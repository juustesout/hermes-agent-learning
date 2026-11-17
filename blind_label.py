"""Blind labels for the HN aidevforum evidence run.

Raters see title/url/summary only. Arm, policy, rank, and auto scores stay
in assignments.jsonl, never in the rater card.
Sampling is a stable 20 percent of posted item ids.
Slot assignment and ranking live in hn_ranker.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import env_setup  # noqa: F401
import opslog

WORKFLOW = "hn-aidevforum"
SUCCESS_DEFINITION = "2026-11-17.3"
SAMPLE_RATE = 0.2
AGREEMENT_FLOOR = 0.9
ASSIGN_SALT = "hn-aidevforum-2026-11-17.2"
P_LEARNED = 0.5
ARMS = ("B", "L")
ARM_ALIASES = {"B": "B", "L": "L", "baseline": "B", "learned": "L"}
EXPECTED_POSTS_PER_DAY = 8
EXPECTED_RUNS_PER_DAY = 4
DECISION_RECORD = "decision"
OUTCOME_RECORD = "outcome"
HIDDEN_FIELDS = frozenset(
    {
        "arm",
        "policy",
        "policy_id",
        "policy_version",
        "rank",
        "choice_prob",
        "auto_relevant",
        "auto_score",
        "rag_score",
        "source_rank",
        "selection_basis",
        "learned_score",
        "baseline",
        "action_logprob",
        "http_status",
        "auto_duplicate",
        "fallback",
        "kill_switch",
        "ranker_score",
        "pool_ids",
        "model_version",
        "slot",
        "chosen_id",
        "baseline_pick_id",
        "features",
        "run_id",
    }
)
VISIBLE_FIELDS = ("id", "title", "url", "summary", "posted_at", "hn_id", "workflow")
YES = {"1", "true", "yes", "y", "on"}
NO = {"0", "false", "no", "n", "off"}


def evidence_dir() -> Path:
    env = os.environ.get("AGENT_LEARNING_EVIDENCE_DIR", "").strip()
    if env:
        path = Path(env)
    else:
        path = Path(opslog.default_store_dir()) / "evidence" / WORKFLOW
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(name: str) -> Path:
    return evidence_dir() / name


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in YES:
        return True
    if text in NO:
        return False
    raise ValueError(f"expected yes/no, got {value!r}")


def quality_success(*, relevant: bool, duplicate: bool) -> bool:
    return bool(relevant) and not bool(duplicate)


def _unit(item_id: str, purpose: str) -> float:
    digest = hashlib.sha256(f"{ASSIGN_SALT}:{purpose}:{item_id}".encode("utf-8")).hexdigest()
    return (int(digest, 16) % 10000) / 10000.0


def should_sample(item_id: str, rate: float = SAMPLE_RATE) -> bool:
    return _unit(item_id, "sample") < rate


def normalize_arm(value: Any) -> Optional[str]:
    if value is None:
        return None
    return ARM_ALIASES.get(str(value).strip())


def assign_arm(item_id: str, p_learned: float = P_LEARNED) -> dict:
    """Legacy per-item coin. Prefer hn_ranker.assign_slot_arm for posting."""
    draw = _unit(item_id, "arm")
    arm = "L" if draw < p_learned else "B"
    return {
        "id": item_id,
        "arm": arm,
        "p_learned": p_learned,
        "choice_prob": P_LEARNED,
        "draw": draw,
        "salt": ASSIGN_SALT,
    }


def load_decisions() -> list[dict]:
    return [row for row in load_assignments() if row.get("record") == DECISION_RECORD]


def load_outcomes() -> list[dict]:
    return [row for row in load_assignments() if row.get("record") == OUTCOME_RECORD]


def latest_decision_for(item_id: str) -> Optional[dict]:
    found = None
    for row in load_decisions():
        chosen = str(row.get("chosen_id") or row.get("id") or "")
        if chosen == item_id:
            found = row
    return found


def blind_view(item: dict) -> dict:
    view = {key: item.get(key, "") for key in VISIBLE_FIELDS}
    leaked = sorted(HIDDEN_FIELDS & set(view))
    if leaked:
        raise ValueError(f"hidden fields in blind view: {leaked}")
    for key in HIDDEN_FIELDS:
        if key in view:
            raise ValueError(f"hidden field {key} leaked")
    return view


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    def _write():
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()

    opslog._with_lock(str(path), _write)


def load_items() -> list[dict]:
    return _read_jsonl(_path("items.jsonl"))


def load_queue() -> list[dict]:
    return _read_jsonl(_path("queue.jsonl"))


def load_labels() -> list[dict]:
    return _read_jsonl(_path("labels.jsonl"))


def load_assignments() -> list[dict]:
    return _read_jsonl(_path("assignments.jsonl"))


def load_health() -> dict:
    path = _path("health.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _ids(rows: Iterable[dict]) -> set[str]:
    return {str(row.get("id")) for row in rows if row.get("id")}


def public_item(item: dict) -> dict:
    record = {key: item.get(key, "") for key in VISIBLE_FIELDS if key in item or key in ("id", "title", "url", "summary")}
    record["id"] = str(item.get("id") or "").strip()
    record["title"] = str(item.get("title") or "").strip()
    record["url"] = str(item.get("url") or "").strip()
    record["summary"] = str(item.get("summary") or "").strip()
    record["workflow"] = WORKFLOW
    if item.get("posted_at"):
        record["posted_at"] = item["posted_at"]
    if item.get("hn_id"):
        record["hn_id"] = item["hn_id"]
    return record


def assignment_record(item: dict, assigned: dict) -> dict:
    http_status = item.get("http_status")
    try:
        http_status = int(http_status) if http_status is not None else None
    except (TypeError, ValueError):
        http_status = None
    arm = normalize_arm(assigned.get("arm") or item.get("arm"))
    run_id = assigned.get("run_id") or item.get("run_id")
    slot = assigned.get("slot") if assigned.get("slot") is not None else item.get("slot")
    payload = {
        "record": OUTCOME_RECORD,
        "id": assigned.get("id") or item.get("id"),
        "chosen_id": assigned.get("chosen_id") or assigned.get("id") or item.get("id"),
        "arm": arm,
        "run_id": run_id,
        "slot": slot,
        "p_learned": assigned.get("p_learned", P_LEARNED),
        "choice_prob": assigned.get("choice_prob", P_LEARNED),
        "http_status": http_status,
        "technical_success": technical_success(http_status),
        "auto_relevant": item.get("auto_relevant"),
        "auto_duplicate": item.get("auto_duplicate"),
        "assigned_at": datetime.now(timezone.utc).isoformat(),
        "success_definition": SUCCESS_DEFINITION,
        "workflow": WORKFLOW,
    }
    if assigned.get("fallback") is not None:
        payload["fallback"] = assigned.get("fallback")
    if assigned.get("kill_switch") is not None:
        payload["kill_switch"] = assigned.get("kill_switch")
    return payload


def technical_success(http_status: Any) -> Optional[bool]:
    if http_status is None:
        return None
    try:
        code = int(http_status)
    except (TypeError, ValueError):
        return None
    return 200 <= code < 300


def enqueue_item(item: dict) -> dict:
    item_id = str(item.get("id") or "").strip()
    if not item_id:
        raise ValueError("item.id is required")
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or "").strip()
    if not title or not url:
        raise ValueError("item.title and item.url are required")
    existing = _ids(load_items())
    if item_id in existing:
        assigned = latest_decision_for(item_id) or next(
            (row for row in load_outcomes() if str(row.get("id") or row.get("chosen_id")) == item_id),
            {},
        )
        return {
            "id": item_id,
            "enqueued": False,
            "sampled": item_id in _ids(load_queue()),
            "arm": normalize_arm(assigned.get("arm")),
            "choice_prob": assigned.get("choice_prob"),
        }
    assigned = latest_decision_for(item_id)
    if assigned is None:
        assigned = assign_arm(item_id)
        assigned["missing_decision"] = True
    record = public_item(item)
    record.setdefault("posted_at", datetime.now(timezone.utc).isoformat())
    _append_jsonl(_path("items.jsonl"), record)
    _append_jsonl(_path("assignments.jsonl"), assignment_record(item, assigned))
    sampled = should_sample(item_id)
    if sampled:
        _append_jsonl(
            _path("queue.jsonl"),
            {
                "id": item_id,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "success_definition": SUCCESS_DEFINITION,
            },
        )
    return {
        "id": item_id,
        "enqueued": True,
        "sampled": sampled,
        "arm": normalize_arm(assigned.get("arm")),
        "choice_prob": assigned.get("choice_prob", P_LEARNED),
        "http_status": item.get("http_status"),
        "missing_decision": bool(assigned.get("missing_decision")),
    }


def unlabeled_queue() -> list[dict]:
    labeled = _ids(load_labels())
    items = {row["id"]: row for row in load_items() if row.get("id")}
    pending = []
    for row in load_queue():
        item_id = str(row.get("id") or "")
        if not item_id or item_id in labeled:
            continue
        item = items.get(item_id)
        if item:
            pending.append(item)
    return pending


def queue_status() -> dict:
    items = load_items()
    queue = load_queue()
    labeled = _ids(load_labels())
    pending = unlabeled_queue()
    last_enqueue = None
    for row in items:
        ts = row.get("posted_at")
        if ts and (last_enqueue is None or str(ts) > str(last_enqueue)):
            last_enqueue = ts
    for row in queue:
        ts = row.get("queued_at")
        if ts and (last_enqueue is None or str(ts) > str(last_enqueue)):
            last_enqueue = ts
    since = 0
    if last_enqueue:
        cutoff = _parse_ts(last_enqueue)
        if cutoff:
            since = sum(1 for row in items if (_parse_ts(row.get("posted_at")) or cutoff) >= cutoff)
    return {
        "pending": len(pending),
        "sampled": len(queue),
        "labeled": sum(1 for row in queue if row.get("id") in labeled),
        "posted": len(items),
        "last_enqueue": last_enqueue,
        "items_since_last_enqueue": since,
    }


def next_card() -> Optional[dict]:
    pending = unlabeled_queue()
    if not pending:
        return None
    return blind_view(pending[0])


def record_label(
    item_id: str,
    *,
    relevant: bool,
    duplicate: bool,
    needs_correction: bool,
    notes: str = "",
    rater: str = "juust",
) -> dict:
    queued = _ids(load_queue())
    if item_id not in queued:
        raise ValueError(f"{item_id} is not in the blind sample")
    payload = {
        "id": item_id,
        "relevant": relevant,
        "duplicate": duplicate,
        "needs_correction": needs_correction,
        "quality_success": quality_success(relevant=relevant, duplicate=duplicate),
        "notes": notes,
        "rater": rater,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "success_definition": SUCCESS_DEFINITION,
        "workflow": WORKFLOW,
    }
    _append_jsonl(_path("labels.jsonl"), payload)
    return payload


def _latest_labels() -> dict[str, dict]:
    latest = {}
    for row in load_labels():
        item_id = str(row.get("id") or "")
        if item_id:
            latest[item_id] = row
    return latest


def auto_quality(assignment: dict) -> bool:
    if assignment.get("auto_duplicate") is True:
        return False
    relevant = assignment.get("auto_relevant")
    if relevant is None:
        return False
    return bool(relevant)


def _by_id(rows: Iterable[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        item_id = str(row.get("chosen_id") or row.get("id") or "")
        if item_id:
            out[item_id] = row
    return out


def report() -> dict:
    items = load_items()
    queue = load_queue()
    assignments = _by_id(load_decisions())
    assignments.update(_by_id(load_outcomes()))
    latest = _latest_labels()
    labeled_n = sum(1 for row in queue if row.get("id") in latest)
    remaining = len(queue) - labeled_n
    compared = 0
    agree = 0
    human_ok = 0
    by_arm = {arm: {"n": 0, "human_ok": 0, "technical_ok": 0} for arm in ARMS}
    for row in items:
        assigned = assignments.get(str(row.get("id") or ""), {})
        arm = normalize_arm(assigned.get("arm"))
        if arm in by_arm:
            by_arm[arm]["n"] += 1
            if assigned.get("technical_success") is True:
                by_arm[arm]["technical_ok"] += 1
    for row in queue:
        item_id = str(row.get("id") or "")
        label = latest.get(item_id)
        if not label:
            continue
        assigned = assignments.get(item_id, {})
        compared += 1
        if label.get("quality_success"):
            human_ok += 1
            arm = normalize_arm(assigned.get("arm"))
            if arm in by_arm:
                by_arm[arm]["human_ok"] += 1
        if bool(label.get("quality_success")) == auto_quality(assigned):
            agree += 1
    agreement = (agree / compared) if compared else None
    return {
        "workflow": WORKFLOW,
        "success_definition": SUCCESS_DEFINITION,
        "posted": len(items),
        "assignments": len(assignments),
        "sampled": len(queue),
        "labeled": labeled_n,
        "remaining": remaining,
        "sample_rate": SAMPLE_RATE,
        "p_learned": P_LEARNED,
        "by_arm": by_arm,
        "human_success_rate": (human_ok / compared) if compared else None,
        "agreement": agreement,
        "agreement_floor": AGREEMENT_FLOOR,
        "automatic_measure_ok": None if agreement is None else agreement >= AGREEMENT_FLOOR,
    }


def record_run(*, attempted: int, posted: int, failed: int, run_id: str = "") -> dict:
    payload = {
        "run_id": run_id or datetime.now(timezone.utc).isoformat(),
        "attempted": int(attempted),
        "posted": int(posted),
        "failed": int(failed),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workflow": WORKFLOW,
    }
    _append_jsonl(_path("runs.jsonl"), payload)
    return payload


def load_runs() -> list[dict]:
    return _read_jsonl(_path("runs.jsonl"))


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def audit(*, expected_posts: int = EXPECTED_POSTS_PER_DAY, hours: int = 24) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items = [row for row in load_items() if (_parse_ts(row.get("posted_at")) or cutoff) >= cutoff]
    outcomes = [
        row
        for row in load_outcomes()
        if (_parse_ts(row.get("assigned_at")) or cutoff) >= cutoff
    ]
    decisions = [
        row
        for row in load_decisions()
        if (_parse_ts(row.get("decided_at")) or cutoff) >= cutoff
    ]
    runs = [row for row in load_runs() if (_parse_ts(row.get("recorded_at")) or cutoff) >= cutoff]
    attempted = sum(int(row.get("attempted") or 0) for row in runs)
    posted_runs = sum(int(row.get("posted") or 0) for row in runs)
    failed_runs = sum(int(row.get("failed") or 0) for row in runs)
    item_ids = {str(row.get("id")) for row in items if row.get("id")}
    decision_ids = {str(row.get("chosen_id") or row.get("id")) for row in decisions}
    outcome_ids = {str(row.get("chosen_id") or row.get("id")) for row in outcomes}
    alarms = []
    if len(items) != len(item_ids):
        alarms.append("duplicate_item_ids")
    if item_ids != outcome_ids:
        alarms.append("items_assignments_mismatch")
    missing_decision = sorted(item_ids - decision_ids)
    if missing_decision:
        alarms.append("posted_without_decision")
    if any(row.get("kill_switch") for row in decisions):
        alarms.append("kill_switch")
    if any(row.get("fallback") for row in decisions):
        alarms.append("ranker_fallback")
    if attempted and posted_runs + failed_runs != attempted:
        alarms.append("run_counts_inconsistent")
    if len(items) < expected_posts - 1:
        alarms.append("posted_below_expected")
    if attempted and len(items) < attempted:
        alarms.append("enqueue_lag")
    payload = {
        "ok": not alarms,
        "hours": hours,
        "expected_posts": expected_posts,
        "items": len(items),
        "outcomes": len(outcomes),
        "decisions": len(decisions),
        "assignments": len(load_assignments()),
        "runs": len(runs),
        "attempted": attempted,
        "posted_runs": posted_runs,
        "failed_runs": failed_runs,
        "posted_without_decision": missing_decision,
        "alarms": alarms,
        "success_definition": SUCCESS_DEFINITION,
    }
    _path("health.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _print_card(card: dict) -> None:
    print(f"id: {card.get('id')}")
    print(f"posted_at: {card.get('posted_at')}")
    print(f"title: {card.get('title')}")
    print(f"url: {card.get('url')}")
    print(f"summary: {card.get('summary')}")
    print("")
    print(
        "python3 blind_label.py label --id "
        f"{card.get('id')} --relevant yes --duplicate no --needs-correction no"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="blind_label")
    sub = parser.add_subparsers(dest="cmd")

    p_en = sub.add_parser("enqueue", help="Add a posted or failed HN item")
    p_en.add_argument("--json", required=True, help="Posted item as JSON object")

    p_imp = sub.add_parser("import-jsonl", help="Bulk enqueue posted items")
    p_imp.add_argument("path")

    p_as = sub.add_parser("assign", help="Show coin-flip arm for an item id")
    p_as.add_argument("--id", required=True)

    p_run = sub.add_parser("record-run", help="Record a 6h job attempt count")
    p_run.add_argument("--attempted", type=int, required=True)
    p_run.add_argument("--posted", type=int, required=True)
    p_run.add_argument("--failed", type=int, required=True)
    p_run.add_argument("--run-id", default="")

    sub.add_parser("next", help="Show the next unlabeled blind card")
    sub.add_parser("report", help="Sample progress and rater/auto agreement")
    sub.add_parser("audit", help="Compare posted vs enqueued vs expected")

    p_lab = sub.add_parser("label", help="Record a blind label")
    p_lab.add_argument("--id", required=True)
    p_lab.add_argument("--relevant", required=True)
    p_lab.add_argument("--duplicate", required=True)
    p_lab.add_argument("--needs-correction", required=True)
    p_lab.add_argument("--notes", default="")
    p_lab.add_argument("--rater", default="juust")

    args = parser.parse_args(argv)
    if args.cmd == "enqueue":
        item = json.loads(args.json)
        print(json.dumps(enqueue_item(item), indent=2))
        return 0
    if args.cmd == "import-jsonl":
        n = 0
        sampled = 0
        for row in _read_jsonl(Path(args.path)):
            result = enqueue_item(row)
            if result["enqueued"]:
                n += 1
            if result["sampled"]:
                sampled += 1
        print(json.dumps({"imported": n, "sampled": sampled}, indent=2))
        return 0
    if args.cmd == "next":
        card = next_card()
        if card is None:
            status = queue_status()
            last = status["last_enqueue"] or "never"
            print(
                "No unlabeled items in the blind sample. "
                f"Last enqueue: {last}. "
                f"Posted items: {status['posted']}. "
                f"Sampled: {status['sampled']}. "
                f"Labeled: {status['labeled']}."
            )
            return 0
        _print_card(card)
        return 0
    if args.cmd == "label":
        try:
            payload = record_label(
                args.id,
                relevant=parse_bool(args.relevant),
                duplicate=parse_bool(args.duplicate),
                needs_correction=parse_bool(args.needs_correction),
                notes=args.notes,
                rater=args.rater,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "report":
        print(json.dumps(report(), indent=2))
        return 0
    if args.cmd == "assign":
        print(json.dumps(assign_arm(args.id), indent=2))
        return 0
    if args.cmd == "record-run":
        print(json.dumps(
            record_run(
                attempted=args.attempted,
                posted=args.posted,
                failed=args.failed,
                run_id=args.run_id,
            ),
            indent=2,
        ))
        return 0
    if args.cmd == "audit":
        payload = audit()
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

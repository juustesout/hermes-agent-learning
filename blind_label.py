"""Blind labels for the HN aidevforum evidence run.

Raters see title/url/summary only. Arm, policy, rank, and auto scores stay hidden.
Sampling is a stable 20 percent of posted item ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import env_setup  # noqa: F401
import opslog

WORKFLOW = "hn-aidevforum"
SUCCESS_DEFINITION = "2026-11-17.1"
SAMPLE_RATE = 0.2
AGREEMENT_FLOOR = 0.9
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


def should_sample(item_id: str, rate: float = SAMPLE_RATE) -> bool:
    digest = hashlib.sha256(str(item_id).encode("utf-8")).hexdigest()
    return (int(digest, 16) % 10000) / 10000.0 < rate


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


def _ids(rows: Iterable[dict]) -> set[str]:
    return {str(row.get("id")) for row in rows if row.get("id")}


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
        return {"id": item_id, "enqueued": False, "sampled": item_id in _ids(load_queue())}
    record = dict(item)
    record["id"] = item_id
    record["workflow"] = WORKFLOW
    record.setdefault("posted_at", datetime.now(timezone.utc).isoformat())
    _append_jsonl(_path("items.jsonl"), record)
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
    return {"id": item_id, "enqueued": True, "sampled": sampled}


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


def auto_quality(item: dict) -> bool:
    if item.get("auto_duplicate") is True:
        return False
    relevant = item.get("auto_relevant")
    if relevant is None:
        return False
    return bool(relevant)


def report() -> dict:
    items = load_items()
    queue = load_queue()
    latest = _latest_labels()
    labeled_n = sum(1 for row in queue if row.get("id") in latest)
    remaining = len(queue) - labeled_n
    compared = 0
    agree = 0
    human_ok = 0
    for row in queue:
        item_id = str(row.get("id") or "")
        label = latest.get(item_id)
        if not label:
            continue
        item = next((it for it in items if it.get("id") == item_id), {})
        compared += 1
        if label.get("quality_success"):
            human_ok += 1
        if bool(label.get("quality_success")) == auto_quality(item):
            agree += 1
    agreement = (agree / compared) if compared else None
    return {
        "workflow": WORKFLOW,
        "success_definition": SUCCESS_DEFINITION,
        "posted": len(items),
        "sampled": len(queue),
        "labeled": labeled_n,
        "remaining": remaining,
        "sample_rate": SAMPLE_RATE,
        "human_success_rate": (human_ok / compared) if compared else None,
        "agreement": agreement,
        "agreement_floor": AGREEMENT_FLOOR,
        "automatic_measure_ok": None if agreement is None else agreement >= AGREEMENT_FLOOR,
    }


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

    p_en = sub.add_parser("enqueue", help="Add a posted HN item")
    p_en.add_argument("--json", required=True, help="Posted item as JSON object")

    p_imp = sub.add_parser("import-jsonl", help="Bulk enqueue posted items")
    p_imp.add_argument("path")

    sub.add_parser("next", help="Show the next unlabeled blind card")
    sub.add_parser("report", help="Sample progress and rater/auto agreement")

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
            print("queue empty")
            return 0
        _print_card(card)
        return 0
    if args.cmd == "label":
        payload = record_label(
            args.id,
            relevant=parse_bool(args.relevant),
            duplicate=parse_bool(args.duplicate),
            needs_correction=parse_bool(args.needs_correction),
            notes=args.notes,
            rater=args.rater,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "report":
        print(json.dumps(report(), indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

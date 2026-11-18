"""Offline Track A advice replay. Measurement gate. Read-only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import features
import opslog
import shadow_policy
from features import PRE_CALL_FEATURE_NAMES, FeatureNameMismatch, pre_call_features

advice_features = pre_call_features

SOURCE = "replay"
DEFAULT_TASK_ID = "choose-scrape-strategy"
DEFAULT_AGENT_ID = "hermes"
DEFAULT_MIN_N = 12
SUCCESS_STATUSES = {"completed", "success", "ok"}
FAIL_STATUSES = {"failed", "error"}


def default_episodes_dir() -> str:
    return os.path.join(opslog.STORE_DIR, "episodes", "hermes")


def default_operations_path() -> str:
    return os.path.join(opslog.STORE_DIR, "operations.jsonl")


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _episode_from_op(op: dict) -> dict:
    action_id = str(op.get("action_id") or op.get("action_name") or "")
    return {
        "id": str(op.get("id") or ""),
        "target": str(op.get("target") or ""),
        "created_at": str(op.get("created_at") or op.get("timestamp") or ""),
        "action_id": action_id,
        "action_name": str(op.get("action_name") or action_id),
        "execution_status": str(op.get("execution_status") or ""),
        "task_id": str(op.get("task_id") or ""),
    }


def load_episodes_dir(path: str) -> list[dict]:
    if not os.path.isdir(path):
        return []
    rows: list[dict] = []
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return []
    for name in names:
        if not name.endswith(".json"):
            continue
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        try:
            data = _read_json(full)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            rows.append(_episode_from_op(data))
    return rows


def load_corpus(*, jsonl: str = "", episodes_dir: str = "") -> list[dict]:
    if jsonl:
        if not os.path.isfile(jsonl):
            raise FileNotFoundError(jsonl)
        return [_episode_from_op(row) for row in _read_jsonl(jsonl)]
    if episodes_dir:
        if not os.path.isdir(episodes_dir):
            raise FileNotFoundError(episodes_dir)
        return load_episodes_dir(episodes_dir)
    rows = load_episodes_dir(default_episodes_dir())
    if rows:
        return rows
    ops_path = default_operations_path()
    if os.path.isfile(ops_path):
        return [_episode_from_op(row) for row in _read_jsonl(ops_path)]
    return []


def has_host(target: str) -> bool:
    return bool(features.host_from_url(target))


def outcome_success(status: str) -> bool | None:
    name = str(status or "").strip().lower()
    if name in SUCCESS_STATUSES:
        return True
    if name in FAIL_STATUSES:
        return False
    return None


def filter_episodes(rows: list[dict], task_id: str) -> list[dict]:
    wanted = str(task_id or "").strip()
    kept: list[dict] = []
    for row in rows:
        if wanted and str(row.get("task_id") or "").strip() != wanted:
            continue
        if not has_host(str(row.get("target") or "")):
            continue
        kept.append(row)
    kept.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or "")))
    return kept


def load_snapshot_file(path: str) -> dict | None:
    data = _read_json(path)
    return data if isinstance(data, dict) else None


def replay(episodes: list[dict], snapshot: dict) -> list[dict]:
    history: list[dict] = []
    out: list[dict] = []
    for ep in episodes:
        target = str(ep.get("target") or "")
        created_at = str(ep.get("created_at") or "")
        action_id = str(ep.get("action_id") or ep.get("action_name") or "")
        phi = advice_features(
            target, history=history, at=created_at or None, action_id=action_id
        )
        advice = shadow_policy.advise(snapshot, phi)
        track = list(advice.get("feature_track") or PRE_CALL_FEATURE_NAMES)
        used = {name: float(phi.get(name, 0.0)) for name in track}
        probs = advice.get("probabilities") or {}
        if not isinstance(probs, dict):
            probs = {}
        out.append(
            {
                "id": str(ep.get("id") or ""),
                "target": target,
                "created_at": created_at,
                "requested_tool": action_id,
                "advised_tool": str(advice.get("advised_tool") or ""),
                "probabilities": {str(k): float(v) for k, v in probs.items()},
                "context_features": used,
            }
        )
        known = outcome_success(str(ep.get("execution_status") or ""))
        if known is None:
            continue
        host = features.host_from_url(target)
        if not host:
            continue
        history.append(
            {
                "domain": host,
                "success": known,
                "action_id": action_id,
                "timestamp": created_at,
            }
        )
    return out


def gate(rows: list[dict], *, min_n: int) -> dict:
    advised = [str(r.get("advised_tool") or "") for r in rows if r.get("advised_tool")]
    n = len(advised)
    counts: dict[str, int] = {}
    for aid in advised:
        counts[aid] = counts.get(aid, 0) + 1
    n_distinct = len(counts)
    top_action = ""
    top_share = 0.0
    if n and counts:
        top_action = max(counts, key=lambda k: (counts[k], k))
        top_share = counts[top_action] / n
    return {
        "n": n,
        "min_n": min_n,
        "n_distinct": n_distinct,
        "top_action": top_action,
        "top_share": round(top_share, 6),
        "counts": counts,
        "gate_open": n >= min_n and n_distinct >= 2,
    }


def exit_code(report: dict) -> int:
    if report.get("gate_open"):
        return 0
    if int(report.get("n") or 0) < int(report.get("min_n") or DEFAULT_MIN_N):
        return 3
    return 2


def build_report(
    rows: list[dict],
    *,
    snapshot: dict,
    task_id: str,
    min_n: int,
) -> dict:
    status = gate(rows, min_n=min_n)
    track = list(PRE_CALL_FEATURE_NAMES)
    ctx = rows[0].get("context_features") if rows else None
    if isinstance(ctx, dict) and ctx:
        track = list(ctx.keys())
    return {
        "gate_open": status["gate_open"],
        "n": status["n"],
        "min_n": status["min_n"],
        "n_distinct": status["n_distinct"],
        "top_action": status["top_action"],
        "top_share": status["top_share"],
        "counts": status["counts"],
        "task_id": task_id,
        "policy_id": str(snapshot.get("id") or ""),
        "policy_version": snapshot.get("version"),
        "feature_track": track,
        "source": SOURCE,
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="replay_advice")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--episodes-dir", default="")
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--agent-id", default=DEFAULT_AGENT_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code in (0, None):
            return 0
        return 1
    if args.min_n < 1:
        print("min-n must be >= 1", file=sys.stderr)
        return 1
    try:
        raw = load_corpus(jsonl=args.jsonl, episodes_dir=args.episodes_dir)
    except FileNotFoundError as exc:
        print(f"corpus not found: {exc}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        print(f"corpus error: {exc}", file=sys.stderr)
        return 1
    episodes = filter_episodes(raw, args.task_id)
    try:
        if args.snapshot:
            if not os.path.isfile(args.snapshot):
                print(f"snapshot not found: {args.snapshot}", file=sys.stderr)
                return 4
            snapshot = load_snapshot_file(args.snapshot)
        else:
            snapshot = shadow_policy.load_snapshot(args.task_id, agent_id=args.agent_id)
    except (OSError, json.JSONDecodeError) as cop:
        print(f"snapshot error: {cop}", file=sys.stderr)
        return 4
    if not isinstance(snapshot, dict):
        print("snapshot missing", file=sys.stderr)
        return 4
    try:
        rows = replay(episodes, snapshot)
        report = build_report(
            rows, snapshot=snapshot, task_id=args.task_id, min_n=args.min_n
        )
    except FeatureNameMismatch as cop:
        print(str(cop), file=sys.stderr)
        return 4
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        try:
            parent = os.path.dirname(args.out)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.write("\n")
        except OSError as cop:
            print(f"write error: {cop}", file=sys.stderr)
            return 1
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())

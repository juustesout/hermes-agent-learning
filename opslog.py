"""
opslog — automatische operation logger voor Hermes agent-learning.

Gebruik (CLI):
  python3 opslog.py log --task-id choose-scrape-strategy --action use_webclaw_scrape \\
    --target agentcursus.nl --success true --duration-ms 3100 --cost 1 \\
    --summary "WebClaw scrape, 24500 chars"

Gebruik (Python):
  from opslog import log_operation
  log_operation(task_id="choose-scrape-strategy", action_id="use_webclaw_scrape", ...)
"""

import json
import uuid
import os
from datetime import datetime, timezone
from typing import Optional

STORE_DIR = "/opt/data/profiles/operator/agent-learning/store"
OPSLOG_PATH = os.path.join(STORE_DIR, "operations.jsonl")


def _ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)


def log_operation(
    task_id: str,
    action_id: str,
    target: str = "",
    success: bool = True,
    duration_ms: Optional[int] = None,
    cost: Optional[float] = None,
    summary: str = "",
    action_type: str = "",
    action_logprob: Optional[float] = None,
    context_features: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """Log één operatie naar operations.jsonl (append-only)."""
    _ensure_store()
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": "hermes",
        "task_id": task_id,
        "action_id": action_id,
        "action_logprob": action_logprob,
        "target": target,
        "execution_status": "completed" if success else "failed",
        "result_summary": summary,
        "action_type": action_type,
        "request_latency_ms": duration_ms,
        "metadata": {
            "cost_credits": cost,
            "task_completed": success,
            **(metadata or {}),
        },
        "context_features": context_features or {},
    }
    with open(OPSLOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry["id"]


def count_operations() -> int:
    """Aantal gelogde operaties."""
    if not os.path.exists(OPSLOG_PATH):
        return 0
    with open(OPSLOG_PATH) as f:
        return sum(1 for _ in f)


def tail(n: int = 10) -> list[dict]:
    """Laatste n operaties."""
    if not os.path.exists(OPSLOG_PATH):
        return []
    with open(OPSLOG_PATH) as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines[-n:]]


def stats() -> dict:
    """Basis statistieken over alle operaties."""
    if not os.path.exists(OPSLOG_PATH):
        return {"total": 0, "by_task": {}, "by_status": {}, "avg_duration_ms": None}
    by_task = {}
    by_status = {}
    durations = []
    total = 0
    with open(OPSLOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            tid = op.get("task_id", "unknown")
            by_task[tid] = by_task.get(tid, 0) + 1
            status = op.get("execution_status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            d = op.get("request_latency_ms")
            if d is not None:
                durations.append(d)
    avg_dur = round(sum(durations) / len(durations), 1) if durations else None
    return {
        "total": total,
        "by_task": by_task,
        "by_status": by_status,
        "avg_duration_ms": avg_dur,
        "file": OPSLOG_PATH,
    }


# ---- CLI ----

def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="opslog", description="Hermes operation logger")
    sub = parser.add_subparsers(dest="cmd")

    p_log = sub.add_parser("log", help="Log een operatie")
    p_log.add_argument("--task-id", required=True)
    p_log.add_argument("--action-id", required=True)
    p_log.add_argument("--target", default="")
    p_log.add_argument("--success", default="true", choices=["true", "false"])
    p_log.add_argument("--duration-ms", type=int, default=None)
    p_log.add_argument("--cost", type=float, default=None)
    p_log.add_argument("--summary", default="")
    p_log.add_argument("--action-type", default="")
    p_log.add_argument("--logprob", type=float, default=None,
                       help="Action log probability (voor off-policy correctie)")
    p_log.add_argument("--context-features", default=None,
                       help="JSON string met context features (bv. '{\"cloudflare\":1.0}')")

    sub.add_parser("count", help="Aantal operaties")
    p_tail = sub.add_parser("tail", help="Laatste N operaties")
    p_tail.add_argument("--n", type=int, default=10)

    sub.add_parser("stats", help="Statistieken")

    args = parser.parse_args(argv)

    if args.cmd == "log":
        cf = None
        if args.context_features:
            try:
                cf = json.loads(args.context_features)
            except json.JSONDecodeError:
                print(f"Invalid context-features JSON: {args.context_features}", file=sys.stderr)
                return 1
        oid = log_operation(
            task_id=args.task_id,
            action_id=args.action_id,
            target=args.target,
            success=args.success == "true",
            duration_ms=args.duration_ms,
            cost=args.cost,
            summary=args.summary,
            action_type=args.action_type,
            action_logprob=args.logprob,
            context_features=cf,
        )
        print(f"ok {oid}")
        return 0
    elif args.cmd == "count":
        print(count_operations())
        return 0
    elif args.cmd == "tail":
        for op in tail(args.n):
            print(json.dumps(op, indent=2)[:500])
            print("---")
        return 0
    elif args.cmd == "stats":
        s = stats()
        print(json.dumps(s, indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
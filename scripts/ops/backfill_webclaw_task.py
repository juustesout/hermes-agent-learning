"""backfill_webclaw_task.py — herlabel oude webclaw-operaties naar choose-scrape-strategy

Gebruik:
    python3 backfill_webclaw_task.py               # droge run (rapport alleen)
    python3 backfill_webclaw_task.py --run          # echte backfill

Backup: store/operations.jsonl.backup-<YYYYMMDD>
Aanroep: 18 nov 2026, na mapping-wijziging in hermes_plugin/__init__.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from shutil import copy2

STORE_DIR = (
    os.environ.get("AGENT_LEARNING_OPSLOG_DIR")
    or os.environ.get("AGENT_LEARNING_LOCAL_STORE_DIR")
    or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "store")
)
OPSLOG = os.path.join(STORE_DIR, "operations.jsonl")
BACKUP = OPSLOG + f".backup-{date.today().isoformat()}"

DRY_RUN = "--run" not in sys.argv


def load_ops(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


def count_by_task(ops: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in ops:
        tid = str(op.get("task_id", "?"))
        counts[tid] = counts.get(tid, 0) + 1
    return counts


def report(ops: list[dict], label: str) -> None:
    total = len(ops)
    unique = len({op.get("id") for op in ops})
    by_task = count_by_task(ops)
    print(f"\n=== {label} ===")
    print(f"  Totaal regels: {total}")
    print(f"  Unieke IDs:    {unique}")
    print(f"  Duplicaten:    {total - unique}")
    for tid in sorted(by_task):
        print(f"    {tid}: {by_task[tid]}")
    # Webclaw-specifiek
    wc = [op for op in ops if op.get("action_id") == "mcp_webclaw_webclaw_scrape"]
    print(f"  mcp_webclaw_webclaw_scrape: {len(wc)}")
    wc_by_task = count_by_task(wc)
    for tid in sorted(wc_by_task):
        print(f"    → {tid}: {wc_by_task[tid]}")


def main() -> int:
    if not os.path.isfile(OPSLOG):
        print(f"Geen opslog gevonden: {OPSLOG}")
        return 1

    ops = load_ops(OPSLOG)
    report(ops, "HUIDIGE OPERATIONS.JSONL")

    has_backup = os.path.isfile(BACKUP)
    if has_backup:
        before = load_ops(BACKUP)
        report(before, "VOOR BACKFILL (backup)")
        print(f"\n  Backup: {BACKUP}")

        # Vergelijk aantallen
        delta = len(ops) - len(before)
        print(f"\n  Verschil in regels: {delta}")
        if delta != 0:
            print(f"  ⚠ Aantal regels is veranderd! Backfill verandert alleen task_id.")
            print(f"    Oorzaak: consolidatie heeft nieuwe rijen toegevoegd of")
            print(f"    de backfill is niet de enige wijziging.")
    else:
        print(f"\n  Geen backup gevonden op {BACKUP}")

    # Webclaw hermes-tool → klaar voor backfill
    to_fix = [
        op
        for op in ops
        if op.get("action_id") == "mcp_webclaw_webclaw_scrape"
        and op.get("task_id") == "hermes-tool"
    ]
    print(f"\n  Te herlabelen: {len(to_fix)}")
    for op in to_fix:
        print(f"    {op.get('id','?')[:8]} → {op.get('target','')[:80]}")

    if DRY_RUN:
        print("\n  DRY RUN — geen wijzigingen. Gebruik --run voor echte backfill.")
        return 0

    # Backup
    if not has_backup:
        copy2(OPSLOG, BACKUP)
        print(f"\n  Backup gemaakt: {BACKUP}")

    # Backfill
    changes = 0
    for op in ops:
        if op.get("action_id") == "mcp_webclaw_webclaw_scrape" and op.get("task_id") == "hermes-tool":
            op["task_id"] = "choose-scrape-strategy"
            op.setdefault("metadata", {})["backfill_task_id"] = "hermes-tool->choose-scrape-strategy"
            changes += 1

    with open(OPSLOG, "w") as f:
        for op in ops:
            f.write(json.dumps(op, ensure_ascii=False) + "\n")

    print(f"\n  Backfill voltooid: {changes} operaties her-labeld")
    return 0


if __name__ == "__main__":
    sys.exit(main())
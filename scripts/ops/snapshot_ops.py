"""Snapshot: split niet-geconsolideerde ops op task_id."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import opslog

# 1. Laad alle operaties
path = os.path.join(opslog.STORE_DIR, "operations.jsonl")
rows = []
with open(path) as f:
    for line in f:
        raw = line.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

# 2. Laad episode IDs
ep_dir = os.path.join(opslog.STORE_DIR, "episodes", "hermes")
ep_ids = set()
if os.path.isdir(ep_dir):
    for name in os.listdir(ep_dir):
        if name.endswith(".json"):
            ep_ids.add(name[:-5])

op_ids = {r.get("id", "") for r in rows if r.get("id")}
only_ops_ids = op_ids - ep_ids
orphan_ids = ep_ids - op_ids

# 3. Snapshot timestamp
import datetime
ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

print(f"=== SNAPSHOT {ts} ===")
print(f"Operations.jsonl rijen: {len(rows)}")
print(f"Operations IDs: {len(op_ids)}")
print(f"Episode IDs: {len(ep_ids)}")
print(f"In beide: {len(op_ids & ep_ids)}")
print(f"Alleen opslog (niet-geconsolideerd): {len(only_ops_ids)}")
print(f"Weeskinderen: {len(orphan_ids)}")
print(f"  Waarvan 0d846871 in orphan: {'0d846871-ee23-4e28-9e68-bb08a2656f23' in orphan_ids}")
print()

# 4. Split alleen-ops by task_id + URL/niet-URL
eligible = Counter()  # choose-scrape-strategy met URL
noneligible = Counter()  # andere task_ids
eligible_nourl = 0

for r in rows:
    rid = r.get("id", "")
    if rid not in only_ops_ids:
        continue
    task = r.get("task_id", "unknown")
    target = str(r.get("target", "") or "")
    has_url = bool(target and ("://" in target or "." in target[:50]))
    if task == "choose-scrape-strategy":
        if has_url:
            eligible["choose-scrape-strategy (URL)"] += 1
        else:
            eligible_nourl += 1
    else:
        noneligible[task] += 1

print("== Niet-geconsolideerd: eligible ==")
for k, v in eligible.most_common():
    print(f"  {k}: {v}")
print(f"  choose-scrape-strategy (geen URL): {eligible_nourl}")
print()
print("== Niet-geconsolideerd: overige tasks ==")
for k, v in noneligible.most_common(10):
    print(f"  {k}: {v}")
print()

# 5. Gate-stand
print("== Gate ==")
print(f"URL-eligible in opslog: {sum(eligible.values())}")
print(f"URL-weeskind 0d846871: 1")
print(f"Telling met weeskind: {sum(eligible.values()) + 1}")
print(f"Telling zonder weeskind: {sum(eligible.values())}")
print(f"min_n: 12")
print(f"Nog nodig (zonder weeskind): {12 - sum(eligible.values())}")
print()

# 6. Details van eligible ops
print("== URL-eligible ops details ==")
for r in rows:
    rid = r.get("id", "")
    if rid not in only_ops_ids:
        continue
    task = r.get("task_id", "")
    target = str(r.get("target", "") or "")
    has_url = bool(target and ("://" in target or "." in target[:50]))
    if task == "choose-scrape-strategy" and has_url:
        ts_op = r.get("created_at") or r.get("timestamp") or "?"
        cf = r.get("context_features")
        cf_status = "null" if cf is None else f"len={len(cf)}" if isinstance(cf, dict) else "?"
        print(f"  {rid[:8]} target={target[:40]} ts={ts_op[:19]} features={cf_status}")
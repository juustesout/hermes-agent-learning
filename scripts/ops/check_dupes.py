"""Check operations.jsonl: rows vs unique IDs, show duplicates."""
import json
import operator
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import opslog

path = os.path.join(opslog.STORE_DIR, "operations.jsonl")
rows = []
with open(path) as f:
    for i, line in enumerate(f, 1):
        raw = line.strip()
        if raw:
            try:
                rows.append((i, json.loads(raw)))
            except json.JSONDecodeError:
                pass

all_ids = [r[1].get("id", "") for r in rows if r[1].get("id")]
unique_ids = set(all_ids)

print(f"Total lines in operations.jsonl: {len(rows)}")
print(f"Rows with non-empty ID:          {len(all_ids)}")
print(f"Unique IDs:                      {len(unique_ids)}")
print(f"Duplicate rows (extra copies):   {len(all_ids) - len(unique_ids)}")
print()

if len(all_ids) > len(unique_ids):
    from collections import Counter
    dupes = {k: v for k, v in Counter(all_ids).items() if v > 1}
    print(f"Distinct duplicate IDs: {len(dupes)}")
    print()
    for id_, count in sorted(dupes.items(), key=operator.itemgetter(1), reverse=True)[:15]:
        matching = [(lnum, r) for lnum, r in rows if r.get("id") == id_]
        line_nums = [lnum for lnum, r in matching]
        task_ids = {r.get("task_id", "?") for _, r in matching}
        context_features_set = {
            str(r.get("context_features"))[:20]
            for _, r in matching
        }
        print(f"  {id_[:20]}...  x{count}")
        print(f"    lines:   {line_nums}")
        print(f"    tasks:   {task_ids}")
        print(f"    context: {context_features_set}")
    if len(dupes) > 15:
        print(f"  ... and {len(dupes) - 15} more")
else:
    print("No duplicates found.")
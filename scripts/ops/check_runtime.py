"""Check latest operations in opslog after scrape."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import opslog

path = os.path.join(opslog.STORE_DIR, "operations.jsonl")
with open(path) as f:
    lines = f.readlines()

# Last 3 lines
for i in range(max(0, len(lines) - 3), len(lines)):
    try:
        op = json.loads(lines[i].strip())
        print(f"--- Line {i+1} ---")
        print(f'id: {op.get("id", "")[:16]}')
        print(f'task_id: {op.get("task_id")}')
        print(f'action_id: {op.get("action_id")}')
        target = str(op.get("target", ""))[:60]
        print(f'target: {target}')
        cf = op.get("context_features")
        print(f"context_features: {cf is not None}")
        if cf:
            non_zero = [(k, v) for k, v in cf.items() if v != 0.0]
            print(f"  non-zero features: {non_zero}")
            non_host = {k: v for k, v in cf.items() if v != 0.0 and k not in (
                "url_host_nieuws","url_host_code","url_host_social",
                "url_host_blog","url_host_overig"
            )}
            print(f"  non-zero outside host category: {non_host}")
        meta = op.get("metadata", {})
        print(f'metadata.import_source: {meta.get("import_source", "MISSING")}')
        print()
    except Exception as e:
        print(f"Error: {e}")

# Check health
health_path = os.path.join(opslog.STORE_DIR, "opslog-health.json")
if os.path.isfile(health_path):
    with open(health_path) as f:
        health = json.load(f)
    print(f"Health: ok={health.get('ok')}, last_error={health.get('last_error')}")
else:
    print("Health: opslog-health.json bestaat niet")
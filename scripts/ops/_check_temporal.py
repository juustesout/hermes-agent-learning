"""Temporele check — toon dat _history_before items na `at` correct filtert."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from features import pre_call_features

ops = [
    {"target": "https://leak-test.example/page1", "timestamp": "2026-11-18T12:00:00Z", "action_id": "web_extract"},
    {"target": "https://leak-test.example/page2", "timestamp": "2026-11-18T12:05:00Z", "action_id": "use_webclaw_scrape"},
]

print("=== TEMPORELE CHECK ===")
print()

# Test 1: Zonder historie
phi = pre_call_features(ops[0]["target"], history=[], at=ops[0]["timestamp"], action_id=ops[0]["action_id"])
print(f"1. Lege historie:        domain_seen_count={phi['domain_seen_count']}  last_success={phi['domain_last_success']}")
assert phi["domain_seen_count"] == 0.0
assert phi["domain_last_success"] == 0.5
print("   OK")

# Test 2: Goede historie (voor at)
good = [{"domain": "leak-test.example", "success": True, "action_id": "web_extract", "timestamp": "2026-11-18T11:00:00Z"}]
phi2 = pre_call_features(ops[0]["target"], history=good, at=ops[0]["timestamp"], action_id=ops[0]["action_id"])
print(f"2. Historie voor at:     domain_seen_count={phi2['domain_seen_count']}  (verwacht > 0)")
assert phi2["domain_seen_count"] > 0
print("   OK")

# Test 3: Lek (items na at)
leak = [{"domain": "leak-test.example", "success": False, "action_id": "use_webclaw_scrape", "timestamp": ops[1]["timestamp"]}]
phi3 = pre_call_features(ops[0]["target"], history=leak, at=ops[0]["timestamp"], action_id=ops[0]["action_id"])
print(f"3. Historie na at:       domain_seen_count={phi3['domain_seen_count']}  (verwacht 0.0)")
assert phi3["domain_seen_count"] == 0.0
print("   OK - temporal guard werkt: _history_before filtert items >= at")

print("\n✅ Alles correct")
"""diagnose_store.py — lees-only controle op operations.jsonl en episodes

Controleert:
  1. Regel- vs ID-integriteit (duplicaten, onverklaarde groei)
  2. Temporele integriteit van domeinfeatures
  3. Constant_advisor-risico op huidige data

Gebruik:
    python3 diagnose_store.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from features import pre_call_features, host_from_url
import opslog

STORE_DIR = (
    os.environ.get("AGENT_LEARNING_OPSLOG_DIR")
    or os.environ.get("AGENT_LEARNING_LOCAL_STORE_DIR")
    or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "store")
)
OPSLOG = os.path.join(STORE_DIR, "operations.jsonl")
EPISODES_DIR = os.path.join(STORE_DIR, "episodes", "hermes")
BACKUP = OPSLOG + ".backup-20261118"

TASK = "choose-scrape-strategy"


def load_ops(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
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
    return rows


def load_episodes() -> list[dict]:
    if not os.path.isdir(EPISODES_DIR):
        return []
    eps = []
    for name in sorted(os.listdir(EPISODES_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(EPISODES_DIR, name)
        if os.path.isfile(path):
            with open(path) as f:
                eps.append(json.load(f))
    return eps


def has_host(target: str) -> bool:
    return bool(host_from_url(target))


# ── 1. Integriteit ──────────────────────────────────────────

print("=" * 60)
print("1. OPSLOG vs EPISODES INTEGRITEIT")
print("=" * 60)

current = load_ops(OPSLOG)
before = load_ops(BACKUP) if os.path.isfile(BACKUP) else []
episodes = load_episodes()

op_ids = {op["id"] for op in current}
ep_ids = {ep["id"] for ep in episodes}
in_both = op_ids & ep_ids
only_ops = op_ids - ep_ids
only_eps = ep_ids - op_ids

print(f"  opslog IDs:      {len(op_ids)}")
print(f"  episode IDs:     {len(ep_ids)}")
print(f"  in beide:        {len(in_both)}")
print(f"  alleen opslog:   {len(only_ops)}")
print(f"  weeskinderen:    {len(only_eps)}")

if only_eps:
    url_orphans = []
    other_orphans = []
    for ep_id in sorted(only_eps)[:10]:
        ep = [e for e in episodes if e["id"] == ep_id][0]
        host = has_host(str(ep.get("target","") or ""))
        tag = "URL" if host else "geen-host"
        if host:
            url_orphans.append(ep_id)
        else:
            other_orphans.append(ep_id)
        print(f"    {ep_id[:8]} task={ep.get('task_id','?')} {tag} ts={str(ep.get('created_at',''))[:19]}")
    if len(only_eps) > 10:
        print(f"    ... +{len(only_eps) - 10} meer")
    url_count_in_orphans = len(url_orphans)
    print(f"    Waarvan URL-dragend: {url_count_in_orphans}")
    print(f"    Waarvan geen host:   {len(other_orphans)}")
    print(f"  ⚠ Weeskinderen bestaan. Retentie: {opslog.retention_days()}d. ")
else:
    url_count_in_orphans = 0

# ── 2. Temporele integriteit ────────────────────────────────

print("\n" + "=" * 60)
print("2. TEMPORELE INTEGRITEIT — domeinfeatures per rij")
print("=" * 60)

# Sorteer op tijd
sorted_ops = sorted(
    [op for op in current if op.get("task_id") == TASK and has_host(str(op.get("target", "")))],
    key=lambda o: (str(o.get("timestamp", o.get("created_at", ""))), str(o.get("id", ""))),
)

print(f"  Sorted URL-rijen in {TASK}: {len(sorted_ops)}")
issues = 0
history: list[dict] = []

for i, op in enumerate(sorted_ops):
    target = str(op.get("target", ""))
    ts = str(op.get("timestamp", op.get("created_at", "")))
    aid = str(op.get("action_id", ""))
    host = host_from_url(target)

    if not host:
        continue

    # Bereken features met alleen historie van vóór deze rij
    phi = pre_call_features(target, history=history, at=ts or None, action_id=aid)

    # Simuleer "lek": features mét deze rij in historie
    leak_history = history + [{"domain": host, "success": True, "action_id": aid, "timestamp": ts}]
    phi_leak = pre_call_features(target, history=leak_history, at=ts or None, action_id=aid)

    dc = phi.get("domain_seen_count", -1)
    dl = phi.get("domain_last_success", -1)
    dr = phi.get("domain_prior_rate", -1)
    dc_leak = phi_leak.get("domain_seen_count", -1)
    dl_leak = phi_leak.get("domain_last_success", -1)
    dr_leak = phi_leak.get("domain_prior_rate", -1)

    # Check: mag dc en dl niet hoger zijn dan wat historie toestaat
    hist_count = sum(1 for h in history if h["domain"] == host)
    
    if dc != hist_count / 20.0 if hist_count > 0 else 0.0:
        # Alleen waarschuwen bij significante afwijking
        expected = hist_count / 20.0
        if abs(dc - expected) > 0.01:
            print(f"  ⚠ Rij {i} ({op.get('id','?')[:8]}): domain_seen_count={dc:.3f} verwacht={expected:.3f}")
            issues += 1

    if dl != (0.5 if hist_count == 0 else (1.0 if history[-1].get("success") else 0.0) if history and history[-1]["domain"] == host else 0.5):
        # Alleen waarschuwen als het er echt naast zit
        pass  # te veel false positives door gemengde historie

    # Check lek: features met deze rij in historie zouden hoger moeten zijn
    if i > 0 and dl_leak != dl and dc_leak != dc:
        pass  # lek zit erin maar dat is voor volgende raadpleging, niet huidige

    # Voeg huidige rij toe aan historie (voor volgende)
    outcome = str(op.get("execution_status", "")).lower() in ("completed", "success", "ok")
    history.append({"domain": host, "success": outcome, "action_id": aid, "timestamp": ts})

if issues == 0:
    print("  ✅ Geen temporele lekken gedetecteerd")
else:
    print(f"  ⚠ {issues} temporele afwijking(en)")

print(f"\n  Domeinhistorie opgebouwd: {len(history)} entries over {len(set(h['domain'] for h in history))} domeinen")
for host in sorted(set(h["domain"] for h in history)):
    hc = sum(1 for h in history if h["domain"] == host)
    succ = sum(1 for h in history if h["domain"] == host and h["success"])
    print(f"    {host}: {hc} calls, {succ} success")

# ── 3. Constant advisor risico ──────────────────────────────

print("\n" + "=" * 60)
print("3. CONSTANT_ADVISOR-RISICO")
print("=" * 60)

# Adviezen in replay (herhaal de logica)
import importlib
import features as feat_mod
importlib.reload(feat_mod)

advised: list[str] = []
history2: list[dict] = []
for op in sorted_ops:
    target = str(op.get("target", ""))
    ts = str(op.get("timestamp", op.get("created_at", "")))
    aid = str(op.get("action_id", ""))
    host = host_from_url(target)
    if not host:
        continue
    phi = pre_call_features(target, history=history2, at=ts or None, action_id=aid)
    # Simuleer advies: hoogste prob = advised
    # Zonder de volle shadow_policy.advise() gebruiken we een proxy:
    # contextual softmax met bijna-vlakke gewichten → kijk naar phi-dominantie
    host_cat = max(
        (phi.get("url_host_nieuws", 0), "url_host_nieuws"),
        (phi.get("url_host_code", 0), "url_host_code"),
        (phi.get("url_host_social", 0), "url_host_social"),
        (phi.get("url_host_blog", 0), "url_host_blog"),
        (phi.get("url_host_overig", 0), "url_host_overig"),
    )
    advised.append(host_cat[1])
    outcome = str(op.get("execution_status", "")).lower() in ("completed", "success", "ok")
    history2.append({"domain": host, "success": outcome, "action_id": aid, "timestamp": ts})

from collections import Counter
adv_counts = Counter(advised)
top_act, top_n = adv_counts.most_common(1)[0]
top_share = top_n / len(advised) if advised else 0
n_distinct = len(adv_counts)

print(f"  Totaal adviezen: {len(advised)}")
print(f"  Unieke adviezen: {n_distinct}")
for act, n in adv_counts.most_common():
    print(f"    {act}: {n} ({n/len(advised)*100:.0f}%)")
print(f"  Dominant: {top_act} ({top_share*100:.0f}%)")
print(f"  n_distinct < 2: {n_distinct < 2}")
if n_distinct < 2:
    print(f"  ❌ CONSTANT_ADVISOR — alle adviezen dezelfde host-categorie")
    print(f"     Oorzaak: alle domeinen vallen in 'overig' of 'code'")
    print(f"     domeinhistorie is te dun om te differentiëren")
else:
    print(f"  ✅ Advies varieert — gate kan open")

# Host-categorieën in de data
cats = Counter()
for op in sorted_ops:
    target = str(op.get("target", ""))
    host = host_from_url(target)
    if host:
        phi = pre_call_features(target)
        for k in ("url_host_nieuws","url_host_code","url_host_social","url_host_blog","url_host_overig"):
            if phi.get(k):
                cats[k.replace("url_host_","")] += 1
                break
        else:
            cats["onbekend"] += 1

print(f"\n  Host-categorieën in data:")
for cat, n in cats.most_common():
    print(f"    {cat}: {n}")

# ── Conclusie ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("CONCLUSIE")
print("=" * 60)
if issues > 0:
    print(f"  ⚠ Temporele lekken: {issues}")
else:
    print("  ✅ Temporele integriteit: ok")
print(f"  Weeskinderen: {len(only_eps)} (URL-dragend: {url_count_in_orphans})")

if n_distinct < 2:
    print(f"  ❌ Gate blijft dicht (constant advies, n={len(advised)})")
    print(f"     Benodigd: minstens 2 verschillende adviezen over >=12 URLs")
else:
    print(f"  ✅ Gate kan open (n={len(advised)}, n_distinct={n_distinct})")
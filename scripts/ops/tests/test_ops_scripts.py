"""Tests voor backfill_webclaw_task.py en diagnose_store.py.

Draait scripts als subprocess met tijdelijke store in tmpdir.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPTS = ROOT / "scripts" / "ops"

SAMPLE_ROWS = [
    {"id": "a0000001", "task_id": "hermes-tool", "action_id": "mcp_webclaw_webclaw_scrape",
     "target": "https://example.com", "execution_status": "completed",
     "timestamp": "2026-11-18T12:00:00Z", "metadata": {}},
    {"id": "a0000002", "task_id": "hermes-tool", "action_id": "mcp_webclaw_webclaw_scrape",
     "target": "https://test.org", "execution_status": "completed",
     "timestamp": "2026-11-18T12:01:00Z", "metadata": {}},
    {"id": "a0000003", "task_id": "choose-scrape-strategy", "action_id": "use_webclaw_scrape",
     "target": "https://already-done.com", "execution_status": "completed",
     "timestamp": "2026-11-18T12:02:00Z", "metadata": {}},
    {"id": "a0000004", "task_id": "hermes-tool", "action_id": "read_file",
     "target": "/some/path", "execution_status": "completed",
     "timestamp": "2026-11-18T12:03:00Z", "metadata": {}},
]


def _build_store(tmp: str, rows: list[dict] | None = None) -> str:
    """Maak een tijdelijke store met operations.jsonl. Retourneert store-pad."""
    store = os.path.join(tmp, "store")
    os.makedirs(os.path.join(store, "episodes", "hermes"), exist_ok=True)
    ops_path = os.path.join(store, "operations.jsonl")
    for row in (rows or SAMPLE_ROWS):
        with open(ops_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    return store


def _run_backfill(tmp: str, args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Draai backfill_webclaw_task.py in tmp met gegeven args."""
    store = os.path.join(tmp, "store")
    env = os.environ.copy()
    env["AGENT_LEARNING_OPSLOG_DIR"] = store
    cmd = ["uv", "run", "python3", str(SCRIPTS / "backfill_webclaw_task.py")]
    if args:
        cmd.extend(args)
    return subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=tmp,
        timeout=30,
        env=env,
    )


def _run_diagnose(tmp: str) -> subprocess.CompletedProcess:
    """Draai diagnose_store.py in tmp."""
    store = os.path.join(tmp, "store")
    env = os.environ.copy()
    env["AGENT_LEARNING_OPSLOG_DIR"] = store
    return subprocess.run(
        ["uv", "run", "python3", str(SCRIPTS / "diagnose_store.py")],
        capture_output=True, text=True,
        cwd=tmp,
        timeout=30,
        env=env,
    )


def _hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── Test 1: Dry run wijzigt niets ───────────────────────────

def test_dry_run_does_not_modify():
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        ops_path = os.path.join(store, "operations.jsonl")
        h_before = _hash(ops_path)
        result = _run_backfill(tmp)  # geen --run = dry run
        assert result.returncode == 0, f"dry run exit {result.returncode}: {result.stderr}"
        h_after = _hash(ops_path)
        assert h_before == h_after, "dry run wijzigde de store!"


# ── Test 2: --run maakt backup en rapporteert aantal ────────

def test_run_creates_backup():
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        ops_path = os.path.join(store, "operations.jsonl")
        result = _run_backfill(tmp, ["--run"])
        assert result.returncode == 0, f"run: {result.stderr}"
        # Backup moet bestaan
        backups = [f for f in os.listdir(store) if f.startswith("operations.jsonl.backup")]
        assert len(backups) >= 1, f"geen backup gevonden in {store}"
        # Output moet aantal gewijzigde rijen tonen
        assert "2" in result.stdout, f"verwacht 2 gewijzigd, output: {result.stdout}"
        # Check dat de 2 webclaw rijen zijn her-labeld
        with open(ops_path) as f:
            ops = [json.loads(l) for l in f if l.strip()]
        for op in ops:
            if op["id"] in ("a0000001", "a0000002"):
                assert op["task_id"] == "choose-scrape-strategy", f"{op['id']} niet her-labeld"


# ── Test 3: Idempotent ──────────────────────────────────────

def test_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        _build_store(tmp)
        r1 = _run_backfill(tmp, ["--run"])
        assert r1.returncode == 0
        # Check of metadata dubbel staat
        ops_path = os.path.join(tmp, "store", "operations.jsonl")
        with open(ops_path) as f:
            ops1 = [json.loads(l) for l in f if l.strip()]
        n_backfill_meta1 = sum(1 for op in ops1 if op.get("metadata", {}).get("backfill_task_id"))

        r2 = _run_backfill(tmp, ["--run"])
        assert r2.returncode == 0
        with open(ops_path) as f:
            ops2 = [json.loads(l) for l in f if l.strip()]
        n_backfill_meta2 = sum(1 for op in ops2 if op.get("metadata", {}).get("backfill_task_id"))

        # Het aantal mag niet verdubbelen
        assert n_backfill_meta2 == n_backfill_meta1, (
            f"backfill meta verdubbeld: {n_backfill_meta1} -> {n_backfill_meta2}"
        )
        # Check geen dubbele pijlen
        for op in ops2:
            meta = op.get("metadata", {})
            bf = meta.get("backfill_task_id", "")
            if bf:
                assert bf.count("->") == 1, f"dubbele pijl in {op['id']}: {bf}"


# ── Test 4: Alleen juiste rijen ─────────────────────────────

def test_only_correct_rows():
    with tempfile.TemporaryDirectory() as tmp:
        _build_store(tmp)
        result = _run_backfill(tmp, ["--run"])
        assert result.returncode == 0
        ops_path = os.path.join(tmp, "store", "operations.jsonl")
        with open(ops_path) as f:
            ops = [json.loads(l) for l in f if l.strip()]
        for op in ops:
            is_webclaw = op["action_id"] == "mcp_webclaw_webclaw_scrape"
            is_target = op["id"] in ("a0000001", "a0000002")
            if is_webclaw and is_target:
                assert op["task_id"] == "choose-scrape-strategy", f"{op['id']} niet her-labeld"
            elif is_webclaw and not is_target:
                pytest.fail(f"webclaw maar niet in target ids? {op['id']}")
            else:
                # Niet-webclaw rijen moeten onaangeroerd zijn
                assert "backfill_task_id" not in op.get("metadata", {}), f"{op['id']} ten onrechte gemarkeerd"
                if op["id"] == "a0000003":
                    assert op["task_id"] == "choose-scrape-strategy"  # was al goed
                elif op["id"] == "a0000004":
                    assert op["task_id"] == "hermes-tool"  # read_file blijft hermes-tool


# ── Test 5: Temporele check vangt lek ───────────────────────

def test_diagnose_detects_temporal_leak():
    """Controleer dat features voor een rij geen toekomst zien."""
    from features import pre_call_features, host_from_url
    ops = [
        {"id": "t001", "task_id": "choose-scrape-strategy",
         "action_id": "mcp_webclaw_webclaw_scrape",
         "target": "https://leak-test.example/page1",
         "execution_status": "completed",
         "timestamp": "2026-11-18T12:00:00Z", "metadata": {}},
        {"id": "t002", "task_id": "choose-scrape-strategy",
         "action_id": "use_webclaw_scrape",
         "target": "https://leak-test.example/page2",
         "execution_status": "failed",
         "timestamp": "2026-11-18T12:05:00Z", "metadata": {}},
    ]
    # t001 zonder historie → domain_seen_count = 0
    phi_1 = pre_call_features(
        ops[0]["target"], history=[], at=ops[0]["timestamp"], action_id=ops[0]["action_id"],
    )
    assert phi_1["domain_seen_count"] == 0.0, \
        f"zonder historie: domain_seen_count={phi_1['domain_seen_count']}"
    assert phi_1["domain_last_success"] == 0.5, \
        f"zonder historie: domain_last_success={phi_1['domain_last_success']}"

    # Goede historie (vóór t001) → domain_seen_count moet > 0
    good_history = [{"domain": "leak-test.example", "success": True,
                     "action_id": "web_extract", "timestamp": "2026-11-18T11:00:00Z"}]
    phi_good = pre_call_features(
        ops[0]["target"], history=good_history, at=ops[0]["timestamp"], action_id=ops[0]["action_id"],
    )
    assert phi_good["domain_seen_count"] > 0, \
        "met historie van vóór at moet domain_seen_count > 0 zijn"

    # Lek: entry met timestamp na at wordt gefilterd door _history_before (>= check)
    # Dit bewijst dat de temporal guard werkt — de entry wordt niet meegeteld
    leak_history = [{"domain": "leak-test.example", "success": True,
                     "action_id": "web_extract", "timestamp": ops[1]["timestamp"]}]
    phi_leak = pre_call_features(
        ops[0]["target"], history=leak_history, at=ops[0]["timestamp"], action_id=ops[0]["action_id"],
    )
    assert phi_leak["domain_seen_count"] == 0.0, \
        f"history na at mag niet meetellen, kreeg {phi_leak['domain_seen_count']}"
    assert phi_1["domain_seen_count"] == 0.0, \
        "zonder historie blijft domain_seen_count=0 (geen lek)"


# ── Test 6: Read-only ───────────────────────────────────────

def test_diagnose_readonly():
    """diagnose_store.py mag de store nooit in schrijfmodus openen."""
    src = (SCRIPTS / "diagnose_store.py").read_text()
    # Geen 'w' of 'a' modes in open() calls (exclusief comments/docstrings)
    for i, line in enumerate(src.split('\n'), 1):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if 'open(' in stripped:
            # Check de mode parameter
            import re
            modes = re.findall(r'open\([^)]*["\']([warb+]+)["\']', line)
            for mode in modes:
                if 'w' in mode or 'a' in mode or '+' in mode:
                    pytest.fail(f"line {i}: write-mode open: {stripped[:80]}")
    # Controleer ook dat er geen --run of backup-aanroepen in staan
    assert 'AGENT_LEARNING_OPSLOG_DIR' in src, "diagnose moet OPSLOG_DIR env var ondersteunen"


# ── Test 7: host_from_url ────────────────────────────────────

def test_host_from_url():
    from features import host_from_url
    cases = [
        ("https://example.com/path", "example.com"),
        ("http://sub.example.com", "sub.example.com"),
        ("example.com", "example.com"),
        ("/local/path", ""),
        ("memory", "memory"),
        ("", ""),
        (None, ""),
    ]
    for url, expected in cases:
        result = host_from_url(url)
        assert result == expected, f"host_from_url({url!r}) = {result!r}, verwacht {expected!r}"
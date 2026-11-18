from __future__ import annotations

import json
import random
import time
from pathlib import Path
from urllib.parse import quote

import hermes_plugin
import opslog
import shadow_policy


def _safe(value: str) -> str:
    return quote(str(value), safe="").replace(".", "%2E") or "_empty_"


def _write_snapshot(root: Path, snapshot: dict) -> None:
    aid = snapshot.get("agent_id", "hermes")
    pid = snapshot["id"]
    tid = snapshot["task_id"]
    policy_dir = root / "policies" / _safe(aid)
    active_dir = root / "active-policies" / _safe(aid)
    policy_dir.mkdir(parents=True, exist_ok=True)
    active_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / f"{_safe(pid)}.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (active_dir / f"{_safe(tid)}.json").write_text(
        json.dumps({"agent_id": aid, "task_id": tid, "policy_id": pid}),
        encoding="utf-8",
    )


def _scrape_snapshot(**overrides) -> dict:
    names = [
        "url_host_nieuws",
        "url_host_code",
        "url_host_social",
        "url_host_blog",
        "url_host_overig",
        "url_path_depth",
        "url_has_query",
        "domain_seen_count",
        "domain_last_success",
        "domain_prior_rate",
    ]
    row = {
        "id": "pol-scrape",
        "agent_id": "hermes",
        "task_id": "choose-scrape-strategy",
        "version": 3,
        "actions": [
            {"id": "use_webclaw_scrape"},
            {"id": "use_jina_reader"},
            {"id": "use_browser"},
        ],
        "logits": {
            "use_webclaw_scrape": 0.0,
            "use_jina_reader": 0.0,
            "use_browser": 0.0,
        },
        "metadata": {
            "policy_kind": "contextual_softmax",
            "feature_dim": 10,
            "feature_names": names,
            "feature_track": "pre_call",
            "context_weights": {
                "use_webclaw_scrape": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                "use_jina_reader": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "use_browser": [0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "created_at": "2026-11-18T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_pre_hook_returns_none_for_random_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    shadow_policy.reset_cache()
    rng = random.Random(20261118)
    tools = ["web_extract", "web_search", "browser_navigate", "", None, 12, "terminal"]
    for i in range(1000):
        kwargs = {
            "tool_name": rng.choice(tools),
            "args": rng.choice([None, "bad", {}, {"url": "https://example.com"}]),
            "session_id": rng.choice(["", "s", None]),
            "tool_call_id": rng.choice(["", f"c{i}", None]),
            "turn_id": rng.choice(["", "t", None]),
            "status": rng.choice(["ok", "error", None]),
        }
        assert hermes_plugin.on_pre_tool_call(**kwargs) is None


def test_policy_lookup_exception_returns_none_and_is_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    shadow_policy.reset_cache()

    def boom(task_id: str):
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(shadow_policy, "cached_snapshot", boom)
    assert hermes_plugin.on_pre_tool_call(tool_name="web_extract") is None
    h = opslog.health()
    assert h["ok"] is False
    assert "lookup failed" in h["last_error"]


def test_hook_under_50ms_cold_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    _write_snapshot(tmp_path, _scrape_snapshot())
    shadow_policy.reset_cache()
    start = time.perf_counter()
    result = hermes_plugin.on_pre_tool_call(
        tool_name="web_extract",
        args={"url": "https://example.com/page"},
        session_id="s1",
        tool_call_id="c1",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert result is None
    assert elapsed_ms < 50


def test_kill_switch_skips_advice_and_log(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_SHADOW_KILL", "1")
    _write_snapshot(tmp_path, _scrape_snapshot())
    shadow_policy.reset_cache()
    assert hermes_plugin.on_pre_tool_call(
        tool_name="web_extract",
        args={"url": "https://example.com"},
        tool_call_id="c1",
        session_id="s1",
    ) is None
    hermes_plugin.on_post_tool_call(
        tool_name="web_extract",
        status="ok",
        duration_ms=12,
        tool_call_id="c1",
        session_id="s1",
    )
    assert not Path(shadow_policy.log_path()).exists()
    assert shadow_policy.load_decisions() == []


def test_post_joins_one_decision_on_tool_call_id(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    _write_snapshot(tmp_path, _scrape_snapshot())
    shadow_policy.reset_cache()
    hermes_plugin.on_pre_tool_call(
        tool_name="web_extract",
        args={"url": "https://example.com"},
        session_id="s1",
        tool_call_id="c9",
    )
    hermes_plugin.on_post_tool_call(
        tool_name="web_extract",
        status="ok",
        duration_ms=41.2,
        session_id="s1",
        tool_call_id="c9",
    )
    rows = shadow_policy.load_decisions()
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "shadow"
    assert row["record"] == "decision"
    assert row["requested_tool"] == "web_extract"
    assert row["tool_call_id"] == "c9"
    assert row["status"] == "ok"
    assert row["duration_ms"] == 41
    assert row["advised_tool"]
    assert row["probabilities"]
    assert row["feature_track"]
    assert set(row["feature_track"]) <= set(row["context_features"])
    ops_path = tmp_path / "operations.jsonl"
    ops = [json.loads(line) for line in ops_path.read_text().splitlines() if line.strip()]
    assert all(op.get("metadata", {}).get("source") != "shadow" for op in ops)


def test_shadow_not_in_opslog_when_opslog_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_ENABLED", "0")
    _write_snapshot(tmp_path, _scrape_snapshot())
    shadow_policy.reset_cache()
    hermes_plugin.on_pre_tool_call(
        tool_name="web_extract",
        session_id="s",
        tool_call_id="c",
    )
    hermes_plugin.on_post_tool_call(
        tool_name="web_extract",
        status="ok",
        session_id="s",
        tool_call_id="c",
    )
    assert not (tmp_path / "operations.jsonl").exists()
    assert len(shadow_policy.load_decisions()) == 1


def test_advise_argmax_contextual():
    snap = _scrape_snapshot()
    features = {
        "url_host_nieuws": 0.0,
        "url_host_code": 1.0,
        "url_host_social": 0.0,
        "url_host_blog": 0.0,
        "url_host_overig": 0.0,
        "url_path_depth": 0.0,
        "url_has_query": 0.0,
        "domain_seen_count": 0.0,
        "domain_last_success": 0.5,
        "domain_prior_rate": 0.5,
    }
    advice = shadow_policy.advise(snap, features, task_id="choose-scrape-strategy")
    assert advice["advised_tool"] == "use_jina_reader"
    assert advice["probabilities"]["use_jina_reader"] > advice["probabilities"]["use_browser"]
    assert advice["feature_track"][1] == "url_host_code"


def test_advise_thompson_uses_means():
    snap = {
        "id": "pol-t",
        "task_id": "web-research",
        "version": 1,
        "actions": [{"id": "web_search"}, {"id": "skip"}],
        "logits": {},
        "metadata": {
            "policy_kind": "thompson",
            "thompson_alpha": {"web_search": 9.0, "skip": 1.0},
            "thompson_beta": {"web_search": 1.0, "skip": 9.0},
        },
    }
    advice = shadow_policy.advise(snap, {}, task_id="web-research")
    assert advice["advised_tool"] == "web_search"
    assert advice["probabilities"]["web_search"] == 0.9


def test_load_snapshot_from_active_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    snap = _scrape_snapshot()
    _write_snapshot(tmp_path, snap)
    loaded = shadow_policy.load_snapshot("choose-scrape-strategy")
    assert loaded is not None
    assert loaded["id"] == "pol-scrape"
    assert loaded["version"] == 3

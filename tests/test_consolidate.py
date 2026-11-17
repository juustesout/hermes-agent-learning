from __future__ import annotations

import json
from pathlib import Path

import consolidate_ops
import opslog
from cost_aware_rewards import CostAwareReward
from workflow_dag import NodeResult


def _patch_store(monkeypatch, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    ops_path = store / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(store))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(ops_path))
    monkeypatch.setattr(consolidate_ops, "STORE_DIR", str(store))
    monkeypatch.setattr(consolidate_ops, "OPSLOG_PATH", str(ops_path))
    monkeypatch.setattr(consolidate_ops, "CONSOLIDATED_LOG", str(store / "consolidated.txt"))
    monkeypatch.setattr(
        consolidate_ops, "EPISODES_DIR", str(store / "episodes" / "hermes")
    )
    return store


def test_op_to_episode_fills_full_fields_on_failure():
    ep = consolidate_ops.op_to_episode({
        "id": "fail-1",
        "task_id": "choose-scrape-strategy",
        "action_id": "use_jina_reader",
        "target": "",
        "execution_status": "failed",
        "result_summary": "",
        "action_type": "web_scrape",
    })
    assert ep["intent_summary"] == "use_jina_reader → no-target"
    assert ep["expected_outcome"]
    assert ep["result_summary"] == "failed"
    assert ep["action_name"] == "use_jina_reader"
    assert ep["execution_status"] == "failed"


def test_op_to_episode_round_trip_fields():
    op = {
        "id": "abc",
        "task_id": "choose-scrape-strategy",
        "action_id": "use_webclaw_scrape",
        "target": "example.com",
        "execution_status": "completed",
        "result_summary": "ok",
        "action_type": "web_scrape",
        "action_logprob": -0.2,
        "request_latency_ms": 1200,
        "context_features": {"cloudflare": 0.0, "model": "x"},
        "metadata": {"task_completed": True, "cost_credits": 1.0, "content_length": 10},
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    ep = consolidate_ops.op_to_episode(op)
    assert ep["id"] == "abc"
    assert ep["agent_id"] == "hermes"
    assert ep["task_id"] == "choose-scrape-strategy"
    assert ep["action_id"] == "use_webclaw_scrape"
    assert ep["action_logprob"] == -0.2
    assert ep["context_features"]["cloudflare"] == 0.0
    assert ep["model_deployment"] == "x"
    assert ep["metadata"]["cost_credits"] == 1.0
    assert ep["execution_status"] == "completed"


def test_consolidate_dry_run_does_not_write(tmp_path, monkeypatch):
    store = _patch_store(monkeypatch, tmp_path)
    opslog.log_operation(task_id="t", action_id="a", success=True, duration_ms=100, cost=0.5)
    result = consolidate_ops.consolidate(dry_run=True)
    assert result["new"] == 1
    assert result["dry_run"] is True
    ep_dir = store / "episodes" / "hermes"
    if ep_dir.exists():
        assert list(ep_dir.glob("*.json")) == []
    assert not (store / "consolidated.txt").exists()
    assert not (store / "rewards" / "hermes").exists()


def test_consolidate_writes_episode_and_reward(tmp_path, monkeypatch):
    store = _patch_store(monkeypatch, tmp_path)
    oid = opslog.log_operation(
        task_id="choose-scrape-strategy",
        action_id="use_jina_reader",
        success=True,
        duration_ms=2000,
        cost=0.0,
        summary="ok",
        action_logprob=-0.1,
        context_features={"cloudflare": 0.0},
    )
    result = consolidate_ops.consolidate(dry_run=False)
    assert result["new"] == 1
    ep_path = store / "episodes" / "hermes" / f"{oid}.json"
    rw_path = store / "rewards" / "hermes" / f"{oid}.json"
    assert ep_path.exists()
    assert rw_path.exists()
    ep = json.loads(ep_path.read_text())
    rewards = json.loads(rw_path.read_text())
    assert ep["id"] == oid
    assert ep["agent_id"] == "hermes"
    assert rewards[0]["episode_id"] == oid
    assert rewards[0]["agent_id"] == "hermes"
    assert rewards[0]["source"] == "aggregate"
    assert -1.0 <= rewards[0]["value"] <= 1.0
    assert rewards[0]["metadata"]["reward_preset"] == "balanced"


def test_consolidate_idempotent(tmp_path, monkeypatch):
    _patch_store(monkeypatch, tmp_path)
    opslog.log_operation(task_id="t", action_id="a", duration_ms=100)
    first = consolidate_ops.consolidate(dry_run=False)
    second = consolidate_ops.consolidate(dry_run=False)
    assert first["new"] == 1
    assert second["new"] == 0


def test_failed_op_lower_reward(tmp_path, monkeypatch):
    store = _patch_store(monkeypatch, tmp_path)
    ok_id = opslog.log_operation(
        task_id="t", action_id="draft", success=True, duration_ms=100, cost=0.1
    )
    fail_id = opslog.log_operation(
        task_id="t", action_id="draft", success=False, duration_ms=100, cost=0.1
    )
    consolidate_ops.consolidate(dry_run=False)
    ok = json.loads((store / "rewards" / "hermes" / f"{ok_id}.json").read_text())[0]["value"]
    fail = json.loads((store / "rewards" / "hermes" / f"{fail_id}.json").read_text())[0]["value"]
    assert fail < ok


def test_write_reward_uses_annotate_execution(tmp_path, monkeypatch):
    store = _patch_store(monkeypatch, tmp_path)
    ep = {
        "id": "e-annot",
        "action_id": "draft",
        "task_id": "opslog",
        "request_latency_ms": 1000,
        "metadata": {"cost_credits": 0.1},
        "execution_status": "completed",
    }
    consolidate_ops._write_reward(ep)
    recorded = json.loads((store / "rewards" / "hermes" / "e-annot.json").read_text())[0]

    ctx = {}
    CostAwareReward.annotate_execution(ctx, "opslog", "draft", 1.0, cost=0.1)
    expected, _ = CostAwareReward().compute(
        ctx,
        [NodeResult(
            node_id="opslog", node_name="opslog", task_id="opslog",
            action_id="draft", action_name="draft", execution_status="completed",
        )],
    )
    assert recorded["value"] == round(expected, 4)

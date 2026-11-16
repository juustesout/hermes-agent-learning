from __future__ import annotations

import pytest

from cost_aware_rewards import (
    CostAwareReward,
    RewardFactors,
    balanced_reward,
    cost_focused,
    estimate_cost,
    estimate_latency,
    quality_focused,
    speed_focused,
)
from workflow_dag import NodeResult


def _node(node_id: str, action_id: str, status: str = "completed") -> NodeResult:
    return NodeResult(
        node_id=node_id,
        node_name=node_id,
        task_id=node_id,
        action_id=action_id,
        action_name=action_id,
        execution_status=status,
    )


def test_empty_nodes_returns_zero():
    r, reason = CostAwareReward().compute({}, [])
    assert r == 0.0
    assert "no nodes" in reason


def test_all_completed_positive_reward():
    reward_fn = balanced_reward()
    nodes = [_node("n1", "draft"), _node("n2", "skip")]
    ctx = {}
    CostAwareReward.annotate_execution(ctx, "n1", "draft", 0.5, cost=0.1)
    CostAwareReward.annotate_execution(ctx, "n2", "skip", 0.1, cost=0.0)
    value, reason = reward_fn.compute(ctx, nodes)
    assert value > 0
    assert "2/2 ok" in reason


def test_all_failed_lower_than_success():
    reward_fn = CostAwareReward()
    ok_nodes = [_node("n1", "draft")]
    fail_nodes = [_node("n1", "draft", status="failed")]
    ctx_ok = {}
    ctx_fail = {}
    CostAwareReward.annotate_execution(ctx_ok, "n1", "draft", 1.0, cost=0.1)
    CostAwareReward.annotate_execution(ctx_fail, "n1", "draft", 1.0, cost=0.1)
    ok, _ = reward_fn.compute(ctx_ok, ok_nodes)
    fail, _ = reward_fn.compute(ctx_fail, fail_nodes)
    assert fail < ok


def test_high_latency_penalised():
    reward_fn = CostAwareReward(weight_latency=1.0, weight_cost=0.0, weight_quality=0.0)
    nodes = [_node("n1", "draft")]
    fast, slow = {}, {}
    CostAwareReward.annotate_execution(fast, "n1", "draft", 1.0, cost=0.0)
    CostAwareReward.annotate_execution(slow, "n1", "draft", 60.0, cost=0.0)
    r_fast, _ = reward_fn.compute(fast, nodes)
    r_slow, reason = reward_fn.compute(slow, nodes)
    assert r_slow < r_fast
    assert "slow" in reason or "lat=" in reason


def test_high_cost_penalised():
    reward_fn = CostAwareReward(weight_cost=1.0, weight_latency=0.0, weight_quality=0.0)
    cheap_nodes = [_node("n1", "draft")]
    costly_nodes = [_node("n1", "explainer_real")]
    cheap, costly = {}, {}
    CostAwareReward.annotate_execution(cheap, "n1", "draft", 1.0, cost=0.1)
    CostAwareReward.annotate_execution(costly, "n1", "explainer_real", 1.0, cost=15.0)
    r_cheap, _ = reward_fn.compute(cheap, cheap_nodes)
    r_costly, reason = reward_fn.compute(costly, costly_nodes)
    assert r_costly < r_cheap
    assert "cost=" in reason


def test_reward_clamped_to_unit_interval():
    reward_fn = CostAwareReward(
        weight_completion=10.0, weight_latency=0.0, weight_cost=0.0, weight_quality=10.0
    )
    nodes = [_node("n1", "draft")]
    ctx = {"_quality_score": 10.0}
    CostAwareReward.annotate_execution(ctx, "n1", "draft", 0.0, cost=0.0)
    value, _ = reward_fn.compute(ctx, nodes)
    assert -1.0 <= value <= 1.0


def test_annotate_execution_writes_context_keys():
    ctx = {}
    CostAwareReward.annotate_execution(ctx, "audio", "tts_edge", 3.2)
    assert ctx["_latency:audio"] == 3.2
    assert ctx["_cost:audio"] == estimate_cost("tts_edge")


def test_annotate_execution_honours_explicit_cost():
    ctx = {}
    CostAwareReward.annotate_execution(ctx, "n", "use_browser", 1.0, cost=9.9)
    assert ctx["_cost:n"] == 9.9


def test_node_reward_success_vs_failure():
    fn = CostAwareReward()
    ok = fn.node_reward("n", "draft", "completed", latency_s=1.0, cost=0.1)
    fail = fn.node_reward("n", "draft", "failed", latency_s=1.0, cost=0.1)
    assert isinstance(ok, RewardFactors)
    assert ok.completion == 1.0
    assert fail.completion == -0.5
    assert fail.total < ok.total


def test_presets_differ_in_weights():
    b = balanced_reward()
    s = speed_focused()
    c = cost_focused()
    q = quality_focused()
    assert s.w_l > b.w_l
    assert c.w_k > b.w_k
    assert q.w_q > b.w_q


def test_estimate_unknown_action_fallback():
    assert estimate_cost("unknown_tool_xyz") == 1.0
    assert estimate_latency("unknown_tool_xyz") == 5.0

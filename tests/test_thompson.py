from __future__ import annotations

import pytest

from agent_learning.types import Action, Episode, Reward, RewardSource
from thompson_policy import (
    ThompsonLearner,
    ThompsonPolicy,
    _reward_to_beta_obs,
)


def test_beta_prior_is_uniform():
    p = ThompsonPolicy.from_actions([Action(id="a")], agent_id="t", task_id="t")
    a, b = p.get_params("a")
    assert a == 1.0 and b == 1.0


def test_no_double_count():
    p = ThompsonPolicy.from_actions([Action(id="a")], agent_id="t", task_id="t")
    l = ThompsonLearner()
    ep = Episode(
        id="e1",
        agent_id="t",
        task_id="t",
        action_id="a",
        action_logprob=None,
        execution_status="completed",
        result_summary="ok",
    )
    rw = Reward(episode_id="e1", value=0.0, source=RewardSource.AGGREGATE)
    l.update(p, [ep], [rw])
    a, b = p.get_params("a")
    assert (a, b) == (1.5, 1.5)


def test_observe_success_increments_alpha():
    p = ThompsonPolicy.from_actions([Action(id="a")], agent_id="t", task_id="t")
    p.observe("a", reward=1.0)
    a, b = p.get_params("a")
    assert a == pytest.approx(2.0)
    assert b == pytest.approx(1.0)


def test_observe_failure_increments_beta():
    p = ThompsonPolicy.from_actions([Action(id="a")], agent_id="t", task_id="t")
    p.observe("a", reward=-1.0)
    a, b = p.get_params("a")
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(2.0)


def test_reward_to_beta_obs_clamps():
    assert _reward_to_beta_obs(1.0) == (1.0, 0.0)
    assert _reward_to_beta_obs(-1.0) == (0.0, 1.0)
    assert _reward_to_beta_obs(0.0) == (0.5, 0.5)
    s, f = _reward_to_beta_obs(5.0)
    assert s == 1.0 and f == 0.0


def test_update_skips_missing_reward():
    p = ThompsonPolicy.from_actions([Action(id="a")], agent_id="t", task_id="t")
    l = ThompsonLearner()
    ep = Episode(id="e1", agent_id="t", task_id="t", action_id="a")
    result = l.update(p, [ep], [])
    assert result["episodes_used"] == 0
    assert p.get_params("a") == (1.0, 1.0)


def test_update_skips_none_action_id():
    p = ThompsonPolicy.from_actions([Action(id="a")], agent_id="t", task_id="t")
    l = ThompsonLearner()
    ep = Episode(id="e1", agent_id="t", task_id="t", action_id=None)
    rw = Reward(episode_id="e1", value=1.0, source=RewardSource.AGGREGATE)
    result = l.update(p, [ep], [rw])
    assert result["episodes_used"] == 0


def test_failure_boost_applied():
    p = ThompsonPolicy.from_actions([Action(id="a")], agent_id="t", task_id="t")
    l = ThompsonLearner(lr_boost=3.0)
    ep = Episode(
        id="e1",
        agent_id="t",
        task_id="t",
        action_id="a",
        execution_status="failed",
    )
    rw = Reward(episode_id="e1", value=-1.0, source=RewardSource.AGGREGATE)
    result = l.update(p, [ep], [rw])
    assert result["boost_applied"] is True
    a, b = p.get_params("a")
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(4.0)


def test_beta_decay_fades_old_evidence():
    p = ThompsonPolicy.from_actions(
        [Action(id="a"), Action(id="b")], agent_id="t", task_id="t"
    )
    p.observe("a", reward=1.0)
    p.observe("a", reward=1.0)
    a_before, b_before = p.get_params("a")
    unused_a, unused_b = p.get_params("b")
    l = ThompsonLearner(beta_decay=0.5)
    ep = Episode(
        id="e1",
        agent_id="t",
        task_id="t",
        action_id="b",
        execution_status="completed",
    )
    rw = Reward(episode_id="e1", value=1.0, source=RewardSource.AGGREGATE)
    l.update(p, [ep], [rw])
    a_after, beta_after = p.get_params("a")
    assert a_after == pytest.approx(a_before * 0.5)
    assert beta_after == pytest.approx(b_before * 0.5)
    b_a, b_beta = p.get_params("b")
    assert b_a == pytest.approx(unused_a * 0.5 + 1.0)
    assert b_beta == pytest.approx(unused_b * 0.5)


def test_beta_decay_unupdated_only():
    p = ThompsonPolicy.from_actions(
        [Action(id="a"), Action(id="b")], agent_id="t", task_id="t"
    )
    p.observe("a", reward=1.0)
    a_before, _ = p.get_params("a")
    l = ThompsonLearner(beta_decay=0.5, decay_unupdated_only=True)
    ep = Episode(
        id="e1",
        agent_id="t",
        task_id="t",
        action_id="a",
        execution_status="completed",
    )
    rw = Reward(episode_id="e1", value=1.0, source=RewardSource.AGGREGATE)
    l.update(p, [ep], [rw])
    a_after, _ = p.get_params("a")
    assert a_after == pytest.approx(a_before + 1.0)


def test_choose_returns_known_action():
    p = ThompsonPolicy.from_actions(
        [Action(id="a"), Action(id="b")],
        agent_id="t",
        task_id="t",
        rng=__import__("random").Random(0),
    )
    decision = p.choose()
    assert decision.action.id in {"a", "b"}
    assert decision.logprob <= 0.0


def test_learner_rejects_non_thompson_policy():
    l = ThompsonLearner()
    with pytest.raises(TypeError):
        l.update(object(), [], [])

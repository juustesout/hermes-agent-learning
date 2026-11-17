from __future__ import annotations

from agent_learning.types import Action
from holdout_eval import (
    BASELINES,
    DEFAULT_TEST_SET,
    EvalResult,
    HoldoutEvaluator,
    TOOL_COST,
)
from thompson_policy import ThompsonPolicy


def test_default_test_set_has_ten_cases():
    assert len(DEFAULT_TEST_SET) == 10


def test_four_baselines_registered():
    assert set(BASELINES) == {
        "always_first",
        "always_webclaw",
        "cheapest",
        "fastest",
    }


def test_always_webclaw_metrics():
    ev = HoldoutEvaluator()
    result = ev.evaluate_baseline("always_webclaw")
    assert result.total == 10
    assert result.action_counts == {"use_webclaw_scrape": 10}
    expected_correct = sum(1 for _, _, correct, _ in DEFAULT_TEST_SET if correct == "use_webclaw_scrape")
    assert result.correct == expected_correct
    assert result.accuracy == expected_correct / 10
    assert result.total_cost == 10 * TOOL_COST["use_webclaw_scrape"]
    assert "wordpress" in result.per_category


def test_cheapest_picks_jina():
    ev = HoldoutEvaluator()
    result = ev.evaluate_baseline("cheapest")
    assert result.action_counts.get("use_jina_reader") == 10


def test_fastest_picks_jina():
    ev = HoldoutEvaluator()
    result = ev.evaluate_baseline("fastest")
    assert result.action_counts.get("use_jina_reader") == 10


def test_unknown_baseline_records_error():
    ev = HoldoutEvaluator()
    result = ev.evaluate_baseline("not_a_baseline")
    assert result.errors
    assert result.total == 0


def test_baseline_cache_returns_same_object():
    ev = HoldoutEvaluator()
    a = ev.evaluate_baseline("always_first")
    b = ev.evaluate_baseline("always_first")
    assert a is b


def test_print_comparison_includes_delta():
    ev = HoldoutEvaluator()
    learned = EvalResult(policy_name="learned", total=10, correct=8, accuracy=0.8, total_cost=5.0)
    baseline = ev.evaluate_baseline("always_webclaw")
    text = ev.print_comparison(learned, baseline)
    assert "Accuracy" in text
    assert "Delta accuracy" in text
    assert "  80.0%" in text
    assert "0.0 %" not in text
    assert "% 50.0%" not in text


def test_evaluate_learned_missing_policy():
    from agent_learning.storage.local import LocalFileStore
    import tempfile

    store = LocalFileStore(root_dir=tempfile.mkdtemp(prefix="holdout_"))
    ev = HoldoutEvaluator()
    result = ev.evaluate_learned(store, agent_id="hermes", task_id="choose-scrape-strategy")
    assert "No active policy" in result.errors[0]


def test_evaluate_learned_thompson_policy(store):
    actions = [
        Action(id="use_webclaw_scrape"),
        Action(id="use_jina_reader"),
        Action(id="use_browser"),
    ]
    policy = ThompsonPolicy.from_actions(
        actions, agent_id="hermes", task_id="choose-scrape-strategy"
    )
    store.store_policy(policy.snapshot())
    ev = HoldoutEvaluator()
    result = ev.evaluate_learned(store, agent_id="hermes", task_id="choose-scrape-strategy")
    assert not result.errors
    assert result.total == 10
    assert result.accuracy == result.correct / result.total
    assert sum(result.action_counts.values()) == 10

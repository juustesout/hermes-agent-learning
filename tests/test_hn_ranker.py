from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import blind_label
import hn_ranker


def _isolate(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "store"))
    monkeypatch.delenv(hn_ranker.KILL_SWITCH_ENV, raising=False)
    return evidence


def _item(item_id: str, rank: int, rag: float, **extra) -> dict:
    payload = {
        "id": item_id,
        "title": f"Title {item_id} about AI agents",
        "url": extra.pop("url", f"https://github.com/example/{item_id}"),
        "summary": "AI development tooling.",
        "source_rank": rank,
        "rag_score": rag,
        "hn_points": extra.pop("hn_points", 10),
        "hn_comments": extra.pop("hn_comments", 2),
        "age_hours": extra.pop("age_hours", 5),
    }
    payload.update(extra)
    return payload


GOLDEN_POOL = [
    _item("a", 0, 0.10, hn_points=50),
    _item("b", 1, 0.90, hn_points=5),
    _item("c", 2, 0.80, hn_points=20),
    _item("d", 3, 0.20, hn_points=1),
]


def test_learned_differs_from_baseline_on_fixture():
    baseline = hn_ranker.baseline_select_ids(GOLDEN_POOL, n=2)
    learned = hn_ranker.select_run(
        GOLDEN_POOL,
        "run-diff",
        force_arms=["L", "L"],
        write=False,
        model={"model_version": hn_ranker.PRIOR_VERSION},
    )
    learned_ids = [row["chosen_id"] for row in learned]
    assert baseline == ["a", "b"]
    assert learned_ids == ["b", "c"]
    assert learned_ids != baseline


def test_baseline_matches_frozen_source_order():
    assert hn_ranker.baseline_select_ids(GOLDEN_POOL) == ["a", "b"]
    decisions = hn_ranker.select_run(
        GOLDEN_POOL,
        "run-base",
        force_arms=["B", "B"],
        write=False,
        model={"model_version": hn_ranker.PRIOR_VERSION},
    )
    assert [row["chosen_id"] for row in decisions] == ["a", "b"]
    assert [row["arm"] for row in decisions] == ["B", "B"]
    assert decisions[0]["baseline_pick_id"] == "a"
    assert decisions[1]["baseline_pick_id"] == "b"


def test_run_never_repeats_an_item():
    decisions = hn_ranker.select_run(
        GOLDEN_POOL,
        "run-unique",
        force_arms=["L", "B"],
        write=False,
    )
    ids = [row["chosen_id"] for row in decisions]
    assert len(ids) == len(set(ids))
    assert decisions[0]["chosen_id"] not in decisions[1]["pool_ids"]


def test_short_pool_does_not_crash(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    decisions = hn_ranker.select_run(
        GOLDEN_POOL[:1],
        "run-short",
        force_arms=["L", "L"],
        write=True,
    )
    assert len(decisions) == 1
    assert decisions[0]["chosen_id"] == "a"
    logged = blind_label.load_decisions()
    assert len(logged) == 1
    audit = blind_label.audit(expected_posts=8, hours=24)
    assert audit["decisions"] == 1
    assert "posted_below_expected" in audit["alarms"]


def test_feature_allowlist_rejects_forbidden_fields():
    assert hn_ranker.FEATURE_ALLOWLIST.isdisjoint(hn_ranker.FORBIDDEN_FIELDS)
    with pytest.raises(ValueError, match="forbidden feature fields"):
        hn_ranker.ensure_allowed({"rag_score": 0.1, "auto_relevant": True, "domain_cat": "github"})
    with pytest.raises(ValueError, match="unknown feature fields"):
        hn_ranker.ensure_allowed({"rag_score": 0.1, "posted_score": 1, "domain_cat": "github"})


def test_posted_item_has_matching_decision(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    decisions = hn_ranker.select_run(
        GOLDEN_POOL,
        "run-match",
        force_arms=["B", "L"],
        write=True,
    )
    for row in decisions:
        chosen = next(item for item in GOLDEN_POOL if item["id"] == row["chosen_id"])
        blind_label.enqueue_item(
            {
                **chosen,
                "http_status": 201,
                "run_id": "run-match",
                "auto_relevant": True,
            }
        )
    items = {row["id"] for row in blind_label.load_items()}
    chosen = {row["chosen_id"] for row in decisions}
    outcomes = {row["chosen_id"] for row in blind_label.load_outcomes()}
    assert items == chosen == outcomes
    audit = blind_label.audit(expected_posts=2, hours=24)
    assert "posted_without_decision" not in audit["alarms"]
    assert "items_assignments_mismatch" not in audit["alarms"]


def test_retrain_rejects_labels_after_cutoff():
    cutoff = datetime(2026, 11, 16, tzinfo=timezone.utc)
    rows = [
        {
            "id": "early",
            "auto_relevant": True,
            "labeled_at": "2026-11-15T00:00:00+00:00",
            "features": hn_ranker.extract_features(GOLDEN_POOL[0]),
        },
        {
            "id": "late",
            "auto_relevant": False,
            "labeled_at": (cutoff + timedelta(days=1)).isoformat(),
            "features": hn_ranker.extract_features(GOLDEN_POOL[1]),
        },
    ]
    with pytest.raises(ValueError, match="after cutoff"):
        hn_ranker.select_training_rows(rows, cutoff)


def test_kill_switch_forces_baseline(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setenv(hn_ranker.KILL_SWITCH_ENV, "1")
    decisions = hn_ranker.select_run(
        GOLDEN_POOL,
        "run-kill",
        force_arms=["L", "L"],
        write=True,
        model={"model_version": hn_ranker.PRIOR_VERSION},
    )
    assert [row["arm"] for row in decisions] == ["B", "B"]
    assert [row["chosen_id"] for row in decisions] == ["a", "b"]
    assert all(row["kill_switch"] for row in decisions)
    audit = blind_label.audit(expected_posts=8, hours=24)
    assert "kill_switch" in audit["alarms"]


def test_ranker_exception_falls_back(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("ranker exploded")

    monkeypatch.setattr(hn_ranker, "_pick_learned", boom)
    decisions = hn_ranker.select_run(
        GOLDEN_POOL,
        "run-fallback",
        force_arms=["L", "L"],
        write=True,
    )
    assert [row["arm"] for row in decisions] == ["B", "B"]
    assert [row["chosen_id"] for row in decisions] == ["a", "b"]
    assert all(row["fallback"] is True for row in decisions)
    audit = blind_label.audit(expected_posts=8, hours=24)
    assert "ranker_fallback" in audit["alarms"]


def test_slot_coin_is_stable():
    first = hn_ranker.assign_slot_arm("run-x", 1)
    second = hn_ranker.assign_slot_arm("run-x", 1)
    assert first["arm"] == second["arm"]
    assert first["draw"] == second["draw"]
    assert first["choice_prob"] == 0.5


def test_prior_scores_by_rag():
    chosen, score = hn_ranker.ranker_pick(GOLDEN_POOL, {"model_version": hn_ranker.PRIOR_VERSION})
    assert chosen["id"] == "b"
    assert score == 0.9

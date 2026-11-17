from __future__ import annotations

import json

import pytest

import blind_label
import hn_ranker
import hn_sim


def _isolate(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "store"))
    monkeypatch.delenv(hn_ranker.KILL_SWITCH_ENV, raising=False)
    return evidence


def test_200_runs_write_decision_before_post(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    summary = hn_sim.run_sim(runs=200, error_rate=0.0, auto_label=True, seed=1)
    assert summary["runs"] == 200
    assert summary["jobs_posted"] == summary["wp_posts"]
    assert summary["posted"] == summary["decisions"]
    assert summary["posted"] == 400
    assert summary["jobs_failed"] == 0
    assert summary["missing_decision"] == []
    assert "posted_without_decision" not in summary["alarms"]
    assert "items_assignments_mismatch" not in summary["alarms"]
    assert summary["elapsed_s"] < 60
    items = {row["id"] for row in blind_label.load_items()}
    chosen = {row["chosen_id"] for row in hn_ranker.load_decisions()}
    outcomes = {row["chosen_id"] for row in hn_ranker.load_outcomes()}
    assert items == chosen == outcomes
    assert summary["decision_arms"]["L"] > 0
    assert summary["decision_arms"]["B"] > 0
    assert summary["labeled"] == summary["sampled"]
    assert summary["agreement"] is not None
    assert summary["agreement"] >= 0.9


def test_failures_are_enqueued(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    summary = hn_sim.run_sim(runs=40, error_rate=0.5, auto_label=False, seed=7)
    assert summary["jobs_failed"] > 0
    outcomes = hn_ranker.load_outcomes()
    failed = [row for row in outcomes if row.get("technical_success") is False]
    assert failed
    assert summary["jobs_posted"] == summary["wp_posts"]
    assert summary["decisions"] == summary["posted"]
    assert summary["posted"] == summary["jobs_posted"] + summary["jobs_failed"]


def test_learned_and_baseline_can_differ_in_sim_catalog():
    catalog = hn_sim.make_catalog(n=80, seed=1)
    pool = hn_sim.fetch_pool(catalog, set(), fetch_size=30, offset=0)
    baseline = hn_ranker.baseline_select_ids(pool, n=2)
    learned = hn_ranker.select_run(
        pool,
        "sim-diff",
        force_arms=["L", "L"],
        write=False,
        model={"model_version": hn_ranker.PRIOR_VERSION},
    )
    learned_ids = [row["chosen_id"] for row in learned]
    assert learned_ids
    assert learned_ids != baseline


def test_cli_run_uses_isolated_evidence(tmp_path, monkeypatch, capsys):
    evidence = _isolate(tmp_path, monkeypatch)
    rc = hn_sim.main(["run", "--runs", "4", "--seed", "2", "--no-auto-label"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc in {0, 2}
    assert payload["runs"] == 4
    assert (evidence / "assignments.jsonl").exists()
    assert (evidence / "items.jsonl").exists()

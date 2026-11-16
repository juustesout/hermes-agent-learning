from __future__ import annotations

from test_adaptive_vs_standard import make_learner, run_benchmark, run_simulation


def test_benchmark_tiny_multi_seed_smoke():
    report = run_benchmark(
        seeds=(42, 123),
        n_good=3,
        n_bad=3,
        kinds=("reinforce", "adaptive_full", "decay_only"),
    )
    assert set(report) == {"reinforce", "adaptive_full", "decay_only"}
    for row in report.values():
        assert 0.0 <= row["switch_rate"] <= 1.0
        assert row["n_good"] == 3
        assert len(row["per_seed"]) == 2


def test_simulation_history_length():
    learner = make_learner("reinforce")
    history = run_simulation(learner, 2, 2, seed=42)
    assert len(history) == 4
    assert history[0]["phase"] == "good"
    assert history[-1]["phase"] == "bad"

#!/usr/bin/env python3
"""
test_adaptive_vs_standard.py — Multi-seed benchmark + ablation.

Scenario (30/30 as originally documented):
  1. Clean SoftmaxPolicy (3 actions, uniform)
  2. 30 episodes where WebClaw is correct (reward=0.7)
  3. 30 episodes where WebClaw fails and Jina is correct
  4. Repeat over seeds and report mean ± stddev

Ablation isolates AdaptiveBanditLearner's three mechanisms:
  - decay_only: logit decay, no LR boost
  - failure_lr_only: failure detection + LR boost, no decay
  - lr_boost_hair_trigger: decay off, threshold=1 so any miss boosts LR
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AGENT_LEARNING_STORE_BACKEND", "local")
os.environ.setdefault(
    "AGENT_LEARNING_LOCAL_STORE_DIR",
    str(Path(tempfile.gettempdir()) / "agent-learning-test-store"),
)
for _k in (
    "AGENT_LEARNING_ADAPTIVE_LOGIT_DECAY",
    "AGENT_LEARNING_ADAPTIVE_FAILURE_THRESHOLD",
    "AGENT_LEARNING_ADAPTIVE_LR_BOOST",
):
    os.environ.pop(_k, None)

from agent_learning.config import LearnerConfig
from agent_learning.learners.reinforce import ReinforceLearner
from agent_learning.policy.softmax_bandit import SoftmaxPolicy
from agent_learning.types import Action, Episode, Reward, RewardSource

from adaptive_learner import AdaptiveBanditLearner

SEEDS: Sequence[int] = (42, 123, 256, 512, 1024)
N_GOOD = 30
N_BAD = 30

LEARNER_CONFIG = LearnerConfig(learning_rate=0.05, baseline_decay=0.9, entropy_bonus=0.01)


def make_policy() -> SoftmaxPolicy:
    actions = [
        Action(id="use_webclaw_scrape", description="WebClaw", parameters={"cost": 1}),
        Action(id="use_jina_reader", description="Jina", parameters={"cost": 0}),
        Action(id="use_browser", description="Browser", parameters={"cost": 0}),
    ]
    return SoftmaxPolicy.from_actions(actions, agent_id="hermes", task_id="scrape-test")


def make_episode(policy: SoftmaxPolicy, correct_action_id: str, task_done: bool) -> Tuple[Episode, Reward]:
    snapshot = policy.snapshot()
    action_ids = [a.id for a in snapshot.actions]
    probs = policy.probabilities()
    r = random.random()  # nosec B311
    cum = 0.0
    chosen_id = action_ids[-1]
    for idx, p in enumerate(probs):
        cum += p
        if r <= cum:
            chosen_id = action_ids[idx]
            break
    logprob = math.log(max(probs[action_ids.index(chosen_id)], 1e-12))

    reward_val = 0.7 if chosen_id == correct_action_id and task_done else -0.5
    if not task_done:
        reward_val = -0.8

    ep = Episode(
        id=f"ep-{random.randint(10000, 99999)}",  # nosec B311
        agent_id="hermes",
        task_id="scrape-test",
        action_id=chosen_id,
        action_logprob=logprob,
        execution_status="completed",
        metadata={
            "correct_action_id": correct_action_id,
            "task_completed": task_done,
        },
    )
    rw = Reward(
        id=f"rw-{random.randint(10000, 99999)}",  # nosec B311
        episode_id="",
        source=RewardSource.AGGREGATE,
        value=reward_val,
    )
    return ep, rw


def make_learner(kind: str):
    if kind == "reinforce":
        return ReinforceLearner(config=LEARNER_CONFIG)
    if kind == "adaptive_full":
        return AdaptiveBanditLearner(
            config=LEARNER_CONFIG, logit_decay=0.85, failure_threshold=2, lr_boost=5.0
        )
    if kind == "decay_only":
        return AdaptiveBanditLearner(
            config=LEARNER_CONFIG, logit_decay=0.85, failure_threshold=10_000, lr_boost=1.0
        )
    if kind == "failure_lr_only":
        return AdaptiveBanditLearner(
            config=LEARNER_CONFIG, logit_decay=1.0, failure_threshold=2, lr_boost=5.0
        )
    if kind == "lr_boost_hair_trigger":
        return AdaptiveBanditLearner(
            config=LEARNER_CONFIG, logit_decay=1.0, failure_threshold=1, lr_boost=5.0
        )
    raise ValueError(f"unknown learner kind: {kind}")


def run_simulation(
    learner,
    n_good: int,
    n_bad: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    policy = make_policy()
    history: List[Dict[str, Any]] = []

    for step in range(n_good):
        ep, rw = make_episode(policy, "use_webclaw_scrape", True)
        rw.episode_id = ep.id
        result = learner.update(policy, [ep], [rw])
        probs = policy.probabilities()
        action_ids = [a.id for a in policy.snapshot().actions]
        extra = getattr(result, "extra", {}) or {}
        history.append({
            "step": step + 1,
            "phase": "good",
            "action": ep.action_id,
            "reward": rw.value,
            "probs": {aid: float(p) for aid, p in zip(action_ids, probs)},
            "baseline": float(result.baseline_after),
            "extra": extra,
        })

    for step in range(n_good, n_good + n_bad):
        ep, rw = make_episode(policy, "use_jina_reader", True)
        rw.episode_id = ep.id
        result = learner.update(policy, [ep], [rw])
        probs = policy.probabilities()
        action_ids = [a.id for a in policy.snapshot().actions]
        extra = getattr(result, "extra", {}) or {}
        history.append({
            "step": step + 1,
            "phase": "bad",
            "action": ep.action_id,
            "reward": rw.value,
            "probs": {aid: float(p) for aid, p in zip(action_ids, probs)},
            "baseline": float(result.baseline_after),
            "extra": extra,
        })

    return history


def summarise_history(history: List[Dict[str, Any]], n_good: int) -> Dict[str, Any]:
    last = history[-1]["probs"]
    after_good = history[n_good - 1]["probs"]
    switched_at = None
    for h in history[n_good:]:
        p = h["probs"]
        if p.get("use_jina_reader", 0) > p.get("use_webclaw_scrape", 0):
            switched_at = h["step"] - n_good
            break
    return {
        "webclaw_after_good": after_good.get("use_webclaw_scrape", 0.0),
        "jina_after_good": after_good.get("use_jina_reader", 0.0),
        "webclaw_final": last.get("use_webclaw_scrape", 0.0),
        "jina_final": last.get("use_jina_reader", 0.0),
        "browser_final": last.get("use_browser", 0.0),
        "switched": last.get("use_jina_reader", 0.0) > last.get("use_webclaw_scrape", 0.0),
        "steps_to_switch": switched_at,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def run_benchmark(
    seeds: Sequence[int] = SEEDS,
    n_good: int = N_GOOD,
    n_bad: int = N_BAD,
    kinds: Sequence[str] = (
        "reinforce",
        "adaptive_full",
        "decay_only",
        "failure_lr_only",
        "lr_boost_hair_trigger",
    ),
) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for kind in kinds:
        per_seed = []
        for seed in seeds:
            learner = make_learner(kind)
            history = run_simulation(learner, n_good, n_bad, seed=seed)
            per_seed.append(summarise_history(history, n_good))
        jina = [s["jina_final"] for s in per_seed]
        webclaw = [s["webclaw_final"] for s in per_seed]
        after_good = [s["webclaw_after_good"] for s in per_seed]
        switch_steps = [s["steps_to_switch"] for s in per_seed if s["steps_to_switch"] is not None]
        jina_m, jina_s = mean_std(jina)
        web_m, web_s = mean_std(webclaw)
        good_m, good_s = mean_std(after_good)
        sw_m, sw_s = mean_std(switch_steps) if switch_steps else (float("nan"), float("nan"))
        report[kind] = {
            "seeds": list(seeds),
            "n_good": n_good,
            "n_bad": n_bad,
            "jina_final_mean": jina_m,
            "jina_final_std": jina_s,
            "webclaw_final_mean": web_m,
            "webclaw_final_std": web_s,
            "webclaw_after_good_mean": good_m,
            "webclaw_after_good_std": good_s,
            "switch_rate": sum(1 for s in per_seed if s["switched"]) / len(per_seed),
            "steps_to_switch_mean": sw_m,
            "steps_to_switch_std": sw_s,
            "per_seed": per_seed,
        }
    return report


def format_report(report: Dict[str, Dict[str, Any]]) -> str:
    labels = {
        "reinforce": "REINFORCE (baseline)",
        "adaptive_full": "Adaptive full (decay+fail+LR)",
        "decay_only": "Ablation: decay only",
        "failure_lr_only": "Ablation: failure detect + LR",
        "lr_boost_hair_trigger": "Ablation: LR boost (threshold=1)",
    }
    lines = [
        "=" * 88,
        "AdaptiveBandit vs REINFORCE — multi-seed benchmark + ablation",
        f"Episodes: {N_GOOD} good (WebClaw) + {N_BAD} bad (Jina)  seeds={list(SEEDS)}",
        "=" * 88,
        f"{'variant':<36} {'Jina final':>16} {'WebClaw final':>16} {'switch%':>8} {'steps':>10}",
        "-" * 88,
    ]
    for kind, row in report.items():
        jina = f"{row['jina_final_mean']:.3f}±{row['jina_final_std']:.3f}"
        web = f"{row['webclaw_final_mean']:.3f}±{row['webclaw_final_std']:.3f}"
        sw = f"{row['switch_rate']*100:.0f}%"
        if math.isnan(row["steps_to_switch_mean"]):
            steps = "never"
        else:
            steps = f"{row['steps_to_switch_mean']:.1f}±{row['steps_to_switch_std']:.1f}"
        lines.append(f"{labels.get(kind, kind):<36} {jina:>16} {web:>16} {sw:>8} {steps:>10}")
    lines.append("-" * 88)
    lines.append("switch% = fraction of seeds where final P(Jina) > P(WebClaw)")
    lines.append("steps   = mean episodes into the bad phase until Jina overtakes")
    return "\n".join(lines)


def main() -> int:
    report = run_benchmark()
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

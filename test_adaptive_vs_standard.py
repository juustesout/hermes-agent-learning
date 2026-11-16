#!/usr/bin/env python3
"""
test_adaptive_vs_standard.py — Vergelijk AdaptiveBanditLearner met standaard ReinforceLearner.

Scenario:
  1. Initialiseer een schone TaskPolicy (3 actions, uniform)
  2. Voer 10 episodes in waar WebClaw werkt (correct, reward=0.7)
  3. Train na elke episode en kijk hoe WebClaw dominant wordt
  4. Voer dan 5 episodes in waar WebClaw faalt, Jina werkt
  5. Kijk hoe snel de policy omslaat

Dit is een in-memory test — geen store of CLI nodig.
"""

from __future__ import annotations

import copy
import math
import random
import sys
import tempfile
from typing import Dict, List
from pathlib import Path

# Python path voor onze custom learner
import os
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AGENT_LEARNING_STORE_BACKEND", "local")
os.environ.setdefault(
    "AGENT_LEARNING_LOCAL_STORE_DIR",
    str(Path(tempfile.gettempdir()) / "agent-learning-test-store"),
)

from agent_learning.config import LearnerConfig
from agent_learning.learners.reinforce import ReinforceLearner
from agent_learning.policy.softmax_bandit import SoftmaxPolicy
from agent_learning.types import Action, Episode, PolicySnapshot, Reward, RewardSource

from adaptive_learner import AdaptiveBanditLearner


def make_policy() -> SoftmaxPolicy:
    actions = [
        Action(id="use_webclaw_scrape", description="WebClaw", parameters={"cost": 1}),
        Action(id="use_jina_reader", description="Jina", parameters={"cost": 0}),
        Action(id="use_browser", description="Browser", parameters={"cost": 0}),
    ]
    return SoftmaxPolicy.from_actions(actions, agent_id="hermes", task_id="scrape-test")


def make_episode(policy: SoftmaxPolicy, correct_action_id: str, task_done: bool) -> Episode:
    """Simuleer een episode: policy beslist, wij bepalen of het correct was."""
    snapshot = policy.snapshot()
    action_ids = [a.id for a in snapshot.actions]
    probs = policy.probabilities()
    # Sample from softmax (same as policy would)
    # Simulation randomness for learning behavior tests; crypto randomness is not required.
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

    return Episode(
        # Non-security identifier values for test episodes.
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
    ), Reward(
        id=f"rw-{random.randint(10000, 99999)}",  # nosec B311
        episode_id="",
        source=RewardSource.AGGREGATE,
        value=reward_val,
    )


def run_simulation(
    name: str,
    learner,
    n_good: int,
    n_bad: int,
    *,
    seed: int = 42,
) -> List[Dict]:
    """Run een simulatie en rapporteer probas na elke episode."""
    random.seed(seed)
    policy = make_policy()
    history = []

    for step in range(n_good):
        if step < n_good:  # WebClaw werkt
            ep, rw = make_episode(policy, "use_webclaw_scrape", True)
        rw.episode_id = ep.id
        result = learner.update(policy, [ep], [rw])
        probs = policy.probabilities()
        action_ids = [a.id for a in policy.snapshot().actions]
        history.append({
            "step": step + 1,
            "phase": "WebClaw werkt",
            "action": ep.action_id,
            "reward": rw.value,
            "probs": {aid: round(p, 4) for aid, p in zip(action_ids, probs)},
            "baseline": round(result.baseline_after, 4),
            "extra": result.extra,
        })

    for step in range(n_good, n_good + n_bad):
        # WebClaw faalt nu, Jina werkt
        ep, rw = make_episode(policy, "use_jina_reader", True)
        rw.episode_id = ep.id
        result = learner.update(policy, [ep], [rw])
        probs = policy.probabilities()
        action_ids = [a.id for a in policy.snapshot().actions]
        history.append({
            "step": step + 1,
            "phase": "WebClaw DOOD - Jina werkt",
            "action": ep.action_id,
            "reward": rw.value,
            "probs": {aid: round(p, 4) for aid, p in zip(action_ids, probs)},
            "baseline": round(result.baseline_after, 4),
            "extra": result.extra,
        })

    return history


# ============================================================================
# RUN
# ============================================================================

N_GOOD = 10
N_BAD = 5

print(f"{'='*80}")
print(f"VERGELIJKING: Standaard ReinforceLearner vs AdaptiveBanditLearner")
print(f"{'='*80}")
print(f"Scenario: {N_GOOD}x WebClaw werkt, daarna {N_BAD}x WebClaw faalt (Jina werkt)")
print()

# ---- Standaard learner ----------------------------------------------------
std_config = LearnerConfig(learning_rate=0.05, baseline_decay=0.9, entropy_bonus=0.01)
std_learner = ReinforceLearner(config=std_config)
std_history = run_simulation(
    "REINFORCE (standaard)", std_learner, N_GOOD, N_BAD, seed=42
)

# ---- Adaptive learner -----------------------------------------------------
ada_config = LearnerConfig(learning_rate=0.05, baseline_decay=0.9, entropy_bonus=0.01)
ada_learner = AdaptiveBanditLearner(
    config=ada_config,
    logit_decay=0.85,        # Sterk verval voor demo-doeleinden
    failure_threshold=2,     # Na 2 failures boost
    lr_boost=5.0,            # 5x learning rate bij regime change
)
# Dezelfde seed = dezelfde acties worden gekozen (eerlijke vergelijking)
ada_history = run_simulation(
    "AdaptiveBandit", ada_learner, N_GOOD, N_BAD, seed=42
)

# ---- Print resultaten -----------------------------------------------------
def print_history(history, label):
    print(f"\n{'─'*80}")
    print(f"  {label}")
    print(f"{'─'*80}")
    print(f"{'Step':>5} {'Phase':<25} {'Action':<20} {'Reward':>7} {'WebClaw':>9} {'Jina':>9} {'Browser':>9} {'Baseline':>9}")
    print(f"{'─'*80}")
    for h in history:
        p = h["probs"]
        extra = h.get("extra", {})
        spike = ""
        if extra.get("regime_change_detected"):
            spike = " 🚨REGIME CHANGE"
        print(
            f"{h['step']:>5} {h['phase']:<25} {h['action']:<20} {h['reward']:>7.2f} "
            f"{p.get('use_webclaw_scrape',0):>9.4f} {p.get('use_jina_reader',0):>9.4f} "
            f"{p.get('use_browser',0):>9.4f} {h['baseline']:>9.4f}{spike}"
        )

print_history(std_history, "STANDAARD REINFORCE")
print_history(ada_history, "ADAPTIVE BANDIT (logit_decay=0.85, failure_threshold=2, lr_boost=5x)")

# ---- Conclusie ------------------------------------------------------------
std_last = std_history[-1]["probs"]
ada_last = ada_history[-1]["probs"]

print(f"\n{'='*80}")
print(f"CONCLUSIE na {N_GOOD + N_BAD} episodes:")
print(f"{'='*80}")
print(f"  Standaard REINFORCE:  WebClaw={std_last.get('use_webclaw_scrape',0):.1%}, "
      f"Jina={std_last.get('use_jina_reader',0):.1%}, "
      f"Browser={std_last.get('use_browser',0):.1%}")
print(f"  AdaptiveBandit:      WebClaw={ada_last.get('use_webclaw_scrape',0):.1%}, "
      f"Jina={ada_last.get('use_jina_reader',0):.1%}, "
      f"Browser={ada_last.get('use_browser',0):.1%}")
print()

# Check after first failure
std_after_fail1 = std_history[N_GOOD]["probs"]
ada_after_fail1 = ada_history[N_GOOD]["probs"]
print(f"  Na 1e failure:")
print(f"    Standaard:  WebClaw={std_after_fail1.get('use_webclaw_scrape',0):.1%}")
print(f"    Adaptive:   WebClaw={ada_after_fail1.get('use_webclaw_scrape',0):.1%}")

# Check after all failures
std_after_all = std_history[-1]
ada_after_all = ada_history[-1]
std_switched = std_last.get("use_jina_reader", 0) > std_last.get("use_webclaw_scrape", 0)
ada_switched = ada_last.get("use_jina_reader", 0) > ada_last.get("use_webclaw_scrape", 0)

print(f"\n  Is de policy omgeslagen naar Jina?")
print(f"    Standaard:  {'JA ✅' if std_switched else 'NEE ❌ — blijft WebClaw dominant'}")
print(f"    Adaptive:   {'JA ✅' if ada_switched else 'NEE ❌ — blijft WebClaw dominant'}")
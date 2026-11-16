#!/usr/bin/env python3
"""
train_adaptive.py — Train Hermes task policies.

Detecteert automatisch of de policy SoftmaxPolicy (marginal) of
ContextualSoftmaxPolicy (contextual) is en gebruikt de juiste learner.

Usage:
  python3 train_adaptive.py --agent-id hermes --task-id choose-scrape-strategy

Environment:
  AGENT_LEARNING_STORE_BACKEND=local
  AGENT_LEARNING_LOCAL_STORE_DIR=...
  AGENT_LEARNING_ADAPTIVE_LOGIT_DECAY=0.97    (default)
  AGENT_LEARNING_ADAPTIVE_FAILURE_THRESHOLD=3
  AGENT_LEARNING_ADAPTIVE_LR_BOOST=5.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import env_setup  # noqa: F401 — ensures LocalFileStore backend

from collections import Counter

from agent_learning.config import LearnerConfig
from agent_learning.policy.softmax_bandit import SoftmaxPolicy
from agent_learning.policy.contextual_softmax import ContextualSoftmaxPolicy
from agent_learning.storage.cosmos import get_default_store
from agent_learning.training.runner import LearningRunner

from adaptive_learner import AdaptiveBanditLearner
from contextual_learner import ContextualAdaptiveBanditLearner


def _is_contextual(snapshot) -> bool:
    """Detecteer of snapshot een ContextualSoftmaxPolicy is."""
    kind = snapshot.metadata.get("policy_kind", "")
    return kind == "contextual_softmax" or "context_weights" in snapshot.metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Hermes task policies (adaptive / contextual)"
    )
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--logit-decay", type=float, default=None)
    parser.add_argument("--failure-threshold", type=int, default=None)
    parser.add_argument("--lr-boost", type=float, default=None)
    args = parser.parse_args()

    # ---- Resolve learner config -------------------------------------------
    learner_config = LearnerConfig()
    for key in ("logit_decay", "failure_threshold", "lr_boost"):
        val = getattr(args, key, None)
        if val is not None:
            os.environ[f"AGENT_LEARNING_ADAPTIVE_{key.upper()}"] = str(val)

    store = get_default_store()
    task_ids = [args.task_id] if args.task_id else [t.id for t in store.list_agent_tasks(args.agent_id)]
    if not task_ids:
        print(f"No tasks found for agent {args.agent_id!r}", file=sys.stderr)
        return 1

    selected_episodes = store.query_episodes(
        args.agent_id, task_id=args.task_id, limit=args.limit,
        start_date=args.start_date, end_date=args.end_date, full_only=True,
    )
    episode_limits = Counter(ep.task_id for ep in selected_episodes)
    results = []
    skipped = []

    for task_id in task_ids:
        snapshot = store.get_active_policy(args.agent_id, task_id)
        if snapshot is None:
            skipped.append({"task_id": task_id, "reason": "no active policy"})
            continue

        n_eps = episode_limits.get(task_id, 0)
        if n_eps == 0:
            skipped.append({"task_id": task_id, "reason": "no episodes in selected batch"})
            continue
        if n_eps < args.min_episodes:
            skipped.append({"task_id": task_id, "reason": f"batch has {n_eps} eps; min {args.min_episodes}"})
            continue

        contextual = _is_contextual(snapshot)
        learner = ContextualAdaptiveBanditLearner(learner_config) if contextual else AdaptiveBanditLearner(learner_config)
        policy_cls = ContextualSoftmaxPolicy if contextual else SoftmaxPolicy
        policy = policy_cls.from_snapshot(snapshot)

        print(f"\n=== Task: {task_id} ({n_eps} episodes) ===")
        print(f"Policy: {'ContextualSoftmax' if contextual else 'Softmax (marginal)'}")
        print(f"Learner: {type(learner).__name__}")
        print(f"  logit_decay={learner.logit_decay}, failure_threshold={learner.failure_threshold}")

        if contextual:
            from scrape_features import FEATURE_NAMES
            phi = [0.0] * policy.feature_dim
            probs_before = policy.probabilities(phi)
        else:
            probs_before = policy.probabilities()
        print(f"  probs BEFORE: {dict(zip([a.id for a in snapshot.actions], [f'{p:.2%}' for p in probs_before]))}")

        runner = LearningRunner(store=store, policy=policy, learner=learner, learner_config=learner_config)
        run = runner.run_offline_batch(
            args.agent_id, task_id=task_id, episode_limit=n_eps,
            start_date=args.start_date, end_date=args.end_date,
            score_missing=not args.skip_scoring,
        )

        if contextual:
            probs_after = policy.probabilities(phi)
        else:
            probs_after = policy.probabilities()
        print(f"  probs AFTER:  {dict(zip([a.id for a in snapshot.actions], [f'{p:.2%}' for p in probs_after]))}")

        run_dict = run.to_dict()
        run_dict["diagnostics"] = {
            "contextual": contextual,
            "logit_decay": learner.logit_decay,
            "failure_threshold": learner.failure_threshold,
            "probabilities_before": {a.id: round(p, 6) for a, p in zip(snapshot.actions, probs_before)},
            "probabilities_after": {a.id: round(p, 6) for a, p in zip(snapshot.actions, probs_after)},
        }
        results.append(run_dict)

    print(f"\n{'='*60}")
    print(json.dumps({"agent_id": args.agent_id, "runs": results, "skipped": skipped}, indent=2))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
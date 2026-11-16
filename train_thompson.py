#!/usr/bin/env python3
"""
train_thompson.py — Train Thompson sampling policies.

Usage:
  # Train one task
  python3 train_thompson.py --agent-id hermes --task-id podcast-research

  # Train all podcast tasks
  python3 train_thompson.py --agent-id hermes --task-prefix podcast-

  # With custom prior
  python3 train_thompson.py --agent-id hermes --task-id podcast-research --prior-alpha 3 --prior-beta 2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from thompson_policy import train_thompson
from agent_learning.storage.cosmos import get_default_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Thompson sampling policies")
    parser.add_argument("--agent-id", required=True, default="hermes")
    parser.add_argument("--task-id", default=None, help="Specific task to train")
    parser.add_argument("--task-prefix", default=None, help="Train all tasks matching prefix")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-episodes", type=int, default=3)
    parser.add_argument("--prior-alpha", type=float, default=2.0)
    parser.add_argument("--prior-beta", type=float, default=2.0)
    args = parser.parse_args()

    store = get_default_store()

    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_prefix:
        tasks = store.list_agent_tasks(args.agent_id)
        task_ids = [t.id for t in tasks if t.id.startswith(args.task_prefix)]
        if not task_ids:
            print(f"No tasks found with prefix {args.task_prefix!r}")
            return 1
    else:
        tasks = store.list_agent_tasks(args.agent_id)
        task_ids = [t.id for t in tasks]

    if not task_ids:
        print(f"No tasks found for agent {args.agent_id!r}")
        return 1

    results = []
    for task_id in task_ids:
        print(f"\n=== Thompson: {task_id} ===")
        result = train_thompson(
            store, args.agent_id, task_id,
            limit=args.limit,
            min_episodes=args.min_episodes,
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
        )
        print(json.dumps(result, indent=2, default=str))
        results.append(result)

    print(f"\n{'='*50}")
    print(f"Trained {len(results)} tasks")
    print(json.dumps({"agent_id": args.agent_id, "results": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
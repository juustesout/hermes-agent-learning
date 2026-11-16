"""
holdout_eval.py — Holdout-evaluatie: learned policy vs fixed baseline.

Vergelijkt het geleerde scrape-beleid met een vaste baseline op een
gelabelde test set. Meet accuracy, cost, latency, en geprefereerde actie.

Gebruik:
    python3 holdout_eval.py                              # default test set
    python3 holdout_eval.py --baseline always_webclaw     # andere baseline
    python3 holdout_eval.py --json                        # machine-readable
    python3 holdout_eval.py --store-results               # sla op voor trend
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import env_setup  # noqa: F401 — ensures LocalFileStore backend

from scrape_features import SITE_PROFILES, FEATURE_NAMES, phi_to_vector

# ---------------------------------------------------------------------------
# Test set: gelabelde scenarios
# ---------------------------------------------------------------------------
# Elke entry: (name, context_features_dict, correct_action_id, site_category)

LabeledTest = Tuple[str, Dict[str, float], str, str]

DEFAULT_TEST_SET: List[LabeledTest] = [
    # (naam, features, correcte actie, categorie)
    ("wordpress_astra", SITE_PROFILES["wordpress_astra"], "use_webclaw_scrape", "wordpress"),
    ("wordpress_elementor", SITE_PROFILES["wordpress_elementor"], "use_webclaw_scrape", "wordpress"),
    ("cloudflare_wp", SITE_PROFILES["cloudflare_wp"], "use_webclaw_scrape", "cloudflare"),
    ("cloudflare_js_heavy", SITE_PROFILES["cloudflare_js_heavy"], "use_webclaw_scrape", "cloudflare"),
    ("simple_html", SITE_PROFILES["simple_html"], "use_jina_reader", "simple"),
    ("api_endpoint", SITE_PROFILES["api_endpoint"], "use_jina_reader", "structured"),
    # Extra variaties
    ("wp_geen_cf", {"cloudflare": 0.0, "js_heavy": 0.2, "is_wordpress": 1.0,
     "is_elementor": 0.0, "content_len_norm": 0.2, "requires_antibot": 0.0,
     "site_known": 0.5, "has_structured_data": 0.3}, "use_jina_reader", "wordpress"),
    ("js_app_geen_cf", {"cloudflare": 0.0, "js_heavy": 1.0, "is_wordpress": 0.0,
     "is_elementor": 0.0, "content_len_norm": 0.8, "requires_antibot": 0.0,
     "site_known": 0.0, "has_structured_data": 0.0}, "use_browser", "js_app"),
    ("zware_cf_site", {"cloudflare": 1.0, "js_heavy": 0.8, "is_wordpress": 0.0,
     "is_elementor": 0.0, "content_len_norm": 0.6, "requires_antibot": 1.0,
     "site_known": 0.0, "has_structured_data": 0.0}, "use_webclaw_scrape", "cloudflare"),
    ("simpele_api", {"cloudflare": 0.0, "js_heavy": 0.0, "is_wordpress": 0.0,
     "is_elementor": 0.0, "content_len_norm": 0.05, "requires_antibot": 0.0,
     "site_known": 1.0, "has_structured_data": 1.0}, "use_jina_reader", "structured"),
]

# ---------------------------------------------------------------------------
# Cost / latency model (zelfde als cost_aware_rewards.py)
# ---------------------------------------------------------------------------

TOOL_COST = {
    "use_webclaw_scrape": 1.0,
    "use_jina_reader": 0.0,
    "use_browser": 5.0,
}
TOOL_LATENCY = {
    "use_webclaw_scrape": 3.0,
    "use_jina_reader": 1.5,
    "use_browser": 10.0,
}

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def baseline_always_first(action_ids: List[str]) -> str:
    """Kies altijd de eerste action (default fallback)."""
    return action_ids[0] if action_ids else "use_webclaw_scrape"


def baseline_always_webclaw(action_ids: List[str]) -> str:
    """Kies altijd WebClaw (veilige default)."""
    return "use_webclaw_scrape"


def baseline_cheapest(action_ids: List[str]) -> str:
    """Kies de goedkoopste tool."""
    if not action_ids:
        return "use_webclaw_scrape"
    return min(action_ids, key=lambda a: TOOL_COST.get(a, 999))


def baseline_fastest(action_ids: List[str]) -> str:
    """Kies de snelste tool."""
    if not action_ids:
        return "use_webclaw_scrape"
    return min(action_ids, key=lambda a: TOOL_LATENCY.get(a, 999))


BASELINES = {
    "always_first": baseline_always_first,
    "always_webclaw": baseline_always_webclaw,
    "cheapest": baseline_cheapest,
    "fastest": baseline_fastest,
}

# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Resultaat van één evaluatie-run."""

    policy_name: str
    policy_version: int = 0
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_case: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy_name,
            "version": self.policy_version,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "total_cost": round(self.total_cost, 2),
            "avg_latency": round(self.avg_latency, 2),
            "action_distribution": dict(sorted(self.action_counts.items())),
            "per_category": self.per_category,
        }

    def summary(self) -> str:
        lines = [
            f"Policy:     {self.policy_name} (v{self.policy_version})",
            f"Accuracy:   {self.correct}/{self.total} = {self.accuracy*100:.1f}%",
            f"Total cost: {self.total_cost:.1f} credits",
            f"Avg delay:  {self.avg_latency:.1f}s",
            f"Actions:    {dict(sorted(self.action_counts.items()))}",
        ]
        if self.per_category:
            lines.append("Per category:")
            for cat, stats in sorted(self.per_category.items()):
                lines.append(
                    f"  {cat:<15} {stats['correct']}/{stats['total']} "
                    f"({stats['accuracy']*100:.0f}%) "
                    f"cost={stats['cost']:.1f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class HoldoutEvaluator:
    """Vergelijk een learned policy met een baseline op een vaste test set.

    Usage:
        evaluator = HoldoutEvaluator()
        result = evaluator.evaluate_learned(store, agent_id="hermes")
        baseline = evaluator.evaluate_baseline("always_webclaw")
        evaluator.print_comparison(result, baseline)
    """

    ACTION_IDS = ["use_webclaw_scrape", "use_jina_reader", "use_browser"]

    def __init__(
        self,
        test_set: Optional[List[LabeledTest]] = None,
    ):
        self.test_set = test_set or DEFAULT_TEST_SET
        self._baseline_cache: Dict[str, EvalResult] = {}

    # ------------------------------------------------------------------
    # Evaluate learned policy from store
    # ------------------------------------------------------------------

    def evaluate_learned(
        self,
        store,
        agent_id: str = "hermes",
        task_id: str = "choose-scrape-strategy",
    ) -> EvalResult:
        """Evaluate the active learned policy from the store."""
        result = EvalResult(policy_name="learned")

        # Load policy from store
        snapshot = store.get_active_policy(agent_id, task_id)
        if snapshot is None:
            result.errors.append("No active policy in store")
            return result

        result.policy_version = snapshot.version

        # Reconstruct policy
        kind = snapshot.metadata.get("policy_kind", "")
        if kind == "contextual_softmax" or "context_weights" in snapshot.metadata:
            from agent_learning.policy.contextual_softmax import ContextualSoftmaxPolicy
            policy = ContextualSoftmaxPolicy.from_snapshot(snapshot)
        elif "thompson_alpha" in snapshot.metadata:
            from thompson_policy import ThompsonPolicy
            policy = ThompsonPolicy.from_snapshot(snapshot)
        else:
            from agent_learning.policy.softmax_bandit import SoftmaxPolicy
            policy = SoftmaxPolicy.from_snapshot(snapshot)

        self._run_eval(policy, result)
        return result

    # ------------------------------------------------------------------
    # Evaluate a baseline
    # ------------------------------------------------------------------

    def evaluate_baseline(self, name: str = "always_webclaw") -> EvalResult:
        """Evaluate a fixed baseline strategy."""
        if name in self._baseline_cache:
            return self._baseline_cache[name]

        fn = BASELINES.get(name)
        if fn is None:
            result = EvalResult(policy_name=name)
            result.errors.append(f"Unknown baseline: {name}")
            return result

        result = EvalResult(policy_name=name)

        for site_name, features, correct_action, category in self.test_set:
            chosen = fn(self.ACTION_IDS)
            is_correct = chosen == correct_action
            cost = TOOL_COST.get(chosen, 1.0)
            latency = TOOL_LATENCY.get(chosen, 5.0)

            result.total += 1
            if is_correct:
                result.correct += 1
            result.total_cost += cost
            result.avg_latency += latency
            result.action_counts[chosen] = result.action_counts.get(chosen, 0) + 1

            cat_key = category
            if cat_key not in result.per_category:
                result.per_category[cat_key] = {
                    "total": 0, "correct": 0, "cost": 0.0, "accuracy": 0.0
                }
            result.per_category[cat_key]["total"] += 1
            if is_correct:
                result.per_category[cat_key]["correct"] += 1
            result.per_category[cat_key]["cost"] += cost

            result.per_case.append({
                "site": site_name,
                "category": category,
                "chosen": chosen,
                "correct": correct_action,
                "is_correct": is_correct,
                "cost": cost,
            })

        # Finalise
        n = max(result.total, 1)
        result.accuracy = result.correct / n
        result.avg_latency /= n
        for cat_key in result.per_category:
            t = max(result.per_category[cat_key]["total"], 1)
            result.per_category[cat_key]["accuracy"] = (
                result.per_category[cat_key]["correct"] / t
            )

        self._baseline_cache[name] = result
        return result

    # ------------------------------------------------------------------
    # Internal: run evaluation for a policy
    # ------------------------------------------------------------------

    def _run_eval(self, policy, result: EvalResult) -> None:
        for site_name, features, correct_action, category in self.test_set:
            # Get decision from policy
            try:
                phi_vec = phi_to_vector(features)
                decision = policy.choose(state=phi_vec)
                chosen = decision.action.id
                logprob = decision.logprob
            except Exception as exc:
                result.errors.append(f"{site_name}: policy error: {exc}")
                continue

            is_correct = chosen == correct_action
            cost = TOOL_COST.get(chosen, 1.0)
            latency = TOOL_LATENCY.get(chosen, 5.0)

            result.total += 1
            if is_correct:
                result.correct += 1
            result.total_cost += cost
            result.avg_latency += latency
            result.action_counts[chosen] = result.action_counts.get(chosen, 0) + 1

            cat_key = category
            if cat_key not in result.per_category:
                result.per_category[cat_key] = {
                    "total": 0, "correct": 0, "cost": 0.0, "accuracy": 0.0
                }
            result.per_category[cat_key]["total"] += 1
            if is_correct:
                result.per_category[cat_key]["correct"] += 1
            result.per_category[cat_key]["cost"] += cost

            result.per_case.append({
                "site": site_name,
                "category": category,
                "chosen": chosen,
                "correct": correct_action,
                "is_correct": is_correct,
                "cost": cost,
                "logprob": round(logprob, 4) if logprob is not None else None,
            })

        n = max(result.total, 1)
        result.accuracy = result.correct / n
        result.avg_latency /= n
        for cat_key in result.per_category:
            t = max(result.per_category[cat_key]["total"], 1)
            result.per_category[cat_key]["accuracy"] = (
                result.per_category[cat_key]["correct"] / t
            )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def print_comparison(
        learned: EvalResult,
        baseline: EvalResult,
        label: str = "Comparison",
    ) -> str:
        lines = [
            f"{'=' * 50}",
            f"  {label}",
            f"{'=' * 50}",
            "",
            f"{'Metric':<25} {'Learned':<20} {'Baseline':<20}",
            f"{'─'*25} {'─'*20} {'─'*20}",
            f"{'Accuracy':<25} {learned.accuracy*100:<19.1f}% {baseline.accuracy*100:.1f}%",
            f"{'Correct':<25} {learned.correct}/{learned.total:<14} {baseline.correct}/{baseline.total}",
            f"{'Total cost':<25} {learned.total_cost:<19.1f} {baseline.total_cost:.1f}",
            f"{'Avg latency':<25} {learned.avg_latency:<19.1f}s {baseline.avg_latency:.1f}s",
        ]

        delta_acc = learned.accuracy - baseline.accuracy
        delta_cost = baseline.total_cost - learned.total_cost
        lines.append("")
        lines.append(f"Delta accuracy: {delta_acc*100:+.1f}%")
        lines.append(f"Cost savings:   {delta_cost:+.1f} credits (neg = learned costs more)")

        if learned.errors:
            lines.append("")
            lines.append(f"Errors ({len(learned.errors)}):")
            for e in learned.errors[:5]:
                lines.append(f"  ! {e}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Holdout evaluation")
    parser.add_argument("--baseline", default="always_webclaw",
                        choices=list(BASELINES),
                        help="Baseline strategy to compare against")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--store-results", action="store_true",
                        help="Save results to store/eval/ for trend tracking")
    args = parser.parse_args()

    try:
        from agent_learning.storage.cosmos import get_default_store
        store = get_default_store()
    except Exception as exc:
        print(f"Store not available: {exc}")
        return 1

    evaluator = HoldoutEvaluator()

    # Evaluate
    learned = evaluator.evaluate_learned(store)
    baseline = evaluator.evaluate_baseline(args.baseline)

    if args.json:
        print(json.dumps({
            "learned": learned.to_dict(),
            "baseline": baseline.to_dict(),
            "delta": {
                "accuracy": round(learned.accuracy - baseline.accuracy, 4),
                "cost": round(baseline.total_cost - learned.total_cost, 2),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    else:
        print(evaluator.print_comparison(learned, baseline))
        print()
        print("--- Learned per-case ---")
        for case in learned.per_case:
            mark = "✓" if case["is_correct"] else "✗"
            print(f"  {mark} {case['site']:<22} → {case['chosen']:<22} "
                  f"(correct: {case['correct']})")

    # Optioneel: sla resultaten op voor trend
    if args.store_results:
        eval_dir = os.path.join(
            os.environ.get("AGENT_LEARNING_LOCAL_STORE_DIR",
                           os.path.join(os.path.dirname(__file__), "store")),
            "eval",
        )
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = os.path.join(eval_dir, f"holdout_{timestamp}.json")
        with open(path, "w") as f:
            json.dump({
                "learned": learned.to_dict(),
                "baseline": baseline.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        print(f"\nResults saved: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
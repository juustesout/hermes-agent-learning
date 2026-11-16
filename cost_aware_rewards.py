"""
cost_aware_rewards.py — Reward functions that incorporate cost, latency, and quality.

Standard reward: +1 for success, -1 for failure.
Cost-aware reward: weighted combination of:
  - Task completion (base)
  - Latency penalty (slower → lower reward)
  - Cost penalty (more expensive tools → lower reward)
  - Quality bonus (content quality, retry rate)

Usage:
    from cost_aware_rewards import CostAwareReward
    reward_fn = CostAwareReward(
        weight_completion=1.0,
        weight_latency=0.3,
        weight_cost=0.3,
        weight_quality=0.2,
    )
    runner = WorkflowRunner(dag, reward_fn=reward_fn.compute)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from workflow_runner import RewardFn

logger = logging.getLogger("cost_aware_rewards")


# ---------------------------------------------------------------------------
# Cost model — known tool costs
# ---------------------------------------------------------------------------

# Estimated cost per action (in credits / compute units)
# Used when no real cost data is available in the episode
TOOL_COST_ESTIMATES: Dict[str, float] = {
    # Web scraping
    "use_webclaw_scrape": 1.0,
    "use_webclaw_batch": 2.0,
    "use_webclaw_search": 0.5,
    "use_webclaw_deep": 3.0,
    "use_webclaw_quick": 0.5,
    "use_jina_reader": 1.0,
    "use_browser": 5.0,
    # LLM
    "llm_full": 2.0,
    "llm_suggest": 0.3,
    "llm_only": 1.0,
    "llm_outline": 0.5,
    # Audio / Video
    "tts_edge": 0.5,
    "tts_elevenlabs": 3.0,
    "record": 10.0,
    "explainer_mock": 1.0,
    "explainer_real": 15.0,
    "static": 0.1,
    # Publishing
    "wordpress": 0.3,
    "youtube": 0.5,
    "spotify": 0.3,
    "all": 1.0,
    "draft": 0.1,
    # Misc
    "template": 0.2,
    "from_queue": 0.1,
    "manual": 5.0,
    "skip": 0.0,
    "done": 0.1,
    "ok": 0.0,
    "fail": 0.0,
}

# Estimated latency per action (seconds)
TOOL_LATENCY_ESTIMATES: Dict[str, float] = {
    "use_webclaw_scrape": 3.0,
    "use_webclaw_batch": 6.0,
    "use_webclaw_search": 1.5,
    "use_webclaw_deep": 8.0,
    "use_webclaw_quick": 1.0,
    "use_jina_reader": 2.0,
    "use_browser": 10.0,
    "llm_full": 8.0,
    "llm_suggest": 1.0,
    "llm_only": 5.0,
    "llm_outline": 2.0,
    "tts_edge": 3.0,
    "tts_elevenlabs": 5.0,
    "record": 600.0,
    "explainer_mock": 5.0,
    "explainer_real": 120.0,
    "static": 1.0,
    "wordpress": 2.0,
    "youtube": 10.0,
    "spotify": 3.0,
    "all": 15.0,
    "draft": 0.5,
    "skip": 0.0,
}


def estimate_cost(action_id: str) -> float:
    """Estimated cost for an action, with fallback."""
    return TOOL_COST_ESTIMATES.get(action_id, 1.0)


def estimate_latency(action_id: str) -> float:
    """Estimated latency in seconds, with fallback."""
    return TOOL_LATENCY_ESTIMATES.get(action_id, 5.0)


# ---------------------------------------------------------------------------
# CostAwareReward
# ---------------------------------------------------------------------------


@dataclass
class RewardFactors:
    """Breakdown of a cost-aware reward computation."""
    completion: float = 0.0
    latency_penalty: float = 0.0
    cost_penalty: float = 0.0
    quality_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "completion": round(self.completion, 4),
            "latency_penalty": round(self.latency_penalty, 4),
            "cost_penalty": round(self.cost_penalty, 4),
            "quality_bonus": round(self.quality_bonus, 4),
            "total": round(self.total, 4),
        }


class CostAwareReward:
    """Reward function that balances completion, speed, cost, and quality.

    Formula:
        reward = w_c * completion
               - w_l * latency_penalty
               - w_k * cost_penalty
               + w_q * quality_bonus

    All factors are normalised to [0, 1] before weighting.
    Final reward is clamped to [-1, 1].

    Args:
        weight_completion: Weight for task completion (0-1). Default 1.0.
        weight_latency: Penalty weight for slow execution. Default 0.3.
        weight_cost: Penalty weight for expensive tools. Default 0.3.
        weight_quality: Bonus for quality metrics. Default 0.2.
        max_latency_s: Latency above this gets max penalty. Default 60s.
        max_cost: Cost above this gets max penalty. Default 10 credits.
    """

    def __init__(
        self,
        weight_completion: float = 1.0,
        weight_latency: float = 0.3,
        weight_cost: float = 0.3,
        weight_quality: float = 0.2,
        max_latency_s: float = 60.0,
        max_cost: float = 10.0,
    ) -> None:
        self.w_c = max(0, weight_completion)
        self.w_l = max(0, weight_latency)
        self.w_k = max(0, weight_cost)
        self.w_q = max(0, weight_quality)
        self.max_latency = max(1, max_latency_s)
        self.max_cost = max(1, max_cost)
        self._total_weight = self.w_c + self.w_l + self.w_k + self.w_q

    # ------------------------------------------------------------------
    # Per-action factors
    # ------------------------------------------------------------------

    def _latency_factor(self, action_id: str,
                        actual_s: Optional[float] = None) -> float:
        """Normalised latency penalty: 0 = instant, 1 = max penalty."""
        lat = actual_s if actual_s is not None else estimate_latency(action_id)
        return min(lat / self.max_latency, 1.0)

    def _cost_factor(self, action_id: str,
                     actual_cost: Optional[float] = None) -> float:
        """Normalised cost penalty: 0 = free, 1 = max penalty."""
        cost = actual_cost if actual_cost is not None else estimate_cost(action_id)
        return min(cost / self.max_cost, 1.0)

    # ------------------------------------------------------------------
    # Full reward computation (for WorkflowRunner)
    # ------------------------------------------------------------------

    def compute(
        self,
        context: Dict[str, Any],
        node_results: List[Any],
    ) -> Tuple[float, str]:
        """Compute global reward for a completed workflow run.

        This is a RewardFn-compatible callback for WorkflowRunner.

        Args:
            context: Run context with cost/latency data.
            node_results: List of NodeResult from the run.

        Returns:
            (reward, reason_string)
        """
        total = len(node_results)
        if total == 0:
            return 0.0, "no nodes"

        # --- Completion factor ---
        n_ok = sum(1 for nr in node_results
                   if nr.execution_status == "completed")
        completion = n_ok / total

        # --- Latency penalty ---
        latencies = []
        for nr in node_results:
            action = nr.action_id or "?"
            # Actual latency from context if available
            actual = context.get(f"_latency:{nr.node_id}")
            lat = actual if actual is not None else estimate_latency(action)
            latencies.append(lat)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        latency_penalty = self._latency_factor("", avg_latency)

        # --- Cost penalty ---
        costs = []
        for nr in node_results:
            action = nr.action_id or "?"
            actual = context.get(f"_cost:{nr.node_id}")
            c = actual if actual is not None else estimate_cost(action)
            costs.append(c)
        total_cost = sum(costs)
        avg_cost = total_cost / len(costs) if costs else 0
        cost_penalty = self._cost_factor("", total_cost)

        # --- Quality bonus ---
        quality = context.get("_quality_score", 1.0)
        # Penalise retries
        retries = context.get("_retry_count", 0)
        quality = quality * (0.9 ** retries)

        # --- Weighted sum ---
        raw = (
            self.w_c * completion
            - self.w_l * latency_penalty
            - self.w_k * cost_penalty
            + self.w_q * quality
        )
        # Normalise to [-1, 1]
        norm_raw = raw / max(self._total_weight, 0.01)
        reward = max(-1.0, min(1.0, norm_raw))

        # Build reason
        parts = [
            f"{n_ok}/{total} ok",
            f"lat={avg_latency:.1f}s",
            f"cost={total_cost:.1f}",
            f"qual={quality:.2f}",
        ]
        if latency_penalty > 0.3:
            parts.append("⚠slow")
        if cost_penalty > 0.3:
            parts.append("⚠costly")

        return round(reward, 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # Per-node reward breakdown
    # ------------------------------------------------------------------

    def node_reward(
        self,
        node_id: str,
        action_id: str,
        status: str,
        latency_s: Optional[float] = None,
        cost: Optional[float] = None,
        quality: float = 1.0,
    ) -> RewardFactors:
        """Compute reward factors for a single node/action."""
        completion = 1.0 if status == "completed" else -0.5
        lat_factor = self._latency_factor(action_id, latency_s)
        cost_factor = self._cost_factor(action_id, cost)

        raw = (
            self.w_c * completion
            - self.w_l * lat_factor
            - self.w_k * cost_factor
            + self.w_q * quality
        )
        total = raw / max(self._total_weight, 0.01)
        total = max(-1.0, min(1.0, total))

        return RewardFactors(
            completion=completion,
            latency_penalty=lat_factor,
            cost_penalty=cost_factor,
            quality_bonus=quality,
            total=total,
        )

    # ------------------------------------------------------------------
    # Helpers: annotate context with cost/latency data
    # ------------------------------------------------------------------

    @staticmethod
    def annotate_execution(
        context: Dict[str, Any],
        node_id: str,
        action_id: str,
        latency_s: float,
        cost: Optional[float] = None,
    ) -> None:
        """Store cost/latency data in context for the reward function."""
        context[f"_latency:{node_id}"] = latency_s
        context[f"_cost:{node_id}"] = (
            cost if cost is not None else estimate_cost(action_id)
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def balanced_reward() -> CostAwareReward:
    """Balanced: completion + moderate cost/latency awareness."""
    return CostAwareReward(
        weight_completion=1.0,
        weight_latency=0.3,
        weight_cost=0.3,
        weight_quality=0.2,
    )


def speed_focused() -> CostAwareReward:
    """Speed优先: latency heavily penalised."""
    return CostAwareReward(
        weight_completion=1.0,
        weight_latency=0.8,
        weight_cost=0.2,
        weight_quality=0.1,
    )


def cost_focused() -> CostAwareReward:
    """Budget mode: cost heavily penalised."""
    return CostAwareReward(
        weight_completion=1.0,
        weight_latency=0.2,
        weight_cost=0.8,
        weight_quality=0.1,
    )


def quality_focused() -> CostAwareReward:
    """Quality first: completion + quality dominate."""
    return CostAwareReward(
        weight_completion=1.0,
        weight_latency=0.1,
        weight_cost=0.1,
        weight_quality=0.6,
    )


__all__ = [
    "CostAwareReward", "RewardFactors",
    "estimate_cost", "estimate_latency",
    "balanced_reward", "speed_focused", "cost_focused", "quality_focused",
]
"""
AdaptiveBanditLearner — extends ReinforceLearner with:

1. Logit decay: old evidence fades so ancient history doesn't dominate
2. Consecutive failure detection: N failures → regime change response
3. Dynamic learning rate: boosts adaptation when failure rate spikes
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from agent_learning.config import LearnerConfig
from agent_learning.learners.base import Learner, LearnerResult
from agent_learning.learners.reinforce import ReinforceLearner
from agent_learning.policy.base import Policy
from agent_learning.policy.softmax_bandit import SoftmaxPolicy
from agent_learning.types import ConsumedInput, Episode, Reward, RewardSource


class AdaptiveBanditLearner(ReinforceLearner):
    """REINFORCE-with-baseline + logit decay + consecutive-failure detection.

    Hyperparameters (via env vars or config):
      AGENT_LEARNING_ADAPTIVE_LOGIT_DECAY  (default 0.995)
      AGENT_LEARNING_ADAPTIVE_FAILURE_THRESHOLD  (default 3)
      AGENT_LEARNING_ADAPTIVE_LR_BOOST  (default 3.0)
    """

    def __init__(
        self,
        config: Optional[LearnerConfig] = None,
        *,
        logit_decay: float = 0.995,
        failure_threshold: int = 3,
        lr_boost: float = 3.0,
    ) -> None:
        import os
        self._logit_decay = float(os.getenv(
            "AGENT_LEARNING_ADAPTIVE_LOGIT_DECAY", str(logit_decay)
        ))
        self._failure_threshold = int(os.getenv(
            "AGENT_LEARNING_ADAPTIVE_FAILURE_THRESHOLD", str(failure_threshold)
        ))
        self._lr_boost = float(os.getenv(
            "AGENT_LEARNING_ADAPTIVE_LR_BOOST", str(lr_boost)
        ))
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def logit_decay(self) -> float:
        return self._logit_decay

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    def update(
        self,
        policy: Policy,
        episodes: Iterable[Episode],
        rewards: Iterable[Reward],
    ) -> LearnerResult:
        if not isinstance(policy, SoftmaxPolicy):
            raise TypeError("AdaptiveBanditLearner requires a SoftmaxPolicy")

        # ---- Step 1: Logit decay -------------------------------------------
        # Fade old logits so ancient history doesn't dominate.
        # This runs BEFORE the REINFORCE update so the decay is baked into
        # the snapshot that REINFORCE reads AND the final applied update.
        for aid in policy._snapshot.logits:
            policy._snapshot.logits[aid] *= self._logit_decay

        # ---- Step 2: Analyze recent episodes for regime change -------------
        episode_list = list(episodes)
        recent = episode_list[-self._failure_threshold * 5:]
        failures_in_window = 0
        total_in_window = len(recent)

        for ep in recent:
            if ep.execution_status != "completed":
                failures_in_window += 1
                continue
            # Episode is completed — check metadata for correctness
            correct_id = ep.metadata.get("correct_action_id")
            if correct_id and ep.action_id != correct_id:
                failures_in_window += 1
            elif ep.metadata.get("task_completed") is False:
                failures_in_window += 1

        failure_rate = (
            failures_in_window / max(total_in_window, 1)
            if total_in_window > 0
            else 0.0
        )

        # Consecutive-failure check: last N episodes all failures
        trail = recent[-self._failure_threshold:] if len(recent) >= self._failure_threshold else []
        consecutive_failures = len(trail) > 0 and all(
            (ep.metadata.get("correct_action_id") and ep.action_id != ep.metadata["correct_action_id"])
            or ep.metadata.get("task_completed") is False
            for ep in trail
        )

        # ---- Step 3: Dynamic learning rate ---------------------------------
        boosted = False
        original_lr = self._config.learning_rate

        if consecutive_failures or failure_rate > 0.5:
            # Regime change detected — boost learning rate
            boosted_lr = min(original_lr * self._lr_boost, 0.5)
            self._config.learning_rate = boosted_lr
            boosted = True

        result = super().update(policy, episode_list, rewards)

        # Restore original learning rate
        if boosted:
            self._config.learning_rate = original_lr

        # ---- Step 4: Extra diagnostics -------------------------------------
        result.extra["logit_decay"] = self._logit_decay
        result.extra["failure_rate_window"] = failure_rate
        result.extra["consecutive_failures"] = consecutive_failures
        result.extra["learning_rate_used"] = (
            boosted_lr if boosted else original_lr
        )
        result.extra["regime_change_detected"] = boosted

        return result


__all__ = ["AdaptiveBanditLearner"]
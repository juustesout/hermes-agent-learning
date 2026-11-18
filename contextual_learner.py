"""
ContextualAdaptiveBanditLearner — extends AdaptiveBanditLearner for
ContextualSoftmaxPolicy (W * phi bandit).

Leert per site-type welke scrape strategie werkt in plaats van één
globale voorkeur. Context features worden uit episode.context_features
of episode.metadata["context_features"] gelezen.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from agent_learning.config import LearnerConfig
from agent_learning.learners.base import LearnerResult
from agent_learning.learners.reinforce import ReinforceLearner
from agent_learning.policy.base import Policy
from agent_learning.policy.contextual_softmax import ContextualSoftmaxPolicy
from agent_learning.policy.softmax_bandit import SoftmaxPolicy
from agent_learning.types import ConsumedInput, Episode, Reward, RewardSource

from adaptive_learner import AdaptiveBanditLearner
from eligibility import SCRAPE_TASK_ID, eligible_scrape
from features import PRE_CALL_FEATURE_NAMES, phi_vector, pre_call_features


# Naam in episode metadata / context_features voor de feature vector
_CONTEXT_KEY = "context_features"


class ContextualAdaptiveBanditLearner(AdaptiveBanditLearner):
    """REINFORCE-with-baseline voor ContextualSoftmaxPolicy +

    logit decay + failure detection + dynamic LR.
    """

    def __init__(
        self,
        config: Optional[LearnerConfig] = None,
        *,
        logit_decay: float = 0.97,
        failure_threshold: int = 3,
        lr_boost: float = 5.0,
    ) -> None:
        super().__init__(
            config,
            logit_decay=logit_decay,
            failure_threshold=failure_threshold,
            lr_boost=lr_boost,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        policy: Policy,
        episodes: Iterable[Episode],
        rewards: Iterable[Reward],
    ) -> LearnerResult:
        if not isinstance(policy, ContextualSoftmaxPolicy):
            raise TypeError(
                "ContextualAdaptiveBanditLearner requires a "
                "ContextualSoftmaxPolicy"
            )

        feature_dim = policy.feature_dim
        action_ids = [a.id for a in policy.snapshot().actions]

        # ---- Step 1: Logit decay op gewichtsmatrix ------------------------
        weights = policy._snapshot.metadata.get("context_weights", {})
        for aid in action_ids:
            row = weights.get(aid)
            if row is not None:
                for j in range(len(row)):
                    row[j] *= self._logit_decay

        # ---- Step 2: Analyseer episodes voor failure detection -------------
        episode_list = [
            ep
            for ep in episodes
            if getattr(ep, "task_id", "") != SCRAPE_TASK_ID or eligible_scrape(ep)
        ]
        recent = episode_list[-self._failure_threshold * 5:]
        failures = 0
        total = len(recent)
        for ep in recent:
            correct = ep.metadata.get("correct_action_id")
            if correct and ep.action_id != correct:
                failures += 1
            elif ep.metadata.get("task_completed") is False:
                failures += 1

        trail = recent[-self._failure_threshold:] if len(recent) >= self._failure_threshold else []
        consecutive_failures = all(
            (ep.metadata.get("correct_action_id") and ep.action_id != ep.metadata["correct_action_id"])
            or ep.metadata.get("task_completed") is False
            for ep in trail
        ) if trail else False

        failure_rate = failures / max(total, 1)
        failure_rate = min(failure_rate, 1.0)

        # ---- Step 3: REINFORCE-with-baseline voor contextual policy --------

        # Aggregate rewards by episode id
        episode_map = {ep.id: ep for ep in episode_list}
        agg_rewards: Dict[str, Reward] = {}
        for r in rewards:
            if r.source != RewardSource.AGGREGATE:
                continue
            if not math.isfinite(r.value) or not -1.0 <= r.value <= 1.0:
                raise ValueError("aggregate reward must be finite and within [-1, 1]")
            cur = agg_rewards.get(r.episode_id)
            if cur is None or r.created_at > cur.created_at:
                agg_rewards[r.episode_id] = r

        snapshot = policy.snapshot()
        baseline_before = snapshot.baseline

        # Initialiseer per-dimension deltas voor elke action
        deltas: Dict[str, List[float]] = {
            aid: [0.0] * feature_dim for aid in action_ids
        }
        consumed: List[ConsumedInput] = []
        used = 0
        reward_sum = 0.0

        # Entropy op uniforme probs (voor bonus — ruwe benadering)
        uniform_entropy = -sum(
            (1.0 / len(action_ids)) * math.log(1.0 / len(action_ids))
            for _ in action_ids
        )

        for episode in episode_list:
            if episode.action_id is None or episode.action_id not in action_ids:
                continue

            # Extracteer context vector
            phi = self._extract_phi(episode, feature_dim)

            # Compute probabilities GEVEN context
            probs_now = policy.probabilities(phi)

            reward = agg_rewards.get(episode.id)
            if reward is None:
                continue

            reward_value = reward.value
            advantage = reward_value - baseline_before

            # Importance sampling weight
            iw = self._importance_weight(episode, probs_now, action_ids)
            advantage *= iw

            chosen_idx = action_ids.index(episode.action_id)

            # Gradient: ΔW[j] = lr * A * (1_{a==k} - π(k)) * φ[j]  per action k
            for idx, aid in enumerate(action_ids):
                indicator = 1.0 if idx == chosen_idx else 0.0
                grad = indicator - probs_now[idx]
                for j in range(feature_dim):
                    deltas[aid][j] += advantage * grad * float(phi[j])

                # Entropy regularisation per action
                for j in range(feature_dim):
                    deltas[aid][j] += (
                        self._config.entropy_bonus
                        * (-math.log(max(probs_now[idx], 1e-12)) - uniform_entropy)
                        * float(phi[j])
                    )

            used += 1
            reward_sum += reward_value
            consumed.append(ConsumedInput(
                episode_id=episode.id, reward_id=reward.id
            ))

        if used == 0:
            return LearnerResult(
                episodes_used=0,
                mean_reward=0.0,
                baseline_before=baseline_before,
                baseline_after=baseline_before,
                logit_deltas={aid: 0.0 for aid in action_ids},
                extra={"reason": "no actionable episodes", "contextual": True},
                consumed_inputs=consumed,
            )

        # Average deltas across the batch
        for aid in action_ids:
            for j in range(feature_dim):
                deltas[aid][j] = (
                    self._config.learning_rate * deltas[aid][j] / used
                )

        # Update EMA baseline
        mean_reward = reward_sum / used
        decay = self._config.baseline_decay
        baseline_after = decay * baseline_before + (1.0 - decay) * mean_reward

        # Dynamic LR boost bij regime change
        boosted = False
        if consecutive_failures or failure_rate > 0.5:
            boost = self._lr_boost
            scale = min(boost * self._config.learning_rate, 0.5) / self._config.learning_rate
            for aid in action_ids:
                for j in range(feature_dim):
                    deltas[aid][j] *= scale
            boosted = True

        policy.apply_update(
            deltas,
            baseline=baseline_after,
            episodes_seen=snapshot.episodes_seen + used,
        )

        return LearnerResult(
            episodes_used=used,
            mean_reward=mean_reward,
            baseline_before=baseline_before,
            baseline_after=baseline_after,
            logit_deltas={aid: sum(deltas[aid]) / feature_dim for aid in action_ids},
            extra={
                "contextual": True,
                "feature_dim": feature_dim,
                "logit_decay": self._logit_decay,
                "failure_rate_window": failure_rate,
                "consecutive_failures": consecutive_failures,
                "regime_change_detected": boosted,
                "learning_rate_used": (
                    self._config.learning_rate * (scale if boosted else 1.0)
                ),
            },
            consumed_inputs=consumed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_phi(self, episode: Episode, expected_dim: int) -> np.ndarray:
        """Haal Track A vector uit episode, aligned op het featureregister."""
        ctx = episode.context_features
        if not ctx:
            ctx = episode.metadata.get(_CONTEXT_KEY, {})
        if not isinstance(ctx, dict) or not ctx:
            target = getattr(episode, "target", "") or ""
            if target:
                ctx = pre_call_features(target, at=getattr(episode, "created_at", None))
            else:
                return np.zeros(expected_dim, dtype=np.float64)
        names = list(PRE_CALL_FEATURE_NAMES)
        vec = phi_vector(ctx, names)
        if len(vec) < expected_dim:
            vec = vec + [0.0] * (expected_dim - len(vec))
        return np.asarray(vec[:expected_dim], dtype=np.float64)

    def _importance_weight(
        self,
        episode: Episode,
        probs_now: List[float],
        action_ids: List[str],
    ) -> float:
        if episode.action_logprob is None or episode.action_id is None:
            return 1.0
        idx = action_ids.index(episode.action_id)
        p_target = max(probs_now[idx], 1e-12)
        p_behaviour = max(math.exp(episode.action_logprob), 1e-12)
        ratio = p_target / p_behaviour
        return min(ratio, self._config.importance_clip)


__all__ = ["ContextualAdaptiveBanditLearner"]
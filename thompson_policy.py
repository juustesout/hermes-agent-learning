"""
thompson_policy.py — Thompson Sampling policy + learner.

Thompson sampling maintains a Beta distribution per action.
At decision time, it samples from each Beta and picks the max.
This gives better cold-start exploration than Softmax (which is
uniform and slow to converge).

Beta(α, β):
  α = prior_successes + observed_successes
  β = prior_failures  + observed_failures
  Prior: Beta(1, 1) — uniform (flat) prior

Reward scaling (REINFORCE reward ∈ [-1, 1] → Beta observation):
  success_weight = (reward + 1) / 2  → maps [-1,1] to [0,1]
  α_update += success_weight
  β_update += 1 - success_weight

Usage:
    from thompson_policy import ThompsonPolicy
    policy = ThompsonPolicy.from_actions(actions, agent_id="hermes")
    decision = policy.choose()
    policy.apply_update(action_id, reward=0.7)
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_learning.policy.base import Decision, Policy
from agent_learning.types import Action, PolicySnapshot


# ---------------------------------------------------------------------------
# Beta helpers
# ---------------------------------------------------------------------------

_BETA_PRIOR_ALPHA = 1.0
_BETA_PRIOR_BETA = 1.0
_MAX_LOGIT_ABS = 10.0  # placeholder for compatibility


def _beta_mean(alpha: float, beta: float) -> float:
    """Expected value of Beta(α, β)."""
    denom = alpha + beta
    return alpha / denom if denom > 0 else 0.5


def _sample_beta(rng: random.Random, alpha: float, beta: float) -> float:
    """Sample from Beta(α, β) using the gamma approximation."""
    if alpha <= 0 or beta <= 0:
        return 0.5
    # Use random.gammavariate for each parameter
    try:
        x = rng.gammavariate(alpha, 1.0)
        y = rng.gammavariate(beta, 1.0)
        return x / (x + y)
    except (ValueError, ZeroDivisionError):
        return _beta_mean(alpha, beta)


def _reward_to_beta_obs(reward: float) -> Tuple[float, float]:
    """Scale REINFORCE reward ∈ [-1, 1] to Beta observation weights.

    Returns (success_weight, failure_weight).
    A reward of +1.0 → (1.0, 0.0)  — pure success
    A reward of -1.0 → (0.0, 1.0)  — pure failure
    A reward of  0.0 → (0.5, 0.5)  — neutral
    """
    clamped = max(-1.0, min(1.0, reward))
    success = (clamped + 1.0) / 2.0
    failure = 1.0 - success
    return success, failure


# ---------------------------------------------------------------------------
# ThompsonPolicy
# ---------------------------------------------------------------------------


class ThompsonPolicy(Policy):
    """Discrete Thompson-sampling policy with Beta posteriors per action.

    Construction:
        policy = ThompsonPolicy.from_actions(actions, agent_id="hermes")

    Continuation from a snapshot:
        policy = ThompsonPolicy.from_snapshot(snapshot)
    """

    def __init__(
        self,
        snapshot: PolicySnapshot,
        *,
        rng: Optional[random.Random] = None,
        prior_alpha: float = _BETA_PRIOR_ALPHA,
        prior_beta: float = _BETA_PRIOR_BETA,
    ) -> None:
        if not snapshot.actions:
            raise ValueError("ThompsonPolicy requires at least one action.")
        self._snapshot = snapshot
        # Thompson sampling intentionally uses pseudo-random exploration, not crypto randomness.
        self._rng = rng or random.Random()  # nosec B311
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

        # Ensure every action has Beta parameters
        alphas = snapshot.metadata.get("thompson_alpha", {})
        betas = snapshot.metadata.get("thompson_beta", {})
        for action in snapshot.actions:
            alphas.setdefault(action.id, prior_alpha)
            betas.setdefault(action.id, prior_beta)
        snapshot.metadata["thompson_alpha"] = alphas
        snapshot.metadata["thompson_beta"] = betas

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_actions(
        cls,
        actions: Sequence[Action],
        *,
        agent_id: str = "default",
        task_id: str = "default",
        rng: Optional[random.Random] = None,
        prior_alpha: float = _BETA_PRIOR_ALPHA,
        prior_beta: float = _BETA_PRIOR_BETA,
    ) -> "ThompsonPolicy":
        alphas = {a.id: prior_alpha for a in actions}
        betas = {a.id: prior_beta for a in actions}
        snapshot = PolicySnapshot(
            agent_id=agent_id,
            task_id=task_id,
            version=0,
            actions=list(actions),
            logits={a.id: 0.0 for a in actions},
            metadata={
                "thompson_alpha": alphas,
                "thompson_beta": betas,
                "policy_kind": "thompson",
            },
        )
        return cls(snapshot, rng=rng, prior_alpha=prior_alpha, prior_beta=prior_beta)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PolicySnapshot,
        *,
        rng: Optional[random.Random] = None,
    ) -> "ThompsonPolicy":
        return cls(snapshot, rng=rng)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def actions(self) -> Sequence[Action]:
        return tuple(self._snapshot.actions)

    def probabilities(self) -> List[float]:
        """Expected value of each action's Beta distribution."""
        alphas = self._snapshot.metadata.get("thompson_alpha", {})
        betas = self._snapshot.metadata.get("thompson_beta", {})
        return [
            _beta_mean(
                alphas.get(a.id, self._prior_alpha),
                betas.get(a.id, self._prior_beta),
            )
            for a in self._snapshot.actions
        ]

    def choose(self, state: Optional[Any] = None) -> Decision:
        """Thompson sample: draw from each Beta, pick max.

        Returns the action with the highest sampled value, along with
        its expected probability for logprob.
        """
        del state
        alphas = self._snapshot.metadata.get("thompson_alpha", {})
        betas = self._snapshot.metadata.get("thompson_beta", {})
        probs = self.probabilities()

        # Sample from each Beta
        samples: List[float] = []
        for a in self._snapshot.actions:
            sample = _sample_beta(
                self._rng,
                alphas.get(a.id, self._prior_alpha),
                betas.get(a.id, self._prior_beta),
            )
            samples.append(sample)

        # Pick max sample
        idx = max(range(len(samples)), key=lambda i: samples[i])
        chosen = self._snapshot.actions[idx]
        prob = max(probs[idx], 1e-12)
        return Decision(action=chosen, logprob=math.log(prob), probabilities=probs)

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot.from_dict(self._snapshot.to_dict())

    def apply_update(
        self,
        deltas: Dict[str, float],
        *,
        baseline: float,
        episodes_seen: int,
    ) -> None:
        """Apply Beta updates (deltas are ignored — use observe() instead).

        For compatibility with the Policy interface. Real updates happen
        via observe() or ThompsonLearner.update().
        """
        self._snapshot.baseline = float(baseline)
        self._snapshot.episodes_seen = int(episodes_seen)
        self._snapshot.updates_applied += 1
        self._snapshot.advance_version()

    # ------------------------------------------------------------------
    # Thompson-specific
    # ------------------------------------------------------------------

    def observe(self, action_id: str, reward: float) -> None:
        """Update Beta posterior with a single observation.

        Args:
            action_id: Which action was taken.
            reward: REINFORCE reward ∈ [-1, 1].
        """
        success_w, failure_w = _reward_to_beta_obs(reward)
        alphas = self._snapshot.metadata.get("thompson_alpha", {})
        betas = self._snapshot.metadata.get("thompson_beta", {})
        alphas[action_id] = alphas.get(action_id, self._prior_alpha) + success_w
        betas[action_id] = betas.get(action_id, self._prior_beta) + failure_w

    def get_params(self, action_id: str) -> Tuple[float, float]:
        """Return (alpha, beta) for an action."""
        alphas = self._snapshot.metadata.get("thompson_alpha", {})
        betas = self._snapshot.metadata.get("thompson_beta", {})
        return (
            alphas.get(action_id, self._prior_alpha),
            betas.get(action_id, self._prior_beta),
        )

    def uncertainty(self, action_id: str) -> float:
        """Return the standard deviation of Beta for an action.

        High uncertainty = explore more.
        """
        a, b = self.get_params(action_id)
        denom = (a + b) ** 2 * (a + b + 1)
        if denom <= 0:
            return 0.5
        return math.sqrt((a * b) / denom)


# ---------------------------------------------------------------------------
# ThompsonLearner — batch update from episodes
# ---------------------------------------------------------------------------


class ThompsonLearner:
    """Learner for ThompsonPolicy: batch Beta updates from episodes.

    Unlike REINFORCE, this is a Bayesian update: each episode contributes
    a weighted observation to the action's Beta posterior.

    Supports the same adaptive features as AdaptiveBanditLearner:
    - Failure detection (consecutive failures → boost)
    - Dynamic learning rate (via observation weighting)
    """

    def __init__(
        self,
        *,
        prior_alpha: float = _BETA_PRIOR_ALPHA,
        prior_beta: float = _BETA_PRIOR_BETA,
        failure_threshold: int = 3,
        lr_boost: float = 3.0,
        min_observations: int = 1,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.failure_threshold = failure_threshold
        self.lr_boost = lr_boost
        self.min_observations = min_observations

    def update(
        self,
        policy: ThompsonPolicy,
        episodes: Sequence[Any],
        rewards: Sequence[Any],
    ) -> Dict[str, Any]:
        """Batch-update Beta posteriors from episodes and rewards.

        Args:
            policy: A ThompsonPolicy instance.
            episodes: List of Episode objects (from agent-learning store).
            rewards: List of Reward objects.

        Returns:
            Dict with stats: episodes_used, mean_reward, action_updates, etc.
        """
        if not isinstance(policy, ThompsonPolicy):
            raise TypeError("ThompsonLearner requires a ThompsonPolicy")

        # Aggregate rewards by episode id
        reward_map: Dict[str, float] = {}
        for r in rewards:
            reward_map[r.episode_id] = r.value

        updates: Dict[str, Tuple[float, float]] = {}  # action_id → (α_total, β_total)
        used = 0
        reward_sum = 0.0
        failures = 0

        for ep in episodes:
            if ep.action_id is None:
                continue
            reward = reward_map.get(ep.id)
            if reward is None:
                continue

            success_w, failure_w = _reward_to_beta_obs(reward)
            cur = updates.get(ep.action_id, (0.0, 0.0))
            updates[ep.action_id] = (cur[0] + success_w, cur[1] + failure_w)
            used += 1
            reward_sum += reward
            if ep.execution_status != "completed":
                failures += 1

        if used == 0:
            return {
                "episodes_used": 0,
                "mean_reward": 0.0,
                "action_updates": {},
                "total_observations": 0,
            }

        # Detect regime change
        failure_rate = failures / max(used, 1)
        boost = self.lr_boost if failure_rate > 0.5 else 1.0

        # Apply updates
        action_updates: Dict[str, Dict[str, float]] = {}
        for action_id, (alpha_delta, beta_delta) in updates.items():
            alpha_add = alpha_delta * boost
            beta_add = beta_delta * boost
            # Direct Beta update (aggregated deltas, not per-step observe)
            alphas = policy._snapshot.metadata.get("thompson_alpha", {})
            betas = policy._snapshot.metadata.get("thompson_beta", {})
            alphas[action_id] = alphas.get(action_id, self.prior_alpha) + alpha_add
            betas[action_id] = betas.get(action_id, self.prior_beta) + beta_add
            action_updates[action_id] = {
                "alpha_add": round(alpha_add, 4),
                "beta_add": round(beta_add, 4),
                "new_alpha": round(alphas[action_id], 4),
                "new_beta": round(betas[action_id], 4),
                "new_mean": round(_beta_mean(alphas[action_id], betas[action_id]), 4),
            }

        policy._snapshot.episodes_seen += used
        policy._snapshot.updates_applied += 1
        policy._snapshot.advance_version()

        mean_reward = reward_sum / used
        return {
            "episodes_used": used,
            "mean_reward": round(mean_reward, 4),
            "failure_rate": round(failure_rate, 4),
            "boost_applied": boost > 1.0,
            "action_updates": action_updates,
            "total_observations": used,
            "policy_version": policy._snapshot.version,
        }


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def train_thompson(
    store,
    agent_id: str,
    task_id: str,
    *,
    limit: int = 200,
    min_episodes: int = 3,
    prior_alpha: float = _BETA_PRIOR_ALPHA,
    prior_beta: float = _BETA_PRIOR_BETA,
) -> Dict[str, Any]:
    """Load episodes from store, train a ThompsonPolicy, save it back.

    Usage:
        result = train_thompson(store, "hermes", "podcast-research")
        print(result["action_updates"])
    """
    from agent_learning.storage.cosmos import get_default_store

    snapshot = store.get_active_policy(agent_id, task_id)
    if snapshot is None:
        # Create new policy from scratch
        # We need the action list — query for task actions in episode metadata
        episodes = store.query_episodes(agent_id, task_id=task_id, limit=limit)
        if not episodes:
            return {"error": f"no episodes for {task_id}", "task_id": task_id}
        # Infer actions from episodes
        action_ids = sorted(set(
            ep.action_id for ep in episodes if ep.action_id
        ))
        if not action_ids:
            return {"error": f"no actions found in episodes for {task_id}"}
        actions = [Action(id=aid) for aid in action_ids]
        policy = ThompsonPolicy.from_actions(
            actions, agent_id=agent_id, task_id=task_id,
            prior_alpha=prior_alpha, prior_beta=prior_beta,
        )
    else:
        policy = ThompsonPolicy.from_snapshot(snapshot)

    # Fetch episodes
    episodes = store.query_episodes(agent_id, task_id=task_id, limit=limit, full_only=True)
    if len(episodes) < min_episodes:
        return {
            "task_id": task_id,
            "skipped": True,
            "reason": f"only {len(episodes)} episodes, need {min_episodes}",
            "episodes_found": len(episodes),
        }

    # Fetch rewards
    from agent_learning.types import Reward, RewardSource
    rewards = store.query_rewards(agent_id, limit=limit * 2)

    learner = ThompsonLearner(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        failure_threshold=3,
        lr_boost=3.0,
    )
    result = learner.update(policy, episodes, rewards)

    # Persist
    store.store_policy(policy.snapshot())
    result["task_id"] = task_id
    result["policy_id"] = policy._snapshot.id
    result["policy_version"] = policy._snapshot.version

    # Show probabilities
    probs = policy.probabilities()
    result["probabilities"] = {
        a.id: round(p, 4) for a, p in zip(policy._snapshot.actions, probs)
    }

    return result


__all__ = [
    "ThompsonPolicy",
    "ThompsonLearner",
    "train_thompson",
    "_beta_mean",
    "_sample_beta",
    "_reward_to_beta_obs",
]
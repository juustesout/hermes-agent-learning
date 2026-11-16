"""
workflow_runner.py — Sequentiële executor voor WorkflowDAGs met credit assignment.

How it works:

1. **Topological sort** — nodes worden in afhankelijkheidsvolgorde uitgevoerd.
2. **Policy decision** — per node wordt via TaskPolicy.decide() een actie gekozen,
   of via een custom decision_callback als er geen policy is.
3. **Execution** — de gekozen actie wordt uitgevoerd via node.execute() of
   een custom execute_callback.
4. **Episode logging** — elke beslissing + uitvoeringsresultaat wordt als Episode
   weggeschreven naar de agent-learning store.
5. **Global reward** — na de laatste node wordt een globale reward bepaald
   (via reward_callback). Dit is het "path-level" leer signaal.
6. **Credit assignment** — de globale reward wordt teruggepropageerd langs de
   path: elke node krijgt reward × γ^distance_from_end. Deze credit rewards
   worden als Reward records in de store gezet, gelinkt aan de episodes.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from workflow_dag import (
    NodeResult,
    WorkflowDAG,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRun,
)

# ---------------------------------------------------------------------------
# Type aliases for callbacks
# ---------------------------------------------------------------------------

# Decision callback: (node, run_context) → (action_id, action_logprob_or_None, metadata)
DecisionFn = Callable[
    [WorkflowNode, Dict[str, Any]],
    Tuple[str, Optional[float], Dict[str, Any]],
]

# Execute callback: (node, action_id, run_context) → (status, summary, metadata)
ExecuteFn = Callable[
    [WorkflowNode, str, Dict[str, Any]],
    Tuple[str, str, Dict[str, Any]],
]

# Reward callback: (run_context, node_results) → (reward_float, reason_string)
RewardFn = Callable[
    [Dict[str, Any], List[NodeResult]],
    Tuple[float, str],
]

# Local reward callback: (node, action_id, status, summary, exec_meta, run_context)
# → (reward_float, reason_string)
# Called immediately after each node executes, before the global reward.
# Gives a per-node signal independent of the global outcome.
LocalRewardFn = Callable[
    [WorkflowNode, str, str, str, Dict[str, Any], Dict[str, Any]],
    Tuple[float, str],
]

logger = logging.getLogger("workflow_runner")


# ---------------------------------------------------------------------------
# Default callbacks
# ---------------------------------------------------------------------------


def _default_decision(
    node: WorkflowNode,
    context: Dict[str, Any],
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """Fallback: kies de eerste action_id, tenzij context een override heeft."""
    override = context.get(f"override_action:{node.id}")
    if override:
        return override, None, {"source": "context_override"}
    if node.action_ids:
        return node.action_ids[0], None, {"source": "fallback_first"}
    return "skip", None, {"source": "no_actions"}


def _default_execute(
    node: WorkflowNode,
    action_id: str,
    context: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    """Fallback: roep node.execute() aan als die bestaat, anders stempel 'simulated'."""
    if node.execute:
        try:
            result = node.execute(node, action_id, context)
            status = result.get("status", "completed")
            summary = result.get("summary", f"Executed {node.id} via {action_id}")
            meta = result.get("metadata", {})
            return status, summary, meta
        except Exception as exc:
            return "failed", f"Exception: {exc}", {"error": str(exc)}
    return "completed", f"Simulated {node.id} (no executor)", {"simulated": True}


def _default_reward(
    context: Dict[str, Any],
    node_results: List[NodeResult],
) -> Tuple[float, str]:
    """Fallback: reward op basis van hoeveel nodes succeeded zijn."""
    total = len(node_results)
    ok = sum(1 for nr in node_results if nr.execution_status == "completed")
    if total == 0:
        return 0.0, "no nodes executed"
    return ok / total, f"{ok}/{total} nodes completed"


def _default_local_reward(
    node: WorkflowNode,
    action_id: str,
    status: str,
    summary: str,
    exec_meta: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[float, str]:
    """Standaard lokale reward met kwaliteitschecks.

    Niet alleen 'completed' vs 'failed' — checkt ook of de output
    daadwerkelijk zinvol is (geen error page, minimale lengte, etc.).

    Bonusfactoren:
    - completed: +0.5
    - Content > 20 chars: +0.2
    - Geen error-patroon in output: +0.2
    - Bevat URLs (research): +0.1
    """
    if status != "completed":
        return -0.5, "failed"

    base = 0.5
    reasons = ["completed"]

    # Kwaliteitscheck 1: minimale content
    if summary and len(summary) > 20:
        base += 0.2
        reasons.append("content")
    else:
        base -= 0.2
        reasons.append("empty")

    # Kwaliteitscheck 2: error-patroon detectie
    if summary:
        error_patterns = [
            "403", "404", "500", "error", "timeout", "blocked",
            "cloudflare", "captcha", "access denied", "forbidden",
            "not found", "could not", "failed", "exception",
            "geen resultaat", "niets gevonden", "fout",
        ]
        lower_summary = summary.lower()
        has_error = any(p in lower_summary for p in error_patterns)
        if has_error:
            base -= 0.3
            reasons.append("error_content")
        else:
            base += 0.1
            reasons.append("clean_output")

    # Kwaliteitscheck 3: URLs in output (research nodes)
    if summary and ("http" in summary or "www." in summary):
        base += 0.1
        reasons.append("has_urls")

    # Kwaliteitscheck 4: structured data markers
    if summary and any(m in summary for m in ["{", "[", "<html", "<div", "class="]):
        base += 0.1
        reasons.append("structured")

    return max(-1.0, min(1.0, base)), "+".join(reasons)


# ---------------------------------------------------------------------------
# WorkflowRunner
# ---------------------------------------------------------------------------


class WorkflowRunner:
    """Sequential executor for a WorkflowDAG with policy learning.

    Typical usage::

        dag = WorkflowDAG(id="podcast", nodes=[...], edges=[...])
        runner = WorkflowRunner(dag)
        run = runner.execute(
            context={"topic": "AI Agents in 2026"},
            decision_fn=my_decision_fn,
            execute_fn=my_execute_fn,
            reward_fn=my_reward_fn,
        )
        print(run.to_dict())
    """

    def __init__(
        self,
        dag: WorkflowDAG,
        *,
        store: Optional[Any] = None,
        agent_id: str = "hermes",
        decision_fn: Optional[DecisionFn] = None,
        execute_fn: Optional[ExecuteFn] = None,
        reward_fn: Optional[RewardFn] = None,
        local_reward_fn: Optional[LocalRewardFn] = None,
        auto_reward: bool = True,
        discount_factor: Optional[float] = None,
    ) -> None:
        """

        Args:
            dag: The workflow DAG to execute.
            store: Optional agent-learning store for persisting episodes
                and rewards. If None, episodes are only held in memory
                (no learning persistence).
            agent_id: Agent identifier used when storing episodes.
            decision_fn: Callback that chooses an action for each node.
                Default: _default_decision (first action / context override).
            execute_fn: Callback that executes the chosen action.
                Default: _default_execute (calls node.execute if set).
            reward_fn: Callback that determines the global reward after
                all nodes complete. Default: _default_reward.
            local_reward_fn: Optional per-node reward callback, called
                immediately after each node executes. Gives a local signal
                independent of the global outcome. Default: None.
            auto_reward: If True, run credit assignment immediately after
                the run finishes. Set False to assign rewards later.
            discount_factor: Override the DAG's discount factor for
                credit assignment. Uses dag.discount_factor if None.
        """
        self.dag = dag
        self.store = store
        self.agent_id = agent_id
        self.decision_fn = decision_fn or _default_decision
        self.execute_fn = execute_fn or _default_execute
        self.reward_fn = reward_fn or _default_reward
        self.local_reward_fn = local_reward_fn
        self.auto_reward = auto_reward
        self._discount = (
            discount_factor if discount_factor is not None
            else dag.discount_factor
        )

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        *,
        decision_fn: Optional[DecisionFn] = None,
        execute_fn: Optional[ExecuteFn] = None,
        reward_fn: Optional[RewardFn] = None,
        auto_reward: Optional[bool] = None,
    ) -> WorkflowRun:
        """Run the DAG end-to-end.

        Args:
            context: Mutable dict passed through callbacks.
                Pre-populate with initial data (topic, config, etc.).
            decision_fn: Per-run override for the constructor's decision_fn.
            execute_fn: Per-run override for the constructor's execute_fn.
            reward_fn: Per-run override for the constructor's reward_fn.
            auto_reward: Per-run override for credit assignment.

        Returns:
            WorkflowRun with node_results populated.
        """
        dec = decision_fn or self.decision_fn
        exe = execute_fn or self.execute_fn
        rwd = reward_fn or self.reward_fn
        do_reward = auto_reward if auto_reward is not None else self.auto_reward

        run = WorkflowRun(
            dag_id=self.dag.id,
            dag_name=self.dag.name,
            context=context or {},
        )

        order = self.dag.topological_sort()
        logger.info(
            "WorkflowRunner[%s] executing %d nodes: %s",
            self.dag.id, len(order), " → ".join(order),
        )

        for node_id in order:
            node = self.dag.nodes[node_id]

            # ---- Step 1: Decision -----------------------------------------
            started_at = datetime.now(timezone.utc).isoformat()
            action_id, logprob, dec_meta = dec(node, run.context)

            node_result = NodeResult(
                node_id=node.id,
                node_name=node.name,
                task_id=node.task_id,
                action_id=action_id,
                action_name=node.actions_short.get(action_id, action_id),
                execution_status="running",
                started_at=started_at,
                distance_from_end=self.dag.distance_from_end(node.id),
                metadata={"decision_meta": dec_meta},
            )

            # ---- Step 2: Execute -------------------------------------------
            try:
                status, summary, exec_meta = exe(node, action_id, run.context)
                node_result.execution_status = status
                node_result.result_summary = summary
            except Exception as exc:
                node_result.execution_status = "failed"
                node_result.result_summary = f"Exception during execute: {exc}"
                node_result.error = str(exc)
                status = "failed"
                summary = node_result.result_summary

            node_result.completed_at = datetime.now(timezone.utc).isoformat()

            # ---- Step 2b: Store latency/cost in context for reward function --
            # CostAwareReward.compute() reads these keys from context.
            # Actual latency from timestamps (fallback to exec_meta if available).
            if started_at and node_result.completed_at:
                try:
                    t0 = datetime.fromisoformat(started_at)
                    t1 = datetime.fromisoformat(node_result.completed_at)
                    actual_latency = max((t1 - t0).total_seconds(), 0.001)
                except Exception:
                    actual_latency = exec_meta.get("latency_s", 5.0)
            else:
                actual_latency = exec_meta.get("latency_s", 5.0)
            run.context[f"_latency:{node.id}"] = actual_latency
            run.context[f"_cost:{node.id}"] = exec_meta.get("cost",
                None)  # None = fallback to estimate_cost in CostAwareReward

            # ---- Step 2c: Local reward (per-node, independent of global) ----
            local_reward_val: Optional[float] = None
            if self.local_reward_fn is not None:
                try:
                    lr_val, lr_reason = self.local_reward_fn(
                        node, action_id, status, summary, exec_meta, run.context
                    )
                    local_reward_val = round(lr_val, 4)
                    node_result.local_reward = local_reward_val
                    logger.info("  [%s] local reward = %+.4f (%s)",
                                node.id, local_reward_val, lr_reason)
                except Exception as exc:
                    logger.warning("  [%s] local reward callback failed: %s",
                                   node.id, exc)

            # ---- Step 3: Episode logging -----------------------------------
            if self.store and node.task_id:
                episode_id = self._log_episode(node, action_id, logprob,
                                               status, summary, run)
                node_result.episode_id = episode_id
                # Store local reward linked to this episode
                if local_reward_val is not None and episode_id:
                    self._store_local_reward(episode_id, local_reward_val, node.id)
            elif self.store and not node.task_id:
                logger.info("  [%s] no task_id — skipping episode log", node.id)
            else:
                logger.info("  [%s] no store — skipping episode log", node.id)

            run.node_results.append(node_result)

            # Short-circuit op failure? Nee — we blijven doorschieten naar
            # de volgende nodes zodat de reward functie kan zien wat er misging.
            # De DAG ontwerper bepaalt of een failure fatal is.

        # ---- Step 4: Global reward + credit assignment --------------------
        if do_reward:
            try:
                global_reward, reason = rwd(run.context, run.node_results)
                run.global_reward = global_reward
                run.global_reward_reason = reason
                self._assign_credit(run)
            except Exception as exc:
                logger.warning("Reward callback failed: %s", exc)
                run.global_reward = 0.0
                run.global_reward_reason = f"reward callback error: {exc}"

        return run

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------

    def _log_episode(
        self,
        node: WorkflowNode,
        action_id: str,
        logprob: Optional[float],
        status: str,
        summary: str,
        run: WorkflowRun,
    ) -> str:
        """Create and persist an Episode for this node's decision+execution."""
        try:
            from agent_learning.types import Episode
        except ImportError:
            logger.warning("agent_learning not installed — skipping episode")
            return ""

        episode = Episode(
            agent_id=self.agent_id,
            task_id=node.task_id or node.id,
            task_name=node.name,
            action_id=action_id,
            action_logprob=logprob,
            intent_summary=node.description or f"Execute {node.name}",
            input_summary=json.dumps({
                "node_id": node.id,
                "workflow_run": run.run_id,
                "workflow_dag": self.dag.id,
            }, default=str)[:500],
            expected_outcome=f"Complete {node.name} step in {self.dag.id}",
            execution_status=status,
            result_summary=summary[:500] if summary else "",
            context_features=node.metadata.get("context_features", {}),
            metadata={
                "workflow_dag": self.dag.id,
                "workflow_run": run.run_id,
                "node_id": node.id,
                "distance_from_end": self.dag.distance_from_end(node.id),
                "action_name": node.actions_short.get(action_id, action_id),
                "correct_action_id": node.metadata.get("correct_action_id"),
                "task_completed": status == "completed",
            },
            session_id=run.run_id,
        )
        try:
            stored_id = self.store.store_episode(episode)
            logger.info("  [%s] episode %s → action=%s status=%s",
                        node.id, stored_id[:8], action_id, status)
            return stored_id
        except Exception as exc:
            logger.warning("  [%s] failed to store episode: %s", node.id, exc)
            return ""

    # ------------------------------------------------------------------
    # Credit assignment
    # ------------------------------------------------------------------

    def _assign_credit(self, run: WorkflowRun) -> None:
        """Backpropagate global reward along the DAG path.

        Each node receives: global_reward × discount_factor^distance_from_end

        The credit reward is stored as a Reward record in the agent-learning
        store, linked to the node's episode. This allows the learner to
        pick it up during training via ``store_episode_reward()`` or the
        standard reward query path.
        """
        if run.global_reward is None:
            return

        gamma = self._discount
        global_r = run.global_reward
        logger.info("Credit assignment: global_reward=%.3f γ=%s", global_r, gamma)

        for nr in run.node_results:
            dist = nr.distance_from_end
            credit = global_r * (gamma ** dist)
            nr.credit_reward = round(credit, 6)

            logger.info("  [%s] dist=%d → credit=%.4f (γ^%d=%.4f)",
                        nr.node_id, dist, credit, dist, gamma ** dist)

            # Persist reward in store if we have an episode id
            if self.store and nr.episode_id:
                self._store_reward(nr.episode_id, nr.credit_reward, dist, global_r)

    def _store_reward(
        self,
        episode_id: str,
        credit: float,
        distance: int,
        global_reward: float,
    ) -> None:
        """Write a Reward record to the agent-learning store."""
        try:
            from agent_learning.types import Reward, RewardSource
        except ImportError:
            return

        reward = Reward(
            episode_id=episode_id,
            value=credit,
            source=RewardSource.AGGREGATE,
            agent_id=self.agent_id,
            metadata={
                "type": "credit_assignment",
                "discount_factor": self._discount,
                "distance_from_end": distance,
                "global_reward": global_reward,
                "workflow_dag": self.dag.id,
            },
        )
        try:
            self.store.store_reward(reward)
        except Exception as exc:
            logger.warning("Failed to store reward for episode %s: %s",
                           episode_id[:8], exc)

    def _store_local_reward(
        self,
        episode_id: str,
        value: float,
        node_id: str,
    ) -> None:
        """Store a per-node local reward (independent of global credit)."""
        try:
            from agent_learning.types import Reward, RewardSource
        except ImportError:
            return
        reward = Reward(
            episode_id=episode_id,
            value=value,
            source=RewardSource.METRIC,
            agent_id=self.agent_id,
            metadata={
                "type": "local_reward",
                "node_id": node_id,
                "workflow_dag": self.dag.id,
            },
        )
        try:
            self.store.store_reward(reward)
        except Exception as exc:
            logger.warning("Failed to store local reward: %s", exc)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, run: WorkflowRun) -> str:
        """Human-readable run summary."""
        lines = [
            f"Workflow: {self.dag.id} / run {run.run_id}",
            f"Nodes:    {len(run.node_results)}",
        ]
        for nr in run.node_results:
            flag = "✓" if nr.execution_status == "completed" else "✗"
            reward = f" [{nr.credit_reward:+.3f}]" if nr.credit_reward is not None else ""
            local = ""
            if hasattr(nr, "local_reward") and nr.local_reward is not None:
                local = f" local={nr.local_reward:+.3f}"
            lines.append(
                f"  {flag} {nr.node_name:<20} {nr.action_name or nr.action_id:<15}"
                f" {nr.execution_status}{reward}{local}"
            )
        if run.global_reward is not None:
            lines.append(f"Global reward: {run.global_reward:+.3f} — "
                         f"{run.global_reward_reason}")
        return "\n".join(lines)


__all__ = [
    "WorkflowRunner",
    "DecisionFn", "ExecuteFn", "RewardFn", "LocalRewardFn",
    "_default_decision", "_default_execute", "_default_reward",
    "_default_local_reward",
]
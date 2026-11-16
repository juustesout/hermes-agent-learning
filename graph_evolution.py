"""
graph_evolution.py — Level 3: Graph Evolution, meta-policy, multi-DAG orchestration.

Core capabilities:
1. **DagRegistry** — beheert meerdere DAGs, meta-policy voor DAG-selectie
2. **RunAnalyzer** — query historische runs uit store, per-node stats, reward trends
3. **GraphOptimizer** — optimaliseert γ, detecteert bottlenecks, suggesties
4. **MultiDagOrchestrator** — sequentiële/parallelle/conditionele DAG-compositie
5. **Dashboard** — HTML + terminal rapportage
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from html import escape
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from workflow_dag import NodeResult, WorkflowDAG, WorkflowRun
from workflow_runner import WorkflowRunner, DecisionFn, ExecuteFn, RewardFn
from workflow_visualize import ascii_graph, mermaid_run, run_report

logger = logging.getLogger("graph_evolution")


# ======================================================================
# 1. DAG REGISTRY + Meta-policy
# ======================================================================


class DagRegistry:
    """Registry of available workflow DAGs with meta-policy selection.

    The meta-policy is itself a TaskPolicy (reuse the agent-learning stack)
    that chooses which DAG to run based on context (time of day, queue,
    last run success, etc.).
    """

    def __init__(self, store=None, agent_id: str = "hermes") -> None:
        self._dags: Dict[str, WorkflowDAG] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}  # dag_id → metadata
        self.store = store
        self.agent_id = agent_id

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, dag: WorkflowDAG, **metadata) -> None:
        """Register a DAG with optional metadata (priority, schedule, tags)."""
        self._dags[dag.id] = dag
        self._meta[dag.id] = {
            "name": dag.name,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "run_count": 0,
            "last_run": None,
            "avg_global_reward": None,
            "total_runs": 0,
            "successful_runs": 0,
            **metadata,
        }

    def get(self, dag_id: str) -> Optional[WorkflowDAG]:
        return self._dags.get(dag_id)

    def list(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return (dag_id, name, metadata) for all registered DAGs."""
        return [(dag_id, dag.name, dict(self._meta.get(dag_id, {})))
                for dag_id, dag in self._dags.items()]

    # ------------------------------------------------------------------
    # Meta-policy: kies de beste DAG obv context
    # ------------------------------------------------------------------

    def select_dag(
        self,
        context: Dict[str, Any],
        *,
        preference: Optional[str] = None,
    ) -> Tuple[Optional[WorkflowDAG], str]:
        """Kies een DAG op basis van context en historische data.

        Returns (dag, reason) — dag is None als er geen geschikte is.
        """
        if not self._dags:
            return None, "no DAGs registered"

        # 1. Preference override
        if preference and preference in self._dags:
            return self._dags[preference], f"preference: {preference}"

        # 2. Context override
        context_dag = context.get("dag_id")
        if context_dag and context_dag in self._dags:
            return self._dags[context_dag], f"context dag_id: {context_dag}"

        # 3. Kies obv meta-data: DAG met hoogste avg reward of minst gedraaid
        candidates = []
        for dag_id, dag in self._dags.items():
            meta = self._meta[dag_id]
            avg_r = meta.get("avg_global_reward")
            score = (
                avg_r if avg_r is not None else 0.5,
                -meta.get("run_count", 0),  # prefer less-run DAGs for exploration
                meta.get("priority", 0),
            )
            candidates.append((score, dag_id, dag))

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, dag_id, dag = candidates[0]
        return dag, f"meta-policy: best score (reward={candidates[0][0][0]:.3f})"

    def _update_run_stats(self, dag_id: str, run: WorkflowRun) -> None:
        """Update metadata na een run."""
        meta = self._meta.get(dag_id)
        if meta is None:
            return
        meta["run_count"] = meta.get("run_count", 0) + 1
        meta["last_run"] = run.created_at
        if run.global_reward is not None:
            prev = meta.get("avg_global_reward")
            n = meta.get("total_runs", 0)
            if prev is None:
                meta["avg_global_reward"] = run.global_reward
            else:
                meta["avg_global_reward"] = (prev * n + run.global_reward) / (n + 1)
            meta["total_runs"] = n + 1
            if run.global_reward > 0.5:
                meta["successful_runs"] = meta.get("successful_runs", 0) + 1

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dags": {dag_id: dag.to_dict() for dag_id, dag in self._dags.items()},
            "meta": dict(self._meta),
        }


# ======================================================================
# 2. RUN ANALYZER
# ======================================================================


@dataclass
class NodeStats:
    """Aggregated statistics for one node across multiple runs."""
    node_id: str
    node_name: str
    total_runs: int = 0
    succeeded: int = 0
    failed: int = 0
    avg_credit: Optional[float] = None
    actions_used: Dict[str, int] = field(default_factory=Counter)
    actions_success: Dict[str, int] = field(default_factory=Counter)
    avg_latency: Optional[float] = None


@dataclass
class RunHistoryStats:
    """Aggregated statistics across multiple workflow runs."""
    dag_id: str
    dag_name: str
    total_runs: int = 0
    node_stats: Dict[str, NodeStats] = field(default_factory=dict)
    global_rewards: List[float] = field(default_factory=list)
    avg_global_reward: Optional[float] = None
    success_rate: Optional[float] = None
    avg_nodes_per_run: Optional[float] = None
    first_run: Optional[str] = None
    last_run: Optional[str] = None


class RunAnalyzer:
    """Query and analyze historical workflow runs from the store.

    Uses episodes tagged with workflow_dag/workflow_run in metadata
    to reconstruct run histories.
    """

    def __init__(self, store, agent_id: str = "hermes") -> None:
        self.store = store
        self.agent_id = agent_id

    def analyze_dag(self, dag_id: str, limit: int = 500) -> RunHistoryStats:
        """Aggregate statistics for all runs of a DAG."""
        eps = self.store.query_episodes(
            self.agent_id, limit=limit
        )
        # Filter to episodes belonging to this DAG
        dag_eps = [
            ep for ep in eps
            if ep.metadata.get("workflow_dag") == dag_id
        ]

        stats = RunHistoryStats(dag_id=dag_id, dag_name=dag_id)

        # Group by run_id (session_id = run_id in our setup)
        runs: Dict[str, List] = defaultdict(list)
        for ep in dag_eps:
            run_id = ep.metadata.get("workflow_run") or ep.session_id
            if run_id:
                runs[run_id].append(ep)

        stats.total_runs = len(runs)
        if not runs:
            return stats

        # Per-run aggregatie
        per_run_rewards: List[float] = []
        per_run_nodes: List[int] = []
        all_ok_runs = 0

        for run_id, episodes in runs.items():
            n_nodes = len(episodes)

            # Zoek globale reward — die zit in de reward records of metadata
            global_reward = None
            for ep in episodes:
                # Check metadata for credit_reward with distance=0 (publish)
                dist = ep.metadata.get("distance_from_end")
                if dist == 0:
                    # De publish-node heeft de volle reward gekregen
                    pass

            # Gebruik rewards uit store
            try:
                rewards = self.store.query_rewards(self.agent_id, limit=100)
                # Find aggregate rewards matching this run's episodes
                run_ep_ids = {ep.id for ep in episodes}
                for r in rewards:
                    if r.episode_id in run_ep_ids and r.source.value == "aggregate":
                        if global_reward is None or r.value > global_reward:
                            global_reward = r.value
            except Exception:
                pass

            if global_reward is not None:
                per_run_rewards.append(global_reward)

            per_run_nodes.append(n_nodes)
            ok = sum(1 for ep in episodes if ep.execution_status == "completed")
            if ok == n_nodes and n_nodes > 0:
                all_ok_runs += 1

            # Per-node stats
            for ep in episodes:
                nid = ep.metadata.get("node_id", ep.task_id)
                if nid not in stats.node_stats:
                    stats.node_stats[nid] = NodeStats(
                        node_id=nid,
                        node_name=ep.task_name or nid,
                    )
                ns = stats.node_stats[nid]
                ns.total_runs += 1
                if ep.execution_status == "completed":
                    ns.succeeded += 1
                    ns.actions_success[ep.action_id or "?"] += 1
                else:
                    ns.failed += 1
                ns.actions_used[ep.action_id or "?"] = \
                    ns.actions_used.get(ep.action_id or "?", 0) + 1

        # Compute aggregates
        if per_run_rewards:
            stats.global_rewards = per_run_rewards
            stats.avg_global_reward = sum(per_run_rewards) / len(per_run_rewards)
        if stats.total_runs > 0:
            stats.success_rate = all_ok_runs / stats.total_runs
        if per_run_nodes:
            stats.avg_nodes_per_run = sum(per_run_nodes) / len(per_run_nodes)

        if dag_eps:
            dates = sorted(set(ep.created_at for ep in dag_eps))
            if dates:
                stats.first_run = dates[0]
                stats.last_run = dates[-1]

        return stats

    def compare_dags(self, dag_ids: List[str]) -> Dict[str, RunHistoryStats]:
        """Compare statistics across multiple DAGs."""
        return {did: self.analyze_dag(did) for did in dag_ids}


# ======================================================================
# 3. GRAPH OPTIMIZER
# ======================================================================


@dataclass
class OptimizationSuggestion:
    """A concrete suggestion to improve a DAG."""
    type: str  # "gamma", "reorder", "bottleneck", "new_node", "merge_nodes"
    dag_id: str
    description: str
    current_value: Optional[Any] = None
    suggested_value: Optional[Any] = None
    expected_impact: Optional[str] = None
    confidence: float = 0.5  # 0-1


class GraphOptimizer:
    """Analyze run history and suggest DAG improvements.

    Capabilities:
    - Optimal γ bepalen obv reward volatility
    - Bottleneck nodes detecteren (hoge failure rate)
    - Overbodige nodes voorstellen (altijd success, weinig impact)
    - Nieuwe paden voorstellen (parallelle splitsingen)
    """

    def __init__(self, store, agent_id: str = "hermes") -> None:
        self.analyzer = RunAnalyzer(store, agent_id)

    def optimize_gamma(self, dag_id: str) -> OptimizationSuggestion:
        """Bepaal optimale discount factor obv reward volatility.

        Hoe hoger de reward volatility, hoe lager γ moet zijn
        (meer nadruk op immediate rewards in onvoorspelbare omgevingen).
        """
        stats = self.analyzer.analyze_dag(dag_id)
        if len(stats.global_rewards) < 3:
            return OptimizationSuggestion(
                type="gamma",
                dag_id=dag_id,
                description="Nog te weinig runs voor gamma-optimalisatie (min 3 nodig)",
                current_value=0.8,
                suggested_value=0.8,
                confidence=0.0,
            )

        rewards = stats.global_rewards
        volatility = (
            sum(abs(rewards[i] - rewards[i-1])
                for i in range(1, len(rewards)))
            / (len(rewards) - 1)
        )

        # Vuistregel: hoge volatility → lagere gamma
        if volatility > 0.3:
            suggested = 0.6
            impact = "hoge reward volatility — korte horizon beter"
        elif volatility > 0.15:
            suggested = 0.7
            impact = "gemiddelde volatility — matige discount"
        else:
            suggested = 0.85
            impact = "lage volatility — lange horizon mogelijk"

        return OptimizationSuggestion(
            type="gamma",
            dag_id=dag_id,
            description=f"Reward volatility = {volatility:.3f}",
            current_value=0.8,
            suggested_value=suggested,
            expected_impact=f"γ: 0.8 → {suggested} ({impact})",
            confidence=min(len(rewards) / 10, 0.9),
        )

    def find_bottlenecks(self, dag_id: str) -> List[OptimizationSuggestion]:
        """Detecteer nodes met hoge failure rate."""
        stats = self.analyzer.analyze_dag(dag_id)
        suggestions = []

        for nid, ns in stats.node_stats.items():
            if ns.total_runs < 2:
                continue
            failure_rate = ns.failed / ns.total_runs
            if failure_rate > 0.3:
                suggestions.append(OptimizationSuggestion(
                    type="bottleneck",
                    dag_id=dag_id,
                    description=f"Node {ns.node_name}: {failure_rate:.0%} failures",
                    current_value=ns.failed,
                    suggested_value=ns.succeeded,
                    expected_impact=f"Verlaag failure rate van {failure_rate:.0%} naar <15%",
                    confidence=min(failure_rate * 2, 0.95),
                ))

        return suggestions

    def suggest_reorder(self, dag_id: str) -> List[OptimizationSuggestion]:
        """Detecteer of node volgorde geoptimaliseerd kan worden."""
        stats = self.analyzer.analyze_dag(dag_id)
        suggestions = []

        # Als een vroege node vaak faalt en latere nodes wel slagen,
        # overweeg parallelisatie
        node_list = list(stats.node_stats.values())
        for i, ns in enumerate(node_list[:-1]):
            early_fail = ns.failed / max(ns.total_runs, 1)
            if early_fail > 0.3:
                # Zijn er latere nodes die het goed doen?
                later_ok = [
                    n for n in node_list[i+1:]
                    if n.succeeded / max(n.total_runs, 1) > 0.8
                ]
                if later_ok:
                    suggestions.append(OptimizationSuggestion(
                        type="reorder",
                        dag_id=dag_id,
                        description=f"Node {ns.node_name} faalt vaak ({early_fail:.0%}), "
                                    f"maar {len(later_ok)} latere nodes slagen wel",
                        expected_impact="Overweeg parallelle uitvoering of early-exit",
                        confidence=min(early_fail, 0.8),
                    ))

        return suggestions

    def full_report(self, dag_id: str) -> str:
        """Generate a full optimization report for a DAG."""
        stats = self.analyzer.analyze_dag(dag_id)

        lines = [
            f"Graph Optimization Report: {dag_id}",
            "=" * 50,
            f"Total runs analyzed: {stats.total_runs}",
            f"Avg global reward:   {stats.avg_global_reward:.4f}" if stats.avg_global_reward else "",
            f"Success rate:        {stats.success_rate:.1%}" if stats.success_rate else "",
            "",
            "Per-node statistics:",
        ]

        for nid, ns in sorted(stats.node_stats.items()):
            fail_pct = ns.failed / max(ns.total_runs, 1)
            actions_str = ", ".join(
                f"{a}({c})" for a, c in sorted(ns.actions_used.items(), key=lambda x: -x[1])
            )
            lines.append(
                f"  {ns.node_name:<20} "
                f"runs={ns.total_runs:<4} "
                f"✓{ns.succeeded:<3} ✗{ns.failed:<3} "
                f"({fail_pct:.0%} fail)"
            )
            if actions_str:
                lines.append(f"  {'':>20} actions: {actions_str}")

        lines.extend(["", "Suggestions:"])
        gamma_sugg = self.optimize_gamma(dag_id)
        lines.append(f"  [γ] {gamma_sugg.description}")
        if gamma_sugg.suggested_value != gamma_sugg.current_value:
            lines.append(f"       {gamma_sugg.current_value} → {gamma_sugg.suggested_value}")

        for b in self.find_bottlenecks(dag_id):
            lines.append(f"  [⚠] {b.description}")
            lines.append(f"       {b.expected_impact}")

        for r in self.suggest_reorder(dag_id):
            lines.append(f"  [↻] {r.description}")
            lines.append(f"       {r.expected_impact}")

        return "\n".join(lines)


# ======================================================================
# 4. MULTI-DAG ORCHESTRATOR
# ======================================================================


class MultiDagOrchestrator:
    """Voer meerdere DAGs uit in sequentie, parallel, of conditioneel.

    Composities:
    - Sequential DAGs: dag_a → dag_b → dag_c
    - Parallel DAGs:   dag_a || dag_b (simulated parallel)
    - Conditional:     if context.X then dag_a else dag_b
    - Loop:            repeat dag until condition
    - Meta-DAG:        DAG waarin elke node zelf een sub-DAG is
    """

    def __init__(
        self,
        registry: DagRegistry,
        store=None,
        agent_id: str = "hermes",
        global_decision_fn: Optional[DecisionFn] = None,
        global_execute_fn: Optional[ExecuteFn] = None,
        global_reward_fn: Optional[RewardFn] = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.agent_id = agent_id
        self.global_decision_fn = global_decision_fn
        self.global_execute_fn = global_execute_fn
        self.global_reward_fn = global_reward_fn
        self._runs: Dict[str, WorkflowRun] = {}  # compose_run_id → WorkflowRun

    # ------------------------------------------------------------------
    # Sequential execution
    # ------------------------------------------------------------------

    def run_sequential(
        self,
        dag_ids: List[str],
        context: Optional[Dict[str, Any]] = None,
        *,
        chain_context: bool = True,
        compose_reward: Optional[RewardFn] = None,
    ) -> Dict[str, WorkflowRun]:
        """Voer DAGs uit in sequentie: A → B → C.

        Als chain_context=True wordt de context van de vorige DAG
        doorgegeven aan de volgende (inclusief _result keys).
        """
        results: Dict[str, WorkflowRun] = {}

        for dag_id in dag_ids:
            dag = self.registry.get(dag_id)
            if dag is None:
                logger.warning("DAG %s not found — skipping", dag_id)
                continue

            runner = WorkflowRunner(
                dag,
                store=self.store,
                agent_id=self.agent_id,
                decision_fn=self.global_decision_fn,
                execute_fn=self.global_execute_fn,
                reward_fn=self.global_reward_fn,
            )
            run = runner.execute(context=context or {})
            results[dag_id] = run

            # Chain: geef resultaten door aan volgende DAG
            if chain_context and context is not None:
                for nr in run.node_results:
                    key = f"_dag:{dag_id}:{nr.node_id}"
                    context[key] = {
                        "status": nr.execution_status,
                        "summary": nr.result_summary,
                        "credit": nr.credit_reward,
                    }

            # Update registry stats
            self.registry._update_run_stats(dag_id, run)

        return results

    # ------------------------------------------------------------------
    # Conditional execution
    # ------------------------------------------------------------------

    def run_conditional(
        self,
        condition_fn: Callable[[Dict[str, Any]], str],
        dag_map: Dict[str, str],  # condition_value → dag_id
        context: Optional[Dict[str, Any]] = None,
        default_dag: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[WorkflowRun]]:
        """Kies een DAG obv condition_fn en voer die uit.

        condition_fn(context) returns een key die in dag_map staat.
        Returns (dag_id, run).
        """
        if context is None:
            context = {}
        try:
            choice = condition_fn(context)
        except Exception as exc:
            logger.warning("Condition function failed: %s", exc)
            choice = default_dag

        dag_id = dag_map.get(choice, default_dag)
        if dag_id is None:
            logger.warning("No matching DAG for choice=%s, no default", choice)
            return choice, None

        dag = self.registry.get(dag_id)
        if dag is None:
            logger.warning("DAG %s not found", dag_id)
            return choice, None

        runner = WorkflowRunner(
            dag, store=self.store, agent_id=self.agent_id,
            decision_fn=self.global_decision_fn,
            execute_fn=self.global_execute_fn,
            reward_fn=self.global_reward_fn,
        )
        run = runner.execute(context=context)
        self.registry._update_run_stats(dag_id, run)
        return dag_id, run

    # ------------------------------------------------------------------
    # Looped execution
    # ------------------------------------------------------------------

    def run_looped(
        self,
        dag_id: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        max_iterations: int = 5,
        stop_condition: Optional[Callable[[Dict[str, Any], WorkflowRun], bool]] = None,
        improve_threshold: float = 0.8,
    ) -> List[WorkflowRun]:
        """Herhaal een DAG tot stop_condition of max_iterations.

        Handig voor iteratieve verbetering: draai podcast, evalueer,
        verbeter obv resultaat, draai opnieuw.
        """
        dag = self.registry.get(dag_id)
        if dag is None:
            logger.error("DAG %s not found", dag_id)
            return []

        if context is None:
            context = {}

        runs: List[WorkflowRun] = []
        for iteration in range(1, max_iterations + 1):
            context["_iteration"] = iteration
            context["_previous_runs"] = len(runs)

            runner = WorkflowRunner(
                dag, store=self.store, agent_id=self.agent_id,
                decision_fn=self.global_decision_fn,
                execute_fn=self.global_execute_fn,
                reward_fn=self.global_reward_fn,
            )
            run = runner.execute(context=context)
            runs.append(run)
            self.registry._update_run_stats(dag_id, run)

            # Check stop condition
            if stop_condition is not None and stop_condition(context, run):
                logger.info("Looped run stopped by condition at iteration %d", iteration)
                break

            # Auto-stop bij hoge reward
            if run.global_reward is not None and run.global_reward >= improve_threshold:
                logger.info("Looped run: reward %.3f ≥ %.2f, stopping", run.global_reward, improve_threshold)
                break

        return runs

    # ------------------------------------------------------------------
    # Meta-DAG: run a DAG of DAGs
    # ------------------------------------------------------------------

    def run_meta_dag(
        self,
        meta_dag: WorkflowDAG,
        node_to_dag: Dict[str, str],  # meta node_id → dag_id
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Voer een meta-DAG uit waarbij elke node een sub-DAG is.

        De meta-DAG bepaalt de volgorde, elke node = een sub-DAG.
        """
        results: Dict[str, Any] = {}
        context = context or {}
        order = meta_dag.topological_sort()

        for meta_node_id in order:
            meta_node = meta_dag.nodes[meta_node_id]
            sub_dag_id = node_to_dag.get(meta_node_id)
            if sub_dag_id is None:
                logger.warning("Meta node %s has no sub-DAG mapping", meta_node_id)
                continue

            sub_dag = self.registry.get(sub_dag_id)
            if sub_dag is None:
                logger.warning("Sub-DAG %s not found", sub_dag_id)
                continue

            runner = WorkflowRunner(
                sub_dag, store=self.store, agent_id=self.agent_id,
                decision_fn=self.global_decision_fn,
                execute_fn=self.global_execute_fn,
                reward_fn=self.global_reward_fn,
            )
            run = runner.execute(context=context)
            results[meta_node_id] = run
            self.registry._update_run_stats(sub_dag_id, run)

            # Forward results
            context[f"_meta:{meta_node_id}"] = {
                "dag_id": sub_dag_id,
                "global_reward": run.global_reward,
                "n_nodes": len(run.node_results),
            }

        return results

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, WorkflowRun]:
        return dict(self._runs)


# ======================================================================
# 5. DASHBOARD
# ======================================================================


class Dashboard:
    """Generate terminal and HTML reports from run history."""

    def __init__(self, store, agent_id: str = "hermes") -> None:
        self.analyzer = RunAnalyzer(store, agent_id)

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------

    def terminal_report(self, dag_id: str, limit: int = 20) -> str:
        """Generate a terminal-friendly run history report."""
        eps = self.analyzer.store.query_episodes(
            self.analyzer.agent_id, limit=limit
        )
        dag_eps = [
            ep for ep in eps
            if ep.metadata.get("workflow_dag") == dag_id
        ]

        if not dag_eps:
            return f"No runs found for DAG {dag_id!r}"

        # Group by run
        runs: Dict[str, List] = defaultdict(list)
        for ep in dag_eps:
            run_id = ep.metadata.get("workflow_run") or ep.session_id or "?"
            runs[run_id].append(ep)

        lines = [
            f"DASHBOARD: {dag_id}",
            f"{'=' * 55}",
            f"Total runs: {len(runs)}  |  Total episodes: {len(dag_eps)}",
            "",
        ]

        # Per-run summary, newest first
        for run_id, episodes in sorted(
            runs.items(),
            key=lambda x: max(ep.created_at for ep in x[1]),
            reverse=True,
        )[:10]:
            run_date = max(ep.created_at[:16] for ep in episodes)
            n_ok = sum(1 for ep in episodes if ep.execution_status == "completed")
            n_total = len(episodes)
            status = "✓" if n_ok == n_total else f"✗ ({n_ok}/{n_total})"

            lines.append(f"  [{run_date}] run_{run_id[:6]}  {status}")
            for ep in sorted(episodes, key=lambda e: e.metadata.get("distance_from_end", 0), reverse=True):
                dist = ep.metadata.get("distance_from_end", "?")
                action = ep.metadata.get("action_name", ep.action_id or "?")
                ep_status = "✓" if ep.execution_status == "completed" else "✗"
                lines.append(f"             [{dist}] {ep.task_name or ep.task_id:<20} "
                            f"{ep_status} {action:<20} {ep.execution_status}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML dashboard (self-contained, no external deps)
    # ------------------------------------------------------------------

    def html_dashboard(
        self,
        dag_ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> str:
        """Generate a self-contained HTML dashboard with charts."""
        def _esc(value: Any) -> str:
            return escape(str(value), quote=True)

        all_eps = self.analyzer.store.query_episodes(
            self.analyzer.agent_id, limit=limit
        )
        if not all_eps:
            return "<html><body><h1>No data</h1></body></html>"

        # Filter by DAG
        if dag_ids:
            dag_eps = [ep for ep in all_eps
                      if ep.metadata.get("workflow_dag") in dag_ids]
        else:
            dag_eps = all_eps

        # Per-DAG data
        by_dag: Dict[str, List] = defaultdict(list)
        for ep in dag_eps:
            by_dag[ep.metadata.get("workflow_dag", "ungrouped")].append(ep)

        # Build HTML
        html = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Hermes Workflow Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:system-ui,sans-serif}
body{background:#0d1117;color:#c9d1d9;padding:2rem}
h1{color:#58a6ff;margin-bottom:1rem}
h2{color:#f0f6fc;margin:1.5rem 0 0.5rem}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;margin-bottom:1rem}
.card h3{color:#c9d1d9;font-size:1rem;margin-bottom:0.5rem}
.stat{display:inline-block;margin:0.25rem;padding:0.25rem 0.6rem;background:#21262d;border-radius:4px;
  color:#8b949e;font-size:0.85rem}
.stat strong{color:#58a6ff}
.bar{display:flex;height:1.5rem;margin:0.25rem 0;border-radius:4px;overflow:hidden}
.bar-ok{background:#238636;flex:1}
.bar-fail{background:#da3633;flex:1}
table{width:100%;border-collapse:collapse;font-size:0.85rem}
th{text-align:left;color:#8b949e;border-bottom:1px solid #30363d;padding:0.5rem;font-weight:600}
td{padding:0.4rem 0.5rem;border-bottom:1px solid #21262d}
tr:hover td{background:#1c2128}
.credit{color:#d29922}
.fail{color:#da3633}
.ok{color:#3fb950}
</style></head>
<body>
<h1>Hermes Workflow Dashboard</h1>
<p style="color:#8b949e">"""+str(len(dag_eps))+""" episodes across """+str(len(by_dag))+""" DAGs</p>
<hr style="border-color:#30363d;margin:1rem 0">
"""
        for dag_id, episodes in sorted(by_dag.items()):
            # Per-DAG stats
            by_run: Dict[str, List] = defaultdict(list)
            for ep in episodes:
                rid = ep.metadata.get("workflow_run") or ep.session_id or "?"
                by_run[rid].append(ep)

            n_runs = len(by_run)
            n_eps = len(episodes)
            total_ok = sum(1 for ep in episodes if ep.execution_status == "completed")
            total_fail = n_eps - total_ok
            ok_rate = total_ok / n_eps if n_eps > 0 else 0

            html += f"""<div class="card">
<h2>{_esc(dag_id)}</h2>
<span class="stat"><strong>{n_runs}</strong> runs</span>
<span class="stat"><strong>{n_eps}</strong> episodes</span>
<span class="stat"><strong>{total_ok}✓</strong> / {total_fail}✗</span>
<span class="stat"><strong>{ok_rate:.0%}</strong> success</span>
<div class="bar"><div class="bar-ok" style="flex:{total_ok}"></div>
<div class="bar-fail" style="flex:{total_fail or 1}"></div></div>
<table><tr><th>Run</th><th>Node</th><th>Action</th><th>Status</th><th>Credit</th></tr>
"""
            # Last 3 runs
            for run_id, run_eps in sorted(by_run.items(),
                key=lambda x: max(ep.created_at for ep in x[1]), reverse=True)[:3]:
                for ep in sorted(run_eps, key=lambda e: e.metadata.get("distance_from_end", 0), reverse=True):
                    dist = ep.metadata.get("distance_from_end", "")
                    node = ep.task_name or ep.task_id or "?"
                    action = ep.metadata.get("action_name", ep.action_id or "?")
                    status = ep.execution_status or "?"
                    credit = ep.metadata.get("credit_reward", "")
                    scls = "ok" if status == "completed" else "fail"
                    credit_str = f"<span class='credit'>{credit:+.3f}</span>" if isinstance(credit, (int, float)) else ""
                    html += (
                        f"<tr><td>{_esc(str(run_id)[:6])}</td>"
                        f"<td>[{_esc(dist)}] {_esc(node)}</td>"
                        f"<td>{_esc(action)}</td>"
                    )
                    html += f"<td class='{scls}'>{_esc(status)}</td><td>{credit_str}</td></tr>\n"

            html += "</table></div>\n"

        html += """
<hr style="border-color:#30363d;margin:2rem 0">
<p style="color:#8b949e;font-size:0.8rem">
Generated by Hermes agent-learning · """ + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + """
</p>
</body></html>
"""
        return html


# ======================================================================
# Helpers
# ======================================================================


def get_default_store():
    """Get the default agent-learning store."""
    from agent_learning.storage.cosmos import get_default_store as _gds
    return _gds()


__all__ = [
    "DagRegistry",
    "RunAnalyzer", "RunHistoryStats", "NodeStats",
    "GraphOptimizer", "OptimizationSuggestion",
    "MultiDagOrchestrator",
    "Dashboard",
]
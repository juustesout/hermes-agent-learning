"""
skill_discovery.py — Mine recurring action sequences from the store.

A "skill" is a recurring sequence of (node_id, action_id) pairs that
produces a successful outcome. Think of it as a reusable workflow fragment:
instead of learning each edge independently, the agent can recognise
"oh, this is the 'quick research' skill" and bias its decisions accordingly.

Discovery pipeline:
1. Load episodes from store, group by workflow_run
2. Extract action sequences per run (ordered by distance_from_end)
3. Find common subsequences across runs (prefixspan / n-gram)
4. Cluster runs by action pattern
5. Report discovered skills with frequency, avg reward, confidence

Usage:
    from skill_discovery import SkillDiscovery
    disc = SkillDiscovery(store)
    skills = disc.discover(min_support=2)
    for s in skills:
        print(s.name, s.confidence)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("skill_discovery")


@dataclass
class DiscoveredSkill:
    """A recurring action-sequence pattern mined from the store.

    Attributes:
        name: Human-readable name (auto-generated from sequence).
        sequence: List of (node_id, action_id) tuples.
        frequency: How many runs this pattern appears in.
        total_runs: Total runs analyzed.
        avg_reward: Average global reward of runs using this pattern.
        confidence: 0-1 score based on frequency × success.
        nodes: Set of node ids involved.
        actions: Set of action ids involved.
    """
    name: str
    sequence: List[Tuple[str, str]]
    frequency: int = 0
    total_runs: int = 0
    avg_reward: Optional[float] = None
    confidence: float = 0.0
    nodes: set = field(default_factory=set)
    actions: set = field(default_factory=set)


@dataclass
class RunTrace:
    """A single workflow run reconstructed from episodes."""
    run_id: str
    dag_id: str
    steps: List[Tuple[str, str, int]]  # (node_id, action_id, distance_from_end)
    global_reward: Optional[float] = None
    success: bool = False

    @property
    def action_sequence(self) -> List[Tuple[str, str]]:
        """Ordered sequence of (node_id, action_id) from start to finish."""
        # Sort by distance_from_end descending (high dist = early in DAG)
        sorted_steps = sorted(self.steps, key=lambda s: -s[2])
        return [(n, a) for n, a, _ in sorted_steps]

    @property
    def action_ids(self) -> List[str]:
        return [a for _, a in self.action_sequence]

    @property
    def node_ids(self) -> List[str]:
        return [n for n, _ in self.action_sequence]


class SkillDiscovery:
    """Mine the episode store for recurring action-sequence patterns.

    Uses n-gram based subsequence mining: finds all contiguous action
    subsequences of length 2+ that appear in multiple runs.

    Also supports:
    - Clustering runs by edit-distance on action sequences
    - Reporting skill stats (frequency, avg reward, confidence)
    """

    def __init__(self, store, agent_id: str = "hermes", min_ngram: int = 2) -> None:
        self.store = store
        self.agent_id = agent_id
        self.min_ngram = min_ngram

    # ------------------------------------------------------------------
    # Reconstruct runs from episodes
    # ------------------------------------------------------------------

    def _load_runs(self, dag_id: Optional[str] = None,
                   limit: int = 500) -> List[RunTrace]:
        """Reconstruct workflow runs from stored episodes."""
        eps = self.store.query_episodes(self.agent_id, limit=limit)

        # Group by run
        by_run: Dict[str, List] = defaultdict(list)
        for ep in eps:
            dag = ep.metadata.get("workflow_dag")
            if dag_id and dag != dag_id:
                continue
            run_id = ep.metadata.get("workflow_run") or ep.session_id
            if not run_id:
                continue
            by_run[run_id].append(ep)

        runs: List[RunTrace] = []
        for run_id, episodes in by_run.items():
            dag = episodes[0].metadata.get("workflow_dag", "?")
            steps = []
            for ep in episodes:
                node = ep.metadata.get("node_id", ep.task_id)
                action = ep.action_id or "?"
                dist = ep.metadata.get("distance_from_end", 0)
                if isinstance(dist, str):
                    try:
                        dist = int(dist)
                    except ValueError:
                        dist = 0
                steps.append((node, action, dist))

            # Find reward from store
            reward = None
            try:
                for r in self.store.query_rewards(self.agent_id, limit=100):
                    if r.episode_id in {ep.id for ep in episodes}:
                        if r.source.value == "aggregate":
                            reward = r.value
                            break
            except Exception:
                pass

            all_ok = all(
                ep.execution_status == "completed" for ep in episodes
            )
            runs.append(RunTrace(
                run_id=run_id,
                dag_id=dag,
                steps=steps,
                global_reward=reward,
                success=all_ok,
            ))

        return runs

    # ------------------------------------------------------------------
    # N-gram mining
    # ------------------------------------------------------------------

    def _mine_ngrams(
        self,
        runs: List[RunTrace],
        min_support: int = 2,
    ) -> List[Tuple[Tuple, int, float]]:
        """Find all contiguous action subsequences that appear in ≥min_support runs.

        Returns list of (ngram_tuple, frequency, avg_reward).
        """
        # Index: ngram → set of run_ids
        ngram_index: Dict[Tuple, set] = defaultdict(set)
        ngram_rewards: Dict[Tuple, List[float]] = defaultdict(list)

        for run in runs:
            actions = run.action_ids
            for start in range(len(actions)):
                for length in range(self.min_ngram, len(actions) - start + 1):
                    ngram = tuple(actions[start:start + length])
                    ngram_index[ngram].add(run.run_id)
                    if run.global_reward is not None:
                        ngram_rewards[ngram].append(run.global_reward)

        # Filter by support
        results = []
        for ngram, run_ids in ngram_index.items():
            freq = len(run_ids)
            if freq >= min_support:
                avg_r = (
                    sum(ngram_rewards[ngram]) / len(ngram_rewards[ngram])
                    if ngram_rewards.get(ngram) else None
                )
                results.append((ngram, freq, avg_r))

        # Sort by frequency desc, then length desc
        results.sort(key=lambda x: (-x[1], -len(x[0])))
        return results

    # ------------------------------------------------------------------
    # Main discovery
    # ------------------------------------------------------------------

    def discover(
        self,
        dag_id: Optional[str] = None,
        min_support: int = 2,
        limit: int = 500,
    ) -> List[DiscoveredSkill]:
        """Run full skill discovery pipeline.

        Args:
            dag_id: Optional filter to one DAG.
            min_support: Minimum number of runs a pattern must appear in.
            limit: Max episodes to query.

        Returns:
            List of DiscoveredSkill, sorted by confidence descending.
        """
        runs = self._load_runs(dag_id=dag_id, limit=limit)
        if not runs:
            logger.info("No runs found for discovery")
            return []

        ngrams = self._mine_ngrams(runs, min_support=min_support)
        total_runs = len(runs)

        skills: List[DiscoveredSkill] = []
        seen = set()

        for ngram, freq, avg_r in ngrams:
            # Skip if this ngram is fully contained in a longer one with same support
            key = str(ngram)
            if key in seen:
                continue

            # Build a name
            name = " → ".join(ngram[:4])
            if len(ngram) > 4:
                name += " → …"

            # Confidence: frequency ratio × reward factor
            freq_ratio = freq / max(total_runs, 1)
            reward_factor = max(0, (avg_r or 0.5))  # 0-1
            confidence = min(freq_ratio * reward_factor * 2, 1.0)

            # Map back to nodes — find a run that has this pattern
            example_nodes = set()
            example_actions = set(ngram)
            for run in runs:
                seq_str = str(run.action_ids)
                if all(a in seq_str for a in ngram):
                    for n, a in run.action_sequence:
                        if a in ngram:
                            example_nodes.add(n)
                    break

            skills.append(DiscoveredSkill(
                name=name,
                sequence=[("?", a) for a in ngram],
                frequency=freq,
                total_runs=total_runs,
                avg_reward=round(avg_r, 4) if avg_r else None,
                confidence=round(confidence, 4),
                nodes=example_nodes,
                actions=set(ngram),
            ))

            seen.add(key)

        return skills

    # ------------------------------------------------------------------
    # Cluster runs by action similarity
    # ------------------------------------------------------------------

    def cluster_runs(
        self,
        dag_id: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, List[RunTrace]]:
        """Group runs by identical action sequences.

        Returns dict: action_sequence_string → [runs]
        """
        runs = self._load_runs(dag_id=dag_id, limit=limit)
        clusters: Dict[str, List[RunTrace]] = defaultdict(list)
        for run in runs:
            key = "|".join(run.action_ids)
            clusters[key].append(run)
        return dict(clusters)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, dag_id: Optional[str] = None, min_support: int = 2) -> str:
        """Human-readable discovery report."""
        skills = self.discover(dag_id=dag_id, min_support=min_support)
        runs = self._load_runs(dag_id=dag_id)

        lines = [
            f"Skill Discovery Report",
            f"{'=' * 50}",
            f"DAG:      {dag_id or 'all'}",
            f"Runs:     {len(runs)}",
            f"Skills:   {len(skills)}",
            f"Min support: {min_support}",
            "",
        ]

        if not skills:
            lines.append("(no skills discovered — need more runs)")
            return "\n".join(lines)

        lines.append(f"{'Skill':<40} {'Freq':<6} {'Conf':<6} {'AvgReward':<10} {'Actions':<20}")
        lines.append(f"{'─'*39} {'─'*5} {'─'*5} {'─'*9} {'─'*19}")

        for s in skills:
            actions = ", ".join(sorted(s.actions)[:4])
            reward = f"{s.avg_reward:+.3f}" if s.avg_reward else "—"
            lines.append(
                f"{s.name:<39} {s.frequency:<5} {s.confidence:<5.2f} "
                f"{reward:<9} {actions:<19}"
            )

        # Clusters
        clusters = self.cluster_runs(dag_id=dag_id)
        if len(clusters) > 1:
            lines.extend(["", "Action clusters (unique sequences):"])
            for seq, cluster_runs in sorted(
                clusters.items(), key=lambda x: -len(x[1])
            ):
                ok = sum(1 for r in cluster_runs if r.success)
                lines.append(
                    f"  [{len(cluster_runs)} runs, {ok} OK]  {seq[:60]}"
                )

        return "\n".join(lines)


__all__ = [
    "SkillDiscovery", "DiscoveredSkill", "RunTrace",
]
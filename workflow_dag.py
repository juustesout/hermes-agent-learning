"""
workflow_dag.py — Workflow DAG dataclasses for multi-step path learning.

Defines the graph structure: nodes (decisions/steps) and edges (ordering).

A WorkflowDAG is a directed acyclic graph where each node represents one
step in a workflow (e.g. "research", "script", "audio"). Each node has an
associated task_id that maps to a TaskPolicy in the agent-learning store,
so each step learns independently which action works best for its context.

Example podcast DAG:
    topic_select → research → script → audio → video → publish
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class WorkflowNode:
    """One step in a workflow DAG.

    Each node carries the metadata needed to make a policy decision,
    execute the step, and log an episode for learning.

    Attributes:
        id: Unique node identifier (e.g. "research", "script").
        name: Human-readable name for logging / visualisation.
        description: What this step does.
        task_id: agent-learning task id for the TaskPolicy that decides
            *how* to execute this step (e.g. "choose-research-source").
        action_ids: List of valid action ids the policy chooses from.
            Each action corresponds to a concrete execution strategy.
        actions_short: Short human labels for each action_id, used in
            visualisation and logging (e.g. "webclaw|jina|manual").
        execute: Optional callback that performs the step. Called with
            (node, chosen_action_id, run_context) and returns a dict
            with at least {"status": "completed"|"failed", ...}.
        metadata: Arbitrary extra data (LLM prompts, config, etc.).
    """
    id: str
    name: str
    description: str = ""
    task_id: Optional[str] = None
    action_ids: List[str] = field(default_factory=list)
    actions_short: Dict[str, str] = field(default_factory=dict)
    execute: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_id": self.task_id,
            "action_ids": list(self.action_ids),
            "actions_short": dict(self.actions_short),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowNode":
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            description=data.get("description", ""),
            task_id=data.get("task_id"),
            action_ids=data.get("action_ids", []),
            actions_short=data.get("actions_short", {}),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@dataclass
class WorkflowEdge:
    """A directed dependency between two workflow steps.

    source ---> target  means source must complete before target starts.
    """
    source: str  # source node id
    target: str  # target node id
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

_ExecuteResult = Dict[str, Any]
_RunContext = Dict[str, Any]  # mutable dict passed through the entire run


@dataclass
class WorkflowRun:
    """Result of executing a WorkflowDAG once.

    Stores the per-node outcomes so the runner can propagate rewards
    back along the path after the final result is known.
    """
    dag_id: str
    dag_name: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # Ordered list of node results (topological order + executed order)
    node_results: List["NodeResult"] = field(default_factory=list)
    # Global reward set after the workflow completes
    global_reward: Optional[float] = None
    global_reward_reason: Optional[str] = None
    # Mutable context shared across all nodes in this run
    context: _RunContext = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: __import__("datetime")
                             .datetime.now(__import__("datetime").timezone.utc)
                             .isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "dag_name": self.dag_name,
            "run_id": self.run_id,
            "node_results": [nr.to_dict() for nr in self.node_results],
            "global_reward": self.global_reward,
            "global_reward_reason": self.global_reward_reason,
            "context": {k: v for k, v in self.context.items()
                        if not k.startswith("_")},
            "created_at": self.created_at,
        }


@dataclass
class NodeResult:
    """Outcome of executing one node in a workflow run."""
    node_id: str
    node_name: str
    task_id: Optional[str]
    action_id: Optional[str]
    action_name: Optional[str]
    episode_id: Optional[str] = None
    execution_status: str = "pending"
    result_summary: Optional[str] = None
    # Reward assigned via credit assignment (backpropagated)
    credit_reward: Optional[float] = None
    # Distance from end of the path (for discounting)
    distance_from_end: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "task_id": self.task_id,
            "action_id": self.action_id,
            "action_name": self.action_name,
            "episode_id": self.episode_id,
            "execution_status": self.execution_status,
            "result_summary": self.result_summary,
            "credit_reward": self.credit_reward,
            "distance_from_end": self.distance_from_end,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class WorkflowDAG:
    """A directed acyclic graph of workflow steps.

    Typical usage:

        dag = WorkflowDAG(
            id="podcast",
            name="Podcast Productie",
            nodes=[...],
            edges=[("topic_select", "research"), ...],
        )
        runner = WorkflowRunner(dag, store)
        run = runner.execute(context={"topic": "..."})
    """

    def __init__(
        self,
        id: str,
        name: str = "",
        nodes: Optional[List[WorkflowNode]] = None,
        edges: Optional[List[WorkflowEdge]] = None,
        discount_factor: float = 0.8,
        metadata: Dict[str, Any] = None,
    ) -> None:
        self.id = id
        self.name = name or id
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self._outgoing: Dict[str, List[str]] = {}  # node_id → [target ids]
        self._incoming: Dict[str, List[str]] = {}  # node_id → [source ids]
        self.discount_factor = discount_factor
        self.metadata = metadata or {}

        if nodes:
            for n in nodes:
                self.add_node(n)
        if edges:
            for e in edges:
                self.add_edge(e)

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    def add_node(self, node: WorkflowNode) -> None:
        self.nodes[node.id] = node
        self._outgoing.setdefault(node.id, [])
        self._incoming.setdefault(node.id, [])

    def add_edge(self, edge: WorkflowEdge) -> None:
        if edge.source not in self.nodes:
            raise KeyError(f"Source node {edge.source!r} not in DAG")
        if edge.target not in self.nodes:
            raise KeyError(f"Target node {edge.target!r} not in DAG")
        self.edges.append(edge)
        self._outgoing.setdefault(edge.source, []).append(edge.target)
        self._incoming.setdefault(edge.target, []).append(edge.source)

    def add(self, source: str, target: str) -> "WorkflowEdge":
        """Shorthand: dag.add("a", "b") creates edge a→b."""
        edge = WorkflowEdge(source=source, target=target)
        self.add_edge(edge)
        return edge

    # ------------------------------------------------------------------
    # Topological order
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Return node ids in topological order (leaves-first linearisation).

        Raises ValueError if the graph contains a cycle.
        """
        # Kahn's algorithm
        in_degree = {
            nid: len(self._incoming[nid]) for nid in self.nodes
        }
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for target in self._outgoing[nid]:
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        if len(order) != len(self.nodes):
            raise ValueError(
                f"Cycle detected — only {len(order)}/{len(self.nodes)} "
                f"nodes can be ordered"
            )
        return order

    def root_nodes(self) -> List[str]:
        """Nodes with no incoming edges (DAG entry points)."""
        return [nid for nid in self.nodes if not self._incoming[nid]]

    def leaf_nodes(self) -> List[str]:
        """Nodes with no outgoing edges (DAG exit points / final steps)."""
        return [nid for nid in self.nodes if not self._outgoing[nid]]

    def predecessors(self, node_id: str) -> List[str]:
        return list(self._incoming.get(node_id, []))

    def successors(self, node_id: str) -> List[str]:
        return list(self._outgoing.get(node_id, []))

    # ------------------------------------------------------------------
    # Distance computation (for credit assignment)
    # ------------------------------------------------------------------

    def _max_distance_to_leaf(self, node_id: str, memo: Dict[str, int]) -> int:
        """Longest path from this node to any leaf (edge count)."""
        if node_id in memo:
            return memo[node_id]
        successors = self._outgoing[node_id]
        if not successors:
            memo[node_id] = 0
            return 0
        best = 0
        for s in successors:
            dist = self._max_distance_to_leaf(s, memo) + 1
            if dist > best:
                best = dist
        memo[node_id] = best
        return best

    def distance_from_end(self, node_id: str) -> int:
        """How many edges from this node to the farthest leaf.

        Used as the step count for discounting: γ^distance_from_end.
        """
        return self._max_distance_to_leaf(node_id, {})

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "discount_factor": self.discount_factor,
            "metadata": self.metadata,
        }

    def to_mermaid(self) -> str:
        """Return a Mermaid flowchart definition for this DAG."""
        lines = ["```mermaid", "flowchart LR"]
        for nid, node in self.nodes.items():
            label = node.name.replace('"', "'")
            lines.append(f'    {nid}["{label}"]')
        lines.append("")
        for edge in self.edges:
            lines.append(f"    {edge.source} --> {edge.target}")
        lines.append("```")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDAG":
        nodes = [WorkflowNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [WorkflowEdge.from_dict(e) for e in data.get("edges", [])]
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            nodes=nodes,
            edges=edges,
            discount_factor=data.get("discount_factor", 0.8),
            metadata=data.get("metadata"),
        )


__all__ = [
    "WorkflowNode", "WorkflowEdge", "WorkflowDAG",
    "WorkflowRun", "NodeResult",
    "_ExecuteResult", "_RunContext",
]
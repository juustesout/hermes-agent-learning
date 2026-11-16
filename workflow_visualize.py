"""
workflow_visualize.py — Visualiseer WorkflowDAGs en runs.

Output formats:
- Mermaid flowchart (ingebouwd in WorkflowDAG.to_mermaid())
- ASCII art graph (voor CLI gebruik zonder browser)
- Run summary met rewards per node
"""

from __future__ import annotations

from typing import List, Optional

from workflow_dag import NodeResult, WorkflowDAG, WorkflowRun


def ascii_graph(dag: WorkflowDAG) -> str:
    """Teken de DAG als ASCII art, met nodes horizontaal."""
    order = dag.topological_sort()
    if not order:
        return "(empty DAG)"

    # Max node name length for alignment
    col_width = max(
        len(dag.nodes[nid].name) for nid in order
    ) + 2
    col_width = max(col_width, 12)

    lines: List[str] = []
    lines.append(f"DAG: {dag.id} — {dag.name}")
    lines.append(f"Discount γ = {dag.discount_factor}")
    lines.append("")

    # Arrow connector line
    arrow_line = "  "
    for nid in order:
        node = dag.nodes[nid]
        label = f" [{node.id}] {node.name}"
        arrow_line += f"{'─' * 3}▶ {label}"
    lines.append(arrow_line)

    # Second line: action sets
    action_line = "  " + " " * 6
    for nid in order:
        node = dag.nodes[nid]
        actions = ", ".join(node.actions_short.values()) if node.actions_short else "—"
        action_line += f"  ({actions})  "
    lines.append(action_line)

    # Legend
    lines.append("")
    for nid in order:
        node = dag.nodes[nid]
        task = node.task_id or "(no policy)"
        lines.append(f"  {node.id}: {node.description or node.name}")
        lines.append(f"         task={task}")
        if node.action_ids:
            short = {v: k for k, v in node.actions_short.items()}
            actions_str = ", ".join(
                f"{a}" + (f" ({short[a]})" if a in short else "")
                for a in node.action_ids
            )
            lines.append(f"         actions=[{actions_str}]")

    return "\n".join(lines)


def run_report(run: WorkflowRun) -> str:
    """ASCII report van een uitgevoerde run met rewards."""
    lines: List[str] = []
    lines.append(f"Run:       {run.run_id}")
    lines.append(f"Workflow:  {run.dag_name} ({run.dag_id})")
    lines.append(f"Timestamp: {run.created_at}")
    lines.append("")

    if not run.node_results:
        lines.append("(no nodes executed)")
        lines.append("")
        return "\n".join(lines)

    # Header
    lines.append(f"{'':>4} {'Node':<20} {'Action':<18} {'Status':<12} "
                 f"{'Dist':<6} {'Credit':<10} {'Episode':<12}")
    lines.append(f"{'':>4} {'─'*19} {'─'*17} {'─'*11} {'─'*5} {'─'*9} {'─'*11}")

    for nr in run.node_results:
        status_char = "✓" if nr.execution_status == "completed" else "✗"
        credit_str = f"{nr.credit_reward:+.4f}" if nr.credit_reward is not None else "—"
        ep_str = nr.episode_id[:10] if nr.episode_id else "—"
        lines.append(
            f"  {status_char} {nr.node_name:<19} {nr.action_name or '—':<17} "
            f"{nr.execution_status:<11} {nr.distance_from_end:<5} "
            f"{credit_str:<9} {ep_str:<11}"
        )

    lines.append("")
    if run.global_reward is not None:
        bar_len = 30
        reward = run.global_reward
        # Clamp to [-1, 1] for bar display
        clamped = max(-1.0, min(1.0, reward))
        fill = int(clamped * bar_len) if clamped >= 0 else 0
        empty = bar_len - fill
        lines.append(
            f"GLOBAL REWARD:  {reward:+.4f}  "
            f"|{'█' * fill}{'░' * empty}|  "
            f"{run.global_reward_reason or ''}"
        )
    else:
        lines.append("GLOBAL REWARD:  (not set)")

    if run.context:
        ctx_str = ", ".join(
            f"{k}={v}" for k, v in run.context.items()
            if not k.startswith("_")
        )
        if ctx_str:
            lines.append(f"Context:       {ctx_str[:100]}")

    return "\n".join(lines)


def mermaid_run(dag: WorkflowDAG, run: Optional[WorkflowRun] = None) -> str:
    """Mermaid diagram met optionele run-status kleuren.

    Kleurt nodes op basis van execution_status:
    - completed: groen
    - failed: rood
    - running/skip: grijs
    """
    lines = ["```mermaid", "flowchart LR"]

    # Stijlen
    lines.append("    %% Styles")
    lines.append("    classDef completed fill:#8f8,stroke:#393,stroke-width:2px")
    lines.append("    classDef failed fill:#f88,stroke:#c33,stroke-width:2px")
    lines.append("    classDef pending fill:#eee,stroke:#999,stroke-width:1px")

    status_map: dict = {}
    if run:
        for nr in run.node_results:
            status_map[nr.node_id] = nr.execution_status

    for nid, node in dag.nodes.items():
        label = node.name.replace('"', "'")
        actions = ", ".join(node.actions_short.values()) if node.actions_short else ""
        if actions:
            label += f"<br><small>{actions}</small>"
        lines.append(f'    {nid}["{label}"]')

    lines.append("")
    for edge in dag.edges:
        lines.append(f"    {edge.source} --> {edge.target}")

    if status_map:
        completed = [nid for nid, s in status_map.items() if s == "completed"]
        failed = [nid for nid, s in status_map.items() if s in ("failed", "error")]
        pending = [nid for nid in dag.nodes if nid not in status_map]

        if completed:
            lines.append(f"    class {' '.join(completed)} completed")
        if failed:
            lines.append(f"    class {' '.join(failed)} failed")
        if pending:
            lines.append(f"    class {' '.join(pending)} pending")

    lines.append("```")
    return "\n".join(lines)


__all__ = ["ascii_graph", "run_report", "mermaid_run"]
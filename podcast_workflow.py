"""
podcast_workflow.py — Concrete podcast productie DAG.

Defines the 6-step podcast workflow:

    topic_select → research → script → audio → video → publish

Each step is a WorkflowNode with its own task_id so it learns independently
via the agent-learning TaskPolicy infrastructure.

CLI usage::

    # Toon de DAG (visualisatie)
    python3 podcast_workflow.py --show

    # Droge run (gesimuleerd, geen store)
    python3 podcast_workflow.py --dry-run --topic "AI Agents in 2026"

    # Echte run (met store + episodes)
    python3 podcast_workflow.py --run --topic "AI Agents in 2026"

    # Toon credit assignment
    python3 podcast_workflow.py --simulate-rewards
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from workflow_dag import WorkflowDAG, WorkflowNode, WorkflowRun
from workflow_runner import WorkflowRunner, DecisionFn, ExecuteFn, RewardFn
from workflow_visualize import ascii_graph, mermaid_run, run_report


# ---------------------------------------------------------------------------
# Podcast nodes configuration
# ---------------------------------------------------------------------------

PODCAST_NODES = [
    WorkflowNode(
        id="topic_select",
        name="Onderwerp Kiezen",
        description="Kies een podcast onderwerp uit trending AI nieuws of voorraad",
        task_id="podcast-topic-select",
        action_ids=["llm_suggest", "from_queue", "manual"],
        actions_short={
            "llm_suggest": "LLM-suggestie",
            "from_queue": "Uit wachtrij",
            "manual": "Handmatig",
        },
        metadata={
            "correct_action_id": "llm_suggest",
            "context_features": {
                "queue_size": 0.0,
                "trending_score": 0.5,
                "has_previous": 0.0,
            },
        },
    ),
    WorkflowNode(
        id="research",
        name="Research Doen",
        description="Verzamel bronnen en feiten over het onderwerp",
        task_id="podcast-research",
        action_ids=["webclaw_deep", "webclaw_quick", "llm_only", "manual"],
        actions_short={
            "webclaw_deep": "WebClaw diep",
            "webclaw_quick": "WebClaw snel",
            "llm_only": "LLM-only",
            "manual": "Handmatig",
        },
        metadata={
            "correct_action_id": "webclaw_deep",
            "context_features": {
                "topic_novelty": 0.5,
                "required_depth": 0.7,
                "time_budget_hrs": 2.0,
            },
        },
    ),
    WorkflowNode(
        id="script",
        name="Script Schrijven",
        description="Schrijf een podcast script op basis van research",
        task_id="podcast-script",
        action_ids=["llm_full", "llm_outline", "template", "manual"],
        actions_short={
            "llm_full": "LLM volledig",
            "llm_outline": "LLM outline",
            "template": "Template",
            "manual": "Handmatig",
        },
        metadata={
            "correct_action_id": "llm_full",
            "context_features": {
                "research_quality": 0.5,
                "script_length_min": 10.0,
                "requires_review": 0.0,
            },
        },
    ),
    WorkflowNode(
        id="audio",
        name="Audio Genereren",
        description="Genereer audiobestand uit script (TTS of opname)",
        task_id="podcast-audio",
        action_ids=["tts_edge", "tts_elevenlabs", "record", "skip"],
        actions_short={
            "tts_edge": "Edge TTS",
            "tts_elevenlabs": "ElevenLabs TTS",
            "record": "Opnemen",
            "skip": "Overslaan",
        },
        metadata={
            "correct_action_id": "tts_edge",
            "context_features": {
                "script_quality": 0.5,
                "audio_length_min": 10.0,
                "has_host": 0.0,
            },
        },
    ),
    WorkflowNode(
        id="video",
        name="Video Maken",
        description="Maak video van audio + visuals",
        task_id="podcast-video",
        action_ids=["explainer_mock", "explainer_real", "static", "skip"],
        actions_short={
            "explainer_mock": "Explainer mock",
            "explainer_real": "Explainer real",
            "static": "Stilstaand beeld",
            "skip": "Overslaan",
        },
        metadata={
            "correct_action_id": "explainer_mock",
            "context_features": {
                "audio_quality": 0.5,
                "visual_complexity": 0.3,
                "has_assets": 0.5,
            },
        },
    ),
    WorkflowNode(
        id="publish",
        name="Publiceren",
        description="Upload naar podcast platformen",
        task_id="podcast-publish",
        action_ids=["wordpress", "youtube", "spotify", "all", "draft"],
        actions_short={
            "wordpress": "WordPress",
            "youtube": "YouTube",
            "spotify": "Spotify",
            "all": "Alle platformen",
            "draft": "Concept houden",
        },
        metadata={
            "correct_action_id": "all",
            "context_features": {
                "has_video": 0.5,
                "has_audio": 0.5,
                "is_approved": 1.0,
            },
        },
    ),
]

# Edges
PODCAST_EDGES = [
    ("topic_select", "research"),
    ("research", "script"),
    ("script", "audio"),
    ("audio", "video"),
    ("video", "publish"),
]


# ---------------------------------------------------------------------------
# Build DAG
# ---------------------------------------------------------------------------


def build_podcast_dag(discount_factor: float = 0.8) -> WorkflowDAG:
    """Bouw de podcast productie DAG."""
    dag = WorkflowDAG(
        id="podcast",
        name="Podcast Productie",
        nodes=PODCAST_NODES,
        discount_factor=discount_factor,
    )
    for src, tgt in PODCAST_EDGES:
        dag.add(src, tgt)
    return dag


# ---------------------------------------------------------------------------
# Custom callbacks voor podcast
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Decision factory — gebruikt TaskPolicy wanneer beschikbaar
# ---------------------------------------------------------------------------


def make_podcast_decision(store=None):
    """Factory: returns a decision function that uses a TaskPolicy when available.

    When no policy exists yet, initializes a ThompsonPolicy from the node's
    action_ids so the first run already explores via Thompson sampling instead
    of always picking action_ids[0].

    Args:
        store: Optional agent-learning store. When set, queries the store
            for a trained TaskPolicy per node and uses policy.decide().
            Falls back to first action when no policy exists.
    """
    # Cache for auto-created policies per task_id
    _auto_policies: Dict[str, ThompsonPolicy] = {}

    def _get_or_create_policy(node: WorkflowNode) -> tuple:
        """Get existing policy or create a fresh ThompsonPolicy."""
        if store is None or not node.task_id:
            return None, None, "no_store"

        try:
            snapshot = store.get_active_policy("hermes", node.task_id)
            if snapshot is not None:
                # Check if it's a Thompson policy
                kind = snapshot.metadata.get("policy_kind", "")
                if kind == "thompson" or "thompson_alpha" in snapshot.metadata:
                    from thompson_policy import ThompsonPolicy
                    policy = ThompsonPolicy.from_snapshot(snapshot)
                    return policy, snapshot.version, "thompson_policy"
                else:
                    # Try SoftmaxPolicy for backward compatibility
                    from agent_learning.policy.softmax_bandit import SoftmaxPolicy
                    policy = SoftmaxPolicy.from_snapshot(snapshot)
                    return policy, snapshot.version, "softmax_policy"
        except Exception:
            pass

        # No policy exists — create one from node's action_ids
        if node.action_ids and node.task_id:
            # Check if we already created one this session
            if node.task_id in _auto_policies:
                p = _auto_policies[node.task_id]
                return p, p._snapshot.version, "auto_thompson_cached"

            try:
                from thompson_policy import ThompsonPolicy
                from agent_learning.types import Action
                actions = [Action(id=aid) for aid in node.action_ids]
                policy = ThompsonPolicy.from_actions(
                    actions, agent_id="hermes", task_id=node.task_id,
                    prior_alpha=1.0, prior_beta=1.0,  # uniform prior
                )
                # Store it so subsequent runs find it
                if store is not None:
                    try:
                        store.store_policy(policy._snapshot)
                    except Exception:
                        pass
                _auto_policies[node.task_id] = policy
                return policy, 0, "auto_thompson"
            except Exception:
                pass

        return None, None, "fallback"

    def _decide(node: WorkflowNode, context: Dict[str, Any]) -> Tuple[str, Optional[float], Dict[str, Any]]:
        """Echte decision: probeer TaskPolicy, dan context override, dan fallback."""
        override = context.get(f"override_action:{node.id}")
        if override:
            return override, None, {"source": "context_override"}

        correct_path = context.get("correct_path", {})
        if node.id in correct_path:
            return correct_path[node.id], None, {"source": "correct_path"}

        # Probeer policy uit store of auto-create
        policy, version, source = _get_or_create_policy(node)
        if policy is not None:
            try:
                decision = policy.choose()
                chosen = decision.action.id
                # Apply constraints: filter out disallowed actions
                from constraints import apply_podcast_constraints
                allowed = apply_podcast_constraints(node.id, node.action_ids, context)
                if chosen not in allowed and allowed:
                    # Policy koos een geblokkeerde actie — gebruik eerste toegestane
                    chosen = allowed[0]
                    logprob = None
                    source = f"{source}_constrained"
                else:
                    logprob = decision.logprob
                return chosen, logprob, {
                    "source": source,
                    "policy_version": version,
                    "allowed_actions": allowed,
                }
            except Exception:
                pass

        # Fallback: eerste action
        if node.action_ids:
            return node.action_ids[0], None, {"source": "fallback"}
        return "skip", None, {"source": "no_actions"}
    return _decide


def podcast_execute(node: WorkflowNode, action_id: str, context: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Simuleer uitvoering van een podcast stap.

    In de echte workflow roept dit de daadwerkelijke tools aan
    (WebClaw, TTS, Explainer, WP REST API, etc.).
    """
    topic = context.get("topic", "onbekend onderwerp")
    started = time.time()

    # Simuleer latency per node type
    sim_latency = {
        "topic_select": 0.3,
        "research": 2.0,
        "script": 1.5,
        "audio": 3.0,
        "video": 5.0,
        "publish": 1.0,
    }
    delay = sim_latency.get(node.id, 0.5)

    # Check voor forced failures (voor reward testing)
    fail_node = context.get("force_fail")
    if fail_node == node.id:
        time.sleep(delay * 0.3)
        return "failed", f"Geforceerde fout in {node.id}", {"forced": True}

    # Check voor partial failures
    fail_rate = context.get("fail_rate", 0.0)
    if fail_rate > 0 and hash(f"{context.get('run_seed','')}:{node.id}") % 100 < fail_rate * 100:
        time.sleep(delay * 0.5)
        return "failed", f"Willekeurige fout in {node.id}", {"random_failure": True}

    time.sleep(min(delay, 1.0))  # cap simulated delay for testing

    # Node-specific results
    results = {
        "topic_select": f"Onderwerp gekozen: {topic}",
        "research": f"Research voltooid: 3 bronnen gevonden voor {topic}",
        "script": f"Script geschreven: 5 minuten over {topic}",
        "audio": f"Audio gegenereerd: 5:12 minuten via {action_id}",
        "video": f"Video gecreëerd: 1920x1080, 5:12 minuten",
        "publish": f"Gepubliceerd op WordPress + YouTube",
    }
    summary = results.get(node.id, f"{node.id} voltooid via {action_id}")

    context[f"_result:{node.id}"] = {
        "action": action_id,
        "summary": summary,
        "latency_s": round(time.time() - started, 2),
    }

    return "completed", summary, {"simulated": True, "latency_s": round(time.time() - started, 2)}


def podcast_reward(context: Dict[str, Any], node_results: List) -> Tuple[float, str]:
    """Bepaal globale reward voor een podcast run.
    Domein-specifiek voor de podcast simulatie (niet CostAwareReward).

    Factoren (verschillend van CostAwareReward — deze gebruikt
    critical-path bonus i.p.v. latency/cost penalties):
    - Basis: succeeded / total nodes
    - Critical path bonus: topic + research + script + publish (0–0.3)
    - Publish bonus: +0.2 als publish OK, -0.3 als niet
    - Fail penalty: -0.2 per failed node (max -0.5)
    """
    total = len(node_results)
    if total == 0:
        return 0.0, "geen nodes uitgevoerd"

    succeeded = [nr for nr in node_results if nr.execution_status == "completed"]
    failed = [nr for nr in node_results if nr.execution_status != "completed"]
    n_ok = len(succeeded)
    n_fail = len(failed)

    # Basis: ratio succeeded
    base = n_ok / total

    # Bonus voor critical path
    critical = {"topic_select", "research", "script", "publish"}
    critical_ok = sum(
        1 for nr in succeeded if nr.node_id in critical
    )
    critical_bonus = critical_ok / len(critical) * 0.3

    # Publish is de belangrijkste — extra bonus
    publish_ok = any(
        nr.node_id == "publish" and nr.execution_status == "completed"
        for nr in succeeded
    )
    publish_bonus = 0.2 if publish_ok else -0.3

    # Straffactor voor falen
    fail_penalty = min(n_fail * 0.2, 0.5)

    reward = base + critical_bonus + publish_bonus - fail_penalty
    reward = max(-1.0, min(1.0, reward))

    reasons = []
    reasons.append(f"{n_ok}/{total} succeeded")
    if critical_bonus > 0:
        reasons.append(f"critical path {critical_ok}/{len(critical)}")
    if publish_ok:
        reasons.append("publish OK +0.2")
    else:
        reasons.append("geen publish -0.3")
    if fail_penalty > 0:
        reasons.append(f"fail penalty -{fail_penalty:.1f}")

    return round(reward, 4), "; ".join(reasons)


# ---------------------------------------------------------------------------
# Simulatie helpers
# ---------------------------------------------------------------------------


def simulate_credit_assignment(discount: float = 0.8) -> str:
    """Toon hoe credit assignment uitpakt voor de podcast DAG."""
    dag = build_podcast_dag(discount)
    order = dag.topological_sort()

    lines = [
        f"Podcast DAG credit assignment (γ = {discount})",
        "=" * 50,
        "",
        f"{'Node':<20} {'Dist':<6} {'γ^dist':<10} {'Als global=1.0':<16} {'Als global=0.5':<16}",
        f"{'─'*19} {'─'*5} {'─'*9} {'─'*15} {'─'*15}",
    ]

    for nid in order:
        node = dag.nodes[nid]
        dist = dag.distance_from_end(nid)
        gamma_pow = discount ** dist
        credit_1 = 1.0 * gamma_pow
        credit_05 = 0.5 * gamma_pow
        lines.append(
            f"{node.name:<19} {dist:<5} {gamma_pow:<9.4f} "
            f"{credit_1:<15.4f} {credit_05:<15.4f}"
        )

    lines.extend([
        "",
        "Formule: credit = global_reward × γ^distance_from_end",
        "  distance = aantal edges naar de verste leaf node",
        f"  publish heeft distance=0 (laatste node) → krijgt volledige reward",
        f"  topic_select heeft distance=5 → krijgt γ^5 = {discount**5:.4f} van de reward",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Podcast Workflow — DAG definitie, simulatie, en uitvoering"
    )
    parser.add_argument("--show", action="store_true",
                        help="Toon DAG visualisatie (ASCII + Mermaid)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Droge run (gesimuleerd, geen store)")
    parser.add_argument("--run", action="store_true",
                        help="Echte run (met store)")
    parser.add_argument("--topic", default="AI Agents in 2026",
                        help="Podcast onderwerp")
    parser.add_argument("--discount", type=float, default=0.8,
                        help="Discount factor voor credit assignment")
    parser.add_argument("--force-fail", default=None,
                        help="Forceer een failure in deze node (test)")
    parser.add_argument("--fail-rate", type=float, default=0.0,
                        help="Kans op willekeurige failure per node (0-1)")
    parser.add_argument("--correct-path", nargs="*", default=[],
                        help="Correcte acties per node, format: node_id=action_id")
    parser.add_argument("--simulate-rewards", action="store_true",
                        help="Toon credit assignment tabel")
    parser.add_argument("--json", action="store_true",
                        help="Output als JSON")

    args = parser.parse_args()

    # ---- Show DAG ----------------------------------------------------------
    if args.show:
        dag = build_podcast_dag(args.discount)
        print(ascii_graph(dag))
        print()
        print(mermaid_run(dag))

    # ---- Simulate credit assignment ----------------------------------------
    if args.simulate_rewards:
        print(simulate_credit_assignment(args.discount))

    # ---- Dry run / Run -----------------------------------------------------
    if args.dry_run or args.run:
        dag = build_podcast_dag(args.discount)

        # Parse correct path
        correct_path = {}
        for entry in args.correct_path:
            if "=" in entry:
                k, v = entry.split("=", 1)
                correct_path[k] = v

        store = None
        if args.run:
            try:
                from agent_learning.storage.cosmos import get_default_store
                store = get_default_store()
                print(f"Store: {type(store).__name__}")
            except Exception as exc:
                print(f"Store niet beschikbaar: {exc}")
                print("Val terug naar droge run zonder store.")

        context: Dict[str, Any] = {
            "topic": args.topic,
            "correct_path": correct_path,
            "force_fail": args.force_fail,
            "fail_rate": args.fail_rate,
            "run_seed": str(time.time()),
        }

        runner = WorkflowRunner(
            dag,
            store=store,
            agent_id="hermes",
            decision_fn=make_podcast_decision(store),
            execute_fn=podcast_execute,
            reward_fn=podcast_reward,
            discount_factor=args.discount,
        )

        print(f"\n{'='*60}")
        print(f"Uitvoeren: {dag.name} — onderwerp: {args.topic}")
        print(f"Discount:  γ = {args.discount}")
        if correct_path:
            print(f"Correct path: {correct_path}")
        if args.force_fail:
            print(f"Force fail: {args.force_fail}")
        print()

        run = runner.execute(context)

        print(runner.summary(run))
        print()
        print(run_report(run))

        if args.json:
            print()
            print(json.dumps(run.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
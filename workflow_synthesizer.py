"""
workflow_synthesizer.py — Dynamische workflow synthese uit een taakomschrijving.

In plaats van workflows hard te coderen (zoals podcast_workflow.py), maakt
deze module van een natuurlijke-taakomschrijving een complete WorkflowDAG
met nodes, constraints, policies en rewards — en registreert die meteen
in de DagRegistry voor uitvoering en optimalisatie.

Usage:
    synthesizer = WorkflowSynthesizer(store)
    dag = synthesizer.synthesize(
        task="Scrape website X en extraheer contactgegevens",
        context={"url": "https://example.com"}
    )
    # dag is klaar voor WorkflowRunner.execute()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import env_setup  # noqa: F401 — ensures LocalFileStore backend

from workflow_dag import WorkflowDAG, WorkflowNode, WorkflowRun
from workflow_runner import WorkflowRunner, LocalRewardFn

logger = logging.getLogger("workflow_synthesizer")

# ---------------------------------------------------------------------------
# Standaard actie-sets per node-type
# ---------------------------------------------------------------------------

# Een node-type definieert welke acties mogelijk zijn, constraints, en
# een default lokale reward. Synthesizer kiest obv taakomschrijving.

@dataclass
class NodeType:
    """Template voor een workflow node."""
    id: str
    name: str
    description: str
    action_ids: List[str]
    actions_short: Dict[str, str]
    default_correct_action: str
    context_features: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_node(self, task_suffix: str = "") -> WorkflowNode:
        task_id = f"{self.id}{task_suffix}"
        return WorkflowNode(
            id=self.id,
            name=self.name,
            description=self.description,
            task_id=task_id,
            action_ids=list(self.action_ids),
            actions_short=dict(self.actions_short),
            metadata={
                "correct_action_id": self.default_correct_action,
                "context_features": dict(self.context_features),
                "node_type": self.id,
            },
        )


# Registry van bekende node-types. Uitbreidbaar.
NODE_TYPES: Dict[str, NodeType] = {
    "research": NodeType(
        id="research",
        name="Research Uitvoeren",
        description="Verzamel informatie over het onderwerp",
        action_ids=["web_search", "web_scrape", "llm_only", "manual"],
        actions_short={
            "web_search": "Web search",
            "web_scrape": "Web scrape",
            "llm_only": "LLM-only",
            "manual": "Handmatig",
        },
        default_correct_action="web_scrape",
        context_features={"topic_novelty": 0.5, "required_depth": 0.7},
        tags=["research", "information"],
    ),
    "extract": NodeType(
        id="extract",
        name="Data Extraheren",
        description="Extraheer gestructureerde data uit bronnen",
        action_ids=["llm_extract", "regex", "manual"],
        actions_short={
            "llm_extract": "LLM extractie",
            "regex": "Regex patroon",
            "manual": "Handmatig",
        },
        default_correct_action="llm_extract",
        context_features={"data_complexity": 0.5, "has_schema": 0.0},
        tags=["extract", "data"],
    ),
    "transform": NodeType(
        id="transform",
        name="Data Transformeren",
        description="Transformeer data naar gewenst formaat",
        action_ids=["llm_transform", "template", "script", "manual"],
        actions_short={
            "llm_transform": "LLM transform",
            "template": "Template",
            "script": "Script",
            "manual": "Handmatig",
        },
        default_correct_action="llm_transform",
        context_features={"complexity": 0.5, "volume": 0.3},
        tags=["transform", "data"],
    ),
    "create": NodeType(
        id="create",
        name="Content Creëren",
        description="Genereer content op basis van research",
        action_ids=["llm_full", "llm_outline", "template", "manual"],
        actions_short={
            "llm_full": "LLM volledig",
            "llm_outline": "LLM outline",
            "template": "Template",
            "manual": "Handmatig",
        },
        default_correct_action="llm_full",
        context_features={"quality": 0.5, "length": 0.5},
        tags=["create", "content"],
    ),
    "publish": NodeType(
        id="publish",
        name="Publiceren",
        description="Publiceer het resultaat",
        action_ids=["wordpress", "youtube", "spotify", "draft"],
        actions_short={
            "wordpress": "WordPress",
            "youtube": "YouTube",
            "spotify": "Spotify",
            "draft": "Concept",
        },
        default_correct_action="wordpress",
        context_features={"has_media": 0.5, "is_approved": 0.0},
        tags=["publish", "output"],
    ),
    "monitor": NodeType(
        id="monitor",
        name="Monitoren",
        description="Bewaak een bron voor veranderingen",
        action_ids=["poll_api", "rss_feed", "web_check", "manual"],
        actions_short={
            "poll_api": "API pollen",
            "rss_feed": "RSS feed",
            "web_check": "Web check",
            "manual": "Handmatig",
        },
        default_correct_action="poll_api",
        context_features={"frequency": 0.5, "volume": 0.3},
        tags=["monitor", "watch"],
    ),
    "validate": NodeType(
        id="validate",
        name="Valideren",
        description="Controleer kwaliteit en correctheid",
        action_ids=["llm_check", "schema_check", "manual_review", "auto_pass"],
        actions_short={
            "llm_check": "LLM check",
            "schema_check": "Schema check",
            "manual_review": "Handmatige review",
            "auto_pass": "Automatisch goedkeuren",
        },
        default_correct_action="llm_check",
        context_features={"criticality": 0.5, "has_schema": 0.0},
        tags=["validate", "quality"],
    ),
    "notify": NodeType(
        id="notify",
        name="Notificeren",
        description="Stuur een notificatie over het resultaat",
        action_ids=["telegram", "email", "slack", "log_only"],
        actions_short={
            "telegram": "Telegram",
            "email": "E-mail",
            "slack": "Slack",
            "log_only": "Alleen loggen",
        },
        default_correct_action="telegram",
        context_features={"urgency": 0.3, "audience": 0.5},
        tags=["notify", "output"],
    ),
}


# ---------------------------------------------------------------------------
# Workflow templates — veelgebruikte DAG-patronen
# ---------------------------------------------------------------------------

# Template: (naam, [lijst van node-type ids in volgorde], [edges])
WORKFLOW_TEMPLATES: Dict[str, Tuple[str, List[str], List[Tuple[str, str]]]] = {
    "research_publish": (
        "Research → Creëer → Publiceer",
        ["research", "create", "publish"],
        [("research", "create"), ("create", "publish")],
    ),
    "scrape_extract_store": (
        "Scrape → Extraheer → Opslaan",
        ["research", "extract", "publish"],
        [("research", "extract"), ("extract", "publish")],
    ),
    "monitor_validate_notify": (
        "Monitor → Valideer → Notificeer",
        ["monitor", "validate", "notify"],
        [("monitor", "validate"), ("validate", "notify")],
    ),
    "research_create_validate_publish": (
        "Research → Creëer → Valideer → Publiceer",
        ["research", "create", "validate", "publish"],
        [("research", "create"), ("create", "validate"), ("validate", "publish")],
    ),
}


# ---------------------------------------------------------------------------
# Lokale reward-fabriek per node-type
# ---------------------------------------------------------------------------


def make_local_reward(node_type: str) -> LocalRewardFn:
    """Maak een lokale reward functie voor een node-type.

    De reward checkt of de output zinvol is, niet alleen of het 'completed' is.
    Dit is de eerste stap richting LLM-as-a-Judge voor quality scores.
    """
    def _reward(
        node: WorkflowNode,
        action_id: str,
        status: str,
        summary: str,
        exec_meta: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, str]:
        if status != "completed":
            return -0.5, "failed"

        base = 0.5
        reasons = ["completed"]

        # Content check: heeft de node iets opgeleverd?
        if summary and len(summary) > 20:
            base += 0.2
            reasons.append("content")
        else:
            base -= 0.2
            reasons.append("empty")

        # Type-specifieke checks
        if node_type == "research":
            # Research moet minstens een URL of bron hebben opgeleverd
            if any(marker in (summary or "") for marker in ["http", "bron", "found", "gevonden"]):
                base += 0.2
                reasons.append("sources")
        elif node_type == "create":
            # Content moet een bepaalde lengte hebben
            if summary and len(summary) > 100:
                base += 0.1
                reasons.append("length")
        elif node_type == "validate":
            # Validatie moet een oordeel hebben
            if any(marker in (summary or "") for marker in ["ok", "passed", "goedgekeurd", "valid"]):
                base += 0.2
                reasons.append("approved")
            elif any(marker in (summary or "") for marker in ["fail", "error", "afgekeurd"]):
                base -= 0.3
                reasons.append("rejected")
        elif node_type == "publish":
            # Publicatie moet gelukt zijn (geen draft als het live moest)
            if "draft" in action_id and not context.get("_is_approved", False):
                base -= 0.1
                reasons.append("draft_ok")
            elif "draft" in action_id:
                base -= 0.3
                reasons.append("draft_not_live")

        return max(-1.0, min(1.0, base)), "+".join(reasons)

    return _reward


# ---------------------------------------------------------------------------
# Constraint-fabriek per node-type
# ---------------------------------------------------------------------------


def make_constraints(node_type: str) -> Callable[[str, List[str], Dict], List[str]]:
    """Maak een constraint functie voor een node-type.

    Gebruikt de constraints uit constraints.py waar mogelijk, met
    extra type-specifieke regels.
    """
    if node_type == "publish":
        from constraints import apply_podcast_constraints
        return apply_podcast_constraints
    elif node_type == "research":
        from constraints import apply_scrape_constraints
        def _research_constraints(node_id: str, actions: List[str], ctx: Dict) -> List[str]:
            # Als er context_features zijn, pas scrape constraints toe
            cf = ctx.get("_context_features", {})
            if cf:
                return apply_scrape_constraints(cf, actions)
            return list(actions)
        return _research_constraints
    else:
        # Default: geen extra constraints
        def _noop(node_id: str, actions: List[str], ctx: Dict) -> List[str]:
            return list(actions)
        return _noop


# ---------------------------------------------------------------------------
# WorkflowSynthesizer
# ---------------------------------------------------------------------------


class WorkflowSynthesizer:
    """Synthetiseer een WorkflowDAG uit een taakomschrijving.

    Gebruikt een combinatie van templates en node-types om een werkende
    DAG te bouwen, complete met constraints, policies, en rewards.

    Usage:
        syn = WorkflowSynthesizer(store)
        dag = syn.synthesize(
            task="Research AI trends en publiceer een blog",
            context={"topic": "AI Agents"}
        )
        runner = WorkflowRunner(dag, store=store,
            decision_fn=syn.make_decision(dag),
            execute_fn=syn.make_execute(dag),
            reward_fn=syn.make_reward(),
            local_reward_fn=syn.make_local_reward(dag),
        )
        run = runner.execute(context={"topic": "AI Agents"})
    """

    def __init__(self, store=None, agent_id: str = "hermes"):
        self.store = store
        self.agent_id = agent_id
        self._last_dag: Optional[WorkflowDAG] = None

    # ------------------------------------------------------------------
    # Template suggestie — ontworpen voor LLM-gebruik (Hermes roept dit)
    # ------------------------------------------------------------------

    def suggest_template(self, task: str) -> Dict[str, Any]:
        """Analyseer een taak en geef template-suggestie.

        Dit is de intelligente variant van _resolve_template. Ontworpen
        om door Hermes (LLM) gebruikt te worden: Hermes analyseert de
        taak met eigen reasoning, roept deze methode met expliciete
        parameters.

        Returns dict met:
            template: template naam of None
            node_types: suggestie voor node types (kan afwijken van template)
            context: suggestie voor context variabelen
            reasoning: korte uitleg waarom deze suggestie
        """
        task_lower = task.lower()

        # Bouw een structured analysis
        analysis = {
            "task": task[:200],
            "has_research": any(w in task_lower for w in
                ["onderzoek", "research", "zoek", "analyse", "verzamel", "informatie",
                 "research", "search", "find", "gather", "analyze"]),
            "has_create": any(w in task_lower for w in
                ["schrijf", "maak", "genereer", "creëer", "blog", "artikel", "post",
                 "write", "create", "generate", "make", "blog", "article"]),
            "has_publish": any(w in task_lower for w in
                ["publiceer", "deel", "upload", "stuur", "blog", "website",
                 "publish", "share", "upload", "send", "post"]),
            "has_extract": any(w in task_lower for w in
                ["extraheer", "haal", "data", "scrape", "contact",
                 "extract", "get", "scrape", "data"]),
            "has_monitor": any(w in task_lower for w in
                ["monitor", "bewaak", "volg", "check", "watch",
                 "monitor", "watch", "track", "check"]),
            "has_validate": any(w in task_lower for w in
                ["valideer", "check", "controleer", "review", "keur",
                 "validate", "check", "verify", "review"]),
            "has_notify": any(w in task_lower for w in
                ["notificeer", "waarschuw", "mail", "bericht", "stuur",
                 "notify", "alert", "send", "message", "email"]),
        }

        # Bepaal template obv analysis
        template = None
        reasoning = []

        if analysis["has_research"] and analysis["has_create"] and analysis["has_publish"] and analysis["has_validate"]:
            template = "research_create_validate_publish"
            reasoning.append("research + create + validate + publish")
        elif analysis["has_research"] and analysis["has_publish"]:
            template = "research_publish"
            reasoning.append("research + publish")
        elif analysis["has_extract"]:
            template = "scrape_extract_store"
            reasoning.append("scrape + extract + store")
        elif analysis["has_monitor"]:
            template = "monitor_validate_notify"
            reasoning.append("monitor + notify")
        else:
            template = "research_publish"
            reasoning.append("default: research + publish")

        # Context suggesties
        context = {}
        if analysis["has_extract"]:
            context["data_type"] = "structured"
        if analysis["has_monitor"]:
            context["frequency"] = "daily"

        return {
            "template": template,
            "reasoning": "; ".join(reasoning),
            "context_suggestions": context,
            "analysis": analysis,
        }

    # ------------------------------------------------------------------
    # Main synthesis method
    # ------------------------------------------------------------------

    def synthesize(
        self,
        task: str,
        template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        discount_factor: float = 0.8,
    ) -> WorkflowDAG:
        """Synthetiseer een workflow DAG uit een taakomschrijving.

        Args:
            task: Natuurlijke taal omschrijving van de taak.
            template: Optionele template naam (anders auto-detect).
            context: Optionele context voor de nodes.
            discount_factor: Gamma voor credit assignment.

        Returns:
            Een geregistreerde WorkflowDAG klaar voor uitvoering.
        """
        ctx = context or {}

        # ---- Stap 1: Bepaal template en node-volgorde ------------------
        template_name, node_order, edges = self._resolve_template(task, template)

        # ---- Stap 2: Bouw nodes -----------------------------------------
        task_suffix = f"-{uuid.uuid4().hex[:6]}"
        nodes: List[WorkflowNode] = []
        for i, node_type_id in enumerate(node_order):
            nt = NODE_TYPES.get(node_type_id)
            if nt is None:
                logger.warning("Unknown node type %s, skipping", node_type_id)
                continue
            node = nt.to_node(task_suffix)
            # Voeg context-features uit taakcontext toe
            for key, val in ctx.items():
                if key in nt.context_features:
                    node.metadata.setdefault("context_features", {})[key] = val
            nodes.append(node)

        # ---- Stap 3: Bouw DAG -------------------------------------------
        dag_id = f"{template_name}_{uuid.uuid4().hex[:6]}"
        dag = WorkflowDAG(
            id=dag_id,
            name=task[:60] if len(task) > 60 else task,
            nodes=nodes,
            discount_factor=discount_factor,
        )
        for src, tgt in edges:
            dag.add(src, tgt)

        # ---- Stap 4: Initialiseer Thompson policies ---------------------
        self._init_policies(dag)

        # ---- Stap 5: Registreer in DagRegistry --------------------------
        self._register_dag(dag, template_name, ctx)

        self._last_dag = dag
        logger.info("Synthesized DAG %s: %d nodes, template=%s",
                     dag.id, len(nodes), template_name)
        return dag

    # ------------------------------------------------------------------
    # Run — synthesize + execute in one call
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        discount_factor: float = 0.8,
        execute_fn: Optional[Callable] = None,
    ) -> WorkflowRun:
        """Synthetiseer en voer direct uit.

        Dit is de high-level API voor Hermes: één call synthetiseert
        de DAG, initialiseert policies, en voert uit.

        Args:
            task: Taakomschrijving.
            template: Optionele template (anders auto-detect).
            context: Uitvoeringscontext (topic, flags, etc.).
            discount_factor: Gamma voor credit assignment.
            execute_fn: Optionele execute callback (default: simulated).

        Returns:
            WorkflowRun met resultaten.
        """
        ctx = context or {}

        # Stap 1: Synthetiseer DAG
        dag = self.synthesize(
            task=task,
            template=template,
            context=ctx,
            discount_factor=discount_factor,
        )

        # Stap 2: Bouw callbacks
        decision_fn = self.make_decision(dag)
        exe_fn = execute_fn or self.make_execute(dag)
        reward_fn = self.make_reward()
        local_reward_fn = self.make_local_reward(dag)

        # Stap 3: Voer uit
        runner = WorkflowRunner(
            dag,
            store=self.store,
            agent_id=self.agent_id,
            decision_fn=decision_fn,
            execute_fn=exe_fn,
            reward_fn=reward_fn,
            local_reward_fn=local_reward_fn,
            discount_factor=discount_factor,
        )
        run = runner.execute(context=ctx)

        logger.info("Run completed: global_reward=%.4f, %d nodes",
                     run.global_reward or 0, len(run.node_results))
        return run

    # ------------------------------------------------------------------
    # Template resolution
    # ------------------------------------------------------------------

    def _resolve_template(
        self,
        task: str,
        template: Optional[str] = None,
    ) -> Tuple[str, List[str], List[Tuple[str, str]]]:
        """Bepaal welke template bij de taak hoort.

        Gebruikt suggest_template() voor analyse — ook de keyword-gebaseerde
        variant, want Hermes gebruikt deze methode met expliciete template
        parameter. De CLI fallback gebruikt suggest_template() voor betere
        auto-detect.
        """
        # Expliciete template override (Hermes roept dit met template=...)
        if template and template in WORKFLOW_TEMPLATES:
            name, nodes, edges = WORKFLOW_TEMPLATES[template]
            return template, nodes, edges

        # Auto-detect via suggest_template
        suggestion = self.suggest_template(task)
        t = suggestion["template"]
        logger.info("Template suggestie: %s (%s)", t, suggestion["reasoning"])

        if t in WORKFLOW_TEMPLATES:
            name, nodes, edges = WORKFLOW_TEMPLATES[t]
            return t, nodes, edges

        # Fallback
        name, nodes, edges = WORKFLOW_TEMPLATES["research_publish"]
        return "research_publish", nodes, edges

    # ------------------------------------------------------------------
    # Policy initialization
    # ------------------------------------------------------------------

    def _init_policies(self, dag: WorkflowDAG) -> None:
        """Initialiseer Thompson policies voor elke node in de DAG."""
        if self.store is None:
            return

        for node_id, node in dag.nodes.items():
            if not node.task_id or not node.action_ids:
                continue

            try:
                from thompson_policy import ThompsonPolicy
                from agent_learning.types import Action

                # Check of er al een policy bestaat
                existing = self.store.get_active_policy(self.agent_id, node.task_id)
                if existing is not None:
                    continue

                actions = [Action(id=aid) for aid in node.action_ids]
                policy = ThompsonPolicy.from_actions(
                    actions,
                    agent_id=self.agent_id,
                    task_id=node.task_id,
                    prior_alpha=1.0,
                    prior_beta=1.0,
                )
                # Sla op in store
                policy._snapshot.metadata["policy_kind"] = "thompson"
                policy._snapshot.metadata["workflow_dag"] = dag.id
                policy._snapshot.metadata["node_id"] = node_id
                self.store.store_policy(policy._snapshot)
                logger.info("  Init policy %s -> %s", node.task_id, node_id)
            except Exception as exc:
                logger.warning("  Failed to init policy for %s: %s", node_id, exc)

    # ------------------------------------------------------------------
    # DAG registration
    # ------------------------------------------------------------------

    def _register_dag(
        self, dag: WorkflowDAG, template_name: str, context: Dict[str, Any]
    ) -> None:
        """Registreer DAG in DagRegistry als beschikbaar."""
        try:
            from graph_evolution import DagRegistry
            if self.store is not None:
                registry = DagRegistry(self.store, agent_id=self.agent_id)
                registry.register(dag, priority=1, tags=[template_name, "synthesized"])
                logger.info("  Registered in DagRegistry")
        except Exception as exc:
            logger.warning("  Failed to register DAG: %s", exc)

    # ------------------------------------------------------------------
    # Callback factories
    # ------------------------------------------------------------------

    def make_decision(self, dag: WorkflowDAG) -> Callable:
        """Maak een decision functie voor deze DAG met constraints.

        De decision functie:
        1. Checkt constraints (filtert acties)
        2. Gebruikt Thompson policy uit store
        3. Valt terug op eerste toegestane actie als policy kiest
           voor een geblokkeerde actie
        """
        # Bepaal per node-type de constraint functie
        node_types: Dict[str, str] = {}
        for node_id in dag.nodes:
            # Haal node_type uit metadata (ingepland tijdens synthesize)
            node = dag.nodes[node_id]
            nt = node.metadata.get("node_type", "research")
            node_types[node_id] = nt

        constraint_fns = {
            nid: make_constraints(nt) for nid, nt in node_types.items()
        }

        from thompson_policy import ThompsonPolicy

        def _decide(node, context):
            # 1: Context override
            override = context.get(f"override_action:{node.id}")
            if override:
                return override, None, {"source": "context_override"}

            # 2: Constraints filteren acties
            cf = constraint_fns.get(node.id, make_constraints("default"))
            allowed = cf(node.id, list(node.action_ids or []), context)
            if not allowed:
                allowed = list(node.action_ids or [])

            # 3: Policy beslissing (als store beschikbaar)
            if self.store is not None and node.task_id:
                try:
                    snapshot = self.store.get_active_policy(
                        self.agent_id, node.task_id
                    )
                    if snapshot is not None:
                        policy = ThompsonPolicy.from_snapshot(snapshot)
                        decision = policy.choose()
                        chosen = decision.action.id
                        if chosen in allowed:
                            return chosen, decision.logprob, {
                                "source": "thompson_policy",
                                "policy_version": snapshot.version,
                                "allowed": allowed,
                            }
                        else:
                            # Policy koos geblokkeerde actie → fallback
                            chosen = allowed[0]
                            return chosen, None, {
                                "source": "constrained_fallback",
                                "policy_choice": decision.action.id,
                                "allowed": allowed,
                            }
                except Exception:
                    pass

            # 4: Fallback naar eerste toegestane actie
            if allowed:
                return allowed[0], None, {"source": "fallback", "allowed": allowed}
            return "skip", None, {"source": "no_actions"}

        return _decide

    def make_execute(self, dag: WorkflowDAG) -> Callable:
        """Maak een generieke execute functie.

        In de echte implementatie roept dit de juiste tool aan obv
        action_id. Voor nu een simpele stub die de actie logged.
        """
        def _execute(node, action_id, context):
            import time
            started = time.time()
            time.sleep(0.1)  # minimale simulatie

            summary = f"{node.name} via {action_id}"
            if context.get("topic"):
                summary += f" over {context['topic']}"

            context[f"_result:{node.id}"] = {
                "action": action_id,
                "summary": summary,
                "latency_s": round(time.time() - started, 2),
            }

            return "completed", summary, {
                "simulated": True,
                "latency_s": round(time.time() - started, 2),
            }

        return _execute

    def make_reward(self) -> Callable:
        """Maak een reward functie obv CostAwareReward."""
        from cost_aware_rewards import CostAwareReward
        reward_fn = CostAwareReward()
        return reward_fn.compute

    def make_local_reward(self, dag: WorkflowDAG) -> LocalRewardFn:
        """Maak lokale reward functies per node-type."""
        node_types: Dict[str, str] = {}
        for node_id, node in dag.nodes.items():
            nt = node.metadata.get("node_type", "research")
            node_types[node_id] = nt

        local_fns = {
            nid: make_local_reward(nt) for nid, nt in node_types.items()
        }

        def _local_reward(node, action_id, status, summary, exec_meta, context):
            fn = local_fns.get(node.id, make_local_reward("default"))
            return fn(node, action_id, status, summary, exec_meta, context)

        return _local_reward


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Workflow Synthesizer")
    parser.add_argument("task", nargs="?", default=None,
                        help="Taakomschrijving (of --interactive)")
    parser.add_argument("--template", default=None,
                        choices=list(WORKFLOW_TEMPLATES),
                        help="Forceer een template")
    parser.add_argument("--topic", default="onbekend onderwerp",
                        help="Context: topic")
    parser.add_argument("--list-templates", action="store_true",
                        help="Toon beschikbare templates")
    parser.add_argument("--list-node-types", action="store_true",
                        help="Toon beschikbare node types")
    parser.add_argument("--dry-run", action="store_true",
                        help="Alleen tonen, niet registreren")
    args = parser.parse_args()

    if args.list_templates:
        print("Beschikbare workflow templates:")
        for name, (desc, nodes, _) in WORKFLOW_TEMPLATES.items():
            print(f"  {name:<35} {desc:<30} → {' → '.join(nodes)}")
        return 0

    if args.list_node_types:
        print("Beschikbare node types:")
        for nid, nt in NODE_TYPES.items():
            print(f"  {nid:<20} {nt.name:<25} acties={nt.action_ids}")
        return 0

    if not args.task:
        print("Geef een taakomschrijving of --list-templates")
        return 1

    # Synthesize
    try:
        from agent_learning.storage.cosmos import get_default_store
        store = get_default_store()
    except Exception:
        store = None

    syn = WorkflowSynthesizer(store)
    dag = syn.synthesize(
        task=args.task,
        template=args.template,
        context={"topic": args.topic},
    )

    # Toon resultaat
    print(f"\n{'=' * 50}")
    print(f"Workflow: {dag.name}")
    print(f"DAG ID:   {dag.id}")
    print(f"Template: {dag.id.split('_')[0]}")
    print(f"{'=' * 50}")
    print()

    order = dag.topological_sort()
    print(f"{'Node':<25} {'Task ID':<30} {'Acties'}")
    print(f"{'─'*25} {'─'*30} {'─'*30}")
    for nid in order:
        node = dag.nodes[nid]
        acts = ", ".join(node.action_ids or [])
        print(f"{node.name:<25} {node.task_id:<30} {acts}")

    print(f"\nEdges: {dag.edges}")
    print(f"Discount: γ={dag.discount_factor}")

    if not args.dry_run and store is not None:
        print(f"\nPolicies: {len(order)} Thompson policies geïnitialiseerd ✅")
        print(f"Geregistreerd in DagRegistry ✅")
        print(f"\nKlaar voor uitvoering met WorkflowRunner.")

    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "WorkflowSynthesizer", "NodeType", "NODE_TYPES", "WORKFLOW_TEMPLATES",
    "make_local_reward", "make_constraints",
]
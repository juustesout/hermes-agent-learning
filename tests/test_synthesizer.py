from __future__ import annotations

from workflow_synthesizer import (
    NODE_TYPES,
    WORKFLOW_TEMPLATES,
    WorkflowSynthesizer,
    make_constraints,
    make_local_reward,
)
from workflow_dag import WorkflowNode


def test_suggest_research_publish():
    syn = WorkflowSynthesizer()
    s = syn.suggest_template("Research AI trends and publish a blog")
    assert s["template"] == "research_publish"


def test_suggest_extract_template():
    syn = WorkflowSynthesizer()
    s = syn.suggest_template("Scrape the site and extract contact data")
    assert s["template"] == "scrape_extract_store"


def test_suggest_monitor_template():
    syn = WorkflowSynthesizer()
    s = syn.suggest_template("Monitor the feed and notify on changes")
    assert s["template"] == "monitor_validate_notify"


def test_suggest_validate_publish_template():
    syn = WorkflowSynthesizer()
    s = syn.suggest_template(
        "Research the topic, write an article, validate it, and publish"
    )
    assert s["template"] == "research_create_validate_publish"


def test_synthesize_builds_dag_from_template():
    syn = WorkflowSynthesizer()
    dag = syn.synthesize("ignored", template="research_publish")
    order = dag.topological_sort()
    assert order == ["research", "create", "publish"]
    assert dag.discount_factor == 0.8
    assert dag.distance_from_end("research") == 2
    assert dag.distance_from_end("publish") == 0


def test_synthesize_explicit_unknown_template_falls_back():
    syn = WorkflowSynthesizer()
    dag = syn.synthesize("totally unrelated xyz", template="does_not_exist")
    assert "research" in dag.nodes
    assert "publish" in dag.nodes


def test_node_types_have_actions():
    for nt in NODE_TYPES.values():
        assert nt.action_ids
        assert nt.default_correct_action in nt.action_ids
        node = nt.to_node("-ab12")
        assert node.task_id.endswith("-ab12")
        assert node.metadata["node_type"] == nt.id


def test_templates_reference_known_node_types():
    for name, (_, nodes, edges) in WORKFLOW_TEMPLATES.items():
        for n in nodes:
            assert n in NODE_TYPES
        for src, tgt in edges:
            assert src in nodes and tgt in nodes


def test_make_decision_honours_publish_constraints():
    syn = WorkflowSynthesizer()
    dag = syn.synthesize("x", template="research_publish")
    decide = syn.make_decision(dag)
    node = dag.nodes["publish"]
    action, _, meta = decide(node, {"_is_approved": False})
    assert action == "draft"
    assert "allowed" in meta or meta.get("source") in {"fallback", "constrained_fallback"}


def test_make_execute_simulated():
    syn = WorkflowSynthesizer()
    dag = syn.synthesize("x", template="research_publish")
    exe = syn.make_execute(dag)
    status, summary, meta = exe(dag.nodes["research"], "web_search", {"topic": "AI"})
    assert status == "completed"
    assert meta["simulated"] is True
    assert "latency_s" in meta


def test_local_reward_research_bonus():
    fn = make_local_reward("research")
    node = WorkflowNode(id="research", name="R")
    val, reason = fn(
        node, "web_search", "completed",
        "gevonden bron http://example.com extra content here", {}, {},
    )
    assert "sources" in reason
    assert val > 0.5


def test_publish_constraints_factory():
    fn = make_constraints("publish")
    allowed = fn("publish", ["wordpress", "draft"], {"_is_approved": False})
    assert allowed == ["draft"]

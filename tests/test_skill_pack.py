from __future__ import annotations

import json

import pytest

from skill_pack import (
    SCHEMA_VERSION,
    SkillPackError,
    dag_to_workflow_json,
    export_from_task,
    read_skill_pack,
    validate_cases,
    workflow_json_to_dag,
)
from workflow_synthesizer import WorkflowSynthesizer


def test_export_round_trip_stable_ids(tmp_path):
    dest = tmp_path / "skill"
    export_from_task(
        "Scrape contacts",
        dest,
        template="scrape_extract_store",
        skill_id="scrape-contacts",
        skill_version="2.0.0",
        description="store extracted contacts",
    )
    payload, cases = read_skill_pack(dest)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["skill_id"] == "scrape-contacts"
    assert payload["skill_version"] == "2.0.0"
    assert (dest / "README.md").exists()
    dag = workflow_json_to_dag(payload)
    assert dag.id == "scrape-contacts-v2.0.0"
    assert dag.topological_sort() == ["research", "extract", "store"]
    assert dag.nodes["store"].task_id == "store-scrape-contacts-v2.0.0"
    assert cases == []


def test_same_skill_exports_identical_workflow():
    syn = WorkflowSynthesizer()
    a = syn.synthesize("x", template="research_publish",
                       skill_id="blog-post", skill_version="1.0.0")
    b = syn.synthesize("x", template="research_publish",
                       skill_id="blog-post", skill_version="1.0.0")
    pa = dag_to_workflow_json(a, skill_id="blog-post", skill_version="1.0.0")
    pb = dag_to_workflow_json(b, skill_id="blog-post", skill_version="1.0.0")
    assert pa["workflow"] == pb["workflow"]


def test_cases_reject_raw_episodes():
    with pytest.raises(SkillPackError):
        validate_cases([{"name": "x", "correct_action": "a", "episode": {}}])


def test_invalid_skill_id_rejected():
    syn = WorkflowSynthesizer()
    dag = syn.synthesize("x", template="research_publish", skill_id="ok")
    with pytest.raises(SkillPackError):
        dag_to_workflow_json(dag, skill_id="Not Valid")


def test_cli_validate(tmp_path, capsys, monkeypatch):
    dest = tmp_path / "pack"
    export_from_task("x", dest, template="research_publish", skill_id="demo")
    from skill_pack import main
    monkeypatch.setattr("sys.argv", ["skill_pack.py", "validate", str(dest)])
    assert main() == 0
    out = capsys.readouterr().out
    assert "ok demo" in out


def test_workflow_json_has_no_policy_blob(tmp_path):
    dest = tmp_path / "pack"
    export_from_task("x", dest, template="research_publish", skill_id="demo")
    raw = json.loads((dest / "workflow.json").read_text())
    dumped = json.dumps(raw)
    assert "thompson_alpha" not in dumped
    assert "policy_snapshot" not in dumped
    assert "conversation_history" not in dumped

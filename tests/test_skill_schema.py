from __future__ import annotations

import pytest

import consolidate_ops
from skill_pack import export_from_task, read_skill_pack
from skill_schema import SchemaError, is_full_episode, validate


def _skill_payload(tmp_path):
    dest = tmp_path / "pack"
    export_from_task("x", dest, template="research_publish", skill_id="demo")
    payload, _ = read_skill_pack(dest)
    return payload


def test_exported_skill_validates(tmp_path):
    validate(_skill_payload(tmp_path), "skill")


def test_invalid_skill_id_rejected(tmp_path):
    payload = _skill_payload(tmp_path)
    payload["skill_id"] = "Not Valid"
    with pytest.raises(SchemaError):
        validate(payload, "skill")


def test_unknown_edge_rejected(tmp_path):
    payload = _skill_payload(tmp_path)
    payload["workflow"]["edges"].append({"source": "missing", "target": "publish"})
    with pytest.raises(SchemaError, match="unknown edge source"):
        validate(payload, "skill")


def test_episode_schema_and_is_full():
    base = {"id": "e1", "agent_id": "hermes", "task_id": "t"}
    validate(base, "episode")
    assert is_full_episode(base) is False
    full = {
        **base,
        "intent_summary": "do t",
        "action_id": "a",
        "expected_outcome": "ok",
        "execution_status": "failed",
        "result_summary": "failed",
    }
    validate(full, "episode")
    assert is_full_episode(full) is True


def test_reward_value_clamped():
    rw = {
        "id": "r1",
        "episode_id": "e1",
        "agent_id": "hermes",
        "source": "aggregate",
        "value": 0.5,
    }
    validate(rw, "reward")
    rw["value"] = 1.5
    with pytest.raises(SchemaError):
        validate(rw, "reward")
    rw["value"] = 0.5
    rw["source"] = "not-a-source"
    with pytest.raises(SchemaError):
        validate(rw, "reward")


def test_case_privacy():
    validate({"name": "simple", "correct_action": "use_jina_reader"}, "case")
    with pytest.raises(SchemaError, match="not allowed"):
        validate(
            {"name": "x", "correct_action": "a", "episode": {}},
            "case",
        )


def test_unknown_kind():
    with pytest.raises(SchemaError, match="unknown schema kind"):
        validate({}, "policy")


def test_consolidated_episode_and_reward_match_schema():
    ep = consolidate_ops.op_to_episode({
        "id": "fail-1",
        "task_id": "choose-scrape-strategy",
        "action_id": "use_jina_reader",
        "target": "",
        "execution_status": "failed",
        "result_summary": "",
        "action_type": "web_scrape",
    })
    validate(ep, "episode")
    assert is_full_episode(ep) is True
    rw = {
        "id": "r1",
        "episode_id": ep["id"],
        "agent_id": "hermes",
        "source": "aggregate",
        "value": 0.11,
    }
    validate(rw, "reward")

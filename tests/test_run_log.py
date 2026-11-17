"""Automations log their own steps; a failed or silent step is an error."""
from __future__ import annotations

import json

import pytest

import hermes_plugin
import opslog
import run_log


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    return tmp_path


def _ops(store):
    path = store / "operations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_ok_step_writes_automation_op(store):
    with run_log.run_log("podcast-daily", run_id="r1") as run:
        with run.step("research", action_id="llm_suggest") as s:
            s.ok(cost=1.0, http_status=200)
    ops = _ops(store)
    assert len(ops) == 1
    entry = ops[0]
    assert entry["task_id"] == "podcast-daily"
    assert entry["action_id"] == "llm_suggest"
    assert entry["execution_status"] == "completed"
    assert entry["metadata"]["source"] == "automation"
    assert entry["metadata"]["run_id"] == "r1"
    assert entry["metadata"]["step"] == "research"
    assert entry["metadata"]["http_status"] == 200
    assert entry["metadata"]["cost_credits"] == 1.0
    assert entry["metadata"]["step_status"] == "ok"
    ends = [row for row in run_log.load_runs("podcast-daily") if row.get("record") == "end"]
    assert len(ends) == 1
    assert ends[0]["status"] == "ok"
    assert ends[0]["steps"] == 1
    assert ends[0]["failed"] == 0


def test_failed_step_is_logged_as_error(store):
    with run_log.run_log("podcast-daily", run_id="r-fail") as run:
        with run.step("publish", action_id="buzzsprout", target="https://example/ep") as s:
            s.fail(http_status=502, summary="upstream 502")
    entry = _ops(store)[0]
    assert entry["execution_status"] == "failed"
    assert entry["metadata"]["step_status"] == "fail"
    assert entry["metadata"]["http_status"] == 502
    assert "upstream 502" in entry["result_summary"]
    end = [row for row in run_log.load_runs() if row.get("record") == "end"][0]
    assert end["status"] == "partial"
    assert end["failed"] == 1


def test_exception_is_logged_and_reraised(store):
    with pytest.raises(RuntimeError, match="boom"):
        with run_log.run_log("podcast-daily", run_id="r-exc") as run:
            with run.step("audio", action_id="tts"):
                raise RuntimeError("boom")
    entry = _ops(store)[0]
    assert entry["execution_status"] == "failed"
    assert entry["metadata"]["step_status"] == "exception"
    assert "boom" in entry["result_summary"]
    end = [row for row in run_log.load_runs() if row.get("record") == "end"][0]
    assert end["status"] == "crashed"
    assert end["failed"] == 1


def test_silent_step_counts_as_unknown_failure(store):
    with run_log.run_log("podcast-daily") as run:
        with run.step("script", action_id="write"):
            pass
    entry = _ops(store)[0]
    assert entry["execution_status"] == "failed"
    assert entry["metadata"]["step_status"] == "unknown"
    end = [row for row in run_log.load_runs() if row.get("record") == "end"][0]
    assert end["status"] == "partial"
    assert end["failed"] == 1


def test_unfinished_run_is_visible_after_dropped_end(store):
    with run_log.run_log("podcast-daily", run_id="r-hang") as run:
        with run.step("research", action_id="llm_suggest") as s:
            s.ok()
    path = store / "runs_automation.jsonl"
    kept = [
        line
        for line in path.read_text().splitlines()
        if '"record": "end"' not in line
    ]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    hanging = run_log.unfinished_runs("podcast-daily")
    assert [row["run_id"] for row in hanging] == ["r-hang"]
    beat = run_log.heartbeat("podcast-daily", expected=1)
    assert beat["ok"] is False
    assert "unfinished_runs" in beat["alarms"]
    assert "missing_runs" in beat["alarms"]


def test_item_id_links_step_to_outcome(store):
    with run_log.run_log("hn-aidevforum", run_id="job-1", item_id="sim-0001") as run:
        with run.step("publish", action_id="wp_draft") as s:
            s.ok(http_status=201, item_id="sim-0001")
    entry = _ops(store)[0]
    assert entry["metadata"]["item_id"] == "sim-0001"
    assert entry["metadata"]["run_id"] == "job-1"


def test_hermes_hook_is_tagged_hermes_not_automation(store):
    hermes_plugin.on_post_tool_call(
        tool_name="web_search",
        args={"query": "bandits"},
        status="ok",
        duration_ms=12,
    )
    entry = _ops(store)[0]
    assert entry["metadata"]["source"] == "hermes"
    assert entry["metadata"].get("workflow") is None

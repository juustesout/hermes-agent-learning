from __future__ import annotations

import json
from pathlib import Path

import opslog


def test_log_operation_writes_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))

    oid = opslog.log_operation(
        task_id="choose-scrape-strategy",
        action_id="use_webclaw_scrape",
        target="example.com",
        success=True,
        duration_ms=3100,
        cost=1.0,
        summary="ok",
        action_logprob=-0.4,
        context_features={"cloudflare": 1.0},
    )
    assert oid
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["id"] == oid
    assert entry["task_id"] == "choose-scrape-strategy"
    assert entry["action_id"] == "use_webclaw_scrape"
    assert entry["action_logprob"] == -0.4
    assert entry["context_features"]["cloudflare"] == 1.0
    assert entry["execution_status"] == "completed"
    assert entry["metadata"]["cost_credits"] == 1.0
    assert entry["request_latency_ms"] == 3100


def test_log_failure_status(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    opslog.log_operation(task_id="t", action_id="a", success=False)
    entry = json.loads(log_path.read_text().strip())
    assert entry["execution_status"] == "failed"
    assert entry["metadata"]["task_completed"] is False


def test_count_tail_stats(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    assert opslog.count_operations() == 0
    assert opslog.tail() == []
    assert opslog.stats()["total"] == 0

    opslog.log_operation(task_id="t1", action_id="a", duration_ms=10, success=True)
    opslog.log_operation(task_id="t1", action_id="b", duration_ms=20, success=False)
    assert opslog.count_operations() == 2
    s = opslog.stats()
    assert s["total"] == 2
    assert s["by_task"]["t1"] == 2
    assert s["by_status"]["completed"] == 1
    assert s["by_status"]["failed"] == 1
    assert s["avg_duration_ms"] == 15.0
    assert len(opslog.tail(1)) == 1


def test_cli_log_with_logprob_and_context(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    rc = opslog.main([
        "log",
        "--task-id", "choose-scrape-strategy",
        "--action-id", "use_jina_reader",
        "--target", "site.nl",
        "--success", "false",
        "--duration-ms", "800",
        "--cost", "0",
        "--summary", "Jina failed",
        "--logprob", "-0.3",
        "--context-features", '{"cloudflare":1.0}',
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("ok ")
    entry = json.loads(Path(log_path).read_text().strip())
    assert entry["action_logprob"] == -0.3
    assert entry["context_features"] == {"cloudflare": 1.0}
    assert entry["execution_status"] == "failed"


def test_cli_invalid_context_features(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    rc = opslog.main([
        "log",
        "--task-id", "t",
        "--action-id", "a",
        "--context-features", "not-json",
    ])
    assert rc == 1


def test_cli_count_stats(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(tmp_path / "operations.jsonl"))
    opslog.log_operation(task_id="t", action_id="a")
    assert opslog.main(["count"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert opslog.main(["stats"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1

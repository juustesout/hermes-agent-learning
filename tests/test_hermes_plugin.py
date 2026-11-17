from __future__ import annotations

import json

import hermes_plugin
import opslog


class FakeCtx:
    def __init__(self, with_cli: bool = True):
        self.hooks = []
        self.cli = []
        if not with_cli:
            return

        def register_cli_command(*, name, help, setup_fn, handler_fn):
            self.cli.append(
                {
                    "name": name,
                    "help": help,
                    "setup_fn": setup_fn,
                    "handler_fn": handler_fn,
                }
            )

        self.register_cli_command = register_cli_command

    def register_hook(self, name, fn):
        self.hooks.append((name, fn))


def test_register_hook_and_cli():
    ctx = FakeCtx()
    hermes_plugin.register(ctx)
    assert ctx.hooks == [("post_tool_call", hermes_plugin.on_post_tool_call)]
    assert len(ctx.cli) == 1
    assert ctx.cli[0]["name"] == "agent-learning"


def test_register_without_cli_command():
    ctx = FakeCtx(with_cli=False)
    hermes_plugin.register(ctx)
    assert ctx.hooks[0][0] == "post_tool_call"
    assert ctx.cli == []


def test_build_operation_maps_web_extract():
    payload = hermes_plugin.build_operation(
        tool_name="web_extract",
        args={"url": "https://example.com/page"},
        status="ok",
        duration_ms=12.9,
        session_id="s1",
        turn_id="t1",
        tool_call_id="c1",
    )
    assert payload["task_id"] == "choose-scrape-strategy"
    assert payload["action_id"] == "web_extract"
    assert payload["target"] == "https://example.com/page"
    assert payload["success"] is True
    assert payload["duration_ms"] == 12
    assert payload["metadata"]["session_id"] == "s1"
    assert "result" not in payload
    assert "args" not in payload


def test_build_operation_skips_empty_tool():
    assert hermes_plugin.build_operation(tool_name="  ") is None


def test_blocked_is_failure():
    payload = hermes_plugin.build_operation(
        tool_name="terminal",
        status="blocked",
        error_message="denied",
    )
    assert payload["task_id"] == "hermes-tool"
    assert payload["success"] is False
    assert "denied" in payload["summary"]


def test_extract_target_ignores_command():
    assert hermes_plugin.extract_target({"command": "rm -rf /", "query": "q"}) == "q"
    assert hermes_plugin.extract_target("not-a-dict") == ""


def test_hook_writes_opslog(tmp_path, monkeypatch):
    log_path = tmp_path / "operations.jsonl"
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setattr(opslog, "OPSLOG_PATH", str(log_path))
    hermes_plugin.on_post_tool_call(
        tool_name="web_search",
        args={"query": "bandits"},
        status="ok",
        duration_ms=40,
        result={"hits": 9},
    )
    entry = json.loads(log_path.read_text().strip())
    assert entry["task_id"] == "web-research"
    assert entry["action_id"] == "web_search"
    assert entry["target"] == "bandits"
    assert entry["execution_status"] == "completed"
    assert "hits" not in json.dumps(entry)


def test_hook_fail_open(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(opslog, "log_operation", boom)
    hermes_plugin.on_post_tool_call(tool_name="web_search", status="ok")


def test_kernel_roots_honor_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LEARNING_REPO", str(tmp_path))
    roots = hermes_plugin._kernel_roots()
    assert roots[0] == tmp_path

"""Thin Hermes plugin: post_tool_call -> opslog. Kernel stays Hermes-free."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


def _kernel_roots() -> list[Path]:
    here = Path(__file__).resolve().parent
    roots = [here.parent, here]
    env = os.environ.get("AGENT_LEARNING_REPO", "").strip()
    if env:
        roots.insert(0, Path(env))
    return roots


for _root in _kernel_roots():
    if (_root / "opslog.py").is_file():
        _root_s = str(_root)
        if _root_s not in sys.path:
            sys.path.insert(0, _root_s)
        break

try:
    import opslog
except Exception:  # pragma: no cover - fail-open when kernel is absent
    opslog = None

PLUGIN_NAME = "agent-learning"
DEFAULT_TASK_ID = "hermes-tool"
_SUMMARY_MAX = 500
_TARGET_MAX = 300
_TARGET_KEYS = ("url", "uri", "path", "query", "target", "href")
_TOOL_TASK = {
    "web_search": "web-research",
    "web_extract": "choose-scrape-strategy",
    "browser_navigate": "choose-scrape-strategy",
}


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_target(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    for key in _TARGET_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value.strip(), _TARGET_MAX)
    return ""


def task_id_for_tool(tool_name: str) -> str:
    return _TOOL_TASK.get(tool_name, DEFAULT_TASK_ID)


def summarize_tool_call(
    *,
    tool_name: str,
    status: str,
    error_message: Any = None,
) -> str:
    err = _truncate(error_message, _SUMMARY_MAX) if error_message else ""
    if err:
        return _truncate(f"{tool_name}:{status} {err}", _SUMMARY_MAX)
    return _truncate(f"{tool_name}:{status}", _SUMMARY_MAX)


def build_operation(
    *,
    tool_name: str = "",
    args: Any = None,
    status: str = "",
    duration_ms: Any = None,
    error_message: Any = None,
    session_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> Optional[dict]:
    name = str(tool_name or "").strip()
    if not name:
        return None
    outcome = str(status or "ok").strip().lower() or "ok"
    latency = None
    if duration_ms is not None:
        try:
            latency = int(duration_ms)
        except (TypeError, ValueError):
            latency = None
    return {
        "task_id": task_id_for_tool(name),
        "action_id": name,
        "target": extract_target(args),
        "success": outcome == "ok",
        "duration_ms": latency,
        "summary": summarize_tool_call(
            tool_name=name, status=outcome, error_message=error_message
        ),
        "action_type": name,
        "metadata": {
            "hermes_status": outcome,
            "session_id": session_id or "",
            "turn_id": turn_id or "",
            "tool_call_id": tool_call_id or "",
        },
    }


def on_post_tool_call(**kwargs) -> None:
    try:
        if opslog is None:
            return
        payload = build_operation(**kwargs)
        if payload is None:
            return
        opslog.log_operation(**payload)
    except Exception:
        return


def _handle_cli(args) -> None:
    if opslog is None:
        print("opslog unavailable", file=sys.stderr)
        return
    cmd = getattr(args, "al_command", None)
    if cmd == "count":
        print(opslog.count_operations())
        return
    if cmd == "tail":
        n = getattr(args, "n", 10)
        for op in opslog.tail(n):
            print(json.dumps(op, indent=2)[:500])
            print("---")
        return
    if cmd == "stats":
        print(json.dumps(opslog.stats(), indent=2))
        return
    print("Usage: hermes agent-learning <stats|count|tail>")


def _setup_argparse(subparser) -> None:
    subs = subparser.add_subparsers(dest="al_command")
    subs.add_parser("stats", help="Show opslog stats")
    subs.add_parser("count", help="Count logged operations")
    p_tail = subs.add_parser("tail", help="Show recent operations")
    p_tail.add_argument("-n", type=int, default=10)
    subparser.set_defaults(func=_handle_cli)


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    register_cli = getattr(ctx, "register_cli_command", None)
    if register_cli is not None:
        register_cli(
            name=PLUGIN_NAME,
            help="Inspect agent-learning opslog",
            setup_fn=_setup_argparse,
            handler_fn=_handle_cli,
        )

"""Thin Hermes plugin: observer hooks -> opslog/shadow. Kernel stays Hermes-free."""

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

try:
    import shadow_policy
except Exception:  # pragma: no cover - fail-open when kernel is absent
    shadow_policy = None

PLUGIN_NAME = "agent-learning"
DEFAULT_TASK_ID = "hermes-tool"
_SUMMARY_MAX = 500
_TARGET_MAX = 300
_TARGET_KEYS = ("url", "uri", "path", "query", "target", "href")
_TOOL_TASK = {
    "web_search": "web-research",
    "web_extract": "choose-scrape-strategy",
    "browser_navigate": "choose-scrape-strategy",
    "mcp_webclaw_webclaw_scrape": "choose-scrape-strategy",
}


def logging_enabled() -> bool:
    raw = os.environ.get("AGENT_LEARNING_OPSLOG_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def log_session_ids() -> bool:
    raw = os.environ.get("AGENT_LEARNING_OPSLOG_SESSION_IDS", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _load_json_map(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items() if str(key).strip()}


def load_tool_task_map() -> dict[str, str]:
    mapping = dict(_TOOL_TASK)
    mapping.update(_load_json_map(os.environ.get("AGENT_LEARNING_TOOL_TASK_MAP", "")))
    path = os.environ.get("AGENT_LEARNING_TOOL_TASK_MAP_FILE", "").strip()
    if path:
        try:
            mapping.update(_load_json_map(Path(path).read_text(encoding="utf-8")))
        except OSError:
            pass
    return mapping


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
            text = _truncate(value.strip(), _TARGET_MAX)
            if opslog is not None:
                return opslog.redact_target(text)
            return text
    return ""


def task_id_for_tool(tool_name: str, mapping: Optional[dict[str, str]] = None) -> str:
    table = mapping if mapping is not None else load_tool_task_map()
    return table.get(tool_name, DEFAULT_TASK_ID)


def summarize_tool_call(
    *,
    tool_name: str,
    status: str,
    error_message: Any = None,
) -> str:
    err = _truncate(error_message, _SUMMARY_MAX) if error_message else ""
    if err:
        text = _truncate(f"{tool_name}:{status} {err}", _SUMMARY_MAX)
    else:
        text = _truncate(f"{tool_name}:{status}", _SUMMARY_MAX)
    if opslog is not None:
        return opslog.redact_text(text)
    return text


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
    metadata = {
        "source": "hermes",
        "hermes_status": outcome,
        "tool_ok": outcome == "ok",
        "task_success": False,
    }
    if log_session_ids():
        metadata["session_id"] = session_id or ""
        metadata["turn_id"] = turn_id or ""
        metadata["tool_call_id"] = tool_call_id or ""
    raw_target = ""
    if isinstance(args, dict):
        for key in _TARGET_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                raw_target = value.strip()
                break
    target = extract_target(args)
    context_features = None
    try:
        from .. import features as feature_register
    except ImportError:
        try:
            import features as feature_register
        except Exception:
            feature_register = None
    if feature_register is not None:
        try:
            context_features = feature_register.advice_features(raw_target, action_id=name)
        except Exception as exc:
            context_features = None
            try:
                if opslog is not None:
                    opslog.record_health(ok=False, error=f"features: {exc}")
            except Exception:
                pass
    return {
        "task_id": task_id_for_tool(name),
        "action_id": name,
        "target": target,
        "success": outcome == "ok",
        "duration_ms": latency,
        "summary": summarize_tool_call(
            tool_name=name, status=outcome, error_message=error_message
        ),
        "action_type": name,
        "context_features": context_features,
        "metadata": metadata,
    }


def _health_error(exc: BaseException) -> None:
    try:
        if opslog is not None:
            opslog.record_health(ok=False, error=str(exc))
    except Exception:
        return


def on_pre_tool_call(**kwargs) -> None:
    try:
        if shadow_policy is None:
            return None
        name = str(kwargs.get("tool_name") or "").strip()
        shadow_policy.record_pre(
            task_id=task_id_for_tool(name),
            tool_name=name,
            args=kwargs.get("args"),
            session_id=str(kwargs.get("session_id") or ""),
            conversation_id=str(kwargs.get("conversation_id") or ""),
            turn_id=str(kwargs.get("turn_id") or ""),
            tool_call_id=str(kwargs.get("tool_call_id") or ""),
        )
    except Exception as exc:
        _health_error(exc)
    return None


def on_post_tool_call(**kwargs) -> None:
    try:
        if logging_enabled() and opslog is not None:
            payload = build_operation(**kwargs)
            if payload is not None:
                opslog.log_operation(**payload)
    except OSError:
        pass
    except Exception as exc:
        _health_error(exc)
    try:
        if shadow_policy is not None:
            shadow_policy.record_post(
                tool_name=str(kwargs.get("tool_name") or ""),
                session_id=str(kwargs.get("session_id") or ""),
                conversation_id=str(kwargs.get("conversation_id") or ""),
                turn_id=str(kwargs.get("turn_id") or ""),
                tool_call_id=str(kwargs.get("tool_call_id") or ""),
                status=str(kwargs.get("status") or ""),
                duration_ms=kwargs.get("duration_ms"),
                result=kwargs.get("result"),
            )
    except Exception as exc:
        _health_error(exc)


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
    if cmd == "health":
        print(json.dumps(opslog.health(), indent=2))
        return
    print("Usage: hermes agent-learning <stats|count|tail|health>")


def _setup_argparse(subparser) -> None:
    subs = subparser.add_subparsers(dest="al_command")
    subs.add_parser("stats", help="Show opslog stats")
    subs.add_parser("count", help="Count logged operations")
    subs.add_parser("health", help="Show opslog write/error health")
    p_tail = subs.add_parser("tail", help="Show recent operations")
    p_tail.add_argument("-n", type=int, default=10)
    subparser.set_defaults(func=_handle_cli)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    register_cli = getattr(ctx, "register_cli_command", None)
    if register_cli is not None:
        register_cli(
            name=PLUGIN_NAME,
            help="Inspect agent-learning opslog",
            setup_fn=_setup_argparse,
            handler_fn=_handle_cli,
        )

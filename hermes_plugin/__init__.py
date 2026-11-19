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
HOOKS_REGISTERED: dict[str, bool] = {
    "pre_tool_call": False,
    "post_tool_call": False,
}
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


def _load_features():
    """Return (module or None, "relative" | "absolute" | None, [import errors]).

    Relative first (works when the plugin loads as a subpackage), absolute as
    fallback. Every failed attempt is recorded so nothing fails silently.
    """
    errors: list[str] = []
    try:
        from .. import features as module

        return module, "relative", errors
    except Exception as exc:
        errors.append(f"relative: {type(exc).__name__}: {exc}")
    try:
        import features as module

        return module, "absolute", errors
    except Exception as exc:
        errors.append(f"absolute: {type(exc).__name__}: {exc}")
    return None, None, errors


def _health_error_text(message: str) -> None:
    try:
        if opslog is not None:
            opslog.record_health(ok=False, error=message)
    except Exception:
        return


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
    feature_register, import_source, import_errors = _load_features()
    metadata["import_source"] = import_source or "none"
    if feature_register is None:
        _health_error_text("features import failed: " + "; ".join(import_errors))
    else:
        try:
            context_features = feature_register.pre_call_features(
                raw_target, action_id=name
            )
        except Exception as exc:
            context_features = None
            _health_error_text(f"features: {type(exc).__name__}: {exc}")
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
    return None


def _handle_cli(args) -> None:
    cmd = getattr(args, "al_command", None)
    if cmd == "status":
        import plugin_status

        print(plugin_status.format_status(), end="")
        return
    if cmd == "selftest":
        import plugin_status

        print(plugin_status.format_selftest(), end="")
        return
    if opslog is None:
        print("opslog unavailable", file=sys.stderr)
        return
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
    try:
        import pool_v1
    except Exception:
        pool_v1 = None
    if pool_v1 is not None:
        if cmd == "pool" and getattr(args, "pool_cmd", None) == "import":
            print(json.dumps(pool_v1.import_pool(args.path), indent=2, default=str))
            return
        if cmd == "select":
            print(json.dumps(pool_v1.select_for_run(args.run_id), indent=2, default=str))
            return
        if cmd == "baseline":
            print(json.dumps(pool_v1.baseline_from_pool(args.pool), indent=2, default=str))
            return
        if cmd == "record-post":
            raw = str(getattr(args, "fallback", "false")).strip().lower()
            fallback = raw in {"1", "true", "yes", "y", "on"}
            print(
                json.dumps(
                    pool_v1.record_post(
                        run_id=args.run_id,
                        item_id=args.id,
                        http_status=args.http_status,
                        fallback=fallback,
                    ),
                    indent=2,
                    default=str,
                )
            )
            return
    print(
        "Usage: hermes agent-learning "
        "<stats|count|tail|health|status|selftest|pool|select|baseline|record-post>"
    )


def _setup_argparse(subparser) -> None:
    subs = subparser.add_subparsers(dest="al_command")
    subs.add_parser("stats", help="Show opslog stats")
    subs.add_parser("count", help="Count logged operations")
    subs.add_parser("health", help="Show opslog write/error health")
    subs.add_parser("status", help="Show hooks, features, opslog, gate")
    subs.add_parser("selftest", help="Run hooks against a temporary store")
    p_tail = subs.add_parser("tail", help="Show recent operations")
    p_tail.add_argument("-n", type=int, default=10)
    p_pool = subs.add_parser("pool", help="Import a pool.v1 candidate file")
    pool_sub = p_pool.add_subparsers(dest="pool_cmd")
    p_imp = pool_sub.add_parser("import", help="Validate and queue a pool.v1 file")
    p_imp.add_argument("path")
    p_sel = subs.add_parser("select", help="Choose two items for a run")
    p_sel.add_argument("--run-id", required=True)
    p_base = subs.add_parser("baseline", help="Baseline two items without writing the store")
    p_base.add_argument("--pool", required=True)
    p_post = subs.add_parser("record-post", help="Record HTTP result for a selected id")
    p_post.add_argument("--run-id", required=True)
    p_post.add_argument("--id", required=True)
    p_post.add_argument("--http-status", type=int, required=True)
    p_post.add_argument("--fallback", default="false")
    subparser.set_defaults(func=_handle_cli)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    HOOKS_REGISTERED["pre_tool_call"] = True
    HOOKS_REGISTERED["post_tool_call"] = True
    register_cli = getattr(ctx, "register_cli_command", None)
    if register_cli is not None:
        register_cli(
            name=PLUGIN_NAME,
            help="Inspect agent-learning opslog",
            setup_fn=_setup_argparse,
            handler_fn=_handle_cli,
        )
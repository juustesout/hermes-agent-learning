"""Minimal JSON Schema (2020-12 subset) validator + skill graph checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union
from urllib.parse import urlparse

SCHEMA_VERSION = "1.0.0"
_KINDS = ("skill", "workflow", "node", "episode", "reward", "case")
_BANNED_CASE_KEYS = frozenset(
    {"episode", "policy", "user_input", "conversation_history"}
)
_FULL_EPISODE_FIELDS = (
    "intent_summary",
    "expected_outcome",
    "execution_status",
    "result_summary",
)


class SchemaError(ValueError):
    def __init__(self, message: str, path: str = "$") -> None:
        self.path = path
        super().__init__(f"{path}: {message}" if path != "$" else message)


def schema_dir() -> Path:
    return Path(__file__).resolve().parent / "schemas"


def load_schema(kind: str) -> Dict[str, Any]:
    if kind not in _KINDS:
        raise SchemaError(f"unknown schema kind {kind!r}")
    path = schema_dir() / f"{kind}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def is_full_episode(episode: Mapping[str, Any]) -> bool:
    if not all(episode.get(k) for k in _FULL_EPISODE_FIELDS):
        return False
    return bool(episode.get("action_id") or episode.get("action_name"))


def validate(instance: Any, kind: str) -> None:
    schema = load_schema(kind)
    ctx = _Ctx(root_kind=kind)
    _check(instance, schema, "$", ctx, doc=schema)
    if kind == "skill":
        _check_skill_graph(instance)
        _check_no_policy_blob(instance)
    elif kind == "workflow":
        _check_workflow_graph(instance)
    elif kind == "case":
        _check_case_privacy(instance)
    elif kind == "episode" and not isinstance(instance, Mapping):
        raise SchemaError("episode must be an object")


def _check_case_privacy(instance: Mapping[str, Any]) -> None:
    banned = _BANNED_CASE_KEYS & set(instance)
    if banned:
        raise SchemaError(
            f"raw policy/episode fields are not allowed: {sorted(banned)}"
        )


def _check_no_policy_blob(instance: Mapping[str, Any]) -> None:
    dumped = json.dumps(instance)
    for needle in ("thompson_alpha", "policy_snapshot", "conversation_history"):
        if needle in dumped:
            raise SchemaError(f"skill pack must not contain {needle}")


def _check_skill_graph(instance: Mapping[str, Any]) -> None:
    wf = instance.get("workflow")
    if isinstance(wf, Mapping):
        _check_workflow_graph(wf)


def _check_workflow_graph(wf: Mapping[str, Any]) -> None:
    nodes = wf.get("nodes") or []
    node_ids = {
        n.get("id") for n in nodes if isinstance(n, Mapping) and n.get("id")
    }
    for i, edge in enumerate(wf.get("edges") or []):
        if not isinstance(edge, Mapping):
            continue
        src, tgt = edge.get("source"), edge.get("target")
        if src not in node_ids:
            raise SchemaError(f"unknown edge source {src!r}", f"$.edges[{i}]")
        if tgt not in node_ids:
            raise SchemaError(f"unknown edge target {tgt!r}", f"$.edges[{i}]")


class _Ctx:
    def __init__(self, root_kind: str) -> None:
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.root_kind = root_kind
        self.root = load_schema(root_kind)
        self.cache[f"{root_kind}.schema.json"] = self.root


def _resolve_ref(
    ref: str, doc: Dict[str, Any], ctx: _Ctx
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (target_schema, document_root_for_target)."""
    if ref.startswith("#/"):
        cur: Any = doc
        for part in ref[2:].split("/"):
            if not isinstance(cur, Mapping) or part not in cur:
                raise SchemaError(f"unresolved $ref {ref}")
            cur = cur[part]
        if not isinstance(cur, Mapping):
            raise SchemaError(f"$ref {ref} is not an object")
        return dict(cur), doc
    parsed = urlparse(ref)
    name = Path(parsed.path).name if parsed.scheme else Path(ref).name
    if name not in ctx.cache:
        path = schema_dir() / name
        if not path.exists():
            raise SchemaError(f"unresolved $ref {ref}")
        ctx.cache[name] = json.loads(path.read_text(encoding="utf-8"))
    resolved = ctx.cache[name]
    return resolved, resolved


def _join(path: str, key: Union[str, int]) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path != "$" else f"$.{key}"


def _types_ok(value: Any, declared: Union[str, Sequence[str]]) -> bool:
    names = (declared,) if isinstance(declared, str) else tuple(declared)
    for name in names:
        if name == "object" and isinstance(value, Mapping):
            return True
        if name == "array" and isinstance(value, list):
            return True
        if name == "string" and isinstance(value, str):
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
        if name == "null" and value is None:
            return True
    return False


def _check(value: Any, schema: Any, path: str, ctx: _Ctx, doc: Dict[str, Any]) -> None:
    if schema is True:
        return
    if schema is False:
        raise SchemaError("not allowed", path)
    if not isinstance(schema, Mapping):
        raise SchemaError("invalid schema", path)

    if "$ref" in schema:
        target, target_doc = _resolve_ref(schema["$ref"], doc, ctx)
        _check(value, target, path, ctx, doc=target_doc)
        extra = {k: v for k, v in schema.items() if k != "$ref"}
        if extra:
            _check(value, extra, path, ctx, doc=doc)
        return

    if "const" in schema and value != schema["const"]:
        raise SchemaError(f"must be {schema['const']!r}", path)
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"must be one of {schema['enum']!r}", path)
    if "type" in schema and not _types_ok(value, schema["type"]):
        raise SchemaError(f"expected type {schema['type']!r}", path)

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaError(f"minLength {schema['minLength']}", path)
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise SchemaError(f"must match {schema['pattern']}", path)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(f"minimum {schema['minimum']}", path)
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaError(f"maximum {schema['maximum']}", path)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaError(f"minItems {schema['minItems']}", path)
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(value):
                _check(item, item_schema, _join(path, i), ctx, doc=doc)

    if isinstance(value, Mapping):
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise SchemaError(f"missing required {key!r}", path)
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            if key in props:
                _check(child, props[key], _join(path, key), ctx, doc=doc)
            elif additional is False:
                raise SchemaError(f"additional property {key!r}", path)
            elif additional is not True:
                _check(child, additional, _join(path, key), ctx, doc=doc)

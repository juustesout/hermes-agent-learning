"""Shadow policy advisor. Reads a cached snapshot; never changes Hermes behavior."""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import opslog
from constraints import apply_scrape_constraints
from scrape_features import FEATURE_NAMES, describe_site, phi_to_vector

SOURCE = "shadow"
AGENT_ID = "hermes"
SCRAPE_TASK = "choose-scrape-strategy"
KILL_ENV = "AGENT_LEARNING_SHADOW_KILL"
ENABLED_ENV = "AGENT_LEARNING_SHADOW_ENABLED"
LOG_NAME = "shadow_decisions.jsonl"
_PENDING_MAX = 1024
_PROB_DIGITS = 6

_CACHE: dict[str, Optional[dict]] = {}
_PENDING: OrderedDict[str, dict] = OrderedDict()


def reset_cache() -> None:
    _CACHE.clear()
    _PENDING.clear()


def shadow_enabled() -> bool:
    kill = os.environ.get(KILL_ENV, "").strip().lower()
    if kill in {"1", "true", "yes", "on"}:
        return False
    raw = os.environ.get(ENABLED_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def log_path() -> str:
    return os.path.join(opslog.STORE_DIR, LOG_NAME)


def _safe_component(value: str) -> str:
    encoded = quote(str(value), safe="").replace(".", "%2E")
    return encoded or "_empty_"


def _policy_root() -> Path:
    env = os.environ.get("AGENT_LEARNING_LOCAL_STORE_DIR", "").strip()
    if env:
        return Path(env)
    return Path(opslog.default_store_dir())


def _agent_id() -> str:
    return os.environ.get("AGENT_LEARNING_AGENT_ID", AGENT_ID).strip() or AGENT_ID


def join_key(
    *,
    session_id: str = "",
    conversation_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    tool_name: str = "",
) -> str:
    sid = str(session_id or "").strip() or str(conversation_id or "").strip()
    tc = str(tool_call_id or "").strip()
    if tc:
        return f"{sid}|{tc}"
    return f"{sid}|{str(turn_id or '').strip()}|{str(tool_name or '').strip()}"


def features_from_args(args: Any) -> dict[str, float]:
    target = ""
    if isinstance(args, dict):
        for key in ("url", "uri", "href", "path", "query", "target"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                target = value.strip()
                break
    if target:
        try:
            phi = describe_site(target, html="", headers=None)
        except Exception:
            phi = {name: 0.0 for name in FEATURE_NAMES}
    else:
        phi = {name: 0.0 for name in FEATURE_NAMES}
    return {name: float(phi.get(name, 0.0)) for name in FEATURE_NAMES}


def _action_ids(snapshot: dict) -> list[str]:
    ids: list[str] = []
    for item in snapshot.get("actions") or []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
        elif isinstance(item, str) and item:
            ids.append(item)
    return ids


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        n = len(logits)
        return [1.0 / n] * n
    return [x / total for x in exps]


def _round_probs(action_ids: list[str], probs: list[float]) -> dict[str, float]:
    return {
        aid: round(float(p), _PROB_DIGITS)
        for aid, p in zip(action_ids, probs)
        if aid
    }


def _phi_vector(snapshot: dict, features: dict[str, float]) -> list[float]:
    meta = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    dim = int(meta.get("feature_dim") or len(FEATURE_NAMES))
    names = meta.get("feature_names")
    if isinstance(names, list) and names:
        vec = [float(features.get(str(name), 0.0)) for name in names]
    else:
        vec = phi_to_vector(features)
    if len(vec) < dim:
        vec = vec + [0.0] * (dim - len(vec))
    return vec[:dim]


def _probs_for(snapshot: dict, action_ids: list[str], features: dict[str, float]) -> list[float]:
    n = len(action_ids)
    if n == 0:
        return []
    meta = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    kind = str(meta.get("policy_kind") or "")
    if kind == "contextual_softmax" or "context_weights" in meta:
        dim = int(meta.get("feature_dim") or len(FEATURE_NAMES))
        vec = _phi_vector(snapshot, features)
        weights = meta.get("context_weights") if isinstance(meta.get("context_weights"), dict) else {}
        logits = []
        for aid in action_ids:
            row = weights.get(aid) or []
            if not isinstance(row, list):
                row = []
            acc = 0.0
            for i, w in enumerate(row[:dim]):
                try:
                    acc += float(w) * vec[i]
                except (TypeError, ValueError, IndexError):
                    continue
            logits.append(acc)
        return _softmax(logits)
    if "thompson_alpha" in meta:
        alphas = meta.get("thompson_alpha") if isinstance(meta.get("thompson_alpha"), dict) else {}
        betas = meta.get("thompson_beta") if isinstance(meta.get("thompson_beta"), dict) else {}
        means = []
        for aid in action_ids:
            try:
                a = float(alphas.get(aid, 1.0))
                b = float(betas.get(aid, 1.0))
            except (TypeError, ValueError):
                a, b = 1.0, 1.0
            denom = a + b
            means.append(a / denom if denom > 0 else 1.0 / n)
        total = sum(means)
        if total <= 0:
            return [1.0 / n] * n
        return [m / total for m in means]
    raw_logits = snapshot.get("logits") if isinstance(snapshot.get("logits"), dict) else {}
    logits = []
    for aid in action_ids:
        try:
            logits.append(float(raw_logits.get(aid, 0.0)))
        except (TypeError, ValueError):
            logits.append(0.0)
    return _softmax(logits)


def advise(
    snapshot: Optional[dict],
    features: Optional[dict] = None,
    *,
    task_id: str = "",
) -> dict:
    if not isinstance(snapshot, dict):
        return {"advised_tool": "", "probabilities": {}}
    action_ids = _action_ids(snapshot)
    if not action_ids:
        return {"advised_tool": "", "probabilities": {}}
    phi = features or {}
    if str(task_id or snapshot.get("task_id") or "") == SCRAPE_TASK:
        action_ids = apply_scrape_constraints(phi, action_ids)
    probs = _probs_for(snapshot, action_ids, phi)
    if not probs:
        return {"advised_tool": "", "probabilities": {}}
    idx = max(range(len(probs)), key=lambda i: probs[i])
    meta = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    return {
        "advised_tool": action_ids[idx],
        "probabilities": _round_probs(action_ids, probs),
        "policy_id": str(snapshot.get("id") or ""),
        "policy_version": snapshot.get("version"),
        "policy_kind": str(meta.get("policy_kind") or ""),
    }


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_snapshot(task_id: str, *, agent_id: str = "") -> Optional[dict]:
    aid = agent_id or _agent_id()
    tid = str(task_id or "").strip()
    if not tid:
        return None
    root = _policy_root()
    pointer = _read_json(
        root / "active-policies" / _safe_component(aid) / f"{_safe_component(tid)}.json"
    )
    policy_id = str((pointer or {}).get("policy_id") or "").strip()
    if policy_id:
        doc = _read_json(
            root / "policies" / _safe_component(aid) / f"{_safe_component(policy_id)}.json"
        )
        if doc and str(doc.get("task_id") or tid) == tid:
            return doc
    policy_dir = root / "policies" / _safe_component(aid)
    if not policy_dir.is_dir():
        return None
    latest = None
    latest_key = (-1, "")
    try:
        entries = list(policy_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        if entry.suffix != ".json" or not entry.is_file():
            continue
        doc = _read_json(entry)
        if not doc or str(doc.get("task_id") or "") != tid:
            continue
        try:
            version = int(doc.get("version") or 0)
        except (TypeError, ValueError):
            version = 0
        key = (version, str(doc.get("created_at") or ""))
        if key > latest_key:
            latest = doc
            latest_key = key
    return latest


def cached_snapshot(task_id: str) -> Optional[dict]:
    tid = str(task_id or "").strip()
    if not tid:
        return None
    if tid in _CACHE:
        return _CACHE[tid]
    snap = load_snapshot(tid)
    _CACHE[tid] = snap
    return snap


def _stash(key: str, row: dict) -> None:
    if key in _PENDING:
        _PENDING.move_to_end(key)
    _PENDING[key] = row
    while len(_PENDING) > _PENDING_MAX:
        _PENDING.popitem(last=False)


def _append(row: dict) -> None:
    path = log_path()
    payload = json.dumps(row, default=str) + "\n"

    def _write():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()

    opslog._with_lock(path, _write)


def load_decisions() -> list[dict]:
    path = log_path()
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def record_pre(
    *,
    task_id: str,
    tool_name: str,
    args: Any = None,
    session_id: str = "",
    conversation_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
) -> None:
    if not shadow_enabled():
        return
    name = str(tool_name or "").strip()
    if not name:
        return
    tid = str(task_id or "").strip() or "hermes-tool"
    snapshot = cached_snapshot(tid)
    features = features_from_args(args)
    advice = advise(snapshot, features, task_id=tid)
    key = join_key(
        session_id=session_id,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        tool_name=name,
    )
    _stash(
        key,
        {
            "record": "decision",
            "source": SOURCE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "join_key": key,
            "session_id": str(session_id or ""),
            "conversation_id": str(conversation_id or ""),
            "turn_id": str(turn_id or ""),
            "tool_call_id": str(tool_call_id or ""),
            "requested_tool": name,
            "advised_tool": advice.get("advised_tool") or "",
            "probabilities": advice.get("probabilities") or {},
            "context_features": features,
            "task_id": tid,
            "policy_id": advice.get("policy_id") or "",
            "policy_version": advice.get("policy_version"),
            "policy_kind": advice.get("policy_kind") or "",
        },
    )


def record_post(
    *,
    tool_name: str = "",
    session_id: str = "",
    conversation_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
    status: str = "",
    duration_ms: Any = None,
) -> None:
    if not shadow_enabled():
        return
    key = join_key(
        session_id=session_id,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        tool_name=str(tool_name or "").strip(),
    )
    row = _PENDING.pop(key, None)
    if row is None:
        return
    outcome = str(status or "").strip().lower()
    latency = None
    if duration_ms is not None:
        try:
            latency = int(duration_ms)
        except (TypeError, ValueError):
            latency = None
    row["status"] = outcome
    row["duration_ms"] = latency
    row["hermes_status"] = outcome
    _append(row)

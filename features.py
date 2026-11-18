"""Feature register: Track A (pre_call) for advice, Track B (post_fetch) for memory."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

import opslog

PRE_CALL = "pre_call"
POST_FETCH = "post_fetch"
HOST_CATEGORIES = ("nieuws", "code", "social", "blog", "overig")
SEEN_CAP = 20
BETA_PRIOR = 1.0
CONSTANT_ADVISOR_THRESHOLD = 0.95
CONSTANT_ADVISOR_WINDOW = 200
OUTCOMES_NAME = "domain_outcomes.jsonl"

PRE_CALL_FEATURE_NAMES: list[str] = [
    "url_host_nieuws",
    "url_host_code",
    "url_host_social",
    "url_host_blog",
    "url_host_overig",
    "url_path_depth",
    "url_has_query",
    "domain_seen_count",
    "domain_last_success",
    "domain_prior_rate",
]
PRE_CALL_FEATURE_DIM = len(PRE_CALL_FEATURE_NAMES)

POST_FETCH_FEATURE_NAMES: list[str] = [
    "cloudflare",
    "requires_antibot",
    "js_heavy",
    "is_wordpress",
    "is_elementor",
    "content_len_norm",
    "has_structured_data",
]

_HOST_SUFFIXES: dict[str, tuple[str, ...]] = {
    "nieuws": (
        "bbc.com",
        "bbc.co.uk",
        "nytimes.com",
        "theguardian.com",
        "reuters.com",
        "apnews.com",
        "cnn.com",
        "npr.org",
        "nos.nl",
        "nu.nl",
        "volkskrant.nl",
        "telegraaf.nl",
        "nrc.nl",
        "ad.nl",
    ),
    "code": (
        "github.com",
        "githubusercontent.com",
        "gitlab.com",
        "bitbucket.org",
        "stackoverflow.com",
        "stackexchange.com",
        "pypi.org",
        "npmjs.com",
        "crates.io",
        "godbolt.org",
        "git.sr.ht",
        "codeberg.org",
    ),
    "social": (
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "reddit.com",
        "tiktok.com",
        "threads.net",
        "bsky.app",
        "mastodon.social",
    ),
    "blog": (
        "medium.com",
        "substack.com",
        "wordpress.com",
        "blogger.com",
        "ghost.io",
        "hashnode.dev",
        "tumblr.com",
    ),
}

_OUTCOMES: Optional[list[dict]] = None
_OUTCOMES_PATH = ""


class FeatureNameMismatch(ValueError):
    pass


def reset_memory() -> None:
    global _OUTCOMES, _OUTCOMES_PATH
    _OUTCOMES = None
    _OUTCOMES_PATH = ""


def outcomes_path() -> str:
    return os.path.join(opslog.STORE_DIR, OUTCOMES_NAME)


def require_pre_call_names(names: Any) -> list[str]:
    got = [str(n) for n in (names or [])]
    expected = list(PRE_CALL_FEATURE_NAMES)
    if got != expected:
        raise FeatureNameMismatch(
            f"snapshot feature_names {got!r} != register {expected!r}"
        )
    return expected


def host_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if raw and "://" not in raw:
        raw = "https://" + raw
    try:
        host = (urlsplit(raw).hostname or "").lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _path_query(url: str) -> tuple[float, float]:
    raw = str(url or "").strip()
    if raw and "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
    except ValueError:
        return 0.0, 0.0
    segs = [s for s in (parts.path or "").split("/") if s]
    depth = min(len(segs), 8) / 8.0
    has_query = 1.0 if parts.query else 0.0
    return depth, has_query


def host_category(host: str) -> str:
    name = str(host or "").lower()
    if name.startswith("www."):
        name = name[4:]
    if not name:
        return "overig"
    for cat in ("nieuws", "code", "social", "blog"):
        for suffix in _HOST_SUFFIXES[cat]:
            if name == suffix or name.endswith("." + suffix):
                return cat
    if "news" in name or name.startswith("nieuws."):
        return "nieuws"
    if "blog" in name:
        return "blog"
    if name.startswith("git.") or name.endswith(".github.io"):
        return "code"
    return "overig"


def _host_onehot(host: str) -> dict[str, float]:
    cat = host_category(host)
    return {f"url_host_{name}": 1.0 if name == cat else 0.0 for name in HOST_CATEGORIES}


def _ts_key(value: Any) -> str:
    return str(value or "")


def load_outcomes() -> list[dict]:
    global _OUTCOMES, _OUTCOMES_PATH
    path = outcomes_path()
    if _OUTCOMES is not None and _OUTCOMES_PATH == path:
        return _OUTCOMES
    _OUTCOMES_PATH = path
    rows: list[dict] = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("domain"):
                        rows.append(row)
        except OSError:
            rows = []
    _OUTCOMES = rows
    return _OUTCOMES


def remember_outcome(
    *,
    domain: str,
    success: bool,
    action_id: str = "",
    timestamp: str = "",
) -> None:
    host = str(domain or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return
    row = {
        "domain": host,
        "success": bool(success),
        "action_id": str(action_id or ""),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    load_outcomes().append(row)
    path = outcomes_path()
    payload = json.dumps(row, default=str) + "\n"

    def _write():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()

    try:
        opslog._with_lock(path, _write)
    except OSError:
        return


def _history_before(history: list[dict], at: Optional[str], domain: str) -> list[dict]:
    cutoff = _ts_key(at) if at else ""
    rows = []
    for item in history:
        host = str(item.get("domain") or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host != domain:
            continue
        if cutoff and _ts_key(item.get("timestamp")) >= cutoff:
            continue
        rows.append(item)
    return rows


def domain_stats(
    domain: str,
    *,
    history: Optional[list[dict]] = None,
    at: Optional[str] = None,
    action_id: str = "",
) -> dict[str, float]:
    host = str(domain or "").lower()
    if host.startswith("www."):
        host = host[4:]
    rows = _history_before(history if history is not None else load_outcomes(), at, host)
    n = len(rows)
    seen = min(n, SEEN_CAP) / float(SEEN_CAP)
    if n == 0:
        last = 0.5
        prior = BETA_PRIOR / (BETA_PRIOR + BETA_PRIOR)
    else:
        last = 1.0 if rows[-1].get("success") else 0.0
        succ = sum(1 for item in rows if item.get("success"))
        fail = n - succ
        prior = (BETA_PRIOR + succ) / (2.0 * BETA_PRIOR + succ + fail)
    stats = {
        "domain_seen_count": float(seen),
        "domain_last_success": float(last),
        "domain_prior_rate": float(prior),
    }
    if action_id:
        matched = [item for item in rows if str(item.get("action_id") or "") == action_id]
        a = BETA_PRIOR + sum(1 for item in matched if item.get("success"))
        b = BETA_PRIOR + sum(1 for item in matched if not item.get("success"))
        stats["domain_action_prior"] = float(a / (a + b))
    return stats


def action_priors(
    domain: str,
    action_ids: list[str],
    *,
    history: Optional[list[dict]] = None,
    at: Optional[str] = None,
) -> dict[str, float]:
    host = str(domain or "").lower()
    if host.startswith("www."):
        host = host[4:]
    rows = _history_before(history if history is not None else load_outcomes(), at, host)
    out = {}
    for aid in action_ids:
        matched = [item for item in rows if str(item.get("action_id") or "") == aid]
        a = BETA_PRIOR + sum(1 for item in matched if item.get("success"))
        b = BETA_PRIOR + sum(1 for item in matched if not item.get("success"))
        out[aid] = round(a / (a + b), 6)
    return out


def pre_call_features(
    url: str,
    *,
    history: Optional[list[dict]] = None,
    at: Optional[str] = None,
    action_id: str = "",
) -> dict[str, float]:
    host = host_from_url(url)
    depth, has_query = _path_query(url)
    phi = {name: 0.0 for name in PRE_CALL_FEATURE_NAMES}
    phi.update(_host_onehot(host))
    phi["url_path_depth"] = depth
    phi["url_has_query"] = has_query
    stats = domain_stats(host, history=history, at=at, action_id=action_id)
    phi["domain_seen_count"] = stats["domain_seen_count"]
    phi["domain_last_success"] = stats["domain_last_success"]
    if action_id and "domain_action_prior" in stats:
        phi["domain_prior_rate"] = stats["domain_action_prior"]
    else:
        phi["domain_prior_rate"] = stats["domain_prior_rate"]
    return {name: float(phi.get(name, 0.0)) for name in PRE_CALL_FEATURE_NAMES}


def post_fetch_features(
    url: str,
    html: str = "",
    headers: Optional[dict] = None,
) -> dict[str, float]:
    from scrape_features import describe_site

    raw = describe_site(url, html=html or "", headers=headers)
    return {name: float(raw.get(name, 0.0)) for name in POST_FETCH_FEATURE_NAMES}


def phi_vector(features: Optional[dict], names: Optional[list[str]] = None) -> list[float]:
    keys = names if names is not None else PRE_CALL_FEATURE_NAMES
    src = features or {}
    return [float(src.get(name, 0.0)) for name in keys]


def constant_advisor_status(
    rows: list[dict],
    *,
    window: int = CONSTANT_ADVISOR_WINDOW,
    threshold: float = CONSTANT_ADVISOR_THRESHOLD,
) -> dict:
    advised = [str(r.get("advised_tool") or "") for r in rows[-window:] if r.get("advised_tool")]
    n = len(advised)
    payload = {
        "alarm": False,
        "name": "constant_advisor",
        "n": n,
        "window": window,
        "threshold": threshold,
        "top_action": "",
        "top_share": 0.0,
    }
    if n < window:
        return payload
    counts: dict[str, int] = {}
    for aid in advised:
        counts[aid] = counts.get(aid, 0) + 1
    top_action = max(counts, key=lambda k: counts[k])
    share = counts[top_action] / n
    payload["top_action"] = top_action
    payload["top_share"] = round(share, 6)
    payload["alarm"] = share > threshold
    return payload


advice_features = pre_call_features

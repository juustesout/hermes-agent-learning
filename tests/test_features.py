from __future__ import annotations

import ast
import inspect

import pytest

import features
import shadow_policy
from features import FeatureNameMismatch, pre_call_features


def test_pre_call_works_without_html_headers_and_does_not_read_them():
    phi = pre_call_features("https://github.com/org/repo")
    assert phi["url_host_code"] == 1.0
    with pytest.raises(TypeError):
        pre_call_features("https://example.com", html="SENTINEL_HTML", headers={"SENTINEL": "1"})
    src = inspect.getsource(pre_call_features)
    tree = ast.parse(src)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "html" not in names
    assert "headers" not in names
    assert "html" not in attrs
    assert "headers" not in attrs


def test_advice_varies_across_twenty_urls_with_domain_history():
    history = [
        {"domain": "example.com", "success": True, "action_id": "web_extract", "timestamp": "2026-11-01T00:00:00+00:00"},
        {"domain": "example.com", "success": True, "action_id": "web_extract", "timestamp": "2026-11-02T00:00:00+00:00"},
        {"domain": "reddit.com", "success": False, "action_id": "web_extract", "timestamp": "2026-11-01T00:00:00+00:00"},
    ]
    urls = [
        "https://github.com/a/b",
        "https://gitlab.com/a/b",
        "https://stackoverflow.com/q/1",
        "https://nytimes.com/2026/11/18/tech",
        "https://nos.nl/artikel/1",
        "https://reddit.com/r/python",
        "https://twitter.com/x",
        "https://medium.com/@x/post",
        "https://example.com/",
        "https://example.com/docs/api/v1/users?q=1",
        "https://bbc.com/news/world",
        "https://pypi.org/project/x",
        "https://linkedin.com/in/x",
        "https://substack.com/p/x",
        "https://unknown-site.test/a/b/c/d",
        "https://nrc.nl/nieuws/1",
        "https://facebook.com/x",
        "https://bitbucket.org/x/y",
        "https://blog.example.org/post",
        "https://example.com/deep/path/here",
    ]
    snap = {
        "id": "pol-a",
        "task_id": "choose-scrape-strategy",
        "version": 1,
        "actions": [{"id": "use_webclaw_scrape"}, {"id": "use_jina_reader"}, {"id": "use_browser"}],
        "metadata": {
            "policy_kind": "contextual_softmax",
            "feature_dim": 10,
            "feature_names": list(features.PRE_CALL_FEATURE_NAMES),
            "context_weights": {
                "use_webclaw_scrape": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                "use_jina_reader": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "use_browser": [0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
    }
    advised = set()
    for url in urls:
        phi = pre_call_features(url, history=history, at="2026-11-18T00:00:00+00:00")
        advice = shadow_policy.advise(snap, phi)
        advised.add(advice["advised_tool"])
    assert len(advised) >= 2


def test_domain_last_success_ignores_later_outcomes():
    history = [
        {"domain": "example.com", "success": True, "timestamp": "2026-11-10T00:00:00+00:00"},
    ]
    at = "2026-11-11T00:00:00+00:00"
    before = pre_call_features("https://example.com/x", history=history, at=at)
    later = list(history) + [
        {"domain": "example.com", "success": False, "timestamp": "2026-11-12T00:00:00+00:00"},
    ]
    after = pre_call_features("https://example.com/x", history=later, at=at)
    assert before["domain_last_success"] == after["domain_last_success"] == 1.0
    future = pre_call_features(
        "https://example.com/x",
        history=later,
        at="2026-11-13T00:00:00+00:00",
    )
    assert future["domain_last_success"] == 0.0


def test_training_and_shadow_share_pre_call_function():
    assert shadow_policy.advice_features is features.pre_call_features
    assert features.advice_features is features.pre_call_features


def test_snapshot_feature_name_mismatch_is_explicit():
    snap = {
        "id": "old",
        "task_id": "choose-scrape-strategy",
        "actions": [{"id": "use_webclaw_scrape"}],
        "metadata": {
            "policy_kind": "contextual_softmax",
            "feature_dim": 8,
            "feature_names": ["cloudflare", "js_heavy"],
            "context_weights": {"use_webclaw_scrape": [1.0, 0.0]},
        },
    }
    with pytest.raises(FeatureNameMismatch, match="feature_names"):
        shadow_policy.advise(snap, {})


def test_shadow_record_contains_feature_track_and_values(tmp_path, monkeypatch):
    import hermes_plugin
    import opslog
    from urllib.parse import quote

    def _safe(value: str) -> str:
        return quote(str(value), safe="").replace(".", "%2E") or "_empty_"

    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path))
    features.reset_memory()
    shadow_policy.reset_cache()
    snap = {
        "id": "pol-a",
        "agent_id": "hermes",
        "task_id": "choose-scrape-strategy",
        "version": 1,
        "actions": [{"id": "use_webclaw_scrape"}, {"id": "use_jina_reader"}],
        "metadata": {
            "policy_kind": "contextual_softmax",
            "feature_dim": 10,
            "feature_names": list(features.PRE_CALL_FEATURE_NAMES),
            "context_weights": {
                "use_webclaw_scrape": [0.0] * 10,
                "use_jina_reader": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
    }
    policy_dir = tmp_path / "policies" / _safe("hermes")
    active_dir = tmp_path / "active-policies" / _safe("hermes")
    policy_dir.mkdir(parents=True)
    active_dir.mkdir(parents=True)
    import json

    (policy_dir / f"{_safe('pol-a')}.json").write_text(json.dumps(snap), encoding="utf-8")
    (active_dir / f"{_safe('choose-scrape-strategy')}.json").write_text(
        json.dumps({"agent_id": "hermes", "task_id": "choose-scrape-strategy", "policy_id": "pol-a"}),
        encoding="utf-8",
    )
    hermes_plugin.on_pre_tool_call(
        tool_name="web_extract",
        args={"url": "https://github.com/org/repo"},
        session_id="s",
        tool_call_id="c",
    )
    hermes_plugin.on_post_tool_call(
        tool_name="web_extract",
        status="ok",
        session_id="s",
        tool_call_id="c",
        duration_ms=9,
    )
    row = shadow_policy.load_decisions()[0]
    assert row["feature_track"] == list(features.PRE_CALL_FEATURE_NAMES)
    assert row["context_features"]["url_host_code"] == 1.0
    assert "cloudflare" not in row["context_features"]


def test_constant_advisor_alarm_over_window():
    rows = [{"advised_tool": "use_webclaw_scrape"}] * 200
    alarm = features.constant_advisor_status(rows)
    assert alarm["alarm"] is True
    assert alarm["name"] == "constant_advisor"
    mixed = [{"advised_tool": "use_webclaw_scrape"}] * 100 + [{"advised_tool": "use_jina_reader"}] * 100
    assert features.constant_advisor_status(mixed)["alarm"] is False
    short = [{"advised_tool": "use_webclaw_scrape"}] * 50
    assert features.constant_advisor_status(short)["alarm"] is False

from __future__ import annotations

from constraints import apply_podcast_constraints, apply_scrape_constraints

SCRAPE_ACTIONS = ["use_webclaw_scrape", "use_jina_reader", "use_browser"]
PUBLISH_ACTIONS = ["wordpress", "youtube", "spotify", "all", "draft"]


def test_cloudflare_blocks_jina():
    allowed = apply_scrape_constraints({"cloudflare": 1.0}, list(SCRAPE_ACTIONS))
    assert "use_jina_reader" not in allowed
    assert "use_webclaw_scrape" in allowed


def test_antibot_blocks_jina():
    allowed = apply_scrape_constraints({"requires_antibot": 0.9}, list(SCRAPE_ACTIONS))
    assert "use_jina_reader" not in allowed


def test_js_heavy_keeps_only_capable():
    allowed = apply_scrape_constraints({"js_heavy": 1.0}, list(SCRAPE_ACTIONS))
    assert set(allowed) <= {"use_browser", "use_webclaw_scrape"}
    assert "use_jina_reader" not in allowed


def test_simple_html_blocks_browser():
    allowed = apply_scrape_constraints(
        {"cloudflare": 0.0, "js_heavy": 0.0, "content_len_norm": 0.1},
        list(SCRAPE_ACTIONS),
    )
    assert "use_browser" not in allowed
    assert "use_jina_reader" in allowed


def test_known_simple_wp_blocks_browser():
    allowed = apply_scrape_constraints(
        {"site_known": 1.0, "cloudflare": 0.0, "js_heavy": 0.2},
        list(SCRAPE_ACTIONS),
    )
    assert "use_browser" not in allowed


def test_empty_filter_falls_back_to_all():
    allowed = apply_scrape_constraints(
        {"js_heavy": 1.0},
        ["use_jina_reader"],
    )
    assert allowed == ["use_jina_reader"]


def test_empty_features_treated_as_simple_html():
    allowed = apply_scrape_constraints({}, list(SCRAPE_ACTIONS))
    assert "use_browser" not in allowed
    assert "use_jina_reader" in allowed
    assert "use_webclaw_scrape" in allowed


def test_publish_without_approval_only_draft():
    allowed = apply_podcast_constraints(
        "publish", list(PUBLISH_ACTIONS), {"_is_approved": False}
    )
    assert allowed == ["draft"]


def test_publish_dry_run_only_draft():
    allowed = apply_podcast_constraints(
        "publish",
        list(PUBLISH_ACTIONS),
        {"_is_approved": True, "_is_dry_run": True},
    )
    assert allowed == ["draft"]


def test_publish_approved_keeps_live():
    allowed = apply_podcast_constraints(
        "publish",
        list(PUBLISH_ACTIONS),
        {"_is_approved": True, "_is_dry_run": False},
    )
    assert "wordpress" in allowed
    assert "draft" in allowed


def test_audio_keeps_skip():
    allowed = apply_podcast_constraints(
        "audio", ["tts_edge", "skip"], {}
    )
    assert "skip" in allowed
    assert allowed == ["tts_edge", "skip"]


def test_non_publish_node_unfiltered():
    allowed = apply_podcast_constraints("research", ["webclaw_deep", "llm_only"], {})
    assert allowed == ["webclaw_deep", "llm_only"]

from __future__ import annotations

import json

import pytest

import blind_label


def _isolate(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "store"))
    return evidence


def _item(item_id: str, **extra) -> dict:
    payload = {
        "id": item_id,
        "title": f"Title {item_id}",
        "url": f"https://news.ycombinator.com/item?id={item_id}",
        "summary": "A post about AI development tooling.",
        "arm": "learned",
        "policy_id": "hn-ranker",
        "rank": 1,
        "auto_relevant": True,
        "auto_duplicate": False,
        "choice_prob": 0.5,
    }
    payload.update(extra)
    return payload


def test_should_sample_is_stable():
    first = [blind_label.should_sample(f"id-{i}") for i in range(200)]
    second = [blind_label.should_sample(f"id-{i}") for i in range(200)]
    assert first == second
    assert 20 <= sum(first) <= 60


def test_blind_view_hides_arm_and_scores():
    view = blind_label.blind_view(_item("x1"))
    assert view["title"].startswith("Title")
    assert view["url"].startswith("https://")
    for key in blind_label.HIDDEN_FIELDS:
        assert key not in view


def test_enqueue_samples_and_next_is_blind(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    sampled = None
    for i in range(200):
        result = blind_label.enqueue_item(_item(f"post-{i}"))
        if result["sampled"]:
            sampled = result["id"]
            break
    assert sampled
    card = blind_label.next_card()
    assert card is not None
    assert card["id"] == sampled
    assert "arm" not in json.dumps(card)
    assert "policy" not in json.dumps(card)
    assert "auto_relevant" not in json.dumps(card)


def test_label_and_agreement(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(blind_label, "should_sample", lambda item_id, rate=0.2: True)
    blind_label.enqueue_item(_item("a", auto_relevant=True, auto_duplicate=False))
    blind_label.enqueue_item(_item("b", auto_relevant=True, auto_duplicate=False))
    blind_label.record_label("a", relevant=True, duplicate=False, needs_correction=False)
    blind_label.record_label("b", relevant=False, duplicate=False, needs_correction=False)
    report = blind_label.report()
    assert report["sampled"] == 2
    assert report["labeled"] == 2
    assert report["remaining"] == 0
    assert report["agreement"] == 0.5
    assert report["automatic_measure_ok"] is False
    assert report["human_success_rate"] == 0.5


def test_label_rejects_unsampled_id(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(blind_label, "should_sample", lambda item_id, rate=0.2: False)
    blind_label.enqueue_item(_item("skip-me"))
    with pytest.raises(ValueError, match="not in the blind sample"):
        blind_label.record_label("skip-me", relevant=True, duplicate=False, needs_correction=False)


def test_cli_next_and_label(tmp_path, monkeypatch, capsys):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(blind_label, "should_sample", lambda item_id, rate=0.2: True)
    rc = blind_label.main([
        "enqueue",
        "--json",
        json.dumps(_item("cli-1", arm="baseline")),
    ])
    assert rc == 0
    assert blind_label.main(["next"]) == 0
    out = capsys.readouterr().out
    assert "cli-1" in out
    assert "baseline" not in out
    assert "arm" not in out
    assert blind_label.main([
        "label",
        "--id", "cli-1",
        "--relevant", "yes",
        "--duplicate", "no",
        "--needs-correction", "no",
    ]) == 0
    capsys.readouterr()
    assert blind_label.main(["report"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["labeled"] == 1
    assert report["automatic_measure_ok"] is True


def test_quality_success():
    assert blind_label.quality_success(relevant=True, duplicate=False) is True
    assert blind_label.quality_success(relevant=True, duplicate=True) is False
    assert blind_label.quality_success(relevant=False, duplicate=False) is False

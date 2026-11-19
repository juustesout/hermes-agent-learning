from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import blind_label
import label_ui


def _isolate(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("AGENT_LEARNING_OPSLOG_DIR", str(tmp_path / "store"))
    return evidence


def _pool_item(item_id: str, **extra) -> dict:
    payload = {
        "id": item_id,
        "title": f"Title {item_id}",
        "url": f"https://example.com/{item_id}",
        "summary": f"Summary {item_id}",
        "rag_score": 0.8,
        "source_rank": 1,
        "hn_points": 40,
        "arm": "L",
        "auto_relevant": True,
    }
    payload.update(extra)
    return payload


def _json(url, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_card_payload_only_visible_fields():
    item = _pool_item("hidden-1", ranker_score=0.99, run_id="secret")
    payload = label_ui.card_payload(item)
    assert set(payload) == set(blind_label.VISIBLE_FIELDS)
    for key in blind_label.HIDDEN_FIELDS:
        assert key not in payload


def test_bind_host_rejects_non_loopback():
    with pytest.raises(ValueError, match="loopback"):
        label_ui.bind_host("0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        label_ui.bind_host("192.168.1.10")
    assert label_ui.bind_host("localhost") == "127.0.0.1"


def test_next_endpoint_is_blind(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    hidden = _pool_item("c0")
    hidden["ranker_score"] = 0.99
    blind_label.enqueue_pool([hidden], run_id="run-ui")
    ui = label_ui.LabelUI()
    url = ui.start(port=0)
    try:
        status, payload = _json(url + "/api/next")
        assert status == 200
        item = payload["item"]
        assert set(item) <= set(blind_label.VISIBLE_FIELDS)
        assert set(item) == set(blind_label.VISIBLE_FIELDS)
        dumped = json.dumps(payload)
        for key in ("rag_score", "source_rank", "hn_points", "arm", "auto_relevant", "ranker_score"):
            assert key not in dumped
        assert payload["counts"]["remaining"] == 1
        assert payload["counts"]["remaining"] == blind_label.report()["remaining"]
    finally:
        ui.stop()


def test_incomplete_label_rejected(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    blind_label.enqueue_pool([_pool_item("c1")], run_id="run-ui")
    ui = label_ui.LabelUI()
    url = ui.start(port=0)
    try:
        status, payload = _json(
            url + "/api/label",
            method="POST",
            body={"id": "c1", "relevant": "yes", "duplicate": "no"},
        )
        assert status == 400
        assert "needs_correction" in payload["error"]
        assert len(blind_label.load_labels()) == 0
        status, payload = _json(
            url + "/api/label",
            method="POST",
            body={
                "id": "c1",
                "relevant": "yes",
                "duplicate": "no",
                "needs_correction": "no",
            },
        )
        assert status == 200
        assert payload["item"] is None
        labels = blind_label.load_labels()
        assert len(labels) == 1
        assert labels[0]["rater"] == "juust"
        assert labels[0]["relevant"] is True
        assert labels[0]["duplicate"] is False
        assert labels[0]["needs_correction"] is False
    finally:
        ui.stop()


def test_relabel_keeps_log_and_updates_counts(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    blind_label.enqueue_pool([_pool_item("a"), _pool_item("b")], run_id="run-ui")
    ui = label_ui.LabelUI()
    url = ui.start(port=0)
    try:
        first = {
            "id": "a",
            "relevant": "yes",
            "duplicate": "no",
            "needs_correction": "no",
        }
        status, payload = _json(url + "/api/label", method="POST", body=first)
        assert status == 200
        assert payload["item"]["id"] == "b"
        assert payload["counts"]["remaining"] == 1
        assert payload["counts"]["labeled"] == 1
        assert payload["counts"]["labeled"] == blind_label.report()["labeled"]
        status, payload = _json(
            url + "/api/label",
            method="POST",
            body={
                "id": "a",
                "relevant": "no",
                "duplicate": "yes",
                "needs_correction": "yes",
                "notes": "changed",
            },
        )
        assert status == 200
        assert payload["counts"]["labeled"] == 1
        assert payload["counts"]["remaining"] == 1
        assert payload["counts"]["labeled"] == blind_label.report()["labeled"]
        rows = [row for row in blind_label.load_labels() if row["id"] == "a"]
        assert len(rows) == 2
        assert rows[0]["relevant"] is True
        assert rows[1]["relevant"] is False
        assert rows[1]["notes"] == "changed"
        latest = blind_label._latest_labels()["a"]
        assert latest["relevant"] is False
    finally:
        ui.stop()


def test_counts_match_report_windows(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pool = [_pool_item("t1"), _pool_item("t2"), _pool_item("old")]
    blind_label.enqueue_pool(pool, run_id="run-ui")
    now = datetime.now(timezone.utc)
    blind_label.record_label("t1", relevant=True, duplicate=False, needs_correction=False)
    blind_label.record_label("t2", relevant=False, duplicate=False, needs_correction=False)
    old = dict(blind_label.load_labels()[-1])
    old["id"] = "old"
    old["labeled_at"] = (now - timedelta(days=8)).isoformat()
    blind_label._append_jsonl(blind_label._path("labels.jsonl"), old)
    today, week = label_ui.labeled_windows(now=now)
    counts = label_ui.counts_payload()
    report = blind_label.report()
    assert counts["remaining"] == report["remaining"]
    assert counts["labeled"] == report["labeled"]
    assert counts["labeled"] == 3
    assert today == 2
    assert week == 2


def test_store_error_is_visible(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ui = label_ui.LabelUI()
    url = ui.start(port=0)
    try:
        monkeypatch.setattr(
            label_ui.blind_label,
            "unlabeled_queue",
            lambda: (_ for _ in ()).throw(OSError("store down")),
        )
        status, payload = _json(url + "/api/next")
        assert status == 500
        assert "store unavailable" in payload["error"]
    finally:
        ui.stop()


def test_html_has_no_horizontal_scroll_at_375():
    html = label_ui.HTML_PATH.read_text(encoding="utf-8")
    assert "overflow-x: hidden" in html
    assert "max-width: 100%" in html
    assert "width=device-width" in html
    assert "target=\"_blank\"" in html
    for leak in ("rag_score", "source_rank", "hn_points", "auto_relevant"):
        assert leak not in html

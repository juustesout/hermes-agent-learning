from __future__ import annotations

import json
from pathlib import Path

import features
import opslog
import replay_advice
import shadow_policy

TWELVE_URLS = [
    "https://github.com/a/b",
    "https://gitlab.com/a/b",
    "https://stackoverflow.com/q/1",
    "https://nytimes.com/2026/11/18/tech",
    "https://nos.nl/artikel/1",
    "https://reddit.com/r/python",
    "https://twitter.com/x",
    "https://medium.com/@x/post",
    "https://bbc.com/news/world",
    "https://pypi.org/project/x",
    "https://nrc.nl/nieuws/1",
    "https://bitbucket.org/x/y",
]


def _varied_snapshot() -> dict:
    return {
        "id": "pol-a",
        "agent_id": "hermes",
        "task_id": "choose-scrape-strategy",
        "version": 1,
        "actions": [
            {"id": "use_webclaw_scrape"},
            {"id": "use_jina_reader"},
            {"id": "use_browser"},
        ],
        "metadata": {
            "policy_kind": "contextual_softmax",
            "feature_dim": 10,
            "feature_names": list(features.PRE_CALL_FEATURE_NAMES),
            "feature_track": "pre_call",
            "context_weights": {
                "use_webclaw_scrape": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                "use_jina_reader": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "use_browser": [0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
    }


def _constant_snapshot() -> dict:
    snap = _varied_snapshot()
    snap["id"] = "pol-const"
    snap["metadata"]["context_weights"] = {
        "use_webclaw_scrape": [0.0] * 10,
        "use_jina_reader": [5.0] * 10,
        "use_browser": [0.0] * 10,
    }
    return snap


def _old_snapshot() -> dict:
    return {
        "id": "old",
        "task_id": "choose-scrape-strategy",
        "version": 1,
        "actions": [{"id": "use_webclaw_scrape"}],
        "metadata": {
            "policy_kind": "contextual_softmax",
            "feature_dim": 8,
            "feature_names": ["cloudflare", "js_heavy"],
            "context_weights": {"use_webclaw_scrape": [1.0, 0.0]},
        },
    }


def _episodes(urls: list[str], *, extra: list[dict] | None = None) -> list[dict]:
    rows = []
    for i, url in enumerate(urls, start=1):
        rows.append(
            {
                "id": f"ep-{i:02d}",
                "task_id": "choose-scrape-strategy",
                "target": url,
                "created_at": f"2026-11-{i:02d}T00:00:00+00:00",
                "action_id": "web_extract",
                "execution_status": "completed" if i % 2 else "failed",
            }
        )
    if extra:
        rows.extend(extra)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_snapshot(path: Path, snap: dict) -> Path:
    path.write_text(json.dumps(snap), encoding="utf-8")
    return path


def test_advice_features_is_pre_call():
    assert replay_advice.advice_features is features.pre_call_features


def test_gate_open_on_twelve_varied(tmp_path, capsys):
    jsonl = _write_jsonl(tmp_path / "eps.jsonl", _episodes(TWELVE_URLS))
    snap = _write_snapshot(tmp_path / "snap.json", _varied_snapshot())
    code = replay_advice.main(
        ["--jsonl", str(jsonl), "--snapshot", str(snap), "--min-n", "12"]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert code == 0
    assert report["gate_open"] is True
    assert report["n"] == 12
    assert report["n_distinct"] >= 2
    assert report["policy_id"] == "pol-a"
    assert report["mean_top_margin"] > 0
    assert report["skipped_no_host"] == 0
    assert report["rows"][0]["probabilities"]
    assert "use_jina_reader" in report["rows"][0]["probabilities"]
    assert "gate=open" in captured.err


def test_constant_advice_exit_two(tmp_path, capsys):
    jsonl = _write_jsonl(tmp_path / "eps.jsonl", _episodes(TWELVE_URLS))
    snap = _write_snapshot(tmp_path / "snap.json", _constant_snapshot())
    code = replay_advice.main(
        ["--jsonl", str(jsonl), "--snapshot", str(snap), "--min-n", "12"]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["gate_open"] is False
    assert report["n"] == 12
    assert report["n_distinct"] == 1
    assert report["top_share"] == 1.0
    assert report["top_action"] == "use_jina_reader"


def test_eleven_rows_exit_three(tmp_path, capsys):
    jsonl = _write_jsonl(tmp_path / "eps.jsonl", _episodes(TWELVE_URLS[:11]))
    snap = _write_snapshot(tmp_path / "snap.json", _varied_snapshot())
    code = replay_advice.main(
        ["--jsonl", str(jsonl), "--snapshot", str(snap), "--min-n", "12"]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 3
    assert report["gate_open"] is False
    assert report["n"] == 11


def test_old_snapshot_exit_four(tmp_path, capsys):
    jsonl = _write_jsonl(tmp_path / "eps.jsonl", _episodes(TWELVE_URLS))
    snap = _write_snapshot(tmp_path / "snap.json", _old_snapshot())
    code = replay_advice.main(["--jsonl", str(jsonl), "--snapshot", str(snap)])
    err = capsys.readouterr().err
    assert code == 4
    assert "feature_names" in err


def test_does_not_write_store_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(opslog, "STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", str(tmp_path / "store"))
    features.reset_memory()
    shadow_policy.reset_cache()
    jsonl = _write_jsonl(tmp_path / "eps.jsonl", _episodes(TWELVE_URLS))
    snap = _write_snapshot(tmp_path / "snap.json", _varied_snapshot())
    replay_advice.main(["--jsonl", str(jsonl), "--snapshot", str(snap)])
    assert not (tmp_path / "store" / "domain_outcomes.jsonl").exists()
    assert not (tmp_path / "store" / "shadow_decisions.jsonl").exists()


def test_later_outcome_does_not_leak_into_earlier_phi():
    snap = _varied_snapshot()
    first = {
        "id": "ep-1",
        "task_id": "choose-scrape-strategy",
        "target": "https://example.com/a",
        "created_at": "2026-11-10T00:00:00+00:00",
        "action_id": "web_extract",
        "execution_status": "completed",
    }
    second = {
        "id": "ep-2",
        "task_id": "choose-scrape-strategy",
        "target": "https://example.com/b",
        "created_at": "2026-11-12T00:00:00+00:00",
        "action_id": "web_extract",
        "execution_status": "failed",
    }
    rows = replay_advice.replay([first, second], snap)
    expected = features.pre_call_features(
        first["target"], history=[], at=first["created_at"], action_id="web_extract"
    )
    assert rows[0]["context_features"]["domain_last_success"] == expected["domain_last_success"]
    later_hist = [
        {
            "domain": "example.com",
            "success": True,
            "action_id": "web_extract",
            "timestamp": first["created_at"],
        }
    ]
    after = features.pre_call_features(
        second["target"],
        history=later_hist,
        at=second["created_at"],
        action_id="web_extract",
    )
    assert rows[1]["context_features"]["domain_last_success"] == after["domain_last_success"]
    assert (
        rows[0]["context_features"]["domain_last_success"]
        != rows[1]["context_features"]["domain_last_success"]
    )


def test_timeout_counts_as_fail_for_memory():
    assert replay_advice.outcome_success("timeout") is False
    assert replay_advice.outcome_success("blocked") is False
    assert replay_advice.outcome_success("cancelled") is False
    assert replay_advice.outcome_success("ok") is True
    assert replay_advice.outcome_success("running") is None


def test_rows_without_host_do_not_count(tmp_path, capsys):
    rows = _episodes(TWELVE_URLS[:11])
    rows.append(
        {
            "id": "ep-bad",
            "task_id": "choose-scrape-strategy",
            "target": "/just/a/path",
            "created_at": "2026-11-20T00:00:00+00:00",
            "action_id": "web_extract",
            "execution_status": "completed",
        }
    )
    jsonl = _write_jsonl(tmp_path / "eps.jsonl", rows)
    snap = _write_snapshot(tmp_path / "snap.json", _varied_snapshot())
    code = replay_advice.main(
        ["--jsonl", str(jsonl), "--snapshot", str(snap), "--min-n", "12"]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert code == 3
    assert report["n"] == 11
    assert report["skipped_no_host"] == 1
    assert report["skipped_samples"][0]["target"] == "/just/a/path"
    assert "skipped_no_host=1" in captured.err

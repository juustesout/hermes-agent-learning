"""Sim-rijen zijn gescheiden van live meting; overlap en balans zijn zichtbaar in de audit."""
import json
import pytest

import blind_label as bl


@pytest.fixture
def ev(tmp_path, monkeypatch):
    d = tmp_path / "hn-aidevforum-sim"
    d.mkdir()
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(d))
    bl.set_write_source(bl.SOURCE_LIVE)
    yield d
    bl.set_write_source(bl.SOURCE_LIVE)


def _write(d, name, rows):
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_sim_refused_for_live_dir_and_unset(tmp_path, monkeypatch):
    import opslog
    from pathlib import Path
    live = Path(opslog.default_store_dir()) / "evidence" / bl.WORKFLOW
    monkeypatch.setenv("AGENT_LEARNING_EVIDENCE_DIR", str(live))
    with pytest.raises(RuntimeError):
        bl.set_write_source(bl.SOURCE_SIM)
    monkeypatch.delenv("AGENT_LEARNING_EVIDENCE_DIR")
    with pytest.raises(RuntimeError):
        bl.set_write_source(bl.SOURCE_SIM)


def test_report_sim_view_sees_only_sim(ev):
    _write(ev, "items.jsonl", [{"id": "s1", "title": "t", "url": "https://e.x/3", "source": "sim"}])
    _write(ev, "queue.jsonl", [{"id": "s1", "title": "t", "url": "https://e.x/3", "source": "sim"}])
    _write(ev, "labels.jsonl", [{"id": "s1", "relevant": True, "duplicate": False,
                                 "quality_success": True, "source": "sim", "rater": "sim"}])
    assert bl.report(view="live")["labeled"] == 0
    sim = bl.report(view="sim")
    assert sim["labeled"] == 1 and sim["posted"] == 1


def test_writes_are_tagged(ev):
    bl.set_write_source(bl.SOURCE_SIM)
    bl._append_jsonl(ev / "labels.jsonl", {"id": "a"})
    row = json.loads((ev / "labels.jsonl").read_text().splitlines()[0])
    assert row["source"] == "sim"
    assert bl.is_live(row) is False


def test_legacy_sim_rows_are_not_live():
    assert bl.is_live({"id": "x", "rater": "sim"}) is False
    assert bl.is_live({"id": "x", "notes": "sim"}) is False
    assert bl.is_live({"id": "x", "rater": "juust", "source": "live"}) is True


def test_agreement_ignores_sim_labels(ev):
    live_item = {"id": "live-1", "title": "t", "url": "https://e.x/1"}
    sim_item = {"id": "sim-1", "title": "t", "url": "https://e.x/2"}
    _write(ev, "items.jsonl", [dict(live_item, source="live"), dict(sim_item, source="sim")])
    _write(ev, "queue.jsonl", [dict(live_item, source="live"), dict(sim_item, source="sim")])
    _write(ev, "labels.jsonl", [
        {"id": "live-1", "relevant": True, "duplicate": False, "quality_success": True, "source": "live", "rater": "juust"},
        {"id": "sim-1", "relevant": False, "duplicate": False, "quality_success": False, "source": "sim", "rater": "sim"},
    ])
    rep = bl.report()
    assert rep["labeled"] == 1
    assert rep["sampled"] == 1


def test_overlap_and_balance_stats():
    decisions = [
        {"chosen_id": "a", "baseline_pick_id": "a", "arm": "L", "source": "live"},
        {"chosen_id": "b", "baseline_pick_id": "c", "arm": "B", "source": "live"},
    ]
    assert bl.overlap_stats(decisions) == {
        "n": 2,
        "same": 1,
        "n_contrast": 1,
        "rate": 0.5,
        "contrast_rate": 0.5,
        "primary_claim": "itt_all_posted",
        "explanatory": "n_contrast",
    }
    assert bl.balance_stats(["L", "L", "B", "B"])["p_value"] == pytest.approx(1.0)
    assert bl.balance_stats(["L"] * 30 + ["B"] * 10)["p_value"] < 0.01


def test_alarms_fire_on_overlap_and_imbalance():
    same = [{"chosen_id": f"i{k}", "baseline_pick_id": f"i{k}", "arm": "L", "source": "live"}
            for k in range(40)]
    alarms, overlap, _ = bl.ranker_health_alarms(same)
    assert "high_overlap" in alarms and overlap["rate"] == 1.0

    skew = [{"chosen_id": f"i{k}", "baseline_pick_id": f"j{k}", "arm": "L", "source": "live"}
            for k in range(40)]
    skew += [{"chosen_id": f"x{k}", "baseline_pick_id": f"y{k}", "arm": "B", "source": "live"}
             for k in range(4)]
    alarms, _, balance = bl.ranker_health_alarms(skew)
    assert "arm_imbalance" in alarms and balance["L"] == 40


def test_sim_decisions_do_not_trigger_live_alarms():
    sim_only = [{"chosen_id": f"i{k}", "baseline_pick_id": f"i{k}", "arm": "L", "source": "sim"}
                for k in range(40)]
    alarms, overlap, _ = bl.ranker_health_alarms(sim_only)
    assert alarms == [] and overlap["n"] == 0

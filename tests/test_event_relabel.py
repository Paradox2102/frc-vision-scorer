"""scripts/event_relabel.py: replay selection (with the spec 5b fallbacks),
schedule order, incremental vs full, outputs and the CLI guards. No network,
no video, no Jev."""
import json
import sqlite3
import threading
import time

import pytest

import scripts.event_relabel as er
from web.database import SCHEMA

GOOD = {"config": {"window_labels": True}, "stopped": False, "duration_sec": 168.8,
        "calibration_version": "v"}
OLD_GOOD = {"window_labels": {"aborted": None}, "duration_sec": 168.8}


def _replay(tmp_path, rid, mk, created, summary, status="complete"):
    d = tmp_path / "replays" / rid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"summary": summary, "frames": [], "events": []}))
    video = tmp_path / f"{rid}.mp4"
    video.write_bytes(b"")
    return {"id": rid, "match_key": mk, "created_at": created, "status": status,
            "video": str(video)}


def _select(tmp_path, rows, matches=None):
    db = tmp_path / "frc.db"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    for r in rows:
        con.execute("INSERT INTO replays (id, created_at, match_key, status, source_video_path, "
                    "meta_path) VALUES (?, ?, ?, ?, ?, NULL)",
                    (r["id"], r["created_at"], r["match_key"], r["status"], r["video"]))
    con.commit()
    con.close()
    return er.select_replays("2026cur", matches, db_path=db, replays_dir=tmp_path / "replays")


def test_the_latest_qualifying_replay_wins(tmp_path):
    rows = [_replay(tmp_path, "a-good", "2026cur_qm8", "2026-11-10T09:00:00", GOOD),
            _replay(tmp_path, "b-stopped", "2026cur_qm8", "2026-11-10T10:00:00",
                    {**GOOD, "stopped": True}),
            _replay(tmp_path, "c-auto", "2026cur_qm8", "2026-11-10T11:00:00",
                    {**GOOD, "config": {"window_labels": True, "auto_period_only": True}}),
            _replay(tmp_path, "d-wl-off", "2026cur_qm8", "2026-11-10T12:00:00",
                    {**GOOD, "config": {}}),
            _replay(tmp_path, "e-partial", "2026cur_qm8", "2026-11-10T13:00:00", GOOD,
                    status="partial"),
            _replay(tmp_path, "f-short", "2026cur_qm8", "2026-11-10T14:00:00",
                    {**GOOD, "duration_sec": 20.0})]
    chosen, skipped = _select(tmp_path, rows)
    assert [c["row"]["id"] for c in chosen] == ["a-good"] and skipped == []


def test_old_replays_qualify_through_the_fallbacks(tmp_path):
    rows = [_replay(tmp_path, "old", "2026cur_qm11", "2026-11-01T09:00:00", OLD_GOOD),
            _replay(tmp_path, "nowl", "2026cur_qm11", "2026-11-02T09:00:00",
                    {"duration_sec": 168.8})]
    chosen, _ = _select(tmp_path, rows)
    assert [c["row"]["id"] for c in chosen] == ["old"]


def test_matches_without_a_qualifying_replay_are_skipped_and_listed(tmp_path):
    rows = [_replay(tmp_path, "x", "2026cur_qm5", "2026-11-10T09:00:00",
                    {**GOOD, "stopped": True})]
    chosen, skipped = _select(tmp_path, rows, matches="qm5,qm6")
    assert chosen == []
    assert skipped[0][0] == "2026cur_qm5" and "user-stopped" in skipped[0][1]
    assert skipped[1] == ("2026cur_qm6", "no replays")


def test_selection_is_in_schedule_order(tmp_path):
    rows = [_replay(tmp_path, f"r-{m}", f"2026cur_{m}", "2026-11-10T09:00:00", GOOD)
            for m in ("qm11", "sf1m1", "qm8", "qm118", "f1m1")]
    chosen, _ = _select(tmp_path, rows)
    assert [c["match_key"] for c in chosen] == [
        "2026cur_qm8", "2026cur_qm11", "2026cur_qm118", "2026cur_sf1m1", "2026cur_f1m1"]
    chosen, _ = _select(tmp_path, rows, matches="qm118, qm8")
    assert [c["match_key"] for c in chosen] == ["2026cur_qm8", "2026cur_qm118"]


def test_incremental_borrows_below_its_own_number_and_full_has_no_cutoff():
    assert er.before_match_for("incremental", "2026cur_qm11") == 11
    assert er.before_match_for("full", "2026cur_qm11") is None


def test_incremental_runs_one_at_a_time_in_order_and_full_runs_in_parallel():
    lock, state = threading.Lock(), {"active": 0, "peak": 0, "order": []}

    def fn(it):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["order"].append(it)
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return {"match_key": it}

    items = ["a", "b", "c", "d"]
    assert [r["match_key"] for r in er.run_all(items, fn, "incremental", workers=4)] == items
    assert state["peak"] == 1 and state["order"] == items
    state["peak"] = 0
    assert [r["match_key"] for r in er.run_all(items, fn, "full", workers=4)] == items
    assert state["peak"] > 1


def test_incremental_store_write_stops_at_the_first_error():
    seen = []

    def fn(it):
        seen.append(it)
        return {"match_key": it, "error": "boom"} if it == "b" else {"match_key": it}

    out = er.run_all(["a", "b", "c"], fn, "incremental", workers=1, stop_on_error=True)
    assert seen == ["a", "b"] and [r["match_key"] for r in out] == ["a", "b"]


def test_unlabelled_share_counts_slot_rows_without_a_team():
    frames = [{"robots": [{"slot": "B1", "team_number": "3128"},
                          {"slot": "R1", "team_number": None}]},
              {"robots": [{"slot": "B1", "team_number": None}, {"team_number": None}]}]
    assert er.unlabelled_share(frames) == pytest.approx(2 / 3)
    assert er.unlabelled_share([]) is None


def test_result_row_writes_the_relabelled_meta_and_its_unlabelled_share(tmp_path):
    item = {"match_key": "2026cur_qm8", "row": {"id": "abcdef123456"},
            "video": tmp_path / "v.mp4"}
    relabelled = {"summary": {}, "frames": [{"robots": [{"slot": "B1", "team_number": None}]}],
                  "events": []}
    out = {"result": {"aborted": None, "stats": {"windows": 1},
                      "row_sources": {"6773": {"own": 0}}},
           "relabelled": relabelled, "stats": {"windows": 1, "anchors_written": 0},
           "cache_origin": "file", "gco": 0.0}
    rec = er.result_row(item, out, tmp_path, score=False, seconds=12.34)
    assert rec["unlabelled_share"] == 1.0 and rec["stats"]["anchors_written"] == 0
    assert json.loads((tmp_path / "2026cur_qm8_abcdef12.meta.json").read_text()) == relabelled
    assert rec["row_sources"] == {"6773": {"own": 0}} and rec["seconds"] == 12.3
    assert rec["meta_out"].endswith("2026cur_qm8_abcdef12.meta.json") and rec["score"] is None


def test_no_write_back_flag_and_reset_never_clears_live():
    with pytest.raises(SystemExit):
        er.main(["--event", "2026cur", "--write"])            # phase 2: not built
    with pytest.raises(SystemExit):
        er.main(["--event", "2026cur", "--reset"])            # namespace defaults to live
    with pytest.raises(SystemExit):
        er.main(["--event", "2026cur", "--store-write", "--no-event-lineup"])


# ---- end to end with fakes -------------------------------------------------
import argparse  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import scripts.relabel_replay as rr  # noqa: E402
from web.field import event_anchors as ea  # noqa: E402
from tests.field.test_window_labels import (  # noqa: E402
    FPS, _Cal, _lineup_frames, _lineup_session, _source)

RED, BLUE = ["2637", "2724", "6497"], ["3128", "6773", "9066"]


def _args(tmp_path, **kw):
    d = dict(lineup_min_p=0.8, no_event_lineup=False, namespace="eval-x", mode="incremental",
             store_write=True, digit_cache_dir=str(tmp_path / "dc"), no_elim=False,
             alliance_rows=False, score=False)
    d.update(kw)
    return argparse.Namespace(**d)


def _item(tmp_path, mk):
    frames = _lineup_frames()
    mp = tmp_path / f"{mk}.json"
    mp.write_text(json.dumps({"summary": {"duration_sec": 168.8}, "frames": frames,
                              "events": []}))
    return {"match_key": mk, "row": {"id": f"id-{mk}", "match_key": mk, "created_at": "t"}, "meta_path": mp,
            "video": tmp_path / "v.mp4"}, frames


@pytest.fixture()
def fakes(tmp_path, monkeypatch):
    real_one, real_fetch = rr.relabel_one, ea.fetch
    fetched = {}
    state = {"frames": None}
    monkeypatch.setattr(rr, "calibration_for", lambda mk, meta: ({"calibration": _Cal()}, None))
    monkeypatch.setattr(rr, "job_roster", lambda mk: (RED, BLUE))
    monkeypatch.setattr(er, "_probe_fps", lambda v: FPS)
    monkeypatch.setattr(er, "load_layout",
                        lambda: {"main": SimpleNamespace(crop=(0, 0, 1920, 723))})
    monkeypatch.setattr(er.jev_reader, "JevSession",
                        lambda *a, **k: _lineup_session(state["frames"], who_p={"B": 0.9}))
    monkeypatch.setattr(rr, "relabel_one", lambda *a, **k: real_one(
        *a, frame_source=_source, db_path=tmp_path / "s.db", anchor_root=tmp_path / "tiles", **k))

    def spy_fetch(*a, **k):
        out = real_fetch(*a, **k)
        fetched[a[3]] = out
        return out
    monkeypatch.setattr(ea, "fetch", spy_fetch)
    return SimpleNamespace(fetched=fetched, state=state, tmp=tmp_path)


def test_incremental_match_borrows_what_the_previous_match_just_wrote(tmp_path, fakes):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    recs = []
    for mk in ("2026cur_qm3", "2026cur_qm8"):
        item, fakes.state["frames"] = _item(tmp_path, mk)
        recs.append(er.relabel_match(item, _args(tmp_path), out_dir))
    assert not any(r.get("error") for r in recs), recs
    assert fakes.fetched["2026cur_qm3"] == {}
    borrowed = fakes.fetched["2026cur_qm8"]
    assert borrowed and all(a.match_key == "2026cur_qm3" for v in borrowed.values() for a in v)
    assert recs[0]["stats"]["anchors_written"] == 8 and recs[1]["stats"]["rows_borrowed"] >= 0


def test_a_failed_borrow_fetch_is_an_error_not_a_silent_arm_1(tmp_path, fakes, monkeypatch):
    def boom(*a, **k):
        raise OSError("db locked")
    monkeypatch.setattr(ea, "fetch", boom)
    item, fakes.state["frames"] = _item(tmp_path, "2026cur_qm8")
    rec = er.relabel_match(item, _args(tmp_path), tmp_path)
    assert rec["error"].startswith("borrow fetch failed") and "db locked" in rec["error"]
    out = er.run_all([item, item], lambda it: er.relabel_match(it, _args(tmp_path), tmp_path),
                     "incremental", workers=1, stop_on_error=True)
    assert len(out) == 1


def test_incremental_refuses_a_match_it_cannot_order():
    with pytest.raises(ValueError):
        er.before_match_for("incremental", "garbage")
    assert er.before_match_for("full", "garbage") is None


def test_a_scoring_failure_keeps_the_stats_and_is_not_a_stage_error(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "score_numbers", lambda *a: (_ for _ in ()).throw(KeyError("gt")))
    item = {"match_key": "2026cur_qm8", "row": {"id": "abcdef123456"}, "video": tmp_path}
    out = {"result": {"aborted": None, "stats": {"windows": 1}, "row_sources": {}},
           "relabelled": {"summary": {}, "frames": [], "events": []}, "stats": {"windows": 1}}
    rec = er.result_row(item, out, tmp_path, score=True, seconds=1.0)
    assert "error" not in rec and "KeyError" in rec["score_error"] and rec["stats"] == {"windows": 1}
    assert rec["meta_out"]


def test_arm_tag_names_the_arm():
    ns = argparse.Namespace
    assert er.arm_tag(ns(no_event_lineup=True, no_elim=False, alliance_rows=False)) == "own"
    assert er.arm_tag(ns(no_event_lineup=False, no_elim=True, alliance_rows=True)) == \
        "event-noelim-alliance"


def _main_with_one_item(tmp_path, monkeypatch, extra):
    item, _ = _item(tmp_path, "2026cur_qm8")
    cleared = []
    monkeypatch.setattr(er, "select_replays", lambda *a, **k: ([item], []))
    monkeypatch.setattr(er, "relabel_match", lambda it, args, od: {"match_key": it["match_key"]})
    monkeypatch.setattr(er.ea, "clear", lambda ns, ev, **k: cleared.append((ns, ev)) or 5)
    return cleared, er.main(["--event", "2026cur", "--namespace", "eval-x", *extra])


def test_reset_clears_a_non_live_namespace(tmp_path, monkeypatch):
    cleared, _ = _main_with_one_item(tmp_path, monkeypatch,
                                     ["--reset", "--out-dir", str(tmp_path / "o")])
    assert cleared == [("eval-x", "2026cur")]


def test_a_non_empty_out_dir_is_refused_before_a_reset(tmp_path, monkeypatch):
    od = tmp_path / "o"
    od.mkdir()
    (od / "old.json").write_text("{}")
    with pytest.raises(SystemExit):
        _main_with_one_item(tmp_path, monkeypatch, ["--reset", "--out-dir", str(od)])
    # (the clear spy is local to the helper; the refusal comes before any clear call)


def test_bad_namespace_is_a_usage_error():
    with pytest.raises(SystemExit):
        er.main(["--event", "2026cur", "--namespace", "a/b"])


def test_full_store_write_runs_one_worker(tmp_path, monkeypatch):
    seen = {}
    item, _ = _item(tmp_path, "2026cur_qm8")
    monkeypatch.setattr(er, "select_replays", lambda *a, **k: ([item], []))
    monkeypatch.setattr(er, "run_all", lambda items, fn, mode, workers, stop_on_error=False:
                        (seen.setdefault("w", workers), [{"match_key": "2026cur_qm8"}])[1])
    er.main(["--event", "2026cur", "--namespace", "eval-x", "--mode", "full", "--store-write",
             "--workers", "4", "--out-dir", str(tmp_path / "o")])
    assert seen["w"] == 1

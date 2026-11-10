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

"""scripts/relabel_replay.py: replay provenance with the spec 5b fallbacks,
calibration choice, digit cache, roster order, and relabel_one end to end
with a fake reader, a fake frame source and a tmp anchor store. No network,
no video, nothing written outside tmp_path."""
import json
import sqlite3

import pytest

import scripts.relabel_replay as rr
from web.field import event_anchors as ea
from tests.field.test_window_labels import (
    FPS, FakeSession, _Cal, _lineup_frames, _lineup_session, _source)

RED, BLUE = ["2637", "2724", "6497"], ["3128", "6773", "9066"]
ROW = {"id": "replay-qm8", "match_key": "2026cur_qm8"}


def test_provenance_reads_the_recorded_fields():
    s = {"config": {"window_labels": True, "ground_center_offset_m": 0.2}, "stopped": False,
         "calibration_version": "v7", "duration_sec": 168.8}
    assert rr.provenance(s) == {"window_labels_on": True, "full_match": True,
                                "ground_center_offset_m": 0.2, "stopped": False,
                                "calibration_version": "v7", "fallbacks": []}


def test_provenance_fallbacks_for_old_replays():
    p = rr.provenance({"window_labels": {"aborted": None}, "duration_sec": 168.8})
    assert p["window_labels_on"] and p["full_match"] and p["ground_center_offset_m"] == 0.0
    assert p["stopped"] is False and p["calibration_version"] is None
    assert p["fallbacks"] == ["config", "stopped", "calibration_version"]
    assert rr.provenance({"duration_sec": 168.8})["window_labels_on"] is False
    assert rr.provenance({"window_labels": {}, "duration_sec": 140.0})["full_match"] is False
    assert rr.provenance(None)["full_match"] is False


def test_auto_only_and_stopped_runs_are_flagged():
    auto = rr.provenance({"config": {"window_labels": True, "auto_period_only": True},
                          "duration_sec": 168.8, "stopped": False})
    assert auto["full_match"] is False
    assert rr.provenance({"config": {}, "stopped": True, "duration_sec": 168.8})["stopped"]


VERSIONS = [{"version_id": "v1", "start_match": None, "created_at": "2026-06-01",
             "calibration": "C1"},
            {"version_id": "v2", "start_match": "2026cur_qm7", "created_at": "2026-06-02",
             "calibration": "C2"}]


def test_calibration_override_then_recorded_then_select_epoch_with_a_warning():
    v, warn = rr.choose_calibration(VERSIONS, "2026cur_qm8", override="v1", recorded="v2")
    assert v["version_id"] == "v1" and warn is None
    v, warn = rr.choose_calibration(VERSIONS, "2026cur_qm8", recorded="v1")
    assert v["version_id"] == "v1" and warn is None
    v, warn = rr.choose_calibration(VERSIONS, "2026cur_qm8")
    assert v["version_id"] == "v2" and "select_epoch" in warn
    v, warn = rr.choose_calibration(VERSIONS, "2026cur_qm3", recorded="gone")
    assert v["version_id"] == "v1" and "gone" in warn
    with pytest.raises(SystemExit):
        rr.choose_calibration(VERSIONS, "2026cur_qm8", override="nope")


def test_job_roster_is_driver_station_order(tmp_path):
    db = tmp_path / "r.db"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE matches (id INTEGER PRIMARY KEY, tba_match_key TEXT);"
                      "CREATE TABLE match_robots (match_id INT, alliance TEXT, position INT, "
                      "team_number TEXT);")
    con.execute("INSERT INTO matches VALUES (1, '2026cur_qm8')")
    con.executemany("INSERT INTO match_robots VALUES (1, ?, ?, ?)",
                    [("red", 3, "6497"), ("blue", 1, "3128"), ("red", 1, "2637"),
                     ("red", 2, "2724"), ("blue", 2, "6773"), ("blue", 3, "9066")])
    con.commit()
    con.close()
    assert rr.job_roster("2026cur_qm8", db_path=db) == (RED, BLUE)


def test_digit_cache_prefers_an_existing_file_then_the_replay_summary(tmp_path):
    f = tmp_path / "c.json"
    meta = {"summary": {"window_labels_digit_calls": {"calls": {"0": []}}}}
    in_summary = meta["summary"]["window_labels_digit_calls"]
    assert rr.digit_cache_for(meta, None) == (in_summary, "summary")
    assert rr.digit_cache_for(meta, str(f)) == (in_summary, "summary")
    f.write_text(json.dumps({"calls": {"1": []}}))
    assert rr.digit_cache_for(meta, str(f)) == ({"calls": {"1": []}}, "file")
    assert rr.digit_cache_for({"summary": {}}, None) == (None, "none")


@pytest.fixture()
def store(tmp_path):
    return {"db_path": tmp_path / "s.db", "anchor_root": tmp_path / "tiles"}


def _meta(frames):
    return {"summary": {"duration_sec": 168.8}, "frames": frames, "events": []}


def _go(meta, session, opts, store, row=ROW):
    return rr.relabel_one(row, meta, "video.mp4", _Cal(), RED, BLUE, opts, session, fps=FPS,
                          view=(0, 0, 1920, 723), frame_source=_source, **store)


def test_relabel_one_never_touches_its_input_and_is_read_only_by_default(store):
    frames = _lineup_frames()
    meta = _meta(frames)
    snapshot = json.dumps(meta)
    out = _go(meta, _lineup_session(frames, who_p={"B": 0.9}), rr.Options(), store)
    assert json.dumps(meta) == snapshot
    s = out["relabelled"]["summary"]
    assert s["window_labels"]["digit_named"] == 8 and "anchors_written" not in s["window_labels"]
    assert s["window_labels_digit_calls"]["calls"]
    assert out["gco"] == 0.0 and out["cache_origin"] == "none"
    assert not store["db_path"].exists()


def test_before_match_n_borrows_from_matches_numbered_below_n(store, monkeypatch):
    seen = []
    monkeypatch.setattr(ea, "fetch", lambda *a, **kw: seen.append(a) or {})
    frames = _lineup_frames()
    _go(_meta(frames), _lineup_session(frames, who_p={"B": 0.9}),
        rr.Options(event_lineup=True, namespace="eval-x", before_match=8), store)
    assert seen == [("eval-x", "2026cur", RED + BLUE, "2026cur_qm8", 7)]


def test_store_write_puts_this_replays_anchors_in_its_namespace_only(store):
    frames = _lineup_frames()
    qm3 = {"id": "replay-qm3", "match_key": "2026cur_qm3"}
    out = _go(_meta(frames), _lineup_session(frames, who_p={"B": 0.9}),
              rr.Options(event_lineup=True, namespace="eval-x", store_write=True), store, row=qm3)
    st = out["relabelled"]["summary"]["window_labels"]
    assert (st["anchors_written"], st["elim_anchors_written"], st["borrow_fetch_error"]) == (8, 0, None)
    got = ea.fetch("eval-x", "2026cur", ["3128", "6773"], "2026cur_qm8",
                   db_path=store["db_path"], root=store["anchor_root"])
    assert len(got["3128"]) == 4 and len(got["6773"]) == 4
    assert got["3128"][0].match_number == 3 and got["3128"][0].replay_id == "replay-qm3"
    assert ea.fetch("live", "2026cur", ["3128"], "2026cur_qm8", db_path=store["db_path"],
                    root=store["anchor_root"]) == {}


def test_the_digit_cache_file_is_created_once_and_reused(store, tmp_path):
    frames = _lineup_frames()
    cache = tmp_path / "cache" / "replay-qm8.json"
    first = _go(_meta(frames), _lineup_session(frames, who_p={"B": 0.9}),
                rr.Options(digit_cache=str(cache)), store)
    assert cache.exists() and first["cache_origin"] == "none"
    s = FakeSession(who_p={"B": 0.9})
    again = _go(_meta(frames), s, rr.Options(digit_cache=str(cache)), store)
    assert again["cache_origin"] == "file" and [q for q, _n in s.asked] == ["who"]
    assert again["result"]["labels"] == first["result"]["labels"]


def test_store_write_needs_the_event_lineup():
    with pytest.raises(SystemExit):
        rr.main(["--replay", "x", "--store-write"])


def test_knob_defaults_cover_every_knob():
    from web.job_config import KNOBS
    assert set(rr._knob_defaults()) == {k.name for k in KNOBS}


def test_provenance_missing_config_keys_fall_back_to_the_knob_defaults(monkeypatch):
    monkeypatch.setattr(rr, "_knob_defaults",
                        lambda: {"window_labels": True, "ground_center_offset_m": 0.3})
    p = rr.provenance({"config": {"sam2_track_fps": 5}, "duration_sec": 168.8})
    assert p["window_labels_on"] is True and p["ground_center_offset_m"] == 0.3
    assert rr.provenance({"config": {"window_labels": False},
                          "duration_sec": 168.8})["window_labels_on"] is False


def test_pct_is_none_safe():
    assert rr.pct(None) == "-" and rr.pct(0.5) == "50%"


def test_an_aborted_stage_writes_no_cache_and_no_store_rows(store, tmp_path):
    cache = tmp_path / "cache" / "x.json"
    out = rr.relabel_one(ROW, _meta(_lineup_frames()), "video.mp4", _Cal(), [], [],
                         rr.Options(event_lineup=True, namespace="eval-x", store_write=True,
                                    digit_cache=str(cache)),
                         FakeSession(), fps=FPS, view=(0, 0, 1920, 723),
                         frame_source=_source, **store)
    assert out["result"]["aborted"] and out["relabelled"] is None
    assert not cache.exists() and not cache.with_name("x.json.tmp").exists()
    assert not store["db_path"].exists()


def test_the_digit_cache_write_leaves_no_temp_file(store, tmp_path):
    frames = _lineup_frames()
    cache = tmp_path / "c" / "r.json"
    _go(_meta(frames), _lineup_session(frames, who_p={"B": 0.9}),
        rr.Options(digit_cache=str(cache)), store)
    assert [p.name for p in cache.parent.iterdir()] == ["r.json"]


def test_a_bad_namespace_is_refused_before_anything_runs():
    with pytest.raises(SystemExit):
        rr.main(["--replay", "x", "--event-lineup", "--namespace", "../evil"])


def test_deinterlace_option_reaches_the_stage(store):
    frames = _lineup_frames()
    on = _go(_meta(frames), _lineup_session(frames, who_p={"B": 0.9}),
             rr.Options(deinterlace=True), store)
    assert on["stats"]["tiles_deinterlaced"] == 0          # blank fake video: nothing combs
    assert on["result"]["digit_calls"]["fingerprints"][0]["deinterlace"] is True
    off = _go(_meta(frames), _lineup_session(frames, who_p={"B": 0.9}), rr.Options(), store)
    assert "tiles_deinterlaced" not in off["stats"]


def test_reader_option_reaches_the_stage(store):
    from tests.field.test_window_labels_student import FakeStudent
    frames = _lineup_frames()
    out = rr.relabel_one(ROW, _meta(frames), "video.mp4", _Cal(), RED, BLUE,
                         rr.Options(reader="student_only", lineup_min_p=1.01),
                         FakeSession(), fps=FPS, view=(0, 0, 1920, 723), frame_source=_source,
                         student=FakeStudent(slots=()), **store)
    assert out["stats"]["student"]["reader"] == "student_only"
    assert out["relabelled"]["summary"]["window_labels_digit_calls"]["reader"] == "student_only"
    assert rr.Options().reader == "jev"

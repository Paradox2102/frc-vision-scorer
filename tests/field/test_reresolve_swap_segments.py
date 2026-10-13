"""identity_swap_resolve (default OFF) through InferenceJob._reresolve_replay_frames:
replay per-frame labels and replay events follow the per-slot team segments;
knob OFF leaves the whole-match (gir) path byte-identical.

Synthetic match: R1 and R3 exchange robots at the crossing around f800 (R1
reads 4444 then 6666, R3 the opposite); every other slot keeps its robot.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

import web.app as app_mod
import web.database as database
from web.app import InferenceJob

BLUE = ["1111", "2222", "3333"]
RED = ["4444", "5555", "6666"]
TEAM = {0: "1111", 1: "2222", 2: "3333", 3: "4444", 4: "5555", 5: "6666"}
FRAMES = list(range(0, 1600, 8))
FPS = 60.0


def _pos(s, f):
    x, y = 2.0 + 0.01 * f + (6.0 if s == 4 else 0.0), 1.0 + 3.0 * s
    if s in (3, 5):                       # R1/R3 converge to y=13 around f800
        t = min(abs(f - 800) / (6.0 * FPS), 1.0)
        y = 13.0 + (y - 13.0) * t
    return [x, y]


def _job(tmp_path, monkeypatch, swap_on):
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", tmp_path)
    job = InferenceJob.__new__(InferenceJob)
    job._identity_log = []
    job._replay = SimpleNamespace(frames=[], events=[])
    job.identity_diagnostics = SimpleNamespace(gauge=lambda *a, **k: None)
    job.config = SimpleNamespace(global_identity_resolve=True,
                                 identity_swap_resolve=swap_on,
                                 identity_swap_lambda=3.0, identity_swap_prox_m=1.5)
    job.video_fps = FPS
    job.job_id = "swap_segments_test"
    job.gallery = None
    job._slot_source = {}
    job._resolved_team_by_slot = {}
    job._resolved_confirmed_by_slot = {}
    job.match_blue_teams = list(BLUE)
    job.match_red_teams = list(RED)
    job._get_display_team_number = lambda r: r.team_number if r else None
    canonical = {}
    for f in FRAMES:
        for s in range(6):
            job._identity_log.append({
                "frame_num": f, "identity_id": 100 + s, "slot_id": s,
                "alliance": "blue" if s < 3 else "red", "pos": _pos(s, f),
                "person_overlap": False, "confirmed": True})
    for s in range(6):
        canonical[s] = 100 + s
    job.robot_tracks = {iid: SimpleNamespace(team_number=None, team_number_confirmed=False,
                                             team_ocr_votes={}, appearance_embedding=None)
                        for iid in canonical.values()}
    job._canonical_identity_slots = lambda: dict(canonical)
    reads = []
    for s in (0, 1, 2, 4):
        reads += [{"frame": f, "robot_id": 100 + s, "matched_team": TEAM[s]}
                  for f in (200, 1200)]
    for i in range(4):
        reads += [{"frame": 200 + 40 * i, "robot_id": 103, "matched_team": "4444"},
                  {"frame": 200 + 40 * i, "robot_id": 105, "matched_team": "6666"},
                  {"frame": 1200 + 40 * i, "robot_id": 103, "matched_team": "6666"},
                  {"frame": 1200 + 40 * i, "robot_id": 105, "matched_team": "4444"}]
    with (tmp_path / "diagnostic_swap_segments_test.jsonl").open("w") as fh:
        for r in reads:
            fh.write(json.dumps({"type": "vlm_read", "vote_weight": 1.0, **r}) + "\n")
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in FRAMES]
    # ScoringEvent.frame_num is the PROCESSED-frame index (one replay frame
    # per processed frame): index 50 = source f400, index 175 = source f1400.
    job._replay.events = [
        {"frame_num": 50, "alliance": "red", "slot": "R1", "team_number": None,
         "confidence": "high", "robot_label": "R1"},
        {"frame_num": 175, "alliance": "red", "slot": "R1", "team_number": "4444",
         "confidence": "high", "robot_label": "4444"},
        {"frame_num": 175, "alliance": "blue", "slot": "B2", "team_number": "2222",
         "confidence": "low", "robot_label": "2222"},
        {"frame_num": 5000, "alliance": "red", "slot": "R3", "team_number": "6666",
         "confidence": "low", "robot_label": "6666"}]
    return job


def _labels_at(job, frame):
    fr = next(f for f in job._replay.frames if f["frame_num"] == frame)
    return {r["slot"]: r["team_number"] for r in fr["robots"]}


def test_swap_on_replay_rows_and_events_follow_segments(tmp_path, monkeypatch, capsys):
    job = _job(tmp_path, monkeypatch, swap_on=True)
    job._reresolve_replay_frames()
    out = capsys.readouterr().out
    assert "[swap-resolve] lambda=3 prox=1.5m | blue: none | red: R1<->R3 @f" in out
    segs = job._slot_segments
    assert [g["team_number"] for g in segs[3]] == ["4444", "6666"]
    assert [g["team_number"] for g in segs[5]] == ["6666", "4444"]
    assert all(len(segs[s]) == 1 for s in (0, 1, 2, 4))
    early, late = _labels_at(job, 400), _labels_at(job, 1400)
    assert early == {"B1": "1111", "B2": "2222", "B3": "3333",
                     "R1": "4444", "R2": "5555", "R3": "6666"}
    assert late == {**early, "R1": "6666", "R3": "4444"}
    ev = job._replay.events
    assert (ev[0]["team_number"], ev[0]["robot_label"]) == ("4444", "4444")
    assert (ev[1]["team_number"], ev[1]["robot_label"]) == ("6666", "6666")
    assert ev[2]["team_number"] == "2222"
    assert ev[3]["team_number"] == "6666"           # index beyond the recording: untouched
    # gir alone ties R1/R3 (4 reads each way) -> the whole-match labels and
    # the match_robot_identity overrides are unchanged by the knob.
    assert job._final_slot_meta[3]["team_number"] is None


def test_swap_off_is_identical_to_gir_only(tmp_path, monkeypatch, capsys):
    base = _job(tmp_path, monkeypatch, swap_on=False)
    del base.config.identity_swap_resolve                    # pre-knob config object
    base._reresolve_replay_frames()
    off = _job(tmp_path, monkeypatch, swap_on=False)
    off._reresolve_replay_frames()
    assert "[swap-resolve]" not in capsys.readouterr().out
    assert getattr(off, "_slot_segments", None) is None
    assert off._replay.frames == base._replay.frames
    assert off._replay.events == base._replay.events
    assert off._resolved_team_by_slot == base._resolved_team_by_slot
    assert off._final_slot_meta == base._final_slot_meta


def test_swap_with_no_swap_found_matches_gir_labels(tmp_path, monkeypatch):
    """Knob ON but nothing to swap: replay rows equal the knob-OFF rows."""
    on = _job(tmp_path, monkeypatch, swap_on=True)
    on.config.identity_swap_lambda = 1e9
    on._reresolve_replay_frames()
    off = _job(tmp_path, monkeypatch, swap_on=False)
    off._reresolve_replay_frames()
    assert all(len(v) == 1 for v in on._slot_segments.values())
    assert on._replay.frames == off._replay.frames


def test_swap_requires_global_resolve(tmp_path, monkeypatch, capsys):
    job = _job(tmp_path, monkeypatch, swap_on=True)
    job.config.global_identity_resolve = False
    job._reresolve_replay_frames()
    assert "[swap-resolve] skipped: requires global_identity_resolve" in capsys.readouterr().out
    assert getattr(job, "_slot_segments", None) is None


def test_relabel_events_maps_processed_index_to_source_frame():
    rec = SimpleNamespace(
        frames=[{"frame_num": 420 + 8 * i, "i": i} for i in range(10)],
        events=[{"frame_num": 1, "slot": "B1", "team_number": None},     # src 428
                {"frame_num": 9, "slot": "B1", "team_number": None},     # src 492
                {"frame_num": 10, "slot": "??", "team_number": "9"},
                {"frame_num": None, "slot": "B1", "team_number": "9"},
                {"frame_num": 99, "slot": "B1", "team_number": "9"}])
    segs = {0: [{"start_frame": 420, "team_number": "1"},
                {"start_frame": 460, "team_number": "2"}]}
    assert InferenceJob._relabel_replay_events(rec, segs) == 2
    assert [e["team_number"] for e in rec.events] == ["1", "2", "9", "9", "9"]
    assert rec.events[1]["robot_label"] == "2"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def test_segment_table_roundtrip_replaces(db):
    rows = [{"slot": 3, "slot_label": "R1", "start_frame": 0, "end_frame": 792,
             "team_number": "4444", "confirmed": True, "score": 4.0},
            {"slot": 3, "slot_label": "R1", "start_frame": 800, "end_frame": 1592,
             "team_number": None, "confirmed": False, "score": 0.0}]
    asyncio.run(db.replace_match_identity_segments("j1", rows))
    asyncio.run(db.replace_match_identity_segments("j1", rows))      # idempotent
    asyncio.run(db.replace_match_identity_segments("j2", rows[:1]))
    got = asyncio.run(db.get_match_identity_segments("j1"))
    assert [(r["slot"], r["start_frame"], r["end_frame"], r["team_number"], r["confirmed"])
            for r in got] == [(3, 0, 792, "4444", 1), (3, 800, 1592, None, 0)]
    assert len(asyncio.run(db.get_match_identity_segments("j2"))) == 1
    assert asyncio.run(db.get_match_identity_segments("nope")) == []

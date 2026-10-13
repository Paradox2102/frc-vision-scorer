"""identity_segment_resolve / clef_min_evidence_conf / slot_boundary_log (all
default OFF) through InferenceJob: finalize labels, replay rows + events,
segment persistence, live boundary recording.

Synthetic match: R1 is re-seeded onto another robot at f800 (a 6 m teleport):
4444 before, 6666 after; R3 carries 6666 then 4444 far away. R2 expires at
f600 and is re-seeded 0.8 s later 0.7 m from where it vanished, never read
after -> its second segment is stitched to 5555."""
import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

import web.app as app_mod
import web.database as database
from web.app import InferenceJob
from web.diagnostic import DiagnosticRecorder

BLUE = ["1111", "2222", "3333"]
RED = ["4444", "5555", "6666"]
FRAMES = list(range(0, 1600, 8))
FPS = 60.0


def _present(s, f):
    return not (s == 4 and 600 < f < 648)


def _pos(s, f):
    if s == 3:
        return [3.0, 2.0] if f < 800 else [9.0, 2.0]
    if s == 4:
        return [12.0, 8.0] if f <= 600 else [12.5, 8.5]
    if s == 5:
        return [9.0, 13.0] if f < 800 else [3.0, 13.0]
    return [2.0 + 2.5 * s, 4.0 + s]


def _job(tmp_path, monkeypatch, seg_on=False, swap_on=False, floor=0.0, gir=True,
         extra_reads=()):
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", tmp_path)
    job = InferenceJob.__new__(InferenceJob)
    job._identity_log = []
    job._replay = SimpleNamespace(frames=[], events=[])
    job.identity_diagnostics = SimpleNamespace(gauge=lambda *a, **k: None)
    job.config = SimpleNamespace(global_identity_resolve=gir,
                                 identity_swap_resolve=swap_on,
                                 identity_swap_lambda=3.0, identity_swap_prox_m=1.5,
                                 identity_segment_resolve=seg_on,
                                 clef_min_evidence_conf=floor)
    job.video_fps = FPS
    job.job_id = "segment_labels_test"
    job.gallery = None
    job._slot_source = {}
    job._resolved_team_by_slot = {}
    job._resolved_confirmed_by_slot = {}
    job.match_blue_teams = list(BLUE)
    job.match_red_teams = list(RED)
    job._get_display_team_number = lambda r: r.team_number if r else None
    for f in FRAMES:
        for s in range(6):
            if not _present(s, f):
                continue
            job._identity_log.append({
                "frame_num": f, "identity_id": 100 + s, "slot_id": s,
                "alliance": "blue" if s < 3 else "red", "pos": _pos(s, f),
                "person_overlap": False, "confirmed": True})
    canonical = {s: 100 + s for s in range(6)}
    job.robot_tracks = {iid: SimpleNamespace(team_number=None, team_number_confirmed=False,
                                             team_ocr_votes={}, appearance_embedding=None)
                        for iid in canonical.values()}
    job._canonical_identity_slots = lambda: dict(canonical)
    team = dict(enumerate(BLUE + RED))
    reads = []
    for s in (0, 1, 2):
        reads += [{"frame": f, "robot_id": 100 + s, "matched_team": team[s]}
                  for f in (200, 1200)]
    reads += [{"frame": 200, "robot_id": 104, "matched_team": "5555"},
              {"frame": 400, "robot_id": 104, "matched_team": "5555"}]
    for f in (200, 400, 600):
        reads += [{"frame": f, "robot_id": 103, "matched_team": "4444"},
                  {"frame": f, "robot_id": 105, "matched_team": "6666"}]
    for f in (1000, 1200):
        reads += [{"frame": f, "robot_id": 103, "matched_team": "6666"},
                  {"frame": f, "robot_id": 105, "matched_team": "4444"}]
    reads += list(extra_reads)
    with (tmp_path / "diagnostic_segment_labels_test.jsonl").open("w") as fh:
        for r in reads:
            fh.write(json.dumps({"type": "vlm_read", "vote_weight": 1.0, **r}) + "\n")
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in FRAMES]
    job._replay.events = [   # frame_num = processed-frame index (source f = 8 * index)
        {"frame_num": 50, "alliance": "red", "slot": "R1", "team_number": None,
         "confidence": "high", "robot_label": "R1"},
        {"frame_num": 150, "alliance": "red", "slot": "R1", "team_number": "4444",
         "confidence": "high", "robot_label": "4444"},
        {"frame_num": 100, "alliance": "red", "slot": "R2", "team_number": None,
         "confidence": "low", "robot_label": "R2"}]
    return job


def _labels_at(job, frame):
    fr = next(f for f in job._replay.frames if f["frame_num"] == frame)
    return {r["slot"]: r["team_number"] for r in fr["robots"]}


def test_segment_on_replay_rows_and_events_follow_segments(tmp_path, monkeypatch, capsys):
    job = _job(tmp_path, monkeypatch, seg_on=True)
    job._reresolve_replay_frames()
    out = capsys.readouterr().out
    assert "[segment-resolve] segments=" in out
    segs = job._slot_segments
    assert [g["team_number"] for g in segs[3]] == ["4444", "6666"]
    assert [g["team_number"] for g in segs[5]] == ["6666", "4444"]
    assert [(g["team_number"], g["source"]) for g in segs[4]] == [
        ("5555", "read"), ("5555", "stitched")]
    assert all("source" in g and "support" in g for v in segs.values() for g in v)
    early, late = _labels_at(job, 400), _labels_at(job, 1400)
    assert early == {"B1": "1111", "B2": "2222", "B3": "3333",
                     "R1": "4444", "R2": "5555", "R3": "6666"}
    assert late == {**early, "R1": "6666", "R3": "4444"}
    ev = job._replay.events
    assert [(e["team_number"], e["robot_label"]) for e in ev] == [
        ("4444", "4444"), ("6666", "6666"), ("5555", "5555")]
    # The whole-match (gir) labels -- what match_robot_identity keeps -- are
    # untouched by the segments.
    assert job._final_slot_meta[4]["team_number"] == "5555"


def test_segment_wins_over_swap_when_both_on(tmp_path, monkeypatch, capsys):
    job = _job(tmp_path, monkeypatch, seg_on=True, swap_on=True)
    job._reresolve_replay_frames()
    out = capsys.readouterr().out
    assert "[swap-resolve] skipped: identity_segment_resolve wins" in out
    assert "[segment-resolve] segments=" in out
    assert job._slot_segments[4][1]["source"] == "stitched"


def test_knobs_off_is_identical_to_pre_knob_config(tmp_path, monkeypatch, capsys):
    base = _job(tmp_path, monkeypatch)
    for attr in ("identity_segment_resolve", "clef_min_evidence_conf"):
        delattr(base.config, attr)
    base._reresolve_replay_frames()
    off = _job(tmp_path, monkeypatch)
    off._reresolve_replay_frames()
    out = capsys.readouterr().out
    assert "[segment-resolve]" not in out and "[clef-floor]" not in out
    assert getattr(off, "_slot_segments", None) is None
    assert off._replay.frames == base._replay.frames
    assert off._replay.events == base._replay.events
    assert off._resolved_team_by_slot == base._resolved_team_by_slot
    assert off._final_slot_meta == base._final_slot_meta


def test_segment_requires_global_resolve(tmp_path, monkeypatch, capsys):
    job = _job(tmp_path, monkeypatch, seg_on=True, gir=False)
    job._reresolve_replay_frames()
    assert "[segment-resolve] skipped: requires global_identity_resolve" in capsys.readouterr().out
    assert getattr(job, "_slot_segments", None) is None


def _weak_b3_reads():
    """Two blurry low-p Clef reads naming 2222 on B3 (weights exaggerated so
    gir's alliance optimum flips: B3=2222 + B1 = 7.0 > B1 + B2 + B3 = 6.0)."""
    probs = {"2222": 0.58, "3333": 0.1, "none": 0.3, "other": 0.02}
    return [{"frame": f, "robot_id": 102, "matched_team": "2222", "confidence": 0.58,
             "clef_probs": probs, "vote_weight": 2.5} for f in (400, 600)]


def test_floor_drops_weak_clef_reads_from_finalize_evidence(tmp_path, monkeypatch, capsys):
    off = _job(tmp_path, monkeypatch, extra_reads=_weak_b3_reads())
    off._reresolve_replay_frames()
    assert "[clef-floor]" not in capsys.readouterr().out
    on = _job(tmp_path, monkeypatch, floor=0.65, extra_reads=_weak_b3_reads())
    on._reresolve_replay_frames()
    assert "[clef-floor] ignoring 2 team read(s) with p(team) < 0.65" in capsys.readouterr().out
    assert off._final_slot_meta[2]["team_number"] == "2222"
    assert off._final_slot_meta[1]["team_number"] is None
    assert on._final_slot_meta[2]["team_number"] == "3333"
    assert on._final_slot_meta[1]["team_number"] == "2222"


# --------------------------------------------------------------------------- #
# Live boundary recording
# --------------------------------------------------------------------------- #
def _bare_job(**cfg):
    job = InferenceJob.__new__(InferenceJob)
    job.config = SimpleNamespace(**cfg)
    return job


def test_boundary_logging_switch():
    assert not _bare_job()._slot_boundary_logging()
    assert not _bare_job(slot_boundary_log=False, identity_segment_resolve=False
                         )._slot_boundary_logging()
    assert _bare_job(slot_boundary_log=True)._slot_boundary_logging()
    assert _bare_job(identity_segment_resolve=True)._slot_boundary_logging()


def test_record_slot_boundaries_collects_and_persists(tmp_path):
    job = _bare_job(slot_boundary_log=True)
    job.diagnostic = DiagnosticRecorder(tmp_path / "diag.jsonl")
    job.field_tracker = SimpleNamespace(slot_events=[
        {"event": "seed", "slot_id": 4, "seed_dist_m": 12.9, "track_id": 3}])
    job.identity_registry = SimpleNamespace(frame_rejections=[
        {"slot_id": 3, "distance_m": 0.4, "slot_disrupted": True, "reason": "appearance_veto",
         "outcome": "born"}])
    outs = [SimpleNamespace(slot_id=3, pos=(1.0, 2.0), corner_only=False),
            SimpleNamespace(slot_id=4, pos=(5.0, 6.0), corner_only=True)]
    job._record_slot_boundaries(800, outs, {3: (7, 8)})
    job.field_tracker.slot_events = []
    job.identity_registry.frame_rejections = []
    outs[0].corner_only = True
    job._record_slot_boundaries(808, outs, {})
    job.diagnostic.close()
    kinds = [(r["frame"], r["slot_id"], r["kinds"]) for r in job._slot_boundaries]
    assert kinds == [(800, 3, ["feed_change", "birth"]), (800, 4, ["seed"]),
                     (808, 3, ["corner_only_on"])]
    rows = [json.loads(line) for line in (tmp_path / "diag.jsonl").read_text().splitlines()]
    assert [r["type"] for r in rows] == ["slot_boundary"] * 3
    assert rows[1]["seed_dist_m"] == 12.9 and rows[0]["prev_track_id"] == 7


def test_record_slot_boundaries_never_raises():
    job = _bare_job(slot_boundary_log=True)          # no tracker / registry at all
    job._record_slot_boundaries(8, [SimpleNamespace(slot_id=0)], {})
    assert getattr(job, "_slot_boundaries", []) == []


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def test_segment_rows_persist_their_source(db):
    rows = [{"slot": 3, "slot_label": "R1", "start_frame": 0, "end_frame": 792,
             "team_number": "4444", "confirmed": True, "score": 2.0, "source": "read"},
            {"slot": 3, "slot_label": "R1", "start_frame": 800, "end_frame": 1592,
             "team_number": "4444", "confirmed": False, "score": 2.0, "source": "stitched"},
            {"slot": 4, "slot_label": "R2", "start_frame": 0, "end_frame": 8,
             "team_number": None, "confirmed": False, "score": 0.0}]
    asyncio.run(db.replace_match_identity_segments("j1", rows))
    got = asyncio.run(db.get_match_identity_segments("j1"))
    assert [r["source"] for r in got] == ["read", "stitched", None]


def test_init_db_adds_source_to_a_pre_existing_segment_table(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE match_robot_identity_segment (id INTEGER PRIMARY KEY "
                "AUTOINCREMENT, job_id TEXT NOT NULL, slot INTEGER NOT NULL, slot_label TEXT,"
                " start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL, team_number TEXT,"
                " confirmed INTEGER DEFAULT 0, score REAL, created_at TEXT)")
    con.execute("INSERT INTO match_robot_identity_segment (job_id, slot, start_frame, "
                "end_frame, team_number) VALUES ('old', 3, 0, 8, '4444')")
    con.commit()
    con.close()
    monkeypatch.setattr(database, "DB_PATH", path)
    asyncio.run(database.init_db())
    asyncio.run(database.init_db())                       # idempotent
    got = asyncio.run(database.get_match_identity_segments("old"))
    assert got[0]["team_number"] == "4444" and got[0]["source"] is None

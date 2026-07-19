from types import SimpleNamespace
from web.app import InferenceJob
from web.field.track_evidence import MatchEvidence


def _job():
    job = InferenceJob.__new__(InferenceJob)
    job._identity_log = []
    job._replay = SimpleNamespace(frames=[])
    job.robot_tracks = {}
    job.identity_diagnostics = SimpleNamespace(gauge=lambda *a, **k: None)
    return job


def test_reresolve_rebuilds_frames_excluding_human():
    job = _job()
    for f in range(3):
        job._identity_log.append({"frame_num": f, "identity_id": 1, "slot_id": 0,
                                  "alliance": "blue", "pos": [1.0, 4.0],
                                  "person_overlap": False, "confirmed": f >= 2})
    job._identity_log.append({"frame_num": 1, "identity_id": 9, "slot_id": 3,
                              "alliance": "red", "pos": [8.0, 7.0],
                              "person_overlap": True, "confirmed": False})
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in range(3)]
    job.robot_tracks = {1: SimpleNamespace(team_number="254", team_number_confirmed=True)}
    job._canonical_identity_slots = lambda: {0: 1}
    job._get_display_team_number = lambda r: r.team_number if r else None
    job._slot_alliance = lambda sid: "blue" if sid < 3 else "red"
    job._reresolve_replay_frames()
    assert all(len(f["robots"]) == 1 and f["robots"][0]["slot"] == "B1"
               and f["robots"][0]["team_number"] == "254" for f in job._replay.frames)
    assert job._replay.frames[0]["scores"] == {}   # original keys preserved


def test_reresolve_drops_phantom_prefix_from_frames_and_auto_paths():
    """Per-segment resolution: a slot that is a 9 s static phantom then a real
    moving robot (stitched by a gap-jump) keeps only the moving segment — the
    phantom prefix is dropped from both replay frames and autonomous_paths."""
    job = _job()
    job.video_fps = 60.0
    # slot 0 (B1): frames 0..9 parked static at x=15 (phantom), then a >1 s gap
    # and a big jump, then frames 120..131 moving (real robot).
    for f in range(10):
        job._identity_log.append({"frame_num": f, "identity_id": 1, "slot_id": 0,
                                  "alliance": "blue", "pos": [15.0, 2.4],
                                  "person_overlap": False, "confirmed": False})
    for i in range(12):
        job._identity_log.append({"frame_num": 120 + i, "identity_id": 1, "slot_id": 0,
                                  "alliance": "blue", "pos": [9.0 + 0.3 * i, 5.0],
                                  "person_overlap": False, "confirmed": False})
    frames = list(range(10)) + [120 + i for i in range(12)]
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in frames]
    job.autonomous_paths = {"B1": [{"frame_num": f, "timestamp": f / 60.0,
                                    "field_pos": [15.0, 2.4]} for f in range(10)]
                            + [{"frame_num": 120 + i, "timestamp": (120 + i) / 60.0,
                                "field_pos": [9.0 + 0.3 * i, 5.0]} for i in range(12)]}
    job.robot_tracks = {1: SimpleNamespace(team_number=None, team_number_confirmed=False)}
    job._canonical_identity_slots = lambda: {0: 1}
    job._get_display_team_number = lambda r: None
    job._slot_alliance = lambda sid: "blue"
    job._reresolve_replay_frames()

    by_frame = {f["frame_num"]: f for f in job._replay.frames}
    assert all(len(by_frame[f]["robots"]) == 0 for f in range(10))       # phantom gone
    assert all(len(by_frame[120 + i]["robots"]) == 1 for i in range(12))  # robot kept
    path_frames = [s["frame_num"] for s in job.autonomous_paths["B1"]]
    assert min(path_frames) == 120 and len(path_frames) == 12            # prefix pruned


def _global_resolve_job(knob_on):
    """A reresolve stub with the global_identity_resolve wiring in scope. Slot 0
    (B1) has greedy team '9999' but its team_ocr_votes point at '1111' — with the
    knob on, the exact assignment overrides it (no diag file -> votes fallback)."""
    job = _job()
    job.config = SimpleNamespace(global_identity_resolve=knob_on)
    job.job_id = "test_gir_override_nonexistent"   # diag file absent -> votes fallback
    job.gallery = None
    job._slot_source = {}
    job._resolved_team_by_slot = {}
    job._resolved_confirmed_by_slot = {}
    job.match_blue_teams = ["1111", "2222", "3333"]
    job.match_red_teams = ["4444", "5555", "6666"]
    for f in range(3):
        job._identity_log.append({"frame_num": f, "identity_id": 1, "slot_id": 0,
                                  "alliance": "blue", "pos": [1.0, 4.0],
                                  "person_overlap": False, "confirmed": True})
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in range(3)]
    job.robot_tracks = {1: SimpleNamespace(
        team_number="9999", team_number_confirmed=False,
        team_ocr_votes={"1111": 5.0}, appearance_embedding=None, alliance="blue")}
    job._canonical_identity_slots = lambda: {0: 1}
    job._get_display_team_number = lambda r: r.team_number if r else None
    job._slot_alliance = InferenceJob._slot_alliance
    job._slot_label = InferenceJob._slot_label
    return job


def test_global_resolve_populates_override_dicts_on_change():
    """Knob ON: a slot whose exact-assignment team differs from the greedy live
    value records the resolved team + confirmed in the DB-write override dicts
    (identity_id-keyed), leaving the live robot.team_number untouched."""
    job = _global_resolve_job(knob_on=True)
    job._reresolve_replay_frames()
    assert job._resolved_team_by_slot == {1: "1111"}       # override recorded
    assert job._resolved_confirmed_by_slot == {1: True}    # margin 5.0 >= 2.0
    assert job._slot_source == {1: "global_resolve"}
    assert job.robot_tracks[1].team_number == "9999"       # live value NOT mutated
    # DB-write consultation semantics: override wins for the changed identity,
    # falls back to the live value for any identity without an override.
    r = job.robot_tracks[1]
    assert job._resolved_team_by_slot.get(1, r.team_number) == "1111"
    assert job._resolved_team_by_slot.get(99, r.team_number) == "9999"


def test_global_resolve_off_leaves_override_dicts_empty():
    """Knob OFF: the whole block is skipped, so both override dicts stay empty and
    the DB write falls back to the live team_number (byte-identical to legacy)."""
    job = _global_resolve_job(knob_on=False)
    job._reresolve_replay_frames()
    assert job._resolved_team_by_slot == {}
    assert job._resolved_confirmed_by_slot == {}
    assert job._slot_source == {}
    r = job.robot_tracks[1]
    assert job._resolved_team_by_slot.get(1, r.team_number) == "9999"  # live fallback


def test_reresolve_falls_back_on_catastrophic_drop():
    """Identity-churn can make the rebuild retain almost no rows; the guard must
    fall back to the slice-2 prune rather than ship a gutted replay."""
    job = _job()
    # live forward output: 5 frames x 5 robot rows = 25 rows
    job._replay.frames = [{"frame_num": f, "scores": {},
                           "robots": [{"slot": s} for s in ("B1", "B2", "B3", "R1", "R2")]}
                          for f in range(5)]
    # only ONE identity present in the log, in one frame -> rebuild yields 1 row << 25
    job._identity_log = [{"frame_num": 0, "identity_id": 1, "slot_id": 0, "alliance": "blue",
                          "pos": [1.0, 4.0], "person_overlap": False, "confirmed": True}]
    job.robot_tracks = {1: SimpleNamespace(team_number="254", team_number_confirmed=True)}
    job._canonical_identity_slots = lambda: {0: 1}
    job._get_display_team_number = lambda r: r.team_number if r else None
    job._slot_alliance = lambda sid: "blue" if sid < 3 else "red"
    # fallback path calls _apply_track_verdicts; empty evidence => it keeps all rows
    job._match_evidence = MatchEvidence()
    job._binding_log = []
    job._reresolve_replay_frames()
    # guard tripped: original 25 rows preserved, NOT replaced by the 1-row rebuild
    assert sum(len(f["robots"]) for f in job._replay.frames) == 25

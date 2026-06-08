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

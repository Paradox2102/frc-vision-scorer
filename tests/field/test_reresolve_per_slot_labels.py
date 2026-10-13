"""End-of-job labels from whole-match per-slot evidence (global_identity_resolve,
default ON) through InferenceJob._reresolve_replay_frames.

qm11 full match (80a1cd4c): identities churned (575 in 168 s, 83-130 per
slot), 5 of the 6 final (canonical) identities were born in the last ~10 s
with no votes, and the last-identity labels stamped NULL into all 1,273
replay frames and match_robot_identity — even R2, which had 8 correct reads
spread over 8 identities.
"""
import json
from types import SimpleNamespace


import web.app as app_mod
from web.app import InferenceJob

TRUTH = {0: "4786", 1: "3181", 2: "2992", 3: "6800", 4: "1477", 5: "4256"}
BLUE = ["4786", "3181", "2992"]
RED = ["6800", "1477", "4256"]


# Replay rows are rebuilt only for slots whose whole-match verdict is "robot":
# the churny match moves its robots (>= 10 samples, pos x grows with the
# frame); the short fixtures mark a frame "confirmed" (team-read evidence).


def _job(tmp_path, monkeypatch, knob_on=True):
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", tmp_path)
    job = InferenceJob.__new__(InferenceJob)
    job._identity_log = []
    job._replay = SimpleNamespace(frames=[])
    job.identity_diagnostics = SimpleNamespace(gauge=lambda *a, **k: None)
    job.config = SimpleNamespace(global_identity_resolve=knob_on)
    job.job_id = "per_slot_labels_test"
    job.gallery = None
    job._slot_source = {}
    job._resolved_team_by_slot = {}
    job._resolved_confirmed_by_slot = {}
    job.match_blue_teams = list(BLUE)
    job.match_red_teams = list(RED)
    job.robot_tracks = {}
    job._get_display_team_number = lambda r: r.team_number if r else None
    return job


def _track(team=None, votes=None):
    return SimpleNamespace(team_number=team, team_number_confirmed=False,
                           team_ocr_votes=dict(votes or {}),
                           appearance_embedding=None)


def _write_reads(tmp_path, rows):
    path = tmp_path / "diagnostic_per_slot_labels_test.jsonl"
    with path.open("w") as f:
        f.write("not json\n")                                   # tolerated
        f.write(json.dumps({"type": "snapshot", "frame": 0}) + "\n")
        for r in rows:
            f.write(json.dumps({"type": "vlm_read", "vote_weight": 1.0,
                                "crop_b64": "xx", **r}) + "\n")


def _churny_match(job, tmp_path, per_slot=12, hold=3, noise=None):
    """Each slot is held by `per_slot` short-lived identities (hold frames
    each) that get exactly ONE read of the slot's true team, then by a final
    canonical identity with no votes at all."""
    next_id = 100
    frame_nums = list(range(0, (per_slot + 1) * hold * 8, 8))
    reads, canonical = [], {}
    for sid in range(6):
        for k in range(per_slot + 1):
            iid = next_id
            next_id += 1
            frames = frame_nums[k * hold:(k + 1) * hold]
            for fn in frames:
                job._identity_log.append({
                    "frame_num": fn, "identity_id": iid, "slot_id": sid,
                    "alliance": "blue" if sid < 3 else "red",
                    "pos": [0.1 * fn, 1.0 + sid], "person_overlap": False,
                    "confirmed": False})
            if k < per_slot:
                reads.append({"frame": frames[1], "robot_id": iid,
                              "matched_team": TRUTH[sid]})
            else:
                canonical[sid] = iid
                job.robot_tracks[iid] = _track()                # no votes
    for r in noise or []:
        reads.append(r)
    _write_reads(tmp_path, reads)
    job._replay.frames = [{"frame_num": fn, "scores": {}, "robots": []}
                          for fn in frame_nums]
    job._canonical_identity_slots = lambda: dict(canonical)
    return canonical, frame_nums


def test_churny_slots_get_correct_final_labels(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    # One misread on B1 (a slot-switch artefact) must be outvoted.
    canonical, _ = _churny_match(job, tmp_path, noise=[
        {"frame": 8, "robot_id": 100, "matched_team": "2992"}])
    job._reresolve_replay_frames()

    for sid, iid in canonical.items():
        assert job._resolved_team_by_slot[iid] == TRUTH[sid]
        assert job._resolved_confirmed_by_slot[iid] is True     # 12 vs <=1
        assert job._slot_source[iid] == "global_resolve"
    labels = {r["slot"]: r["team_number"] for fr in job._replay.frames for r in fr["robots"]}
    assert labels == {"B1": "4786", "B2": "3181", "B3": "2992",
                      "R1": "6800", "R2": "1477", "R3": "4256"}
    # every replay row of every frame carries its slot's team
    assert all(r["team_number"] for fr in job._replay.frames for r in fr["robots"])
    assert job._final_slot_meta[0]["team_number"] == "4786"


def test_churny_slots_knob_off_keeps_last_identity_nulls(tmp_path, monkeypatch):
    """GLOBAL_IDENTITY_RESOLVE=0 restores today's behaviour (all NULL)."""
    job = _job(tmp_path, monkeypatch, knob_on=False)
    _churny_match(job, tmp_path)
    job._reresolve_replay_frames()
    assert job._resolved_team_by_slot == {}
    assert all(r["team_number"] is None for fr in job._replay.frames for r in fr["robots"])


def test_canonical_identity_without_votes_gets_slot_label(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    for fn, iid in ((8, 11), (16, 11), (24, 43), (32, 77)):     # 77 = canonical
        job._identity_log.append({"frame_num": fn, "identity_id": iid, "slot_id": 4,
                                  "alliance": "red", "pos": [0.2 * fn, 3.0],
                                  "person_overlap": False, "confirmed": True})
    _write_reads(tmp_path, [{"frame": 8, "robot_id": 11, "matched_team": "1477"},
                            {"frame": 24, "robot_id": 43, "matched_team": "1477"}])
    job.robot_tracks = {77: _track()}
    job._canonical_identity_slots = lambda: {4: 77}
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in (8, 16, 24, 32)]
    job._reresolve_replay_frames()
    assert job._resolved_team_by_slot == {77: "1477"}
    assert job._resolved_confirmed_by_slot == {77: True}      # margin 2.0 >= 2.0
    assert all(fr["robots"][0]["team_number"] == "1477" for fr in job._replay.frames)
    assert job.robot_tracks[77].team_number is None           # live value untouched


def test_dead_identity_label_cannot_survive_on_another_slot(tmp_path, monkeypatch):
    """Collision sweep with a vote-less canonical identity: B3's last identity
    carries a live label (e.g. process-of-elimination) for a team the
    whole-match evidence gives to B1 -> B3 is stripped, never duplicated."""
    job = _job(tmp_path, monkeypatch)
    for fn in (8, 16):
        job._identity_log.append({"frame_num": fn, "identity_id": 5, "slot_id": 0,
                                  "alliance": "blue", "pos": [0.3 * fn, 1],
                                  "person_overlap": False, "confirmed": True})
        job._identity_log.append({"frame_num": fn, "identity_id": 6, "slot_id": 2,
                                  "alliance": "blue", "pos": [0.3 * fn, 3],
                                  "person_overlap": False, "confirmed": True})
    _write_reads(tmp_path, [{"frame": 8, "robot_id": 5, "matched_team": "2992"}])
    job.robot_tracks = {5: _track(), 6: _track(team="2992")}
    job._canonical_identity_slots = lambda: {0: 5, 2: 6}
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in (8, 16)]
    job._reresolve_replay_frames()
    assert job._resolved_team_by_slot == {5: "2992", 6: None}
    by_slot = {r["slot"]: r["team_number"] for r in job._replay.frames[0]["robots"]}
    assert by_slot == {"B1": "2992", "B3": None}


def test_tied_slot_is_left_unassigned_and_live_label_stripped(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch)
    for fn, iid in ((8, 1), (16, 2)):
        job._identity_log.append({"frame_num": fn, "identity_id": iid, "slot_id": 5,
                                  "alliance": "red", "pos": [0.3 * fn, 3],
                                  "person_overlap": False, "confirmed": True})
    _write_reads(tmp_path, [{"frame": 8, "robot_id": 1, "matched_team": "4256"},
                            {"frame": 16, "robot_id": 2, "matched_team": "6800"}])
    job.robot_tracks = {2: _track(team="6800")}
    job._canonical_identity_slots = lambda: {5: 2}
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in (8, 16)]
    job._reresolve_replay_frames()
    assert job._resolved_team_by_slot == {2: None}
    assert all(fr["robots"][0]["team_number"] is None for fr in job._replay.frames)

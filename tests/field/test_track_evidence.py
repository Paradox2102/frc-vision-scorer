import math
from web.field.track_evidence import MatchEvidence, TrackEvidence


def _ev(positions):
    e = TrackEvidence(track_id=1)
    for f, x, y in positions:
        e.add_position(f, x, y)
    return e


def test_net_displacement_zero_for_jittery_static_track():
    import itertools
    jit = itertools.cycle([0.2, -0.15, 0.1, -0.2, 0.05])
    pos = [(f, 10.5 + next(jit), 7.6 + next(jit)) for f in range(40)]
    assert _ev(pos).net_displacement_m(window=5) < 0.5


def test_net_displacement_catches_real_drive():
    pos = [(f, 2.0 + 0.13 * f, 4.0) for f in range(40)]  # 0..~5 m
    assert _ev(pos).net_displacement_m(window=5) > 3.0


def test_net_displacement_robust_to_single_outlier():
    pos = [(f, 3.0, 3.0) for f in range(20)]
    pos[10] = (10, 30.0, 30.0)  # one wild outlier frame
    assert _ev(pos).net_displacement_m(window=5) < 0.5


def test_net_displacement_too_few_samples_is_zero():
    assert _ev([(0, 1.0, 1.0), (1, 2.0, 2.0)]).net_displacement_m(window=5) == 0.0


def test_person_overlap_frac():
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=(f < 4))  # 4 of 10
    assert abs(e.person_overlap_frac() - 0.4) < 1e-9


def test_person_overlap_frac_no_frames_is_zero():
    assert TrackEvidence(track_id=1).person_overlap_frac() == 0.0


def test_first_confident_frame_is_earliest():
    e = TrackEvidence(track_id=1)
    e.observe_frame(5, overlapped_person=False)
    e.confirm(7, team_read=True)
    e.confirm(6, appearance=True)   # earlier confirmation arrives later in code
    assert e.ever_team_read and e.ever_appearance_match
    assert e.first_confident_frame == 6


def test_confirm_bumper_color_is_not_confident_frame():
    e = TrackEvidence(track_id=1)
    e.confirm(3, bumper_color=True)
    assert e.ever_bumper_color is True
    assert e.first_confident_frame is None


def test_match_evidence_aggregates_per_track():
    m = MatchEvidence()
    m.observe(track_id=1, frame=0, field_xy=(2.0, 4.0), overlapped_person=False)
    m.observe(track_id=1, frame=1, field_xy=(2.6, 4.0), overlapped_person=False)
    m.observe(track_id=2, frame=0, field_xy=(8.0, 4.0), overlapped_person=True)
    assert set(m.by_track) == {1, 2}
    assert len(m.by_track[1].field_positions) == 2
    assert m.by_track[2].person_overlap_frames == 1


def test_match_evidence_tolerates_missing_position():
    m = MatchEvidence()
    m.observe(track_id=1, frame=0, field_xy=None, overlapped_person=False)
    assert m.by_track[1].frames_seen == [0]
    assert m.by_track[1].field_positions == []  # coasting frame: no motion sample


def test_match_evidence_confirm_routes_to_track():
    m = MatchEvidence()
    m.observe(track_id=3, frame=2, field_xy=(5.0, 5.0), overlapped_person=False)
    m.confirm(track_id=3, frame=2, team_read=True)
    assert m.by_track[3].ever_team_read
    assert m.by_track[3].first_confident_frame == 2


def test_match_evidence_confirm_before_observe_autocreates():
    m = MatchEvidence()
    m.confirm(track_id=9, frame=1, appearance=True)  # confirmation can precede observe
    assert m.by_track[9].ever_appearance_match


from web.field.track_evidence import classify_track


def _te(**kw):
    e = TrackEvidence(track_id=kw.pop("track_id", 1))
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def test_classify_read_track_is_robot_even_if_stationary():
    e = _te(ever_team_read=True)  # no motion, no positions
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "robot"


def test_classify_moving_ref_is_human_not_robot():
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=True)        # 100% overlap
    for f in range(10):
        e.add_position(f, 2.0 + 0.5 * f, 4.0)             # ~4.5 m of motion
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "human"


def test_classify_moving_robot_is_robot():
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=False)
    for f in range(10):
        e.add_position(f, 2.0 + 0.5 * f, 4.0)
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "robot"


def test_classify_never_moving_no_evidence_is_static():
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=False)
        e.add_position(f, 5.0, 5.0)                        # parked
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "static"


def test_classify_read_beats_person_overlap():
    e = TrackEvidence(track_id=1, ever_appearance_match=True)
    for f in range(10):
        e.observe_frame(f, overlapped_person=True)
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "robot"


from web.field.track_evidence import (
    track_verdicts, slot_label_to_id, prune_replay_frames, MatchEvidence)


def test_slot_label_to_id_roundtrip():
    assert [slot_label_to_id(s) for s in ("B1", "B2", "B3", "R1", "R2", "R3")] == [0, 1, 2, 3, 4, 5]
    assert slot_label_to_id("??") is None


def test_track_verdicts_maps_each_track():
    m = MatchEvidence()
    m.confirm(track_id=1, frame=0, team_read=True)          # robot
    for f in range(10):                                     # human: overlap, no read/motion
        m.observe(track_id=2, frame=f, field_xy=(5.0, 5.0), overlapped_person=True)
    for f in range(10):                                     # static: parked, no evidence
        m.observe(track_id=3, frame=f, field_xy=(8.0, 8.0), overlapped_person=False)
    v = track_verdicts(m, motion_confirm_m=0.75, person_overlap_frac=0.5)
    assert v == {1: "robot", 2: "human", 3: "static"}


def test_prune_replay_frames_drops_non_robot_rows():
    frames = [{"frame_num": 7, "robots": [
        {"slot": "B1", "alliance": "blue"},
        {"slot": "R1", "alliance": "red"},
    ]}]
    binding_log = [{"frame_num": 7, "bindings": {1: 0, 9: 3}}]
    verdicts = {1: "robot", 9: "human"}
    out = prune_replay_frames(frames, binding_log, verdicts)
    assert [r["slot"] for r in out[0]["robots"]] == ["B1"]


def test_prune_keeps_rows_with_unknown_binding():
    frames = [{"frame_num": 1, "robots": [{"slot": "B2", "alliance": "blue"}]}]
    out = prune_replay_frames(frames, [{"frame_num": 1, "bindings": {}}], {})
    assert len(out[0]["robots"]) == 1


from web.field.track_evidence import identity_evidence, classify_track


def test_identity_evidence_aggregates_and_classifies():
    log = []
    # identity 1: confirmed robot, parked (no motion) -> robot via confirmed
    for f in range(6):
        log.append({"frame_num": f, "identity_id": 1, "slot_id": 0, "alliance": "blue",
                    "pos": [2.0, 4.0], "person_overlap": False, "confirmed": (f >= 4)})
    # identity 2: moving ref (overlap, no confirm) -> human
    for f in range(10):
        log.append({"frame_num": f, "identity_id": 2, "slot_id": 3, "alliance": "red",
                    "pos": [2.0 + 0.5 * f, 7.0], "person_overlap": True, "confirmed": False})
    ev = identity_evidence(log)
    assert set(ev) == {1, 2}
    assert ev[1].ever_team_read is True
    assert classify_track(ev[1]) == "robot"
    assert classify_track(ev[2]) == "human"

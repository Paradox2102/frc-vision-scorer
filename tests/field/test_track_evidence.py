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

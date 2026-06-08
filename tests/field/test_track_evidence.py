import math
from web.field.track_evidence import TrackEvidence


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

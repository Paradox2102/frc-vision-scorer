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

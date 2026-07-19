import numpy as np
import pytest

from web.field.smoother import rts_smooth_segment, smooth_history

DT = 0.125  # 8 fps, the default SAM2_TRACK_FPS


def _rec(fn, z, r_scale=1.0, seed=False, corner_only=False, dt=DT):
    return {"frame_num": fn, "dt": dt, "z": z,
            "r_scale": r_scale if z is not None else None,
            "seed": seed, "corner_only": corner_only}


def _line_records(n, v=1.0, y=1.0):
    """Noiseless 1D motion along x at v m/s, one record per DT."""
    recs = [_rec(0, (0.0, y), seed=True)]
    recs += [_rec(k, (v * k * DT, y)) for k in range(1, n)]
    return recs


def test_noiseless_line_recovered():
    out = rts_smooth_segment(_line_records(40))
    assert len(out) == 40
    for o in out:
        truth_x = o["frame_num"] * DT
        assert abs(o["pos"][0] - truth_x) < 0.05
        assert abs(o["pos"][1] - 1.0) < 0.05


def test_gap_interpolates_two_sided():
    # measured 0..16, coast 17..32 (2s gap), measured 33..48, all on x=t line
    recs = _line_records(17)
    recs += [_rec(k, None) for k in range(17, 33)]
    recs += [_rec(k, (k * DT, 1.0)) for k in range(33, 49)]
    out = rts_smooth_segment(recs)
    assert len(out) == 49
    mid = next(o for o in out if o["frame_num"] == 25)   # t = 3.125 s
    assert mid["measured"] is False
    assert abs(mid["pos"][0] - 25 * DT) < 0.15
    assert abs(mid["pos"][1] - 1.0) < 0.10
    # A causal coast with velocity decay 0.85 would have stalled ~1 m short
    # by mid-gap; two-sided smoothing must not.


def test_trailing_coast_truncated():
    recs = _line_records(11) + [_rec(k, None) for k in range(11, 19)]
    out = rts_smooth_segment(recs)
    assert len(out) == 11
    assert out[-1]["frame_num"] == 10


def test_single_record_segment():
    out = rts_smooth_segment([_rec(5, (2.0, 3.0), seed=True)])
    assert len(out) == 1
    assert out[0]["pos"] == pytest.approx((2.0, 3.0))


def test_high_r_scale_measurement_downweighted():
    # One wild corner measurement (r_scale=8) mid-line gets pulled toward the line.
    recs = _line_records(21)
    recs[10] = _rec(10, (10 * DT + 2.0, 1.0), r_scale=8.0, corner_only=True)
    out = rts_smooth_segment(recs)
    o = next(o for o in out if o["frame_num"] == 10)
    assert abs(o["pos"][0] - 10 * DT) < 0.6          # << the 2.0 m raw error
    assert o["corner_only"] is True


def test_smooth_history_shapes_and_none_frames():
    seg = [_rec(None, (0.0, 1.0), seed=True)] + [_rec(k, (k * DT, 1.0))
                                                 for k in range(1, 5)]
    result = smooth_history({3: [seg]})
    assert set(result.keys()) == {3}
    assert set(result[3].keys()) == {1, 2, 3, 4}      # frame None omitted
    assert result[3][4]["measured"] is True

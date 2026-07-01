"""Tests for gap-aware path quality: segmentation, honest length, coverage.

Our stored auto paths record only *visible* frames, so a tracking dropout
(occlusion / lost track) becomes a large timestamp gap between consecutive
samples. Consumers that treat the samples as one polyline fabricate
straight-line motion across the gap. These helpers let consumers break the
path at gaps and quantify how much of the window was actually tracked.
"""
import numpy as np

from web.field.path_quality import gap_segments, tracked_length_m, coverage


# --------------------------------------------------------------------------- #
# gap_segments
# --------------------------------------------------------------------------- #
def test_gap_segments_uniform_cadence_is_one_segment():
    ts = [0.0, 0.13, 0.26, 0.39, 0.52]
    assert gap_segments(ts, max_gap_s=1.0) == [(0, 5)]


def test_gap_segments_splits_at_a_large_gap():
    # ~0.13 s cadence with a 4.87 s dropout between index 1 and 2.
    ts = [0.0, 0.13, 5.0, 5.13]
    assert gap_segments(ts, max_gap_s=1.0) == [(0, 2), (2, 4)]


def test_gap_segments_multiple_gaps():
    ts = [0.0, 0.1, 3.0, 3.1, 3.2, 9.0]
    assert gap_segments(ts, max_gap_s=1.0) == [(0, 2), (2, 5), (5, 6)]


def test_gap_segments_empty_and_single():
    assert gap_segments([], max_gap_s=1.0) == []
    assert gap_segments([2.5], max_gap_s=1.0) == [(0, 1)]


# --------------------------------------------------------------------------- #
# tracked_length_m — polyline length that does NOT bridge gaps
# --------------------------------------------------------------------------- #
def test_tracked_length_excludes_gap_bridge():
    # Two 1 m runs separated by a 9 m straight "bridge" across a 4.9 s gap.
    pts = [(0, 0), (1, 0), (10, 0), (11, 0)]
    ts = [0.0, 0.1, 5.0, 5.1]
    # Naive polyline would be 1 + 9 + 1 = 11 m; honest length drops the 9 m bridge.
    assert tracked_length_m(pts, ts, max_gap_s=1.0) == 2.0


def test_tracked_length_no_gaps_equals_full_polyline():
    pts = [(0, 0), (3, 0), (3, 4)]
    ts = [0.0, 0.1, 0.2]
    assert tracked_length_m(pts, ts, max_gap_s=1.0) == 7.0  # 3 + 4


# --------------------------------------------------------------------------- #
# coverage — how much of the window was actually tracked
# --------------------------------------------------------------------------- #
def test_coverage_reports_gaps_and_fraction():
    # 3 samples, then a 4.8 s dropout, then 2 samples. Window is 6 s long.
    ts = [0.0, 0.1, 0.2, 5.0, 5.1]
    c = coverage(ts, max_gap_s=1.0, window_s=6.0)
    assert c["n_samples"] == 5
    assert c["n_gaps"] == 1
    assert abs(c["max_gap_s"] - 4.8) < 1e-9
    # tracked time = 0.2 s (first run) + 0.1 s (second run) = 0.3 s of 6 s
    assert abs(c["tracked_s"] - 0.3) < 1e-9
    assert abs(c["coverage_frac"] - 0.3 / 6.0) < 1e-9


def test_coverage_full_window_is_one():
    ts = list(np.arange(0.0, 5.0001, 0.1))
    c = coverage(ts, max_gap_s=1.0, window_s=5.0)
    assert c["n_gaps"] == 0
    assert abs(c["coverage_frac"] - 1.0) < 1e-6


def test_coverage_defaults_denominator_to_span():
    ts = [10.0, 10.1, 10.2]  # no window given -> fraction over the sample span
    c = coverage(ts, max_gap_s=1.0)
    assert c["n_gaps"] == 0
    assert abs(c["coverage_frac"] - 1.0) < 1e-9

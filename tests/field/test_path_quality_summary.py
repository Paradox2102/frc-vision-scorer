"""quality_summary aggregates the per-sample quality signals now attached to
auto-path samples (tracker_prob / corner_only / occluded), tolerating legacy
samples that predate those keys by excluding them from prob aggregates and
reporting the counts honestly.
"""
import math

from web.field.path_quality import quality_summary


def test_quality_summary_all_new_samples():
    samples = [
        {"field_pos": [0, 0], "timestamp": 0.0, "tracker_prob": 0.9,
         "corner_only": False, "occluded": None},
        {"field_pos": [1, 0], "timestamp": 0.1, "tracker_prob": 0.3,
         "corner_only": True, "occluded": "hub"},
        {"field_pos": [2, 0], "timestamp": 0.2, "tracker_prob": 0.7,
         "corner_only": False, "occluded": None},
    ]
    q = quality_summary(samples)
    assert q["n_with_prob"] == 3
    assert abs(q["mean_tracker_prob"] - (0.9 + 0.3 + 0.7) / 3) < 1e-3
    assert q["min_tracker_prob"] == 0.3
    assert abs(q["frac_low_conf"] - 1 / 3) < 1e-3   # only 0.3 < 0.5
    assert abs(q["frac_corner_only"] - 1 / 3) < 1e-3
    assert abs(q["frac_occluded"] - 1 / 3) < 1e-3


def test_quality_summary_excludes_legacy_from_prob_aggregate():
    # Two legacy samples (no keys) + two new ones. Prob aggregates only span
    # the two that carry a prob; fractions over all samples for the flags they
    # do carry.
    samples = [
        {"field_pos": [0, 0], "timestamp": 0.0},                       # legacy
        {"field_pos": [1, 0], "timestamp": 0.1},                       # legacy
        {"field_pos": [2, 0], "timestamp": 0.2, "tracker_prob": 0.8,
         "corner_only": False, "occluded": None},
        {"field_pos": [3, 0], "timestamp": 0.3, "tracker_prob": 0.4,
         "corner_only": True, "occluded": "bumper"},
    ]
    q = quality_summary(samples)
    assert q["n_samples"] == 4
    assert q["n_with_prob"] == 2
    assert abs(q["mean_tracker_prob"] - 0.6) < 1e-9
    assert q["min_tracker_prob"] == 0.4
    assert abs(q["frac_low_conf"] - 0.5) < 1e-9    # 0.4 of the 2 with prob


def test_quality_summary_all_legacy_reports_none():
    samples = [
        {"field_pos": [0, 0], "timestamp": 0.0},
        {"field_pos": [1, 0], "timestamp": 0.1},
    ]
    q = quality_summary(samples)
    assert q["n_samples"] == 2
    assert q["n_with_prob"] == 0
    assert q["mean_tracker_prob"] is None
    assert q["min_tracker_prob"] is None
    assert q["frac_low_conf"] is None
    # Flags default to False when absent, so the fractions are honestly 0.0.
    assert q["frac_corner_only"] == 0.0
    assert q["frac_occluded"] == 0.0


def test_quality_summary_empty():
    q = quality_summary([])
    assert q["n_samples"] == 0
    assert q["n_with_prob"] == 0
    assert q["mean_tracker_prob"] is None


def test_quality_summary_prob_none_treated_as_missing():
    # An explicit tracker_prob=None (slot tracked but SAM gave no prob this
    # frame) is excluded from prob aggregates, same as a missing key.
    samples = [
        {"field_pos": [0, 0], "timestamp": 0.0, "tracker_prob": None,
         "corner_only": True, "occluded": None},
        {"field_pos": [1, 0], "timestamp": 0.1, "tracker_prob": 0.6,
         "corner_only": False, "occluded": None},
    ]
    q = quality_summary(samples)
    assert q["n_with_prob"] == 1
    assert q["mean_tracker_prob"] == 0.6
    assert abs(q["frac_corner_only"] - 0.5) < 1e-9

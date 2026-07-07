"""Quality-aware Manta metrics: confidence-weighted nearest-neighbor error and
confidence-scaled coverage, added ALONGSIDE (never replacing) the existing
unweighted metrics. The point is that deleting tracked points must not silently
improve the score — low-confidence points still count, weighted by tracker_prob.

Legacy replays (samples without tracker_prob) must weight every point 1.0, so
weighted == unweighted and nothing regresses for historical comparability.
"""
import numpy as np

from scripts.compare_manta import weighted_mean_nn_m, quality_metrics


def _samples(pts, probs=None):
    out = []
    for i, p in enumerate(pts):
        s = {"timestamp": round(0.1 * i, 4), "field_pos": [float(p[0]), float(p[1])]}
        if probs is not None:
            s["tracker_prob"] = probs[i]
        out.append(s)
    return out


def test_weighted_equals_unweighted_when_all_probs_none():
    # Legacy: no tracker_prob anywhere -> every weight 1.0 -> weighted == the
    # plain directed mean nearest-neighbor distance.
    ours = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    truth = np.array([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]])
    samples = _samples(ours)  # no probs
    w = weighted_mean_nn_m(ours, truth, samples)
    d = np.linalg.norm(ours[:, None, :] - truth[None, :, :], axis=2).min(axis=1)
    assert abs(w - float(d.mean())) < 1e-9   # unweighted mean == 0.5


def test_weighted_downweights_a_low_conf_outlier():
    # Three points on-truth (nn 0.0) plus one wild low-confidence outlier far
    # away. Unweighted mean is dragged up by the outlier; weighting by
    # tracker_prob (outlier prob ~0) pulls the weighted mean back toward 0.
    ours = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [50.0, 50.0]])
    truth = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    samples = _samples(ours, probs=[0.9, 0.9, 0.9, 0.01])
    nn = np.linalg.norm(ours[:, None, :] - truth[None, :, :], axis=2).min(axis=1)
    unweighted = float(nn.mean())
    weighted = weighted_mean_nn_m(ours, truth, samples)
    assert weighted < unweighted                 # right direction
    assert weighted < 1.0                          # outlier nearly ignored


def test_weighted_mean_nn_missing_prob_weights_one():
    # Mixed: some samples carry a prob, some don't. Missing ones weight 1.0.
    ours = np.array([[0.0, 0.0], [10.0, 0.0]])
    truth = np.array([[0.0, 0.0]])   # nn: 0.0 for first, 10.0 for second
    samples = [
        {"timestamp": 0.0, "field_pos": [0.0, 0.0], "tracker_prob": 0.5},
        {"timestamp": 0.1, "field_pos": [10.0, 0.0]},  # legacy -> weight 1.0
    ]
    w = weighted_mean_nn_m(ours, truth, samples)
    # weights (0.5, 1.0), nn (0.0, 10.0) -> (0.5*0 + 1.0*10)/(1.5) = 6.667
    assert abs(w - (10.0 / 1.5)) < 1e-6


def test_quality_metrics_legacy_weighted_equals_unweighted():
    ours = np.array([[0.0, 0.0], [1.0, 0.0]])
    truth = np.array([[0.0, 0.5], [1.0, 0.5]])
    samples = _samples(ours)  # legacy, no probs
    q = quality_metrics(samples, ours, truth)
    assert abs(q["weighted_mean_nn_m"] - 0.5) < 1e-9
    # No probs -> coverage_conf is None (nothing to scale coverage by).
    assert q["coverage_conf"] is None
    assert q["n_with_prob"] == 0


def test_quality_metrics_coverage_conf_scales_coverage():
    ours = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    truth = ours.copy()
    samples = _samples(ours, probs=[0.8, 0.8, 0.8])
    q = quality_metrics(samples, ours, truth, coverage_frac=1.0)
    # coverage_conf = coverage_frac (1.0) * mean prob (0.8)
    assert abs(q["coverage_conf"] - 0.8) < 1e-9
    assert abs(q["mean_tracker_prob"] - 0.8) < 1e-9

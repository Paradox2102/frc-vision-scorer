import numpy as np

from scripts.compare_manta import (
    compare_paths,
    match_number_from_key,
    slot_to_alliance_pos,
    summarize_our_path,
)


def test_slot_to_alliance_pos():
    assert slot_to_alliance_pos("B1") == ("blue", 1)
    assert slot_to_alliance_pos("R3") == ("red", 3)
    assert slot_to_alliance_pos("X9") is None
    assert slot_to_alliance_pos("") is None


def test_match_number_from_key():
    assert match_number_from_key("2026cur_qm2") == ("qual", 2)
    assert match_number_from_key("2026cur_qm115") == ("qual", 115)
    assert match_number_from_key("2026cur_sf3m1") == ("sf", 3)
    assert match_number_from_key("garbage") is None


def test_compare_paths_identical_is_zero():
    p = np.array([[0, 0], [1, 0], [2, 1]], float)
    m = compare_paths(p, p)
    assert m.mean_nn_m == 0.0
    assert m.hausdorff_m == 0.0
    assert m.start_err_m == 0.0
    assert m.end_err_m == 0.0


def test_compare_paths_constant_offset():
    a = np.array([[0, 0], [1, 0], [2, 0]], float)
    b = a + np.array([0.0, 0.5])  # shifted 0.5 m in y
    m = compare_paths(a, b)
    # Every point's nearest neighbor is the matching shifted point: 0.5 m.
    assert abs(m.mean_nn_m - 0.5) < 1e-6
    assert abs(m.hausdorff_m - 0.5) < 1e-6
    assert abs(m.start_err_m - 0.5) < 1e-6


def test_compare_paths_handles_different_sampling():
    # Same straight line, different point counts. The symmetric mean NN stays
    # small (bounded by the coarse spacing), and lengths match exactly — the
    # metric is robust to the two sources sampling at different rates.
    a = np.array([[x, 0.0] for x in np.linspace(0, 10, 5)])
    b = np.array([[x, 0.0] for x in np.linspace(0, 10, 50)])
    m = compare_paths(a, b)
    assert m.mean_nn_m < 0.5  # off-line error is zero; only along-line gaps remain
    assert m.hausdorff_m < 1.5
    assert m.len_ours_m == 10.0 and m.len_truth_m == 10.0


def test_summarize_our_path_flags_dropout():
    # Two short tracked runs bridged by a 9 m straight line across a 4.9 s
    # dropout: the naive length counts the bridge, the honest length does not,
    # and coverage exposes the gap that a raw length comparison would hide.
    samples = [
        {"timestamp": 0.0, "field_pos": [0.0, 0.0]},
        {"timestamp": 0.1, "field_pos": [1.0, 0.0]},
        {"timestamp": 5.0, "field_pos": [10.0, 0.0]},
        {"timestamp": 5.1, "field_pos": [11.0, 0.0]},
    ]
    s = summarize_our_path(samples, max_gap_s=1.0)
    assert s["naive_len_m"] == 11.0
    assert s["tracked_len_m"] == 2.0
    assert s["n_gaps"] == 1
    assert abs(s["max_gap_s"] - 4.9) < 1e-6


def _samples(t0, t1, xy=(1.0, 1.0), step=0.2):
    ts = np.arange(t0, t1 + 1e-9, step)
    return [{"timestamp": round(float(t), 4), "field_pos": [xy[0] + 0.01 * i, xy[1]]}
            for i, t in enumerate(ts)]


def test_assign_alliance_reports_start_lag_and_window_coverage():
    """A slot that starts 5 s into the match must NOT score 100% coverage:
    coverage is measured against the whole-match window (earliest to latest
    sample across all slots), and the late start is reported as start_lag_s.
    qm11: R1 started 5.1 s late yet scored coverage_frac=1.0, hiding the
    untracked opening sweep."""
    from scripts.compare_manta import assign_alliance
    path = np.array([[1.0, 1.0], [1.5, 1.0]], float)
    ours = {
        (11, "red", 1): {"replay_id": "x", "match_key": "t_qm11", "slot": "R1",
                         "path_m": path, "samples": _samples(7.5, 27.5)},
        (11, "red", 2): {"replay_id": "x", "match_key": "t_qm11", "slot": "R2",
                         "path_m": path, "samples": _samples(12.5, 27.5)},
    }
    truth = {
        (11, "red", 1): {"team_number": 111, "path_m": path},
        (11, "red", 2): {"team_number": 222, "path_m": path + 5.0},
    }
    pairs = assign_alliance(ours, {False: truth, True: {}}, 11, "red", "off")
    by_slot = {p["our_slot"]: p for p in pairs}
    assert abs(by_slot["R1"]["start_lag_s"] - 0.0) < 1e-6
    assert abs(by_slot["R2"]["start_lag_s"] - 5.0) < 1e-6
    # R2 tracked 15 of the 20 s window -> 75%, not 100%.
    assert abs(by_slot["R1"]["coverage_frac"] - 1.0) < 0.02
    assert abs(by_slot["R2"]["coverage_frac"] - 0.75) < 0.02

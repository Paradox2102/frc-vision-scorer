import numpy as np

from scripts.compare_manta import (
    compare_paths,
    match_number_from_key,
    slot_to_alliance_pos,
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

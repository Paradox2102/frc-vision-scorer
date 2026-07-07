import numpy as np
from web.field.occlusion_portal import predict_exit, path_crosses_footprint

SQUARE = np.array([[4.0, 2.0], [6.0, 2.0], [6.0, 4.0], [4.0, 4.0]])


def test_predict_exit_straight_through():
    # Track at the west edge moving +x at 2 m/s exits near the east edge.
    res = predict_exit((4.1, 3.0), (2.0, 0.0), SQUARE, margin_m=0.0)
    assert res is not None
    (ex, ey), eta = res
    assert abs(ex - 6.0) < 0.15 and abs(ey - 3.0) < 0.15
    assert abs(eta - (6.0 - 4.1) / 2.0) < 0.15


def test_predict_exit_stationary_returns_none():
    assert predict_exit((5.0, 3.0), (0.05, 0.0), SQUARE) is None


def test_predict_exit_pointing_away_returns_none():
    # Already outside and moving away: never inside the footprint again.
    assert predict_exit((8.0, 3.0), (2.0, 0.0), SQUARE, margin_m=0.0) is None


def test_path_crosses_footprint():
    assert path_crosses_footprint((3.0, 3.0), (7.0, 3.0), SQUARE, margin_m=0.0)
    assert not path_crosses_footprint((3.0, 5.5), (7.0, 5.5), SQUARE, margin_m=0.0)

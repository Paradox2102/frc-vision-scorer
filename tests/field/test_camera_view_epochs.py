import numpy as np
from web.field import store
from web.field.calibration import CameraCalibration


def _cal(tc):
    c = CameraCalibration(K=np.eye(3), D=np.zeros(4), rvec=np.zeros((3, 1)),
                          tvec=np.ones((3, 1)), image_size=(1920, 1080),
                          field_season="2026-rebuilt-welded", mode="pinhole")
    c.robot_track_top_crop = tc
    return c


def test_load_views_selects_requested_version(tmp_path):
    store.UPLOAD_DIR = tmp_path
    v1 = store.save_version("evt", "main", _cal(0.1), start_match="evt_qm1")
    v2 = store.save_version("evt", "main", _cal(0.4), start_match="evt_qm50")
    from web.field.camera_view import load_views
    main = next(v for v in load_views("evt", selection={"main": v1}) if v.name == "main")
    assert abs(main.calibration.robot_track_top_crop - 0.1) < 1e-9


def test_load_views_default_uses_latest_version(tmp_path):
    store.UPLOAD_DIR = tmp_path
    store.save_version("evt", "main", _cal(0.1), start_match="evt_qm1")
    store.save_version("evt", "main", _cal(0.4), start_match="evt_qm50")
    from web.field.camera_view import load_views
    main = next(v for v in load_views("evt") if v.name == "main")
    # latest (most recent created_at) -> 0.4
    assert abs(main.calibration.robot_track_top_crop - 0.4) < 1e-9


def test_load_views_omits_uncalibrated_cameras(tmp_path):
    store.UPLOAD_DIR = tmp_path
    # no calibration saved for any camera
    from web.field.camera_view import load_views
    assert load_views("evt") == []

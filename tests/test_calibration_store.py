import numpy as np
from web.field.calibration import CameraCalibration
from web.field import store


def _cal():
    return CameraCalibration(
        K=np.eye(3), D=np.zeros(4), rvec=np.zeros((3,1)), tvec=np.ones((3,1)),
        image_size=(1920,1080), crop_region=(0,0,1920,724),
        field_season="2026-rebuilt-welded", rms_error=2.5)


def test_json_roundtrip():
    c = _cal()
    c2 = CameraCalibration.from_json(c.to_json())
    np.testing.assert_allclose(c2.K, c.K)
    np.testing.assert_allclose(c2.tvec, c.tvec)
    assert c2.image_size == (1920,1080)
    assert c2.crop_region == (0,0,1920,724)
    assert c2.rms_error == 2.5


def test_store_save_load(tmp_path):
    store.UPLOAD_DIR = tmp_path
    store.save("evt1", "main", _cal())
    loaded = store.load("evt1", "main")
    assert loaded is not None
    np.testing.assert_allclose(loaded.tvec, np.ones((3,1)))
    assert store.load("evt1", "missing") is None


def test_load_intrinsics_only(tmp_path):
    store.UPLOAD_DIR = tmp_path
    store.save("evt1", "main", _cal())
    K, D = store.load_intrinsics("evt1", "main")
    np.testing.assert_allclose(K, np.eye(3))

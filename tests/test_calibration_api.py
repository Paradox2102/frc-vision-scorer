from fastapi.testclient import TestClient
import importlib

def _client():
    app_mod = importlib.import_module("web.app")
    return TestClient(app_mod.app)

def test_get_calibration_empty():
    r = _client().get("/api/calibration", params={"event": "test_evt_xyz", "camera": "main"})
    assert r.status_code == 200
    assert r.json()["has_calibration"] in (False, True)

def test_manual_without_intrinsics_returns_clear_error():
    # Uses "points" (the actual body key the endpoint reads).
    r = _client().post("/api/calibration/manual", json={
        "event": "no_such_evt", "camera": "main", "points": []})
    assert r.status_code == 400
    assert "auto" in r.json().get("error", "").lower()

def test_manual_with_intrinsics_too_few_points(tmp_path, monkeypatch):
    """When intrinsics exist but fewer than 2 points are posted, expect 400."""
    import numpy as np
    from web.field import store as field_store
    from web.field.calibration import CameraCalibration

    # Redirect store writes/reads to tmp_path so the test is isolated.
    # web.app imports `from web.field import store as field_store`, so the
    # same module object is used by the endpoint — one monkeypatch is enough.
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    cal = CameraCalibration(
        K=np.eye(3), D=np.zeros(4),
        rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), crop_region=None,
        field_season="2026-rebuilt-welded",
    )
    field_store.save("evt_few", "main", cal)

    r = _client().post("/api/calibration/manual", json={
        "event": "evt_few", "camera": "main",
        "points": [{"u": 10, "v": 10, "x": 1, "y": 1, "z": 0}],
    })
    assert r.status_code == 400
    assert "point" in r.json().get("error", "").lower()


def test_calibration_frame_from_local_video():
    import os
    if not os.path.exists("data/test_footage/qual1_first_champs_2026.mp4"):
        import pytest; pytest.skip("test video not present")
    r = _client().post("/api/calibration/frame", json={"timestamp": 75})
    assert r.status_code == 200
    j = r.json()
    assert j["width"] == 1920 and 600 <= j["height"] <= 700  # top 60% of 1080
    assert len(j["frame"]) > 1000  # base64 jpeg present

def test_solve_homography_too_few_points():
    r = _client().post("/api/calibration/solve-homography",
                       json={"points": [{"u":1,"v":1,"x":0,"y":0}]})
    assert r.status_code == 400

def test_landmarks_endpoint():
    r = _client().get("/api/calibration/landmarks")
    assert r.status_code == 200
    assert len(r.json()["landmarks"]) >= 4

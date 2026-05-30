from fastapi.testclient import TestClient
import importlib
import numpy as np
from web.field import store as field_store
from web.field.calibration import CameraCalibration
from web.field.composite_layout import CameraLayout
from web.field.tracker import FieldDetection
import supervision as sv

def _client():
    app_mod = importlib.import_module("web.app")
    return TestClient(app_mod.app)

def test_post_and_get_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    client = _client()
    
    # 1. Post masks
    r_post = client.post("/api/calibration/masks", json={
        "event": "evt_mask",
        "camera": "corner_left",
        "masks": [[10, 20, 100, 200]]
    })
    assert r_post.status_code == 200
    assert r_post.json()["status"] == "ok"
    
    # 2. Retrieve calibration and verify masks are present
    r_get = client.get("/api/calibration", params={"event": "evt_mask", "camera": "corner_left"})
    assert r_get.status_code == 200
    assert r_get.json()["has_calibration"] is True
    assert r_get.json()["masks"] == [[10, 20, 100, 200]]

def test_post_and_get_robot_track_top_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    client = _client()

    r_post = client.post("/api/calibration/robot-track-top-crop", json={
        "event": "evt_crop",
        "camera": "main",
        "robot_track_top_crop": 0.3,
    })
    assert r_post.status_code == 200
    assert r_post.json()["status"] == "ok"
    assert r_post.json()["robot_track_top_crop"] == 0.3

    r_get = client.get("/api/calibration", params={"event": "evt_crop", "camera": "main"})
    assert r_get.status_code == 200
    assert r_get.json()["robot_track_top_crop"] == 0.3


def test_robot_track_top_crop_clamped_to_half(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    client = _client()
    r = client.post("/api/calibration/robot-track-top-crop", json={
        "event": "evt_clamp", "camera": "main", "robot_track_top_crop": 0.9,
    })
    assert r.status_code == 200
    assert r.json()["robot_track_top_crop"] == 0.5  # clamped


def test_save_top_crop_preserves_existing_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    client = _client()
    cal_init = CameraCalibration(
        mode="homography", H=np.eye(3), image_size=(1920, 1080),
        field_season="2026-rebuilt-welded", masks=[(1, 2, 3, 4)],
    )
    field_store.save("evt_keep", "main", cal_init)
    client.post("/api/calibration/robot-track-top-crop", json={
        "event": "evt_keep", "camera": "main", "robot_track_top_crop": 0.25,
    })
    saved = field_store.load("evt_keep", "main")
    assert saved.masks == [(1, 2, 3, 4)]          # masks untouched
    assert saved.robot_track_top_crop == 0.25


def test_solve_tags_preserves_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    client = _client()

    # Pre-save a calibration containing masks
    cal_init = CameraCalibration(
        K=np.eye(3), D=np.zeros(4),
        rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), crop_region=None,
        field_season="2026-rebuilt-welded",
        masks=[(10, 20, 100, 200)]
    )
    field_store.save("evt_tags", "main", cal_init)
    
    # Prepare mock tags project inputs to solve
    from web.field.field_model import FieldModel
    from tests._field_synth import truth_pinhole_camera, project_tag_corners
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)
    auto = {str(tid): corr[tid][0].tolist() for tid in corr}
    
    r = client.post("/api/calibration/solve-tags", json={
        "event": "evt_tags", "camera": "main", "season": "2026-rebuilt-welded",
        "image_size": list(cam.image_size), "auto_tags": auto, "manual_tags": []})
    assert r.status_code == 200
    
    # Reload and verify masks are preserved
    saved = field_store.load("evt_tags", "main")
    assert saved is not None
    assert saved.masks == [(10, 20, 100, 200)]

def test_solve_homography_preserves_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    client = _client()
    
    cal_init = CameraCalibration(
        mode="homography", H=np.eye(3),
        image_size=(1920, 1080), field_season="2026-rebuilt-welded",
        masks=[(50, 60, 10, 20)]
    )
    field_store.save("evt_homog", "main", cal_init)
    
    # 4 points to solve homography
    pts = [
        {"u": 100, "v": 100, "x": 0.0, "y": 0.0},
        {"u": 200, "v": 100, "x": 1.0, "y": 0.0},
        {"u": 200, "v": 200, "x": 1.0, "y": 1.0},
        {"u": 100, "v": 200, "x": 0.0, "y": 1.0}
    ]
    
    import cv2
    class MockCap:
        def __init__(self, *args, **kwargs):
            pass
        def get(self, *args):
            return 30.0
        def set(self, *args):
            return True
        def read(self, *args):
            return True, np.zeros((723, 1920, 3), dtype=np.uint8)
        def release(self, *args):
            pass
    monkeypatch.setattr(cv2, "VideoCapture", MockCap)
    
    r = client.post("/api/calibration/solve-homography", json={
        "event": "evt_homog", "camera": "main", "points": pts, "image_size": [1920, 1080], "timestamp": 75})
    assert r.status_code == 200
    
    saved = field_store.load("evt_homog", "main")
    assert saved is not None
    assert saved.masks == [(50, 60, 10, 20)]

def test_yolo_detections_filtering_by_exclusion_masks(monkeypatch):
    app_mod = importlib.import_module("web.app")
    
    # Mock a CameraView with a custom mask at [5, 5, 2, 2]
    from web.field.camera_view import CameraView
    cal = CameraCalibration(mode="homography", H=np.eye(3), image_size=(1920, 1080))
    view = CameraView(name="main", crop=(0, 0, 1920, 723), masks=((5, 5, 2, 2),), calibration=cal)
    
    # Create two detections:
    # 1. Center at (5.5, 5.5) -> inside mask, should be filtered out
    # 2. Center at (1.5, 1.5) -> outside mask, should be kept
    xyxy = np.array([
        [5.0, 5.0, 6.0, 6.0],
        [1.0, 1.0, 2.0, 2.0]
    ], dtype=np.float32)
    class_id = np.array([1, 1], dtype=np.int32)
    dets = sv.Detections(xyxy=xyxy, class_id=class_id)
    
    # Instantiate InferenceJob and mock details
    job = app_mod.InferenceJob.__new__(app_mod.InferenceJob)
    job.person_model = lambda *args, **kwargs: [sv.Detections.empty()]
    monkeypatch.setattr(sv.Detections, "from_ultralytics", lambda results: results if isinstance(results, sv.Detections) else sv.Detections.empty())
    job.field_dims = (16.54, 8.07)
    job.detect_bumper_color = lambda *args: "red"
    
    # Run _detect_view
    paired, person_boxes = job._detect_view(view, np.zeros((723, 1920, 3), np.uint8), precomputed_dets=dets)
    
    assert len(paired) == 1
    # Verify the only remaining detection is the one at (1.0, 1.0, 2.0, 2.0)
    assert paired[0][0].box == (1.0, 1.0, 2.0, 2.0)

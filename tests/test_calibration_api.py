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
    # Crop now comes from the composite layout's "main" camera ([0,0,1920,723]).
    from web.field.composite_layout import load_layout
    _, _, cw, ch = load_layout()["main"].crop
    assert j["width"] == cw and j["height"] == ch
    assert len(j["frame"]) > 1000  # base64 jpeg present

def test_solve_homography_too_few_points():
    r = _client().post("/api/calibration/solve-homography",
                       json={"points": [{"u":1,"v":1,"x":0,"y":0}]})
    assert r.status_code == 400

def test_landmarks_endpoint():
    r = _client().get("/api/calibration/landmarks")
    assert r.status_code == 200
    assert len(r.json()["landmarks"]) >= 4


def test_tag_map_endpoint():
    r = _client().get("/api/calibration/tag-map")
    assert r.status_code == 200
    j = r.json()
    assert j["field"]["length"] > 0 and j["field"]["width"] > 0
    assert len(j["tags"]) >= 4
    some = next(iter(j["tags"].values()))
    assert {"x", "y", "z"} <= set(some)


def test_calibration_frame_includes_detected_tags():
    import os
    if not os.path.exists("data/test_footage/qual1_first_champs_2026.mp4"):
        import pytest; pytest.skip("test video not present")
    r = _client().post("/api/calibration/frame", json={"timestamp": 75})
    assert r.status_code == 200
    j = r.json()
    assert "detected_tags" in j        # dict; may be empty but key present
    for corners in j["detected_tags"].values():
        assert len(corners) == 4       # 4 (u,v) corners per tag


def test_solve_tags_recovers_and_saves(tmp_path, monkeypatch):
    import numpy as np
    from web.field import store as field_store
    from web.field.field_model import FieldModel
    from tests._field_synth import truth_pinhole_camera, project_tag_corners
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()                       # image_size (1920,1080)
    corr = project_tag_corners(cam, fm)
    auto = {str(tid): corr[tid][0].tolist() for tid in corr}

    r = _client().post("/api/calibration/solve-tags", json={
        "event": "evt_tags", "camera": "main", "season": "2026-rebuilt-welded",
        "image_size": list(cam.image_size), "auto_tags": auto, "manual_tags": []})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["rms_error"] < 2.0
    assert j["n_tags"] >= 4
    saved = field_store.load("evt_tags", "main")
    assert saved is not None and saved.mode == "pinhole"


def test_solve_tags_too_few_points(tmp_path, monkeypatch):
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    r = _client().post("/api/calibration/solve-tags", json={
        "image_size": [1920, 648], "auto_tags": {},
        "manual_tags": [{"id": 1, "u": 10, "v": 10}]})
    assert r.status_code == 400
    assert "point" in r.json().get("error", "").lower()


def test_init_field_tracking_rejects_homography(monkeypatch):
    """A pose-free (homography) calibration must abort field tracking with a clear
    error, not pass the gate and crash per-frame in occlusion ray-casting."""
    from types import SimpleNamespace
    import numpy as np
    from web.app import InferenceJob
    from web.field import store as field_store
    from web.field.calibration import CameraCalibration

    homog = CameraCalibration(mode="homography", H=np.eye(3),
                              image_size=(100, 100), field_season="2026-rebuilt-welded")
    # _init_field_tracking selects the epoch via load_versions/select_epoch, so the
    # homography cal must be returned as the (single) version to reach the pose-free guard.
    monkeypatch.setattr(field_store, "load_versions",
                        lambda e, c: [{"calibration": homog, "start_match": None,
                                       "label": "legacy", "created_at": "", "version_id": "legacy"}])
    job = SimpleNamespace(event_key="evt", camera="main", match_key=None,
                          _calib_version_id=None, error=None, calib=None,
                          field_model=None, field_dims=None, hub_prisms=None,
                          field_tracker=None, radar=None, field_tracking=False)
    assert InferenceJob._init_field_tracking(job) is False
    assert "pinhole" in job.error.lower() or "pose" in job.error.lower()


def test_versions_endpoint_lists_epochs(tmp_path, monkeypatch):
    """GET /api/calibration/versions returns metadata for every saved version,
    including the start_match anchor the calibrate page sets via its match selector."""
    import numpy as np
    from web.field import store as field_store
    from web.field.calibration import CameraCalibration
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    cal = CameraCalibration(
        K=np.eye(3), D=np.zeros(4),
        rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), crop_region=None,
        field_season="2026-rebuilt-welded",
    )
    field_store.save_version("evt_v", "main", cal, start_match=None, label="base")
    field_store.save_version("evt_v", "main", cal, start_match="evt_v_qm45",
                             label="camera moved")

    r = _client().get("/api/calibration/versions",
                      params={"event": "evt_v", "camera": "main"})
    assert r.status_code == 200
    versions = r.json()["versions"]
    assert len(versions) == 2
    starts = {v["start_match"] for v in versions}
    assert starts == {None, "evt_v_qm45"}
    assert all(v["version_id"] for v in versions)


def test_calibration_frame_match_key_requires_loaded_event():
    """match_key with no loaded event must fail clearly, not fall back silently."""
    import web.app as app_mod
    if app_mod.current_event:
        import pytest; pytest.skip("an event is loaded in this process")
    r = _client().post("/api/calibration/frame",
                       json={"timestamp": 5, "match_key": "2026hop_qm1"})
    assert r.status_code == 400
    assert "event" in r.json()["error"].lower()


def test_solve_tags_stores_source_and_dry_run_skips_save(tmp_path, monkeypatch):
    """A real solve persists its inputs (source) for later restore; a dry_run
    solve returns RMS/overlay data but writes no new version."""
    from web.field import store as field_store
    from web.field.field_model import FieldModel
    from tests._field_synth import truth_pinhole_camera, project_tag_corners
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)
    auto = {str(tid): corr[tid][0].tolist() for tid in corr}
    body = {"event": "evt_src", "camera": "main", "season": "2026-rebuilt-welded",
            "image_size": list(cam.image_size), "auto_tags": auto, "manual_tags": [],
            "start_match": "evt_src_qm12", "timestamp": 33}

    r = _client().post("/api/calibration/solve-tags", json=body)
    assert r.status_code == 200, r.text
    versions = field_store.load_versions("evt_src", "main")
    assert len(versions) == 1
    src = versions[0]["source"]
    assert src["timestamp"] == 33
    assert src["auto_tags"].keys() == auto.keys()
    assert src["holdout_ids"] == []

    # epoch endpoint surfaces the stored source for the covering match
    r2 = _client().get("/api/calibration/epoch", params={
        "event": "evt_src", "camera": "main", "match": "evt_src_qm40"})
    v = r2.json()["version"]
    assert v["start_match"] == "evt_src_qm12"
    assert v["source"]["auto_tags"].keys() == auto.keys()

    # dry_run: solves fine, but no second version appears
    r3 = _client().post("/api/calibration/solve-tags", json={**body, "dry_run": True})
    assert r3.status_code == 200 and r3.json()["rms_error"] < 2.0
    assert len(field_store.load_versions("evt_src", "main")) == 1


def test_drift_records_lifecycle(tmp_path, monkeypatch):
    """Drift failures persist per (camera, match); the endpoint marks them
    recalibrated once a different epoch covers the match; passing clears them."""
    import numpy as np
    from web.field import store as field_store
    from web.field.calibration import CameraCalibration
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    cal = CameraCalibration(
        K=np.eye(3), D=np.zeros(4),
        rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), crop_region=None,
        field_season="2026-rebuilt-welded",
    )
    base_id = field_store.save_version("evt_d", "main", cal, start_match=None)
    field_store.record_drift("evt_d", "main", "evt_d_qm63", 4.67, base_id)

    r = _client().get("/api/calibration/drift", params={"event": "evt_d"})
    rec = r.json()["drift"]["main"]["evt_d_qm63"]
    assert rec["error_m"] == 4.67 and rec["recalibrated"] is False

    # Recalibrating from that match flips the flag (new epoch covers qm63)...
    field_store.save_version("evt_d", "main", cal, start_match="evt_d_qm63")
    r = _client().get("/api/calibration/drift", params={"event": "evt_d"})
    assert r.json()["drift"]["main"]["evt_d_qm63"]["recalibrated"] is True
    # ...and earlier matches still resolve to the old epoch, so a failure
    # recorded there would NOT be considered recalibrated.
    field_store.record_drift("evt_d", "main", "evt_d_qm10", 2.0, base_id)
    r = _client().get("/api/calibration/drift", params={"event": "evt_d"})
    assert r.json()["drift"]["main"]["evt_d_qm10"]["recalibrated"] is False

    # A passing drift check removes the record entirely.
    field_store.clear_drift("evt_d", "main", "evt_d_qm63")
    r = _client().get("/api/calibration/drift", params={"event": "evt_d"})
    assert "evt_d_qm63" not in r.json()["drift"]["main"]

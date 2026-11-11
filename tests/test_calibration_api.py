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
    # Realistic, non-folding lens: the shared fixture lens folds back inside its own
    # image and solve-tags now refuses non-monotonic lenses.
    cam = truth_pinhole_camera(D=__import__('numpy').array([-0.12, 0.02, 0.0, 0.0, 0.0]))
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
                       json={"timestamp": 5, "match_key": "2026zzzz_qm1"})   # no local footage
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
    cam = truth_pinhole_camera(D=__import__('numpy').array([-0.12, 0.02, 0.0, 0.0, 0.0]))
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


def _stage_autofix(field_store, event, camera, match_key, base_epoch=None,
                   summary=None):
    """Stage a pending auto-fix the way the pipeline would, for endpoint tests."""
    import numpy as np
    from web.field.calibration import CameraCalibration
    cal = CameraCalibration(
        K=np.eye(3), D=np.zeros(5),
        rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), crop_region=None, mode="pinhole",
        field_season="2026-rebuilt-welded", rms_error=1.5)
    summary = summary or {"passed": True, "best_err_m": 0.12, "n_tags": 5,
                          "center_shift_m": 0.03, "rotation_delta_deg": 1.4}
    field_store.save_pending_autofix(event, camera, match_key, cal, summary,
                                     base_epoch=base_epoch)
    return cal


def test_get_autofix_none_when_unstaged(tmp_path, monkeypatch):
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    r = _client().get("/api/calibration/autofix", params={
        "event": "evt_af", "camera": "main", "match": "evt_af_qm1"})
    assert r.status_code == 200
    assert r.json()["autofix"] is None


def test_get_autofix_returns_staged_summary(tmp_path, monkeypatch):
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    _stage_autofix(field_store, "evt_af", "main", "evt_af_qm5", base_epoch="base1")
    r = _client().get("/api/calibration/autofix", params={
        "event": "evt_af", "camera": "main", "match": "evt_af_qm5"})
    assert r.status_code == 200
    af = r.json()["autofix"]
    assert af["base_epoch"] == "base1"
    assert af["summary"]["best_err_m"] == 0.12
    assert af["created_at"]


def test_approve_autofix_promotes_to_version_and_clears(tmp_path, monkeypatch):
    """Approval anchors a NEW calibration version at the drifted match, carries the
    base epoch's masks, and clears both the drift record and the pending fix."""
    import numpy as np
    from web.field import store as field_store
    from web.field.calibration import CameraCalibration
    from web.field.match_order import select_epoch
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    base = CameraCalibration(
        K=np.eye(3), D=np.zeros(5), rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), crop_region=None, mode="pinhole",
        field_season="2026-rebuilt-welded")
    base.masks = [(10, 20, 30, 40)]
    base.robot_track_top_crop = 0.15
    base_id = field_store.save_version("evt_ap", "main", base, start_match=None)
    field_store.record_drift("evt_ap", "main", "evt_ap_qm7", 3.1, base_id)
    _stage_autofix(field_store, "evt_ap", "main", "evt_ap_qm7", base_epoch=base_id)

    r = _client().post("/api/calibration/autofix/approve", json={
        "event": "evt_ap", "camera": "main", "match_key": "evt_ap_qm7"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "ok" and j["version_id"]

    # A new epoch now covers qm7 (anchored there) and carries the base masks.
    versions = field_store.load_versions("evt_ap", "main")
    cover = select_epoch(versions, "evt_ap_qm7")
    assert cover["start_match"] == "evt_ap_qm7"
    assert cover["version_id"] != base_id
    assert cover["calibration"].masks == [(10, 20, 30, 40)]
    assert cover["calibration"].robot_track_top_crop == 0.15

    # Drift record and pending fix are both gone.
    assert "evt_ap_qm7" not in field_store.load_drift("evt_ap").get("main", {})
    assert field_store.load_pending_autofix("evt_ap", "main", "evt_ap_qm7") is None


def test_approve_autofix_without_pending_is_400(tmp_path, monkeypatch):
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    r = _client().post("/api/calibration/autofix/approve", json={
        "event": "evt_none", "camera": "main", "match_key": "evt_none_qm1"})
    assert r.status_code == 400
    assert "pending" in r.json().get("error", "").lower()


def test_reject_autofix_clears_pending_but_keeps_drift(tmp_path, monkeypatch):
    """Rejecting drops the staged fix (so nothing auto-adopts) but leaves the drift
    record so the match stays flagged for manual recalibration."""
    import numpy as np
    from web.field import store as field_store
    from web.field.calibration import CameraCalibration
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    base = CameraCalibration(
        K=np.eye(3), D=np.zeros(5), rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), mode="pinhole", field_season="2026-rebuilt-welded")
    base_id = field_store.save_version("evt_rj", "main", base, start_match=None)
    field_store.record_drift("evt_rj", "main", "evt_rj_qm9", 3.1, base_id)
    _stage_autofix(field_store, "evt_rj", "main", "evt_rj_qm9", base_epoch=base_id)

    r = _client().post("/api/calibration/autofix/reject", json={
        "event": "evt_rj", "camera": "main", "match_key": "evt_rj_qm9"})
    assert r.status_code == 200
    assert "/calibrate" in r.json().get("redirect", "")
    assert field_store.load_pending_autofix("evt_rj", "main", "evt_rj_qm9") is None
    # drift record survives -> match still flagged for manual fix
    assert "evt_rj_qm9" in field_store.load_drift("evt_rj").get("main", {})
    # no new version was created
    assert len(field_store.load_versions("evt_rj", "main")) == 1


def test_preview_autofix_without_pending_is_400(tmp_path, monkeypatch):
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    r = _client().post("/api/calibration/autofix/preview", json={
        "event": "evt_pv", "camera": "main", "match_key": "evt_pv_qm1"})
    assert r.status_code == 400
    assert "pending" in r.json().get("error", "").lower()


def test_preview_autofix_renders_overlay(tmp_path, monkeypatch):
    """End-to-end preview: stage a fix, read a local frame, draw the wireframe +
    tags, return a JPEG plus the summary."""
    import os
    if not os.path.exists("data/test_footage/qual1_first_champs_2026.mp4"):
        import pytest; pytest.skip("test video not present")
    from web.field import store as field_store
    from tests._field_synth import truth_pinhole_camera
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    cal = truth_pinhole_camera()
    field_store.save_pending_autofix(
        "evt_pv", "main", "evt_pv_qm2", cal,
        {"passed": True, "best_err_m": 0.1, "n_tags": 5}, base_epoch="b0")
    r = _client().post("/api/calibration/autofix/preview", json={
        "event": "evt_pv", "camera": "main", "match_key": "evt_pv_qm2",
        "video_path": "data/test_footage/qual1_first_champs_2026.mp4",
        "timestamp": 75})
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["overlay"]) > 1000            # base64 jpeg present
    assert j["summary"]["best_err_m"] == 0.1


def test_drift_endpoint_flags_pending_autofix(tmp_path, monkeypatch):
    """The drift listing marks a match as having a pending auto-fix so the calibrate
    page can offer the approve/reject panel instead of only manual recalibration."""
    import numpy as np
    from web.field import store as field_store
    from web.field.calibration import CameraCalibration
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    base = CameraCalibration(
        K=np.eye(3), D=np.zeros(5), rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), mode="pinhole", field_season="2026-rebuilt-welded")
    base_id = field_store.save_version("evt_df", "main", base, start_match=None)
    field_store.record_drift("evt_df", "main", "evt_df_qm3", 2.2, base_id)
    field_store.record_drift("evt_df", "main", "evt_df_qm4", 2.2, base_id)
    _stage_autofix(field_store, "evt_df", "main", "evt_df_qm3", base_epoch=base_id)

    r = _client().get("/api/calibration/drift", params={"event": "evt_df"})
    cam = r.json()["drift"]["main"]
    assert cam["evt_df_qm3"]["autofix_pending"] is True
    assert cam["evt_df_qm4"]["autofix_pending"] is False


def _base_pinhole_cal():
    import numpy as np
    from web.field.calibration import CameraCalibration
    return CameraCalibration(
        K=np.eye(3), D=np.zeros(5), rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)),
        image_size=(1920, 1080), mode="pinhole",
        field_season="2026-rebuilt-welded", rms_error=1.2)


def _pose_fix_result(passed):
    from web.field.pose_autofix import PoseFixResult
    return PoseFixResult(
        passed=passed, reason="held" if passed else "camera moved too far",
        calibration=_base_pinhole_cal(), best_err_m=0.12 if passed else 0.9,
        n_tags=5, center_shift_m=0.02, rotation_delta_deg=1.1)


def _autofix_job():
    from types import SimpleNamespace
    return SimpleNamespace(
        video_path="x.mp4", calib=_base_pinhole_cal(), field_model=object(),
        event_key="evt_t", camera="main", match_key="evt_t_qm2",
        _calib_version_id="base9")


def test_attempt_autofix_stages_passing_result(tmp_path, monkeypatch):
    """A passing re-solve is persisted as a pending fix keyed by match, tagged with
    the base epoch and the JSON-safe summary."""
    import web.app as app_mod
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    result = _pose_fix_result(passed=True)
    monkeypatch.setattr(app_mod, "resolve_pose_from_video", lambda *a, **k: result)

    from types import SimpleNamespace
    job = _autofix_job()
    out = app_mod.InferenceJob._attempt_autofix(job, SimpleNamespace(crop=(0, 0, 1920, 1080)), 3.0)

    assert out is result and out.passed
    staged = field_store.load_pending_autofix("evt_t", "main", "evt_t_qm2")
    assert staged is not None
    assert staged["base_epoch"] == "base9"
    assert staged["summary"]["best_err_m"] == 0.12


def test_attempt_autofix_does_not_stage_failing_result(tmp_path, monkeypatch):
    import web.app as app_mod
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    result = _pose_fix_result(passed=False)
    monkeypatch.setattr(app_mod, "resolve_pose_from_video", lambda *a, **k: result)

    from types import SimpleNamespace
    out = app_mod.InferenceJob._attempt_autofix(_autofix_job(), SimpleNamespace(crop=(0, 0, 1920, 1080)), 3.0)
    assert out is result and not out.passed
    assert field_store.load_pending_autofix("evt_t", "main", "evt_t_qm2") is None


def test_drift_abort_state_pending_points_to_review():
    """With a staged fix, the abort payload flags autofix_pending + summary and the
    message directs the operator to review/approve rather than recalibrate."""
    import web.app as app_mod
    job = _autofix_job()
    err, nc = app_mod.InferenceJob._drift_abort_state(job, 3.0, _pose_fix_result(True))
    assert nc["autofix_pending"] is True
    assert nc["autofix_summary"]["best_err_m"] == 0.12
    assert nc["error_m"] == 3.0 and nc["match_key"] == "evt_t_qm2"
    assert "approve" in err.lower() or "review" in err.lower()


def test_drift_abort_state_without_fix_is_manual():
    import web.app as app_mod
    job = _autofix_job()
    err, nc = app_mod.InferenceJob._drift_abort_state(job, 3.0, None)
    assert not nc.get("autofix_pending")
    assert "recalibrate" in err.lower()


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


def test_solve_tags_returns_predicted_tags_and_constrains_one_structure(tmp_path, monkeypatch):
    """One hub's tags only: solve-tags keeps the constrained lens (no runaway free
    fit) and returns predicted positions for one-click placement."""
    import numpy as np
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    from fastapi.testclient import TestClient
    import web.app as app_mod
    from tests._field_synth import truth_pinhole_camera, project_tag_corners
    from web.field.field_model import FieldModel
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera(D=np.array([-0.12, 0.02, 0.0, 0.0, 0.0]))
    corr = project_tag_corners(cam, fm)
    hub = [t for t in (5, 8, 9, 10) if t in corr]
    assert len(hub) >= 3
    auto = {str(t): corr[t][0].tolist() for t in hub}
    client = TestClient(app_mod.app)
    r = client.post("/api/calibration/solve-tags", json={
        "event": "evx", "camera": "corner_right", "image_size": [1920, 1080],
        "auto_tags": auto, "manual_tags": [], "dry_run": True})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] in ("pp_k1", "pp_k0") and j["tag_spread_m"] < 2.0
    assert j["predicted_tags"] and all(len(v) == 2 for v in j["predicted_tags"].values())


def test_predicted_tags_route_uses_auto_candidate(tmp_path, monkeypatch):
    import numpy as np
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    from fastapi.testclient import TestClient
    import web.app as app_mod
    from tests._field_synth import truth_pinhole_camera
    cam = truth_pinhole_camera(D=np.array([-0.12, 0.02, 0.0, 0.0, 0.0]))
    client = TestClient(app_mod.app)
    assert client.get("/api/calibration/predicted-tags?event=evp&camera=corner_left").json()["tags"] == {}
    field_store.save_auto_record("evp", "corner_left", {
        "status": "failed", "calibration": cam.to_json(), "crop": [0, 743, 1920, 1080]})
    j = client.get("/api/calibration/predicted-tags?event=evp&camera=corner_left").json()
    assert j["status"] == "failed" and len(j["tags"]) >= 4
    u, v = next(iter(j["tags"].values()))
    assert 0 <= u < 1920 and 0 <= v < 1080

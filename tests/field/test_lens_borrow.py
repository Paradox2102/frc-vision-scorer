"""Borrowing a sibling corner camera's lens (web.field.lens_borrow) and the
auto-calibration fallback that uses it (web.field.auto_calibrate.run_camera).

Synthetic footage as in test_auto_calibrate: a tiny black mp4 supplies frame timing
and an injected detect_fn returns tag corners projected from a known camera. The
2025hop corner_right regression uses values copied from the real versions (never
web/uploads)."""
import cv2
import numpy as np
import pytest

from tests._field_synth import project_tag_corners, truth_pinhole_camera
from web.field import auto_calibrate as ac
from web.field import lens_borrow as lb
from web.field.calibration import CameraCalibration, CalibrationError
from web.field.field_model import FieldModel

FM = FieldModel.load("2026-rebuilt-welded")
CROP = (0, 0, 1920, 1080)
GOOD_D = np.array([-0.12, 0.02, 0.0, 0.0, 0.0])


def _video(tmp_path, n=30, fps=1.0, name="clip.mp4"):
    p = tmp_path / name
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    for _ in range(n):
        w.write(np.full((36, 64, 3), 10, np.uint8))
    w.release()
    return p


def _detector(cam, only=None):
    corr = project_tag_corners(cam, FM)
    dets = {tid: c2d for tid, (c2d, _) in corr.items() if only is None or tid in only}
    return lambda frame: {k: v.copy() for k, v in dets.items()}


def _lens(f=1200.0, size=(1920, 1080), D=GOOD_D, mode="pinhole", pp=None):
    w, h = size
    cx, cy = pp if pp is not None else (w / 2.0, h / 2.0)
    return CameraCalibration(mode=mode, K=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]),
                             D=np.asarray(D, float), image_size=size,
                             crop_region=(0, 0, w, h), field_season="2026-rebuilt-welded",
                             rvec=np.zeros((3, 1)), tvec=np.ones((3, 1)))


# ---------------------------------------------------------------- lens transfer

def test_transfer_scales_focal_by_height_and_recentres_principal_point():
    src = _lens(f=739.97, size=(965, 347), D=[-0.5373, 0, 0, 0, 0], pp=(480.0, 170.0))
    out = lb.transfer_lens(src, (916, 347))
    assert out.mode == "pinhole"
    assert out.K[0, 0] == pytest.approx(739.97) and out.K[1, 1] == pytest.approx(739.97)
    assert (out.K[0, 2], out.K[1, 2]) == (458.0, 173.5)
    assert np.allclose(out.D, [-0.5373, 0, 0, 0, 0])
    assert tuple(out.image_size) == (916, 347)
    assert out.rvec is None and out.tvec is None        # a lens, not a pose
    big = lb.transfer_lens(src, (1830, 694))             # a 2x panel
    assert big.K[0, 0] == pytest.approx(2 * 739.97) and big.K[1, 1] == pytest.approx(2 * 739.97)
    assert (big.K[0, 2], big.K[1, 2]) == (915.0, 347.0)
    # the source is untouched
    assert src.K[0, 2] == 480.0 and src.rvec is not None


def test_transfer_keeps_a_non_square_pixel_ratio():
    src = _lens(size=(1000, 500))
    src.K[1, 1] = 1100.0
    out = lb.transfer_lens(src, (500, 250))
    assert out.K[0, 0] == pytest.approx(600.0) and out.K[1, 1] == pytest.approx(550.0)


@pytest.mark.parametrize("mode", ["camera", "homography"])
def test_transfer_refuses_non_pinhole_sources(mode):
    src = _lens(mode=mode)
    with pytest.raises(CalibrationError, match="pinhole"):
        lb.transfer_lens(src, (916, 347))


def test_transfer_refuses_a_source_without_a_size():
    src = _lens()
    src.image_size = (0, 0)
    with pytest.raises(CalibrationError):
        lb.transfer_lens(src, (916, 347))


# ------------------------------------------------------------------- the rule

def test_only_corner_cameras_lend_to_corner_cameras():
    assert lb.borrow_rule_error("corner_right", "corner_left") is None
    assert lb.borrow_rule_error("corner_left", "corner_right") is None
    assert "main" in lb.borrow_rule_error("main", "corner_left")
    assert "main" in lb.borrow_rule_error("corner_left", "main")
    assert "own" in lb.borrow_rule_error("corner_left", "corner_left")


def _ver(cal, vid, start=None, created="2026-11-11T00:00:00"):
    return {"calibration": cal, "start_match": start, "version_id": vid,
            "created_at": created, "label": None, "source": None}


def test_find_lenders_prefers_versions_then_ready_records_and_skips_main():
    versions = {"main": [_ver(_lens(), "m1")],
                "corner_left": [_ver(_lens(f=700), "L1")],
                "corner_right": [], "corner_far": []}
    records = {"corner_far": {"status": "ready", "calibration": _lens(f=650).to_json(),
                              "created_at": "2026-11-11T01:00:00"},
               "main": {"status": "ready", "calibration": _lens().to_json()}}
    out = lb.find_lenders("corner_right", ["main", "corner_left", "corner_right",
                                           "corner_far"],
                          versions.__getitem__, "evt_qm1", records)
    assert [(l.camera, l.kind, l.version_id) for l in out] == [
        ("corner_left", "version", "L1"), ("corner_far", "auto_record", None)]
    assert out[0].calibration.K[0, 0] == 700
    assert lb.find_lenders("main", list(versions), versions.__getitem__, "evt_qm1",
                           records) == []


def test_find_lenders_uses_the_covering_epoch_and_skips_non_pinhole():
    vs = [_ver(_lens(f=700), "early", None, "2026-11-10"),
          _ver(_lens(f=710), "late", "evt_qm5", "2026-11-11")]
    out = lb.find_lenders("corner_right", ["corner_left"], lambda c: vs, "evt_qm2", {})
    assert out[0].version_id == "early"
    out = lb.find_lenders("corner_right", ["corner_left"], lambda c: vs, "evt_qm7", {})
    assert out[0].version_id == "late"
    fish = [_ver(_lens(mode="camera"), "fish")]
    assert lb.find_lenders("corner_right", ["corner_left"], lambda c: fish,
                           "evt_qm1", {}) == []
    failed = {"corner_left": {"status": "failed", "calibration": _lens().to_json()}}
    assert lb.find_lenders("corner_right", ["corner_left"], lambda c: [], "evt_qm1",
                           failed) == []


def test_should_retry_with_a_new_lender_only():
    rec = {"status": "failed", "method": "full_solve", "metrics": {"n_tags": 2}}
    lenders = [lb.LensLender("corner_left", _lens(), "L1")]
    assert lb.should_retry(rec, "corner_right", lenders)
    assert not lb.should_retry(rec, "main", lenders)
    assert not lb.should_retry(rec, "corner_right", [])
    assert not lb.should_retry(dict(rec, status="ready"), "corner_right", lenders)
    assert not lb.should_retry(dict(rec, method="pose_resolve"), "corner_right", lenders)
    assert not lb.should_retry(dict(rec, metrics={"n_tags": 1}), "corner_right", lenders)
    assert not lb.should_retry(dict(rec, lens_lenders_tried=["corner_left"]),
                               "corner_right", lenders)
    assert lb.should_retry(dict(rec, lens_lenders_tried=["corner_far"]),
                           "corner_right", lenders)


# ------------------------------------------------- 2025hop corner_right (real)

SEASON25 = "2025-reefscape-welded"
# corner_left 20261111-232802-934489 (1.65 px), crop 965x347
HOP_LEFT = {"K": [[739.967700055877, 0.0, 482.5], [0.0, 739.967700055877, 173.5],
                  [0.0, 0.0, 1.0]],
            "D": [-0.5373262887776691, 0.0, 0.0, 0.0, 0.0],
            "rvec": [1.3426460943393659, 1.3867128809160423, -1.0255371138805007],
            "tvec": [-4.251873273284275, -0.22737555519031116, 9.097189557677899],
            "image_size": [965, 347], "crop_region": [0, 0, 965, 347],
            "field_season": SEASON25, "rms_error": 1.6495825784268812, "mode": "pinhole",
            "H": None, "masks": [], "robot_track_top_crop": 0.0}
HOP_LEFT_VID = "20261111-232802-934489"
# corner_right 20261111-233034-815599's source (crop 916x347)
HOP_RIGHT_SIZE = (916, 347)
HOP_RIGHT_AUTO = {"1": [[725.12, 59.5], [737.16, 61.92], [737.89, 47.69], [725.68, 45.37]]}
HOP_RIGHT_MANUAL = [{"id": 2, "u": 120, "v": 57}, {"id": 11, "u": 553, "v": 327},
                    {"id": 10, "u": 424, "v": 342}, {"id": 9, "u": 291, "v": 333}]


def _hop_corr():
    fm = FieldModel.load(SEASON25)
    corr = {f"a{t}": (np.array(c, float), fm.tag_corners_3d(int(t)))
            for t, c in HOP_RIGHT_AUTO.items()}
    for m in HOP_RIGHT_MANUAL:
        corr[f"m{m['id']}"] = (np.array([[m["u"], m["v"]]], float),
                               fm.tag_corners_3d(m["id"]).mean(axis=0).reshape(1, 3))
    return fm, corr


def test_2025hop_corner_right_own_lens_lands_on_the_wrong_side_of_the_barge():
    fm, corr = _hop_corr()
    cal, _model, _spread = ac.solve_fresh_lens(corr, HOP_RIGHT_SIZE, SEASON25, fm)
    assert cal.camera_center[0] < fm.length_m / 2 - 1.0      # x ~6.9: wrong side
    assert cal.K[0, 0] > 1000                                 # traded focal for distance


def test_2025hop_corner_right_with_corner_left_lens_sits_on_the_red_barge_face():
    fm, corr = _hop_corr()
    lens = lb.transfer_lens(CameraCalibration.from_json(HOP_LEFT), HOP_RIGHT_SIZE)
    cal = lb.solve_pose_with_lens(corr, HOP_RIGHT_SIZE, SEASON25, lens)
    c = np.asarray(cal.camera_center, float)
    assert c[0] == pytest.approx(9.29, abs=0.1)               # red side of the barge
    assert c[1] == pytest.approx(3.99, abs=0.1)
    assert c[2] == pytest.approx(1.88, abs=0.1)
    assert cal.K[0, 0] == pytest.approx(739.97, abs=0.01)     # the lens did not move
    assert cal.rms_error == pytest.approx(11.7, abs=0.5)
    errs = lb.per_tag_residuals(cal, corr)
    assert max(errs["a1"]) < 6.0                              # tag 1's corners fit


# ------------------------------------------------ auto-calibration fallback

def _lender(f=1200.0, cam="corner_left", vid="L1"):
    return lb.LensLender(cam, _lens(f=f), vid, "version")


def test_too_few_tags_corner_borrows_lens_and_stays_honest(tmp_path):
    """2 tags: too few for the camera's own lens (needs 3). With a sibling lens the
    pose is solved, but no frame shows 3 tags to validate on -> failed, with the
    borrowed candidate attached for review."""
    cam = truth_pinhole_camera(D=GOOD_D)
    two = [5, 8]
    assert set(two) <= set(project_tag_corners(cam, FM))
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "corner_right", CROP, FM, versions=[],
                        detect_fn=_detector(cam, only=two), lenders=[_lender()])
    rec = res.record
    assert rec["status"] == "failed" and rec["method"] == "full_solve"
    assert "lens borrowed from corner_left (version L1)" in rec["reason"]
    assert "only 2 known tags" in rec["reason"]                # the own-lens failure
    assert res.calibration is not None and rec["calibration"] is not None
    assert rec["lens_from"] == "corner_left" and rec["lens_version_id"] == "L1"
    assert rec["lens_lenders_tried"] == ["corner_left"]
    assert np.linalg.norm(np.asarray(res.calibration.camera_center)
                          - np.asarray(cam.camera_center)) < 0.05
    assert res.calibration.K[0, 0] == pytest.approx(1200.0)


def test_one_structure_corner_borrows_lens_and_validates_ready(tmp_path):
    """4 tags on one hub corner: too little spread for its own lens (failed), but a
    sibling's lens fixes it and the 4-tag frames validate -> ready."""
    cam = truth_pinhole_camera(D=GOOD_D)
    hub = {5, 8, 9, 10}
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "corner_right", CROP, FM, versions=[],
                        detect_fn=_detector(cam, only=hub), lenders=[_lender()])
    rec = res.record
    assert rec["status"] == "ready", rec["reason"]
    assert "lens borrowed from corner_left (version L1)" in rec["reason"]
    assert "only span" in rec["reason"]                        # why its own lens failed
    assert rec["metrics"]["val_gate_best_m"] < 0.05
    assert rec["lens_from"] == "corner_left"


def test_no_fallback_when_the_camera_solves_itself(tmp_path):
    cam = truth_pinhole_camera(D=GOOD_D)
    video = _video(tmp_path)
    plain = ac.run_camera(video, "evt_qm1", "corner_right", CROP, FM, versions=[],
                          detect_fn=_detector(cam))
    lent = ac.run_camera(video, "evt_qm1", "corner_right", CROP, FM, versions=[],
                         detect_fn=_detector(cam), lenders=[_lender(f=600.0)])
    assert plain.record["status"] == lent.record["status"] == "ready"
    assert "borrowed" not in lent.record["reason"]
    assert "lens_from" not in lent.record
    assert np.allclose(lent.calibration.K, plain.calibration.K)


def test_main_never_borrows(tmp_path):
    cam = truth_pinhole_camera(D=GOOD_D)
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "main", CROP, FM, versions=[],
                        detect_fn=_detector(cam, only=[5, 8]), lenders=[_lender()])
    assert res.record["status"] == "failed" and res.calibration is None
    assert "borrowed" not in res.record["reason"]


def test_one_tag_is_too_few_even_for_a_borrowed_lens(tmp_path):
    cam = truth_pinhole_camera(D=GOOD_D)
    res = ac.run_camera(_video(tmp_path), "evt_qm1", "corner_right", CROP, FM, versions=[],
                        detect_fn=_detector(cam, only=[5]), lenders=[_lender()])
    assert res.record["status"] == "failed" and res.calibration is None
    assert "needs >= 2 tags" in res.record["reason"]
    assert res.record["lens_lenders_tried"] == ["corner_left"]

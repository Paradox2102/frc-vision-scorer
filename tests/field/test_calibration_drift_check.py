"""Integration test for InferenceJob._verify_calibration_alignment (Task 6).

Exercises the real method end-to-end against synthetic pinhole projections: only
the video I/O is stubbed (cv2.VideoCapture + detect_tags), while the actual
tag_alignment_error + calibration projection run for real via the same fixtures
the tag-alignment unit tests use. The four covered paths mirror the drift check's
decision points: aligned -> proceed, drifted -> abort, too-few-tags -> proceed
(unvalidated), disabled -> proceed unconditionally.
"""
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import web.app as app
from web.field.field_model import FieldModel
from tests._field_synth import truth_pinhole_camera, project_tag_corners


class _FakeView:
    name = "main"
    crop = (0, 0, 16, 16)      # (x, y, w, h), as CameraView.crop

    def extract(self, frame):
        return frame


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in: always yields the same dummy frame."""

    def __init__(self, *_a):
        self._frame = np.zeros((16, 16, 3), dtype=np.uint8)

    def set(self, *_a):
        return True

    def read(self):
        return True, self._frame

    def release(self):
        pass


@pytest.fixture
def cam_and_tags():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)          # {tid: (img(4,2), obj(4,3))}
    aligned = {tid: img for tid, (img, _obj) in corr.items()}
    return fm, cam, aligned


def _make_job(cam, fm, **over):
    job = SimpleNamespace(
        calib=cam, field_model=fm, views=[_FakeView()],
        processing_ranges=[(0, 100)], video_path=Path("dummy.mp4"),
        event_key="2026hop", camera="main", match_key="2026hop_qm84",
        _calib_version_id="20260608-1200",
        status=None, error=None, needs_calibration=None,
    )
    # Bind the collaborators _verify_calibration_alignment calls via self so the
    # real methods run end-to-end (the file only stubs video I/O, not logic).
    for name in ("_attempt_autofix", "_drift_abort_state"):
        setattr(job, name, MethodType(getattr(app.InferenceJob, name), job))
    job.__dict__.update(over)
    return job


def _patch_video(monkeypatch, detected):
    monkeypatch.setattr(app, "CALIB_CHECK_ENABLED", True)
    monkeypatch.setattr(app.cv2, "VideoCapture", _FakeCap)
    monkeypatch.setattr(app, "detect_tags", lambda _crop: detected)


def test_aligned_calibration_proceeds(monkeypatch, cam_and_tags):
    fm, cam, aligned = cam_and_tags
    _patch_video(monkeypatch, aligned)
    job = _make_job(cam, fm)
    assert app.InferenceJob._verify_calibration_alignment(job) is True
    assert job.error is None
    assert job.needs_calibration is None
    assert job.status is None


def test_drifted_calibration_aborts(monkeypatch, cam_and_tags):
    fm, cam, aligned = cam_and_tags
    # A 60 px shift at this camera geometry reprojects to well over a metre of
    # field-space error — far past the 0.5 m drift threshold (cf. the unit-level
    # test_shifted_tags_large_error).
    drifted = {tid: img + 60.0 for tid, img in aligned.items()}
    _patch_video(monkeypatch, drifted)
    # Autofix OFF: assert the classic manual-recalibration abort payload.
    monkeypatch.setattr(app, "CALIB_AUTOFIX_ENABLED", False)
    job = _make_job(cam, fm)
    assert app.InferenceJob._verify_calibration_alignment(job) is False
    assert job.status == "error"
    assert "drift" in job.error.lower()
    assert "recalibrate" in job.error.lower()
    assert job.needs_calibration is not None
    assert not job.needs_calibration.get("autofix_pending")
    assert job.needs_calibration["match_key"] == "2026hop_qm84"
    assert job.needs_calibration["camera"] == "main"
    assert job.needs_calibration["error_m"] > 1.0   # not merely past the 0.5 m gate


def test_drifted_calibration_stages_autofix(monkeypatch, cam_and_tags, tmp_path):
    """With autofix ON, a drift abort attempts a pose re-solve and, when it passes,
    STAGES it for approval — flipping the abort payload to autofix_pending and
    persisting a pending fix (never auto-adopting it)."""
    from web.field import store as field_store
    from web.field.pose_autofix import PoseFixResult
    fm, cam, aligned = cam_and_tags
    drifted = {tid: img + 60.0 for tid, img in aligned.items()}
    _patch_video(monkeypatch, drifted)
    monkeypatch.setattr(app, "CALIB_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    result = PoseFixResult(passed=True, reason="held", calibration=truth_pinhole_camera(),
                           best_err_m=0.1, n_tags=6, center_shift_m=0.02,
                           rotation_delta_deg=1.0)
    monkeypatch.setattr(app, "resolve_pose_from_video", lambda *a, **k: result)

    job = _make_job(cam, fm)
    assert app.InferenceJob._verify_calibration_alignment(job) is False
    assert job.status == "error"
    assert job.needs_calibration["autofix_pending"] is True
    assert "approve" in job.error.lower() or "review" in job.error.lower()
    staged = field_store.load_pending_autofix("2026hop", "main", "2026hop_qm84")
    assert staged is not None and staged["summary"]["best_err_m"] == 0.1


def test_too_few_tags_proceeds_unvalidated(monkeypatch, cam_and_tags):
    fm, cam, _aligned = cam_and_tags
    _patch_video(monkeypatch, {})                 # detector finds nothing
    job = _make_job(cam, fm)
    assert app.InferenceJob._verify_calibration_alignment(job) is True
    assert job.error is None
    assert job.needs_calibration is None
    assert job.status is None       # the unvalidated path, not the abort path


def test_disabled_proceeds_without_sampling(monkeypatch, cam_and_tags):
    fm, cam, _aligned = cam_and_tags
    monkeypatch.setattr(app, "CALIB_CHECK_ENABLED", False)
    # Count VideoCapture opens: the disabled guard must short-circuit BEFORE any
    # frame sampling (a bad path wouldn't raise, so absence of a crash proves nothing).
    cap_calls = []
    monkeypatch.setattr(app.cv2, "VideoCapture",
                        lambda *a: cap_calls.append(a) or _FakeCap(*a))
    job = _make_job(cam, fm)
    assert app.InferenceJob._verify_calibration_alignment(job) is True
    assert job.error is None
    assert cap_calls == [], "VideoCapture must not be opened when the check is disabled"


def test_single_sample_frame_no_div_by_zero(monkeypatch, cam_and_tags):
    # CALIB_CHECK_SAMPLE_FRAMES=1 takes the n==1 branch ([lo]) — guards against a
    # divide-by-zero in the even-spacing formula.
    fm, cam, aligned = cam_and_tags
    monkeypatch.setattr(app, "CALIB_CHECK_SAMPLE_FRAMES", 1)
    _patch_video(monkeypatch, aligned)
    job = _make_job(cam, fm)
    assert app.InferenceJob._verify_calibration_alignment(job) is True
    assert job.error is None

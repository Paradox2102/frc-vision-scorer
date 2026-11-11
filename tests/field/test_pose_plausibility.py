"""Physically implausible PnP minima (mirrored / upside-down / behind-the-camera)
must never win a pose solve. Reproduces the 2026hop corner_left case (Nov 11): three
tags on two adjacent hub faces at one height, re-solved against the June lens, gave a
pose with every tag BEHIND the camera (rms 19.05 px, centre moved 2.65 m, rotated
159 deg) and an upside-down overlay."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from web.field.calibration import (CameraCalibration, CalibrationError, Calibrator,
                                   check_solve_plausible, pose_plausibility)
from web.field.field_model import FieldModel
from web.field.pose_autofix import _pose_deltas, resolve_pose
from tests._field_synth import truth_pinhole_camera, project_tag_corners

FM = FieldModel.load("2026-rebuilt-welded")
FIX = json.loads((Path(__file__).parent / "fixtures" / "hop_corner_left_flip.json").read_text())


def _hop():
    ecal = CameraCalibration.from_json(FIX["epoch"]["calibration"])
    corr = {int(t): (np.array(c, float), FM.tag_corners_3d(int(t)))
            for t, c in FIX["auto_tags"].items()}
    return ecal, corr


def _single_start(corr, K, D):
    """The pre-fix solve: DLT/iterative init on undistorted points + LM refine."""
    obj = np.vstack([o for _, o in corr.values()]).astype(np.float64)
    img = np.vstack([i for i, _ in corr.values()]).astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(img, K, D.reshape(-1, 1), P=K)
    ok, r, t = cv2.solvePnP(obj, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    r, t = cv2.solvePnPRefineLM(obj, img, K, D.reshape(-1, 1), r, t)
    return r, t, obj


def test_hop_fixture_reproduces_behind_camera_minimum():
    ecal, corr = _hop()
    r, t, obj = _single_start(corr, ecal.K, ecal.D)
    old = CameraCalibration(K=ecal.K, D=ecal.D, rvec=r, tvec=t, image_size=ecal.image_size,
                            field_season=ecal.field_season, mode="pinhole")
    shift, rot = _pose_deltas(ecal, old)
    assert shift == pytest.approx(FIX["recorded"]["center_shift_m"], abs=0.05)
    assert rot == pytest.approx(FIX["recorded"]["rotation_delta_deg"], abs=1.0)
    why = pose_plausibility(old, obj, FM, list(corr))
    assert why is not None and "behind the camera" in why


def test_hop_fixture_now_solves_upright():
    ecal, corr = _hop()
    cand = resolve_pose(corr, ecal)
    assert check_solve_plausible(cand, corr) is None
    assert cand.rms_error < 5.0
    shift, rot = _pose_deltas(ecal, cand)
    assert shift < 1.0 and rot < 20.0
    R, _ = cv2.Rodrigues(cand.rvec)
    assert (R @ [0, 0, 1])[1] < 0                      # world up is image up


def test_single_face_mirror_minimum_is_escaped_by_multistart():
    """Two tags on ONE hub face (coplanar): the second IPPE solution is a mirrored
    pose that an iterative solve started there stays in; the multi-start solve
    returns the true pose regardless of that bad initial guess."""
    cam = truth_pinhole_camera(eye=(0.6, -0.3, 0.7), target=(4.6, 4.0, 0.9),
                               image_size=(960, 377),
                               K=np.array([[680., 0, 440], [0, 680., 150], [0, 0, 1]]),
                               D=np.array([-0.35, 0.1, 0, 0, 0]))
    corr = {t: (np.array([cam.field_to_image(tuple(p)) for p in FM.tag_corners_3d(t)]),
                FM.tag_corners_3d(t)) for t in (25, 26)}
    obj = np.vstack([o for _, o in corr.values()])
    img = np.vstack([i for i, _ in corr.values()])
    und = cv2.undistortPoints(img.reshape(-1, 1, 2), cam.K, cam.D.reshape(-1, 1), P=cam.K)
    _, rs, ts, _ = cv2.solvePnPGeneric(obj[:4], und[:4], cam.K, None, flags=cv2.SOLVEPNP_IPPE)
    bad = max(zip(rs, ts), key=lambda rt: np.linalg.norm(
        (-cv2.Rodrigues(rt[0])[0].T @ rt[1]).ravel() - cam.camera_center))
    r, t = cv2.solvePnPRefineLM(obj, img.reshape(-1, 1, 2), cam.K, cam.D.reshape(-1, 1),
                                bad[0].copy(), bad[1].copy())
    stuck = (-cv2.Rodrigues(r)[0].T @ t).ravel()
    assert np.linalg.norm(stuck - cam.camera_center) > 2.0          # the mirror minimum
    sol = Calibrator.solve_pose(corr, K=cam.K, D=cam.D, image_size=cam.image_size,
                                field_season="2026-rebuilt-welded", mode="pinhole",
                                init=bad)
    assert np.linalg.norm(np.asarray(sol.camera_center) - cam.camera_center) < 0.05


def _upside_down(cam):
    R, _ = cv2.Rodrigues(cam.rvec)
    Rz = np.diag([-1.0, -1.0, 1.0])
    return CameraCalibration(K=cam.K, D=cam.D, rvec=cv2.Rodrigues(Rz @ R)[0],
                             tvec=Rz @ cam.tvec, image_size=cam.image_size,
                             field_season="2026-rebuilt-welded", mode="pinhole")


def test_solve_pose_refuses_when_only_implausible_poses_fit():
    flipped = _upside_down(truth_pinhole_camera())   # the data really came from a flip
    corr = {t: (np.array([flipped.field_to_image(tuple(p)) for p in o]), o)
            for t, (_, o) in project_tag_corners(truth_pinhole_camera(), FM).items()}
    with pytest.raises(CalibrationError, match="implausible"):
        Calibrator.solve_pose(corr, K=flipped.K, D=flipped.D, image_size=flipped.image_size,
                              field_season="2026-rebuilt-welded", mode="pinhole")


@pytest.mark.parametrize("mutate,expect", [
    (lambda R, t: (np.diag([-1., -1., 1.]) @ R, np.diag([-1., -1., 1.]) @ t), "upside down"),
    (lambda R, t: (R, t + np.array([[0.], [0.], [40.]])), "height"),
])
def test_pose_plausibility_reasons(mutate, expect):
    cam = truth_pinhole_camera()
    R, _ = cv2.Rodrigues(cam.rvec)
    R2, t2 = mutate(R, cam.tvec)
    bad = CameraCalibration(K=cam.K, D=cam.D, rvec=cv2.Rodrigues(R2)[0], tvec=t2,
                            image_size=cam.image_size, field_season="x", mode="pinhole")
    assert expect in (pose_plausibility(bad, fm=FM) or "")
    assert pose_plausibility(cam, fm=FM) is None


def test_full_solve_gate_rejects_upside_down_camera():
    flipped = _upside_down(truth_pinhole_camera())
    corr = {t: (np.array([flipped.field_to_image(tuple(p)) for p in o]), o)
            for t, (_, o) in project_tag_corners(truth_pinhole_camera(), FM).items()}
    with pytest.raises(CalibrationError, match="implausible"):
        Calibrator.solve_camera_from_tags(corr, flipped.image_size, "2026-rebuilt-welded")

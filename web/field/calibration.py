"""Fisheye camera calibration + field<->image projection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


class CalibrationError(Exception):
    pass


MIN_POSE_TAGS = 2          # >=2 tags so points are non-coplanar across the field


@dataclass
class CameraCalibration:
    K: np.ndarray            # (3,3)
    D: np.ndarray            # (4,) fisheye
    rvec: np.ndarray         # (3,1)
    tvec: np.ndarray         # (3,1)
    image_size: tuple        # (w, h)
    crop_region: Optional[tuple]  # (x, y, w, h) of main-cam crop, or None
    field_season: str
    rms_error: float = 0.0

    def field_to_image(self, point_xyz) -> tuple:
        obj = np.array([[point_xyz]], dtype=np.float64)  # (1,1,3)
        img, _ = cv2.fisheye.projectPoints(obj, self.rvec, self.tvec, self.K, self.D)
        u, v = img.reshape(2)
        return (float(u), float(v))

    def image_to_field(self, uv) -> tuple:
        pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
        # Undistort to normalized camera-frame ray (z=1).
        norm = cv2.fisheye.undistortPoints(pt, self.K, self.D).reshape(2)
        ray_cam = np.array([norm[0], norm[1], 1.0])
        R, _ = cv2.Rodrigues(self.rvec)
        cam_center = (-R.T @ self.tvec).reshape(3)
        ray_world = R.T @ ray_cam
        # Intersect with ground plane z=0.
        s = -cam_center[2] / ray_world[2]
        world = cam_center + s * ray_world
        return (float(world[0]), float(world[1]))


class Calibrator:
    @staticmethod
    def solve_pose(correspondences: dict, K, D, image_size: tuple,
                   field_season: str) -> "CameraCalibration":
        """Solve camera POSE against KNOWN intrinsics K, D (the per-event auto step).
        correspondences: {tag_id: (corners_2d (4,2), corners_3d (4,3))}."""
        if len(correspondences) < MIN_POSE_TAGS:
            raise CalibrationError(
                f"need >= {MIN_POSE_TAGS} tags for pose, got {len(correspondences)}")
        K = np.asarray(K, float)
        D = np.asarray(D, float).reshape(-1)
        obj = np.vstack([c3d for (_, c3d) in correspondences.values()]).astype(np.float64)
        img = np.vstack([c2d for (c2d, _) in correspondences.values()]).astype(np.float64)
        # Undistort to a pinhole frame (intrinsics K), then PnP.
        undist = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D, P=K)
        ok, rvec, tvec = cv2.solvePnP(obj, undist.reshape(-1, 1, 2), K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise CalibrationError("solvePnP failed")
        cal = CameraCalibration(K=K, D=D, rvec=rvec, tvec=tvec,
                                image_size=image_size, crop_region=None,
                                field_season=field_season)
        cal.rms_error = _reprojection_rms(cal, correspondences)
        return cal


def _reprojection_rms(cal: "CameraCalibration", correspondences: dict) -> float:
    errs = []
    for (c2d, c3d) in correspondences.values():
        for (u, v), p in zip(c2d, c3d):
            pu, pv = cal.field_to_image(tuple(p))
            errs.append((pu - u) ** 2 + (pv - v) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else 0.0

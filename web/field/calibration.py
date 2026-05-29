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

    def to_json(self) -> dict:
        return {
            "K": self.K.tolist(), "D": np.asarray(self.D).reshape(-1).tolist(),
            "rvec": self.rvec.reshape(-1).tolist(), "tvec": self.tvec.reshape(-1).tolist(),
            "image_size": list(self.image_size),
            "crop_region": list(self.crop_region) if self.crop_region else None,
            "field_season": self.field_season, "rms_error": self.rms_error,
        }

    @classmethod
    def from_json(cls, d: dict) -> "CameraCalibration":
        return cls(
            K=np.array(d["K"], float), D=np.array(d["D"], float),
            rvec=np.array(d["rvec"], float).reshape(3,1),
            tvec=np.array(d["tvec"], float).reshape(3,1),
            image_size=tuple(d["image_size"]),
            crop_region=tuple(d["crop_region"]) if d["crop_region"] else None,
            field_season=d["field_season"], rms_error=d.get("rms_error", 0.0))

    def field_to_image(self, point_xyz) -> tuple:
        obj = np.array([[point_xyz]], dtype=np.float64)  # (1,1,3)
        img, _ = cv2.fisheye.projectPoints(obj, self.rvec, self.tvec, self.K, self.D)
        u, v = img.reshape(2)
        return (float(u), float(v))

    def image_to_field(self, uv) -> "tuple | None":
        """Back-project image pixel *uv* to the field ground plane (z=0).

        Returns ``(x, y)`` in field metres, or ``None`` when the pixel's
        back-projected ray is parallel to / pointing above the ground plane
        (i.e. the pixel lies at or above the camera horizon).  Callers must
        handle the ``None`` case.
        """
        pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
        # Undistort to normalized camera-frame ray (z=1).
        norm = cv2.fisheye.undistortPoints(pt, self.K, self.D).reshape(2)
        ray_cam = np.array([norm[0], norm[1], 1.0])
        R, _ = cv2.Rodrigues(self.rvec)
        cam_center = (-R.T @ self.tvec).reshape(3)
        ray_world = R.T @ ray_cam
        # Intersect with ground plane z=0. Reject rays parallel to / above the
        # ground (no valid forward intersection) so callers can drop the point.
        denom = ray_world[2]
        if abs(denom) < 1e-9:
            return None
        s = -cam_center[2] / denom
        if s <= 0:
            return None
        world = cam_center + s * ray_world
        return (float(world[0]), float(world[1]))


class Calibrator:
    MIN_INTRINSIC_VIEWS = 4

    @staticmethod
    def calibrate_intrinsics(views: list, image_size: tuple):
        """views: list of (obj_pts (N,3), img_pts (N,2)) from DISTINCT viewpoints.
        Returns (K, D). Requires multiple genuinely different camera vantages —
        a single static viewpoint is degenerate for fisheye intrinsics."""
        if len(views) < Calibrator.MIN_INTRINSIC_VIEWS:
            raise CalibrationError(
                f"fisheye intrinsics need >= {Calibrator.MIN_INTRINSIC_VIEWS} "
                f"distinct viewpoints, got {len(views)}")
        obj_arr = [o.reshape(-1, 1, 3).astype(np.float64) for (o, _) in views]
        img_arr = [i.reshape(-1, 1, 2).astype(np.float64) for (_, i) in views]
        K = np.zeros((3, 3)); D = np.zeros((4, 1))
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
        cv2.fisheye.calibrate(
            obj_arr, img_arr, image_size, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9))
        return K, D.reshape(-1)

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

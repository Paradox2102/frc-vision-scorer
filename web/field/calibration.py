"""Fisheye camera calibration + field<->image projection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


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

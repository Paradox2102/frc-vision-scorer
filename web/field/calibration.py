"""Fisheye camera calibration + field<->image projection."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np


class CalibrationError(Exception):
    pass


MIN_POSE_TAGS = 2          # >=2 tags so points are non-coplanar across the field
MIN_CALIB_POINTS = 8       # single-view calibrateCamera: ~10 free params


@dataclass
class CameraCalibration:
    K: Optional[np.ndarray] = None       # (3,3) — None in homography mode
    D: Optional[np.ndarray] = None       # (4,) fisheye — None in homography mode
    rvec: Optional[np.ndarray] = None    # (3,1) — None in homography mode
    tvec: Optional[np.ndarray] = None    # (3,1) — None in homography mode
    image_size: tuple = (0, 0)           # (w, h)
    crop_region: Optional[tuple] = None  # (x, y, w, h) of main-cam crop, or None
    field_season: str = ""
    rms_error: float = 0.0
    mode: str = "camera"                 # "camera" (fisheye), "homography", or "pinhole"
    H: Optional[np.ndarray] = None       # (3,3) ground-plane homography (image->field)
    masks: list[tuple] = field(default_factory=list) # crop-local coordinates (x, y, w, h)
    robot_track_top_crop: float = 0.0    # fraction (0.0–0.5) of frame top excluded from SAM2 robot tracking

    def to_json(self) -> dict:
        return {
            "K": self.K.tolist() if self.K is not None else None,
            "D": np.asarray(self.D).reshape(-1).tolist() if self.D is not None else None,
            "rvec": self.rvec.reshape(-1).tolist() if self.rvec is not None else None,
            "tvec": self.tvec.reshape(-1).tolist() if self.tvec is not None else None,
            "image_size": list(self.image_size),
            "crop_region": list(self.crop_region) if self.crop_region else None,
            "field_season": self.field_season, "rms_error": self.rms_error,
            "mode": self.mode,
            "H": self.H.tolist() if self.H is not None else None,
            "masks": [list(m) for m in self.masks] if self.masks is not None else [],
            "robot_track_top_crop": self.robot_track_top_crop,
        }

    @classmethod
    def from_json(cls, d: dict) -> "CameraCalibration":
        K_raw = d.get("K")
        D_raw = d.get("D")
        rvec_raw = d.get("rvec")
        tvec_raw = d.get("tvec")
        H_raw = d.get("H")
        return cls(
            K=np.array(K_raw, float) if K_raw is not None else None,
            D=np.array(D_raw, float) if D_raw is not None else None,
            rvec=np.array(rvec_raw, float).reshape(3, 1) if rvec_raw is not None else None,
            tvec=np.array(tvec_raw, float).reshape(3, 1) if tvec_raw is not None else None,
            image_size=tuple(d["image_size"]),
            crop_region=tuple(d["crop_region"]) if d.get("crop_region") else None,
            field_season=d["field_season"], rms_error=d.get("rms_error", 0.0),
            mode=d.get("mode", "camera"),
            H=np.array(H_raw, float) if H_raw is not None else None,
            masks=[tuple(m) for m in d.get("masks", [])],
            robot_track_top_crop=float(d.get("robot_track_top_crop", 0.0) or 0.0),
        )

    def field_to_image(self, point_xyz) -> tuple:
        if self.mode == "homography":
            Hinv = np.linalg.inv(self.H)
            pt = np.array([[[float(point_xyz[0]), float(point_xyz[1])]]], dtype=np.float64)
            out = cv2.perspectiveTransform(pt, Hinv).reshape(2)
            return (float(out[0]), float(out[1]))
        if self.mode == "pinhole":
            obj = np.array([[point_xyz]], dtype=np.float64)
            D = np.asarray(self.D, float).reshape(-1, 1)
            img, _ = cv2.projectPoints(obj, self.rvec, self.tvec, self.K, D)
            u, v = img.reshape(2)
            return (float(u), float(v))
        obj = np.array([[point_xyz]], dtype=np.float64)
        img, _ = cv2.fisheye.projectPoints(obj, self.rvec, self.tvec, self.K, self.D)
        u, v = img.reshape(2)
        return (float(u), float(v))

    def _cast_norm_ray_to_z(self, norm, z=0.0) -> "tuple | None":
        """Normalized camera ray (x_n, y_n) -> field (x, y) where it hits plane Z=z.
        z=0 is the carpet (used for projecting robot ground-contact points); a
        nonzero z is used to validate elevated references (e.g. AprilTags at z~1m)."""
        ray_cam = np.array([norm[0], norm[1], 1.0])
        R, _ = cv2.Rodrigues(self.rvec)
        cam_center = (-R.T @ self.tvec).reshape(3)
        ray_world = R.T @ ray_cam
        denom = ray_world[2]
        if abs(denom) < 1e-9:
            return None
        s = (z - cam_center[2]) / denom
        if s <= 0:
            return None
        world = cam_center + s * ray_world
        return (float(world[0]), float(world[1]))

    @property
    def camera_center(self):
        """Camera center in field coords (-R^T t), or None for pose-free modes."""
        if self.rvec is None or self.tvec is None:
            return None
        R, _ = cv2.Rodrigues(self.rvec)
        return (-R.T @ self.tvec).reshape(3)

    def image_to_field(self, uv, z=0.0) -> "tuple | None":
        """Project an image point to field (x, y) on the plane Z=z (default carpet,
        z=0). Homography mode is ground-only and ignores z."""
        if self.mode == "homography":
            pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
            out = cv2.perspectiveTransform(pt, self.H).reshape(2)
            return (float(out[0]), float(out[1]))
        if self.mode == "pinhole":
            pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
            D = np.asarray(self.D, float).reshape(-1, 1)
            norm = cv2.undistortPoints(pt, self.K, D).reshape(2)
            return self._cast_norm_ray_to_z(norm, z)
        # fisheye camera mode:
        pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
        norm = cv2.fisheye.undistortPoints(pt, self.K, self.D).reshape(2)
        return self._cast_norm_ray_to_z(norm, z)


def push_from_camera(cal, xy, offset_m) -> "tuple | None":
    """Push a projected ground point ``offset_m`` metres AWAY from the camera.

    A robot box/mask's lowest image point is its NEAR bumper edge as seen from
    an elevated camera, so the carpet projection of that point sits ~half a
    robot too close to the camera. Moving it along the horizontal ground ray
    (direction = xy - camera_center_xy) recovers the footprint centre:
    ``xy' = xy + offset * (xy - c_xy) / |xy - c_xy|``.

    Returns ``xy`` unchanged (the same object) when ``offset_m`` <= 0, ``xy``
    is None, the calibration has no pose (homography mode: ``camera_center``
    is None), or the point sits (numerically) under the camera.
    """
    if xy is None or cal is None or not offset_m or offset_m <= 0:
        return xy
    cc = getattr(cal, "camera_center", None)
    if cc is None:
        return xy
    dx = float(xy[0]) - float(cc[0])
    dy = float(xy[1]) - float(cc[1])
    d = (dx * dx + dy * dy) ** 0.5
    if d < 1e-6:
        return xy
    k = float(offset_m) / d
    return (float(xy[0]) + k * dx, float(xy[1]) + k * dy)


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
    def solve_ground_homography(image_points, field_xy, image_size, field_season):
        """Fit a ground-plane homography (image px -> field x,y at z=0) from
        well-spread clicked correspondences. Intrinsics-free; for a single static
        view. image_points,(N,2); field_xy,(N,2). Needs >= 4 points."""
        img = np.asarray(image_points, np.float64).reshape(-1, 2)
        fld = np.asarray(field_xy, np.float64).reshape(-1, 2)
        if len(img) < 4 or len(fld) != len(img):
            raise CalibrationError(f"homography needs >= 4 correspondences, got {len(img)}")
        H, _ = cv2.findHomography(img, fld, method=0)   # exact LS fit, no RANSAC
        if H is None:
            raise CalibrationError("findHomography failed (degenerate points?)")
        cal = CameraCalibration(K=None, D=None, rvec=None, tvec=None,
                                image_size=image_size, crop_region=None,
                                field_season=field_season, mode="homography", H=H)
        # rms in PIXELS: reproject field->image via H^-1, compare to clicked px.
        Hinv = np.linalg.inv(H)
        proj = cv2.perspectiveTransform(fld.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        cal.rms_error = float(np.sqrt(((proj - img) ** 2).sum(axis=1).mean()))
        return cal

    @staticmethod
    def solve_camera_from_tags(correspondences: dict, image_size: tuple,
                               field_season: str) -> "CameraCalibration":
        """Single-view full-camera resection from AprilTag correspondences.

        correspondences: {key: (img_pts (N,2), obj_pts (N,3))} — N=4 for auto-detected
        tag corners, N=1 for a manually-labeled tag center. Estimates a pinhole model
        (fx, fy + radial k1,k2) and pose via cv2.calibrateCamera over the non-coplanar
        tag points. The principal point is SOLVED (not fixed): the main view is a crop
        of the full broadcast frame, so the optical center is not at the crop center —
        pinning it there was the dominant reprojection error (RMS ~9px -> ~2px once
        freed). Tangential distortion and k3 are zeroed (with our point spread the extra
        terms overfit without helping). Returns a mode='pinhole' calibration."""
        img_list, obj_list = [], []
        for (i2d, o3d) in correspondences.values():
            img_list.append(np.asarray(i2d, np.float64).reshape(-1, 2))
            obj_list.append(np.asarray(o3d, np.float64).reshape(-1, 3))
        if not img_list:
            raise CalibrationError("no correspondences supplied")
        img_all = np.vstack(img_list).astype(np.float32)
        obj_all = np.vstack(obj_list).astype(np.float32)
        n = len(img_all)
        if n < MIN_CALIB_POINTS:
            raise CalibrationError(
                f"need >= {MIN_CALIB_POINTS} points for single-view calibration, got "
                f"{n} (auto tags give 4 points each, manual tags 1 each)")
        w, h = int(image_size[0]), int(image_size[1])
        K0 = np.array([[float(w), 0., w / 2.0],
                       [0., float(w), h / 2.0],
                       [0., 0., 1.]])
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                 cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
        try:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                [obj_all.reshape(-1, 1, 3)], [img_all.reshape(-1, 1, 2)],
                (w, h), K0.copy(), np.zeros(5), flags=flags)
        except cv2.error as e:
            raise CalibrationError(f"calibrateCamera failed: {str(e)[:120]}")
        return CameraCalibration(
            mode="pinhole", K=np.asarray(K, float),
            D=np.asarray(D, float).reshape(-1),
            rvec=np.asarray(rvecs[0], float).reshape(3, 1),
            tvec=np.asarray(tvecs[0], float).reshape(3, 1),
            image_size=(w, h), crop_region=(0, 0, w, h),
            field_season=field_season, rms_error=float(rms))

    @staticmethod
    def solve_pose(correspondences: dict, K, D, image_size: tuple,
                   field_season: str, mode: str = "camera") -> "CameraCalibration":
        """Solve camera POSE against KNOWN intrinsics K, D (the per-event auto step).
        correspondences: {tag_id: (corners_2d (N,2), corners_3d (N,3))}.

        ``mode`` selects the lens model of the supplied intrinsics and MUST match the
        model they were solved under: "camera" (fisheye, 4-coeff D) or "pinhole"
        (5-coeff radial/tangential D). Undistorting with the wrong model silently
        yields a wrong pose — the two undistort routines are not interchangeable. The
        returned calibration carries the same mode so downstream projection matches."""
        if len(correspondences) < MIN_POSE_TAGS:
            raise CalibrationError(
                f"need >= {MIN_POSE_TAGS} tags for pose, got {len(correspondences)}")
        if mode not in ("camera", "pinhole"):
            raise CalibrationError(f"solve_pose needs a projective mode, got '{mode}'")
        K = np.asarray(K, float)
        D = np.asarray(D, float).reshape(-1)
        obj = np.vstack([c3d for (_, c3d) in correspondences.values()]).astype(np.float64)
        img = np.vstack([c2d for (c2d, _) in correspondences.values()]).astype(np.float64)
        # Recover pose against the correct lens model. The two models take different
        # PnP paths — mixing them is the classic wrong-pose bug:
        #   pinhole: DLT-init on undistorted points, then LM-refine against the true
        #     5-coeff Brown model. Passing distortion straight to solvePnP's ITERATIVE
        #     init is fragile for oblique views — its DLT seed ignores distortion and
        #     can diverge by hundreds of px — while undistort-only init leaves residual
        #     at moderate k1. Init-then-refine is both robust and sub-pixel accurate.
        #   fisheye: solvePnP can't ingest the 4-coeff equidistant D, so undistort to a
        #     pinhole frame (P=K) first, then PnP with no further distortion.
        img_pnp = img.reshape(-1, 1, 2)
        if mode == "pinhole":
            Dp = D.reshape(-1, 1)
            undist = cv2.undistortPoints(img_pnp, K, Dp, P=K)
            ok, rvec, tvec = cv2.solvePnP(obj, undist.reshape(-1, 1, 2), K, None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rvec, tvec = cv2.solvePnPRefineLM(obj, img_pnp, K, Dp, rvec, tvec)
        else:
            undist = cv2.fisheye.undistortPoints(img_pnp, K, D, P=K)
            ok, rvec, tvec = cv2.solvePnP(obj, undist.reshape(-1, 1, 2), K, None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise CalibrationError("solvePnP failed")
        cal = CameraCalibration(K=K, D=D, rvec=rvec, tvec=tvec,
                                image_size=image_size, crop_region=None,
                                field_season=field_season, mode=mode)
        cal.rms_error = _reprojection_rms(cal, correspondences)
        return cal


def holdout_field_error(cal: "CameraCalibration", held: dict) -> float:
    """Mean field-position error in meters over held-out correspondences.
    held: {tag_id: (img_pts (N,2), obj_pts (N,3))}. Each image point is projected
    to the plane at the reference point's OWN height (obj z) — AprilTags sit at
    z~0.55-1.12 m, so casting them to the carpet (z=0) would inject a large parallax
    error and make a good calibration look bad. Compares against the known (x,y)."""
    errs = []
    for (i2d, o3d) in held.values():
        for (u, v), p in zip(np.asarray(i2d).reshape(-1, 2), np.asarray(o3d).reshape(-1, 3)):
            xy = cal.image_to_field((float(u), float(v)), z=float(p[2]))
            if xy is None:
                continue
            errs.append(np.hypot(xy[0] - p[0], xy[1] - p[1]))
    return float(np.mean(errs)) if errs else float("inf")


def tag_alignment_error(cal, detected_tags, field_model):
    """(median per-tag field error in meters, n_tags) between auto-detected
    AprilTags and their calibrated positions.

    detected_tags: {tag_id: corners (4,2)} from detect_tags(). For each tag known to
    the field model, projects its detected corners to field coords at the tag's own
    height and takes the mean corner distance to the known position; returns the
    MEDIAN of those per-tag errors. Median (not mean) is used because a few tags at
    extreme camera obliquity (~90deg) have an ill-conditioned image->field ray-cast
    that inflates a mean — the median stays representative of the well-conditioned
    tags. Returns (inf, 0) when no usable tags.
    """
    per_tag = []
    for tid, corners in detected_tags.items():
        if tid not in field_model.tags:
            continue
        img = np.asarray(corners, float).reshape(-1, 2)
        obj = field_model.tag_corners_3d(tid)
        dists = []
        for (u, v), p in zip(img, obj):
            xy = cal.image_to_field((float(u), float(v)), z=float(p[2]))
            if xy is not None:
                dists.append(float(np.hypot(xy[0] - p[0], xy[1] - p[1])))
        if dists:
            per_tag.append(float(np.mean(dists)))
    if not per_tag:
        return float("inf"), 0
    return float(np.median(per_tag)), len(per_tag)


def _reprojection_rms(cal: "CameraCalibration", correspondences: dict) -> float:
    errs = []
    for (c2d, c3d) in correspondences.values():
        for (u, v), p in zip(c2d, c3d):
            pu, pv = cal.field_to_image(tuple(p))
            errs.append((pu - u) ** 2 + (pv - v) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else 0.0

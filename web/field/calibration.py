"""Fisheye camera calibration + field<->image projection."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np


class CalibrationError(Exception):
    pass


MIN_POSE_TAGS = 2          # >=2 tags so points are non-coplanar across the field
MIN_CALIB_POINTS = 8       # single-view calibrateCamera: ~10 free params

# Physical plausibility of a solved pose (pose_plausibility). Stored calibrations
# (2026cur/2026hop, Nov 2026): corner cameras sit at z 0.67-0.70 m, main 3.3-3.9 m,
# rolls within +-2 deg (one bad hop corner_right version: 26 deg). The band and roll
# limit are loose on purpose — they only have to reject mirror/flip solutions.
POSE_Z_BAND_M = (0.3, 15.0)
POSE_MAX_ROLL_DEG = 45.0
# A camera centre at least this far INSIDE the field rectangle is mounted on a field
# structure (2025: the barge cameras at the midline) and looks outward — the
# field-centre-in-front check does not apply to it. Broadcast cameras outside the
# field (2026 corner cameras, the main camera) keep the check.
POSE_ON_FIELD_INSET_M = 0.25

# Visibility (CameraCalibration.project_checked): a field point closer to the camera
# plane than this (camera-frame depth, metres) is not visible — projection blows up
# near the camera and mirrors points behind it into the image.
NEAR_LIMIT_M = 0.3
_FOLD_SAMPLES = 20000


def _radial_gain_pinhole(D: np.ndarray, r: np.ndarray) -> np.ndarray:
    """d(r * f(r))/dr for OpenCV's rational radial model
    f(r) = (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6),
    D = [k1, k2, p1, p2, k3, k4, k5, k6, ...]; -1 where the denominator is <= 0
    (the rational model's own singularity counts as a fold)."""
    k = np.zeros(8)
    k[:min(8, len(D))] = D[:8]
    k1, k2, k3, k4, k5, k6 = k[0], k[1], k[4], k[5], k[6], k[7]
    r2 = r * r
    num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
    # d/dr of r*num/den = (num + r num')/den - r num den'/den^2
    dnum = 2 * k1 * r + 4 * k2 * r ** 3 + 6 * k3 * r ** 5
    dden = 2 * k4 * r + 4 * k5 * r ** 3 + 6 * k6 * r ** 5
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (num + r * dnum) / den - r * num * dden / (den * den)
    return np.where(den > 0, g, -1.0)


def _radial_gain_fisheye(D: np.ndarray, th: np.ndarray) -> np.ndarray:
    """d(theta_d)/d(theta) for OpenCV's fisheye model
    theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)."""
    k = np.zeros(4)
    k[:min(4, len(D))] = D[:4]
    t2 = th * th
    return (1 + 3 * k[0] * t2 + 5 * k[1] * t2 ** 2 + 7 * k[2] * t2 ** 3
            + 9 * k[3] * t2 ** 4)


def _first_fold(gain, lo: float, hi: float) -> float:
    """Smallest x in (lo, hi) where gain(x) <= 0 (inf if none), to ~1e-12."""
    xs = np.linspace(lo, hi, _FOLD_SAMPLES)
    bad = np.nonzero(~(gain(xs) > 0))[0]
    if len(bad) == 0:
        return float("inf")
    i = int(bad[0])
    if i == 0:
        return float(lo)
    a, b = float(xs[i - 1]), float(xs[i])       # gain(a) > 0 >= gain(b)
    for _ in range(80):
        m = 0.5 * (a + b)
        if gain(np.array([m]))[0] > 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


_fold_cache: dict = {}


def distortion_fold_radius(D, mode: str = "pinhole") -> float:
    """The undistorted normalised radius r = |(X/Z, Y/Z)| at which the lens's radial
    distortion stops being monotonic — d(r * f(r))/dr reaches 0 — or inf when it never
    folds. Past it a point far outside the field of view maps back INTO the image
    (k1 only: r^2 = -1 / (3 k1)). Tangential terms are ignored (small). ``mode``
    "pinhole" uses OpenCV's rational model D = [k1, k2, p1, p2, k3, k4, k5, k6, ...];
    "camera" (fisheye) uses theta_d = theta(1 + k1 theta^2 + ... + k4 theta^8), r = tan
    theta, searched up to 90 degrees off-axis."""
    if D is None:
        return float("inf")
    Dv = np.asarray(D, float).reshape(-1)
    key = (mode, Dv.tobytes())
    hit = _fold_cache.get(key)
    if hit is not None:
        return hit
    if mode == "pinhole":
        # search in angle (uniform resolution out to ~90 deg), then map back to r
        th = _first_fold(lambda t: _radial_gain_pinhole(Dv, np.tan(t)),
                         0.0, np.pi / 2 - 1e-6)
        r = float("inf") if not np.isfinite(th) else float(np.tan(th))
    else:
        th = _first_fold(lambda t: _radial_gain_fisheye(Dv, t), 0.0, np.pi / 2 - 1e-9)
        r = float("inf") if not np.isfinite(th) else float(np.tan(th))
    if len(_fold_cache) > 256:
        _fold_cache.clear()
    _fold_cache[key] = r
    return r


# Fold conditioning (fold_conditioning): a lens fitted from few tags can bend its
# distortion to absorb click error, pulling the fold radius in to just past the
# outermost used tag. Ratio = max normalised radius of the used points / fold radius.
# Stored pinhole solves with tag sources (Nov 11 2026, every event):
#   over-fitted own lenses   2025hop corner_right 0.958 (fx 1168) and 0.856 (fx 1047,
#                            centre x 7.4: wrong side of the barge); 2025hop
#                            corner_left 0.756 (fx 1599 vs the true 740)
#   good own lenses          2026casnd corner_right 0.696, 2025hop corner_left 0.62,
#                            2026casnd corner_left 0.60, 2025hop main 0.37; 2026
#                            lenses that never fold: 0
#   borrowed (fixed) lens    2025hop corner_right 0.69
# The brief's 0.8 misses the 0.756 over-fit; 0.72 separates every case above (good
# max 0.696, bad min 0.756). A warning only, never a pass/fail gate.
FOLD_RATIO_WARN = 0.72


@dataclass
class FoldConditioning:
    ratio: float            # r_max / fold_radius; 0 for a lens that never folds, inf
                            # when a used point is at / behind the camera
    r_max: float            # largest undistorted normalised radius of the used points
    fold_radius: float      # distortion_fold_radius (inf: never folds)
    key: Optional[str]      # correspondence key of the outermost point ("a2", "m11")

    @property
    def tag_id(self) -> Optional[int]:
        return None if self.key is None else _tag_id_of_key(self.key)

    @property
    def behind_camera(self) -> bool:
        return not np.isfinite(self.r_max)

    @property
    def near_fold(self) -> bool:
        return self.ratio >= FOLD_RATIO_WARN

    def to_json(self) -> dict:
        def fin(x, nd):
            return round(float(x), nd) if np.isfinite(x) else None
        return {"ratio": fin(self.ratio, 3), "r_max": fin(self.r_max, 3),
                "fold_radius": fin(self.fold_radius, 3), "tag": self.tag_id,
                "near_fold": bool(self.near_fold)}


def fold_conditioning(cal, correspondences) -> "Optional[FoldConditioning]":
    """How close ``cal``'s used correspondence points ({key: (img, obj (N,3))}) come
    to its lens's fold radius. None for a pose-free (homography) calibration or no
    points. The radius is project_checked's: |(X/Z, Y/Z)| in the camera frame."""
    if (cal is None or cal.mode == "homography" or cal.rvec is None or cal.tvec is None
            or not correspondences):
        return None
    R, _ = cv2.Rodrigues(np.asarray(cal.rvec, float))
    t = np.asarray(cal.tvec, float).reshape(1, 3)
    fold = distortion_fold_radius(cal.D, cal.mode)
    best_r, best_key = -1.0, None
    for key, (_img, obj) in correspondences.items():
        P = np.asarray(obj, float).reshape(-1, 3)
        if not len(P):
            continue
        C = P @ R.T + t
        z = C[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(z > 0, np.hypot(C[:, 0], C[:, 1]) / np.where(z > 0, z, 1.0),
                         np.inf)
        rk = float(np.max(r))
        if rk > best_r:
            best_r, best_key = rk, str(key)
    if best_key is None:
        return None
    if not np.isfinite(best_r):
        ratio = float("inf")
    elif not np.isfinite(fold):
        ratio = 0.0
    else:
        ratio = best_r / fold
    return FoldConditioning(ratio=float(ratio), r_max=float(best_r),
                            fold_radius=float(fold), key=best_key)


def fold_warning(cond: "Optional[FoldConditioning]", suggestion: Optional[str] = None,
                 lens_fixed: bool = False) -> Optional[str]:
    """Operator-facing warning for a near-fold solve, or None. ``lens_fixed``: the lens
    was not fitted here (borrowed / shared), so it cannot be over-fitted — the warning
    only says the points reach its distortion limit. ``suggestion`` (e.g. "try
    borrowing from corner_left") is appended after an em dash."""
    if cond is None or not cond.near_fold:
        return None
    what = f"tag {cond.tag_id}" if cond.tag_id is not None else f"point {cond.key}"
    if cond.behind_camera:
        msg = f"{what} is behind the camera in this solve — the pose is broken"
    elif lens_fixed:
        msg = (f"{what} reaches near the lens's distortion limit (r {cond.r_max:.2f} of "
               f"{cond.fold_radius:.2f}) — projection there is unreliable")
    else:
        msg = (f"lens fit is near its distortion limit at {what} (r {cond.r_max:.2f} "
               f"of {cond.fold_radius:.2f}) — the lens is probably over-fitted")
    if suggestion:
        msg += f"; {suggestion}"
    return msg


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

    def project_checked(self, points, margin_px: Optional[float] = 0.0,
                        near_m: float = NEAR_LIMIT_M, image_size=None):
        """Batch field->image with a visibility test: ((N, 2) pixels, (N,) bool ok).

        A point is visible when, in pose modes (pinhole / fisheye):
          * its camera-frame depth exceeds ``near_m`` (not behind / at the camera),
          * its undistorted normalised radius is inside the lens's monotonic region
            (< distortion_fold_radius: past it a point far outside the field of view
            folds back INTO the image), and
          * its pixel is finite and inside the image grown by ``margin_px`` on every
            side (``image_size`` (w, h), default self.image_size). ``margin_px=None``
            skips the frame test — for drawing, where a line may run off-screen and
            the canvas clips it.
        Homography mode keeps its old behaviour: only the finite + frame test.
        Pixels are field_to_image's for every row (ok or not)."""
        P = np.asarray(points, np.float64).reshape(-1, 3)
        n = len(P)
        if n == 0:
            return np.zeros((0, 2)), np.zeros(0, bool)
        if self.mode == "homography":
            Hinv = np.linalg.inv(self.H)
            uv = cv2.perspectiveTransform(P[:, :2].reshape(-1, 1, 2), Hinv).reshape(-1, 2)
            ok = np.ones(n, bool)
        else:
            obj = P.reshape(-1, 1, 3)
            R, _ = cv2.Rodrigues(np.asarray(self.rvec, float))
            cam = P @ R.T + np.asarray(self.tvec, float).reshape(1, 3)
            z = cam[:, 2]
            ok = z > near_m
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.hypot(cam[:, 0], cam[:, 1]) / np.where(ok, z, 1.0)
            ok &= r < distortion_fold_radius(self.D, self.mode)
            if self.mode == "pinhole":
                D = np.asarray(self.D, float).reshape(-1, 1)
                uv, _ = cv2.projectPoints(obj, self.rvec, self.tvec, self.K, D)
            else:
                uv, _ = cv2.fisheye.projectPoints(obj, self.rvec, self.tvec, self.K, self.D)
            uv = uv.reshape(-1, 2)
        ok &= np.all(np.isfinite(uv), axis=1)
        if margin_px is not None:
            w, h = image_size if image_size is not None else self.image_size
            m = float(margin_px)
            with np.errstate(invalid="ignore"):
                ok &= ((uv[:, 0] >= -m) & (uv[:, 0] < float(w) + m)
                       & (uv[:, 1] >= -m) & (uv[:, 1] < float(h) + m))
        return uv, ok

    def in_view(self, points, margin_px: Optional[float] = 0.0,
                near_m: float = NEAR_LIMIT_M, image_size=None) -> np.ndarray:
        """(N,) bool: project_checked's visibility for each field point."""
        return self.project_checked(points, margin_px, near_m, image_size)[1]

    def field_to_image_visible(self, point_xyz, margin_px: Optional[float] = 0.0,
                               near_m: float = NEAR_LIMIT_M,
                               image_size=None) -> "tuple | None":
        """field_to_image, or None when the point is not visible (see
        project_checked: behind / at the camera, past the lens fold radius, or
        outside the frame + ``margin_px``; ``margin_px=None`` skips the frame test)."""
        uv, ok = self.project_checked([tuple(point_xyz)], margin_px, near_m, image_size)
        if not ok[0]:
            return None
        return (float(uv[0, 0]), float(uv[0, 1]))

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


def visible_point(cal, point_xyz, margin_px: Optional[float] = None) -> "tuple | None":
    """cal.field_to_image_visible(point_xyz, margin_px) — default: depth + lens-fold
    test only, no frame test. A duck-typed calibration without it (test doubles)
    falls back to plain field_to_image, returning None for a non-finite pixel."""
    f = getattr(cal, "field_to_image_visible", None)
    if f is not None:
        return f(point_xyz, margin_px=margin_px)
    uv = cal.field_to_image(point_xyz)
    if uv is None or not np.all(np.isfinite(np.asarray(uv, float))):
        return None
    return (float(uv[0]), float(uv[1]))


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
        cal = CameraCalibration(
            mode="pinhole", K=np.asarray(K, float),
            D=np.asarray(D, float).reshape(-1),
            rvec=np.asarray(rvecs[0], float).reshape(3, 1),
            tvec=np.asarray(tvecs[0], float).reshape(3, 1),
            image_size=(w, h), crop_region=(0, 0, w, h),
            field_season=field_season, rms_error=float(rms))
        why = check_solve_plausible(cal, correspondences, field_season)
        if why is not None:
            raise CalibrationError(
                f"the camera solve is physically implausible ({why}) — check the "
                f"tag correspondences / calibrate manually")
        return cal

    @staticmethod
    def solve_pose(correspondences: dict, K, D, image_size: tuple,
                   field_season: str, mode: str = "camera", init=None,
                   check_plausible: bool = True) -> "CameraCalibration":
        """Solve camera POSE against KNOWN intrinsics K, D (the per-event auto step).
        correspondences: {tag_id: (corners_2d (N,2), corners_3d (N,3))}.

        ``mode`` selects the lens model of the supplied intrinsics and MUST match the
        model they were solved under: "camera" (fisheye, 4-coeff D) or "pinhole"
        (5-coeff radial/tangential D). Undistorting with the wrong model silently
        yields a wrong pose — the two undistort routines are not interchangeable. The
        returned calibration carries the same mode so downstream projection matches.

        Multi-start (``_pose_multistart``): the plain solve, ``init`` (rvec, tvec),
        SQPnP and both IPPE solutions of every tag are refined on all points; the
        lowest-reprojection candidate that passes ``pose_plausibility`` wins. When
        no candidate is plausible (only mirrored / behind-camera / upside-down poses
        fit) it raises CalibrationError with that reason."""
        if len(correspondences) < MIN_POSE_TAGS:
            raise CalibrationError(
                f"need >= {MIN_POSE_TAGS} tags for pose, got {len(correspondences)}")
        if mode not in ("camera", "pinhole"):
            raise CalibrationError(f"solve_pose needs a projective mode, got '{mode}'")
        K = np.asarray(K, float)
        D = np.asarray(D, float).reshape(-1)
        cands = _pose_multistart(correspondences, K, D, mode, init)
        if not cands:
            raise CalibrationError("solvePnP failed")
        first_why = None
        best_rms = cands[0][0]
        for rms, rvec, tvec in cands:
            # A plausible pose that fits far worse than the best (implausible) one
            # is not an answer either — only a mirror of the real problem.
            if rms > max(2.0 * best_rms, best_rms + 3.0):
                break
            cal = CameraCalibration(K=K, D=D, rvec=rvec, tvec=tvec,
                                    image_size=image_size, crop_region=None,
                                    field_season=field_season, mode=mode)
            why = check_solve_plausible(cal, correspondences, field_season) \
                if check_plausible else None
            if why is None:
                cal.rms_error = _reprojection_rms(cal, correspondences)
                return cal
            first_why = first_why or f"{why} (rms {rms:.1f} px)"
        raise CalibrationError(
            "only physically implausible poses fit these tags (upside-down / mirrored / "
            f"behind the camera): {first_why} — calibrate manually")


def _tag_id_of_key(key) -> Optional[int]:
    """Tag id of a correspondence key: 6, "6", "a6" (auto), "m6" (manual centre),
    "r6" (ROI decode). None when it is not a tag."""
    if isinstance(key, (int, np.integer)):
        return int(key)
    k = str(key)
    if k[:1] in ("a", "m", "r"):
        k = k[1:]
    return int(k) if k.isdigit() else None


_FM_CACHE: dict = {}


def _field_model(season: str):
    if not season:
        return None
    fm = _FM_CACHE.get(season)
    if fm is None:
        try:
            from web.field.field_model import FieldModel
            fm = _FM_CACHE[season] = FieldModel.load(season)
        except Exception:
            return None
    return fm


def pose_plausibility(cal: "CameraCalibration", obj_points=None, fm=None, tag_ids=(),
                      z_band=POSE_Z_BAND_M, max_roll_deg=POSE_MAX_ROLL_DEG) -> Optional[str]:
    """None when ``cal``'s pose is physically plausible for a broadcast camera, else a
    one-line reason. A hard gate against mirror / flipped PnP minima:
      * camera centre height within ``z_band`` metres;
      * camera upright — world +z points UP in the image (roll within max_roll_deg);
      * every correspondence point (``obj_points``) in front of the camera;
      * with a field model: the field centre in front of the camera — unless the
        camera centre lies inside the field rectangle (inset POSE_ON_FIELD_INSET_M):
        a camera mounted ON the field (2025 barge cameras) looks outward and has
        the centre behind it — and the used tags (``tag_ids``) facing it: a decoded
        tag cannot be seen from behind. A MAJORITY must face (synthetic fixtures
        include tags seen edge-on/from behind).
    Pose-free (homography) calibrations are always plausible."""
    if cal is None or cal.rvec is None or cal.tvec is None:
        return None
    R, _ = cv2.Rodrigues(np.asarray(cal.rvec, float))
    t = np.asarray(cal.tvec, float).reshape(3)
    c = (-R.T @ t).reshape(3)
    if not np.all(np.isfinite(c)):
        return "non-finite camera centre"
    if not (z_band[0] <= c[2] <= z_band[1]):
        return (f"camera height {c[2]:.2f} m is outside {z_band[0]:g}-{z_band[1]:g} m "
                f"(below the floor / implausible)")
    up = R @ np.array([0.0, 0.0, 1.0])            # world up, camera frame (y = down)
    if abs(up[0]) + abs(up[1]) > 0.2:             # not looking straight down
        if up[1] >= 0:
            return "camera is upside down (world up points down in the image)"
        roll = float(np.degrees(np.arctan2(up[0], -up[1])))
        if abs(roll) > max_roll_deg:
            return f"camera roll {roll:.0f} deg exceeds +-{max_roll_deg:g} deg"
    if obj_points is not None and len(obj_points):
        P = np.asarray(obj_points, float).reshape(-1, 3)
        zc = (P @ R.T + t)[:, 2]
        if np.any(zc <= 0):
            return f"{int(np.sum(zc <= 0))} of {len(zc)} tag points are behind the camera"
    if fm is not None:
        ctr = np.array([fm.length_m / 2.0, fm.width_m / 2.0, 0.0])
        m = POSE_ON_FIELD_INSET_M
        on_field = (m <= c[0] <= fm.length_m - m) and (m <= c[1] <= fm.width_m - m)
        if not on_field and (R @ ctr + t)[2] <= 0:
            return "the field centre is behind the camera"
        ids = [i for i in dict.fromkeys(tag_ids) if i is not None and i in fm.tags]
        if ids:
            from web.field.field_model import _quat_to_matrix
            away = [i for i in ids
                    if float((_quat_to_matrix(fm.tag_quaternion(i)) @ [1.0, 0.0, 0.0])
                             @ (c - fm.tag_translation(i))) <= 0]
            if len(away) * 2 > len(ids):
                return f"tags {away} would be seen from behind"
    return None


# Set by plausibility_gate_off(): the solve-time plausibility gate passes every pose.
_PLAUSIBILITY_GATE_OFF: ContextVar[bool] = ContextVar("plausibility_gate_off",
                                                      default=False)


@contextmanager
def plausibility_gate_off():
    """Within: check_solve_plausible passes every pose, so the solvers return their
    best-fitting pose even when it is upside down / mirrored / facing away. For a
    user-confirmed manual calibration of an unusual camera (an offseason broadcast
    from the far side of the field); the caller re-checks the final pose with
    ``check_solve_plausible(..., gate=False)`` to tell the user what it overrode."""
    tok = _PLAUSIBILITY_GATE_OFF.set(True)
    try:
        yield
    finally:
        _PLAUSIBILITY_GATE_OFF.reset(tok)


def check_solve_plausible(cal: "CameraCalibration", correspondences: dict,
                          season: Optional[str] = None,
                          gate: bool = True) -> Optional[str]:
    """pose_plausibility for a solve from ``correspondences`` ({key: (img, obj)}).
    ``gate`` (the solvers' use): None inside plausibility_gate_off(); pass False to
    always get the real verdict."""
    if gate and _PLAUSIBILITY_GATE_OFF.get():
        return None
    obj = np.vstack([np.asarray(o, float).reshape(-1, 3) for _, o in correspondences.values()])
    ids = [_tag_id_of_key(k) for k in correspondences]
    return pose_plausibility(cal, obj, _field_model(season or cal.field_season), ids)


def _pose_rms(obj, img, rvec, tvec, K, D, mode) -> float:
    if mode == "pinhole":
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.asarray(D, float).reshape(-1, 1))
    else:
        proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
    d = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _pose_multistart(correspondences: dict, K, D, mode: str, init=None):
    """Candidate poses for the PnP problem, each refined on ALL points: the plain
    DLT/iterative solve, the caller's initial pose, SQPnP, and BOTH IPPE solutions of
    every 4-corner tag (planar targets have a two-fold ambiguity, and points on one
    or two hub faces are close to planar — the iterative solve alone can land in the
    mirrored minimum). Returns [(rms_px, rvec, tvec)], best first."""
    obj = np.vstack([np.asarray(o, np.float64).reshape(-1, 3)
                     for _, o in correspondences.values()])
    img = np.vstack([np.asarray(i, np.float64).reshape(-1, 2)
                     for i, _ in correspondences.values()])
    img_pnp = img.reshape(-1, 1, 2)
    if mode == "pinhole":
        Dp = D.reshape(-1, 1)
        und = cv2.undistortPoints(img_pnp, K, Dp, P=K).reshape(-1, 1, 2)
    else:
        und = cv2.fisheye.undistortPoints(img_pnp, K, D, P=K).reshape(-1, 1, 2)
    seeds = []
    try:
        ok, r, t = cv2.solvePnP(obj, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            seeds.append((r, t))
    except cv2.error:
        pass
    if init is not None:
        seeds.append((np.asarray(init[0], float).reshape(3, 1),
                      np.asarray(init[1], float).reshape(3, 1)))
    if len(obj) >= 4:
        try:
            ok, r, t = cv2.solvePnP(obj, und, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                seeds.append((r, t))
        except cv2.error:
            pass
    off = 0
    for i2d, o3d in correspondences.values():
        n = len(np.asarray(o3d).reshape(-1, 3))
        if n == 4:
            try:
                k, rs, ts, _ = cv2.solvePnPGeneric(obj[off:off + 4], und[off:off + 4], K, None,
                                                   flags=cv2.SOLVEPNP_IPPE)
                seeds.extend(zip(rs[:k], ts[:k]))
            except cv2.error:
                pass
        off += n
    out = []
    for r, t in seeds:
        r = np.asarray(r, float).reshape(3, 1).copy()
        t = np.asarray(t, float).reshape(3, 1).copy()
        try:
            if mode == "pinhole":
                r, t = cv2.solvePnPRefineLM(obj, img_pnp, K, D.reshape(-1, 1), r, t)
            else:
                r, t = cv2.solvePnPRefineLM(obj, und, K, None, r, t)
            rms = _pose_rms(obj, img, r, t, K, D, mode)
        except cv2.error:
            continue
        if np.isfinite(rms):
            out.append((rms, r, t))
    out.sort(key=lambda x: x[0])
    return out


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

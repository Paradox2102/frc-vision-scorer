"""One lens shared by several corner cameras, fitted jointly with each camera's pose.

Broadcasts mount identical corner cameras in pairs (2025: two wide-lens cameras on
the barge). Each sees only ~5 tags: its own free lens over-fits (2025hop
corner_right: fx 1047, k1 -1.195, centre on the wrong side of the barge), and a lens
borrowed from a sibling fitted on ITS ~5 tags still leaves offsets. Fitting ONE lens
to both cameras' tags at once — fx = fy, principal point at each crop's centre, k1
only (k2 = 0: freed it diverges, f 45238 px) — plus a 6-DOF pose per camera, by
robust least squares, lines both overlays up (2025hop: f 692, k1 -0.151).

The shared focal is in pixels of the REFERENCE camera (the first in ``cameras``);
a camera of crop height h gets f * h / h_ref (panels are scaled copies of the
camera image; only the vertical scale is reliable — lens_borrow.transfer_lens).

Nothing here does IO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from web.field.calibration import (CalibrationError, CameraCalibration,
                                   check_solve_plausible)

JOINT_LOSS = "soft_l1"
JOINT_F_SCALE_PX = 5.0       # soft_l1 knee: clicks are good to a few px
JOINT_OUTLIER_PX = 10.0      # a point this far off is an outlier (frame-edge clicks)
F_PER_H_BOUNDS = (0.5, 12.0)  # focal / crop height
K1_BOUNDS = (-1.5, 1.0)
# default multi-start lenses (focal / crop height, k1) when no init lens is given;
# 2025hop corner lens: 692 / 347 = 2.0, k1 -0.15
DEFAULT_STARTS = ((2.0, 0.0), (2.0, -0.3), (1.4, -0.2), (3.0, 0.0))
MIN_JOINT_CAMERAS = 2
MIN_TAGS_PER_CAMERA = 2      # a pose needs >= 2 tags (MIN_POSE_TAGS)


@dataclass
class JointLensResult:
    calibrations: dict                  # camera -> CameraCalibration (shared lens)
    reference: str                      # the camera whose pixels f_px is in
    f_px: float
    k1: float
    residuals: dict                     # camera -> {key: [px per point]}
    rms_px: float                       # all points
    rms_inlier_px: float                # points under JOINT_OUTLIER_PX
    outliers: list                      # [{"camera", "key", "px"}], worst first
    plausibility: dict                  # camera -> None | reason (pose_plausibility)
    n_points: int = 0
    starts_tried: int = 0
    notes: list = field(default_factory=list)

    @property
    def plausible(self) -> bool:
        return all(v is None for v in self.plausibility.values())

    def camera_rms(self, camera: str) -> float:
        errs = [e for v in self.residuals[camera].values() for e in v]
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0

    def to_json(self) -> dict:
        return {"reference": self.reference, "f_px": round(self.f_px, 1),
                "k1": round(self.k1, 4), "rms_px": round(self.rms_px, 2),
                "rms_inlier_px": round(self.rms_inlier_px, 2),
                "n_points": self.n_points, "outliers": self.outliers,
                "plausibility": self.plausibility,
                "cameras": {c: {"rms_px": round(self.camera_rms(c), 2),
                                "f_px": round(float(cal.K[0, 0]), 1),
                                "center": [round(float(v), 3) for v in cal.camera_center],
                                "residuals": self.residuals[c]}
                            for c, cal in self.calibrations.items()}}


def _lens_K(f, size):
    w, h = size
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def _stack(corr):
    keys, img, obj, owner = [], [], [], []
    for k, (i, o) in corr.items():
        i = np.asarray(i, float).reshape(-1, 2)
        o = np.asarray(o, float).reshape(-1, 3)
        img.append(i); obj.append(o)
        owner.extend([str(k)] * len(i))
    return np.vstack(img), np.vstack(obj), owner


def _pose_for(corr, size, season, f, k1):
    from web.field import lens_borrow
    lens = CameraCalibration(mode="pinhole", K=_lens_K(f, size),
                             D=np.array([k1, 0.0, 0.0, 0.0, 0.0]), image_size=tuple(size),
                             crop_region=(0, 0, *size), field_season=season)
    cal = lens_borrow.solve_pose_with_lens(corr, size, season, lens)
    return (np.asarray(cal.rvec, float).reshape(3), np.asarray(cal.tvec, float).reshape(3))


def _fit(data, h_ref, f0, k10, poses0):
    """One robust least-squares run from (f0, k10, poses0). Returns (x, cost)."""
    from scipy.optimize import least_squares
    names = list(data)

    def unpack(x):
        return x[0], x[1], {c: (x[2 + 6 * i:5 + 6 * i], x[5 + 6 * i:8 + 6 * i])
                            for i, c in enumerate(names)}

    def resid(x):
        f, k1, poses = unpack(x)
        out = []
        D = np.array([k1, 0.0, 0.0, 0.0, 0.0])
        for c in names:
            img, obj, _owner, size = data[c]
            K = _lens_K(f * size[1] / h_ref, size)
            rv, tv = poses[c]
            uv, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K, D)
            out.append((uv.reshape(-1, 2) - img).reshape(-1))
        r = np.concatenate(out)
        return np.where(np.isfinite(r), r, 1e6)

    x0 = [f0, k10]
    for c in names:
        x0.extend(poses0[c][0]); x0.extend(poses0[c][1])
    x0 = np.asarray(x0, float)
    lo = np.full_like(x0, -np.inf); hi = np.full_like(x0, np.inf)
    lo[0], hi[0] = F_PER_H_BOUNDS[0] * h_ref, F_PER_H_BOUNDS[1] * h_ref
    lo[1], hi[1] = K1_BOUNDS
    x0[0] = np.clip(x0[0], lo[0] + 1e-6, hi[0] - 1e-6)
    x0[1] = np.clip(x0[1], lo[1] + 1e-6, hi[1] - 1e-6)
    sol = least_squares(resid, x0, bounds=(lo, hi), loss=JOINT_LOSS,
                        f_scale=JOINT_F_SCALE_PX, x_scale="jac", max_nfev=2000)
    return sol.x, float(sol.cost), unpack


def solve_joint_lens(cameras: dict, season: str, init: Optional[dict] = None,
                     starts=DEFAULT_STARTS) -> JointLensResult:
    """Fit one shared lens + a pose per camera. ``cameras``: {name: (corr, image_size)},
    corr = {key: (img (N,2), obj (N,3))} as for a single-camera solve; >= 2 cameras
    with >= 2 tags each. ``init`` (optional): {"f": focal in the reference camera's
    px, "k1": ..., "poses": {name: (rvec, tvec)}} — a camera with no init pose gets
    one from a pose-only solve with the start lens. Multi-start: ``init``'s lens first,
    then ``starts`` ((focal / crop height, k1) pairs); the lowest robust cost among
    starts whose poses are all plausible wins (else the lowest cost, flagged).
    Raises CalibrationError when nothing can be fitted."""
    names = [c for c in cameras]
    if len(names) < MIN_JOINT_CAMERAS:
        raise CalibrationError(f"a joint lens needs >= {MIN_JOINT_CAMERAS} cameras")
    data = {}
    for c in names:
        corr, size = cameras[c]
        if len(corr) < MIN_TAGS_PER_CAMERA:
            raise CalibrationError(f"{c}: a joint lens needs >= {MIN_TAGS_PER_CAMERA} "
                                   f"tags per camera (have {len(corr)})")
        size = (int(size[0]), int(size[1]))
        img, obj, owner = _stack(corr)
        data[c] = (img, obj, owner, size)
    ref = names[0]
    h_ref = float(data[ref][3][1])
    init = init or {}
    lens_starts = []
    if init.get("f") is not None:
        lens_starts.append((float(init["f"]), float(init.get("k1", 0.0)), True))
    lens_starts += [(fh * h_ref, k1, False) for fh, k1 in starts]

    best, notes = None, []
    tried = 0
    for f0, k10, explicit in lens_starts:
        poses0 = {}
        try:
            for c in names:
                p = (init.get("poses") or {}).get(c) if explicit else None
                if p is not None:
                    poses0[c] = (np.asarray(p[0], float).reshape(3),
                                 np.asarray(p[1], float).reshape(3))
                else:
                    size = data[c][3]
                    poses0[c] = _pose_for(cameras[c][0], size, season,
                                          f0 * size[1] / h_ref, k10)
        except CalibrationError as e:
            notes.append(f"start f {f0:.0f} k1 {k10:.2f}: no pose ({e})")
            continue
        tried += 1
        x, cost, unpack = _fit(data, h_ref, f0, k10, poses0)
        f, k1, poses = unpack(x)
        cals = {}
        for c in names:
            size = data[c][3]
            cals[c] = CameraCalibration(
                mode="pinhole", K=_lens_K(float(f) * size[1] / h_ref, size),
                D=np.array([float(k1), 0.0, 0.0, 0.0, 0.0]),
                rvec=np.asarray(poses[c][0], float).reshape(3, 1).copy(),
                tvec=np.asarray(poses[c][1], float).reshape(3, 1).copy(),
                image_size=size, crop_region=(0, 0, *size), field_season=season)
        plaus = {c: check_solve_plausible(cals[c], cameras[c][0], season) for c in names}
        ok = all(v is None for v in plaus.values())
        rank = (0 if ok else 1, cost)
        if best is None or rank < best[0]:
            best = (rank, float(f), float(k1), cals, plaus)
    if best is None:
        raise CalibrationError("joint lens: no start could be posed — "
                               + "; ".join(notes))
    _rank, f, k1, cals, plaus = best

    residuals, all_errs, outliers = {}, [], []
    for c in names:
        img, obj, owner, _size = data[c]
        uv, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), cals[c].rvec, cals[c].tvec,
                                  cals[c].K, cals[c].D.reshape(-1, 1))
        e = np.hypot(*(uv.reshape(-1, 2) - img).T)
        per = {}
        for k, err in zip(owner, e):
            per.setdefault(k, []).append(round(float(err), 2))
        residuals[c] = per
        for k, errs in per.items():
            worst = max(errs)
            if worst >= JOINT_OUTLIER_PX:
                outliers.append({"camera": c, "key": k, "px": worst})
        all_errs.extend(float(v) for v in e)
        cals[c].rms_error = float(np.sqrt(np.mean(np.square(e))))
    errs = np.asarray(all_errs)
    inl = errs[errs < JOINT_OUTLIER_PX]
    outliers.sort(key=lambda d: -d["px"])
    return JointLensResult(
        calibrations=cals, reference=ref, f_px=f, k1=k1, residuals=residuals,
        rms_px=float(np.sqrt(np.mean(errs ** 2))),
        rms_inlier_px=float(np.sqrt(np.mean(inl ** 2))) if len(inl) else float("nan"),
        outliers=outliers, plausibility=plaus, n_points=int(len(errs)),
        starts_tried=tried, notes=notes)

"""Locate UNDECODED AprilTags to grow a single-view calibration outward.

A broadcast view usually decodes only a handful of large, central tags; the tags near
the image edges are too small to decode, so a full camera solve from the decoded tags
fits them perfectly but extrapolates the lens badly to the edges. This module grows
the solve outward, staged and guarded:

  Stage A  solve from the decoded tags with a CONSTRAINED model (default "pp_k1":
           principal point at the crop centre, fx == fy, k1 only) that extrapolates
           sanely — on 2026cur qm11 main it predicted the undecoded tags within 17 px
           where the free k1+k2 model was off by up to 220 px. When growing from an
           existing epoch, the ladder starts at "pose" (that epoch's lens fixed).
  Stage B  for each model tag predicted visible + facing but not decoded — nearest to
           the already-solved tags first — search the (temporal-median) frame around
           its prediction:
             1. upscaled-ROI decode (the right id => 4 corners, no judge needed);
             2. else template matching: a synthetic "white printed border around a
                dark square" template drawn from the PREDICTED quad (so perspective and
                size come from the model). Pairs (two tags on one panel, derived from
                the field model) are matched jointly with a few px of slack in their
                separation, so identity comes from map position + pairing — the tags
                are too small to read. Low or ambiguous peaks are dropped;
             3. a template candidate is accepted only if a judge (Jev) says the crop
                shows the expected kind (single tag / pair) with p >= ``jev_min_p``.
           Each accepted location is re-solved first under a model no freer than the
           current one (so extra freedom cannot absorb a wrong point), then escalated
           to the freest model the point count allows. A location is kept only if it
           reprojects within ``max_residual_px``, the decoded tags do not get
           materially worse, and the radial distortion stays monotonic out to the
           points; the final model must stay monotonic over the field region too.

Nothing here does IO beyond the optional judge callable; ``build_jev_tag_payload`` +
``make_jev_judge`` wire Jev in (tests inject a fake judge, never the network).
"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from web.field.calibration import (CameraCalibration, CalibrationError, Calibrator,
                                   check_solve_plausible)

# Model ladder, least -> most free. "pp_*" fix the principal point at the crop centre
# and fx == fy; "k0" = no radial distortion, "k1" = k1 only, "full" = k1 + k2 (the
# existing Calibrator.solve_camera_from_tags model).
_BASE = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
MODEL_FLAGS = {
    "pp_k0": _BASE | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2,
    "pp_k1": _BASE | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_FIX_K2,
    "k0": _BASE | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2,
    "k1": _BASE | cv2.CALIB_FIX_K2,
    "full": _BASE,
}
# Lens-prior rungs (used when growing from an existing epoch's lens, e.g. after a
# pose re-solve): "pose" = lens fixed, only the 6-DOF pose; "f_lens" = also the focal
# length (a zoom change), principal point + distortion kept from the lens.
MODEL_FLAGS["f_lens"] = (_BASE | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                         | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2)
MODEL_ORDER = ("pp_k0", "pp_k1", "k0", "k1", "full")
LENS_MODEL_ORDER = ("pose", "f_lens", "k1", "full")
# Minimum number of tags (decoded + located) before a model may be used.
MODEL_MIN_TAGS = {"pose": 2, "f_lens": 3, "pp_k0": 3, "pp_k1": 3, "k0": 4, "k1": 6,
                  "full": 8}
_NFREE = {"pose": 0, "f_lens": 1, "pp_k0": 1, "pp_k1": 2, "k0": 4, "k1": 5, "full": 6}


def _order(cfg) -> tuple:
    """The model ladder starting at cfg.init_model."""
    seq = LENS_MODEL_ORDER if cfg.init_model in ("pose", "f_lens") else MODEL_ORDER
    return seq[seq.index(cfg.init_model):]


def _rank(cfg, m) -> int:
    seq = LENS_MODEL_ORDER if cfg.init_model in ("pose", "f_lens") else MODEL_ORDER
    return seq.index(m)

JUDGE_OPTIONS = ("single_apriltag", "pair_of_apriltags", "not_an_apriltag", "unclear")
JUDGE_CONTEXT = (
    "`crop` is cut from FRC (FIRST Robotics Competition) match video, centred on a spot "
    "where a camera-calibration tool predicts AprilTag fiducial markers are mounted on a "
    "field structure. An AprilTag is a square printed marker: a black square holding a "
    "pattern of black and white cells, printed with a white border. In low-resolution "
    "video it usually looks like a small bright white frame around a dark square, and the "
    "inner pattern is often unreadable. Some tags are mounted in pairs, side by side on "
    "the same panel. Consider only what is at the centre of the image; robots, balls, "
    "people, lights and other structures may be nearby.")
JUDGE_QUESTION = ("What is at the centre of the image? Choose `unclear` unless you can "
                  "actually see the marker shape.")
JUDGE_CRITERIA = {
    "single_apriltag": "Exactly one AprilTag marker at the centre (a white-framed dark square)",
    "pair_of_apriltags": "Two AprilTag markers side by side at the centre, mounted together",
    "not_an_apriltag": ("No AprilTag at the centre: a robot, ball, person, light, sign, "
                        "bumper, field structure or empty background"),
    "unclear": "Too blurry, dark or blocked to tell",
}


@dataclass
class LocatorConfig:
    init_model: str = "pp_k1"
    jev_verify: bool = True           # False: accept every geometric quad candidate
    jev_min_p: float = 0.6
    max_residual_px: float = 4.0      # a located point must reproject this well
    decoded_worsen_px: float = 1.5    # decoded-tag RMS may grow by at most this
    pair_max_sep_m: float = 0.6
    roi_upscale: int = 4
    min_half_px: float = 20.0
    max_half_px: float = 150.0
    ncc_min: float = 0.6              # template-match floor for a geometric candidate
    ambiguity_margin: float = 0.06    # peak must beat the next-best spot by this


@dataclass
class LocateResult:
    calibration: Optional[CameraCalibration]
    model: Optional[str]
    corr: dict                        # key -> (img (N,2), obj (N,3)); a<id>/r<id>/m<id>
    located: list                     # [{id, method, jev_p, residual_px, u, v}]
    candidates: list = field(default_factory=list)   # every judged/considered candidate
    notes: list = field(default_factory=list)


# ------------------------------------------------------------------- geometry

def _normal(fm, tid):
    from web.field.field_model import _quat_to_matrix
    return _quat_to_matrix(fm.tag_quaternion(tid)) @ np.array([1.0, 0.0, 0.0])


def tag_pairs(fm, max_sep_m: float = 0.6) -> dict:
    """{tag: partner} for tags mounted together: same facing, centres < max_sep_m."""
    ids = sorted(fm.tags)
    out = {}
    for i in ids:
        best, bd = None, max_sep_m
        for j in ids:
            if j == i or float(_normal(fm, i) @ _normal(fm, j)) < 0.95:
                continue
            d = float(np.linalg.norm(fm.tag_translation(i) - fm.tag_translation(j)))
            if d < bd:
                best, bd = j, d
        if best is not None:
            out[i] = best
    return {i: j for i, j in out.items() if out.get(j) == i}


def _cam_z(cal, p):
    R, _ = cv2.Rodrigues(np.asarray(cal.rvec, float))
    return float((R @ np.asarray(p, float) + np.asarray(cal.tvec, float).reshape(3))[2])


def _faces(cal, fm, tid) -> bool:
    c = cal.camera_center
    return c is None or float(_normal(fm, tid) @ (np.asarray(c) - fm.tag_translation(tid))) > 0


def predict_tag(cal, fm, tid, image_size):
    """(centre (2,), corners (4,2), side_px) under ``cal``, or None if the tag is behind
    the camera, faces away, or its centre is outside the image."""
    if not _faces(cal, fm, tid):
        return None
    c3 = fm.tag_corners_3d(tid)
    if min(_cam_z(cal, p) for p in c3) <= 0:
        return None
    ctr = np.array(cal.field_to_image(tuple(fm.tag_translation(tid))), float)
    w, h = image_size
    if not (np.all(np.isfinite(ctr)) and 0 <= ctr[0] < w and 0 <= ctr[1] < h):
        return None
    pc = np.array([cal.field_to_image(tuple(p)) for p in c3], float)
    side = float(np.mean(np.linalg.norm(pc - np.roll(pc, 1, axis=0), axis=1)))
    return ctr, pc, side


def solve_model(corr: dict, image_size, season: str, model: str,
                prev: Optional[CameraCalibration] = None,
                lens: Optional[CameraCalibration] = None) -> CameraCalibration:
    """Single-view calibrateCamera under ``model``'s constraints. ``corr`` values are
    (img (N,2), obj (N,3)). Tries a cold start and (when given) a warm start from
    ``prev`` — calibrateCamera's internal pose init can land in a bad minimum when
    new edge points arrive — and keeps the lower-RMS solve."""
    if model == "pose":
        if lens is None or lens.mode != "pinhole":
            raise CalibrationError("pose model needs a pinhole lens prior")
        if len(corr) < 2:
            raise CalibrationError("pose model needs >= 2 correspondences")
        init = ((prev.rvec, prev.tvec) if prev is not None and prev.rvec is not None
                else ((lens.rvec, lens.tvec) if lens.rvec is not None else None))
        cal = Calibrator.solve_pose(corr, K=lens.K, D=lens.D, image_size=tuple(image_size),
                                    field_season=season, mode="pinhole", init=init)
        cal.crop_region = (0, 0, int(image_size[0]), int(image_size[1]))
        return cal
    best, err = None, None
    if model == "f_lens":
        if lens is None or lens.mode != "pinhole":
            raise CalibrationError("f_lens model needs a pinhole lens prior")
        return _solve_model_once(corr, image_size, season, model, prev, lens)
    for p in ((None, prev) if prev is not None else (None,)):
        try:
            c = _solve_model_once(corr, image_size, season, model, p)
        except CalibrationError as e:
            err = e
            continue
        if best is None or c.rms_error < best.rms_error:
            best = c
    if best is None:
        raise err or CalibrationError(f"model {model} solve failed")
    return best


def _solve_model_once(corr: dict, image_size, season: str, model: str,
                      prev: Optional[CameraCalibration] = None,
                      lens: Optional[CameraCalibration] = None) -> CameraCalibration:
    img = np.vstack([np.asarray(i, np.float64).reshape(-1, 2) for i, _ in corr.values()])
    obj = np.vstack([np.asarray(o, np.float64).reshape(-1, 3) for _, o in corr.values()])
    nfree = _NFREE[model] + 6
    if 2 * len(img) < nfree + 2:
        raise CalibrationError(f"{len(img)} points too few for model {model}")
    w, h = int(image_size[0]), int(image_size[1])
    f0 = float(w)
    K0 = np.array([[f0, 0., w / 2.0], [0., f0, h / 2.0], [0., 0., 1.]])
    D0 = np.zeros(5)
    if prev is not None and prev.K is not None:
        f0 = float(np.sqrt(prev.K[0, 0] * prev.K[1, 1]))
        K0[0, 0] = K0[1, 1] = f0
        if not model.startswith("pp_"):
            K0[0, 0], K0[1, 1] = prev.K[0, 0], prev.K[1, 1]
            K0[0, 2], K0[1, 2] = prev.K[0, 2], prev.K[1, 2]
        if model in ("pp_k1", "k1", "full"):
            D0[0] = float(np.asarray(prev.D).reshape(-1)[0])
    if model == "f_lens":
        K0 = np.asarray(lens.K, float).copy()
        if prev is not None and prev.K is not None:
            s_ = float(np.sqrt(prev.K[0, 0] * prev.K[1, 1] / (K0[0, 0] * K0[1, 1])))
            K0[0, 0] *= s_
            K0[1, 1] *= s_
        D0 = np.zeros(5)
        Dl = np.asarray(lens.D, float).reshape(-1)
        D0[:min(5, len(Dl))] = Dl[:5]
    try:
        rms, K, D, rv, tv = cv2.calibrateCamera(
            [obj.astype(np.float32).reshape(-1, 1, 3)], [img.astype(np.float32).reshape(-1, 1, 2)],
            (w, h), K0, D0, flags=MODEL_FLAGS[model])
    except cv2.error as e:
        raise CalibrationError(f"calibrateCamera ({model}) failed: {str(e)[:120]}")
    cal = CameraCalibration(mode="pinhole", K=np.asarray(K, float),
                            D=np.asarray(D, float).reshape(-1),
                            rvec=np.asarray(rv[0], float).reshape(3, 1),
                            tvec=np.asarray(tv[0], float).reshape(3, 1),
                            image_size=(w, h), crop_region=(0, 0, w, h),
                            field_season=season, rms_error=float(rms))
    why = check_solve_plausible(cal, corr, season)
    if why is not None:                 # mirrored / flipped minimum: not a candidate
        raise CalibrationError(f"{model}: implausible pose ({why})")
    return cal


def distortion_monotonic(cal, image_size, points=None) -> bool:
    """True if the radial distortion r -> r(1 + k1 r^2 + k2 r^4 + k3 r^6) keeps
    increasing until it covers the farthest image corner (no fold-back inside the
    image — the k3-style failure where corners map back toward the centre).

    ``points`` (image px, (N,2)): check only out to the farthest of these instead of
    the crop corners — the region the calibration is actually used in."""
    if cal.K is None or cal.D is None:
        return True
    D = np.asarray(cal.D, float).reshape(-1)
    k1 = D[0] if len(D) > 0 else 0.0
    k2 = D[1] if len(D) > 1 else 0.0
    k3 = D[4] if len(D) > 4 else 0.0
    w, h = image_size
    fx, fy, cx, cy = cal.K[0, 0], cal.K[1, 1], cal.K[0, 2], cal.K[1, 2]
    if points is not None and len(points):
        P = np.asarray(points, float).reshape(-1, 2)
        rd_max = float(np.max(np.hypot((P[:, 0] - cx) / fx, (P[:, 1] - cy) / fy))) * 1.05
    else:
        rd_max = max(np.hypot((u - cx) / fx, (v - cy) / fy) for u in (0, w) for v in (0, h))
    rs = np.linspace(0.0, 4.0 * rd_max + 1.0, 4000)
    r2 = rs * rs
    f = rs * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3)
    reach = np.nonzero(f >= rd_max)[0]
    if len(reach) == 0:
        return False
    return bool(np.all(np.diff(f[: reach[0] + 1]) > 0))


def used_region_points(cal, corr: dict, fm, image_size, step_m: float = 1.0):
    """Image points the calibration must stay monotonic out to: every correspondence
    plus field-boundary samples (z=0) that project inside the image."""
    w, h = image_size
    pts = [np.asarray(i, float).reshape(-1, 2) for i, _ in corr.values()]
    L, W = fm.length_m, fm.width_m
    edge = ([(x, 0.0) for x in np.arange(0, L + 1e-6, step_m)]
            + [(x, W) for x in np.arange(0, L + 1e-6, step_m)]
            + [(0.0, y) for y in np.arange(0, W + 1e-6, step_m)]
            + [(L, y) for y in np.arange(0, W + 1e-6, step_m)])
    extra = []
    for x, y in edge:
        if cal.rvec is not None and _cam_z(cal, (x, y, 0.0)) <= 0:
            continue
        u, v = cal.field_to_image((x, y, 0.0))
        if np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h:
            extra.append((u, v))
    if extra:
        pts.append(np.array(extra, float))
    return np.vstack(pts) if pts else None


def monotonic_ok(cal, corr, fm, image_size) -> bool:
    return distortion_monotonic(cal, image_size,
                                used_region_points(cal, corr, fm, image_size))


def _rms_px(cal, corr: dict, keys) -> float:
    errs = []
    for k in keys:
        i2, o3 = corr[k]
        for (u, v), p in zip(np.asarray(i2).reshape(-1, 2), np.asarray(o3).reshape(-1, 3)):
            pu, pv = cal.field_to_image(tuple(p))
            errs.append((pu - u) ** 2 + (pv - v) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else 0.0


def _tag_of(key: str) -> int:
    return int(key[1:])


# ---------------------------------------------------------------- candidates

def decode_in_roi(img, center, half, upscale, detect_fn) -> dict:
    """Upscaled-ROI decode around ``center``: {tag_id: corners (4,2) in img coords}."""
    h, w = img.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, y0 = max(0, cx - int(half)), max(0, cy - int(half))
    x1, y1 = min(w, cx + int(half)), min(h, cy + int(half))
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return {}
    up = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return {int(t): np.asarray(c, float) / upscale + np.array([x0, y0], float)
            for t, c in detect_fn(up).items()}


_DETECTORS = threading.local()


def thread_detector(**params):
    """A pupil_apriltags Detector private to the calling thread, created once and
    NEVER destroyed: ``Detector.__del__`` segfaults intermittently (seen on qm22 when
    per-run detectors were garbage-collected), and a shared detector is not
    thread-safe — so each worker thread keeps its own for the process lifetime."""
    cache = getattr(_DETECTORS, "cache", None)
    if cache is None:
        cache = _DETECTORS.cache = {}
    key = tuple(sorted(params.items()))
    det = cache.get(key)
    if det is None:
        det = cache[key] = _immortal_detector_cls()(families="tag36h11", **params)
        _KEEPALIVE.append(det)          # survive thread exit too
    return det


_KEEPALIVE: list = []
_IMMORTAL = None


def _immortal_detector_cls():
    """Detector whose destructor never runs the C teardown (it segfaults, including
    during interpreter-exit garbage collection). Cached detectors live for the
    process anyway, so skipping the free only matters at exit."""
    global _IMMORTAL
    if _IMMORTAL is None:
        from pupil_apriltags import Detector

        class _ImmortalDetector(Detector):
            def __del__(self):
                pass
        _IMMORTAL = _ImmortalDetector
    return _IMMORTAL


def make_roi_detect_fn():
    """AprilTag detection tuned for upscaled small ROIs: no quad decimation (the
    default 2x decimation throws away exactly the detail an upscaled tiny tag needs)
    and mild decode sharpening. Uses the calling thread's cached detector."""
    def detect(frame_bgr):
        det = thread_detector(quad_decimate=1.0, decode_sharpening=0.5)
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        return {int(d.tag_id): np.asarray(d.corners, float) for d in det.detect(g)}
    return detect


def group_template(pcs, upscale: int, margin: float = 1.4, inner_gray: int = 110):
    """Synthetic appearance template for one or more PREDICTED tag quads (absolute
    image px): each tag = a white printed border (``margin`` x the tag) around a dark
    square whose inner 6x6-bit area averages to mid grey. Returns (T, M, lo) where
    T/M are the upscaled template/mask and ``lo`` the absolute px of T's top-left."""
    outers = [(np.asarray(pc, float) - np.mean(pc, axis=0)) * margin + np.mean(pc, axis=0)
              for pc in pcs]
    allp = np.vstack(outers)
    lo = np.floor(allp.min(axis=0)) - 2
    hi = np.ceil(allp.max(axis=0)) + 2
    W, H = (int((hi - lo)[0] * upscale) + 1, int((hi - lo)[1] * upscale) + 1)
    T = np.zeros((H, W), np.uint8)
    M = np.zeros((H, W), np.uint8)
    q = lambda P: np.round((np.asarray(P, float) - lo) * upscale).astype(np.int32)
    for pc, outer in zip(pcs, outers):
        pc = np.asarray(pc, float)
        c = pc.mean(axis=0)
        cv2.fillConvexPoly(T, q(outer), 230)
        cv2.fillConvexPoly(M, q(outer), 255)
        cv2.fillConvexPoly(T, q(pc), 30)
        cv2.fillConvexPoly(T, q((pc - c) * 0.75 + c), inner_gray)
    T = cv2.GaussianBlur(T, (0, 0), upscale * 0.8)
    return T, M, lo


def tag_ncc_map(img, pc, half: int, upscale: int = 4):
    """Template NCC of one predicted tag quad over shifts in [-half, +half] px (both
    axes): map[i, j] is the score at shift (j / upscale - half, i / upscale - half).
    The grid is the same for every tag, so maps of different tags can be combined."""
    T, M, lo = group_template([pc], upscale)          # lo is integer px
    th, tw = T.shape
    r0 = (int(lo[0]) - half, int(lo[1]) - half)
    rw = int(np.ceil(tw / upscale)) + 2 * half + 2
    rh = int(np.ceil(th / upscale)) + 2 * half + 2
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    roi = np.full((rh, rw), 128, np.uint8)
    sx0, sy0 = max(0, r0[0]), max(0, r0[1])
    sx1, sy1 = min(w, r0[0] + rw), min(h, r0[1] + rh)
    n = 2 * half * upscale + 1
    if sx1 <= sx0 or sy1 <= sy0:
        return np.full((n, n), -1.0, np.float32)
    roi[sy0 - r0[1]:sy1 - r0[1], sx0 - r0[0]:sx1 - r0[0]] = gray[sy0:sy1, sx0:sx1]
    R = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    res = cv2.matchTemplate(R, T, cv2.TM_CCOEFF_NORMED, mask=M)
    res[~np.isfinite(res) | (res > 0.999)] = -1.0      # flat / padded windows
    # Only shifts whose template window lies fully inside the image are valid.
    xlo, xhi = (sx0 - r0[0]) * upscale, (sx1 - r0[0]) * upscale
    ylo, yhi = (sy0 - r0[1]) * upscale, (sy1 - r0[1]) * upscale
    jj = np.arange(res.shape[1])
    ii = np.arange(res.shape[0])
    res[:, (jj < xlo) | (jj + tw > xhi)] = -1.0
    res[(ii < ylo) | (ii + th > yhi), :] = -1.0
    return res[:n, :n].astype(np.float32)


def _disk(r: int):
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def match_group(img, pcs, half: float, upscale: int = 4, side: float = 10.0,
                 tol: float = 4.0, anchor_shift=None) -> dict:
    """Locate one tag or a PAIR (pcs = predicted quads) near its prediction.

    Each tag gets its own template NCC map; a pair's score at shift s is
    (ncc_a(s) + max ncc_b within ``tol`` px of s) / 2, so the pair geometry fixes
    identity while tolerating a few px of error in the predicted separation.
    ``anchor_shift``: the shift that puts the LAST quad on its already-known position
    — then only the first quad is searched, within ``tol`` of that shift.
    Returns {ncc, shifts: [(dx, dy) per quad], second} (second = best score at least
    ~one tag side away from the peak; ambiguity)."""
    half = int(round(half))
    maps = [tag_ncc_map(img, pc, half, upscale) for pc in pcs]
    to_shift = lambda ij: (ij[1] / upscale - half, ij[0] / upscale - half)
    rad = max(1, int(round(tol * upscale)))
    if anchor_shift is not None:
        a = maps[0].copy()
        ai = (int(round((anchor_shift[1] + half) * upscale)),
              int(round((anchor_shift[0] + half) * upscale)))
        keep = np.zeros(a.shape, np.uint8)
        cv2.circle(keep, (ai[1], ai[0]), rad, 1, -1)
        a[keep == 0] = -1.0
        i, j = np.unravel_index(int(np.argmax(a)), a.shape)
        score = float(a[i, j])
        shifts = [to_shift((i, j)), (float(anchor_shift[0]), float(anchor_shift[1]))]
        comb = maps[0]
    elif len(maps) == 1:
        comb = maps[0]
        i, j = np.unravel_index(int(np.argmax(comb)), comb.shape)
        score = float(comb[i, j])
        shifts = [to_shift((i, j))]
    else:
        bmax = cv2.dilate(maps[1], _disk(rad))
        comb = 0.5 * (maps[0] + bmax)
        i, j = np.unravel_index(int(np.argmax(comb)), comb.shape)
        score = float(comb[i, j])
        b = np.full(maps[1].shape, -1.0, np.float32)
        keep = np.zeros(b.shape, np.uint8)
        cv2.circle(keep, (int(j), int(i)), rad, 1, -1)
        b[keep > 0] = maps[1][keep > 0]
        bi, bj = np.unravel_index(int(np.argmax(b)), b.shape)
        shifts = [to_shift((i, j)), to_shift((bi, bj))]
    sup = comb.copy()
    cv2.circle(sup, (int(j), int(i)), max(2, int(round(0.8 * side * upscale))), -1.0, -1)
    second = float(sup.max()) if np.any(sup > -1.0) else None
    return {"ncc": score, "shifts": [(float(x), float(y)) for x, y in shifts],
            "second": second}


def judge_crop(img, center, size_px, out_px: int = 320) -> bytes:
    """Square crop (gray-padded at the borders) around ``center`` upscaled for the judge."""
    h, w = img.shape[:2]
    half = max(8, int(round(size_px / 2)))
    cx, cy = int(round(center[0])), int(round(center[1]))
    canvas = np.full((2 * half, 2 * half, 3), 128, np.uint8)
    x0, y0 = cx - half, cy - half
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + 2 * half), min(h, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    canvas = cv2.resize(canvas, (out_px, out_px), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def judge_p(probs: dict, kind: str) -> float:
    """Acceptance probability for a candidate of ``kind``. A pair needs
    p(pair_of_apriltags). A single accepts p(single) + p(pair): the crop around a lone
    tag often catches a neighbouring tag too (wide 6x-side context), and "pair" still
    says the centre is AprilTag, which is the question being asked."""
    p = float(probs.get(kind, 0.0))
    if kind == "single_apriltag":
        p += float(probs.get("pair_of_apriltags", 0.0))
    return min(1.0, p)


def build_jev_tag_payload(model_key: str, jpg: bytes):
    """Jev payload (window-payload style) for the tag-kind question."""
    from web.jev_reader import JEV_MODELS, jev_asset
    asset, wh = jev_asset(jpg)
    return ({"model": JEV_MODELS[model_key]["model"],
             "state": {"crop": "crop_image", "context": JUDGE_CONTEXT},
             "assets": {"crop_image": asset},
             "questions": {"kind": {"type": "choice", "instructions": JUDGE_QUESTION,
                                    "criteria": dict(JUDGE_CRITERIA)}}}, [wh])


def make_jev_judge(session) -> Callable:
    """judge(jpg) -> {"probs": {option: p}} | {"error": msg} over a JevSession."""
    def judge(jpg: bytes) -> dict:
        payload, sizes = build_jev_tag_payload(session.model_key, jpg)
        return session.ask([(payload, sizes)], "kind")[0]
    return judge


# ------------------------------------------------------------------- the loop

def _finalize(corr, image_size, season, n_tags, cfg, cal, model, dec_keys, base_dec_rms,
              fm, notes, lens=None):
    """The final model must stay monotonic over the whole used region (field
    boundary + every point), not just near the points so far: try the eligible
    models freest first; keep the intermediate result (with a note) if none passes."""
    if monotonic_ok(cal, corr, fm, image_size):
        return cal, model
    models = [m for m in _order(cfg) if n_tags >= MODEL_MIN_TAGS[m]]
    for m in reversed(models):
        try:
            c = solve_model(corr, image_size, season, m, cal, lens)
        except CalibrationError:
            continue
        if (monotonic_ok(c, corr, fm, image_size)
                and _rms_px(c, corr, dec_keys) <= base_dec_rms + cfg.decoded_worsen_px):
            return c, m
    notes.append(f"final model {model} folds back near the image edge")
    return cal, model


def _points_monotonic(cal, image_size, corr, extra_pts=()) -> bool:
    pts = [np.asarray(i, float).reshape(-1, 2) for i, _ in corr.values()]
    if len(extra_pts):
        pts.append(np.asarray(extra_pts, float).reshape(-1, 2))
    return distortion_monotonic(cal, image_size, np.vstack(pts))


def _check(cal, corr, image_size, dec_keys, base_dec_rms, keys, cfg):
    """None if ``cal`` passes the guardrails for ``keys`` (located points), else why."""
    if not _points_monotonic(cal, image_size, corr):
        return "distortion folds back inside the points"
    dec = _rms_px(cal, corr, dec_keys)
    if dec > base_dec_rms + cfg.decoded_worsen_px:
        return f"decoded tags worsen to {dec:.1f} px"
    worst = max((_rms_px(cal, corr, [k]) for k in keys), default=0.0)
    if worst > cfg.max_residual_px:
        return f"located point residual {worst:.1f} px"
    return None


def _resolve_points(corr, image_size, season, n_tags, cfg, prev, prev_model, dec_keys,
                    base_dec_rms, new_keys, lens=None):
    """Accept the new points under a model NO freer than the current one (so extra
    freedom can't absorb a wrong point), then try to escalate to a freer eligible
    model, kept only if every located point still reprojects within
    cfg.max_residual_px. Returns (cal, model) or (None, why)."""
    cur = _rank(cfg, prev_model)
    eligible = [m for m in _order(cfg) if n_tags >= MODEL_MIN_TAGS[m]]
    base = [m for m in eligible if _rank(cfg, m) <= cur]
    accepted, first = None, None
    for m in reversed(base):
        try:
            cal = solve_model(corr, image_size, season, m, prev, lens)
        except CalibrationError as e:
            first = first or str(e)
            continue
        why = _check(cal, corr, image_size, dec_keys, base_dec_rms, new_keys, cfg)
        if why is None:
            accepted = (cal, m)
            break
        first = first or f"{m}: {why}"
    last = first or "no eligible model"
    if accepted is None:
        return None, last
    all_located = [k for k in corr if k[0] in "rm"]
    for m in reversed([m for m in eligible if _rank(cfg, m) > cur]):
        try:
            cal = solve_model(corr, image_size, season, m, accepted[0], lens)
        except CalibrationError:
            continue
        if _check(cal, corr, image_size, dec_keys, base_dec_rms, all_located, cfg) is None:
            return cal, m
    return accepted


def locate_undecoded(img, fm, decoded: dict, image_size, season: str,
                     cfg: Optional[LocatorConfig] = None, detect_fn=None,
                     judge: Optional[Callable] = None, log: Callable = print,
                     lens: Optional[CameraCalibration] = None) -> LocateResult:
    """Grow a calibration outward from the decoded tags (see module docstring).

    img: the camera crop to search (ideally the temporal median). decoded:
    {tag_id: corners (4,2)} (known tags only). detect_fn: detector for upscaled-ROI
    decodes (``make_roi_detect_fn``); None skips ROI decoding. judge:
    callable(jpg)->{"probs"}|{"error"}; required for template candidates when
    cfg.jev_verify. lens: an existing pinhole calibration whose lens seeds the
    "pose"/"f_lens" ladder (cfg.init_model "pose"). Returns the final calibration
    (None if Stage A failed)."""
    cfg = cfg or LocatorConfig()
    corr = {f"a{t}": (np.asarray(c, float), fm.tag_corners_3d(t)) for t, c in decoded.items()}
    dec_keys = list(corr)
    notes, located, cands_log = [], [], []
    cal, model = None, None
    # Stage A: the constrained model only has to behave out to the decoded points;
    # predictions past a fold simply fail to match and are retried after the model
    # improves. Far-edge fold-back of the final model is checked at the end.
    for m in _order(cfg):
        try:
            c = solve_model(corr, image_size, season, m, None, lens)
        except CalibrationError:
            continue
        if _points_monotonic(c, image_size, corr):
            cal, model = c, m
            break
    if cal is None:
        return LocateResult(None, None, corr, [], [], ["stage A solve failed"])
    base_dec_rms = _rms_px(cal, corr, dec_keys)
    quads_ok = True
    if cfg.jev_verify and judge is None:
        quads_ok = False
        notes.append("Jev unavailable — template candidates skipped (ROI decodes only)")
    pairs = tag_pairs(fm, cfg.pair_max_sep_m)
    known = set(decoded)
    tried = {}                               # tag -> len(known) when last attempted
    centre_of = {t: np.asarray(c, float).mean(axis=0) for t, c in decoded.items()}

    def pending_groups():
        groups = set()
        for t in sorted(fm.tags):
            if t in known or tried.get(t, -1) >= len(known) \
                    or predict_tag(cal, fm, t, image_size) is None:
                continue
            p = pairs.get(t)
            if (p is not None and (p in known or tried.get(p, -1) < len(known))
                    and predict_tag(cal, fm, p, image_size) is not None):
                groups.add(tuple(sorted((t, p))))
            else:
                groups.add((t,))
        solved = [fm.tag_translation(t) for t in known]

        def dist(g):
            pts = [fm.tag_translation(t) for t in g if t not in known]
            return min(np.linalg.norm(a - b) for a in pts for b in solved)
        return sorted(groups, key=dist)

    while True:
        groups = pending_groups()
        if not groups:
            break
        g = groups[0]
        targets = [t for t in g if t not in known]
        tried.update({t: len(known) for t in targets})
        preds = {t: predict_tag(cal, fm, t, image_size) for t in g}
        if any(preds[t] is None for t in g):
            continue
        side = float(np.mean([preds[t][2] for t in targets]))
        ctr = np.mean([preds[t][0] for t in targets], axis=0)
        near = min(np.linalg.norm(ctr - c) for c in centre_of.values()) if centre_of else 500
        half = float(np.clip(1.5 * side + 0.12 * near, cfg.min_half_px, cfg.max_half_px))
        new, found = {}, {}
        # 1. upscaled-ROI decode: the right id gives 4 corners, no judge needed
        if detect_fn is not None:
            span = max(np.linalg.norm(preds[t][0] - ctr) for t in g)
            dec = decode_in_roi(img, ctr, half + span + side, cfg.roi_upscale, detect_fn)
            for t in targets:
                if t in dec:
                    new[f"r{t}"] = (dec[t], fm.tag_corners_3d(t))
                    found[t] = {"method": "roi_decode", "jev_p": None,
                                "u": float(dec[t][:, 0].mean()),
                                "v": float(dec[t][:, 1].mean())}
        # 2. template candidate for the whole group (pair geometry = identity)
        rest = [t for t in targets if t not in found]
        if rest and quads_ok:
            kind = "pair_of_apriltags" if len(g) == 2 else "single_apriltag"
            anchors = [t for t in g if t not in rest]
            order = rest + anchors                    # searched quad(s) first
            sep = (float(np.linalg.norm(preds[g[0]][0] - preds[g[1]][0]))
                   if len(g) == 2 else 0.0)
            tol = max(3.0, 0.35 * sep)
            anchor_shift = None
            if anchors:
                ak = anchors[0]
                kpos = centre_of[ak] if ak in centre_of else np.array(
                    [found[ak]["u"], found[ak]["v"]])
                anchor_shift = tuple(kpos - preds[ak][0])
            mt = match_group(img, [preds[t][1] for t in order], half, cfg.roi_upscale,
                             side, tol=tol, anchor_shift=anchor_shift)
            cand = {t: (float(preds[t][0][0] + sh[0]), float(preds[t][0][1] + sh[1]))
                    for t, sh in zip(order, mt["shifts"])}
            dx, dy = mt["shifts"][0]
            jc = np.mean([cand[t] for t in g], axis=0)
            jsize = ((float(np.linalg.norm(preds[g[0]][0] - preds[g[1]][0])) + 3.0 * side)
                     if len(g) == 2 else 6.0 * side)   # wider context reads better (Jev bench)
            entry = {"ids": sorted(rest), "group": list(g), "kind": kind,
                     "center": [float(jc[0]), float(jc[1])], "size": float(jsize),
                     "ncc": round(mt["ncc"], 3),
                     "second": None if mt["second"] is None else round(mt["second"], 3),
                     "shift": [round(dx, 1), round(dy, 1)], "tol_px": round(tol, 1),
                     "anchored": bool(anchors), "probs": None}
            geo_ok, why = True, ""
            if mt["ncc"] < cfg.ncc_min:
                geo_ok, why = False, f"template ncc {mt['ncc']:.2f} < {cfg.ncc_min}"
            elif mt["second"] is not None and mt["second"] > mt["ncc"] - cfg.ambiguity_margin:
                geo_ok, why = False, (f"ambiguous template peak ({mt['ncc']:.2f} vs "
                                      f"{mt['second']:.2f})")
            if not geo_ok:
                entry["verdict"] = f"no candidate: {why}"
            elif not cfg.jev_verify:
                entry["verdict"] = "accepted (no judge)"
            else:
                res = judge(judge_crop(img, jc, max(jsize, 12.0)))
                if "error" in res:
                    entry["verdict"] = f"judge error: {res['error']}"
                else:
                    entry["probs"] = {k: round(float(v), 3) for k, v in res["probs"].items()}
                    pk = judge_p(res["probs"], kind)
                    entry["verdict"] = ("accepted" if pk >= cfg.jev_min_p
                                        else f"rejected by judge (p={pk:.2f})")
            cands_log.append(entry)
            if entry["verdict"].startswith("accepted"):
                pk = (None if entry["probs"] is None else judge_p(entry["probs"], kind))
                for t in rest:
                    c3 = fm.tag_corners_3d(t).mean(axis=0).reshape(1, 3)
                    new[f"m{t}"] = (np.array([cand[t]], float), c3)
                    found[t] = {"method": "quad_jev" if cfg.jev_verify else "quad",
                                "jev_p": pk, "u": cand[t][0], "v": cand[t][1]}
        if not new:
            continue
        trial = dict(corr)
        trial.update(new)
        n_tags = len({_tag_of(k) for k in trial})
        ncal, nmodel = _resolve_points(trial, image_size, season, n_tags, cfg, cal, model,
                                       dec_keys, base_dec_rms, list(new), lens)
        if ncal is None:
            notes.append(f"tags {sorted(found)} rejected: {nmodel}")
            if cands_log and cands_log[-1]["verdict"].startswith("accepted"):
                cands_log[-1]["verdict"] += f"; re-solve rejected ({nmodel})"
            continue
        cal, model, corr = ncal, nmodel, trial
        for t, f in found.items():
            resid = _rms_px(cal, corr, [k for k in new if _tag_of(k) == t])
            located.append({"id": int(t), "method": f["method"],
                            "jev_p": None if f["jev_p"] is None else round(f["jev_p"], 3),
                            "residual_px": round(resid, 2),
                            "u": round(f["u"], 2), "v": round(f["v"], 2)})
            known.add(t)
            centre_of[t] = np.array([f["u"], f["v"]])
        log(f"  [auto-calib] located {sorted(found)} via "
            f"{sorted({f['method'] for f in found.values()})}; model {model}, "
            f"rms {cal.rms_error:.2f} px")
    n_tags = len({_tag_of(k) for k in corr})
    cal, model = _finalize(corr, image_size, season, n_tags, cfg, cal, model, dec_keys,
                           base_dec_rms, fm, notes, lens)
    for d in located:            # residuals under the final model
        d["residual_px"] = round(_rms_px(cal, corr, [k for k in corr
                                                     if k[0] in "rm" and _tag_of(k) == d["id"]]), 2)
    return LocateResult(cal, model, corr, located, cands_log, notes)

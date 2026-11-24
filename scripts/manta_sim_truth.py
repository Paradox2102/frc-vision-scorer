"""Manta Sim truth -> our ground-truth docs and exact camera calibrations.

Team 3128's Manta Sim renders a synthetic broadcast of a match together with exact
ground truth (``manta-broadcast-v1``): per-frame robot poses + bumper corners and the
exact render cameras. This tool turns that truth into the two things our pipeline
and scorers consume, and grades our own camera solves against it.

Inputs (``--sim-dir``, default data/sim_qm8)
--------------------------------------------
* ``truth-1080p60.json``       — cameras, lens model, per-panel projection matrices,
                                 AprilTag corners, per-frame robot truth.
* ``image-truth-1080p60.json`` — per frame / camera panel / robot: the projected
                                 bumper box (full-frame px, clipped to the panel)
                                 and the visible share (HUB / robot occlusion).
* the broadcast video, as ``data/test_footage/<match_key>.mp4`` (frame index =
  truth frame index; frame 0 = AUTO start).

Conventions (checked against the files, see ``_check_conventions``)
------------------------------------------------------------------
* Field: both sides use the WPILib field frame in metres (origin at the blue
  alliance wall right corner, +x toward red, +y to the blue drivers' left, +z up);
  the sim's tag corners equal our 2026-rebuilt-welded layout's.
* Sim camera: ``fieldToCamera`` is field metres -> OpenGL camera (+x right, +y up,
  looking down -z). OpenCV (ours, x right / y down / z forward) is that frame
  flipped by diag(1, -1, -1), so rvec/tvec = Rodrigues(F R), F t.
* Sim lens (``truth.lens.model``): an out->src radial map,
  ``p_src = p_out (1 + k r^2) / (1 + k)`` with p in half-panel-HEIGHT units from the
  panel centre and r = 1 at the panel corners; p_src is the ideal pinhole image
  (focal fxPx). OpenCV's model is src->out, so we least-squares fit
  D = [k1, k2, 0, 0, k3] (pinhole mode) to the exact inverse over the panel.
* Pixels: the sim's are continuous ((0, 0) = top-left CORNER of the top-left pixel).
  Our calibrations are solved from pupil_apriltags corners, which use the same
  continuous convention (measured on this video: detected - projected = +0.45 px
  in x and y under a pixel-centre convention, ~0 under the continuous one), and
  YOLO box edges are continuous too — so ours = sim (``PX_OFFSET`` = 0) and the
  principal point is the exact panel centre (w/2, h/2).
* Focal: the sim lens magnifies the panel centre by 1 + k (p_out/p_src -> 1 + k as
  r -> 0) while OpenCV's radial factor is 1 at the centre, so K's focal length is
  the paraxial fxPx (1 + k) — what any OpenCV-model solve (ours) recovers.

Subcommands
-----------
``gt``     write data/eval/ground_truth/<match_key>/<team>.json, one per robot, in the
           format of web/field/ground_truth.py (validated before writing).
``calib``  convert the sim cameras to our calibration (pinhole K/D/rvec/tvec per crop),
           validate (bumper boxes vs image-truth, tag corners vs the AprilTag detector
           on a real video frame, overlay JPEGs) and with ``--save`` write versions
           through web/field/store.py.
``grade``  grade the latest saved calibration version per camera for an event
           against the sim truth (camera position / orientation / focal / principal
           point error, bumper-box reprojection px, footprint field error m).

Usage
-----
    .venv/bin/python scripts/manta_sim_truth.py calib --event 2026mantatru --save
    .venv/bin/python scripts/manta_sim_truth.py grade --event 2026mantasim
    .venv/bin/python scripts/manta_sim_truth.py gt --match-key 2026mantasim_qm8 --event 2026mantasim

GT keyframe states
------------------
``visible`` when the robot's visible share (image-truth) is >= ``--visible-min`` (0.5)
in at least one camera, else ``estimated`` (truth still knows where it is). The
scorer's headline is the ``visible`` aggregate (what a vision pipeline can be held
to); its ``all`` aggregate scores every truth position (incl. occluded) and
``estimated`` scores occluded-only. ``uv`` is the projection of (x, y,
click_height_m) into the main view (a corner view when only a corner sees the
point), so ``reproject_doc`` with the exact calibration gives back ``field_xy``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from web.field import ground_truth as gtmod  # noqa: E402
from web.field import store  # noqa: E402
from web.field.calibration import CameraCalibration  # noqa: E402

DEFAULT_SIM_DIR = REPO / "data" / "sim_qm8"
FOOTAGE_DIR = REPO / "data" / "test_footage"
TRUTH_NAME = "truth-1080p60.json"
IMAGE_TRUTH_NAME = "image-truth-1080p60.json"
# Where validation overlay JPEGs go (override with MANTA_SCRATCH).
SCRATCH = Path(os.environ.get("MANTA_SCRATCH", str(REPO / "data" / "eval" / "sim_overlays")))

# our camera name -> sim panel name
CAMERAS = {"main": "main", "corner_left": "blueCorner", "corner_right": "redCorner"}
PX_OFFSET = 0.0                     # ours = sim continuous px - PX_OFFSET (see module doc)
GL_TO_CV = np.diag([1.0, -1.0, -1.0])
INCH = 0.0254


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_truth(sim_dir: str) -> dict:
    d = json.loads((Path(sim_dir) / TRUTH_NAME).read_text())
    if d.get("schema") != "manta-broadcast-v1":
        raise SystemExit(f"unexpected truth schema {d.get('schema')!r}")
    return d


@lru_cache(maxsize=4)
def load_image_truth(sim_dir: str) -> dict:
    d = json.loads((Path(sim_dir) / IMAGE_TRUTH_NAME).read_text())
    if d.get("schema") != "manta-image-truth-v1":
        raise SystemExit(f"unexpected image-truth schema {d.get('schema')!r}")
    return d


# --------------------------------------------------------------------------- #
# Exact sim camera
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SimCamera:
    """One sim panel's exact camera. All pixels here are SIM (continuous) full-frame
    pixels unless a name says otherwise."""
    name: str                      # our camera name
    sim_name: str
    panel: tuple                   # (x, y, w, h) in the full frame
    fx: float
    fy: float
    k: float                       # effective lens k = camera.lensK * quality.lens
    M: np.ndarray                  # 4x4 field metres -> OpenGL camera

    @property
    def R_gl(self) -> np.ndarray:
        return self.M[:3, :3]

    @property
    def t_gl(self) -> np.ndarray:
        return self.M[:3, 3]

    @property
    def R_cv(self) -> np.ndarray:
        return GL_TO_CV @ self.R_gl

    @property
    def t_cv(self) -> np.ndarray:
        return GL_TO_CV @ self.t_gl

    @property
    def centre(self) -> np.ndarray:
        return -self.R_gl.T @ self.t_gl

    @property
    def aspect(self) -> float:
        return self.panel[2] / self.panel[3]

    # -- lens (p in half-panel-height units, x right, y up) ------------------ #
    def lens_src_from_out(self, p_out: np.ndarray) -> np.ndarray:
        p = np.asarray(p_out, float)
        r2 = (p ** 2).sum(-1, keepdims=True) / (self.aspect ** 2 + 1)
        return p * (1 + self.k * r2) / (1 + self.k)

    def lens_out_from_src(self, p_src: np.ndarray) -> np.ndarray:
        """Exact inverse of lens_src_from_out (radial Newton solve)."""
        p = np.asarray(p_src, float)
        s_src = np.linalg.norm(p, axis=-1, keepdims=True)
        A, k = self.aspect ** 2 + 1, self.k
        s = s_src.copy()
        for _ in range(30):
            g = s * (1 + k * s * s / A) / (1 + k) - s_src
            dg = (1 + 3 * k * s * s / A) / (1 + k)
            s = s - g / dg
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(s_src > 0, s / np.where(s_src > 0, s_src, 1), 1.0)
        return p * scale

    # -- projection ---------------------------------------------------------- #
    def project(self, P) -> tuple[np.ndarray, np.ndarray]:
        """Field points (N, 3) -> (full-frame sim px (N, 2), in-front (N,) bool)."""
        P = np.asarray(P, float).reshape(-1, 3)
        C = P @ self.R_gl.T + self.t_gl
        depth = -C[:, 2]
        front = depth > 1e-6
        d = np.where(front, depth, 1.0)
        x, y, w, h = self.panel
        p_src = np.stack([self.fx * C[:, 0] / d, self.fy * C[:, 1] / d], 1) / (h / 2)
        p_out = self.lens_out_from_src(p_src)
        u = x + w / 2 + p_out[:, 0] * (h / 2)
        v = y + h / 2 - p_out[:, 1] * (h / 2)
        return np.stack([u, v], 1), front


def sim_cameras(truth: dict) -> dict[str, SimCamera]:
    lens_q = float((truth.get("quality") or {}).get("lens", 1.0))
    panels = {p["camera"]: p for p in truth["projection"]["panels"]}
    out = {}
    for ours, sim in CAMERAS.items():
        p = panels[sim]
        k = float(truth["cameras"][sim]["lensK"]) * lens_q
        eff = (truth.get("lens") or {}).get("effectiveK", {}).get(sim)
        if eff is not None and abs(float(eff) - k) > 1e-9:
            raise SystemExit(f"{sim}: lensK x quality.lens = {k} != lens.effectiveK {eff}")
        out[ours] = SimCamera(ours, sim, (p["x"], p["y"], p["w"], p["h"]),
                              float(p["fxPx"]), float(p["fyPx"]), k,
                              np.asarray(p["fieldToCamera"], float))
    return out


def _check_conventions(truth: dict, cams: dict[str, SimCamera]) -> list[str]:
    """Sanity checks on the conventions this tool assumes. Returns notes; raises
    on a hard mismatch."""
    notes = []
    conv = truth.get("conventions") or {}
    if "WPILib" not in conv.get("field", "") or "blue alliance wall right corner" not in conv.get("field", ""):
        raise SystemExit(f"unexpected field convention: {conv.get('field')!r}")
    if "OpenGL" not in truth["projection"].get("note", ""):
        raise SystemExit("projection note no longer says OpenGL camera — re-check the flip")
    if "out→src" not in (truth.get("lens") or {}).get("model", ""):
        raise SystemExit(f"unexpected lens model text: {truth['lens'].get('model')!r}")
    from web.field.season import season_for_event
    s = season_for_event(truth.get("event") or "2026")
    fl = truth["tags"].get("mantaFieldM") or truth["tags"].get("layoutField")
    if s.field_m and (abs(fl["length"] - s.field_m[0]) > 0.01 or abs(fl["width"] - s.field_m[1]) > 0.01):
        raise SystemExit(f"field size {fl} != ours {s.field_m}")
    # our tag layout's corners vs the sim's (as sets per tag)
    from web.field.field_model import FieldModel
    fm = FieldModel.load(s.tag_layout)
    worst = 0.0
    for r in truth["tags"]["records"]:
        ours = fm.tag_corners_3d(int(r["tagId"]))
        theirs = np.asarray(r["blackCornersM"], float)
        d = np.linalg.norm(ours[:, None] - theirs[None], axis=2).min(1).max()
        worst = max(worst, float(d))
    notes.append(f"tag corners: sim vs our {s.tag_layout} max {worst * 1000:.2f} mm")
    if worst > 0.01:
        raise SystemExit(f"tag layout differs from ours by {worst:.3f} m")
    # camera centre from the matrix vs cameras.pos (inches, same field frame)
    for c in cams.values():
        pos = np.asarray(truth["cameras"][c.sim_name]["pos"], float) * INCH
        notes.append(f"{c.name}: centre {np.round(c.centre, 3).tolist()} m "
                     f"(cameras.pos {np.round(pos, 3).tolist()} m)")
    return notes


# --------------------------------------------------------------------------- #
# Sim camera -> our CameraCalibration
# --------------------------------------------------------------------------- #
def fit_distortion(cam: SimCamera, step_px: float = 4.0, margin: float = 0.05) -> tuple:
    """Fit OpenCV D = [k1, k2, 0, 0, k3] so that cv2 distortion of the ideal pinhole
    point reproduces the sim output pixel, over a dense grid of OUTPUT pixels
    covering the panel grown by ``margin``. Returns (D (5,), stats dict)."""
    x, y, w, h = cam.panel
    us = np.arange(-margin * w, w * (1 + margin) + 1e-9, step_px)
    vs = np.arange(-margin * h, h * (1 + margin) + 1e-9, step_px)
    U, V = np.meshgrid(us, vs)
    U, V = U.ravel(), V.ravel()                              # crop-local sim px
    p_out = np.stack([(U - w / 2) / (h / 2), -(V - h / 2) / (h / 2)], 1)
    p_src = cam.lens_src_from_out(p_out)
    # OpenCV normalised coords: ideal (n, sim focal) and distorted (d, effective
    # focal of the K we write — OpenCV's radial factor is 1 at the centre, the
    # sim's out/src ratio there is 1 + k); y down
    K = intrinsics(cam)
    fxe, fye = K[0, 0], K[1, 1]
    xn, yn = p_src[:, 0] * (h / 2) / cam.fx, -p_src[:, 1] * (h / 2) / cam.fy
    xd, yd = p_out[:, 0] * (h / 2) / fxe, -p_out[:, 1] * (h / 2) / fye
    r2 = xn ** 2 + yn ** 2
    # pixel residual is linear in (k1, k2, k3): fx (x_n (1 + k1 r2 + ...) - x_d)
    Ax = fxe * xn[:, None] * np.stack([r2, r2 ** 2, r2 ** 3], 1)
    Ay = fye * yn[:, None] * np.stack([r2, r2 ** 2, r2 ** 3], 1)
    A = np.vstack([Ax, Ay])
    b = np.concatenate([fxe * (xd - xn), fye * (yd - yn)])
    kk, *_ = np.linalg.lstsq(A, b, rcond=None)
    D = np.array([kk[0], kk[1], 0.0, 0.0, kk[2]])
    # residual through cv2 itself (identity pose, points on the z=1 plane)
    obj = np.stack([xn, yn, np.ones_like(xn)], 1).reshape(-1, 1, 3)
    img, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, D.reshape(-1, 1))
    img = img.reshape(-1, 2) + PX_OFFSET                     # back to sim px
    err = np.hypot(img[:, 0] - U, img[:, 1] - V)
    inside = (U >= 0) & (U <= w) & (V >= 0) & (V <= h)
    from web.field.calibration import distortion_fold_radius
    fold = distortion_fold_radius(D, "pinhole")
    r_corner = float(np.sqrt(r2[inside].max()))
    stats = {"rms_px": float(np.sqrt((err[inside] ** 2).mean())),
             "max_px": float(err[inside].max()),
             "max_px_margin": float(err.max()),
             "fold_radius": float(fold), "corner_radius": r_corner,
             "n": int(inside.sum())}
    return D, stats


def intrinsics(cam: SimCamera) -> np.ndarray:
    """OpenCV K: the sim lens magnifies the image centre by d p_out / d p_src =
    1 + k, so the effective (paraxial) focal length is fxPx (1 + k) — what any
    OpenCV-model solve recovers, and what the radial fit is anchored to."""
    _, _, w, h = cam.panel
    return np.array([[cam.fx * (1 + cam.k), 0.0, w / 2 - PX_OFFSET],
                     [0.0, cam.fy * (1 + cam.k), h / 2 - PX_OFFSET],
                     [0.0, 0.0, 1.0]])


def to_calibration(cam: SimCamera, field_season: str) -> tuple[CameraCalibration, dict]:
    D, fit = fit_distortion(cam)
    rvec, _ = cv2.Rodrigues(cam.R_cv)
    x, y, w, h = cam.panel
    cal = CameraCalibration(
        K=intrinsics(cam), D=D, rvec=rvec.reshape(3, 1), tvec=cam.t_cv.reshape(3, 1),
        image_size=(int(w), int(h)), crop_region=(int(x), int(y), int(w), int(h)),
        field_season=field_season, rms_error=round(fit["rms_px"], 4), mode="pinhole",
        H=None, masks=[], robot_track_top_crop=0.0)
    return cal, fit


def cal_project_full(cal: CameraCalibration, P) -> tuple[np.ndarray, np.ndarray]:
    """Field points -> (full-frame SIM px, ok) through one of OUR calibrations
    (crop-local px + crop offset + PX_OFFSET). ``ok`` is the pipeline's own
    visibility test without the frame test (in front of the camera and inside the
    lens's monotonic radius — past it a polynomial lens folds far-off points back
    into the image)."""
    P = np.asarray(P, float).reshape(-1, 3)
    uv, ok = cal.project_checked(P, margin_px=None)
    cx, cy = (cal.crop_region or (0, 0, 0, 0))[:2]
    return uv.reshape(-1, 2) + np.array([cx, cy]) + PX_OFFSET, ok


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def box_errors(project, cam: SimCamera, truth: dict, itruth: dict, step: int = 30,
               min_box_px: float = 4.0) -> np.ndarray:
    """Per-box error (px, max |edge difference|) of bbox(project(8 bumper corners))
    vs the image-truth box, over robots fully inside ``cam``'s panel.
    ``project(P) -> (sim px, front)``."""
    x, y, w, h = cam.panel
    errs = []
    for fi in range(0, len(truth["frames"]), step):
        robots = truth["frames"][fi]["truth"]["robots"]
        rows = itruth["frames"][fi]["cams"][cam.sim_name]
        for r, it in zip(robots, rows):
            if it.get("box") is None or it.get("inPanel", 0) < 1:
                continue
            px, front = project(np.asarray(r["bumperCornersM"], float))
            if not front.all():
                continue
            b = np.array([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()])
            b = np.clip(b, [x, y, x, y], [x + w, y + h, x + w, y + h])
            tb = np.asarray(it["box"], float)
            if min(tb[2] - tb[0], tb[3] - tb[1]) < min_box_px:
                continue
            errs.append(float(np.abs(b - tb).max()))
    return np.asarray(errs)


def footprint_errors(cal: CameraCalibration, cam: SimCamera, truth: dict, step: int = 30,
                     z: float = gtmod.DEFAULT_CLICK_HEIGHT_M) -> np.ndarray:
    """Field error (m): truth robot centre at height z -> exact sim pixel -> back
    through ``cal`` onto the z plane, for centres inside both panels."""
    errs = []
    cx, cy, cw, ch = cal.crop_region or (0, 0, *cal.image_size)
    x, y, w, h = cam.panel
    for fi in range(0, len(truth["frames"]), step):
        for r in truth["frames"][fi]["truth"]["robots"]:
            P = np.array([[r["xM"], r["yM"], z]])
            px, front = cam.project(P)
            u, v = px[0]
            if not front[0] or not (x <= u < x + w and y <= v < y + h):
                continue
            if not (cx <= u < cx + cw and cy <= v < cy + ch):
                continue
            xy = cal.image_to_field((u - cx - PX_OFFSET, v - cy - PX_OFFSET), z=z)
            if xy is None:
                continue
            errs.append(math.hypot(xy[0] - r["xM"], xy[1] - r["yM"]))
    return np.asarray(errs)


def read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_idx} of {video}")
    return img


def tag_check(cal: CameraCalibration, cam: SimCamera, truth: dict, frames: list,
              out_jpg: Optional[Path], roi: bool = True) -> dict:
    """Project every camera-facing tag's black corners through ``cal`` and compare
    with our AprilTag detector's corners on the panel of each ``frames`` entry
    ((frame index, full-frame BGR)). Tags the frame-wide detector misses are
    retried with the calibrate UI's upscaled ROI decode around the projected
    centre (``roi``). The first frame's overlay — projected outline + id (green),
    detected corners (red) — is written to ``out_jpg``."""
    from web.field.apriltag_detect import detect_tags, detect_tags_in_roi
    x, y, w, h = cam.panel
    errs, dxy, per_tag, n_roi = [], [], {}, 0
    vis = None
    for k, (fidx, frame) in enumerate(frames):
        crop = frame[y:y + h, x:x + w].copy()
        det = detect_tags(crop)                              # crop-local px
        if k == 0:
            vis = frame.copy()
        for r in truth["tags"]["records"]:
            tid = int(r["tagId"])
            C = np.asarray(r["blackCornersM"], float)
            normal = np.asarray(r["normalField"], float)
            if np.dot(cam.centre - np.asarray(r["frontCenterM"], float), normal) <= 0:
                continue                                     # tag faces away
            px, ok = cal_project_full(cal, C)
            if not ok.all() or not ((px[:, 0] >= x) & (px[:, 0] < x + w)
                                    & (px[:, 1] >= y) & (px[:, 1] < y + h)).all():
                continue
            local = px - np.array([x, y]) - PX_OFFSET        # crop-local px
            d = det.get(tid)
            if d is None and roi:
                side = float(np.ptp(local, axis=0).max())
                d = detect_tags_in_roi(crop, tuple(local.mean(0)), tid,
                                       half=int(max(24, 2 * side)))
                n_roi += d is not None
            if k == 0:
                pts = (px - PX_OFFSET).round().astype(int)
                cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, str(tid), tuple(int(v) for v in pts[0] + [2, -4]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            if d is None:
                continue
            d = np.asarray(d, float)
            dist = np.linalg.norm(d[:, None] - local[None], axis=2)
            j = dist.argmin(1)
            e = dist[np.arange(4), j]
            if e.max() > 10:                                 # mis-decode / wrong tag: report, skip
                print(f"    tag {tid} @ frame {fidx}: corner error {e.max():.1f} px — skipped")
                continue
            errs += e.tolist()
            dxy += (d - local[j]).tolist()
            per_tag.setdefault(tid, []).append(round(float(e.mean()), 2))
            if k == 0:
                for q in d + np.array([x, y]) + PX_OFFSET:
                    cv2.circle(vis, tuple(int(round(v)) for v in q), 2, (0, 0, 255), -1)
    if out_jpg is not None and vis is not None:
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 1)
        cv2.imwrite(str(out_jpg), vis[max(0, y - 10):y + h + 10, max(0, x - 10):x + w + 10],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    errs = np.asarray(errs)
    md = np.mean(dxy, axis=0) if dxy else [float("nan")] * 2
    return {"n_tags": len(per_tag), "n_corners": len(errs), "n_roi_decodes": n_roi,
            "median_px": _pct(errs, 50), "p95_px": _pct(errs, 95),
            "max_px": float(errs.max()) if len(errs) else float("nan"),
            "mean_dx": float(md[0]), "mean_dy": float(md[1]),
            "per_tag": {t: round(float(np.mean(v)), 2) for t, v in sorted(per_tag.items())}}


def _fmt(v, nd=2):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# calib
# --------------------------------------------------------------------------- #
def cmd_calib(args) -> int:
    from web.field.season import season_for_event
    sim_dir = str(args.sim_dir)
    truth, itruth = load_truth(sim_dir), load_image_truth(sim_dir)
    cams = sim_cameras(truth)
    for n in _check_conventions(truth, cams):
        print("  " + n)
    season = season_for_event(args.event).tag_layout
    video = Path(args.video) if args.video else _video_for_event(args.event)
    frames = [(f, read_frame(video, f)) for f in args.frame] if video else []
    print(f"\nvideo {video} frames {args.frame}" if video else "\nno video: tag check skipped")

    ok_all, results = True, {}
    for name, cam in cams.items():
        cal, fit = to_calibration(cam, season)
        exact = box_errors(cam.project, cam, truth, itruth, args.step)
        conv = box_errors(lambda P: cal_project_full(cal, P), cam, truth, itruth, args.step)
        # converted cal vs exact projector, every bumper corner inside the panel
        dev = []
        for fi in range(0, len(truth["frames"]), args.step):
            for r in truth["frames"][fi]["truth"]["robots"]:
                P = np.asarray(r["bumperCornersM"], float)
                a, fa = cam.project(P)
                b, _ = cal_project_full(cal, P)
                x, y, w, h = cam.panel
                m = fa & (a[:, 0] >= x) & (a[:, 0] < x + w) & (a[:, 1] >= y) & (a[:, 1] < y + h)
                dev += np.linalg.norm(a[m] - b[m], axis=1).tolist()
        dev = np.asarray(dev)
        tags = tag_check(cal, cam, truth, frames, SCRATCH / f"truthcal_{name}.jpg") \
            if frames else None
        ok = (fit["max_px"] < 0.5 and _pct(conv, 95) < 2.0
              and (tags is None or not tags["n_corners"] or tags["median_px"] < 2.0))
        ok_all &= ok
        results[name] = (cal, fit, conv, tags)
        print(f"\n[{name}] (sim {cam.sim_name}) panel {cam.panel} fx {cam.fx:.2f} k {cam.k}"
              f"  -> {'OK' if ok else 'FAIL'}")
        print(f"  D = {np.round(cal.D, 6).tolist()}  rvec {np.round(cal.rvec.ravel(), 5).tolist()}"
              f"  tvec {np.round(cal.tvec.ravel(), 4).tolist()}")
        print(f"  lens fit: rms {fit['rms_px']:.3f} px, max {fit['max_px']:.3f} px in panel "
              f"({fit['max_px_margin']:.3f} incl. 5% margin); fold radius "
              f"{fit['fold_radius']:.3f} vs corner {fit['corner_radius']:.3f}")
        print(f"  converted vs exact projector (bumper corners in panel): median "
              f"{_fmt(_pct(dev, 50), 3)} p95 {_fmt(_pct(dev, 95), 3)} max "
              f"{_fmt(dev.max() if len(dev) else float('nan'), 3)} px (n={len(dev)})")
        print(f"  bumper box vs image-truth: exact projector median {_fmt(_pct(exact, 50))} "
              f"p95 {_fmt(_pct(exact, 95))}; converted median {_fmt(_pct(conv, 50))} "
              f"p95 {_fmt(_pct(conv, 95))} px (n={len(conv)})")
        if tags is not None:
            print(f"  tags vs detector ({len(frames)} frames, {tags['n_roi_decodes']} ROI decodes): "
                  f"{tags['n_tags']} tags / {tags['n_corners']} corners, "
                  f"median {_fmt(tags['median_px'])} p95 {_fmt(tags['p95_px'])} max "
                  f"{_fmt(tags['max_px'])} px, mean (dx, dy) = ({_fmt(tags['mean_dx'])}, "
                  f"{_fmt(tags['mean_dy'])}); per tag {tags['per_tag']}")
            print(f"  overlay {SCRATCH / f'truthcal_{name}.jpg'}")

    if not args.save:
        print("\n(dry run: --save writes calibration versions)")
        return 0 if ok_all else 1
    if not ok_all and not args.force:
        print("\nvalidation FAILED — not saving (use --force to override)")
        return 1
    print()
    for name, (cal, fit, conv, tags) in results.items():
        cam = cams[name]
        source = {
            "method": "manta_sim_truth",
            "sim_camera": cam.sim_name,
            "truth_file": str(Path(sim_dir) / TRUTH_NAME),
            "crop": list(cam.panel),
            "match_key": args.match_key,
            # reference-frame time for the corner per-match same-view check, which
            # medians [timestamp - window, timestamp] — so not 0
            "timestamp": round(max(args.frame) / float(truth["video"]["fps"]), 3),
            "lens_fit": {k: round(v, 5) if isinstance(v, float) else v for k, v in fit.items()},
            "validation": {
                "box_median_px": round(_pct(conv, 50), 3), "box_p95_px": round(_pct(conv, 95), 3),
                "tag_median_px": round(tags["median_px"], 3) if tags and tags["n_corners"] else None,
                "tag_corners": tags["n_corners"] if tags else 0},
        }
        vid = store.save_version(args.event, name, cal, start_match=None,
                                 label="Manta sim truth (exact camera)", source=source)
        print(f"saved {args.event}/{name} version {vid} -> "
              f"{store._version_path(args.event, name, vid)}")
    return 0


def _video_for_event(event: str) -> Optional[Path]:
    hits = sorted(FOOTAGE_DIR.glob(f"{event}_*.mp4"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# grade
# --------------------------------------------------------------------------- #
def _latest_version(event: str, camera: str) -> Optional[dict]:
    vs = store.load_versions(event, camera)
    return max(vs, key=lambda v: v.get("created_at") or "") if vs else None


def cmd_grade(args) -> int:
    sim_dir = str(args.sim_dir)
    truth, itruth = load_truth(sim_dir), load_image_truth(sim_dir)
    cams = sim_cameras(truth)
    out = {}
    print(f"grading latest calibration versions of {args.event} against {sim_dir}")
    for name, cam in cams.items():
        v = _latest_version(args.event, name)
        if v is None:
            print(f"\n[{name}] no calibration version")
            continue
        cal: CameraCalibration = v["calibration"]
        # Jobs crop each view from the EVENT LAYOUT (camera_view.load_views uses
        # the layout crop, never the version's crop_region; manual /calibrate
        # solves store crop_region with a 0,0 origin), so grade with that crop.
        from web.field.composite_layout import load_layout
        lay = load_layout(event_key=args.event).get(name)
        if lay is not None:
            cal.crop_region = tuple(int(c) for c in lay.crop)
        row = {"version_id": v["version_id"], "label": v.get("label"), "mode": cal.mode,
               "crop": list(cal.crop_region) if cal.crop_region else None}
        if cal.mode == "homography" or cal.rvec is None:
            print(f"\n[{name}] version {v['version_id']}: {cal.mode} — no pose to grade")
            out[name] = row
            continue
        R, _ = cv2.Rodrigues(np.asarray(cal.rvec, float))
        centre = cal.camera_center
        dR = R @ cam.R_cv.T
        ang = math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        fx, fy = float(cal.K[0, 0]), float(cal.K[1, 1])
        cx0, cy0 = (cal.crop_region or (0, 0))[:2]
        pp = np.array([cal.K[0, 2] + cx0, cal.K[1, 2] + cy0]) + PX_OFFSET
        pp_true = np.array([cam.panel[0] + cam.panel[2] / 2, cam.panel[1] + cam.panel[3] / 2])
        boxes = box_errors(lambda P: cal_project_full(cal, P), cam, truth, itruth, args.step)
        foot = footprint_errors(cal, cam, truth, args.step)
        row.update({
            "position_err_m": float(np.linalg.norm(centre - cam.centre)),
            "position_delta_m": np.round(centre - cam.centre, 4).tolist(),
            "orientation_err_deg": ang,
            # vs the paraxial focal fxPx (1 + k) (see intrinsics())
            "focal_err_pct": 100 * ((fx + fy) / 2 / ((cam.fx + cam.fy) / 2 * (1 + cam.k)) - 1),
            "principal_point_err_px": float(np.linalg.norm(pp - pp_true)),
            "D": np.round(np.asarray(cal.D).ravel(), 5).tolist(),
            "rms_error_px": cal.rms_error,
            "box_median_px": _pct(boxes, 50), "box_p95_px": _pct(boxes, 95), "n_boxes": len(boxes),
            "footprint_median_m": _pct(foot, 50), "footprint_p95_m": _pct(foot, 95),
            "n_footprints": len(foot),
        })
        out[name] = row
        print(f"\n[{name}] version {v['version_id']} ({v.get('label')}) mode {cal.mode} "
              f"crop {row['crop']}")
        print(f"  camera position err {row['position_err_m']:.3f} m "
              f"(delta {row['position_delta_m']}), orientation err {ang:.3f} deg")
        print(f"  focal err {row['focal_err_pct']:+.2f}% (fx {fx:.1f}/{fy:.1f} vs paraxial "
              f"{cam.fx * (1 + cam.k):.1f} = fxPx {cam.fx:.1f} x (1 + k {cam.k})), "
              f"principal point err {row['principal_point_err_px']:.1f} px, D {row['D']}")
        print(f"  bumper boxes vs image-truth: median {_fmt(row['box_median_px'])} p95 "
              f"{_fmt(row['box_p95_px'])} px (n={len(boxes)})")
        print(f"  robot footprints back-projected (z={gtmod.DEFAULT_CLICK_HEIGHT_M} m): median "
              f"{_fmt(row['footprint_median_m'], 3)} p95 {_fmt(row['footprint_p95_m'], 3)} m "
              f"(n={len(foot)})")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {args.json}")
    return 0


# --------------------------------------------------------------------------- #
# gt
# --------------------------------------------------------------------------- #
def ensure_footage(match_key: str, sim_dir: str) -> Path:
    """data/test_footage/<match_key>.mp4; for a key with no file, hard-link it to
    another footage file of the same sim match (same event-agnostic match suffix)
    that is already there."""
    p = FOOTAGE_DIR / f"{match_key}.mp4"
    if p.exists():
        return p
    suffix = match_key.split("_", 1)[1]
    for cand in sorted(FOOTAGE_DIR.glob(f"2026manta*_{suffix}.mp4")):
        os.link(cand, p)
        print(f"hard-linked {p} -> {cand}")
        return p
    raise SystemExit(f"no footage {p} (and no 2026manta*_{suffix}.mp4 to link)")


def build_gt_docs(match_key: str, event: str, sim_dir: str, video: Path, step: int,
                  visible_min: float, cal_versions: Optional[dict] = None) -> list[dict]:
    truth, itruth = load_truth(sim_dir), load_image_truth(sim_dir)
    cams = sim_cameras(truth)
    if [int(t) for t in itruth["teams"]] != [int(r["team"]) for r in truth["frames"][0]["truth"]["robots"]]:
        raise SystemExit("image-truth robot order != truth robot order")
    cap = cv2.VideoCapture(str(video))
    vmeta = {"name": video.name, "fingerprint": gtmod.video_fingerprint(video),
             "fps": float(cap.get(cv2.CAP_PROP_FPS)),
             "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    if vmeta["frame_count"] != len(truth["frames"]) or abs(vmeta["fps"] - truth["video"]["fps"]) > 1e-6:
        raise SystemExit(f"video {vmeta} does not match truth ({len(truth['frames'])} frames "
                         f"@ {truth['video']['fps']})")
    if abs(truth["frames"][0]["truth"]["t"]) > 1.0 / vmeta["fps"]:
        raise SystemExit("truth frame 0 is not AUTO start (match_t0_frame 0 would be wrong)")
    z = gtmod.DEFAULT_CLICK_HEIGHT_M
    station = {}
    for al in ("red", "blue"):
        for i, t in enumerate(truth["match"][al]):
            station[int(t)] = (al, i + 1)
    order = [int(t) for t in itruth["teams"]]
    docs = {}
    for t in order:
        al, st = station[t]
        doc = gtmod.new_gt(match_key, event, str(t), al, st, vmeta, match_t0_frame=0,
                           annotator="manta-sim-truth")
        doc["calibration"] = dict(cal_versions or {})
        doc["click_height_m"] = z
        doc["notes"] = (f"Manta Sim exact truth ({Path(sim_dir).name}/{TRUTH_NAME}); field_xy = "
                        f"robot centre; state visible = visible share >= {visible_min} in >= 1 "
                        f"camera (image-truth), else estimated; uv = exact projection of "
                        f"(x, y, {z}) into the main view (corner view if only a corner sees "
                        f"it); every {step} frames.")
        doc["source"] = {"kind": "manta_sim_truth", "sim_dir": str(sim_dir), "step": step,
                         "visible_min": visible_min, "seed": truth.get("seed")}
        docs[t] = doc
    view_order = ["main", "corner_left", "corner_right"]
    for fi in range(0, len(truth["frames"]), step):
        fr = truth["frames"][fi]
        if int(fr["frame"]) != fi or int(itruth["frames"][fi]["frame"]) != fi:
            raise SystemExit(f"frame index mismatch at {fi}")
        robots = fr["truth"]["robots"]
        cams_it = itruth["frames"][fi]["cams"]
        P = np.array([[r["xM"], r["yM"], z] for r in robots])
        proj = {n: cams[n].project(P) for n in view_order}
        for ri, r in enumerate(robots):
            t = int(r["team"])
            vis = max(float(cams_it[cams[n].sim_name][ri].get("visible") or 0) for n in view_order)
            uv, view = None, None
            for n in view_order:
                (px, front), (x, y, w, h) = proj[n], cams[n].panel
                u, v = px[ri]
                if front[ri] and x <= u < x + w and y <= v < y + h:
                    uv, view = [round(float(u), 2), round(float(v), 2)], n
                    break
            if uv is None:                                  # off every view: main anyway
                uv = [round(float(c), 2) for c in proj["main"][0][ri]]
            kf = {"frame": fi, "state": "visible" if vis >= visible_min else "estimated",
                  "uv": uv, "src": "sim", "view": view,
                  "field_xy": [round(float(r["xM"]), 4), round(float(r["yM"]), 4)],
                  "visible_share": round(vis, 3)}
            docs[t]["keyframes"].append(kf)
    return [docs[t] for t in order]


def cmd_gt(args) -> int:
    sim_dir = str(args.sim_dir)
    video = ensure_footage(args.match_key, sim_dir)
    cal_versions = {}
    for cam in CAMERAS:
        v = _latest_version(args.event, cam)
        if v is not None and (v.get("source") or {}).get("method") == "manta_sim_truth":
            cal_versions[cam] = v["version_id"]
    docs = build_gt_docs(args.match_key, args.event, sim_dir, video, args.step,
                         args.visible_min, cal_versions)
    gt_dir = Path(args.gt_dir)
    for doc in docs:
        gtmod.validate_gt(gtmod.normalize_gt(doc))
        p = gtmod.save_gt(doc, gt_dir)
        back = json.loads(p.read_text())
        gtmod.validate_gt(back)
        n = len(back["keyframes"])
        nv = sum(1 for k in back["keyframes"] if k["state"] == "visible")
        print(f"{p}: {back['alliance']}{back['station']} {n} keyframes, visible {nv} "
              f"({100 * nv / max(n, 1):.0f}%), estimated {n - nv}; video "
              f"{back['video']['name']} {back['video']['fingerprint']}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", default=str(DEFAULT_SIM_DIR),
                    help="directory with the sim truth JSONs (default data/sim_qm8)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gt", help="write ground-truth docs for a match")
    g.add_argument("--match-key", required=True)
    g.add_argument("--event", required=True)
    g.add_argument("--step", type=int, default=3,
                   help="keyframe every N video frames (default 3 = 20 Hz; the scorer "
                        "samples a 0.25 s grid = every 15 frames)")
    g.add_argument("--visible-min", type=float, default=0.5,
                   help="visible share (any camera) for state 'visible' (default 0.5)")
    g.add_argument("--gt-dir", default=str(gtmod.GT_DIR))

    c = sub.add_parser("calib", help="convert + validate the sim cameras; --save writes versions")
    c.add_argument("--event", required=True)
    c.add_argument("--save", action="store_true")
    c.add_argument("--force", action="store_true", help="save even if validation fails")
    c.add_argument("--video", help="video for the tag check (default data/test_footage/<event>_*.mp4)")
    c.add_argument("--frame", type=int, nargs="+", default=[0, 1800, 4800, 7800],
                   help="video frames for the tag check (overlay from the first)")
    c.add_argument("--match-key", default=None,
                   help="recorded as the version's source match (default <event>_qm<N> of the truth)")
    c.add_argument("--step", type=int, default=30, help="truth frame stride for validation")

    r = sub.add_parser("grade", help="grade an event's latest calibration versions")
    r.add_argument("--event", required=True)
    r.add_argument("--step", type=int, default=30, help="truth frame stride")
    r.add_argument("--json", help="write the grade JSON here")

    args = ap.parse_args(argv)
    if args.cmd == "calib" and args.match_key is None:
        q = (load_truth(str(args.sim_dir)).get("match") or {}).get("qual")
        args.match_key = f"{args.event}_qm{q}" if q else None
    return {"gt": cmd_gt, "calib": cmd_calib, "grade": cmd_grade}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

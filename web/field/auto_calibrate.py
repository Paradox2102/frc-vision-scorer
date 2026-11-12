"""Automatic per-camera calibration when an event loads (review-first).

Spec: docs/superpowers/specs/2026-11-10-auto-calibration-review-design.md

Given one reference match's footage, each camera gets ONE of:
  * full_solve  — the event has no calibration for the camera: aggregate AprilTag
                  corners across the match and solve a full pinhole camera.
  * verify      — the event has a covering epoch: check it against the footage. If it
                  still aligns the record is ``verified`` (nothing to confirm).
  * pose_resolve— the covering epoch no longer aligns: re-solve only the pose
                  against its lens (``pose_autofix``) and stage it for review.

Nothing here persists anything or writes a calibration version — the result is a
record dict + candidate calibration + overlay image, which the app stores as a
review artifact. The operator confirms (promotes to a version) or corrects manually.

All heavy functions take explicit inputs so they are testable without the app; the
yt-dlp download is a separate injectable function so tests never touch the network.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from web.field.calibration import (CameraCalibration, Calibrator, CalibrationError,
                                   check_solve_plausible, tag_alignment_error)
from web.field.match_order import match_sort_key, select_epoch
from web.field.pose_autofix import (PoseFixConfig, aggregate_tag_corners, evaluate_fix,
                                    resolve_pose)
from web.field.season import DEFAULT_LAYOUT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOOTAGE_DIRS = (PROJECT_ROOT / "data" / "test_footage", PROJECT_ROOT / "data" / "videos")
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "test_footage"

STATUSES = ("running", "ready", "verified", "confirmed", "failed")
# A failed record is retried on event load only once it is this old (a reload
# should not re-resolve / re-download footage every time); force always re-runs.
FAILED_RETRY_AFTER_S = 6 * 3600
# A run skips cameras already in one of these states unless forced.
SETTLED_STATUSES = frozenset({"ready", "verified", "confirmed"})

_MATCH_KEY_RE = re.compile(r"^[A-Za-z0-9]+_(qm|ef|qf|sf|f)\d+(m\d+)?$")

# BGR colours for the overlay
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)


class AutoCalibError(Exception):
    """A per-camera / per-event failure whose message is the record's reason."""


@dataclass
class AutoCalibConfig:
    window: tuple = (18.0, 1e9)       # skip the pre-auto framing
    stride_sec: float = 1.0
    mad_px: float = 5.0
    min_tags: int = 3                 # solve / verify / pose-resolve conditioning
    check_min_tags: int = 2           # per-frame validation floor (mirrors CALIB_CHECK_MIN_TAGS)
    validate_samples: int = 12
    drift_max_m: float = 0.5          # mirrors CALIB_DRIFT_MAX_M
    max_center_shift_m: float = 1.0   # pose_resolve sanity bound (PoseFixConfig)
    # Tag layout the solve is against. Callers pass the event's season
    # (web.field.season.season_for_event); the default is only for no-event use.
    field_season: str = DEFAULT_LAYOUT
    # Temporal median of the camera crop over the match window: the camera is static
    # in play, so robots/balls/people blocking a tag vanish. Tags decoded only there
    # join the aggregated set, and it is the overlay / locator image. 0 disables.
    median_frames: int = 31
    # Full-solve branch: grow the solve outward by locating undecoded tags
    # (web.field.tag_locator), quad candidates gated by a Jev judgement.
    locate_undecoded: bool = True
    jev_verify: bool = True
    jev_min_p: float = 0.6
    locate_init_model: str = "pp_k1"
    # A fresh full solve frees the lens; tags clustered on one structure (a hub
    # corner spans ~0.6-0.9 m) fit perfectly yet extrapolate wildly (2026casnd
    # corner_left). Below this field-space spread a full solve is never "ready".
    min_full_solve_spread_m: float = 2.0


@dataclass
class AutoCalibResult:
    record: dict
    calibration: Optional[CameraCalibration]
    overlay: Optional[np.ndarray]
    median: Optional[np.ndarray] = None       # raw temporal-median crop (no overlay)
    candidates: list = field(default_factory=list)   # locator candidates (eval/debug)


# --------------------------------------------------------------------------- footage

def valid_match_key(match_key) -> bool:
    return bool(match_key) and bool(_MATCH_KEY_RE.match(str(match_key)))


def local_video_for(match_key: str, dirs=None) -> Optional[Path]:
    """``<dir>/<match_key>.mp4`` from the first footage dir that has it, else None.
    The key is validated first so it can never escape the footage dirs."""
    if not valid_match_key(match_key):
        return None
    for d in (dirs or FOOTAGE_DIRS):
        p = Path(d) / f"{match_key}.mp4"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _local_keys_for_event(event_key: str, dirs) -> set:
    keys = set()
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in d.glob(f"{event_key}_*.mp4"):
            if valid_match_key(p.stem) and p.stem.startswith(f"{event_key}_"):
                keys.add(p.stem)
    return keys


def pick_reference_match(event_key: str, matches, dirs=None):
    """Choose the reference match for an event.

    Returns ``(match_key, local_path, download_match)``:
      * the earliest match (``match_sort_key`` order: quals first) with local footage
        -> (key, path, None). Local files named ``<event>_<level><n>.mp4`` count even
        if the match list is empty (event not loaded in this process);
      * else the earliest QUAL match with a TBA ``video_key`` -> (key, None, match);
      * else (None, None, None).
    """
    dirs = dirs or FOOTAGE_DIRS
    by_key = {m.get("key"): m for m in (matches or []) if m.get("key")}
    keys = set(by_key) | _local_keys_for_event(event_key, dirs)
    for mk in sorted(keys, key=match_sort_key):
        p = local_video_for(mk, dirs)
        if p is not None:
            return mk, p, None
    for mk in sorted(by_key, key=match_sort_key):
        m = by_key[mk]
        if match_sort_key(mk)[0] == 0 and m.get("video_key"):
            return mk, None, m
    return None, None, None


def download_match_video(video_key: str, out_path: Path) -> Path:
    """Download one YouTube match video as OpenCV-decodable H.264 mp4 (the
    scripts/download_match.py recipe: deno on PATH before importing yt_dlp, <=1080p,
    prefer h264, merge to mp4). Raises AutoCalibError carrying the yt-dlp error."""
    deno = os.path.expanduser("~/.deno/bin")
    if os.path.isdir(deno) and deno not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = deno + os.pathsep + os.environ.get("PATH", "")
    import yt_dlp  # after PATH so the JS runtime unlocks >360p formats

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": ("bestvideo[height<=1080]+bestaudio[ext=m4a]/"
                   "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"),
        "format_sort": ["res:1080", "vcodec:h264"],
        "merge_output_format": "mp4",
        "outtmpl": str(out_path),
        "quiet": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_key}"])
    except Exception as e:  # yt_dlp.utils.DownloadError and friends
        raise AutoCalibError(f"download of {video_key} failed: {str(e)[:300]}")
    if not out_path.is_file():
        raise AutoCalibError(f"download of {video_key} produced no file at {out_path}")
    return out_path


FREE_LENS_MIN_TAGS = 8   # a freer lens needs this many tags (not just spread) to be supported


def free_lens_ladder(cal, corr, image_size, season, fm=None):
    """Free the lens step by step (k1, then full) from a constrained fit ``cal`` — a
    freer model is kept only if it fits clearly better (< 0.8 x RMS) and its distortion
    does not fold back — over the field area in view when ``fm`` is given (else over the
    points). Needs >= FREE_LENS_MIN_TAGS tags: 4 hub tags + 2 clicked wall tags span
    4.7 m yet free "k1" fit them at 0.34 px and folded everywhere else (2026casnd
    corner_right). Callers must also check the tags' field spread (``tag_spread_m``).
    Returns (cal, the freest model adopted or None)."""
    from web.field import tag_locator as tl
    pts = np.vstack([np.asarray(i, float).reshape(-1, 2) for i, _ in corr.values()])
    n_tags = len({k for k in corr})
    freed = None
    if n_tags < FREE_LENS_MIN_TAGS:
        return cal, None
    for model in ("k1", "full"):
        if n_tags < tl.MODEL_MIN_TAGS[model]:
            break
        try:
            c = tl.solve_model(corr, image_size, season, model, prev=cal)
        except CalibrationError:
            continue
        mono = (tl.monotonic_ok(c, corr, fm, image_size) if fm is not None
                else tl.distortion_monotonic(c, image_size, pts))
        if c.rms_error < 0.8 * cal.rms_error and mono:
            cal, freed = c, model
    return cal, freed


def solve_fresh_lens(corr, image_size, season, fm, min_spread_m: float = 2.0,
                     init_model: str = "pp_k1"):
    """A camera solve with NO prior lens that cannot run away: the constrained model
    first (centred principal point, square pixels, one radial term), freed only when
    the tags span >= ``min_spread_m`` of the field. Tags on one structure fit a free
    lens perfectly yet extrapolate wildly (2026casnd corner_left). ``corr`` keys may be
    ints or "a6"/"m6"/"r6". Returns (cal, model, spread_m)."""
    from web.field import tag_locator as tl
    from web.field.calibration import _tag_id_of_key
    cal, err, model = None, None, None
    for m in (init_model, "pp_k0"):
        try:
            cal, model = tl.solve_model(corr, image_size, season, m), m
            break
        except CalibrationError as e:
            err = e
    if cal is None:
        raise err or CalibrationError("no constrained model could be solved")
    ids = {_tag_id_of_key(k) for k in corr} - {None}
    spread = tag_spread_m(fm, ids)
    if spread >= min_spread_m:
        cal, freed = free_lens_ladder(cal, corr, image_size, season, fm)
        model = freed or model
    return cal, model, spread


def tag_spread_m(fm, tag_ids) -> float:
    """Largest field-plane (x, y) distance between any two of ``tag_ids`` (metres);
    0 for fewer than two known tags. How much of the field the tags constrain."""
    pts = [np.asarray(fm.tag_translation(t), float)[:2] for t in tag_ids if t in fm.tags]
    if len(pts) < 2:
        return 0.0
    P = np.stack(pts)
    return float(np.max(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)))


def later_reference_videos(event_key: str, matches, after_key: str, dirs=None,
                           download_dir=None, downloader: Callable = None, limit: int = 3):
    """Yield (match_key, video_path) for matches AFTER ``after_key`` (match order) to
    calibrate a camera on when it moved during the reference match: local footage
    first, then (``downloader`` given) qual matches with a TBA video, downloading
    each only when reached. At most ``limit`` candidates; download failures are
    skipped."""
    dirs = dirs or FOOTAGE_DIRS
    after = match_sort_key(after_key)
    by_key = {m.get("key"): m for m in (matches or []) if m.get("key")}
    keys = sorted((set(by_key) | _local_keys_for_event(event_key, dirs)), key=match_sort_key)
    later = [k for k in keys if match_sort_key(k) > after]
    n = 0
    for mk in later:
        p = local_video_for(mk, dirs)
        if p is not None:
            n += 1
            yield mk, p
            if n >= limit:
                return
    if downloader is None:
        return
    for mk in later:
        m = by_key.get(mk)
        if (local_video_for(mk, dirs) is not None or m is None or not m.get("video_key")
                or match_sort_key(mk)[0] != 0):
            continue
        try:
            path = Path(downloader(m["video_key"], Path(download_dir or DOWNLOAD_DIR) / f"{mk}.mp4"))
        except Exception:
            continue
        n += 1
        yield mk, path
        if n >= limit:
            return


def resolve_reference_video(event_key: str, matches, dirs=None, download_dir=None,
                            downloader: Callable = download_match_video):
    """(match_key, video_path) for the event's reference match, downloading it if no
    local footage exists. Raises AutoCalibError (reason) on any failure."""
    mk, path, dl = pick_reference_match(event_key, matches, dirs)
    if mk is None:
        raise AutoCalibError("no local footage and no qual match with a TBA video "
                             "to download")
    if path is not None:
        return mk, path
    out = Path(download_dir or DOWNLOAD_DIR) / f"{mk}.mp4"
    path = Path(downloader(dl["video_key"], out))
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        raise AutoCalibError(f"downloaded {mk} but OpenCV cannot decode {path.name}")
    return mk, path


# ------------------------------------------------------------------------- detection

def make_detect_fn():
    """AprilTag detection with a detector PRIVATE to the calling thread. The shared
    ``apriltag_detect`` singleton holds non-thread-safe C state, and this runs in a
    background thread that may overlap an inference job's drift check or the
    calibrate UI. Detectors are cached per thread and never freed
    (``tag_locator.thread_detector``: pupil_apriltags' destructor can segfault)."""
    from web.field.tag_locator import thread_detector

    def detect(frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # quad_decimate=1: the default (2) halves the image before finding tag
        # outlines, which erases the ~25 px tags of distant corner cameras
        # (2026casnd corner_right: 0 tags at 2, tags 5/8/9/10 at 1). ~5 ms per
        # corner crop, ~30 ms per main crop.
        return {int(d.tag_id): np.asarray(d.corners, float)
                for d in thread_detector(quad_decimate=1.0).detect(gray)}
    return detect


def aggregate_with_best_frame(video, crop, fm, cfg: AutoCalibConfig, detect_fn):
    """``aggregate_tag_corners`` plus the sampled frame with the most detected KNOWN
    tags (the overlay / manual-correction frame). Returns a dict with corners, diag,
    fps, n_frames, best_frame (crop BGR or None), best_t, best_n."""
    best = {"frame": None, "t": None, "n": -1}

    def on_frame(t, sub, dets):
        n = sum(1 for tid in dets if tid in fm.tags)
        if n > best["n"]:
            best.update(frame=sub.copy(), t=float(t), n=n)

    corners, diag, fps, n_frames = aggregate_tag_corners(
        str(video), tuple(crop), cfg.window[0], cfg.window[1], cfg.stride_sec,
        cfg.mad_px, detect_fn=detect_fn, on_frame=on_frame)
    return {"corners": corners, "diag": diag, "fps": fps, "n_frames": n_frames,
            "best_frame": best["frame"], "best_t": best["t"], "best_n": max(0, best["n"])}


def _window_frame_range(n_frames, fps, t_lo, t_hi):
    lo = max(0, int(t_lo * fps))
    hi = min(n_frames - 1, int(t_hi * fps)) if np.isfinite(t_hi) else n_frames - 1
    if hi <= lo:                       # footage shorter than the window: use it all
        lo, hi = 0, max(0, n_frames - 1)
    return lo, hi


def median_of_frames(frames: list):
    """Per-pixel temporal median of same-sized uint8 frames (stays uint8; for an
    even count it takes the upper middle element)."""
    if not frames:
        return None
    st = np.stack(frames)
    k = len(frames) // 2
    return np.partition(st, k, axis=0)[k]


def median_frame(video, crop, t_lo: float, t_hi: float, n: int = 31,
                 return_stack: bool = False):
    """Temporal median of the camera crop over ``n`` frames evenly spaced in
    [t_lo, t_hi] seconds (clamped to the video). None if nothing could be read.
    ``return_stack``: return (median, frames) — the frames let the tag locator test
    that a candidate is STATIC (bolted to the field) rather than a spectator."""
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if n_frames <= 0 or n <= 0:
        cap.release()
        return (None, []) if return_stack else None
    x, y, w, h = (int(c) for c in crop)
    lo, hi = _window_frame_range(n_frames, fps, t_lo, t_hi)
    idxs = sorted({int(round(lo + (hi - lo) * i / max(1, n - 1))) for i in range(n)})
    frames = []
    try:
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if ok and frame is not None:
                sub = frame[y:y + h, x:x + w]
                if not frames or sub.shape == frames[0].shape:
                    frames.append(sub.copy())
    finally:
        cap.release()
    med = median_of_frames(frames)
    return (med, frames) if return_stack else med


def validate_across_window(cal, video, crop, fm, cfg: AutoCalibConfig, detect_fn):
    """``tag_alignment_error`` on ``cfg.validate_samples`` frames evenly spaced over
    the aggregation window (the match proper, like the job's drift check — not the
    pre-match framing). Returns {best_m, median_m, worst_m, frames, gate_best_m,
    gate_tags} where the stats use frames with >= check_min_tags tags and the gate
    (verify / pose-resolve decision) uses frames with >= min_tags tags."""
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x, y, w, h = crop
    lo, hi = _window_frame_range(n_frames, fps, cfg.window[0], cfg.window[1])
    k = max(1, cfg.validate_samples)
    idxs = [int(lo + (hi - lo) * i / (k - 1)) for i in range(k)] if k > 1 else [lo]
    errs, gate_best, gate_n = [], float("inf"), 0
    try:
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            e, n = tag_alignment_error(cal, detect_fn(frame[y:y + h, x:x + w]), fm)
            if n >= cfg.check_min_tags and np.isfinite(e):
                errs.append(e)
            if n >= cfg.min_tags and e < gate_best:
                gate_best, gate_n = e, n
    finally:
        cap.release()
    if not errs:
        return {"best_m": None, "median_m": None, "worst_m": None, "frames": 0,
                "gate_best_m": None, "gate_tags": 0}
    return {"best_m": round(float(min(errs)), 3),
            "median_m": round(float(np.median(errs)), 3),
            "worst_m": round(float(max(errs)), 3), "frames": len(errs),
            "gate_best_m": None if not np.isfinite(gate_best) else round(float(gate_best), 3),
            "gate_tags": int(gate_n)}


# ---------------------------------------------------------------------- geometry/draw

def _project(cal, p):
    """field point -> (u, v) or None if it is behind the camera / not finite."""
    p = np.asarray(p, float)
    if cal.rvec is not None and cal.tvec is not None:
        R, _ = cv2.Rodrigues(np.asarray(cal.rvec, float))
        if (R @ p + np.asarray(cal.tvec, float).reshape(3))[2] <= 0:
            return None
    try:
        u, v = cal.field_to_image(tuple(p))
    except Exception:
        return None
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return float(u), float(v)


def _faces_camera(cal, fm, tid) -> bool:
    """True if tag ``tid``'s face (local +x normal) points toward the camera, i.e. a
    camera could actually decode it. Pose-free (homography) calibrations: True."""
    c = cal.camera_center
    if c is None:
        return True
    from web.field.field_model import _quat_to_matrix
    normal = _quat_to_matrix(fm.tag_quaternion(tid)) @ np.array([1.0, 0.0, 0.0])
    return float(normal @ (np.asarray(c, float) - fm.tag_translation(tid))) > 0


def visible_model_tags(cal, fm, image_size) -> list:
    """Model tags whose centre projects inside the image (in front of the camera,
    face toward it) under ``cal``. Sorted ids."""
    w, h = image_size
    out = []
    for tid in fm.tags:
        if not _faces_camera(cal, fm, tid):
            continue
        uv = _project(cal, fm.tag_translation(tid))
        if uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h:
            out.append(int(tid))
    return sorted(out)


YELLOW = (0, 255, 255)


def render_overlay(frame, cal, fm, auto_corners: dict, used_ids, masks=(),
                   located: Optional[dict] = None):
    """The review overlay on a copy of the camera crop: masks blacked out, field
    wireframe, aggregated detected corners (green), each used tag's model corners
    reprojected under ``cal`` (cyan), and every visible model tag centre labelled
    with its id — green if used, orange if not (an undecoded tag the lens is
    unverified at). ``located``: {tag_id: (u, v)} for tags found by the locator,
    drawn in yellow and labelled "id*". ``cal`` may be None (detections only)."""
    from web import hud_draw
    img = frame.copy()
    for (mx, my, mw, mh) in (masks or ()):
        img[int(my):int(my) + int(mh), int(mx):int(mx) + int(mw)] = 0
    used = {int(t) for t in used_ids}
    H, W = img.shape[:2]
    if cal is not None:
        try:
            hud_draw.draw_field_wireframe(img, cal, fm)
        except Exception as e:
            print(f"  [auto-calib] wireframe draw failed: {e}")
    for tid, corners in (auto_corners or {}).items():
        pts = np.asarray(corners, float).reshape(-1, 2)
        if np.all(np.isfinite(pts)):
            cv2.polylines(img, [pts.astype(np.int32)], True, GREEN, 2)
    if cal is None:
        return img
    for tid in sorted(used):
        if tid not in fm.tags:
            continue
        proj = [_project(cal, p) for p in fm.tag_corners_3d(tid)]
        if all(pp is not None for pp in proj):
            arr = np.array(proj, float)
            if np.all(np.abs(arr) < 4 * max(W, H)):
                cv2.polylines(img, [arr.astype(np.int32)], True, CYAN, 1)
    located = {int(t): uv for t, uv in (located or {}).items()}

    def label(u, v, text, color):
        cv2.putText(img, text, (int(u) + 6, int(v) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (int(u) + 6, int(v) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    for tid in visible_model_tags(cal, fm, (W, H)):
        if tid in located:
            continue
        u, v = _project(cal, fm.tag_translation(tid))
        color = GREEN if tid in used else ORANGE
        cv2.circle(img, (int(u), int(v)), 5, color, -1)
        label(u, v, str(tid), color)
    for tid, (lu, lv) in located.items():
        cv2.drawMarker(img, (int(lu), int(lv)), YELLOW, cv2.MARKER_TILTED_CROSS, 10, 2)
        pr = _project(cal, fm.tag_translation(tid)) if tid in fm.tags else None
        if pr is not None:
            cv2.circle(img, (int(pr[0]), int(pr[1])), 4, YELLOW, 1)
        label(lu, lv, f"{tid}*", YELLOW)
    return img


def encode_jpeg(img, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise AutoCalibError("jpeg encode failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- records

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_record(status: str, reason: str = "", match_key=None, **extra) -> dict:
    """A record in the spec shape with defaults filled in."""
    rec = {"status": status, "method": None, "reason": reason,
           "match_key": match_key, "timestamp": None, "calibration": None,
           "epoch": None, "metrics": {}, "per_tag": {}, "auto_tags": {},
           "preview": None, "created_at": _now(), "resolved_at": None,
           "resolved_version": None}
    rec.update(extra)
    return rec


def _finite(x, nd):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _metrics(n_tags, cal, val, unused, center_shift=None, rot_deg=None) -> dict:
    return {"n_tags": int(n_tags),
            "rms_px": None if cal is None else round(float(cal.rms_error), 2),
            "val_best_m": val.get("best_m") if val else None,
            "val_median_m": val.get("median_m") if val else None,
            "val_worst_m": val.get("worst_m") if val else None,
            "val_frames": val.get("frames", 0) if val else 0,
            "val_gate_best_m": val.get("gate_best_m") if val else None,
            "val_gate_tags": val.get("gate_tags", 0) if val else 0,
            "center_shift_m": _finite(center_shift, 3),
            "rotation_delta_deg": _finite(rot_deg, 2),
            "unused_visible_tags": list(unused or [])}


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def median_decodes(median, detect_fn, roi_detect_fn=None, scales=(2, 3)) -> dict:
    """Decode tags on the temporal-median frame: the normal detector, then the
    no-decimation ROI detector on 2x / 3x upscaled copies — small or oblique tags
    (e.g. a steep overhead view: 1 tag at 1x, 4 at 2x on 2026utwv qm1) only decode
    there. First hit per tag wins; corners in median-frame px."""
    out = {int(t): np.asarray(c, float) for t, c in detect_fn(median).items()}
    if roi_detect_fn is None:
        return out
    for k in scales:
        up = cv2.resize(median, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        for t, c in roi_detect_fn(up).items():
            out.setdefault(int(t), np.asarray(c, float) / k)
    return out


def _grow_with_locator(cal, how, img, fm, corners, image_size, cfg, roi_detect_fn, judge,
                       base, extra, lens=None, stack=None):
    """Run the undecoded-tag locator; adopt its calibration when it located anything.
    Fills base["auto_tags"] (ROI decodes, 4 corners) / base["manual_tags"] (centre
    points) and extra["located"/"candidates"/"notes"]. Returns (cal, description)."""
    from web.field import tag_locator as tl
    lcfg = tl.LocatorConfig(init_model="pose" if lens is not None else cfg.locate_init_model,
                            jev_verify=cfg.jev_verify, jev_min_p=cfg.jev_min_p)
    try:
        res = tl.locate_undecoded(img, fm, corners, image_size, cfg.field_season, lcfg,
                                  detect_fn=roi_detect_fn, judge=judge, lens=lens,
                                  stack=stack)
    except Exception as e:
        extra["notes"].append(f"tag locator failed: {type(e).__name__}: {e}")
        return cal, how
    extra["notes"].extend(res.notes)
    extra["candidates"] = res.candidates
    if res.calibration is None or not res.located:
        if res.calibration is not None:
            extra["notes"].append("locator found no undecoded tags")
        return cal, how
    extra["located"] = res.located
    for key, (img_pts, _obj) in res.corr.items():
        tid = int(key[1:])
        if key.startswith("r"):
            base["auto_tags"][str(tid)] = np.asarray(img_pts, float).round(2).tolist()
        elif key.startswith("m"):
            u, v = np.asarray(img_pts, float).reshape(-1, 2)[0]
            base["manual_tags"].append({"id": tid, "u": round(float(u), 2),
                                        "v": round(float(v), 2)})
    n_roi = sum(1 for d in res.located if d["method"] == "roi_decode")
    if lens is not None:                 # keep the epoch's field season / crop semantics
        res.calibration.image_size = lens.image_size
    return res.calibration, (f"{how} + {len(res.located)} located "
                             f"({n_roi} ROI-decoded, {len(res.located) - n_roi} quad), "
                             f"model {res.model}")


def run_camera(video, match_key: str, camera: str, crop, fm, versions,
               cfg: Optional[AutoCalibConfig] = None, detect_fn=None,
               masks=(), judge=None, roi_detect_fn=None,
               force_resolve: bool = False) -> AutoCalibResult:
    """Auto-calibrate ONE camera from ONE reference match. Pure w.r.t. persistence:
    returns the record (status ready / verified / failed), the candidate calibration
    (or None) and the overlay image (camera crop, or None if no frame was read).

    ``versions``: this camera's existing versions (``store.load_versions``). Empty ->
    full solve; else verify the epoch covering ``match_key`` and, if it drifted,
    re-solve the pose against its lens. ``masks`` are crop-local rects blacked out
    in the overlay only. ``judge``: Jev tag-kind judge for the locator
    (``tag_locator.make_jev_judge``); None with ``cfg.jev_verify`` -> only ROI
    decodes are located. ``force_resolve``: with versions, skip the "verified"
    shortcut and always re-solve the pose from this footage (the caller already
    decided the epoch does not hold, e.g. the per-match corner step)."""
    cfg = cfg or AutoCalibConfig()
    detect_fn = detect_fn or make_detect_fn()
    if roi_detect_fn is None:
        from web.field.tag_locator import make_roi_detect_fn
        roi_detect_fn = make_roi_detect_fn()
    crop = tuple(int(c) for c in crop)
    image_size = (crop[2], crop[3])

    agg = aggregate_with_best_frame(video, crop, fm, cfg, detect_fn)
    if agg["n_frames"] <= 0:
        raise AutoCalibError(f"cannot read frames from {Path(str(video)).name}")
    corners = {tid: c for tid, c in agg["corners"].items() if tid in fm.tags}
    per_tag = {str(int(t)): {"inliers": int(d[0]), "seen": int(d[1]),
                             "std_px": round(float(d[2]), 2)}
               for t, d in agg["diag"].items() if t in fm.tags}
    # Temporal median of the crop: robots/balls blocking a tag vanish. A tag decoded
    # ONLY there joins the set (the multi-frame aggregate wins when both have it).
    median = None
    stack = None
    med_dec = {}
    if cfg.median_frames > 0:
        mf = median_frame(video, crop, cfg.window[0], cfg.window[1],
                          cfg.median_frames, return_stack=True)
        median, stack = mf if isinstance(mf, tuple) else (mf, None)
    if median is not None:
        med_dec = {t: c for t, c in median_decodes(median, detect_fn, roi_detect_fn).items()
                   if t in fm.tags}
        for tid, c in med_dec.items():
            if tid in fm.tags and tid not in corners:
                corners[tid] = np.asarray(c, float)
                per_tag[str(int(tid))] = {"inliers": 0, "seen": 0, "std_px": None,
                                          "source": "median"}
    background = median if median is not None else agg["best_frame"]
    loc_stack = stack if median is not None else None
    corr = {tid: (c, fm.tag_corners_3d(tid)) for tid, c in corners.items()}
    base = dict(
        match_key=match_key,
        timestamp=None if agg["best_t"] is None else round(agg["best_t"], 1),
        per_tag=per_tag, manual_tags=[],
        auto_tags={str(int(t)): np.asarray(c, float).round(2).tolist()
                   for t, c in corners.items()})
    extra = {"located": [], "candidates": [], "notes": []}

    def validate(c):
        """Per-frame validation; when no single frame shows >= min_tags tags (steep
        or distant views decode few tags per frame) fall back to the median frame's
        decodes, flagged with val["source"] = "median"."""
        v = validate_across_window(c, video, crop, fm, cfg, detect_fn)
        if v["gate_best_m"] is None and len(med_dec) >= cfg.min_tags:
            e, n = tag_alignment_error(c, med_dec, fm)
            if n >= cfg.min_tags and np.isfinite(e):
                v.update(gate_best_m=round(float(e), 3), gate_tags=int(n), source="median",
                         best_m=v["best_m"] if v["best_m"] is not None else round(float(e), 3))
        return v

    def finish(status, method, reason, cal, val, overlay_cal, epoch=None,
               center_shift=None, rot_deg=None):
        # Hard gate before anything can be confirmed / adopted / trusted: a mirrored,
        # upside-down or behind-the-camera pose never comes out ready or verified.
        gate_cal = cal if cal is not None else overlay_cal
        if status in ("ready", "verified") and gate_cal is not None and corr:
            why = check_solve_plausible(gate_cal, corr, cfg.field_season)
            if why is not None:
                status = "failed"
                reason = (f"{reason}; REJECTED: physically implausible pose ({why}) — "
                          f"calibrate manually")
        located = {d["id"]: (d["u"], d["v"]) for d in extra["located"]}
        have = set(corr) | set(located)
        unused = []
        if overlay_cal is not None:
            unused = [t for t in visible_model_tags(overlay_cal, fm, image_size)
                      if t not in have]
        metrics = _metrics(len(corr), cal if cal is not None else overlay_cal,
                           val, unused, center_shift, rot_deg)
        metrics["located_tags"] = [{k: d[k] for k in ("id", "method", "jev_p", "residual_px")}
                                   for d in extra["located"]]
        if val and val.get("source") == "median":
            metrics["val_source"] = "median"
            reason = reason + " [validated on the median frame: no single frame shows "\
                              f"{cfg.min_tags}+ tags]"
        if extra["notes"]:
            reason = reason + " [" + "; ".join(extra["notes"]) + "]"
        rec = new_record(status, reason, method=method, epoch=epoch,
                         calibration=None if cal is None else cal.to_json(),
                         metrics=metrics, **base)
        overlay = None
        if background is not None:
            overlay = render_overlay(background, overlay_cal, fm, corners,
                                     sorted(have), masks, located=located)
        return AutoCalibResult(record=rec, calibration=cal, overlay=overlay,
                               median=median, candidates=extra["candidates"])

    # ---- no calibration for this camera yet: full solve -------------------------
    if not versions:
        if len(corr) < cfg.min_tags:
            return finish("failed", "full_solve",
                          f"only {len(corr)} known tags found across {match_key} "
                          f"(need >= {cfg.min_tags}) — calibrate manually",
                          None, None, None)
        # Start from the CONSTRAINED lens model (the locator's best extrapolator), not
        # a fully free one: if the locator then finds nothing, the free model would
        # be kept — perfect on a tight cluster of tags, wild everywhere else.
        from web.field import tag_locator as tl
        cal, err = None, None
        for model in (cfg.locate_init_model, "pp_k0"):
            try:
                cal = tl.solve_model(corr, image_size, cfg.field_season, model)
                break
            except CalibrationError as e:
                err = e
        if cal is None:
            return finish("failed", "full_solve", f"full solve failed: {err}",
                          None, None, None)
        how = f"full solve from {len(corr)} tags"
        if cfg.locate_undecoded and background is not None:
            cal, how = _grow_with_locator(cal, how, background, fm, corners, image_size,
                                          cfg, roi_detect_fn, judge, base, extra,
                                          stack=loc_stack)
        used = set(int(t) for t in corr) | {int(d["id"]) for d in extra.get("located") or []}
        spread = tag_spread_m(fm, used)
        if not extra.get("located") and spread >= cfg.min_full_solve_spread_m:
            # The locator adds nothing but the decoded tags themselves are numerous and
            # spread: free the lens as the locator would (k1, then full), keeping a
            # freer model only if it fits clearly better without distortion fold-back.
            cal, freed = free_lens_ladder(cal, corr, image_size, cfg.field_season, fm)
            if freed:
                how = f"{how}, model {freed}"
        extra["notes"].append(f"tags span {spread:.1f} m of the field")
        if spread < cfg.min_full_solve_spread_m:
            val = validate(cal)
            return finish("failed", "full_solve",
                          f"{how}: the tags only span {spread:.1f} m of the field (one "
                          f"structure) — that can't determine the lens (needs >= "
                          f"{cfg.min_full_solve_spread_m:.1f} m); correct manually by "
                          f"clicking a few far tags",
                          cal, val, cal)
        val = validate(cal)
        gate = val["gate_best_m"]
        if gate is None or gate >= cfg.drift_max_m:
            # Confirming this would only fail the job's drift check — leave it for
            # manual correction, but keep the candidate + overlay for inspection.
            return finish("failed", "full_solve",
                          f"{how} (rms {cal.rms_error:.2f} px) does not validate: best "
                          f"{_fmt(gate)} m on >= {cfg.min_tags}-tag frames vs the "
                          f"{cfg.drift_max_m:.2f} m gate — correct manually",
                          cal, val, cal)
        return finish("ready", "full_solve",
                      f"{how} (rms {cal.rms_error:.2f} px); "
                      f"validation best {_fmt(val['best_m'])} / median "
                      f"{_fmt(val['median_m'])} / worst {_fmt(val['worst_m'])} m over "
                      f"{val['frames']} frames — check the overlay, then confirm",
                      cal, val, cal)

    # ---- existing epoch: verify, then pose re-solve if it drifted ---------------
    epoch = select_epoch(versions, match_key)
    ecal = epoch["calibration"]
    vid = epoch["version_id"]
    val = validate(ecal)
    gate = val["gate_best_m"]
    if (not force_resolve and gate is not None and gate < cfg.drift_max_m
            and val["gate_tags"] >= cfg.min_tags):
        return finish("verified", "verify",
                      f"epoch {vid} still aligns on {match_key}: best {gate:.2f} m over "
                      f"{val['gate_tags']} tags (gate {cfg.drift_max_m:.2f} m)",
                      None, val, ecal, epoch=vid)
    if gate is None:
        drift_txt = (f"could not verify epoch {vid} on {match_key} (no sampled frame "
                     f"showed >= {cfg.min_tags} known tags)")
    else:
        drift_txt = (f"epoch {vid} misaligned on {match_key} (best {gate:.2f} m on "
                     f">= {cfg.min_tags}-tag frames)")
    if ecal.K is None or ecal.mode not in ("camera", "pinhole"):
        return finish("failed", "pose_resolve",
                      f"{drift_txt}; its mode '{ecal.mode}' has no lens to re-solve "
                      f"the pose against — calibrate manually", None, val, ecal, epoch=vid)
    if len(corr) < cfg.min_tags:
        return finish("failed", "pose_resolve",
                      f"{drift_txt}; only {len(corr)} known tags found (need >= "
                      f"{cfg.min_tags}) — calibrate manually", None, val, ecal, epoch=vid)
    try:
        cand = resolve_pose(corr, ecal)
    except CalibrationError as e:
        return finish("failed", "pose_resolve", f"{drift_txt}; pose solve failed: {e}",
                      None, val, ecal, epoch=vid)
    how = ""
    if cfg.locate_undecoded and background is not None and ecal.mode == "pinhole":
        # Few decoded tags (often one structure) condition a pose badly: grow it
        # outward from the epoch's lens ("pose" -> "f_lens" -> ... ladder).
        cand, how = _grow_with_locator(cand, "", background, fm, corners, image_size, cfg,
                                       roi_detect_fn, judge, base, extra, lens=ecal,
                                       stack=loc_stack)
        how = f" [{how.strip(' +')}]" if how else ""
    cval = validate(cand)
    pcfg = PoseFixConfig(min_tags=cfg.min_tags, drift_max_m=cfg.drift_max_m,
                         max_center_shift_m=cfg.max_center_shift_m)
    best = cval["gate_best_m"] if cval["gate_best_m"] is not None else float("inf")
    verdict = evaluate_fix(ecal, cand, best, cval["gate_tags"], pcfg)
    status = "ready" if verdict.passed else "failed"
    reason = f"{drift_txt}; {verdict.reason}{how}"
    if not verdict.passed:
        reason += " — calibrate manually"
    return finish(status, "pose_resolve", reason, cand, cval, cand, epoch=vid,
                  center_shift=verdict.center_shift_m, rot_deg=verdict.rotation_delta_deg)


# --------------------------------------------------------------- event layout

LAYOUT_ORIGIN_TOL_PX = 4
_LAYOUT_COLOURS = {"main": (0, 255, 0), "corner_left": (255, 200, 0),
                   "corner_right": (255, 0, 255)}


def cameras_to_dict(cams: dict, frame_size=(1920, 1080)) -> dict:
    """{name: CameraLayout} -> the {"frame_size", "cameras"} dict save_event_layout takes."""
    return {"frame_size": list(frame_size),
            "cameras": {n: {"crop": list(c.crop), "masks": [list(m) for m in c.masks]}
                        for n, c in cams.items()}}


def solved_crop(version: dict, default_layout: dict, camera: str):
    """The crop a calibration version was solved in: ``source.crop`` when recorded,
    else the global default layout's crop (everything before per-event layouts)."""
    src = version.get("source") or {}
    if src.get("crop"):
        return tuple(int(v) for v in src["crop"])
    cl = default_layout.get(camera)
    return tuple(cl.crop) if cl is not None else None


def layout_conflicts(new_cams: dict, versions_by_cam: dict, default_layout: dict,
                     tol: int = LAYOUT_ORIGIN_TOL_PX) -> list:
    """Why ``new_cams`` ({name: CameraLayout}) can't be adopted: calibrations are
    CROP-LOCAL pixels, so a camera with versions must keep the crop ORIGIN they were
    solved in (within ``tol`` px). Returns human-readable conflicts (empty = safe)."""
    out = []
    for cam, vs in sorted(versions_by_cam.items()):
        for v in vs or ():
            sc = solved_crop(v, default_layout, cam)
            if sc is None:
                continue
            nc = new_cams.get(cam)
            if nc is None:
                msg = f"{cam} has calibrations but the new layout has no {cam} view"
            elif max(abs(nc.crop[0] - sc[0]), abs(nc.crop[1] - sc[1])) > tol:
                axis = "y" if abs(nc.crop[1] - sc[1]) > tol else "x"
                i = 1 if axis == "y" else 0
                msg = f"{cam} {axis} {sc[i]} vs new {nc.crop[i]}"
            else:
                continue
            if msg not in out:
                out.append(msg)
    return out


def render_layout_preview(composite_bgr, cams: dict, other: Optional[dict] = None):
    """The composite frame with each camera's crop box + name and its overlay masks
    (hatched red). ``other``: a second layout drawn thin/dashed (e.g. the detection
    that was not adopted)."""
    img = composite_bgr.copy()
    for n, c in cams.items():
        x, y, w, h = c.crop
        col = _LAYOUT_COLOURS.get(n, (0, 255, 255))
        for (mx, my, mw, mh) in c.masks:
            ov = img.copy()
            cv2.rectangle(ov, (x + mx, y + my), (x + mx + mw, y + my + mh), (0, 0, 255), -1)
            img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), col, 3)
        cv2.putText(img, f"{n} {x},{y} {w}x{h}", (x + 8, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"{n} {x},{y} {w}x{h}", (x + 8, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    for n, c in (other or {}).items():
        x, y, w, h = c.crop
        for i in range(x, x + w, 24):                  # dashed top/bottom
            cv2.line(img, (i, y), (min(i + 12, x + w), y), (0, 165, 255), 2)
            cv2.line(img, (i, y + h - 1), (min(i + 12, x + w), y + h - 1), (0, 165, 255), 2)
        for j in range(y, y + h, 24):
            cv2.line(img, (x, j), (x, min(j + 12, y + h)), (0, 165, 255), 2)
            cv2.line(img, (x + w - 1, j), (x + w - 1, min(j + 12, y + h)), (0, 165, 255), 2)
    return img


def detect_with_preview(video):
    """(detect_layout()-shaped dict, composite median frame) from one sampling pass."""
    from web.field import layout_detect as ld
    t0 = time.time()
    frames, times, dur = ld.sample_frames(str(video), (20, None), 40)
    if len(frames) < 3:
        raise AutoCalibError(f"could not read frames from {Path(str(video)).name} "
                             f"for layout detection")
    out = ld.detect_layout_from_frames(frames)
    out["diag"].update({"video": str(video), "duration_s": round(dur, 2),
                        "t_total_s": round(time.time() - t0, 2)})
    return out, median_of_frames(frames)


def invalidate_stale_auto_records(event_key: str, old_cams: dict, new_cams: dict,
                                  tol: int = LAYOUT_ORIGIN_TOL_PX) -> list:
    """A ``ready`` auto-calibration candidate is crop-local to the crop it was solved
    in: when a camera's crop origin moves (or the camera disappears), mark its ready
    record failed ("layout changed — re-run") so it can't be confirmed into the
    wrong crop. Returns the cameras invalidated."""
    from web.field import store
    out = []
    for cam in sorted(set(old_cams) | set(new_cams)):
        o, n = old_cams.get(cam), new_cams.get(cam)
        moved = (o is None or n is None
                 or max(abs(o.crop[0] - n.crop[0]), abs(o.crop[1] - n.crop[1])) > tol)
        if not moved:
            continue
        rec = store.load_auto_record(event_key, cam)
        if rec and rec.get("status") == "ready":
            store.update_auto_record(event_key, cam, status="failed",
                                     reason="layout changed — re-run")
            out.append(cam)
    return out


def ensure_event_layout(event_key: str, video, *, detector: Optional[Callable] = None,
                        force: bool = False, replace_confirmed: bool = False,
                        log: Callable = print) -> dict:
    """Layout step of an auto-calibration run (once per event, before any camera).

    With no event layout record (or ``force``, i.e. re-detect): detect the broadcast
    layout on the reference video. The detection is ADOPTED (status "auto") only when
    it is safe: it has a main view, no camera with calibration versions would move
    its crop origin (versions are crop-local pixels), and the event's layout is not
    a confirmed/manual one (unless ``replace_confirmed``). Otherwise the layout in
    effect is KEPT — the existing per-event record when it is usable (only its
    ``detected`` copy and reason are updated), else the global default (status
    "kept_default", or "rejected" when no main view was found). Adopting a layout
    that moves a crop invalidates ready auto records for that camera. Renders the
    preview JPEG next to the record. Returns the record."""
    from web.field import composite_layout as cl
    from web.field import store
    rec = cl.load_event_layout_record(event_key)
    if rec is not None and not force:
        return rec
    detector = detector or detect_with_preview
    detected, composite = detector(video)
    det_cams = cl.layout_to_cameras(detected)
    default = cl.load_layout()
    current = cl.load_layout(event_key=event_key)      # in effect before this detection
    usable = rec is not None and rec.get("status") in cl.USABLE_STATUSES
    protected = (usable and not replace_confirmed
                 and (rec.get("status") == "confirmed" or rec.get("source") == "manual"))
    names = sorted(set(default) | set(det_cams) | set(current))
    versions = {c: store.load_versions(event_key, c) for c in names}
    conflicts = layout_conflicts(det_cams, versions, default)
    frame_size = detected.get("frame_size") or (1920, 1080)
    det_dict = cameras_to_dict(det_cams, frame_size)
    keep_why = None
    if "main" not in det_cams:
        keep_why = "layout detection found no main view"
    elif protected:
        keep_why = ("the event's confirmed layout is kept (re-detection stored under "
                    "'detected'; replace it explicitly to adopt)")
    elif conflicts:
        keep_why = ("existing calibrations were solved in other crops ("
                    + "; ".join(conflicts) + ")")
    if keep_why is None:
        status = "auto"
        reason = f"detected {len(det_cams)} view(s): {', '.join(sorted(det_cams))}"
        cl.save_event_layout(event_key, detected, status="auto", source="detect",
                             reason=reason, detected=det_dict["cameras"])
        stale = invalidate_stale_auto_records(event_key, current, det_cams)
        if stale:
            reason += f"; ready results invalidated for {', '.join(stale)}"
    elif usable:
        status = rec["status"]
        reason = keep_why + " — current layout kept"
        cl.save_event_layout(event_key, rec, status=status,
                             source=rec.get("source") or "detect",
                             reason=reason, detected=det_dict["cameras"])
    else:
        status = "rejected" if "main" not in det_cams else "kept_default"
        reason = keep_why + " — default layout kept"
        chosen = (detected if status == "rejected" else
                  dict(cameras_to_dict(default, frame_size),
                       confidence=detected.get("confidence"), diag=detected.get("diag")))
        cl.save_event_layout(event_key, chosen, status=status, source="detect",
                             reason=reason, detected=det_dict["cameras"])
    if composite is not None:
        try:
            fp = layout_frame_path(event_key)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(encode_jpeg(composite, 90))
            refresh_layout_preview(event_key, det_cams if status != "auto" else None)
        except Exception as e:
            log(f"[auto-calib] {event_key}: layout preview failed: {e}")
    log(f"[auto-calib] {event_key}: layout {status} — {reason}")
    return cl.load_event_layout_record(event_key)


def layout_frame_path(event_key: str) -> Path:
    """The raw composite median frame kept next to the layout record, so the
    preview can be re-rendered after a manual layout edit."""
    from web.field import composite_layout as cl
    p = cl.event_layout_preview_path(event_key)
    return p.with_name(p.stem + "_frame.jpg")


def refresh_layout_preview(event_key: str, other: Optional[dict] = None) -> bool:
    """Re-render the layout preview from the saved composite frame and the event's
    current record (crops of the record; ``other`` drawn dashed). False if either
    is missing."""
    from web.field import composite_layout as cl
    rec = cl.load_event_layout_record(event_key)
    fp = layout_frame_path(event_key)
    if rec is None or not fp.is_file():
        return False
    frame = cv2.imread(str(fp))
    if frame is None:
        return False
    if other is None and rec.get("status") != "auto" and rec.get("detected"):
        try:
            other = cl.layout_to_cameras({"cameras": rec["detected"]})
        except (KeyError, TypeError, ValueError):
            other = None
    img = render_layout_preview(frame, cl.layout_to_cameras(rec), other)
    cl.event_layout_preview_path(event_key).write_bytes(encode_jpeg(img, 85))
    return True


def record_age_s(rec: dict, now: Optional[float] = None) -> Optional[float]:
    """Seconds since the record's created_at (local time), or None if unparsable."""
    try:
        t = time.mktime(time.strptime(rec.get("created_at") or "", "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        return None
    return (time.time() if now is None else now) - t


def skip_reason(rec: Optional[dict], force: bool, now: Optional[float] = None):
    """Why a (non-forced) run should leave this camera alone, or None to run it:
    settled records (ready / verified / confirmed) always, failed ones while younger
    than FAILED_RETRY_AFTER_S."""
    if rec is None or force:
        return None
    st = rec.get("status")
    if st in SETTLED_STATUSES:
        return st
    if st == "failed":
        age = record_age_s(rec, now)
        if age is not None and age < FAILED_RETRY_AFTER_S:
            return f"failed {age / 3600:.1f} h ago (retried after {FAILED_RETRY_AFTER_S // 3600} h)"
    return None

"""Automatic camera-pose recovery when a match drifts off its saved calibration.

The lens (K, D) is physically unchanged between matches or days — only where the
camera *points* moves (a pan/tilt/roll about the nodal point), and more rarely where
it *sits* (a small translation). So when the in-match drift check fails we can often
recover without a full manual recalibration by re-solving ONLY the 6-DOF pose against
the same intrinsics, via PnP over AprilTags we can still auto-detect.

This module has three layers, split so the numeric decision is unit-testable without
video or a GPU:
  * ``resolve_pose``      — re-solve pose from correspondences against a base cal.
  * ``evaluate_fix``      — PURE guardrail decision (error gate + camera-center sanity).
  * ``resolve_pose_from_video`` — glue: robustly aggregate tags across in-match frames,
                             solve, validate on held-out frames, then evaluate.

The result is never adopted here — it is returned for human approval (the operator
eyeballs the reprojected wireframe over the replay before it becomes a new epoch).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from web.field.apriltag_detect import detect_tags
from web.field.calibration import (CameraCalibration, Calibrator, CalibrationError,
                                    tag_alignment_error)


@dataclass
class PoseFixConfig:
    """Thresholds for accepting an automatic pose re-solve.

    ``drift_max_m`` mirrors CALIB_DRIFT_MAX_M — the fix must bring alignment back
    under the same gate the original calibration is held to. ``max_center_shift_m``
    is the sanity bound on how far the camera *center* may move: a broadcast camera
    that pivoted keeps its optical center within a few cm, so a large jump means
    either a bad solve or a genuinely re-placed camera — either way, not a safe
    auto-adopt. ``min_tags`` guards conditioning: 6-DOF pose (esp. translation) is
    under-determined from too few or too-clustered tags."""
    min_tags: int = 3
    drift_max_m: float = 0.5
    max_center_shift_m: float = 1.0
    # Video aggregation (skip the pre-auto framing; median-filter occlusion).
    window: tuple = (18.0, 1e9)
    stride_sec: float = 1.0
    mad_px: float = 5.0
    validate_samples: int = 12


@dataclass
class PoseFixResult:
    passed: bool
    reason: str
    calibration: Optional[CameraCalibration] = None
    best_err_m: float = float("inf")
    n_tags: int = 0
    center_shift_m: float = float("inf")
    rotation_delta_deg: float = float("inf")
    diag: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """JSON-safe summary for the approval UI (no ndarrays)."""
        return {
            "passed": self.passed, "reason": self.reason,
            "best_err_m": None if not np.isfinite(self.best_err_m) else round(self.best_err_m, 3),
            "n_tags": self.n_tags,
            "center_shift_m": None if not np.isfinite(self.center_shift_m) else round(self.center_shift_m, 3),
            "rotation_delta_deg": None if not np.isfinite(self.rotation_delta_deg) else round(self.rotation_delta_deg, 2),
            "rms_error_px": round(self.calibration.rms_error, 2) if self.calibration else None,
            "diag": self.diag,
        }


def resolve_pose(correspondences: dict, base_cal: CameraCalibration) -> CameraCalibration:
    """Re-solve pose from tag correspondences against ``base_cal``'s fixed intrinsics.
    correspondences: {tag_id: (corners_2d (N,2), corners_3d (N,3))}. Raises if the base
    calibration has no projective lens model to reuse (e.g. homography mode)."""
    if base_cal.K is None or base_cal.mode not in ("camera", "pinhole"):
        raise CalibrationError(
            "auto pose fix needs a projective (camera/pinhole) base calibration with "
            f"intrinsics; got mode='{base_cal.mode}'")
    init = ((base_cal.rvec, base_cal.tvec)
            if base_cal.rvec is not None and base_cal.tvec is not None else None)
    return Calibrator.solve_pose(
        correspondences, K=base_cal.K, D=base_cal.D, image_size=base_cal.image_size,
        field_season=base_cal.field_season, mode=base_cal.mode, init=init)


def _pose_deltas(base_cal: CameraCalibration, cand_cal: CameraCalibration):
    """(camera-center shift in meters, rotation change in degrees) base -> candidate."""
    c0 = np.asarray(base_cal.camera_center, float)
    c1 = np.asarray(cand_cal.camera_center, float)
    center_shift = float(np.linalg.norm(c1 - c0))
    R0, _ = cv2.Rodrigues(np.asarray(base_cal.rvec, float))
    R1, _ = cv2.Rodrigues(np.asarray(cand_cal.rvec, float))
    cos = (np.trace(R1 @ R0.T) - 1.0) / 2.0
    rot_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return center_shift, rot_deg


def evaluate_fix(base_cal: CameraCalibration, cand_cal: CameraCalibration,
                 best_err_m: float, n_tags: int, cfg: PoseFixConfig) -> PoseFixResult:
    """PURE decision: does the re-solved pose clear every guardrail? No IO.

    Fails (does not adopt) if too few tags conditioned the solve, if alignment is
    still worse than the drift gate, or if the camera center moved implausibly far
    for a pivot. All failing reasons are collected so the operator sees the full
    picture, not just the first tripwire."""
    center_shift, rot_deg = _pose_deltas(base_cal, cand_cal)
    reasons = []
    if n_tags < cfg.min_tags:
        reasons.append(f"only {n_tags} tags aligned (need >= {cfg.min_tags})")
    if not np.isfinite(best_err_m) or best_err_m >= cfg.drift_max_m:
        reasons.append(f"post-fix alignment {best_err_m:.2f} m is not under the "
                       f"{cfg.drift_max_m:.2f} m gate")
    if center_shift > cfg.max_center_shift_m:
        reasons.append(f"camera center moved {center_shift:.2f} m "
                       f"(> {cfg.max_center_shift_m:.2f} m) — reads as a re-placed "
                       f"camera, not a pivot; not safe to auto-adopt")
    passed = not reasons
    return PoseFixResult(
        passed=passed,
        reason="; ".join(reasons) if reasons
               else f"pose re-solve holds: {best_err_m:.2f} m over {n_tags} tags, "
                    f"center moved {center_shift:.2f} m, rotated {rot_deg:.1f} deg",
        calibration=cand_cal, best_err_m=best_err_m, n_tags=n_tags,
        center_shift_m=center_shift, rotation_delta_deg=rot_deg)


def aggregate_tag_corners(video: str, crop: tuple, t_min: float, t_max: float,
                          stride_sec: float, mad_px: float, *, detect_fn=None,
                          on_frame=None):
    """Robustly aggregate each tag's corner pixels over a time window.

    A tag need only be unobscured in SOME sampled frames — we median its corners
    across all detections, then MAD-filter frames whose worst corner strays > mad_px
    from that median (rejecting a minority of off-framing/occluded frames) and median
    again. Returns ({tag_id: median_corners(4,2)}, {tag_id: (n_inliers, n_seen, std_px)},
    fps, n_frames).

    Optional: ``detect_fn(crop_bgr) -> {tag_id: (4,2)}`` replaces the shared
    module-level detector (e.g. a private Detector for a background thread), and
    ``on_frame(t, crop_bgr, detections)`` is called for every sampled frame (e.g. to
    keep the frame with the most tags)."""
    detect_fn = detect_fn or detect_tags
    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x, y, w, h = crop
    raw: dict[int, list] = defaultdict(list)
    stride = max(1, int(stride_sec * fps))
    for fidx in range(0, n_frames, stride):
        t = fidx / fps
        if t < t_min or t > t_max:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        sub = frame[y:y + h, x:x + w]
        dets = detect_fn(sub)
        if on_frame is not None:
            on_frame(t, sub, dets)
        for tid, corners in dets.items():
            raw[tid].append(np.asarray(corners, float))
    cap.release()

    out, diag = {}, {}
    for tid, dets in raw.items():
        stack = np.stack(dets)                                    # (n,4,2)
        med = np.median(stack, axis=0)
        dist = np.linalg.norm(stack - med, axis=2).max(axis=1)    # per-frame worst corner
        inliers = stack[dist <= mad_px]
        if len(inliers) == 0:
            inliers = stack
        out[tid] = np.median(inliers, axis=0)
        std = inliers.reshape(len(inliers), -1).std(axis=0).mean()
        diag[tid] = (len(inliers), len(dets), float(std))
    return out, diag, fps, n_frames


def _validate_across_match(cal: CameraCalibration, video: str, crop: tuple, fm,
                           n_frames: int, min_tags: int, samples: int):
    """Best (lowest) per-frame median tag-alignment error, in meters, over evenly
    sampled frames — mirrors the pipeline's drift check so the gate is apples-to-apples.
    Returns (best_err_m, n_tags_at_best)."""
    cap = cv2.VideoCapture(video)
    x, y, w, h = crop
    best_err, best_n = float("inf"), 0
    try:
        for i in range(samples):
            fidx = int(n_frames * i / max(1, samples - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            err_m, ntags = tag_alignment_error(cal, detect_tags(frame[y:y + h, x:x + w]), fm)
            if ntags >= min_tags and err_m < best_err:
                best_err, best_n = err_m, ntags
    finally:
        cap.release()
    return best_err, best_n


def resolve_pose_from_video(video: str, crop: tuple, base_cal: CameraCalibration,
                            fm, cfg: Optional[PoseFixConfig] = None) -> PoseFixResult:
    """Attempt an automatic pose re-solve for a drifted match. Aggregates tags across
    the window, re-solves pose against the fixed intrinsics, validates on held-out
    frames, and returns the guardrail verdict. Never persists anything."""
    cfg = cfg or PoseFixConfig()
    corners, diag, fps, n_frames = aggregate_tag_corners(
        video, crop, cfg.window[0], cfg.window[1], cfg.stride_sec, cfg.mad_px)
    corr = {tid: (c, fm.tag_corners_3d(tid)) for tid, c in corners.items() if tid in fm.tags}
    agg_diag = {"tags_aggregated": sorted(corr), "n_frames": n_frames,
                "per_tag": {int(t): {"inliers": d[0], "seen": d[1], "std_px": round(d[2], 2)}
                            for t, d in diag.items()}}
    if len(corr) < cfg.min_tags:
        return PoseFixResult(
            passed=False,
            reason=f"only {len(corr)} known tags aggregated across the match "
                   f"(need >= {cfg.min_tags}) — too occluded/off-framing to auto-fix",
            n_tags=len(corr), diag=agg_diag)
    try:
        cand = resolve_pose(corr, base_cal)
    except CalibrationError as e:
        return PoseFixResult(passed=False, reason=f"pose solve failed: {e}", diag=agg_diag)
    best_err, best_n = _validate_across_match(
        cand, video, crop, fm, n_frames, cfg.min_tags, cfg.validate_samples)
    result = evaluate_fix(base_cal, cand, best_err, best_n, cfg)
    result.diag = agg_diag
    return result

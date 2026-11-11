"""Per-match corner-camera calibration (job time, auto-adopted when it validates).

Broadcast corner cameras are re-aimed / re-zoomed between matches (2026cur: the June
corner versions are 1.5-2.1 m off on every match, a qm11 solve holds only around
qm11), and the job's drift check covers only the main view — so without this step
the corners feed metre-off positions into fusion.

For each corner view of a job, ``corner_step``:
  1. CHECK the covering epoch on a few in-match frames of THIS match. Pass -> keep.
  2. Otherwise RE-SOLVE from this match's footage with the auto-calibration machinery
     (aggregated tags + temporal median + undecoded-tag locator + Jev judge), starting
     from the epoch's lens (pose -> focal -> freer models as tags accumulate), and
     validate it within the match against a stricter gate. Pass -> "adopted": the
     caller saves a NEW version anchored at this match and uses it immediately.
  3. Fail -> "dropped": the caller excludes the corner view from this job.
A match whose adopted version an operator rejected is never re-solved again (the
corner is dropped unless the covering epoch passes the check).

Pure w.r.t. persistence: returns a ``CornerOutcome``; the job (web/app.py) saves the
version, the per-match log entry and the preview.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from web.field import auto_calibrate as ac
from web.field import camera_motion
from web.field.calibration import pose_plausibility, tag_alignment_error

LABEL = "auto per-match (corner)"


@dataclass
class CornerStepConfig:
    check_max_m: float = 0.25       # quick check of the covering epoch
    resolve_max_m: float = 0.15     # within-match validation gate of a re-solve
    check_frames: int = 6
    min_tags: int = 3               # tags per check frame
    fallback_min_tags: int = 2      # used only when NO check frame shows min_tags
    median_frames: int = 31
    max_center_shift_m: float = 1.0
    jev_verify: bool = True
    detect_motion: bool = True      # drop a corner re-aimed during the match


@dataclass
class CornerOutcome:
    status: str                     # kept | adopted | dropped | failed_kept
    reason: str
    epoch_checked: Optional[str]
    calibration: object = None      # CameraCalibration to use (adopted), else None
    metrics: dict = field(default_factory=dict)
    auto_tags: dict = field(default_factory=dict)
    manual_tags: list = field(default_factory=list)
    overlay: Optional[np.ndarray] = None
    seconds: float = 0.0


def check_epoch(cal, video, crop, frame_range, fm, detect_fn, cfg: CornerStepConfig) -> dict:
    """tag_alignment_error of ``cal`` on ``cfg.check_frames`` frames evenly spaced in
    ``frame_range`` (lo, hi) of this match. The pass/fail value is the best error on
    frames with >= min_tags tags; only if no frame has that many, on frames with >=
    fallback_min_tags (reported as ``min_tags_used``)."""
    cap = cv2.VideoCapture(str(video))
    lo, hi = int(frame_range[0]), int(frame_range[1])
    n = max(1, cfg.check_frames)
    idxs = [int(lo + (hi - lo) * (i + 0.5) / n) for i in range(n)]
    x, y, w, h = (int(c) for c in crop)
    rows = []
    try:
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            e, k = tag_alignment_error(cal, detect_fn(frame[y:y + h, x:x + w]), fm)
            rows.append((float(e), int(k)))
    finally:
        cap.release()
    for need in (cfg.min_tags, cfg.fallback_min_tags):
        ok_rows = [r for r in rows if r[1] >= need and np.isfinite(r[0])]
        if ok_rows:
            best = min(ok_rows)
            return {"best_m": round(best[0], 3), "n_tags": best[1], "min_tags_used": need,
                    "frames": len(rows), "frames_used": len(ok_rows)}
    return {"best_m": None, "n_tags": 0, "min_tags_used": None, "frames": len(rows),
            "frames_used": 0}


def _overlay_on_median(video, crop, frame_range, fps, cal, fm, masks, detect_fn,
                       roi_detect_fn, n_frames):
    """Median of the match + overlay of ``cal`` with the median-frame decodes."""
    med = ac.median_frame(video, crop, frame_range[0] / fps, frame_range[1] / fps, n_frames)
    if med is None:
        return None
    dec = {t: c for t, c in ac.median_decodes(med, detect_fn, roi_detect_fn).items()
           if t in fm.tags}
    return ac.render_overlay(med, cal, fm, dec, sorted(dec), masks)


def corner_step(video, match_key: str, camera: str, crop, masks, fm, epoch: dict,
                frame_range, fps: float, cfg: Optional[CornerStepConfig] = None, *,
                judge: Optional[Callable] = None, rejected: bool = False,
                drop_on_fail: bool = True, detect_fn=None, roi_detect_fn=None,
                alt_epochs=(), log: Callable = print) -> CornerOutcome:
    """Decide what calibration a corner view uses for THIS match (see module doc).
    ``epoch``: the covering version dict (``calibration``, ``version_id``).
    ``frame_range``: (lo, hi) frames of the match's active play. ``rejected``: an
    operator rejected this match's earlier adoption -> never re-solve.
    ``alt_epochs``: further versions whose LENS to re-solve from when the covering
    epoch's lens fails (e.g. the last operator/auto-calibrated version — a chain
    of per-match adoptions can inherit one poorly-conditioned lens)."""
    cfg = cfg or CornerStepConfig()
    t0 = time.time()
    detect_fn = detect_fn or ac.make_detect_fn()
    if roi_detect_fn is None:
        from web.field.tag_locator import make_roi_detect_fn
        roi_detect_fn = make_roi_detect_fn()
    ecal, vid = epoch["calibration"], epoch["version_id"]
    # A camera re-aimed DURING the match cannot be served by any one calibration —
    # its tracking for this match is invalid. Drop it (never adopt a solve from the
    # mixed footage); the next match's job calibrates on its own, post-move footage.
    if cfg.detect_motion:
        mot = camera_motion.detect_camera_move(
            video, crop, (frame_range[0] / fps, frame_range[1] / fps), detect_fn)
        if mot["moved"]:
            overlay = _overlay_on_median(video, crop, frame_range, fps, ecal, fm, masks,
                                         detect_fn, roi_detect_fn, cfg.median_frames)
            return CornerOutcome(
                "dropped", f"{camera_motion.describe(mot)} — this camera's tracking is "
                           f"invalid for {match_key}; dropped for this match (the next "
                           f"match calibrates on its own footage)",
                vid, None, {"moved_mid_match": {k: mot[k] for k in
                                                ("t_s", "shift_px", "stepping", "two_sided")}},
                overlay=overlay, seconds=time.time() - t0)
    chk = check_epoch(ecal, video, crop, frame_range, fm, detect_fn, cfg)
    metrics = {"check_best_m": chk["best_m"], "check_tags": chk["n_tags"],
               "check_min_tags": chk["min_tags_used"], "check_frames": chk["frames"]}
    implausible = pose_plausibility(ecal, fm=fm)
    if implausible:
        metrics["check_implausible"] = implausible
    if chk["best_m"] is not None and chk["best_m"] <= cfg.check_max_m and not implausible:
        overlay = _overlay_on_median(video, crop, frame_range, fps, ecal, fm, masks,
                                     detect_fn, roi_detect_fn, cfg.median_frames)
        return CornerOutcome(
            "kept", f"epoch {vid} holds on {match_key}: best {chk['best_m']:.2f} m over "
                    f"{chk['n_tags']} tags (gate {cfg.check_max_m:.2f} m)",
            vid, None, metrics, overlay=overlay, seconds=time.time() - t0)
    why_check = (f"epoch {vid} is physically implausible ({implausible})" if implausible else
                 "no check frame showed enough tags" if chk["best_m"] is None else
                 f"epoch {vid} is {chk['best_m']:.2f} m off on {match_key} "
                 f"(gate {cfg.check_max_m:.2f} m)")

    def fail(reason, overlay=None, extra=None):
        status = "dropped" if drop_on_fail else "failed_kept"
        if overlay is None:
            overlay = _overlay_on_median(video, crop, frame_range, fps, ecal, fm, masks,
                                         detect_fn, roi_detect_fn, cfg.median_frames)
        return CornerOutcome(status, reason + ("" if drop_on_fail else
                                                " — kept anyway (corner_drop_on_fail off)"),
                             vid, None, {**metrics, **(extra or {})}, overlay=overlay,
                             seconds=time.time() - t0)

    if rejected:
        return fail(f"{why_check}; this match's per-match calibration was rejected by "
                    f"the operator — calibrate manually")
    acfg = ac.AutoCalibConfig(
        window=(frame_range[0] / fps, frame_range[1] / fps), drift_max_m=cfg.resolve_max_m,
        max_center_shift_m=cfg.max_center_shift_m, median_frames=cfg.median_frames,
        jev_verify=cfg.jev_verify, field_season=getattr(ecal, "field_season", None)
        or "2026-rebuilt-welded")
    res, base_used, errs = None, None, []
    seen_lens = []
    for base in [epoch, *alt_epochs]:
        bcal = base["calibration"]
        lens_key = (None if bcal.K is None else
                    tuple(np.round(np.asarray(bcal.K).ravel(), 1)) +
                    tuple(np.round(np.asarray(bcal.D).ravel(), 3)))
        if lens_key in seen_lens:
            continue                      # same lens: re-solving again changes nothing
        seen_lens.append(lens_key)
        try:
            r = ac.run_camera(video, match_key, camera, crop, fm, [base], cfg=acfg,
                              detect_fn=detect_fn, masks=masks, judge=judge,
                              roi_detect_fn=roi_detect_fn, force_resolve=True)
        except Exception as e:
            errs.append(f"{base['version_id']}: {type(e).__name__}: {e}")
            continue
        if res is None or (r.record["status"] == "ready" and res.record["status"] != "ready"):
            res, base_used = r, base
        if r.record["status"] == "ready":
            break
    if res is None:
        return fail(f"{why_check}; re-solve error: {'; '.join(errs)}")
    rec = res.record
    m = rec.get("metrics") or {}
    extra = {"n_tags": m.get("n_tags"), "rms_px": m.get("rms_px"),
             "val_best_m": m.get("val_gate_best_m"), "val_median_m": m.get("val_median_m"),
             "val_source": m.get("val_source"), "located_tags": m.get("located_tags") or [],
             "center_shift_m": m.get("center_shift_m"),
             "rotation_delta_deg": m.get("rotation_delta_deg")}
    if rec["status"] != "ready" or res.calibration is None:
        return fail(f"{why_check}; per-match re-solve did not validate: {rec['reason']}",
                    overlay=res.overlay, extra=extra)
    cal = res.calibration
    cal.masks = list(getattr(ecal, "masks", None) or [])
    cal.robot_track_top_crop = getattr(ecal, "robot_track_top_crop", 0.0)
    lens_note = ("" if base_used is epoch else
                 f" (lens of {base_used['version_id']}; the covering epoch's lens did not validate)")
    extra["lens_from"] = base_used["version_id"]
    return CornerOutcome(
        "adopted", f"{why_check}; re-solved on {match_key}{lens_note}: {rec['reason']}", vid, cal,
        {**metrics, **extra}, auto_tags=rec.get("auto_tags") or {},
        manual_tags=rec.get("manual_tags") or [], overlay=res.overlay,
        seconds=time.time() - t0)

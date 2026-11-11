"""Calibrate a static camera by AGGREGATING AprilTag detections across a whole match.

Why this exists
---------------
The broadcast camera is static *during* play but is framed differently before auto
(intros / pan-zoom), so a single pre-auto frame calibrates to the wrong framing and
fails the in-match drift check. Mid-match frames have the right framing but tags get
occluded by robots/people/balls. This tool resolves both: it samples many in-match
frames and robustly medians each tag's corner pixels — a tag only needs to be
unobscured in SOME frames, never all at once — then solves one calibration in the
true match framing.

Robustness: a minority of off-framing frames (e.g. the pre-match shot) is rejected
by a MAD filter around each tag's median before the final solve, so a generous time
window is safe.

Usage
-----
    .venv/bin/python scripts/calibrate_from_match.py \
        --video web/uploads/<id>.mp4 --event 2026cur --match-key 2026cur_qm1
    # add --save to persist a new calibration version (default is dry-run)
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from web.field import store as field_store
from web.field.apriltag_detect import detect_tags
from web.field.calibration import Calibrator, tag_alignment_error
from web.field.composite_layout import load_layout
from web.field.field_model import FieldModel
from web.field.pose_autofix import aggregate_tag_corners  # shared robust aggregation

DRIFT_MAX_M = 0.5  # mirrors CALIB_DRIFT_MAX_M default in web/app.py


def validate(cal, video, crop, fm, n_frames, samples=12):
    cap = cv2.VideoCapture(video)
    x, y, w, h = crop
    errs = []
    for i in range(samples):
        fidx = int(n_frames * i / max(1, samples - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        e, n = tag_alignment_error(cal, detect_tags(frame[y:y + h, x:x + w]), fm)
        if n >= 2:
            errs.append(e)
    cap.release()
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--event", required=True)
    ap.add_argument("--camera", default="main")
    ap.add_argument("--match-key", default=None,
                    help="e.g. 2026cur_qm1; saved as start_match so it wins epoch selection")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T_MIN", "T_MAX"), default=[18.0, 1e9],
                    help="time window (s) to sample; default skips the pre-auto shot")
    ap.add_argument("--stride-sec", type=float, default=1.0)
    ap.add_argument("--mad-px", type=float, default=5.0, help="reject frames > this from a tag's median")
    ap.add_argument("--min-tags", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="persist a calibration version (default: dry-run)")
    args = ap.parse_args()

    layout = load_layout(event_key=args.event)
    if args.camera not in layout:
        raise SystemExit(f"unknown camera '{args.camera}'; known: {list(layout)}")
    crop = layout[args.camera].crop

    fm = FieldModel.load("2026-rebuilt-welded")
    corners, diag, fps, n_frames = aggregate_tag_corners(
        args.video, crop, args.window[0], args.window[1], args.stride_sec, args.mad_px)

    print(f"camera '{args.camera}' crop {crop}, {n_frames} frames @ {fps:.1f}fps")
    print("tag : inlier_frames/seen : corner_std_px")
    for tid in sorted(diag):
        ins, seen, std = diag[tid]
        print(f"  {tid:3d} : {ins:4d}/{seen:<4d} : {std:.2f}")

    corr = {tid: (c, fm.tag_corners_3d(tid)) for tid, c in corners.items() if tid in fm.tags}
    if len(corr) < args.min_tags:
        raise SystemExit(f"only {len(corr)} known tags aggregated (< {args.min_tags}); widen --window")

    cal = Calibrator.solve_camera_from_tags(corr, (crop[2], crop[3]), "2026-rebuilt-welded")
    errs = validate(cal, args.video, crop, fm, n_frames)
    best, worst = (min(errs), max(errs)) if errs else (float("inf"), float("inf"))
    print(f"\nsolved from {len(corr)} tags, rms {cal.rms_error:.2f}px")
    print(f"validation across match: best={best:.3f}m worst={worst:.3f}m (drift gate {DRIFT_MAX_M})")
    ok = best < DRIFT_MAX_M
    print("RESULT:", "PASS — will clear the drift check" if ok else "FAIL — widen window / check footage")

    if args.save:
        if not ok:
            raise SystemExit("refusing to save a calibration that fails validation")
        prev = field_store.load(args.event, args.camera)
        if prev is not None:
            cal.masks = prev.masks
            cal.robot_track_top_crop = prev.robot_track_top_crop
        source = {"auto_tags": {str(t): c.tolist() for t, c in corners.items()},
                  "match_key": args.match_key, "method": "aggregate_from_match",
                  "window": args.window, "stride_sec": args.stride_sec,
                  "crop": [int(v) for v in crop]}   # crop-local calibration: record its origin
        vid = field_store.save_version(args.event, args.camera, cal,
                                       start_match=args.match_key,
                                       label="aggregated in-match tags", source=source)
        print(f"saved version_id: {vid} (start_match={args.match_key})")
    else:
        print("\n(dry-run; re-run with --save to persist)")


if __name__ == "__main__":
    main()

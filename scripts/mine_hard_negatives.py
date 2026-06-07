#!/usr/bin/env python3
"""Mine hard negatives from footage by running the bumper detector + field-size
filters and dumping every rejected detection. Mirrors VideoProcessor._detect_view
for an event's calibrated views, without the websocket / event-session machinery.

Usage:
    .venv/bin/python scripts/mine_hard_negatives.py \\
        --event 2026hop --video data/test_footage/2026hop_match.mp4 \\
        --start 0 --end 12000 --step 30

Output: data/hard_negatives/<event>/ — raw frame, manifest JSON, and an annotated
*_preview.jpg (accepted=green, rejected=red) per kept frame.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from web.app import BUMPER_MODEL_PATH, HARD_NEG_DIR, DET_CONF_THRESHOLD
from web.field.camera_view import load_views
from web.field.detect_bridge import to_field_detections
from web.field.detect_schema import DetectionSchema
from web.field.field_model import FieldModel
from web.field.hard_negatives import HardNegativeLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=12000)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--max-files", type=int, default=2000)
    ap.add_argument("--capture-accepted", action="store_true",
                    help="Review mode: also save frames whose ACCEPTED detections need "
                         "human FP triage (hub face, field bumps that pass the filters).")
    ap.add_argument("--views", default="",
                    help="Comma-separated view names to restrict to (e.g. 'main').")
    args = ap.parse_args()

    views = load_views(args.event)
    if args.views:
        want = {v.strip() for v in args.views.split(",")}
        views = [v for v in views if v.name in want]
    if not views:
        sys.exit(f"No calibrated views for event '{args.event}' — calibrate first.")
    model = YOLO(str(BUMPER_MODEL_PATH))
    schema = DetectionSchema.from_names(model.names)
    main = next(v for v in views if v.name == "main")
    fm = FieldModel.load(main.calibration.field_season)
    dims = (fm.length_m, fm.width_m)

    out_dir = HARD_NEG_DIR / args.event
    logger = HardNegativeLogger(out_dir, enabled=True, log_every=1,
                                max_files=args.max_files)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = min(args.end, total) if total else args.end
    print(f"event={args.event} views={[v.name for v in views]} field_dims={dims} "
          f"conf={DET_CONF_THRESHOLD} frames={total} window=[{args.start},{end}) step={args.step}")

    kept = 0
    n_acc = 0
    n_rej = {"off_field": 0, "too_large": 0}
    for fi in range(args.start, end, args.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        for view in views:
            vf = view.extract(frame)
            res = model(vf, conf=DET_CONF_THRESHOLD, verbose=False)[0]
            robots, _balls, _hubs = schema.split(sv.Detections.from_ultralytics(res))
            boxes, colors = [], []
            for box, color, _conf in robots:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                if any(mx <= cx <= mx + mw and my <= cy <= my + mh
                       for (mx, my, mw, mh) in view.masks):
                    continue
                boxes.append(np.asarray(box))
                colors.append(color)
            rejected = []
            dets = to_field_detections(boxes, view.calibration, dims, colors=colors,
                                       frame_shape=vf.shape, rejected_out=rejected)
            n_acc += len(dets)
            for r in rejected:
                n_rej[r["reason"]] = n_rej.get(r["reason"], 0) + 1
            if rejected or args.capture_accepted:
                if logger.maybe_log(vf, [d.box for d in dets], rejected,
                                    view_name=view.name, frame_num=fi,
                                    require_rejected=not args.capture_accepted):
                    kept += 1
        if logger._written >= args.max_files:
            print(f"hit max-files cap ({args.max_files}) at frame {fi}")
            break

    cap.release()
    print(f"\naccepted detections: {n_acc}")
    print(f"rejected detections: {n_rej}")
    print(f"kept {kept} hard-negative frames -> {out_dir}")


if __name__ == "__main__":
    main()

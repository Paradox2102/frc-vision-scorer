#!/usr/bin/env python3
"""A/B a bumper model's false positives on known hub/outpost regions. Runs one or
more models over a match and counts detections that land on the annotated structure
boxes (IoU >= thresh) — i.e. the model calling a hub/outpost a robot. The delta
between old and new models is the verdict on the hard-negative retrain.

Usage:
  .venv/bin/python scripts/eval_hub_fp.py --video <mp4> --boxes <boxes.json> \\
    --model old=runs/train/frc_bumpers/weights/best.pt \\
    --model new=runs/train/frc_bumpers_hardneg/weights/best.pt --step 30
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import supervision as sv
from ultralytics import YOLO

from web.app import DET_CONF_THRESHOLD


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True, help="hub/outpost regions json")
    ap.add_argument("--model", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()

    regions = json.loads(Path(args.boxes).read_text())
    region_boxes = [((b["x1"], b["y1"], b["x2"], b["y2"]), b["label"]) for b in regions]
    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[name] = YOLO(path)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = args.end or total
    frames = list(range(args.start, end, args.step))

    results = {n: {"det": 0, "struct_fp": 0, "by_label": defaultdict(int), "frames_hit": 0}
               for n in models}
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        for name, model in models.items():
            res = model(frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            hit = False
            for i in range(len(dets)):
                box = tuple(float(v) for v in dets.xyxy[i])
                results[name]["det"] += 1
                best = max(((_iou(box, rb), lbl) for rb, lbl in region_boxes), default=(0, None))
                if best[0] >= args.iou:
                    results[name]["struct_fp"] += 1
                    results[name]["by_label"][best[1]] += 1
                    hit = True
            if hit:
                results[name]["frames_hit"] += 1
    cap.release()

    nf = len(frames)
    print(f"video={Path(args.video).name}  frames sampled={nf}  IoU>={args.iou}")
    print(f"hub/outpost regions: {len(region_boxes)}")
    for name, r in results.items():
        print(f"\n[{name}]")
        print(f"  total detections:        {r['det']}")
        print(f"  detections ON structures: {r['struct_fp']}  (by region: {dict(r['by_label'])})")
        print(f"  frames with a structure FP: {r['frames_hit']}/{nf}")


if __name__ == "__main__":
    main()

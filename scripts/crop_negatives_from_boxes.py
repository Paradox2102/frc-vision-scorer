#!/usr/bin/env python3
"""Crop human-drawn boxes (from tools/box_annotator.html) out of a video across many
frames, as hard-negative background images. Operates on the RAW full frame — no
calibration needed, so it works for ANY event's footage, not just calibrated ones.

The boxes are drawn once on a reference frame; the structures are static, so the same
boxes are cropped from frames spread across the match for lighting/ball/occlusion
variety. A contact sheet per label lets you prune any frame with a robot in front.

Usage:
    .venv/bin/python scripts/crop_negatives_from_boxes.py \\
        --video web/uploads/<uuid>.mp4 --boxes boxes.json \\
        --tag 2026hop_qm30 --count 30
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from web.app import HARD_NEG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True, help="boxes.json from the annotator")
    ap.add_argument("--tag", required=True, help="output subdir (e.g. event_match)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = whole video")
    ap.add_argument("--count", type=int, default=30, help="crops per box (evenly spaced)")
    args = ap.parse_args()

    boxes = json.loads(Path(args.boxes).read_text())
    if not boxes:
        sys.exit("no boxes in JSON")
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = args.end or total
    frames = [int(args.start + i * (end - args.start) / args.count) for i in range(args.count)]

    base = HARD_NEG_DIR / "boxes" / args.tag
    # de-dup labels so two boxes with the same label don't collide
    seen = defaultdict(int)
    for b in boxes:
        seen[b["label"]] += 1
        b["_name"] = b["label"] if seen[b["label"]] == 1 else f"{b['label']}{seen[b['label']]}"

    saved = defaultdict(list)
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        for b in boxes:
            x1, y1 = max(0, int(b["x1"])), max(0, int(b["y1"]))
            x2, y2 = min(W, int(b["x2"])), min(H, int(b["y2"]))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            d = base / b["_name"]
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"frame{fi:08d}.jpg"), crop)
            saved[b["_name"]].append(d / f"frame{fi:08d}.jpg")

    for name, paths in saved.items():
        tiles = []
        for p in paths:
            tiles.append(cv2.resize(cv2.imread(str(p)), (160, 120)))
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(base / name / "_contact.jpg"), np.vstack(rows))
        print(f"{name}: {len(paths)} crops -> {base / name}  (review _contact.jpg)")
    cap.release()
    print(f"\nReview each _contact.jpg, delete any crop with a robot in front, then build the dataset.")


if __name__ == "__main__":
    main()

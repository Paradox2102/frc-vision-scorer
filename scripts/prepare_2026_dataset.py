"""Prepare dataset_v2 from 2026 practice footage.

Uses both:
  1. Baseline YOLO v1 model for ball pre-labels
  2. HSV color detection as a secondary signal
  3. NMS merge to combine both sources

Classes:
  0: fuel_ball

Output:
  data/dataset_v2/images/{train,val}/
  data/dataset_v2/labels/{train,val}/
  data/dataset_v2/data.yaml
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIRS = [
    PROJECT_ROOT / "data" / "frames" / "2026_kettering",
    PROJECT_ROOT / "data" / "frames" / "2026_team_practice",
    PROJECT_ROOT / "data" / "frames" / "2026_practice_field",
    PROJECT_ROOT / "data" / "frames" / "2026_frc3255",
    PROJECT_ROOT / "data" / "frames" / "2026_ibots",
    # Also include 2017 data for diversity
    PROJECT_ROOT / "data" / "frames" / "2017_final",
    PROJECT_ROOT / "data" / "frames" / "2017_58ball",
]
MODEL_PATH = PROJECT_ROOT / "runs" / "train" / "frc_v1" / "weights" / "best.pt"
OUT_DIR = PROJECT_ROOT / "data" / "dataset_v2"

# HSV range for yellow balls (tuned from experiments)
HSV_LOWER = np.array([15, 80, 80])
HSV_UPPER = np.array([38, 255, 255])
MIN_BALL_AREA = 50       # px^2 minimum for HSV blobs
MAX_BALL_AREA = 8000     # px^2 maximum
MIN_CIRCULARITY = 0.3    # filter elongated false positives
YOLO_CONF = 0.15         # low threshold to catch more, we'll fix in annotation
NMS_IOU_THRESH = 0.4     # merge overlapping boxes from both sources

TRAIN_SPLIT = 0.85
SEED = 42


def hsv_detect_balls(frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Detect yellow blobs via HSV and return (xc, yc, w, h, conf) in normalized coords."""
    h_img, w_img = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
            continue

        # Circularity check
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # Aspect ratio filter (balls should be roughly square)
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect > 2.5:
            continue

        # Add some padding
        pad = int(max(w, h) * 0.15)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2 * pad)
        h = min(h_img - y, h + 2 * pad)

        xc = (x + w / 2.0) / w_img
        yc = (y + h / 2.0) / h_img
        wn = w / w_img
        hn = h / h_img
        detections.append((xc, yc, wn, hn, 0.5))  # conf=0.5 for HSV

    return detections


def yolo_detect_balls(model: YOLO, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Run YOLO inference and return (xc, yc, w, h, conf) in normalized coords."""
    h_img, w_img = frame.shape[:2]
    results = model.predict(frame, conf=YOLO_CONF, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # fuel_ball only
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            xc = ((x1 + x2) / 2.0) / w_img
            yc = ((y1 + y2) / 2.0) / h_img
            wn = (x2 - x1) / w_img
            hn = (y2 - y1) / h_img
            detections.append((xc, yc, wn, hn, conf))

    return detections


def nms_merge(
    dets: list[tuple[float, float, float, float, float]],
    iou_thresh: float = NMS_IOU_THRESH,
) -> list[tuple[float, float, float, float]]:
    """Simple NMS on normalized (xc, yc, w, h, conf) -> list of (xc, yc, w, h)."""
    if not dets:
        return []

    # Convert to xyxy
    boxes = []
    confs = []
    for xc, yc, w, h, conf in dets:
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        boxes.append([x1, y1, x2, y2])
        confs.append(conf)

    boxes = np.array(boxes)
    confs = np.array(confs)

    # Sort by confidence descending
    order = confs.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    results = []
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        # Clamp
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.001, min(1.0, w))
        h = max(0.001, min(1.0, h))
        results.append((xc, yc, w, h))

    return results


def main():
    # Collect all frames
    all_frames: list[Path] = []
    for d in FRAMES_DIRS:
        if d.exists():
            frames = sorted(d.glob("*.jpg"))
            all_frames.extend(frames)
            print(f"  {d.name}: {len(frames)} frames")

    print(f"\nTotal frames: {len(all_frames)}")

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Process all frames
    labels: dict[Path, list[tuple[float, float, float, float]]] = {}
    stats = {"yolo_only": 0, "hsv_only": 0, "both": 0, "total_boxes": 0}

    for i, fpath in enumerate(all_frames):
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        yolo_dets = yolo_detect_balls(model, frame)
        hsv_dets = hsv_detect_balls(frame)
        merged = nms_merge(yolo_dets + hsv_dets)

        labels[fpath] = merged
        stats["total_boxes"] += len(merged)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_frames)} frames, {stats['total_boxes']} total boxes")

    print(f"\nTotal: {len(labels)} frames, {stats['total_boxes']} boxes")

    # Split into train/val
    frame_list = list(labels.keys())
    random.Random(SEED).shuffle(frame_list)
    split_idx = int(len(frame_list) * TRAIN_SPLIT)
    train_frames = frame_list[:split_idx]
    val_frames = frame_list[split_idx:]

    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    # Write dataset
    for split, frames in [("train", train_frames), ("val", val_frames)]:
        img_dir = OUT_DIR / "images" / split
        lbl_dir = OUT_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for fpath in frames:
            dst_img = img_dir / fpath.name
            shutil.copy2(fpath, dst_img)

            lbl_path = lbl_dir / f"{fpath.stem}.txt"
            with lbl_path.open("w") as f:
                for xc, yc, w, h in labels[fpath]:
                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Write data.yaml
    data_yaml = {
        "path": str(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["fuel_ball"],
    }
    yaml_path = OUT_DIR / "data.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nDataset written to {OUT_DIR}")
    print(f"  data.yaml: {yaml_path}")

    # Summary
    total_labels = sum(len(v) for v in labels.values())
    frames_with_labels = sum(1 for v in labels.values() if len(v) > 0)
    print(f"\n  Frames with labels: {frames_with_labels}/{len(labels)}")
    print(f"  Total annotations: {total_labels}")
    print(f"  Avg per labeled frame: {total_labels / max(1, frames_with_labels):.1f}")


if __name__ == "__main__":
    main()

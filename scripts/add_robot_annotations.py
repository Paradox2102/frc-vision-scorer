#!/usr/bin/env python3
"""Add robot (red/blue bumper) annotations to existing dataset v4 labels.

The auto-annotator only found fuel balls. This script adds robot_red and
robot_blue bounding boxes using HSV bumper color detection on overhead frames.

Only processes overhead frames — alliance views are too noisy for HSV detection.

Classes (matching dataset.yaml):
- 0: fuel_ball  (already annotated)
- 1: robot_red
- 2: robot_blue
- 3: hub (skipped — fixed structure, handle separately)
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_v4"

# HSV ranges from robot_tracker.py — tuned for FRC broadcast
RED_LOWER_1 = np.array([0, 150, 100])
RED_UPPER_1 = np.array([8, 255, 255])
RED_LOWER_2 = np.array([170, 150, 100])
RED_UPPER_2 = np.array([180, 255, 255])
BLUE_LOWER = np.array([100, 150, 80])
BLUE_UPPER = np.array([125, 255, 255])

# Contour area filters (overhead view at 1280x468)
MIN_BUMPER_AREA = 400   # ~20x20 min
MAX_BUMPER_AREA = 5000  # filter large false positives

# Padding around bumper contour to get full robot bbox (fraction of image)
BBOX_PAD_X = 0.015  # ~19px at 1280w
BBOX_PAD_Y = 0.04   # ~19px at 468h

# Merge nearby detections (pixels)
MERGE_DISTANCE = 50

# Max robots per alliance
MAX_PER_ALLIANCE = 3

CLASS_ROBOT_RED = 1
CLASS_ROBOT_BLUE = 2


def detect_bumpers(image: np.ndarray, alliance: str) -> list[tuple]:
    """Detect bumper-colored blobs and return bounding boxes.
    
    Returns list of (area, x_center, y_center, width, height) in normalized coords.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if alliance == "red":
        mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        mask = mask1 | mask2
    else:
        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BUMPER_AREA or area > MAX_BUMPER_AREA:
            continue
        
        bx, by, bw, bh = cv2.boundingRect(cnt)
        
        # Pad to cover full robot body (bumper is just the edge)
        pad_x = int(BBOX_PAD_X * w)
        pad_y = int(BBOX_PAD_Y * h)
        bx = max(0, bx - pad_x)
        by = max(0, by - pad_y)
        bw = min(w - bx, bw + 2 * pad_x)
        bh = min(h - by, bh + 2 * pad_y)
        
        # Normalize
        cx = (bx + bw / 2) / w
        cy = (by + bh / 2) / h
        nw = bw / w
        nh = bh / h
        
        detections.append((area, cx, cy, nw, nh))
    
    return detections


def merge_nearby(detections: list[tuple], img_w: int, img_h: int) -> list[tuple]:
    """Merge detections within MERGE_DISTANCE pixels of each other."""
    if len(detections) <= 1:
        return detections
    
    merged = []
    used = set()
    
    for i, (a1, cx1, cy1, w1, h1) in enumerate(detections):
        if i in used:
            continue
        
        group = [(a1, cx1, cy1, w1, h1)]
        used.add(i)
        
        for j, (a2, cx2, cy2, w2, h2) in enumerate(detections):
            if j in used:
                continue
            
            dx = (cx1 - cx2) * img_w
            dy = (cy1 - cy2) * img_h
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < MERGE_DISTANCE:
                group.append((a2, cx2, cy2, w2, h2))
                used.add(j)
        
        # Use largest detection as representative, expand bbox to cover group
        group.sort(key=lambda x: x[0], reverse=True)
        best_area = group[0][0]
        
        # Compute bounding box of all detections in group
        min_x = min(cx - w/2 for _, cx, cy, w, h in group)
        max_x = max(cx + w/2 for _, cx, cy, w, h in group)
        min_y = min(cy - h/2 for _, cx, cy, w, h in group)
        max_y = max(cy + h/2 for _, cx, cy, w, h in group)
        
        merged_cx = (min_x + max_x) / 2
        merged_cy = (min_y + max_y) / 2
        merged_w = max_x - min_x
        merged_h = max_y - min_y
        
        merged.append((best_area, merged_cx, merged_cy, merged_w, merged_h))
    
    return merged


def top_n(detections: list[tuple], n: int) -> list[tuple]:
    """Keep only the top-n detections by area."""
    detections.sort(key=lambda x: x[0], reverse=True)
    return detections[:n]


def process_frame(image_path: Path) -> list[str]:
    """Detect robots in an overhead frame. Returns YOLO annotation lines."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    lines = []
    
    for alliance, class_id in [("red", CLASS_ROBOT_RED), ("blue", CLASS_ROBOT_BLUE)]:
        dets = detect_bumpers(img, alliance)
        dets = merge_nearby(dets, w, h)
        dets = top_n(dets, MAX_PER_ALLIANCE)
        
        for _, cx, cy, bw, bh in dets:
            # Clamp to [0, 1]
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            bw = max(0.005, min(0.999, bw))
            bh = max(0.005, min(0.999, bh))
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    
    return lines


def main():
    print("=" * 60)
    print("Adding robot annotations to dataset v4")
    print("=" * 60)
    
    total_red = 0
    total_blue = 0
    frames_processed = 0
    
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        
        # Only process overhead frames
        overhead_imgs = sorted(img_dir.glob("overhead_*.jpg"))
        print(f"\n{split}: {len(overhead_imgs)} overhead frames")
        
        for img_path in tqdm(overhead_imgs, desc=f"  {split}"):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            
            robot_lines = process_frame(img_path)
            
            if robot_lines:
                # Read existing fuel ball annotations
                existing = ""
                if label_path.exists():
                    existing = label_path.read_text()
                
                # Append robot annotations
                with open(label_path, 'w') as f:
                    if existing.strip():
                        f.write(existing.rstrip('\n') + '\n')
                    f.write('\n'.join(robot_lines) + '\n')
                
                red_count = sum(1 for l in robot_lines if l.startswith(f"{CLASS_ROBOT_RED} "))
                blue_count = sum(1 for l in robot_lines if l.startswith(f"{CLASS_ROBOT_BLUE} "))
                total_red += red_count
                total_blue += blue_count
            
            frames_processed += 1
    
    print(f"\nDone!")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Red robot annotations: {total_red}")
    print(f"  Blue robot annotations: {total_blue}")
    print(f"  Total robot annotations: {total_red + total_blue}")
    
    # Verify final distribution
    print(f"\nFinal class distribution:")
    for split in ["train", "val"]:
        lbl_dir = DATASET_DIR / "labels" / split
        counts = {}
        for lbl in lbl_dir.glob("*.txt"):
            for line in lbl.read_text().strip().split('\n'):
                if line.strip():
                    cls = int(line.split()[0])
                    counts[cls] = counts.get(cls, 0) + 1
        print(f"  {split}: {counts}")


if __name__ == "__main__":
    main()

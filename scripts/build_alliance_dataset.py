#!/usr/bin/env python3
"""Build YOLO dataset from Label Studio JSON export of alliance crops.

Handles blue/red filename collision by prefixing alliance.
Creates train/val split (80/20).
"""

import json
import shutil
import random
from pathlib import Path

PROJECT = Path(__file__).parent.parent
EXPORT = PROJECT / "data" / "alliance_dataset" / "export.json"
CROP_DIR = PROJECT / "data" / "alliance_crops"
DATASET = PROJECT / "data" / "alliance_yolo_dataset"

# Class mapping (match what we want for training)
CLASS_MAP = {"fuel_ball": 0, "robot": 1, "hub_opening": 2}
VAL_RATIO = 0.2

random.seed(42)


def main():
    # Clean and create dirs
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ["train", "val"]:
        (DATASET / split / "images").mkdir(parents=True)
        (DATASET / split / "labels").mkdir(parents=True)

    with open(EXPORT) as f:
        tasks = json.load(f)

    # Parse all tasks
    items = []
    for task in tasks:
        alliance = task["data"]["alliance"]
        # Extract filename from the image path
        img_path_str = task["data"]["image"]
        # /data/local-files/?d=data/labeling/alliance_crops/blue/week0_f1m1_frame_000286.jpg
        fname = img_path_str.split("/")[-1]
        src_img = CROP_DIR / alliance / fname
        
        if not src_img.exists():
            print(f"SKIP: {src_img} not found")
            continue

        # Build YOLO labels from annotations
        labels = []
        for ann in task.get("annotations", []):
            for r in ann["result"]:
                if r["type"] != "rectanglelabels":
                    continue
                cls_name = r["value"]["rectanglelabels"][0]
                cls_id = CLASS_MAP.get(cls_name)
                if cls_id is None:
                    continue

                # LS gives percentages, convert to YOLO (center_x, center_y, w, h) normalized
                x_pct = r["value"]["x"] / 100.0
                y_pct = r["value"]["y"] / 100.0
                w_pct = r["value"]["width"] / 100.0
                h_pct = r["value"]["height"] / 100.0

                cx = x_pct + w_pct / 2
                cy = y_pct + h_pct / 2

                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w_pct:.6f} {h_pct:.6f}")

        # Prefix with alliance to avoid collision
        out_name = f"{alliance}_{fname}"
        items.append((src_img, out_name, labels))

    # Shuffle and split
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_RATIO))
    val_items = items[:n_val]
    train_items = items[n_val:]

    for split, split_items in [("train", train_items), ("val", val_items)]:
        for src_img, out_name, labels in split_items:
            # Copy image
            shutil.copy2(src_img, DATASET / split / "images" / out_name)
            # Write labels
            label_name = out_name.replace(".jpg", ".txt")
            with open(DATASET / split / "labels" / label_name, "w") as f:
                f.write("\n".join(labels) + "\n" if labels else "")

    print(f"Train: {len(train_items)} images")
    print(f"Val: {len(val_items)} images")
    print(f"Total labels: {sum(len(it[2]) for it in items)}")

    # Write dataset YAML
    yaml_content = f"""path: {DATASET}
train: train/images
val: val/images

names:
  0: fuel_ball
  1: robot
  2: hub_opening
"""
    yaml_path = DATASET / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nDataset YAML: {yaml_path}")

    # Stats
    from collections import Counter
    counts = Counter()
    for _, _, labels in items:
        for l in labels:
            cls_id = int(l.split()[0])
            counts[cls_id] += 1
    for cls_id, name in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
        print(f"  {name}: {counts[CLASS_MAP[name]]}")


if __name__ == "__main__":
    main()

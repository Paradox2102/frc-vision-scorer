#!/usr/bin/env python3
"""Merge Roboflow FRC robot dataset with existing hub-zone training data.

The Roboflow dataset (2006-sk-abtcw/frc-1a4zk) contains FRC robot detections
with classes: FRCrobot=0, object=1. We remap FRCrobot→1 (our robot class)
and drop 'object' (noise class with only 5 annotations).

Our schema: fuel_ball=0, robot=1, hub_opening=2

The merged dataset combines:
- Our alliance_yolo_dataset (fuel_ball, robot, hub_opening labels)
- Roboflow FRC images (robot-only labels, remapped)
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLIANCE_DATASET = PROJECT_ROOT / "data" / "alliance_yolo_dataset"
ROBOFLOW_DATASET = PROJECT_ROOT / "data" / "roboflow_frc"
MERGED_DATASET = PROJECT_ROOT / "data" / "merged_robot_dataset"

# Roboflow class 0 (FRCrobot) → our class 1 (robot)
# Roboflow class 1 (object) → dropped
ROBOFLOW_CLASS_MAP = {0: 1}


def remap_labels(src_label: Path, dst_label: Path):
    """Remap Roboflow label file classes to our schema, dropping unmapped classes."""
    lines = src_label.read_text().strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        if cls_id in ROBOFLOW_CLASS_MAP:
            parts[0] = str(ROBOFLOW_CLASS_MAP[cls_id])
            remapped.append(" ".join(parts))
    if remapped:
        dst_label.write_text("\n".join(remapped) + "\n")
        return True
    return False


def copy_split(split_name: str, src_split: str):
    """Copy and remap one split (train/val/test) from both datasets."""
    merged_img_dir = MERGED_DATASET / split_name / "images"
    merged_lbl_dir = MERGED_DATASET / split_name / "labels"
    merged_img_dir.mkdir(parents=True, exist_ok=True)
    merged_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied_alliance = 0
    copied_roboflow = 0
    skipped = 0

    # 1) Copy alliance dataset (labels already in our schema)
    alliance_split = ALLIANCE_DATASET / split_name
    if alliance_split.exists():
        for img in (alliance_split / "images").glob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dst_img = merged_img_dir / f"alliance_{img.name}"
                shutil.copy2(img, dst_img)
                lbl = alliance_split / "labels" / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, merged_lbl_dir / f"alliance_{img.stem}.txt")
                copied_alliance += 1

    # 2) Copy Roboflow dataset (remap classes)
    rf_split = ROBOFLOW_DATASET / src_split
    if rf_split.exists():
        for img in (rf_split / "images").glob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lbl = rf_split / "labels" / f"{img.stem}.txt"
                if lbl.exists():
                    dst_lbl = merged_lbl_dir / f"rf_{img.stem}.txt"
                    if remap_labels(lbl, dst_lbl):
                        dst_img = merged_img_dir / f"rf_{img.name}"
                        shutil.copy2(img, dst_img)
                        copied_roboflow += 1
                    else:
                        skipped += 1

    print(f"  {split_name}: {copied_alliance} alliance + {copied_roboflow} roboflow "
          f"= {copied_alliance + copied_roboflow} total ({skipped} skipped, no robot labels)")


def write_dataset_yaml():
    """Write the combined dataset YAML config."""
    yaml_path = MERGED_DATASET / "dataset.yaml"
    yaml_path.write_text(
        f"# Merged FRC Robot Training Dataset\n"
        f"# Alliance hub-zone data + Roboflow FRC robot data\n"
        f"path: {MERGED_DATASET}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"\n"
        f"names:\n"
        f"  0: fuel_ball\n"
        f"  1: robot\n"
        f"  2: hub_opening\n"
        f"\n"
        f"nc: 3\n"
    )
    print(f"\nDataset YAML written to: {yaml_path}")


def main():
    if MERGED_DATASET.exists():
        print(f"Removing existing merged dataset at {MERGED_DATASET}")
        shutil.rmtree(MERGED_DATASET)

    print("Merging datasets...")
    print(f"  Alliance source: {ALLIANCE_DATASET}")
    print(f"  Roboflow source: {ROBOFLOW_DATASET}")
    print(f"  Output: {MERGED_DATASET}")
    print()

    # Alliance has train/val; Roboflow has train/valid/test
    copy_split("train", "train")
    copy_split("val", "valid")
    copy_split("test", "test")

    write_dataset_yaml()

    # Summary
    print("\nFinal counts:")
    for split in ["train", "val", "test"]:
        img_dir = MERGED_DATASET / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*")))
            print(f"  {split}: {count} images")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export corrected Label Studio annotations back to YOLO format.

Reads a Label Studio JSON export (from the web UI or API) and writes
YOLO-format .txt label files, ready for training.

Usage:
    python scripts/labelstudio_export_to_yolo.py \
        --input labeling/export.json \
        --output data/dataset_v3/labels \
        --image-dir data/dataset_v2/images \
        --copy-images data/dataset_v3/images \
        --split 0.8

Classes:
    0: fuel_ball
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from urllib.parse import unquote


# Class mapping
CLASS_MAP = {
    "fuel_ball": 0,
}


def parse_label_studio_export(export_path: str) -> list[dict]:
    """Parse Label Studio JSON export into a list of annotation records."""
    with open(export_path) as f:
        tasks = json.load(f)

    records = []
    for task in tasks:
        # Get image path from task data
        image_path = task.get("data", {}).get("image", "")
        
        # Handle local file paths: /data/local-files/?d=path/to/image.jpg
        if "/data/local-files/" in image_path:
            image_path = unquote(image_path.split("?d=")[-1])
        
        # Get the filename
        image_filename = os.path.basename(image_path)
        
        # Get annotations (completed by human) or predictions (auto-labels)
        # Prefer annotations (human-corrected) over predictions
        annotations = task.get("annotations", [])
        if annotations:
            # Use the most recent annotation
            results = annotations[-1].get("result", [])
        else:
            # Fall back to predictions if no human annotations
            predictions = task.get("predictions", [])
            if predictions:
                results = predictions[-1].get("result", [])
            else:
                results = []
        
        boxes = []
        for result in results:
            if result.get("type") != "rectanglelabels":
                continue
            
            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue
            
            label = labels[0]
            if label not in CLASS_MAP:
                print(f"  Warning: unknown label '{label}' in {image_filename}, skipping")
                continue
            
            class_id = CLASS_MAP[label]
            
            # Label Studio uses percentages (0-100)
            x_pct = value.get("x", 0)
            y_pct = value.get("y", 0)
            w_pct = value.get("width", 0)
            h_pct = value.get("height", 0)
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized 0-1
            cx = (x_pct + w_pct / 2) / 100.0
            cy = (y_pct + h_pct / 2) / 100.0
            w = w_pct / 100.0
            h = h_pct / 100.0
            
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            boxes.append((class_id, cx, cy, w, h))
        
        records.append({
            "image_path": image_path,
            "image_filename": image_filename,
            "boxes": boxes,
            "annotated": len(annotations) > 0,
        })
    
    return records


def write_yolo_labels(records: list[dict], output_dir: Path, 
                      image_source_dir: Path | None, 
                      image_output_dir: Path | None,
                      train_split: float = 0.8):
    """Write YOLO format labels and optionally copy/symlink images."""
    
    # Separate annotated records for train/val split
    annotated = [r for r in records if r["boxes"]]  # Has at least one box
    empty = [r for r in records if not r["boxes"]]   # Background images
    
    print(f"\nRecords with boxes: {len(annotated)}")
    print(f"Background (empty) records: {len(empty)}")
    
    # Shuffle and split
    random.seed(42)
    all_records = annotated + empty
    random.shuffle(all_records)
    
    split_idx = int(len(all_records) * train_split)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]
    
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")
    
    for split_name, split_records in [("train", train_records), ("val", val_records)]:
        label_dir = output_dir / split_name
        label_dir.mkdir(parents=True, exist_ok=True)
        
        if image_output_dir:
            img_dir = image_output_dir / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
        
        for record in split_records:
            fname = record["image_filename"]
            stem = Path(fname).stem
            
            # Write label file
            label_path = label_dir / f"{stem}.txt"
            with open(label_path, "w") as f:
                for cls_id, cx, cy, w, h in record["boxes"]:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
            # Copy image if requested
            if image_output_dir and image_source_dir:
                # Try to find the image in train/ or val/ subdirs
                src = None
                for subdir in ["train", "val", ""]:
                    candidate = image_source_dir / subdir / fname
                    if candidate.exists():
                        src = candidate
                        break
                
                if src:
                    dst = image_output_dir / split_name / fname
                    if not dst.exists():
                        shutil.copy2(src, dst)
                else:
                    # Try the full relative path from the record
                    full_path = Path(record["image_path"])
                    if full_path.exists():
                        dst = image_output_dir / split_name / fname
                        if not dst.exists():
                            shutil.copy2(full_path, dst)
                    else:
                        print(f"  Warning: image not found for {fname}")
    
    # Write data.yaml
    yaml_path = output_dir.parent / "data.yaml"
    yaml_content = f"""path: {output_dir.parent.resolve()}
train: images/train
val: images/val

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nWrote data.yaml to {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Label Studio export to YOLO format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to Label Studio JSON export file"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for YOLO labels (will create train/ and val/ subdirs)"
    )
    parser.add_argument(
        "--image-dir", 
        help="Source image directory (to copy images from)"
    )
    parser.add_argument(
        "--copy-images",
        help="Output directory for images (will create train/ and val/ subdirs)"
    )
    parser.add_argument(
        "--split", type=float, default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--annotated-only", action="store_true",
        help="Only export tasks that have human annotations (not just predictions)"
    )
    
    args = parser.parse_args()
    
    print(f"Reading Label Studio export from: {args.input}")
    records = parse_label_studio_export(args.input)
    print(f"Total tasks: {len(records)}")
    
    annotated_count = sum(1 for r in records if r["annotated"])
    print(f"Human-annotated: {annotated_count}")
    print(f"Auto-labels only: {len(records) - annotated_count}")
    
    total_boxes = sum(len(r["boxes"]) for r in records)
    print(f"Total bounding boxes: {total_boxes}")
    
    if args.annotated_only:
        records = [r for r in records if r["annotated"]]
        print(f"\nFiltered to annotated-only: {len(records)} tasks")
    
    output_dir = Path(args.output)
    image_source = Path(args.image_dir) if args.image_dir else None
    image_output = Path(args.copy_images) if args.copy_images else None
    
    write_yolo_labels(records, output_dir, image_source, image_output, args.split)
    
    print("\nDone! To train on the corrected data:")
    print(f"  python scripts/train_yolo.py --data {output_dir.parent / 'data.yaml'}")


if __name__ == "__main__":
    main()

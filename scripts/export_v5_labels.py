#!/usr/bin/env python3
"""Export Label Studio annotations to YOLO format for v5 dataset.

This script reads from the Label Studio SQLite database (project_id=2)
and exports annotations in YOLO format with the following class mapping:
- fuel_ball=0
- robot=1 (generic robot, no red/blue distinction)
- hub=2

Note: v4 had robot_red=1, robot_blue=2, hub=3. v5 detects generic robots
and determines alliance post-detection based on field position.
"""

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_STUDIO_DB = Path.home() / "Library" / "Application Support" / "label-studio" / "label_studio.sqlite3"
IMAGES_DIR = PROJECT_ROOT / "data" / "labeling" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset_v5"

# Class mapping for v5
CLASS_MAPPING = {
    "fuel_ball": 0,
    "robot": 1,      # generic robot (v4 had robot_red/robot_blue)
    "hub": 2
}

# Label Studio project ID for v5 robot detection
PROJECT_ID = 2

# Train/val split ratio
TRAIN_SPLIT = 0.8


def convert_bbox_to_yolo(bbox: Dict, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert Label Studio bbox (percentage) to YOLO format (normalized 0-1).
    
    Label Studio format: {x, y, width, height} in percentages (0-100)
    YOLO format: [cx, cy, w, h] normalized to 0-1
    
    Args:
        bbox: Label Studio bounding box {x, y, width, height}
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        (cx, cy, w, h) normalized to 0-1
    """
    # Convert percentage to normalized (0-1)
    x = bbox["x"] / 100.0
    y = bbox["y"] / 100.0
    w = bbox["width"] / 100.0
    h = bbox["height"] / 100.0
    
    # Convert to center coordinates
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    return cx, cy, w, h


def extract_annotations(db_path: Path, project_id: int) -> Dict[str, List[Dict]]:
    """Extract annotations from Label Studio database.
    
    Args:
        db_path: Path to Label Studio SQLite database
        project_id: Label Studio project ID
    
    Returns:
        Dictionary mapping image filename to list of annotations
        {
            "image.jpg": [
                {"class": "robot", "bbox": (cx, cy, w, h)},
                ...
            ]
        }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query tasks and their completions
    query = """
    SELECT t.data, tc.result
    FROM task t
    JOIN task_completion tc ON t.id = tc.task_id
    WHERE t.project_id = ? AND tc.was_cancelled = 0
    ORDER BY t.id
    """
    
    cursor.execute(query, (project_id,))
    rows = cursor.fetchall()
    
    annotations_by_image = {}
    
    for data_json, result_json in rows:
        # Parse JSON
        data = json.loads(data_json)
        result = json.loads(result_json)
        
        # Extract image filename from data
        # Format: {"image": "/data/local-files/?d=images/week0_f1m1_frame00090.jpg"}
        image_path = data.get("image", "")
        if "?d=images/" in image_path:
            filename = image_path.split("?d=images/")[1]
        else:
            # Try other formats
            filename = Path(image_path).name
        
        if not filename:
            print(f"Warning: Could not extract filename from {image_path}")
            continue
        
        # Parse annotations
        image_annotations = []
        for annotation in result:
            if annotation.get("type") != "rectanglelabels":
                continue
            
            # Get bbox and label
            value = annotation.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue
            
            # Take first label (should only be one per bbox)
            label = labels[0].lower()
            
            # Map label to class ID
            if label not in CLASS_MAPPING:
                print(f"Warning: Unknown label '{label}' in {filename}, skipping")
                continue
            
            class_id = CLASS_MAPPING[label]
            
            # Get image dimensions
            img_width = annotation.get("original_width", 1280)
            img_height = annotation.get("original_height", 480)
            
            # Convert bbox to YOLO format
            cx, cy, w, h = convert_bbox_to_yolo(value, img_width, img_height)
            
            image_annotations.append({
                "class_id": class_id,
                "class_name": label,
                "bbox": (cx, cy, w, h)
            })
        
        if image_annotations:
            annotations_by_image[filename] = image_annotations
    
    conn.close()
    
    return annotations_by_image


def create_dataset(annotations: Dict[str, List[Dict]], output_dir: Path, images_dir: Path, train_split: float = 0.8):
    """Create YOLO dataset with train/val split.
    
    Args:
        annotations: Dictionary mapping image filename to annotations
        output_dir: Output directory for dataset
        images_dir: Directory containing source images
        train_split: Ratio of images to use for training
    """
    # Create output directories
    train_images_dir = output_dir / "images" / "train"
    val_images_dir = output_dir / "images" / "val"
    train_labels_dir = output_dir / "labels" / "train"
    val_labels_dir = output_dir / "labels" / "val"
    
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Split images into train/val
    image_files = list(annotations.keys())
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total annotated images: {len(image_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Class statistics
    class_counts = {"train": {k: 0 for k in CLASS_MAPPING.values()}, 
                   "val": {k: 0 for k in CLASS_MAPPING.values()}}
    
    # Process train set
    for filename in train_files:
        # Copy image
        src_image = images_dir / filename
        dst_image = train_images_dir / filename
        
        if not src_image.exists():
            print(f"Warning: Image not found: {src_image}")
            continue
        
        shutil.copy(src_image, dst_image)
        
        # Write labels
        label_file = train_labels_dir / f"{Path(filename).stem}.txt"
        with open(label_file, 'w') as f:
            for ann in annotations[filename]:
                class_id = ann["class_id"]
                cx, cy, w, h = ann["bbox"]
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                class_counts["train"][class_id] += 1
    
    # Process val set
    for filename in val_files:
        # Copy image
        src_image = images_dir / filename
        dst_image = val_images_dir / filename
        
        if not src_image.exists():
            print(f"Warning: Image not found: {src_image}")
            continue
        
        shutil.copy(src_image, dst_image)
        
        # Write labels
        label_file = val_labels_dir / f"{Path(filename).stem}.txt"
        with open(label_file, 'w') as f:
            for ann in annotations[filename]:
                class_id = ann["class_id"]
                cx, cy, w, h = ann["bbox"]
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                class_counts["val"][class_id] += 1
    
    # Print statistics
    print("\nAnnotation statistics:")
    print("Train set:")
    for label, class_id in CLASS_MAPPING.items():
        print(f"  {label}: {class_counts['train'][class_id]}")
    print("Val set:")
    for label, class_id in CLASS_MAPPING.items():
        print(f"  {label}: {class_counts['val'][class_id]}")
    
    total_train = sum(class_counts["train"].values())
    total_val = sum(class_counts["val"].values())
    print(f"\nTotal annotations - Train: {total_train}, Val: {total_val}")


def create_dataset_yaml(output_dir: Path):
    """Create dataset.yaml for YOLO training.
    
    Args:
        output_dir: Output directory for dataset
    """
    yaml_content = f"""# FRC Vision Scorer v5 Dataset
# Class mapping: fuel_ball=0, robot=1, hub=2
# Note: v4 had robot_red=1, robot_blue=2, hub=3
# v5 detects generic robots; alliance assigned post-detection based on field position

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: 3  # number of classes
names:
  0: fuel_ball
  1: robot
  2: hub
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset.yaml at {yaml_path}")


def main():
    """Main export function."""
    print("=" * 80)
    print("FRC Vision Scorer v5 - Label Studio to YOLO Export")
    print("=" * 80)
    
    # Validate inputs
    if not LABEL_STUDIO_DB.exists():
        print(f"Error: Label Studio database not found at {LABEL_STUDIO_DB}")
        return
    
    if not IMAGES_DIR.exists():
        print(f"Error: Images directory not found at {IMAGES_DIR}")
        return
    
    print(f"\nLabel Studio DB: {LABEL_STUDIO_DB}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Train/val split: {TRAIN_SPLIT:.0%}/{1-TRAIN_SPLIT:.0%}")
    
    # Extract annotations
    print("\nExtracting annotations from Label Studio database...")
    annotations = extract_annotations(LABEL_STUDIO_DB, PROJECT_ID)
    
    if not annotations:
        print("Error: No annotations found!")
        return
    
    print(f"Found {len(annotations)} annotated images")
    
    # Create dataset
    print("\nCreating YOLO dataset...")
    create_dataset(annotations, OUTPUT_DIR, IMAGES_DIR, TRAIN_SPLIT)
    
    # Create dataset.yaml
    create_dataset_yaml(OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("Export complete!")
    print("=" * 80)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print(f"  - images/train: Training images")
    print(f"  - images/val: Validation images")
    print(f"  - labels/train: Training labels (YOLO format)")
    print(f"  - labels/val: Validation labels (YOLO format)")
    print(f"  - dataset.yaml: Dataset configuration")


if __name__ == "__main__":
    main()

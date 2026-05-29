#!/usr/bin/env python3
"""Reorganize dataset v4 into standard YOLO format.

YOLO expects:
  images/train/
  images/val/
  labels/train/
  labels/val/
"""

from pathlib import Path
import shutil
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "dataset_v4"


def main():
    print("Reorganizing dataset v4 for YOLO training...")
    
    # Read train and val splits
    train_txt = DATASET_ROOT / "train.txt"
    val_txt = DATASET_ROOT / "val.txt"
    
    with open(train_txt) as f:
        train_images = [line.strip() for line in f if line.strip()]
    
    with open(val_txt) as f:
        val_images = [line.strip() for line in f if line.strip()]
    
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    
    # Create new directory structure
    new_images_train = DATASET_ROOT / "images_yolo" / "train"
    new_images_val = DATASET_ROOT / "images_yolo" / "val"
    new_labels_train = DATASET_ROOT / "labels_yolo" / "train"
    new_labels_val = DATASET_ROOT / "labels_yolo" / "val"
    
    for d in [new_images_train, new_images_val, new_labels_train, new_labels_val]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy train images and labels
    print("\nCopying train set...")
    for img_rel_path in tqdm(train_images):
        img_path = DATASET_ROOT / img_rel_path
        label_path = DATASET_ROOT / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        
        if img_path.exists():
            # Copy image with unique name
            new_img_name = f"{img_path.parent.name}_{img_path.name}"
            shutil.copy2(img_path, new_images_train / new_img_name)
            
            # Copy label if exists
            if label_path.exists():
                shutil.copy2(label_path, new_labels_train / f"{img_path.parent.name}_{img_path.stem}.txt")
    
    # Copy val images and labels
    print("Copying val set...")
    for img_rel_path in tqdm(val_images):
        img_path = DATASET_ROOT / img_rel_path
        label_path = DATASET_ROOT / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        
        if img_path.exists():
            # Copy image with unique name
            new_img_name = f"{img_path.parent.name}_{img_path.name}"
            shutil.copy2(img_path, new_images_val / new_img_name)
            
            # Copy label if exists
            if label_path.exists():
                shutil.copy2(label_path, new_labels_val / f"{img_path.parent.name}_{img_path.stem}.txt")
    
    # Update dataset.yaml
    print("\nUpdating dataset.yaml...")
    yaml_content = f"""# FRC Vision Scorer - Dataset v4 (YOLO format)
# Auto-annotated from Week 0 Finals match videos

path: {DATASET_ROOT}
train: images_yolo/train
val: images_yolo/val

# Classes
names:
  0: fuel_ball
  1: robot_red
  2: robot_blue
  3: hub

nc: 4
"""
    
    with open(DATASET_ROOT / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print("\nReorganization complete!")
    print(f"  images_yolo/train: {len(list(new_images_train.glob('*.jpg')))} images")
    print(f"  images_yolo/val: {len(list(new_images_val.glob('*.jpg')))} images")
    print(f"  labels_yolo/train: {len(list(new_labels_train.glob('*.txt')))} labels")
    print(f"  labels_yolo/val: {len(list(new_labels_val.glob('*.txt')))} labels")


if __name__ == "__main__":
    main()

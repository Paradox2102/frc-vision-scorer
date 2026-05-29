"""Prepare dataset_v3 by merging v2 + new Label Studio human-corrected labels.

Priority: Label Studio human-corrected labels override v2 ML-generated labels.
Split: 85/15 train/val
"""

import shutil
from pathlib import Path
from collections import defaultdict
import random

# Set seed for reproducibility
random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LS_EXPORT = Path("/tmp/ls_yolo_export_dir")
V2_DIR = PROJECT_ROOT / "data" / "dataset_v2"
V3_DIR = PROJECT_ROOT / "data" / "dataset_v3"

def main():
    print("=" * 70)
    print("FRC Vision Scorer: Preparing v3 dataset")
    print("=" * 70)
    
    # Create v3 directory structure
    (V3_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (V3_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (V3_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (V3_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Collect all images from both sources
    image_sources = {}  # image_name -> (image_path, label_path, source)
    
    # 1. Load Label Studio export (human-corrected - highest priority)
    print("\n1. Loading Label Studio export (human-corrected labels)...")
    ls_images = list((LS_EXPORT / "images").glob("*.jpg"))
    ls_count = 0
    ls_with_labels = 0
    
    for img_path in ls_images:
        img_name = img_path.name
        label_path = LS_EXPORT / "labels" / img_path.with_suffix(".txt").name
        
        # Check if label file has content
        has_labels = label_path.exists() and label_path.stat().st_size > 0
        
        image_sources[img_name] = (img_path, label_path, "label_studio", has_labels)
        ls_count += 1
        if has_labels:
            ls_with_labels += 1
    
    print(f"   Found {ls_count} images from Label Studio")
    print(f"   {ls_with_labels} with bounding boxes, {ls_count - ls_with_labels} negative samples")
    
    # 2. Load v2 dataset (only add images NOT in Label Studio)
    print("\n2. Loading v2 dataset (ML-generated labels)...")
    v2_train_images = list((V2_DIR / "images" / "train").glob("*.jpg"))
    v2_val_images = list((V2_DIR / "images" / "val").glob("*.jpg"))
    v2_all_images = v2_train_images + v2_val_images
    
    v2_added = 0
    v2_skipped = 0
    
    for img_path in v2_all_images:
        img_name = img_path.name
        
        if img_name in image_sources:
            # Skip - Label Studio version takes priority
            v2_skipped += 1
            continue
        
        # Find label path (could be in train or val)
        if img_path.parent.name == "train":
            label_path = V2_DIR / "labels" / "train" / img_path.with_suffix(".txt").name
        else:
            label_path = V2_DIR / "labels" / "val" / img_path.with_suffix(".txt").name
        
        has_labels = label_path.exists() and label_path.stat().st_size > 0
        image_sources[img_name] = (img_path, label_path, "v2", has_labels)
        v2_added += 1
    
    print(f"   Added {v2_added} images from v2 (not in Label Studio)")
    print(f"   Skipped {v2_skipped} images (overridden by Label Studio)")
    
    # 3. Split into train/val (85/15)
    print(f"\n3. Creating 85/15 train/val split...")
    all_images = list(image_sources.keys())
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.85)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"   Total images: {len(all_images)}")
    print(f"   Train: {len(train_images)} ({len(train_images)/len(all_images)*100:.1f}%)")
    print(f"   Val: {len(val_images)} ({len(val_images)/len(all_images)*100:.1f}%)")
    
    # 4. Copy files to v3 directory
    print("\n4. Copying files to dataset_v3...")
    
    stats = defaultdict(int)
    
    for split_name, image_list in [("train", train_images), ("val", val_images)]:
        print(f"\n   Processing {split_name} split...")
        
        for img_name in image_list:
            img_path, label_path, source, has_labels = image_sources[img_name]
            
            # Copy image
            dest_img = V3_DIR / "images" / split_name / img_name
            shutil.copy2(img_path, dest_img)
            
            # Copy label (create empty file if no labels)
            dest_label = V3_DIR / "labels" / split_name / img_path.with_suffix(".txt").name
            if label_path.exists():
                shutil.copy2(label_path, dest_label)
            else:
                dest_label.touch()
            
            stats[f"{split_name}_total"] += 1
            stats[f"{split_name}_{source}"] += 1
            if has_labels:
                stats[f"{split_name}_with_boxes"] += 1
            else:
                stats[f"{split_name}_negatives"] += 1
    
    # 5. Create data.yaml
    print("\n5. Creating data.yaml...")
    data_yaml_content = f"""names:
- fuel_ball
nc: 1
path: {V3_DIR}
train: images/train
val: images/val
"""
    
    with open(V3_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)
    
    print(f"   Created {V3_DIR / 'data.yaml'}")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("DATASET V3 SUMMARY")
    print("=" * 70)
    print(f"\nTrain split ({len(train_images)} images):")
    print(f"  - From Label Studio: {stats['train_label_studio']}")
    print(f"  - From v2: {stats['train_v2']}")
    print(f"  - With bounding boxes: {stats['train_with_boxes']}")
    print(f"  - Negative samples: {stats['train_negatives']}")
    
    print(f"\nVal split ({len(val_images)} images):")
    print(f"  - From Label Studio: {stats['val_label_studio']}")
    print(f"  - From v2: {stats['val_v2']}")
    print(f"  - With bounding boxes: {stats['val_with_boxes']}")
    print(f"  - Negative samples: {stats['val_negatives']}")
    
    total_with_boxes = stats['train_with_boxes'] + stats['val_with_boxes']
    total_negatives = stats['train_negatives'] + stats['val_negatives']
    
    print(f"\nTotal: {len(all_images)} images")
    print(f"  - With bounding boxes: {total_with_boxes}")
    print(f"  - Negative samples: {total_negatives}")
    print(f"\nDataset ready at: {V3_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

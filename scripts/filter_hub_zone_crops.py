#!/usr/bin/env python3
"""Filter hub-zone crops to only those with action (robots or balls).

Reduces ~10K crops to 1-3K interesting crops for labeling.
"""

import cv2
import csv
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5_MODEL = PROJECT_ROOT / "runs/train/frc_v5/weights/best.pt"
INPUT_DIR = PROJECT_ROOT / "data/hub_zones"
OUTPUT_DIR = PROJECT_ROOT / "data/hub_zones_filtered"
INPUT_METADATA = INPUT_DIR / "crop_metadata.csv"
OUTPUT_METADATA = OUTPUT_DIR / "filtered_metadata.csv"

# V5 classes
FUEL_BALL_CLASS = 0
ROBOT_CLASS = 1
HUB_CLASS = 2

# Filtering thresholds
MIN_CONFIDENCE = 0.25  # lower threshold to catch all action
MIN_DETECTIONS = 1  # at least 1 robot or ball


def has_action(crop_path, model):
    """Check if crop has robots or balls."""
    img = cv2.imread(str(crop_path))
    if img is None:
        return False
    
    # Run inference
    results = model(img, verbose=False)
    
    # Check for robot or ball detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Accept robots or balls above threshold
            if conf >= MIN_CONFIDENCE and cls in [FUEL_BALL_CLASS, ROBOT_CLASS]:
                return True
    
    return False


def main():
    print("=" * 80)
    print("Filter Hub-Zone Crops to Interesting Frames")
    print("=" * 80)
    
    # Load v5 model
    print(f"\n📦 Loading v5 model from {V5_MODEL}")
    model = YOLO(str(V5_MODEL))
    
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "blue").mkdir(exist_ok=True)
    (OUTPUT_DIR / "red").mkdir(exist_ok=True)
    
    # Read input metadata
    print(f"\n📋 Reading metadata from {INPUT_METADATA}")
    crops_to_check = []
    with open(INPUT_METADATA, 'r') as f:
        reader = csv.DictReader(f)
        crops_to_check = list(reader)
    
    print(f"   Found {len(crops_to_check):,} total crops")
    
    # Filter crops
    print(f"\n🔍 Filtering crops (keeping those with robots or balls)...")
    
    filtered_crops = []
    
    for row in tqdm(crops_to_check, desc="Processing crops"):
        crop_path = PROJECT_ROOT / row["crop_filename"]
        
        if has_action(crop_path, model):
            filtered_crops.append(row)
            
            # Copy to filtered directory
            output_path = OUTPUT_DIR / row["alliance"] / Path(row["crop_filename"]).name
            shutil.copy2(crop_path, output_path)
    
    # Write filtered metadata
    print(f"\n💾 Saving filtered metadata to {OUTPUT_METADATA}")
    with open(OUTPUT_METADATA, 'w', newline='') as f:
        if filtered_crops:
            writer = csv.DictWriter(f, fieldnames=filtered_crops[0].keys())
            writer.writeheader()
            writer.writerows(filtered_crops)
    
    # Summary
    print(f"\n" + "=" * 80)
    print("✅ Filtering Complete")
    print("=" * 80)
    print(f"📊 Results:")
    print(f"   📥 Input crops: {len(crops_to_check):,}")
    print(f"   ✅ Filtered crops: {len(filtered_crops):,}")
    print(f"   📉 Reduction: {100 * (1 - len(filtered_crops)/len(crops_to_check)):.1f}%")
    
    # Alliance breakdown
    blue_count = sum(1 for c in filtered_crops if c["alliance"] == "blue")
    red_count = sum(1 for c in filtered_crops if c["alliance"] == "red")
    print(f"\n   🔵 Blue: {blue_count:,}")
    print(f"   🔴 Red: {red_count:,}")
    print(f"\n💾 Filtered crops saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare hub-zone crops for Label Studio with v5 pre-annotations.

Generates Label Studio import JSON with:
- All filtered crops
- Pre-annotations from v5 model (fuel_ball, robot)
- New class: hub_opening (for manual labeling)
"""

import json
import csv
from pathlib import Path
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5_MODEL = PROJECT_ROOT / "runs/train/frc_v5/weights/best.pt"
FILTERED_DIR = PROJECT_ROOT / "data/hub_zones_filtered"
METADATA_CSV = FILTERED_DIR / "filtered_metadata.csv"
OUTPUT_JSON = FILTERED_DIR / "labelstudio_import.json"

# V5 classes -> Label Studio classes
V5_TO_LS_CLASS = {
    0: "fuel_ball",
    1: "robot",
    2: "hub"  # will be ignored, user labels hub_opening instead
}

# Label Studio classes for hub-zone model
LS_CLASSES = ["fuel_ball", "robot", "hub_opening"]


def image_to_ls_url(image_path, base_path=PROJECT_ROOT):
    """Convert local image path to Label Studio URL format."""
    # For local Label Studio, we'll use the local file path
    rel_path = image_path.relative_to(base_path)
    return f"/data/local-files/?d={rel_path}"


def generate_preannotations(image_path, model):
    """Generate pre-annotations using v5 model."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    # Run inference
    results = model(img, verbose=False)
    
    annotations = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Skip hub class (user will manually label hub_opening)
            if cls == 2:
                continue
            
            # Convert to Label Studio format (% of image dimensions)
            x_percent = (x1 / w) * 100
            y_percent = (y1 / h) * 100
            width_percent = ((x2 - x1) / w) * 100
            height_percent = ((y2 - y1) / h) * 100
            
            annotations.append({
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rotation": 0,
                    "rectanglelabels": [V5_TO_LS_CLASS[cls]]
                },
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": conf
            })
    
    return annotations


def main():
    print("=" * 80)
    print("Prepare Hub-Zone Crops for Label Studio")
    print("=" * 80)
    
    # Load v5 model
    print(f"\n📦 Loading v5 model for pre-annotations...")
    model = YOLO(str(V5_MODEL))
    
    # Read filtered metadata
    print(f"\n📋 Reading filtered crops metadata...")
    crops = []
    with open(METADATA_CSV, 'r') as f:
        reader = csv.DictReader(f)
        crops = list(reader)
    
    print(f"   Found {len(crops):,} filtered crops")
    
    # Generate Label Studio tasks
    print(f"\n🏷️  Generating Label Studio tasks with pre-annotations...")
    
    tasks = []
    
    for row in tqdm(crops, desc="Processing crops"):
        # Get image path
        alliance = row["alliance"]
        filename = Path(row["crop_filename"]).name
        image_path = FILTERED_DIR / alliance / filename
        
        # Generate pre-annotations
        predictions = generate_preannotations(image_path, model)
        
        # Create task
        task = {
            "data": {
                "image": str(image_path),  # local file path
                "video": row["video"],
                "frame": int(row["frame_number"]),
                "alliance": alliance
            },
            "predictions": [{
                "model_version": "frc_v5",
                "result": predictions
            }] if predictions else []
        }
        
        tasks.append(task)
    
    # Write output JSON
    print(f"\n💾 Saving Label Studio import file...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    # Summary
    print(f"\n" + "=" * 80)
    print("✅ Label Studio Preparation Complete")
    print("=" * 80)
    print(f"📊 Generated {len(tasks):,} tasks")
    print(f"📁 Output: {OUTPUT_JSON}")
    print()
    print("📝 Next steps:")
    print("   1. Import into Label Studio at http://localhost:8080")
    print("   2. Label classes: fuel_ball, robot, hub_opening")
    print("   3. V5 pre-annotations are included for fuel_ball and robot")
    print("   4. Manually label hub_opening (the actual scoring target)")
    print("   5. Export as YOLO format when done")
    print()


if __name__ == "__main__":
    main()

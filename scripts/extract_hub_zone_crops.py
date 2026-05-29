#!/usr/bin/env python3
"""Extract hub-zone crops from Week 0 videos for YOLO26 training.

This script:
1. Uses v5 model to detect hub positions in first 30 frames
2. Crops generous regions around each hub (400x400px minimum)
3. Extracts crops every 3rd frame from all 3 videos
4. Saves organized by alliance: data/hub_zones/{blue|red}/frame_XXXXX.jpg
5. Generates metadata CSV with crop coordinates
"""

import cv2
import csv
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5_MODEL = PROJECT_ROOT / "runs/train/frc_v5/weights/best.pt"
VIDEO_DIR = PROJECT_ROOT / "data/videos/2026"
OUTPUT_DIR = PROJECT_ROOT / "data/hub_zones"
METADATA_CSV = OUTPUT_DIR / "crop_metadata.csv"

# Constants
OVERHEAD_CROP_RATIO = 0.667  # top 67% of frame
HUB_CLASS_ID = 2  # v5 classes: {0: fuel_ball, 1: robot, 2: hub}
CROP_SIZE = 400  # minimum crop size (will be larger if needed)
FRAME_SKIP = 3  # extract every 3rd frame

# Expected hub positions (in cropped overhead coordinates)
BLUE_HUB_EXPECTED = (380, 230)
RED_HUB_EXPECTED = (915, 230)


def crop_overhead_view(frame):
    """Crop to overhead field view (top 67% of frame)."""
    h, w = frame.shape[:2]
    crop_h = int(h * OVERHEAD_CROP_RATIO)
    return frame[:crop_h, :]


def detect_hubs(video_path, model, max_frames=30):
    """Detect hub positions from first N frames, return averaged centers."""
    cap = cv2.VideoCapture(str(video_path))
    
    blue_detections = []
    red_detections = []
    
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop to overhead view
        overhead = crop_overhead_view(frame)
        
        # Run inference
        results = model(overhead, verbose=False)
        
        # Extract hub detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == HUB_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Classify as blue (left) or red (right) based on x position
                    if center_x < overhead.shape[1] / 2:
                        blue_detections.append((center_x, center_y))
                    else:
                        red_detections.append((center_x, center_y))
        
        frame_idx += 1
    
    cap.release()
    
    # Average positions
    blue_center = None
    red_center = None
    
    if blue_detections:
        blue_center = (
            int(np.mean([x for x, y in blue_detections])),
            int(np.mean([y for x, y in blue_detections]))
        )
    
    if red_detections:
        red_center = (
            int(np.mean([x for x, y in red_detections])),
            int(np.mean([y for x, y in red_detections]))
        )
    
    return blue_center, red_center


def extract_crop(frame, center, crop_size=CROP_SIZE):
    """Extract a square crop centered on given position."""
    cx, cy = center
    h, w = frame.shape[:2]
    
    # Calculate crop boundaries
    half_size = crop_size // 2
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + half_size)
    
    crop = frame[y1:y2, x1:x2]
    
    return crop, (x1, y1, x2 - x1, y2 - y1)  # crop, (x, y, w, h)


def extract_crops_from_video(video_path, model, blue_hub, red_hub, output_dir, metadata_writer):
    """Extract hub-zone crops from video every Nth frame."""
    cap = cv2.VideoCapture(str(video_path))
    video_name = video_path.stem
    
    frame_idx = 0
    crop_count = {"blue": 0, "red": 0}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        
        # Crop to overhead view
        overhead = crop_overhead_view(frame)
        
        # Extract crops for each hub
        for alliance, hub_center in [("blue", blue_hub), ("red", red_hub)]:
            if hub_center is None:
                continue
            
            crop, (crop_x, crop_y, crop_w, crop_h) = extract_crop(overhead, hub_center)
            
            # Save crop
            crop_filename = f"frame_{frame_idx:06d}.jpg"
            crop_path = output_dir / alliance / crop_filename
            cv2.imwrite(str(crop_path), crop)
            
            # Write metadata
            metadata_writer.writerow({
                "video": video_name,
                "frame_number": frame_idx,
                "alliance": alliance,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "crop_filename": str(crop_path.relative_to(PROJECT_ROOT))
            })
            
            crop_count[alliance] += 1
        
        frame_idx += 1
    
    cap.release()
    return crop_count


def main():
    print("=" * 80)
    print("Hub-Zone Crop Extraction for YOLO26 Training")
    print("=" * 80)
    
    # Load v5 model
    print(f"\n📦 Loading v5 model from {V5_MODEL}")
    model = YOLO(str(V5_MODEL))
    
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "blue").mkdir(exist_ok=True)
    (OUTPUT_DIR / "red").mkdir(exist_ok=True)
    
    # Find Week 0 videos
    videos = sorted(VIDEO_DIR.glob("week0_f1m*.mp4"))
    print(f"\n📹 Found {len(videos)} Week 0 videos:")
    for v in videos:
        print(f"   - {v.name}")
    
    # Open metadata CSV
    csv_file = open(METADATA_CSV, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "video", "frame_number", "alliance", "crop_x", "crop_y", 
        "crop_w", "crop_h", "crop_filename"
    ])
    csv_writer.writeheader()
    
    total_crops = {"blue": 0, "red": 0}
    
    # Process each video
    for video_path in videos:
        print(f"\n🎬 Processing {video_path.name}")
        
        # Detect hub positions
        print(f"   🔍 Detecting hubs in first 30 frames...")
        blue_hub, red_hub = detect_hubs(video_path, model, max_frames=30)
        
        if blue_hub:
            print(f"   🔵 Blue hub found at {blue_hub}")
        else:
            print(f"   ⚠️  Blue hub not detected, using expected position {BLUE_HUB_EXPECTED}")
            blue_hub = BLUE_HUB_EXPECTED
        
        if red_hub:
            print(f"   🔴 Red hub found at {red_hub}")
        else:
            print(f"   ⚠️  Red hub not detected, using expected position {RED_HUB_EXPECTED}")
            red_hub = RED_HUB_EXPECTED
        
        # Extract crops
        print(f"   ✂️  Extracting crops every {FRAME_SKIP} frames...")
        crop_count = extract_crops_from_video(
            video_path, model, blue_hub, red_hub, OUTPUT_DIR, csv_writer
        )
        
        print(f"   ✅ Extracted {crop_count['blue']} blue crops, {crop_count['red']} red crops")
        total_crops["blue"] += crop_count["blue"]
        total_crops["red"] += crop_count["red"]
    
    csv_file.close()
    
    # Summary
    print(f"\n" + "=" * 80)
    print("✅ Hub-Zone Crop Extraction Complete")
    print("=" * 80)
    print(f"📊 Total crops extracted:")
    print(f"   🔵 Blue: {total_crops['blue']:,}")
    print(f"   🔴 Red: {total_crops['red']:,}")
    print(f"   📈 Total: {total_crops['blue'] + total_crops['red']:,}")
    print(f"\n💾 Crops saved to: {OUTPUT_DIR}")
    print(f"📋 Metadata CSV: {METADATA_CSV}")
    print()


if __name__ == "__main__":
    main()

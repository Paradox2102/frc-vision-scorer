#!/usr/bin/env python3
"""
Prepare training data for YOLO fine-tuning.

1. Extract diverse frames from match videos (720p)
2. Pre-label using color detection + pretrained YOLO
3. Export in YOLO format for manual cleanup and training

Classes:
  0: fuel_ball (yellow ball)
  1: robot (FRC robot)
  2: boiler (scoring structure)
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from ultralytics import YOLO

# Yellow ball HSV range (tightened from initial experiments)
YELLOW_LOW = np.array([20, 100, 100])
YELLOW_HIGH = np.array([32, 255, 255])

# Classes
CLASSES = ["fuel_ball", "robot", "boiler"]


def detect_yellow_balls(frame, min_area=80, max_area=3000, min_circularity=0.5):
    """Detect yellow balls using color filtering. Returns YOLO-format boxes."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    
    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Aspect ratio check (balls should be roughly square)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 2.0:
            continue
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        
        boxes.append((0, cx, cy, nw, nh))  # class 0 = fuel_ball
    
    return boxes


def detect_robots_yolo(model, frame):
    """Use pretrained YOLO to find robot-like objects. Returns YOLO-format boxes."""
    h, w = frame.shape[:2]
    results = model(frame, verbose=False, conf=0.3)
    
    boxes = []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            # Robots often get classified as these
            if cls_name in ("car", "truck", "suitcase", "boat"):
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                nw = (x2 - x1) / w
                nh = (y2 - y1) / h
                
                # Size filter -- robots are medium-large objects
                if nw > 0.03 and nh > 0.03 and nw < 0.4 and nh < 0.4:
                    boxes.append((1, cx, cy, nw, nh))  # class 1 = robot
    
    return boxes


def extract_diverse_frames(video_path, output_dir, target_count=150):
    """Extract frames with diversity (different timestamps, motion levels)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    print(f"Video: {video_path} ({w}x{h}, {duration:.0f}s, {fps:.0f}fps)")
    
    # Sample frames distributed across the video
    # Skip first and last 10% (often pre/post match)
    start = int(total_frames * 0.1)
    end = int(total_frames * 0.9)
    
    # Select frames spread across the video
    indices = np.linspace(start, end, target_count, dtype=int)
    
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        filename = f"frame_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append((filepath, frame))
    
    cap.release()
    print(f"  Extracted {len(frames)} frames")
    return frames


def generate_pre_labels(frames, labels_dir, yolo_model):
    """Generate YOLO-format pre-labels using color detection + YOLO."""
    os.makedirs(labels_dir, exist_ok=True)
    
    total_balls = 0
    total_robots = 0
    
    for filepath, frame in frames:
        # Detect yellow balls
        ball_boxes = detect_yellow_balls(frame)
        
        # Detect robots
        robot_boxes = detect_robots_yolo(yolo_model, frame)
        
        all_boxes = ball_boxes + robot_boxes
        total_balls += len(ball_boxes)
        total_robots += len(robot_boxes)
        
        # Write YOLO label file
        label_name = Path(filepath).stem + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        with open(label_path, "w") as f:
            for cls_id, cx, cy, w, h in all_boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    print(f"  Pre-labeled: {total_balls} balls, {total_robots} robots across {len(frames)} frames")


def create_dataset_yaml(base_dir):
    """Create YOLO dataset configuration."""
    yaml_content = f"""# FRC Vision Scorer Training Dataset
path: {base_dir}
train: images/train
val: images/val

names:
  0: fuel_ball
  1: robot
  2: boiler
"""
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  Dataset config: {yaml_path}")
    return yaml_path


def setup_dataset_structure(base_dir):
    """Create YOLO dataset directory structure."""
    for split in ["train", "val"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)


def split_and_copy(frames, images_dir, labels_dir, val_ratio=0.2):
    """Split frames into train/val and copy to proper dirs."""
    import shutil
    import random
    
    random.seed(42)
    indices = list(range(len(frames)))
    random.shuffle(indices)
    
    val_count = int(len(frames) * val_ratio)
    val_indices = set(indices[:val_count])
    
    for i, (filepath, _) in enumerate(frames):
        split = "val" if i in val_indices else "train"
        
        # Copy image
        img_name = Path(filepath).name
        dst_img = os.path.join(images_dir, split, img_name)
        shutil.copy2(filepath, dst_img)
        
        # Copy label
        label_name = Path(filepath).stem + ".txt"
        src_label = os.path.join(Path(filepath).parent.parent, "pre_labels", label_name)
        dst_label = os.path.join(os.path.dirname(images_dir), "labels", split, label_name)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    train_count = len(frames) - val_count
    print(f"  Split: {train_count} train, {val_count} val")


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    
    # Load YOLO for robot detection
    print("Loading YOLOv8n for pre-labeling...")
    model = YOLO("yolov8n.pt")
    
    # Setup dataset structure
    dataset_dir = str(base / "data" / "dataset_v1")
    setup_dataset_structure(dataset_dir)
    
    # Process each video
    videos = [
        ("data/videos/2017_hq/midwest_final1_720p.mp4", "midwest_final"),
        ("data/videos/2017_hq/einstein_final1_720p.mp4", "einstein_final"),
    ]
    
    all_frames = []
    
    for video_rel, name in videos:
        video_path = str(base / video_rel)
        if not os.path.exists(video_path):
            print(f"Skipping {video_rel} (not found)")
            continue
        
        frames_dir = str(base / "data" / "raw_frames" / name)
        labels_dir = str(base / "data" / "raw_frames" / name / ".." / "pre_labels")
        labels_dir = str(base / "data" / "pre_labels" / name)
        
        # Extract frames
        frames = extract_diverse_frames(video_path, frames_dir, target_count=150)
        
        # Generate pre-labels
        generate_pre_labels(frames, labels_dir, model)
        
        all_frames.extend(frames)
    
    # Split into train/val and copy to dataset
    print(f"\nTotal frames: {len(all_frames)}")
    split_and_copy(
        all_frames,
        os.path.join(dataset_dir, "images"),
        os.path.join(dataset_dir, "labels"),
        val_ratio=0.2
    )
    
    # Create dataset YAML
    create_dataset_yaml(dataset_dir)
    
    print("\nDataset preparation complete!")
    print(f"  Location: {dataset_dir}")
    print(f"  NOTE: Pre-labels are approximate. Manual cleanup recommended before training.")
    print(f"  Use a tool like Roboflow, CVAT, or labelImg to review and correct annotations.")

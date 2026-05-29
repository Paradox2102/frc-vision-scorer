#!/usr/bin/env python3
"""Auto-annotate dataset v4 frames using YOLO v2 model + HSV color detection.

Classes:
- 0: fuel_ball
- 1: robot_red  
- 2: robot_blue
- 3: hub

Strategy:
1. Run existing YOLO v2 model for initial detections
2. Use HSV color filtering for yellow fuel balls (H:18-35, S:80-255, V:80-255)
3. Merge YOLO + HSV detections with NMS
4. Save in YOLO format
5. Create 80/20 train/val split
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "data" / "dataset_v4" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "dataset_v4" / "labels"
MODEL_PATH = PROJECT_ROOT / "runs" / "train" / "frc_v2" / "weights" / "best.pt"

# HSV thresholds for yellow fuel balls
HSV_LOWER = np.array([18, 80, 80])
HSV_UPPER = np.array([35, 255, 255])

# YOLO class IDs (existing v2 model only has fuel_ball)
CLASS_FUEL_BALL = 0
CLASS_ROBOT_RED = 1
CLASS_ROBOT_BLUE = 2
CLASS_HUB = 3

# NMS threshold
NMS_IOU_THRESHOLD = 0.5

# Train/val split
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def detect_yellow_blobs_hsv(image: np.ndarray, min_area: int = 100) -> list[tuple]:
    """Detect yellow fuel balls using HSV color filtering.
    
    Returns list of (x_center, y_center, width, height) normalized 0..1
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image.shape[:2]
    detections = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        
        # Normalize to 0..1
        x_center = (x + box_w / 2) / w
        y_center = (y + box_h / 2) / h
        width = box_w / w
        height = box_h / h
        
        detections.append((x_center, y_center, width, height))
    
    return detections


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two boxes (x_center, y_center, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corners
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def nms(detections: list[tuple], iou_threshold: float = 0.5) -> list[tuple]:
    """Non-maximum suppression on detections.
    
    Each detection: (class_id, confidence, x_center, y_center, width, height)
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        # Remove overlapping detections
        detections = [
            det for det in detections
            if compute_iou(best[2:], det[2:]) < iou_threshold
        ]
    
    return keep


def annotate_image(image_path: Path, model: YOLO, conf_threshold: float = 0.25) -> list[tuple]:
    """Run YOLO + HSV detection and merge with NMS.
    
    Returns list of (class_id, confidence, x_center, y_center, width, height)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    h, w = image.shape[:2]
    all_detections = []
    
    # 1. YOLO detections
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
    for r in results:
        for box in r.boxes:
            # Get normalized coordinates
            x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Map old class IDs to new
            # Assuming v2 model only has fuel_ball (class 0)
            all_detections.append((CLASS_FUEL_BALL, conf, x_center, y_center, width, height))
    
    # 2. HSV color detections
    hsv_detections = detect_yellow_blobs_hsv(image, min_area=100)
    for x, y, w, h in hsv_detections:
        # Use lower confidence for HSV detections
        all_detections.append((CLASS_FUEL_BALL, 0.3, x, y, w, h))
    
    # 3. Merge with NMS
    merged = nms(all_detections, iou_threshold=NMS_IOU_THRESHOLD)
    
    return merged


def save_yolo_annotation(label_path: Path, detections: list[tuple]):
    """Save detections in YOLO format: class x_center y_center width height"""
    with open(label_path, 'w') as f:
        for det in detections:
            class_id, conf, x, y, w, h = det
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def process_all_images(model: YOLO):
    """Process all images in dataset v4."""
    # Find all images
    all_images = []
    for subdir in ["overhead", "alliance_left", "alliance_right", "full"]:
        img_dir = IMAGES_DIR / subdir
        if img_dir.exists():
            all_images.extend(list(img_dir.glob("*.jpg")))
    
    print(f"Found {len(all_images)} images to annotate")
    
    # Create label directories
    for subdir in ["overhead", "alliance_left", "alliance_right", "full"]:
        (LABELS_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    annotated_count = 0
    detection_count = 0
    
    for img_path in tqdm(all_images, desc="Annotating images"):
        detections = annotate_image(img_path, model, conf_threshold=0.25)
        
        # Save label
        label_path = LABELS_DIR / img_path.parent.name / f"{img_path.stem}.txt"
        save_yolo_annotation(label_path, detections)
        
        if detections:
            annotated_count += 1
            detection_count += len(detections)
    
    print(f"\nAnnotation complete:")
    print(f"  Images with detections: {annotated_count}/{len(all_images)}")
    print(f"  Total detections: {detection_count}")
    print(f"  Average detections per image: {detection_count/len(all_images):.2f}")
    
    return all_images


def create_train_val_split(all_images: list[Path]):
    """Create train/val split and generate file lists."""
    random.seed(RANDOM_SEED)
    
    # Shuffle
    shuffled = list(all_images)
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * TRAIN_RATIO)
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]
    
    print(f"\nTrain/val split ({TRAIN_RATIO:.0%}/{(1-TRAIN_RATIO):.0%}):")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")
    
    # Create train.txt and val.txt with relative paths
    dataset_root = PROJECT_ROOT / "data" / "dataset_v4"
    
    train_txt = dataset_root / "train.txt"
    with open(train_txt, 'w') as f:
        for img in train_images:
            rel_path = img.relative_to(dataset_root)
            f.write(f"{rel_path}\n")
    
    val_txt = dataset_root / "val.txt"
    with open(val_txt, 'w') as f:
        for img in val_images:
            rel_path = img.relative_to(dataset_root)
            f.write(f"{rel_path}\n")
    
    print(f"\nCreated split files:")
    print(f"  {train_txt}")
    print(f"  {val_txt}")


def create_dataset_yaml():
    """Create dataset.yaml for YOLO training."""
    dataset_root = PROJECT_ROOT / "data" / "dataset_v4"
    yaml_path = dataset_root / "dataset.yaml"
    
    yaml_content = f"""# FRC Vision Scorer - Dataset v4
# Auto-annotated from Week 0 Finals match videos

path: {dataset_root}
train: train.txt
val: val.txt

# Classes
names:
  0: fuel_ball
  1: robot_red
  2: robot_blue
  3: hub

nc: 4
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset config: {yaml_path}")


def main():
    print("="*70)
    print("FRC Vision Scorer - Auto-annotation (Dataset v4)")
    print("="*70)
    
    # Load YOLO v2 model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Process all images
    all_images = process_all_images(model)
    
    # Create train/val split
    create_train_val_split(all_images)
    
    # Create dataset.yaml
    create_dataset_yaml()
    
    print("\n" + "="*70)
    print("AUTO-ANNOTATION COMPLETE")
    print("="*70)
    print(f"\nDataset ready for training at:")
    print(f"  {PROJECT_ROOT / 'data' / 'dataset_v4'}")


if __name__ == "__main__":
    main()

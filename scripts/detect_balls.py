#!/usr/bin/env python3
"""
Initial ball detection experiment using:
1. Pretrained YOLOv8 (what does it see?)
2. Color-based detection (HSV filtering for yellow balls)
3. Combined approach
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import os

# Yellow ball HSV range (tunable)
YELLOW_LOW = np.array([18, 80, 80])
YELLOW_HIGH = np.array([35, 255, 255])

# Min/max ball radius in pixels (at 640x360)
MIN_RADIUS = 5
MAX_RADIUS = 40


def detect_yellow_circles(frame):
    """Detect yellow circular objects using color filtering + HoughCircles."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create yellow mask
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Skip tiny noise
            continue
        
        # Fit minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        if MIN_RADIUS <= radius <= MAX_RADIUS:
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.4:  # Reasonably circular
                    detections.append({
                        "x": int(x),
                        "y": int(y),
                        "radius": int(radius),
                        "area": int(area),
                        "circularity": round(circularity, 2)
                    })
    
    return detections, mask


def run_yolo_detection(model, frame):
    """Run YOLO detection and return sports ball detections."""
    results = model(frame, verbose=False, conf=0.25)
    
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })
    
    return detections


def annotate_frame(frame, yellow_detections, yolo_detections):
    """Draw detections on frame."""
    annotated = frame.copy()
    
    # Draw yellow ball detections (green circles)
    for det in yellow_detections:
        cv2.circle(annotated, (det["x"], det["y"]), det["radius"], (0, 255, 0), 2)
        cv2.putText(annotated, f"ball({det['circularity']})", 
                    (det["x"] - 20, det["y"] - det["radius"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw YOLO detections (red boxes)
    for det in yolo_detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["class"] == "sports ball" else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return annotated


def process_video_frames(frames_dir: str, output_dir: str, sample_every: int = 5):
    """Process extracted frames with both detection methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO
    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("*.jpg"))
    
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return
    
    print(f"Processing {len(frame_files)} frames from {frames_dir}")
    
    all_results = []
    annotated_count = 0
    
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
        
        # Color-based detection
        yellow_dets, mask = detect_yellow_circles(frame)
        
        # YOLO detection
        yolo_dets = run_yolo_detection(model, frame)
        
        # Filter YOLO to relevant classes
        ball_dets = [d for d in yolo_dets if d["class"] in ("sports ball", "frisbee", "orange")]
        
        result = {
            "frame": frame_file.name,
            "yellow_balls": len(yellow_dets),
            "yolo_balls": len(ball_dets),
            "yolo_all": len(yolo_dets),
            "yellow_details": yellow_dets,
            "yolo_details": yolo_dets[:10],  # Limit detail output
        }
        all_results.append(result)
        
        # Save annotated frames (sample)
        if i % sample_every == 0:
            annotated = annotate_frame(frame, yellow_dets, yolo_dets)
            out_path = os.path.join(output_dir, f"annotated_{frame_file.name}")
            cv2.imwrite(out_path, annotated)
            
            # Also save the yellow mask
            mask_path = os.path.join(output_dir, f"mask_{frame_file.name}")
            cv2.imwrite(mask_path, mask)
            annotated_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames")
    
    # Summary
    total_yellow = sum(r["yellow_balls"] for r in all_results)
    total_yolo = sum(r["yolo_balls"] for r in all_results)
    frames_with_yellow = sum(1 for r in all_results if r["yellow_balls"] > 0)
    frames_with_yolo = sum(1 for r in all_results if r["yolo_balls"] > 0)
    
    # Get all YOLO classes detected
    all_classes = set()
    for r in all_results:
        for d in r["yolo_details"]:
            all_classes.add(d["class"])
    
    print(f"\n{'='*60}")
    print(f"Results for {frames_dir}:")
    print(f"  Total frames: {len(all_results)}")
    print(f"  Yellow (color) detections: {total_yellow} across {frames_with_yellow} frames")
    print(f"  YOLO ball detections: {total_yolo} across {frames_with_yolo} frames")
    print(f"  YOLO classes found: {sorted(all_classes)}")
    print(f"  Annotated frames saved: {annotated_count} to {output_dir}")
    print(f"{'='*60}\n")
    
    # Save results JSON
    results_file = os.path.join(output_dir, "detection_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "summary": {
                "total_frames": len(all_results),
                "yellow_detections": total_yellow,
                "yolo_ball_detections": total_yolo,
                "frames_with_yellow": frames_with_yellow,
                "frames_with_yolo": frames_with_yolo,
                "yolo_classes": sorted(all_classes),
            },
            "frames": all_results
        }, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    
    datasets = [
        ("data/frames/2017_58ball", "data/results/2017_58ball"),
        ("data/frames/2017_final", "data/results/2017_final"),
    ]
    
    for frames_dir, output_dir in datasets:
        frames_path = base / frames_dir
        output_path = base / output_dir
        if frames_path.exists():
            process_video_frames(str(frames_path), str(output_path))
        else:
            print(f"Skipping {frames_dir} (not found)")

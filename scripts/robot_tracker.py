#!/usr/bin/env python3
"""
Robot Tracker for FRC 2026 REBUILT matches.

Uses bumper color detection (red/blue) on the overhead field view to track
robot positions across frames. No YOLO training needed — pure color segmentation.

Pipeline:
1. Crop overhead view (top 65%) from each frame
2. HSV color segmentation for red and blue bumpers
3. Morphological cleanup → contour detection → blob centers
4. ByteTrack for persistent tracking across frames
5. Output: per-frame robot positions with track IDs and alliance color

Usage:
    python scripts/robot_tracker.py --video data/videos/2026/week0_f1m1.mp4
    python scripts/robot_tracker.py  # Process all 3 videos
"""

from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    print("WARNING: supervision not installed. Tracking disabled, detection only.")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos" / "2026"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "week0_analysis"

# Overhead view crop: skip scoreboard (top 80px), take field area only
# Field area in overhead: roughly y=80 to y=420 (exclude scoreboard top and bottom overlap)
OVERHEAD_Y_MIN = 80
OVERHEAD_Y_MAX = 420

# HSV ranges for bumper colors (tuned for FRC broadcast footage)
# Bumpers are fabric nylon, very saturated color, attached to robots
# Need to be STRICT to avoid picking up field elements, crowd, sponsor banners

# Red bumpers (red wraps around in HSV)
RED_LOWER_1 = np.array([0, 150, 100])
RED_UPPER_1 = np.array([8, 255, 255])
RED_LOWER_2 = np.array([170, 150, 100])
RED_UPPER_2 = np.array([180, 255, 255])

# Blue bumpers — strict saturation to avoid picking up the sky, field elements
BLUE_LOWER = np.array([100, 150, 80])
BLUE_UPPER = np.array([125, 255, 255])

# Minimum contour area for a robot bumper (in overhead view pixels)
# A robot is roughly 30-50px wide in overhead view at 720p
MIN_BUMPER_AREA = 400  # ~20x20 pixels minimum
MAX_BUMPER_AREA = 5000  # filter out large false positives (banners, field elements)

# Minimum distance between detected robots (pixels) to merge nearby detections
MERGE_DISTANCE = 50


@dataclass
class RobotDetection:
    """Single robot detection in a frame."""
    frame_number: int
    timestamp: float
    alliance: str  # "red" or "blue"
    center_x: float  # pixel coordinates in overhead view
    center_y: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    area: float
    track_id: Optional[int] = None


def detect_bumpers(frame: np.ndarray, alliance: str) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect bumpers of a specific alliance color using HSV segmentation.
    
    Returns list of (x1, y1, x2, y2, area) bounding boxes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if alliance == "red":
        mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_BUMPER_AREA <= area <= MAX_BUMPER_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio filter — bumpers are wider than tall typically
            aspect = w / max(h, 1)
            if 0.3 < aspect < 5.0:  # reasonable bumper shape
                detections.append((x, y, x + w, y + h, area))
    
    return detections


def merge_nearby_detections(detections: List[Tuple[int, int, int, int, float]], 
                             min_dist: float = MERGE_DISTANCE) -> List[Tuple[int, int, int, int, float]]:
    """Merge detections that are very close (same robot, fragmented bumper)."""
    if len(detections) <= 1:
        return detections
    
    merged = []
    used = set()
    
    for i, (x1, y1, x2, y2, a) in enumerate(detections):
        if i in used:
            continue
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Find all nearby detections to merge
        group_x1, group_y1, group_x2, group_y2 = x1, y1, x2, y2
        group_area = a
        used.add(i)
        
        for j, (ox1, oy1, ox2, oy2, oa) in enumerate(detections):
            if j in used:
                continue
            ocx = (ox1 + ox2) / 2
            ocy = (oy1 + oy2) / 2
            dist = np.sqrt((cx - ocx)**2 + (cy - ocy)**2)
            if dist < min_dist:
                group_x1 = min(group_x1, ox1)
                group_y1 = min(group_y1, oy1)
                group_x2 = max(group_x2, ox2)
                group_y2 = max(group_y2, oy2)
                group_area += oa
                used.add(j)
        
        merged.append((group_x1, group_y1, group_x2, group_y2, group_area))
    
    return merged


def process_video(video_path: Path, sample_rate: int = 3, output_dir: Path = OUTPUT_DIR) -> List[RobotDetection]:
    """
    Process a match video and track robots.
    
    Args:
        video_path: Path to match video
        sample_rate: Process every Nth frame (3 = 10fps for 30fps video)
        output_dir: Where to save results
    
    Returns:
        List of all robot detections across the video
    """
    print(f"\nProcessing: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  {fps:.0f} fps, {total_frames} frames, sampling every {sample_rate} frames")
    
    all_detections: List[RobotDetection] = []
    
    # Set up ByteTrack trackers (one per alliance)
    red_tracker = None
    blue_tracker = None
    if HAS_SUPERVISION:
        red_tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=30,
            minimum_matching_threshold=0.7,
            frame_rate=fps / sample_rate
        )
        blue_tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=30,
            minimum_matching_threshold=0.7,
            frame_rate=fps / sample_rate
        )
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % sample_rate != 0:
            frame_num += 1
            continue
        
        timestamp = frame_num / fps
        
        # Crop overhead view (field area only, skip scoreboard)
        overhead = frame[OVERHEAD_Y_MIN:OVERHEAD_Y_MAX, :]
        
        # Detect red and blue bumpers
        red_dets = detect_bumpers(overhead, "red")
        red_dets = merge_nearby_detections(red_dets)
        
        blue_dets = detect_bumpers(overhead, "blue")
        blue_dets = merge_nearby_detections(blue_dets)
        
        # Keep only top-3 per alliance (exactly 3 robots per alliance)
        # Sort by area descending, take top 3
        red_dets = sorted(red_dets, key=lambda d: d[4], reverse=True)[:3]
        blue_dets = sorted(blue_dets, key=lambda d: d[4], reverse=True)[:3]
        
        # Apply tracking if available
        if HAS_SUPERVISION and red_tracker and blue_tracker:
            # Convert to supervision Detections format
            if red_dets:
                red_xyxy = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in red_dets])
                red_conf = np.array([min(a / 2000, 1.0) for _, _, _, _, a in red_dets])
                red_sv = sv.Detections(xyxy=red_xyxy, confidence=red_conf)
                red_tracked = red_tracker.update_with_detections(red_sv)
                
                for i, (x1, y1, x2, y2) in enumerate(red_tracked.xyxy.astype(int)):
                    track_id = int(red_tracked.tracker_id[i]) if red_tracked.tracker_id is not None else None
                    det = RobotDetection(
                        frame_number=frame_num,
                        timestamp=round(timestamp, 3),
                        alliance="red",
                        center_x=round(float(x1 + x2) / 2, 1),
                        center_y=round(float(y1 + y2) / 2, 1),
                        bbox_x1=int(x1), bbox_y1=int(y1), bbox_x2=int(x2), bbox_y2=int(y2),
                        area=round(float((x2-x1) * (y2-y1)), 1),
                        track_id=track_id
                    )
                    all_detections.append(det)
            else:
                # Empty frame for tracker
                red_tracker.update_with_detections(sv.Detections.empty())
            
            if blue_dets:
                blue_xyxy = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in blue_dets])
                blue_conf = np.array([min(a / 2000, 1.0) for _, _, _, _, a in blue_dets])
                blue_sv = sv.Detections(xyxy=blue_xyxy, confidence=blue_conf)
                blue_tracked = blue_tracker.update_with_detections(blue_sv)
                
                for i, (x1, y1, x2, y2) in enumerate(blue_tracked.xyxy.astype(int)):
                    track_id = int(blue_tracked.tracker_id[i]) if blue_tracked.tracker_id is not None else None
                    det = RobotDetection(
                        frame_number=frame_num,
                        timestamp=round(timestamp, 3),
                        alliance="blue",
                        center_x=round(float(x1 + x2) / 2, 1),
                        center_y=round(float(y1 + y2) / 2, 1),
                        bbox_x1=int(x1), bbox_y1=int(y1), bbox_x2=int(x2), bbox_y2=int(y2),
                        area=round(float((x2-x1) * (y2-y1)), 1),
                        track_id=track_id
                    )
                    all_detections.append(det)
            else:
                blue_tracker.update_with_detections(sv.Detections.empty())
        else:
            # No tracking — just raw detections
            for x1, y1, x2, y2, area in red_dets:
                all_detections.append(RobotDetection(
                    frame_number=frame_num, timestamp=round(timestamp, 3),
                    alliance="red", center_x=(x1+x2)/2, center_y=(y1+y2)/2,
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    area=float(area)
                ))
            for x1, y1, x2, y2, area in blue_dets:
                all_detections.append(RobotDetection(
                    frame_number=frame_num, timestamp=round(timestamp, 3),
                    alliance="blue", center_x=(x1+x2)/2, center_y=(y1+y2)/2,
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    area=float(area)
                ))
        
        if frame_num % (sample_rate * 30) == 0:  # progress every ~1s
            n_red = sum(1 for d in all_detections if d.frame_number == frame_num and d.alliance == "red")
            n_blue = sum(1 for d in all_detections if d.frame_number == frame_num and d.alliance == "blue")
            print(f"  Frame {frame_num:5d} ({timestamp:6.1f}s): {n_red} red, {n_blue} blue robots")
        
        frame_num += 1
    
    cap.release()
    
    # Summarize tracking
    if HAS_SUPERVISION:
        red_tracks = set(d.track_id for d in all_detections if d.alliance == "red" and d.track_id is not None)
        blue_tracks = set(d.track_id for d in all_detections if d.alliance == "blue" and d.track_id is not None)
        print(f"  Red tracks: {len(red_tracks)} (expect 3)")
        print(f"  Blue tracks: {len(blue_tracks)} (expect 3)")
    
    print(f"  Total detections: {len(all_detections)}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem
    
    # JSONL output
    jsonl_path = output_dir / f"{video_name}_robot_tracks.jsonl"
    with open(jsonl_path, 'w') as f:
        for det in all_detections:
            f.write(json.dumps(asdict(det)) + '\n')
    print(f"  Saved: {jsonl_path}")
    
    return all_detections


def analyze_hub_proximity(detections: List[RobotDetection], 
                          hub_center: Tuple[float, float] = (640, 234),
                          proximity_radius: float = 150) -> dict:
    """
    Analyze which robots spend the most time near the hub.
    
    Args:
        detections: All robot detections
        hub_center: Approximate hub center in overhead view (x, y)
        proximity_radius: Distance threshold for "near hub"
    
    Returns:
        Stats per track_id
    """
    hub_x, hub_y = hub_center
    track_stats = {}
    
    for det in detections:
        key = f"{det.alliance}_{det.track_id}"
        if key not in track_stats:
            track_stats[key] = {
                "alliance": det.alliance,
                "track_id": det.track_id,
                "total_frames": 0,
                "near_hub_frames": 0,
                "avg_hub_distance": 0,
                "distances": []
            }
        
        dist = np.sqrt((det.center_x - hub_x)**2 + (det.center_y - hub_y)**2)
        track_stats[key]["total_frames"] += 1
        track_stats[key]["distances"].append(dist)
        if dist < proximity_radius:
            track_stats[key]["near_hub_frames"] += 1
    
    # Compute averages
    for stats in track_stats.values():
        if stats["distances"]:
            stats["avg_hub_distance"] = round(np.mean(stats["distances"]), 1)
            stats["hub_proximity_pct"] = round(100 * stats["near_hub_frames"] / stats["total_frames"], 1)
        del stats["distances"]
    
    return track_stats


def main():
    parser = argparse.ArgumentParser(description="Track robots in FRC match videos")
    parser.add_argument("--video", type=str, help="Single video to process")
    parser.add_argument("--sample-rate", type=int, default=3, help="Process every Nth frame")
    args = parser.parse_args()
    
    if args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(VIDEOS_DIR.glob("week0_f1m*.mp4"))
    
    if not videos:
        print("No videos found!")
        return
    
    for video_path in videos:
        detections = process_video(video_path, sample_rate=args.sample_rate)
        
        # Analyze hub proximity
        hub_stats = analyze_hub_proximity(detections)
        
        print(f"\n  Hub proximity analysis:")
        for key, stats in sorted(hub_stats.items(), key=lambda x: -x[1].get("hub_proximity_pct", 0)):
            print(f"    {stats['alliance']} track {stats['track_id']}: "
                  f"{stats['hub_proximity_pct']:.0f}% near hub, "
                  f"avg dist {stats['avg_hub_distance']:.0f}px, "
                  f"{stats['total_frames']} frames")
        
        # Save hub stats
        stats_path = OUTPUT_DIR / f"{video_path.stem}_hub_proximity.json"
        with open(stats_path, 'w') as f:
            json.dump(hub_stats, f, indent=2)
        print(f"  Saved: {stats_path}")


if __name__ == "__main__":
    main()

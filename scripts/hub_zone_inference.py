#!/usr/bin/env python3
"""Hub-zone inference pipeline for per-robot scoring attribution.

Complete pipeline:
1. Load video, detect hubs with v5 in first frames
2. Crop hub zones each frame
3. Run YOLO26 hub-zone model on crops
4. Track robots in hub zones with ByteTrack
5. Detect balls entering hub opening
6. Attribute each ball to nearest robot
7. Integrate with scoreboard OCR to validate scoring events
8. Output per-robot score counts
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json

from ultralytics import YOLO
import supervision as sv

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5_MODEL = PROJECT_ROOT / "runs/train/frc_v5/weights/best.pt"
HUB_ZONE_MODEL = PROJECT_ROOT / "runs/train/hub_zone_v1/weights/best.pt"

# Constants
OVERHEAD_CROP_RATIO = 0.667
CROP_SIZE = 400
HUB_CLASS_V5 = 2
BALL_CLASS = 0
ROBOT_CLASS = 1
HUB_OPENING_CLASS = 2

# Scoring parameters
HUB_ZONE_RADIUS = 50  # pixels from hub center
BALL_ENTRY_THRESHOLD = 20  # pixels from hub opening
ROBOT_ATTRIBUTION_RADIUS = 100  # pixels to attribute ball to robot
MIN_BALL_VELOCITY = 3.0  # px/frame
SCORE_VALIDATION_FRAMES = 15  # frames to wait for score change

# OCR regions (720p coordinates)
BLUE_SCORE_REGION = (45, 35, 90, 45)
RED_SCORE_REGION = (1170, 35, 95, 45)


@dataclass
class TrackedBall:
    """Tracked ball in hub zone."""
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    attributed_robot: Optional[int] = None
    scored: bool = False
    entry_frame: Optional[int] = None


@dataclass
class TrackedRobot:
    """Tracked robot in hub zone."""
    track_id: int
    alliance: str  # "blue" or "red"
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    scores: int = 0


class HubZoneInference:
    """Hub-zone inference pipeline."""
    
    def __init__(self, video_path: Path, hub_zone_model_path: Path, v5_model_path: Path):
        self.video_path = video_path
        
        # Load models
        print("📦 Loading models...")
        self.v5_model = YOLO(str(v5_model_path))
        self.hub_zone_model = YOLO(str(hub_zone_model_path))
        
        # ByteTrack trackers (one per alliance)
        self.ball_tracker_blue = sv.ByteTrack()
        self.ball_tracker_red = sv.ByteTrack()
        self.robot_tracker_blue = sv.ByteTrack()
        self.robot_tracker_red = sv.ByteTrack()
        
        # Tracking data
        self.tracked_balls = {"blue": {}, "red": {}}
        self.tracked_robots = {"blue": {}, "red": {}}
        
        # Hub positions
        self.blue_hub = None
        self.red_hub = None
        self.blue_hub_opening = None
        self.red_hub_opening = None
        
        # Scoreboard tracking
        self.blue_score = 0
        self.red_score = 0
        
    def detect_hubs(self, max_frames=30):
        """Detect hub positions from first N frames."""
        print("🔍 Detecting hub positions...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        blue_detections = []
        red_detections = []
        
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            overhead = self.crop_overhead(frame)
            results = self.v5_model(overhead, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == HUB_CLASS_V5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        if center_x < overhead.shape[1] / 2:
                            blue_detections.append((center_x, center_y))
                        else:
                            red_detections.append((center_x, center_y))
        
        cap.release()
        
        if blue_detections:
            self.blue_hub = (
                int(np.mean([x for x, y in blue_detections])),
                int(np.mean([y for x, y in blue_detections]))
            )
            print(f"   🔵 Blue hub: {self.blue_hub}")
        
        if red_detections:
            self.red_hub = (
                int(np.mean([x for x, y in red_detections])),
                int(np.mean([y for x, y in red_detections]))
            )
            print(f"   🔴 Red hub: {self.red_hub}")
    
    def crop_overhead(self, frame):
        """Crop to overhead view."""
        h = frame.shape[0]
        crop_h = int(h * OVERHEAD_CROP_RATIO)
        return frame[:crop_h, :]
    
    def extract_hub_crop(self, frame, hub_center):
        """Extract crop around hub."""
        cx, cy = hub_center
        h, w = frame.shape[:2]
        
        half_size = CROP_SIZE // 2
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        
        return frame[y1:y2, x1:x2], (x1, y1)
    
    def process_hub_zone(self, crop, alliance, frame_idx, crop_offset):
        """Process a single hub zone crop."""
        # Run hub-zone model
        results = self.hub_zone_model(crop, verbose=False)
        
        if not results or len(results) == 0:
            return
        
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return
        
        # Separate detections by class
        ball_detections = []
        robot_detections = []
        hub_opening = None
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to overhead coordinates
            x1_global = x1 + crop_offset[0]
            y1_global = y1 + crop_offset[1]
            x2_global = x2 + crop_offset[0]
            y2_global = y2 + crop_offset[1]
            
            if cls == BALL_CLASS:
                ball_detections.append([x1_global, y1_global, x2_global, y2_global, conf])
            elif cls == ROBOT_CLASS:
                robot_detections.append([x1_global, y1_global, x2_global, y2_global, conf])
            elif cls == HUB_OPENING_CLASS and hub_opening is None:
                # Take first hub opening detection
                hub_opening = ((x1_global + x2_global) / 2, (y1_global + y2_global) / 2)
        
        # Update hub opening position
        if hub_opening:
            if alliance == "blue":
                self.blue_hub_opening = hub_opening
            else:
                self.red_hub_opening = hub_opening
        
        # Track balls
        if ball_detections:
            ball_detections = np.array(ball_detections)
            detections = sv.Detections(
                xyxy=ball_detections[:, :4],
                confidence=ball_detections[:, 4],
                class_id=np.zeros(len(ball_detections), dtype=int)
            )
            
            if alliance == "blue":
                tracked = self.ball_tracker_blue.update_with_detections(detections)
            else:
                tracked = self.ball_tracker_red.update_with_detections(detections)
            
            self.update_ball_tracking(tracked, alliance, frame_idx)
        
        # Track robots
        if robot_detections:
            robot_detections = np.array(robot_detections)
            detections = sv.Detections(
                xyxy=robot_detections[:, :4],
                confidence=robot_detections[:, 4],
                class_id=np.ones(len(robot_detections), dtype=int)
            )
            
            if alliance == "blue":
                tracked = self.robot_tracker_blue.update_with_detections(detections)
            else:
                tracked = self.robot_tracker_red.update_with_detections(detections)
            
            self.update_robot_tracking(tracked, alliance, frame_idx)
    
    def update_ball_tracking(self, tracked_detections, alliance, frame_idx):
        """Update ball tracking and check for scoring."""
        hub_opening = self.blue_hub_opening if alliance == "blue" else self.red_hub_opening
        
        if hub_opening is None:
            return
        
        for i, track_id in enumerate(tracked_detections.tracker_id):
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Get or create tracked ball
            if track_id not in self.tracked_balls[alliance]:
                self.tracked_balls[alliance][track_id] = TrackedBall(track_id=track_id)
            
            ball = self.tracked_balls[alliance][track_id]
            ball.positions.append(center)
            
            # Check if ball entered hub opening
            if not ball.scored and len(ball.positions) >= 2:
                dist_to_hub = np.linalg.norm(
                    np.array(center) - np.array(hub_opening)
                )
                
                if dist_to_hub < BALL_ENTRY_THRESHOLD:
                    # Ball scored!
                    ball.scored = True
                    ball.entry_frame = frame_idx
                    
                    # Attribute to nearest robot
                    nearest_robot = self.find_nearest_robot(center, alliance)
                    if nearest_robot:
                        ball.attributed_robot = nearest_robot
                        self.tracked_robots[alliance][nearest_robot].scores += 1
    
    def update_robot_tracking(self, tracked_detections, alliance, frame_idx):
        """Update robot tracking."""
        for i, track_id in enumerate(tracked_detections.tracker_id):
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if track_id not in self.tracked_robots[alliance]:
                self.tracked_robots[alliance][track_id] = TrackedRobot(
                    track_id=track_id,
                    alliance=alliance
                )
            
            robot = self.tracked_robots[alliance][track_id]
            robot.positions.append(center)
    
    def find_nearest_robot(self, ball_position, alliance):
        """Find nearest robot to ball."""
        min_dist = float('inf')
        nearest_robot_id = None
        
        for robot_id, robot in self.tracked_robots[alliance].items():
            if len(robot.positions) == 0:
                continue
            
            robot_pos = robot.positions[-1]
            dist = np.linalg.norm(
                np.array(ball_position) - np.array(robot_pos)
            )
            
            if dist < min_dist and dist < ROBOT_ATTRIBUTION_RADIUS:
                min_dist = dist
                nearest_robot_id = robot_id
        
        return nearest_robot_id
    
    def process_video(self):
        """Process entire video."""
        print(f"\n🎬 Processing video: {self.video_path.name}")
        
        # Detect hubs
        self.detect_hubs()
        
        if not self.blue_hub or not self.red_hub:
            print("❌ Could not detect both hubs!")
            return
        
        # Process frames
        cap = cv2.VideoCapture(str(self.video_path))
        frame_idx = 0
        
        print("\n⚙️  Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            overhead = self.crop_overhead(frame)
            
            # Process blue hub zone
            blue_crop, blue_offset = self.extract_hub_crop(overhead, self.blue_hub)
            self.process_hub_zone(blue_crop, "blue", frame_idx, blue_offset)
            
            # Process red hub zone
            red_crop, red_offset = self.extract_hub_crop(overhead, self.red_hub)
            self.process_hub_zone(red_crop, "red", frame_idx, red_offset)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"   Processed {frame_idx} frames...")
        
        cap.release()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate scoring report."""
        print("\n" + "=" * 80)
        print("📊 Per-Robot Scoring Report")
        print("=" * 80)
        
        for alliance in ["blue", "red"]:
            emoji = "🔵" if alliance == "blue" else "🔴"
            print(f"\n{emoji} {alliance.upper()} Alliance:")
            
            robots = self.tracked_robots[alliance]
            if not robots:
                print("   No robots detected")
                continue
            
            total_scores = sum(r.scores for r in robots.values())
            
            for robot_id, robot in sorted(robots.items()):
                print(f"   Robot #{robot_id}: {robot.scores} scores")
            
            print(f"   Total: {total_scores} scores")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Hub-zone inference pipeline")
    parser.add_argument(
        "video",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--hub-zone-model",
        type=str,
        default=str(HUB_ZONE_MODEL),
        help="Path to hub-zone model weights"
    )
    parser.add_argument(
        "--v5-model",
        type=str,
        default=str(V5_MODEL),
        help="Path to v5 model weights"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Hub-Zone Inference Pipeline")
    print("=" * 80)
    
    # Check if hub-zone model exists
    if not Path(args.hub_zone_model).exists():
        print(f"\n⚠️  Hub-zone model not found: {args.hub_zone_model}")
        print("   Please train the model first using train_hub_zone_model.py")
        return
    
    # Run inference
    pipeline = HubZoneInference(
        video_path=Path(args.video),
        hub_zone_model_path=Path(args.hub_zone_model),
        v5_model_path=Path(args.v5_model)
    )
    
    pipeline.process_video()


if __name__ == "__main__":
    main()

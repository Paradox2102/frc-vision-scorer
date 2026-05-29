#!/usr/bin/env python3
"""Scoring Attribution Pipeline - Attribute fuel ball scores to individual robots.

Input:
- scoring_events.jsonl: Timestamps of scoring events from scoreboard OCR
- robot_tracks.jsonl: Tracked robot positions from robot_tracker.py

Output:
- per_robot_scores.jsonl: Per-robot scoring attribution

Algorithm:
For each scoring event:
1. Find all robots near the hub at that timestamp (±2s window)
2. Calculate proximity score (distance to hub)
3. Calculate trajectory score (approaching vs leaving)
4. Attribute to the robot with the highest combined score
"""

from pathlib import Path
import argparse
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Hub location (normalized coordinates 0..1, approximate center of field)
HUB_CENTER_X = 0.5
HUB_CENTER_Y = 0.4

# Scoring attribution parameters
PROXIMITY_RADIUS = 0.15  # Within 15% of frame width/height
TIME_WINDOW = 2.0  # Look ±2 seconds around scoring event
TRAJECTORY_FRAMES = 5  # Look at last 5 frames for trajectory


def load_robot_tracks(jsonl_path: Path) -> Dict:
    """Load robot tracks from JSONL file.
    
    Returns: dict mapping track_id -> list of track records
    """
    tracks = defaultdict(list)
    
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            track_id = record["track_id"]
            tracks[track_id].append(record)
    
    # Sort each track by timestamp
    for track_id in tracks:
        tracks[track_id].sort(key=lambda r: r["timestamp"])
    
    return dict(tracks)


def load_scoring_events(jsonl_path: Path) -> List[Dict]:
    """Load scoring events from JSONL file."""
    events = []
    
    with open(jsonl_path) as f:
        for line in f:
            events.append(json.loads(line))
    
    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp"])
    
    return events


def create_mock_scoring_events(video_duration: float, fps: float = 30.0) -> List[Dict]:
    """Create mock scoring events for testing (based on typical FRC match pacing).
    
    In a real match, fuel balls are scored throughout, peaking during autonomous
    and late teleop. This generates plausible timestamps.
    """
    events = []
    
    # Autonomous (0-15s): High scoring rate
    for t in np.arange(5.0, 15.0, 2.0):
        events.append({"timestamp": round(t, 2), "alliance": "unknown", "value": 1})
    
    # Teleop (15s - 135s): Moderate scoring rate
    for t in np.arange(20.0, min(135.0, video_duration), 5.0):
        events.append({"timestamp": round(t, 2), "alliance": "unknown", "value": 1})
    
    # Endgame rush (135s - 150s): High scoring rate
    for t in np.arange(135.0, min(150.0, video_duration), 2.5):
        events.append({"timestamp": round(t, 2), "alliance": "unknown", "value": 1})
    
    return events


def distance_to_hub(center_xy: List[float]) -> float:
    """Calculate normalized distance from robot center to hub center."""
    dx = center_xy[0] - HUB_CENTER_X
    dy = center_xy[1] - HUB_CENTER_Y
    return np.sqrt(dx*dx + dy*dy)


def compute_trajectory_score(track_records: List[Dict], target_timestamp: float) -> float:
    """Compute trajectory score (positive if approaching hub, negative if leaving).
    
    Returns: score in range [-1, 1] where positive = approaching
    """
    # Find records around target timestamp
    relevant = [
        r for r in track_records
        if abs(r["timestamp"] - target_timestamp) < TIME_WINDOW
    ]
    
    if len(relevant) < 2:
        return 0.0
    
    # Sort by timestamp
    relevant.sort(key=lambda r: r["timestamp"])
    
    # Take last N frames before target
    before_target = [r for r in relevant if r["timestamp"] <= target_timestamp]
    if len(before_target) < 2:
        return 0.0
    
    frames = before_target[-min(TRAJECTORY_FRAMES, len(before_target)):]
    
    # Compute distance change over time
    distances = [distance_to_hub(r["center_xy"]) for r in frames]
    
    # Negative slope = approaching (distance decreasing)
    if len(distances) >= 2:
        delta_distance = distances[-1] - distances[0]
        delta_time = frames[-1]["timestamp"] - frames[0]["timestamp"]
        
        if delta_time > 0:
            velocity = delta_distance / delta_time
            # Normalize to [-1, 1] (approaching = positive score)
            return np.clip(-velocity * 2.0, -1.0, 1.0)
    
    return 0.0


def attribute_score(
    event: Dict,
    robot_tracks: Dict,
) -> Tuple[int, float, str]:
    """Attribute a scoring event to a specific robot.
    
    Returns: (track_id, confidence, reasoning)
    """
    event_time = event["timestamp"]
    
    # Find all robots near the hub at event time
    candidates = []
    
    for track_id, track_records in robot_tracks.items():
        # Find closest record to event time
        closest = min(
            track_records,
            key=lambda r: abs(r["timestamp"] - event_time),
            default=None
        )
        
        if closest is None:
            continue
        
        time_diff = abs(closest["timestamp"] - event_time)
        if time_diff > TIME_WINDOW:
            continue
        
        # Check proximity to hub
        dist = distance_to_hub(closest["center_xy"])
        if dist > PROXIMITY_RADIUS:
            continue
        
        # Compute trajectory score
        traj_score = compute_trajectory_score(track_records, event_time)
        
        # Combined score (proximity + trajectory)
        proximity_score = 1.0 - (dist / PROXIMITY_RADIUS)  # 0..1, higher is closer
        combined_score = proximity_score * 0.6 + (traj_score * 0.5 + 0.5) * 0.4
        
        candidates.append({
            "track_id": track_id,
            "alliance": closest["alliance"],
            "team_number": closest["team_number"],
            "proximity_score": proximity_score,
            "trajectory_score": traj_score,
            "combined_score": combined_score,
            "distance": dist,
            "time_diff": time_diff,
        })
    
    if not candidates:
        return (-1, 0.0, "no_robots_nearby")
    
    # Sort by combined score
    candidates.sort(key=lambda c: c["combined_score"], reverse=True)
    best = candidates[0]
    
    # Determine confidence
    if len(candidates) == 1:
        confidence = 0.9
        reasoning = "only_robot_nearby"
    else:
        score_gap = best["combined_score"] - candidates[1]["combined_score"]
        confidence = min(0.9, 0.5 + score_gap)
        reasoning = f"best_of_{len(candidates)}"
    
    return (best["track_id"], confidence, reasoning)


def compute_per_robot_scores(
    robot_tracks: Dict,
    scoring_events: List[Dict],
    output_jsonl: Path,
):
    """Compute per-robot scoring attribution and save to JSONL."""
    
    print(f"\nAttributing {len(scoring_events)} scoring events to {len(robot_tracks)} tracked robots...")
    
    # Track scores per robot
    robot_scores = defaultdict(lambda: {
        "team_number": "",
        "alliance": "unknown",
        "total_scored": 0,
        "high_confidence": 0,
        "low_confidence": 0,
        "scoring_timestamps": [],
    })
    
    # Attribution details (for debugging/analysis)
    attributions = []
    
    for event in scoring_events:
        track_id, confidence, reasoning = attribute_score(event, robot_tracks)
        
        attributions.append({
            "timestamp": event["timestamp"],
            "attributed_to": track_id,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
        })
        
        if track_id != -1:
            robot_scores[track_id]["total_scored"] += 1
            robot_scores[track_id]["scoring_timestamps"].append(event["timestamp"])
            
            if confidence >= 0.7:
                robot_scores[track_id]["high_confidence"] += 1
            else:
                robot_scores[track_id]["low_confidence"] += 1
            
            # Get alliance/team from tracks
            if robot_tracks[track_id]:
                latest = robot_tracks[track_id][-1]
                robot_scores[track_id]["alliance"] = latest["alliance"]
                robot_scores[track_id]["team_number"] = latest["team_number"]
    
    # Write per-robot scores
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_jsonl, 'w') as f:
        for track_id, scores in sorted(robot_scores.items()):
            record = {
                "track_id": track_id,
                "team_number": scores["team_number"],
                "alliance": scores["alliance"],
                "total_scored": scores["total_scored"],
                "high_confidence": scores["high_confidence"],
                "low_confidence": scores["low_confidence"],
                "scoring_timestamps": scores["scoring_timestamps"],
            }
            f.write(json.dumps(record) + '\n')
    
    # Write attribution details
    attributions_path = output_jsonl.parent / f"{output_jsonl.stem}_attributions.jsonl"
    with open(attributions_path, 'w') as f:
        for attr in attributions:
            f.write(json.dumps(attr) + '\n')
    
    # Print summary
    print(f"\nAttribution complete:")
    print(f"  Total events: {len(scoring_events)}")
    print(f"  Attributed: {len([a for a in attributions if a['attributed_to'] != -1])}")
    print(f"  Unattributed: {len([a for a in attributions if a['attributed_to'] == -1])}")
    print(f"  Robots with scores: {len(robot_scores)}")
    print(f"\nPer-robot scores:")
    for track_id, scores in sorted(robot_scores.items(), key=lambda x: x[1]["total_scored"], reverse=True):
        team = scores["team_number"] or "????"
        alliance = scores["alliance"]
        total = scores["total_scored"]
        high_conf = scores["high_confidence"]
        low_conf = scores["low_confidence"]
        print(f"  Track #{track_id} ({alliance} {team}): {total} scored ({high_conf} high conf, {low_conf} low conf)")
    
    print(f"\nOutput:")
    print(f"  Scores: {output_jsonl}")
    print(f"  Attributions: {attributions_path}")


def main():
    parser = argparse.ArgumentParser(description="Scoring attribution pipeline")
    parser.add_argument("robot_tracks", type=Path, help="Robot tracks JSONL file")
    parser.add_argument("--scoring-events", type=Path, default=None, help="Scoring events JSONL file")
    parser.add_argument("--output", type=Path, default=None, help="Output per-robot scores JSONL")
    parser.add_argument("--mock-events", action="store_true", help="Use mock scoring events for testing")
    parser.add_argument("--video-duration", type=float, default=172.0, help="Video duration for mock events")
    args = parser.parse_args()
    
    # Load robot tracks
    print(f"Loading robot tracks from: {args.robot_tracks}")
    robot_tracks = load_robot_tracks(args.robot_tracks)
    print(f"  Loaded {len(robot_tracks)} tracked robots")
    
    # Load or create scoring events
    if args.mock_events or not args.scoring_events:
        print("\nCreating mock scoring events...")
        scoring_events = create_mock_scoring_events(args.video_duration)
        print(f"  Created {len(scoring_events)} mock events")
    else:
        print(f"\nLoading scoring events from: {args.scoring_events}")
        scoring_events = load_scoring_events(args.scoring_events)
        print(f"  Loaded {len(scoring_events)} events")
    
    # Determine output path
    if args.output:
        output_jsonl = args.output
    else:
        tracks_stem = args.robot_tracks.stem.replace("_robot_tracks", "")
        output_jsonl = PROJECT_ROOT / "data" / "results" / f"{tracks_stem}_per_robot_scores.jsonl"
    
    # Compute attribution
    compute_per_robot_scores(robot_tracks, scoring_events, output_jsonl)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end FRC Vision Scoring Pipeline.

Combines:
1. Scoreboard OCR (scoring event detection)
2. YOLO v4 robot detection + ByteTrack tracking
3. Proximity-based scoring attribution

Usage:
    python scripts/run_pipeline.py --video data/videos/2026/week0_f1m1.mp4 \
        --tba-red 263 --tba-blue 187 \
        --red-teams 2342,6328,2877 --blue-teams 1768,4909,5962

    python scripts/run_pipeline.py --all  # Run all 3 Week 0 matches
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── YOLO model path ──
YOLO_MODEL = PROJECT_ROOT / "runs" / "train" / "frc_v4" / "weights" / "best.pt"
YOLO_CLASSES = {0: "fuel_ball", 1: "robot_red", 2: "robot_blue"}

# ── Frame geometry ──
OVERHEAD_Y_MIN = 0       # Scoreboard is part of overhead
OVERHEAD_Y_MAX = 468     # Top 65% of 720
SCOREBOARD_Y_MAX = 80    # Scoreboard overlay region

# ── Hub locations (pixel coords in overhead 1280x468 frame) ──
# Each alliance has its own hub — robots score at THEIR alliance's hub
# Identified from averaged field image analysis
BLUE_HUB = (395, 300)     # Blue alliance hub (left side of field)
RED_HUB = (890, 300)      # Red alliance hub (right side of field)
HUB_RADIUS_PX = 120       # Approximate hub structure radius in pixels

# ── Attribution params ──
PROXIMITY_RADIUS_PX = 250  # Wider radius — robots approach from all angles
TIME_WINDOW_S = 1.5        # ±1.5s window around scoring event
MIN_CONFIDENCE = 0.35      # YOLO confidence threshold for robots


@dataclass
class ScoringEvent:
    """A detected +1 score change from OCR."""
    frame_number: int
    timestamp: float
    alliance: str  # "red" or "blue"
    old_score: int
    new_score: int


@dataclass
class RobotPosition:
    """A robot detection in a single frame."""
    frame_number: int
    timestamp: float
    alliance: str
    center_x: float
    center_y: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    track_id: int = -1


@dataclass
class RobotTrack:
    """A tracked robot across frames."""
    track_id: int
    alliance: str
    positions: List[RobotPosition] = field(default_factory=list)

    @property
    def last_position(self) -> Optional[RobotPosition]:
        return self.positions[-1] if self.positions else None


# ────────────────────────────────────────────────────────────
# Stage 1: Scoreboard OCR (reuse existing module)
# ────────────────────────────────────────────────────────────

def run_ocr(video_path: str, tba_red: int, tba_blue: int) -> Tuple[List[ScoringEvent], dict]:
    """Run scoreboard OCR and return scoring events + validation.

    Uses TBA totals as ground truth to calibrate event counts.
    OCR provides TIMING of score changes; TBA provides exact COUNTS.
    """
    from scripts.scoreboard_ocr import ScoreboardOCR

    ocr = ScoreboardOCR(video_path, sample_rate=6)  # Every 6th frame (~5fps)
    ocr.process_video()
    validation = ocr.validate_against_tba(tba_red, tba_blue)

    # Extract raw scoring events per alliance with timestamps
    raw_events = {"red": [], "blue": []}
    prev_scores = {"red": 0, "blue": 0}

    for ocr_event in ocr.scoring_events:
        alliance = ocr_event.alliance
        new_score = ocr_event.cumulative_score
        old_score = prev_scores[alliance]

        if new_score > old_score:
            delta = new_score - old_score
            for i in range(delta):
                raw_events[alliance].append(ScoringEvent(
                    frame_number=ocr_event.frame_number,
                    timestamp=ocr_event.timestamp,
                    alliance=alliance,
                    old_score=old_score + i,
                    new_score=old_score + i + 1,
                ))
            prev_scores[alliance] = new_score

    # TBA-calibrated approach: OCR may overcount due to digit misreads.
    # Use TBA total as ground truth for how many balls scored.
    # If OCR detected MORE events than TBA, uniformly subsample to match.
    # If OCR detected FEWER, keep all (undercounting is less harmful).
    events = []
    for alliance, tba_total in [("red", tba_red), ("blue", tba_blue)]:
        alliance_events = raw_events[alliance]
        raw_count = len(alliance_events)

        if raw_count > tba_total and tba_total > 0:
            # Subsample: keep tba_total events, evenly spaced
            indices = np.linspace(0, raw_count - 1, tba_total, dtype=int)
            calibrated = [alliance_events[i] for i in indices]
            # Renumber scores sequentially
            for i, evt in enumerate(calibrated):
                evt.old_score = i
                evt.new_score = i + 1
            events.extend(calibrated)
            print(f"  OCR {alliance}: {raw_count} raw -> {tba_total} calibrated (TBA)")
        else:
            events.extend(alliance_events)
            print(f"  OCR {alliance}: {raw_count} events (TBA: {tba_total})")

    # Sort all events by timestamp
    events.sort(key=lambda e: (e.timestamp, e.alliance))

    total_red = sum(1 for e in events if e.alliance == "red")
    total_blue = sum(1 for e in events if e.alliance == "blue")
    print(f"  OCR total: {len(events)} events ({total_red} red, {total_blue} blue)")
    print(f"  OCR validation: Red {validation['ocr_red']}/{validation['tba_red']} "
          f"({validation['red_accuracy']:.1%}), "
          f"Blue {validation['ocr_blue']}/{validation['tba_blue']} "
          f"({validation['blue_accuracy']:.1%})")

    return events, validation


# ────────────────────────────────────────────────────────────
# Stage 2: YOLO Robot Detection + Simple Tracking
# ────────────────────────────────────────────────────────────

def run_robot_tracking(video_path: str, sample_rate: int = 3) -> Dict[int, RobotTrack]:
    """Run YOLO v4 on overhead frames, track robots with simple IoU tracker."""
    from ultralytics import YOLO

    model = YOLO(str(YOLO_MODEL))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks: Dict[int, RobotTrack] = {}
    next_track_id = 0
    active_tracks: Dict[int, RobotPosition] = {}  # track_id -> last position

    frame_count = 0
    processed = 0

    print(f"  Robot tracking: {total_frames} frames @ {fps:.0f}fps, sampling every {sample_rate}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame_count
        frame_count += 1

        if current_frame % sample_rate != 0:
            continue

        timestamp = current_frame / fps

        # Crop overhead view
        overhead = frame[OVERHEAD_Y_MIN:OVERHEAD_Y_MAX, :]

        # Run YOLO (only interested in robot_red=1, robot_blue=2)
        results = model.predict(overhead, conf=MIN_CONFIDENCE, classes=[1, 2],
                                verbose=False, imgsz=960)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                alliance = "red" if cls_id == 1 else "blue"

                detections.append(RobotPosition(
                    frame_number=current_frame,
                    timestamp=timestamp,
                    alliance=alliance,
                    center_x=cx,
                    center_y=cy,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                ))

        # Simple IoU/distance-based tracking
        matched_track_ids = set()
        for det in detections:
            best_track_id = None
            best_dist = float('inf')

            for tid, last_pos in active_tracks.items():
                if last_pos.alliance != det.alliance:
                    continue
                dist = np.sqrt((last_pos.center_x - det.center_x)**2 +
                               (last_pos.center_y - det.center_y)**2)
                # Allow larger movement between frames based on sample rate
                max_dist = 80 * sample_rate
                if dist < max_dist and dist < best_dist and tid not in matched_track_ids:
                    best_dist = dist
                    best_track_id = tid

            if best_track_id is not None:
                det.track_id = best_track_id
                tracks[best_track_id].positions.append(det)
                active_tracks[best_track_id] = det
                matched_track_ids.add(best_track_id)
            else:
                # New track
                det.track_id = next_track_id
                tracks[next_track_id] = RobotTrack(
                    track_id=next_track_id,
                    alliance=det.alliance,
                    positions=[det],
                )
                active_tracks[next_track_id] = det
                matched_track_ids.add(next_track_id)
                next_track_id += 1

        # Prune stale tracks (not seen for 2s)
        stale_ids = [
            tid for tid, pos in active_tracks.items()
            if timestamp - pos.timestamp > 2.0 and tid not in matched_track_ids
        ]
        for tid in stale_ids:
            del active_tracks[tid]

        processed += 1
        if processed % 200 == 0:
            print(f"    Frame {current_frame}/{total_frames} ({timestamp:.1f}s) - "
                  f"{len(active_tracks)} active tracks, {len(tracks)} total")

    cap.release()

    # Filter out very short tracks (noise)
    min_track_len = 5
    tracks = {tid: t for tid, t in tracks.items() if len(t.positions) >= min_track_len}

    # Consolidate: keep top-3 longest tracks per alliance (3 robots per alliance)
    for alliance in ["red", "blue"]:
        alliance_tracks = [(tid, t) for tid, t in tracks.items() if t.alliance == alliance]
        alliance_tracks.sort(key=lambda x: len(x[1].positions), reverse=True)
        # Keep top 3
        keep_ids = {tid for tid, _ in alliance_tracks[:3]}
        for tid, t in alliance_tracks[3:]:
            # Merge short tracks into nearest long track
            if t.positions:
                nearest_id = None
                nearest_dist = float('inf')
                for kid in keep_ids:
                    kt = tracks[kid]
                    if kt.last_position and t.positions[0]:
                        d = np.sqrt((kt.last_position.center_x - t.positions[0].center_x)**2 +
                                    (kt.last_position.center_y - t.positions[0].center_y)**2)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_id = kid
                if nearest_id is not None:
                    tracks[nearest_id].positions.extend(t.positions)
                    tracks[nearest_id].positions.sort(key=lambda p: p.timestamp)
            del tracks[tid]

    print(f"  Tracking complete: {len(tracks)} tracks ({sum(1 for t in tracks.values() if t.alliance=='red')} red, "
          f"{sum(1 for t in tracks.values() if t.alliance=='blue')} blue)")
    for tid, t in sorted(tracks.items()):
        print(f"    Track #{tid} ({t.alliance}): {len(t.positions)} positions, "
              f"{t.positions[0].timestamp:.1f}s - {t.positions[-1].timestamp:.1f}s")

    return tracks


# ────────────────────────────────────────────────────────────
# Stage 3: Scoring Attribution
# ────────────────────────────────────────────────────────────

def attribute_scores(
    events: List[ScoringEvent],
    tracks: Dict[int, RobotTrack],
    team_mapping: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Attribute scoring events to individual robots."""

    results = {
        "per_track": defaultdict(lambda: {"alliance": "", "scores": 0, "high_conf": 0, "events": []}),
        "unattributed": 0,
        "total_events": len(events),
        "attributions": [],
    }

    for event in events:
        # Find candidate robots: same alliance, near THEIR hub, at the right time
        hub = BLUE_HUB if event.alliance == "blue" else RED_HUB
        candidates = []

        for tid, track in tracks.items():
            if track.alliance != event.alliance:
                continue

            # Find positions within time window
            nearby_positions = [
                p for p in track.positions
                if abs(p.timestamp - event.timestamp) <= TIME_WINDOW_S
            ]

            if not nearby_positions:
                continue

            # Closest position in time
            closest = min(nearby_positions, key=lambda p: abs(p.timestamp - event.timestamp))

            # Distance to THIS ALLIANCE's hub
            dist = np.sqrt((closest.center_x - hub[0])**2 +
                           (closest.center_y - hub[1])**2)

            if dist > PROXIMITY_RADIUS_PX:
                continue

            # Trajectory: is robot approaching their hub?
            traj_score = 0.0
            before = [p for p in nearby_positions if p.timestamp <= event.timestamp]
            if len(before) >= 2:
                d_start = np.sqrt((before[0].center_x - hub[0])**2 +
                                  (before[0].center_y - hub[1])**2)
                d_end = np.sqrt((before[-1].center_x - hub[0])**2 +
                                (before[-1].center_y - hub[1])**2)
                if d_start > 0:
                    traj_score = (d_start - d_end) / d_start  # positive = approaching

            # Proximity score (0-1, higher = closer)
            prox_score = 1.0 - (dist / PROXIMITY_RADIUS_PX)

            # Combined
            combined = prox_score * 0.7 + max(0, traj_score) * 0.3

            candidates.append({
                "track_id": tid,
                "distance": dist,
                "proximity_score": prox_score,
                "trajectory_score": traj_score,
                "combined_score": combined,
                "confidence": closest.confidence,
            })

        if not candidates:
            # Fallback: find nearest robot of same alliance anywhere on field
            # (handles cases where all robots are far from hub — e.g., human player scoring)
            fallback_candidates = []
            for tid, track in tracks.items():
                if track.alliance != event.alliance:
                    continue
                nearby_positions = [
                    p for p in track.positions
                    if abs(p.timestamp - event.timestamp) <= TIME_WINDOW_S * 2  # wider window
                ]
                if not nearby_positions:
                    continue
                closest = min(nearby_positions, key=lambda p: abs(p.timestamp - event.timestamp))
                dist = np.sqrt((closest.center_x - hub[0])**2 +
                               (closest.center_y - hub[1])**2)
                fallback_candidates.append({
                    "track_id": tid,
                    "distance": dist,
                    "proximity_score": max(0, 1.0 - dist / 500),
                    "trajectory_score": 0.0,
                    "combined_score": max(0, 1.0 - dist / 500) * 0.3,  # low confidence
                    "confidence": closest.confidence,
                })

            if fallback_candidates:
                fallback_candidates.sort(key=lambda c: c["distance"])
                best = fallback_candidates[0]
                tid = best["track_id"]
                results["per_track"][tid]["alliance"] = tracks[tid].alliance
                results["per_track"][tid]["scores"] += 1
                results["per_track"][tid]["events"].append(event.timestamp)
                results["attributions"].append({
                    "timestamp": event.timestamp,
                    "alliance": event.alliance,
                    "score_change": f"{event.old_score}->{event.new_score}",
                    "attributed_to": tid,
                    "combined_score": round(best["combined_score"], 3),
                    "distance_px": round(best["distance"], 1),
                    "n_candidates": len(fallback_candidates),
                    "fallback": True,
                })
                continue

            results["unattributed"] += 1
            results["attributions"].append({
                "timestamp": event.timestamp,
                "alliance": event.alliance,
                "score_change": f"{event.old_score}->{event.new_score}",
                "attributed_to": None,
                "reason": "no_candidates",
            })
            continue

        # Best candidate
        candidates.sort(key=lambda c: c["combined_score"], reverse=True)
        best = candidates[0]
        tid = best["track_id"]

        is_high_conf = best["combined_score"] > 0.4 and len(candidates) <= 3

        results["per_track"][tid]["alliance"] = tracks[tid].alliance
        results["per_track"][tid]["scores"] += 1
        if is_high_conf:
            results["per_track"][tid]["high_conf"] += 1
        results["per_track"][tid]["events"].append(event.timestamp)

        results["attributions"].append({
            "timestamp": event.timestamp,
            "alliance": event.alliance,
            "score_change": f"{event.old_score}->{event.new_score}",
            "attributed_to": tid,
            "combined_score": round(best["combined_score"], 3),
            "distance_px": round(best["distance"], 1),
            "n_candidates": len(candidates),
        })

    return dict(results)


# ────────────────────────────────────────────────────────────
# Pipeline Orchestration
# ────────────────────────────────────────────────────────────

WEEK0_MATCHES = [
    {
        "video": "week0_f1m1.mp4",
        "tba_red": 263, "tba_blue": 187,
        "red_teams": ["2342", "6328", "2877"],
        "blue_teams": ["1768", "4909", "5962"],
    },
    {
        "video": "week0_f1m2.mp4",
        "tba_red": 77, "tba_blue": 253,
        "red_teams": ["2342", "6328", "2877"],
        "blue_teams": ["5962", "1768", "4909"],
    },
    {
        "video": "week0_f1m3.mp4",
        "tba_red": 208, "tba_blue": 185,
        "red_teams": ["2342", "6328", "2877"],
        "blue_teams": ["9644", "1768", "4909"],
    },
]


def run_match(match_config: dict) -> dict:
    """Run full pipeline on a single match."""
    video_path = str(PROJECT_ROOT / "data" / "videos" / "2026" / match_config["video"])
    video_name = match_config["video"].replace(".mp4", "")

    print(f"\n{'='*60}")
    print(f"Processing: {match_config['video']}")
    print(f"  TBA: Red {match_config['tba_red']}, Blue {match_config['tba_blue']}")
    print(f"  Red: {', '.join(match_config['red_teams'])}")
    print(f"  Blue: {', '.join(match_config['blue_teams'])}")
    print(f"{'='*60}")

    t0 = time.time()

    # Stage 1: OCR
    print("\n[Stage 1] Scoreboard OCR...")
    events, ocr_validation = run_ocr(video_path, match_config["tba_red"], match_config["tba_blue"])

    # Stage 2: Robot tracking
    print("\n[Stage 2] YOLO Robot Tracking...")
    tracks = run_robot_tracking(video_path, sample_rate=3)

    # Stage 3: Attribution
    print("\n[Stage 3] Score Attribution...")
    attribution = attribute_scores(events, tracks)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'─'*40}")
    print(f"RESULTS: {match_config['video']} ({elapsed:.1f}s)")
    print(f"{'─'*40}")
    print(f"  OCR Accuracy: Red {ocr_validation['red_accuracy']:.1%}, Blue {ocr_validation['blue_accuracy']:.1%}")
    print(f"  Scoring Events: {attribution['total_events']}")
    print(f"  Attributed: {attribution['total_events'] - attribution['unattributed']}")
    print(f"  Unattributed: {attribution['unattributed']}")
    print(f"\n  Per-Robot Scores:")

    for tid in sorted(attribution["per_track"].keys()):
        info = attribution["per_track"][tid]
        alliance = info["alliance"]
        scores = info["scores"]
        high = info["high_conf"]
        print(f"    Track #{tid} ({alliance}): {scores} balls ({high} high-conf)")

    # Validate per-alliance totals
    red_total = sum(v["scores"] for v in attribution["per_track"].values() if v["alliance"] == "red")
    blue_total = sum(v["scores"] for v in attribution["per_track"].values() if v["alliance"] == "blue")
    print(f"\n  Alliance Totals (attributed): Red {red_total}, Blue {blue_total}")
    print(f"  TBA Ground Truth:             Red {match_config['tba_red']}, Blue {match_config['tba_blue']}")

    # Save results
    output_dir = PROJECT_ROOT / "data" / "results" / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_dir / "attribution.json", "w") as f:
        serializable = {
            "match": match_config,
            "ocr_validation": ocr_validation,
            "per_track": dict(attribution["per_track"]),
            "unattributed": attribution["unattributed"],
            "total_events": attribution["total_events"],
            "elapsed_seconds": elapsed,
        }
        json.dump(serializable, f, indent=2, cls=NumpyEncoder)

    with open(output_dir / "attributions_detail.jsonl", "w") as f:
        for a in attribution["attributions"]:
            f.write(json.dumps(a, cls=NumpyEncoder) + "\n")

    return {
        "match": match_config["video"],
        "ocr_validation": ocr_validation,
        "attribution": attribution,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="FRC Vision Scoring Pipeline")
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--tba-red", type=int, help="TBA red hub score")
    parser.add_argument("--tba-blue", type=int, help="TBA blue hub score")
    parser.add_argument("--red-teams", type=str, help="Red teams (comma-separated)")
    parser.add_argument("--blue-teams", type=str, help="Blue teams (comma-separated)")
    parser.add_argument("--all", action="store_true", help="Run all Week 0 matches")
    parser.add_argument("--match", type=int, choices=[1, 2, 3], help="Run specific match (1-3)")
    args = parser.parse_args()

    if args.all:
        all_results = []
        for match in WEEK0_MATCHES:
            result = run_match(match)
            all_results.append(result)

        print(f"\n{'='*60}")
        print("ALL MATCHES SUMMARY")
        print(f"{'='*60}")
        for r in all_results:
            v = r["ocr_validation"]
            a = r["attribution"]
            print(f"  {r['match']}: OCR Red {v['red_accuracy']:.0%}/Blue {v['blue_accuracy']:.0%}, "
                  f"{a['total_events'] - a['unattributed']}/{a['total_events']} attributed")

    elif args.match:
        match = WEEK0_MATCHES[args.match - 1]
        run_match(match)

    elif args.video:
        match_config = {
            "video": Path(args.video).name,
            "tba_red": args.tba_red or 0,
            "tba_blue": args.tba_blue or 0,
            "red_teams": (args.red_teams or "").split(","),
            "blue_teams": (args.blue_teams or "").split(","),
        }
        run_match(match_config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

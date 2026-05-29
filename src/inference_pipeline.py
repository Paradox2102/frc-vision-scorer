"""FRC Vision Scorer - Live Inference Pipeline

Watches a video (local file or YouTube URL) and tracks:
  1. Yellow fuel balls (YOLO detector)
  2. Ball trajectories (ByteTrack)
  3. Scoring events (ball entering hub zone)
  4. Per-robot attribution (nearest robot to scored ball)

Usage:
  python src/inference_pipeline.py --source path/to/video.mp4
  python src/inference_pipeline.py --source "https://youtu.be/..."
  python src/inference_pipeline.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "train" / "frc_v2" / "weights" / "best.pt"
FALLBACK_MODEL = PROJECT_ROOT / "runs" / "train" / "frc_v1" / "weights" / "best.pt"


@dataclass
class ScoringZone:
    """Rectangular region where balls are considered 'scored'."""
    name: str
    alliance: str  # "red" or "blue"
    x1: float  # normalized 0..1
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class TrackedBall:
    """A ball being tracked across frames."""
    track_id: int
    positions: list[tuple[float, float]] = field(default_factory=list)
    scored: bool = False
    scored_zone: Optional[str] = None
    last_seen_frame: int = 0


@dataclass
class MatchStats:
    """Running statistics for the match."""
    frame_count: int = 0
    balls_detected: int = 0
    balls_scored: int = 0
    scoring_events: list[dict] = field(default_factory=list)
    fps: float = 0.0


class FRCVisionScorer:
    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        show_display: bool = True,
        save_output: str | None = None,
        scoring_zones: list[ScoringZone] | None = None,
    ):
        # Find best available model
        if model_path is None:
            if DEFAULT_MODEL.exists():
                model_path = DEFAULT_MODEL
            elif FALLBACK_MODEL.exists():
                model_path = FALLBACK_MODEL
            else:
                raise FileNotFoundError("No trained model found. Run training first.")

        print(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.show_display = show_display
        self.save_output = save_output
        self.scoring_zones = scoring_zones or []

        # Tracking
        if HAS_SUPERVISION:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            self.annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        else:
            self.tracker = None
            print("Warning: supervision not installed, tracking disabled")

        self.tracked_balls: dict[int, TrackedBall] = {}
        self.stats = MatchStats()

    def _resolve_source(self, source: str) -> str | int:
        """Resolve source to a path or stream URL."""
        if source.isdigit():
            return int(source)

        if "youtu" in source or "youtube" in source:
            try:
                import subprocess
                result = subprocess.run(
                    ["yt-dlp", "-f", "best[height<=720]", "-g", source],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    url = result.stdout.strip()
                    print(f"Resolved YouTube URL: {url[:80]}...")
                    return url
            except Exception as e:
                print(f"Warning: Could not resolve YouTube URL: {e}")

        return source

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process a single frame and return annotated version."""
        h_img, w_img = frame.shape[:2]

        # Run YOLO detection
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections_count = 0

        annotated = frame.copy()

        if HAS_SUPERVISION and self.tracker and results:
            # Convert to supervision detections
            sv_dets = sv.Detections.from_ultralytics(results[0])

            if len(sv_dets) > 0:
                # Track
                sv_dets = self.tracker.update_with_detections(sv_dets)
                detections_count = len(sv_dets)

                # Update tracked balls
                for i in range(len(sv_dets)):
                    x1, y1, x2, y2 = sv_dets.xyxy[i]
                    track_id = int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else i
                    cx = ((x1 + x2) / 2.0) / w_img
                    cy = ((y1 + y2) / 2.0) / h_img

                    if track_id not in self.tracked_balls:
                        self.tracked_balls[track_id] = TrackedBall(track_id=track_id)

                    ball = self.tracked_balls[track_id]
                    ball.positions.append((cx, cy))
                    ball.last_seen_frame = frame_idx

                    # Check scoring zones
                    if not ball.scored:
                        for zone in self.scoring_zones:
                            if zone.contains(cx, cy):
                                ball.scored = True
                                ball.scored_zone = zone.name
                                self.stats.balls_scored += 1
                                self.stats.scoring_events.append({
                                    "frame": frame_idx,
                                    "track_id": track_id,
                                    "zone": zone.name,
                                    "alliance": zone.alliance,
                                    "position": (cx, cy),
                                })
                                break

                # Annotate
                labels = [f"#{int(tid)}" for tid in sv_dets.tracker_id] if sv_dets.tracker_id is not None else []
                annotated = self.annotator.annotate(annotated, sv_dets)
                if labels:
                    annotated = self.label_annotator.annotate(annotated, sv_dets, labels)
        else:
            # No tracking, just draw boxes
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    detections_count += 1

        # Draw scoring zones
        for zone in self.scoring_zones:
            zx1 = int(zone.x1 * w_img)
            zy1 = int(zone.y1 * h_img)
            zx2 = int(zone.x2 * w_img)
            zy2 = int(zone.y2 * h_img)
            color = (0, 0, 255) if zone.alliance == "red" else (255, 0, 0)
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), color, 2)
            cv2.putText(annotated, zone.name, (zx1, zy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # HUD overlay
        self.stats.frame_count = frame_idx
        self.stats.balls_detected = detections_count
        hud_y = 30
        cv2.putText(annotated, f"Frame: {frame_idx}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        hud_y += 25
        cv2.putText(annotated, f"Balls: {detections_count}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        hud_y += 25
        cv2.putText(annotated, f"Scored: {self.stats.balls_scored}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        hud_y += 25
        cv2.putText(annotated, f"FPS: {self.stats.fps:.1f}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def run(self, source: str, max_frames: int | None = None):
        """Run inference on a video source."""
        resolved = self._resolve_source(source)
        cap = cv2.VideoCapture(resolved if isinstance(resolved, str) else resolved)

        if not cap.isOpened():
            print(f"Error: Could not open source: {source}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Source: {source}")
        print(f"Resolution: {w}x{h} @ {fps:.1f}fps")
        print(f"Total frames: {total_frames}")

        # Video writer
        writer = None
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.save_output, fourcc, fps, (w, h))
            print(f"Saving output to: {self.save_output}")

        frame_idx = 0
        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                annotated = self.process_frame(frame, frame_idx)

                # Update FPS
                elapsed = time.time() - t_start
                self.stats.fps = frame_idx / max(0.01, elapsed)

                if writer:
                    writer.write(annotated)

                if self.show_display:
                    cv2.imshow("FRC Vision Scorer", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord(" "):
                        cv2.waitKey(0)  # Pause

                frame_idx += 1

                if frame_idx % 100 == 0:
                    print(f"  Frame {frame_idx}/{total_frames}, "
                          f"balls={self.stats.balls_detected}, "
                          f"scored={self.stats.balls_scored}, "
                          f"fps={self.stats.fps:.1f}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

        # Print summary
        print(f"\n{'='*60}")
        print(f"MATCH SUMMARY")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_idx}")
        print(f"Average FPS: {self.stats.fps:.1f}")
        print(f"Unique balls tracked: {len(self.tracked_balls)}")
        print(f"Scoring events: {self.stats.balls_scored}")
        if self.stats.scoring_events:
            print(f"\nScoring timeline:")
            for evt in self.stats.scoring_events:
                t = evt["frame"] / fps
                print(f"  {t:.1f}s - Ball #{evt['track_id']} scored in {evt['zone']} ({evt['alliance']})")


def main():
    parser = argparse.ArgumentParser(description="FRC Vision Scorer - Live Inference")
    parser.add_argument("--source", required=True, help="Video file, YouTube URL, or camera index")
    parser.add_argument("--model", default=None, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", default=None, help="Save annotated output to file")
    parser.add_argument("--no-display", action="store_true", help="Don't show display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    scorer = FRCVisionScorer(
        model_path=args.model,
        conf_threshold=args.conf,
        show_display=not args.no_display,
        save_output=args.save,
    )
    scorer.run(args.source, max_frames=args.max_frames)


if __name__ == "__main__":
    main()

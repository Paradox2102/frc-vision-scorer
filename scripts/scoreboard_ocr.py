#!/usr/bin/env python3
"""
Scoreboard OCR Pipeline for FRC 2026 Week 0 Finals videos.

Extracts scoreboard data from match videos using OCR to track scoring in real-time.
Validates against TBA ground truth data.
"""

import cv2
import pytesseract
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class ScoreboardFrame:
    """Data extracted from a single frame."""
    frame_number: int
    timestamp: float
    blue_score: Optional[int]
    red_score: Optional[int]
    timer: Optional[str]
    
@dataclass
class ScoringEvent:
    """A detected scoring event (score increment)."""
    frame_number: int
    timestamp: float
    alliance: str  # "red" or "blue"
    cumulative_score: int


class ScoreboardOCR:
    """Extract and track scores from FRC match video scoreboards."""
    
    # Scoreboard region coordinates (top of frame, 1280x720 video)
    SCOREBOARD_HEIGHT = 80
    
    # Two score display locations:
    # 1) Corner indicators: far-left (blue) and far-right (red), ~30x30 digits
    # 2) Center bar: blue box | timer | red box at ~y=40-65
    #
    # Corner indicators are more reliable for OCR (isolated digits, consistent position)
    # Center bar useful as fallback
    
    # Format: (x, y, width, height)
    # Corner score indicators (hub-only scores — what we want)
    # Visually confirmed from multiple frames: blue at far-left, red at far-right
    BLUE_SCORE_REGION = (45, 35, 90, 45)      # Blue corner (far left, "172" etc)
    RED_SCORE_REGION = (1170, 35, 95, 45)     # Red corner (far right, "228" etc)
    
    # Center bar scores (fallback — shows TOTAL score, not hub-only)
    BLUE_SCORE_CENTER = (505, 50, 70, 35)     # Blue score in center bar
    RED_SCORE_CENTER = (670, 50, 70, 35)      # Red score in center bar
    TIMER_REGION = (585, 50, 80, 35)          # Timer in center
    
    def __init__(self, video_path: str, sample_rate: int = 5):
        """
        Initialize the OCR pipeline.
        
        Args:
            video_path: Path to the match video
            sample_rate: Process every Nth frame (default 5 = 6 fps for 30fps video)
        """
        self.video_path = Path(video_path)
        self.sample_rate = sample_rate
        self.frames_data: List[ScoreboardFrame] = []
        self.scoring_events: List[ScoringEvent] = []
        
    def preprocess_score_region(self, region: np.ndarray) -> np.ndarray:
        """
        Preprocess a score region for better OCR accuracy.
        
        Args:
            region: BGR image region
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR (scale up 3x)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # Threshold to get white text on black background
        # Scores are typically white/light text
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        return denoised
    
    def extract_number(self, text: str) -> Optional[int]:
        """
        Extract a number from OCR text.
        
        Args:
            text: Raw OCR output
            
        Returns:
            Extracted number or None if no valid number found
        """
        # Clean up text
        text = text.strip()
        
        # Find all numbers
        numbers = re.findall(r'\d+', text)
        
        if not numbers:
            return None
        
        # Take the first number found
        try:
            return int(numbers[0])
        except ValueError:
            return None
    
    def _ocr_region(self, scoreboard: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[int]:
        """OCR a single score region and return extracted number."""
        x, y, w, h = region
        # Clamp to scoreboard bounds
        y2 = min(y + h, scoreboard.shape[0])
        x2 = min(x + w, scoreboard.shape[1])
        crop = scoreboard[y:y2, x:x2]
        if crop.size == 0:
            return None
        processed = self.preprocess_score_region(crop)
        ocr_config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(processed, config=ocr_config)
        return self.extract_number(text)
    
    def extract_scoreboard_data(self, frame: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Extract blue score, red score, and timer from a frame.
        Uses corner indicators as primary, center bar as fallback.
        
        Args:
            frame: Full video frame
            
        Returns:
            Tuple of (blue_score, red_score, timer)
        """
        # Extract scoreboard region
        scoreboard = frame[0:self.SCOREBOARD_HEIGHT, :]
        
        # Try corner indicators first (more reliable)
        blue_score = self._ocr_region(scoreboard, self.BLUE_SCORE_REGION)
        red_score = self._ocr_region(scoreboard, self.RED_SCORE_REGION)
        
        # Fallback to center bar if corner fails
        if blue_score is None:
            blue_score = self._ocr_region(scoreboard, self.BLUE_SCORE_CENTER)
        if red_score is None:
            red_score = self._ocr_region(scoreboard, self.RED_SCORE_CENTER)
        
        # Timer
        timer_region = scoreboard[
            self.TIMER_REGION[1]:self.TIMER_REGION[1]+self.TIMER_REGION[3],
            self.TIMER_REGION[0]:self.TIMER_REGION[0]+self.TIMER_REGION[2]
        ]
        timer_processed = self.preprocess_score_region(timer_region)
        timer_config = '--psm 7 -c tessedit_char_whitelist=0123456789:'
        timer_text = pytesseract.image_to_string(timer_processed, config=timer_config)
        timer = timer_text.strip() if timer_text.strip() else None
        
        return blue_score, red_score, timer
    
    def process_video(self, debug: bool = False) -> None:
        """
        Process the entire video and extract scoreboard data.
        
        Args:
            debug: If True, save debug images for the first few frames
        """
        print(f"Processing video: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {fps} fps, {total_frames} frames")
        print(f"Sampling every {self.sample_rate} frames ({fps/self.sample_rate:.1f} samples/sec)")
        
        frame_num = 0
        last_blue_score = 0
        last_red_score = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_num % self.sample_rate != 0:
                frame_num += 1
                continue
            
            timestamp = frame_num / fps
            
            # Extract scoreboard data
            blue_score, red_score, timer = self.extract_scoreboard_data(frame)
            
            # Store frame data
            frame_data = ScoreboardFrame(
                frame_number=frame_num,
                timestamp=timestamp,
                blue_score=blue_score,
                red_score=red_score,
                timer=timer
            )
            self.frames_data.append(frame_data)
            
            # Monotonicity: scores can only increase. Reject OCR artifacts that
            # jump backwards or spike unreasonably (>20 pts in one sample interval).
            MAX_JUMP = 20  # max plausible score increase between samples
            
            if blue_score is not None:
                if blue_score < last_blue_score:
                    blue_score = None  # OCR error — reject
                elif blue_score > last_blue_score + MAX_JUMP:
                    blue_score = None  # implausible spike — reject
            
            if red_score is not None:
                if red_score < last_red_score:
                    red_score = None  # OCR error — reject
                elif red_score > last_red_score + MAX_JUMP:
                    red_score = None  # implausible spike — reject
            
            # Detect scoring events
            if blue_score is not None and blue_score > last_blue_score:
                self.scoring_events.append(ScoringEvent(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    alliance="blue",
                    cumulative_score=blue_score
                ))
                last_blue_score = blue_score
            
            if red_score is not None and red_score > last_red_score:
                self.scoring_events.append(ScoringEvent(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    alliance="red",
                    cumulative_score=red_score
                ))
                last_red_score = red_score
            
            # Debug output
            if debug and frame_num < 300:  # First 10 seconds
                scoreboard = frame[0:self.SCOREBOARD_HEIGHT, :]
                debug_dir = self.video_path.parent.parent / "results" / "week0_analysis" / "debug_ocr"
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / f"frame_{frame_num:05d}.jpg"), scoreboard)
            
            if frame_num % 300 == 0:  # Progress update every 10 seconds
                print(f"  Frame {frame_num}/{total_frames} ({timestamp:.1f}s) - Blue: {blue_score}, Red: {red_score}")
            
            frame_num += 1
        
        cap.release()
        print(f"Processed {len(self.frames_data)} frames")
        print(f"Detected {len(self.scoring_events)} scoring events")
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save extracted data to CSV and JSONL files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = self.video_path.stem
        
        # Save all frames data as CSV
        frames_csv = output_dir / f"{video_name}_frames.csv"
        with open(frames_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame_number', 'timestamp', 'blue_score', 'red_score', 'timer'])
            writer.writeheader()
            for frame_data in self.frames_data:
                writer.writerow(asdict(frame_data))
        
        print(f"Saved frames data to {frames_csv}")
        
        # Save scoring events as JSONL
        events_jsonl = output_dir / f"{video_name}_scoring_events.jsonl"
        with open(events_jsonl, 'w') as f:
            for event in self.scoring_events:
                f.write(json.dumps(asdict(event)) + '\n')
        
        print(f"Saved scoring events to {events_jsonl}")
        
    def validate_against_tba(self, tba_red: int, tba_blue: int) -> Dict:
        """
        Validate final scores against TBA ground truth.
        
        Args:
            tba_red: Expected red alliance final score
            tba_blue: Expected blue alliance final score
            
        Returns:
            Dictionary with validation results
        """
        # Get final scores from last valid frame
        final_blue = None
        final_red = None
        
        for frame_data in reversed(self.frames_data):
            if frame_data.blue_score is not None:
                final_blue = frame_data.blue_score
                break
        
        for frame_data in reversed(self.frames_data):
            if frame_data.red_score is not None:
                final_red = frame_data.red_score
                break
        
        results = {
            "video": self.video_path.name,
            "tba_red": tba_red,
            "tba_blue": tba_blue,
            "ocr_red": final_red,
            "ocr_blue": final_blue,
            "red_diff": (final_red - tba_red) if final_red else None,
            "blue_diff": (final_blue - tba_blue) if final_blue else None,
            "red_accuracy": (1 - abs(final_red - tba_red) / tba_red) if final_red else None,
            "blue_accuracy": (1 - abs(final_blue - tba_blue) / tba_blue) if final_blue else None,
        }
        
        return results


def main():
    """Process all three Week 0 Finals videos."""
    
    base_dir = Path(__file__).parent.parent
    videos_dir = base_dir / "data" / "videos" / "2026"
    output_dir = base_dir / "data" / "results" / "week0_analysis"
    
    # TBA ground truth — corner indicators show HUB-ONLY scores
    # Hub scores: F1M1 Red=263 (45 auto+218 teleop), Blue=187 (33 auto+154 teleop)
    #             F1M2 Red=77 (21 auto+56 teleop), Blue=253 (46 auto+207 teleop)
    #             F1M3 Red=208 (38 auto+170 teleop), Blue=185 (36 auto+149 teleop)
    tba_data = {
        "week0_f1m1": {"red": 263, "blue": 187},
        "week0_f1m2": {"red": 77, "blue": 253},
        "week0_f1m3": {"red": 208, "blue": 185},
    }
    
    validation_results = []
    
    for video_name, tba_scores in tba_data.items():
        video_path = videos_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {video_name}")
        print(f"{'='*60}")
        
        # Process video
        ocr = ScoreboardOCR(str(video_path), sample_rate=5)
        ocr.process_video(debug=(video_name == "week0_f1m1"))  # Debug first video only
        
        # Save results
        ocr.save_results(output_dir)
        
        # Validate
        validation = ocr.validate_against_tba(tba_scores["red"], tba_scores["blue"])
        validation_results.append(validation)
        
        print(f"\nValidation Results:")
        print(f"  TBA:  Red={tba_scores['red']}, Blue={tba_scores['blue']}")
        print(f"  OCR:  Red={validation['ocr_red']}, Blue={validation['ocr_blue']}")
        print(f"  Diff: Red={validation['red_diff']}, Blue={validation['blue_diff']}")
        if validation['red_accuracy'] is not None:
            print(f"  Accuracy: Red={validation['red_accuracy']*100:.1f}%, Blue={validation['blue_accuracy']*100:.1f}%")
    
    # Save validation summary
    validation_file = output_dir / "validation_summary.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All videos processed. Validation summary saved to {validation_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

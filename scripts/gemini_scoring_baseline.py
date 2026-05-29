#!/usr/bin/env python3
"""
Gemini Vision Baseline for FRC 2026 Week 0 Finals Scoring.

Extracts key moments from match videos and uses Gemini 2.0 Flash to estimate
scoring activity. Compares against scoreboard OCR data for validation.
"""

import os
import json
import time
import cv2
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ClipAnalysis:
    """Results from analyzing a video clip."""
    video: str
    period: str  # "auto", "teleop_early", "teleop_mid", "teleop_late", "endgame"
    start_time: float
    end_time: float
    gemini_response: str
    estimated_red_scored: int
    estimated_blue_scored: int


class GeminiScoringBaseline:
    """Use Gemini 2.0 Flash to estimate scoring from video clips."""
    
    # Match periods (approximate timings for a 150s match)
    PERIODS = {
        "auto": (0, 15),           # First 15 seconds
        "teleop_early": (15, 45),   # 15-45s
        "teleop_mid": (45, 90),     # 45-90s
        "teleop_late": (90, 135),   # 90-135s
        "endgame": (135, 150),      # Last 15 seconds
    }
    
    def __init__(self, video_path: str, model: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini baseline analyzer.
        
        Args:
            video_path: Path to the match video
            model: Gemini model to use
        """
        self.video_path = Path(video_path)
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.analyses: List[ClipAnalysis] = []
        
    def extract_clip(self, start_sec: float, end_sec: float, output_path: str) -> None:
        """
        Extract a video clip.
        
        Args:
            start_sec: Start time in seconds
            end_sec: End time in seconds
            output_path: Where to save the clip
        """
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
    
    def extract_frames_from_clip(self, clip_path: str, num_frames: int = 5) -> List[str]:
        """
        Extract evenly-spaced frames from a clip.
        
        Args:
            clip_path: Path to the clip
            num_frames: Number of frames to extract
            
        Returns:
            List of paths to extracted frame images
        """
        cap = cv2.VideoCapture(clip_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices
        frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
        
        frame_paths = []
        temp_dir = Path(tempfile.mkdtemp())
        
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_path = temp_dir / f"frame_{idx:02d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
        
        cap.release()
        return frame_paths
    
    def analyze_frames_with_gemini(self, frame_paths: List[str], period: str) -> Tuple[str, int, int]:
        """
        Send frames to Gemini for analysis.
        
        Args:
            frame_paths: Paths to frame images
            period: Match period name
            
        Returns:
            Tuple of (full_response, estimated_red_scored, estimated_blue_scored)
        """
        prompt = f"""
You are analyzing frames from a 2026 FRC (FIRST Robotics Competition) match.

CONTEXT:
- This is the "{period}" period of the match
- Yellow fuel balls are scored into scoring structures (hubs) for points
- Each ball scored = +1 point
- Red alliance (3 robots) competes against Blue alliance (3 robots)
- Robots can be identified by bumper colors: red bumpers = red alliance, blue bumpers = blue alliance

YOUR TASK:
Analyze these {len(frame_paths)} frames taken from a 10-second clip during the {period} period.

Count how many yellow fuel balls you can see being scored into the hubs by each alliance.

IMPORTANT:
- Only count balls you can clearly see entering a scoring structure
- Do not count balls being held, transported, or missed shots
- Be conservative - if you're unsure, don't count it
- Distinguish between red alliance robots (red bumpers) and blue alliance robots (blue bumpers)

Respond in the following format:

RED ALLIANCE SCORED: [number]
BLUE ALLIANCE SCORED: [number]
CONFIDENCE: [low/medium/high]

REASONING: [Explain what you observed in detail. Describe the robot activity, ball movement, 
and any scoring you detected. Be specific about which frames show scoring.]
"""
        
        # Build content parts
        parts = [types.Part.from_text(text=prompt)]
        
        for frame_path in frame_paths:
            with open(frame_path, 'rb') as f:
                image_data = f.read()
            parts.append(types.Part.from_bytes(data=image_data, mime_type="image/jpeg"))
        
        print(f"    Analyzing {len(frame_paths)} frames with {self.model}...")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2000,
            )
        )
        
        response_text = response.text
        
        # Parse response for estimates
        red_scored = 0
        blue_scored = 0
        
        for line in response_text.split('\n'):
            line_upper = line.upper()
            if 'RED ALLIANCE SCORED:' in line_upper:
                try:
                    red_scored = int(''.join(filter(str.isdigit, line)))
                except:
                    pass
            elif 'BLUE ALLIANCE SCORED:' in line_upper:
                try:
                    blue_scored = int(''.join(filter(str.isdigit, line)))
                except:
                    pass
        
        return response_text, red_scored, blue_scored
    
    def analyze_match(self) -> None:
        """Analyze all key periods of the match."""
        print(f"\nAnalyzing match: {self.video_path.name}")
        print("="*60)
        
        temp_dir = Path(tempfile.mkdtemp())
        
        for period_name, (start, end) in self.PERIODS.items():
            print(f"\n{period_name.upper()} ({start}s - {end}s):")
            
            # Extract clip
            clip_path = temp_dir / f"{period_name}.mp4"
            print(f"  Extracting clip...")
            self.extract_clip(start, end, str(clip_path))
            
            # Extract frames
            print(f"  Extracting frames...")
            frame_paths = self.extract_frames_from_clip(str(clip_path), num_frames=5)
            
            # Analyze with Gemini
            response, red_scored, blue_scored = self.analyze_frames_with_gemini(frame_paths, period_name)
            
            # Store results
            analysis = ClipAnalysis(
                video=self.video_path.name,
                period=period_name,
                start_time=start,
                end_time=end,
                gemini_response=response,
                estimated_red_scored=red_scored,
                estimated_blue_scored=blue_scored
            )
            self.analyses.append(analysis)
            
            print(f"  Gemini estimates: Red={red_scored}, Blue={blue_scored}")
            
            # Clean up frames
            for frame_path in frame_paths:
                Path(frame_path).unlink()
        
        print("\n" + "="*60)
    
    def save_results(self, output_dir: Path) -> None:
        """Save analysis results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = self.video_path.stem
        output_file = output_dir / f"{video_name}_gemini_baseline.json"
        
        # Convert to dict
        results = {
            "video": self.video_path.name,
            "model": self.model,
            "analyses": [
                {
                    "period": a.period,
                    "start_time": a.start_time,
                    "end_time": a.end_time,
                    "estimated_red_scored": a.estimated_red_scored,
                    "estimated_blue_scored": a.estimated_blue_scored,
                    "gemini_response": a.gemini_response
                }
                for a in self.analyses
            ],
            "totals": {
                "red": sum(a.estimated_red_scored for a in self.analyses),
                "blue": sum(a.estimated_blue_scored for a in self.analyses)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def compare_with_ocr(self, ocr_data_path: Path) -> Dict:
        """
        Compare Gemini estimates with OCR data.
        
        Args:
            ocr_data_path: Path to OCR scoring events JSONL
            
        Returns:
            Comparison results
        """
        # Load OCR events
        ocr_events = []
        if ocr_data_path.exists():
            with open(ocr_data_path) as f:
                ocr_events = [json.loads(line) for line in f]
        
        # Count scores by period
        ocr_by_period = {period: {"red": 0, "blue": 0} for period in self.PERIODS.keys()}
        
        for event in ocr_events:
            timestamp = event["timestamp"]
            alliance = event["alliance"]
            
            # Find which period this belongs to
            for period_name, (start, end) in self.PERIODS.items():
                if start <= timestamp < end:
                    ocr_by_period[period_name][alliance] += 1
                    break
        
        # Compare with Gemini estimates
        comparison = {
            "video": self.video_path.name,
            "periods": []
        }
        
        for analysis in self.analyses:
            period_name = analysis.period
            ocr_red = ocr_by_period[period_name]["red"]
            ocr_blue = ocr_by_period[period_name]["blue"]
            
            comparison["periods"].append({
                "period": period_name,
                "ocr_red": ocr_red,
                "ocr_blue": ocr_blue,
                "gemini_red": analysis.estimated_red_scored,
                "gemini_blue": analysis.estimated_blue_scored,
                "red_diff": analysis.estimated_red_scored - ocr_red,
                "blue_diff": analysis.estimated_blue_scored - ocr_blue
            })
        
        return comparison


def main():
    """Run Gemini baseline analysis on all three Week 0 Finals videos."""
    
    base_dir = Path(__file__).parent.parent
    videos_dir = base_dir / "data" / "videos" / "2026"
    output_dir = base_dir / "data" / "results" / "week0_analysis"
    
    videos = [
        "week0_f1m1.mp4",
        "week0_f1m2.mp4",
        "week0_f1m3.mp4",
    ]
    
    all_comparisons = []
    
    print("="*60)
    print("Gemini Vision Baseline for FRC 2026 Week 0 Finals")
    print("="*60)
    
    for video_name in videos:
        video_path = videos_dir / video_name
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue
        
        # Run Gemini analysis
        analyzer = GeminiScoringBaseline(str(video_path))
        analyzer.analyze_match()
        analyzer.save_results(output_dir)
        
        # Compare with OCR if available
        ocr_events_path = output_dir / f"{video_path.stem}_scoring_events.jsonl"
        if ocr_events_path.exists():
            comparison = analyzer.compare_with_ocr(ocr_events_path)
            all_comparisons.append(comparison)
            
            print(f"\nComparison with OCR:")
            for period_data in comparison["periods"]:
                print(f"  {period_data['period']:15s} - OCR: R={period_data['ocr_red']:2d} B={period_data['ocr_blue']:2d} | "
                      f"Gemini: R={period_data['gemini_red']:2d} B={period_data['gemini_blue']:2d} | "
                      f"Diff: R={period_data['red_diff']:+3d} B={period_data['blue_diff']:+3d}")
        else:
            print(f"\nNo OCR data found at {ocr_events_path}")
            print("Run scoreboard_ocr.py first to generate comparison data.")
    
    # Save all comparisons
    if all_comparisons:
        comparison_file = output_dir / "gemini_ocr_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(all_comparisons, f, indent=2)
        print(f"\nAll comparisons saved to: {comparison_file}")
    
    print("\n" + "="*60)
    print("Gemini baseline analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()

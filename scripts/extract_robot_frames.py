#!/usr/bin/env python3
"""
Extract robot detection training frames from FRC 2026 Week 0 Finals videos.

Extracts the top 2/3 (overhead field view) from frames at 1fps for YOLO annotation.
"""

import cv2
import csv
from pathlib import Path
from typing import List, Dict

class RobotFrameExtractor:
    """Extract overhead field view frames for robot detection training."""
    
    # Frame dimensions
    FRAME_HEIGHT = 720
    FRAME_WIDTH = 1280
    
    # Extract top 2/3 (overhead view)
    OVERHEAD_HEIGHT = 480  # 2/3 of 720
    
    def __init__(self, video_path: str, output_dir: str, fps_sample: int = 1):
        """
        Initialize the frame extractor.
        
        Args:
            video_path: Path to the match video
            output_dir: Directory to save extracted frames
            fps_sample: Extract one frame per N seconds (default 1 = 1 frame/sec)
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.fps_sample = fps_sample
        self.extracted_frames: List[Dict] = []
        
    def extract_frames(self) -> None:
        """Extract overhead view frames from the video."""
        print(f"Processing video: {self.video_path.name}")
        
        # Create output directory
        video_dir = self.output_dir / self.video_path.stem
        video_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  Video: {fps:.1f} fps, {total_frames} frames, {duration:.1f}s duration")
        print(f"  Extracting 1 frame per {self.fps_sample} seconds")
        
        frame_interval = int(fps * self.fps_sample)
        expected_frames = int(duration / self.fps_sample)
        
        print(f"  Expected ~{expected_frames} frames")
        
        frame_num = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample at specified interval
            if frame_num % frame_interval == 0:
                timestamp = frame_num / fps
                
                # Extract overhead view (top 2/3)
                overhead_view = frame[0:self.OVERHEAD_HEIGHT, :]
                
                # Save frame
                frame_filename = f"{self.video_path.stem}_frame_{frame_num:05d}.jpg"
                frame_path = video_dir / frame_filename
                cv2.imwrite(str(frame_path), overhead_view)
                
                # Record metadata
                self.extracted_frames.append({
                    "video": self.video_path.name,
                    "frame_number": frame_num,
                    "timestamp": f"{timestamp:.2f}",
                    "filename": frame_filename,
                    "relative_path": f"{self.video_path.stem}/{frame_filename}"
                })
                
                extracted_count += 1
                
                if extracted_count % 20 == 0:
                    print(f"    Extracted {extracted_count} frames ({timestamp:.1f}s)")
            
            frame_num += 1
        
        cap.release()
        print(f"  Extracted {extracted_count} frames to {video_dir}")
    
    def get_metadata(self) -> List[Dict]:
        """Return metadata for extracted frames."""
        return self.extracted_frames


def main():
    """Extract frames from all three Week 0 Finals videos."""
    
    base_dir = Path(__file__).parent.parent
    videos_dir = base_dir / "data" / "videos" / "2026"
    output_dir = base_dir / "data" / "robot_training_frames"
    
    videos = [
        "week0_f1m1.mp4",
        "week0_f1m2.mp4",
        "week0_f1m3.mp4",
    ]
    
    all_metadata = []
    
    print("="*60)
    print("FRC Robot Detection Training Frame Extraction")
    print("="*60)
    
    for video_name in videos:
        video_path = videos_dir / video_name
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue
        
        print(f"\n{video_name}:")
        
        extractor = RobotFrameExtractor(
            video_path=str(video_path),
            output_dir=str(output_dir),
            fps_sample=1  # 1 frame per second
        )
        
        extractor.extract_frames()
        all_metadata.extend(extractor.get_metadata())
    
    # Save manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, 'w', newline='') as f:
        if all_metadata:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total frames extracted: {len(all_metadata)}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print(f"\nSummary by video:")
    video_counts = {}
    for entry in all_metadata:
        video = entry['video']
        video_counts[video] = video_counts.get(video, 0) + 1
    
    for video, count in video_counts.items():
        print(f"  {video}: {count} frames")
    
    print(f"\nFrames ready for YOLO annotation at: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract training frames from Week 0 Finals match videos for dataset v4.

Extracts:
- Overhead view (top 65%, y=0 to y=468) at 2fps
- Alliance views (bottom 35%, split left/right) at 2fps  
- Full frames at 0.5fps

Video specs: 1280x720, 30fps, ~172s each
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "2026"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset_v4" / "images"

# Video layout (1280x720)
OVERHEAD_CROP = (0, 0, 1280, 468)  # x, y, w, h - top 65%
ALLIANCE_LEFT_CROP = (0, 468, 640, 252)  # bottom left
ALLIANCE_RIGHT_CROP = (640, 468, 640, 252)  # bottom right

# Frame extraction rates (fps)
OVERHEAD_FPS = 2.0
ALLIANCE_FPS = 2.0
FULL_FPS = 0.5

MATCH_VIDEOS = [
    "week0_f1m1.mp4",
    "week0_f1m2.mp4",
    "week0_f1m3.mp4",
]


def extract_frames_from_video(video_path: Path, match_id: str):
    """Extract frames from a single match video."""
    print(f"\nProcessing: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"  Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {fps:.1f}fps")
    print(f"  Duration: {duration:.1f}s ({total_frames} frames)")
    
    # Calculate frame intervals
    overhead_interval = int(fps / OVERHEAD_FPS)
    alliance_interval = int(fps / ALLIANCE_FPS)
    full_interval = int(fps / FULL_FPS)
    
    # Expected counts
    expected_overhead = int(duration * OVERHEAD_FPS)
    expected_alliance = int(duration * ALLIANCE_FPS)
    expected_full = int(duration * FULL_FPS)
    
    print(f"  Extracting ~{expected_overhead} overhead, ~{expected_alliance} per alliance, ~{expected_full} full frames")
    
    frame_idx = 0
    overhead_count = 0
    alliance_left_count = 0
    alliance_right_count = 0
    full_count = 0
    
    with tqdm(total=total_frames, desc=f"  {match_id}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract overhead view
            if frame_idx % overhead_interval == 0:
                x, y, w, h = OVERHEAD_CROP
                crop = frame[y:y+h, x:x+w]
                out_path = OUTPUT_DIR / "overhead" / f"{match_id}_overhead_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                overhead_count += 1
            
            # Extract alliance views
            if frame_idx % alliance_interval == 0:
                # Left alliance
                x, y, w, h = ALLIANCE_LEFT_CROP
                crop = frame[y:y+h, x:x+w]
                out_path = OUTPUT_DIR / "alliance_left" / f"{match_id}_left_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                alliance_left_count += 1
                
                # Right alliance
                x, y, w, h = ALLIANCE_RIGHT_CROP
                crop = frame[y:y+h, x:x+w]
                out_path = OUTPUT_DIR / "alliance_right" / f"{match_id}_right_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                alliance_right_count += 1
            
            # Extract full frame
            if frame_idx % full_interval == 0:
                out_path = OUTPUT_DIR / "full" / f"{match_id}_full_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                full_count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"  Extracted: {overhead_count} overhead, {alliance_left_count} left, {alliance_right_count} right, {full_count} full")


def main():
    print("="*70)
    print("FRC Vision Scorer - Dataset v4 Frame Extraction")
    print("="*70)
    
    # Create output directories
    output_dirs = [
        OUTPUT_DIR / "overhead",
        OUTPUT_DIR / "alliance_left",
        OUTPUT_DIR / "alliance_right",
        OUTPUT_DIR / "full",
    ]
    
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Extraction rates: overhead={OVERHEAD_FPS}fps, alliance={ALLIANCE_FPS}fps, full={FULL_FPS}fps")
    
    # Process each match video
    total_overhead = 0
    total_alliance_left = 0
    total_alliance_right = 0
    total_full = 0
    
    for video_name in MATCH_VIDEOS:
        video_path = VIDEO_DIR / video_name
        if not video_path.exists():
            print(f"\nWARNING: Video not found: {video_path}")
            continue
        
        match_id = video_name.replace(".mp4", "")
        
        # Count before
        before_overhead = len(list((OUTPUT_DIR / "overhead").glob(f"{match_id}_*.jpg")))
        before_left = len(list((OUTPUT_DIR / "alliance_left").glob(f"{match_id}_*.jpg")))
        before_right = len(list((OUTPUT_DIR / "alliance_right").glob(f"{match_id}_*.jpg")))
        before_full = len(list((OUTPUT_DIR / "full").glob(f"{match_id}_*.jpg")))
        
        extract_frames_from_video(video_path, match_id)
        
        # Count after
        after_overhead = len(list((OUTPUT_DIR / "overhead").glob(f"{match_id}_*.jpg")))
        after_left = len(list((OUTPUT_DIR / "alliance_left").glob(f"{match_id}_*.jpg")))
        after_right = len(list((OUTPUT_DIR / "alliance_right").glob(f"{match_id}_*.jpg")))
        after_full = len(list((OUTPUT_DIR / "full").glob(f"{match_id}_*.jpg")))
        
        total_overhead += (after_overhead - before_overhead)
        total_alliance_left += (after_left - before_left)
        total_alliance_right += (after_right - before_right)
        total_full += (after_full - before_full)
    
    # Final summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total frames extracted:")
    print(f"  Overhead:      {total_overhead}")
    print(f"  Alliance left: {total_alliance_left}")
    print(f"  Alliance right: {total_alliance_right}")
    print(f"  Full frames:   {total_full}")
    print(f"  GRAND TOTAL:   {total_overhead + total_alliance_left + total_alliance_right + total_full}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

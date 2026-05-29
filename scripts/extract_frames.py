#!/usr/bin/env python3
"""Extract frames from FRC match videos for analysis."""

import cv2
import sys
import os
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, fps: int = 5, max_frames: int = 200):
    """Extract frames at specified FPS from a video."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {video_fps:.1f}, Frames: {total_frames}, Duration: {duration:.1f}s")
    print(f"  Extracting at {fps} fps (1 every {int(video_fps/fps)} frames)")
    
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"  Saved {saved_count} frames to {output_dir}")
    return saved_count


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    
    videos = [
        ("data/videos/2017/58ball_auto.mp4", "data/frames/2017_58ball", 5, 100),
        ("data/videos/2017/midwest_final1.mp4", "data/frames/2017_final", 3, 200),
        ("data/videos/2026/game_animation.mp4", "data/frames/2026_animation", 2, 100),
    ]
    
    for video, output, fps, max_f in videos:
        video_path = base / video
        output_path = base / output
        if video_path.exists():
            extract_frames(str(video_path), str(output_path), fps, max_f)
        else:
            print(f"Skipping {video} (not found)")

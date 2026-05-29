#!/usr/bin/env python3
"""Extract alliance-side crops covering hub to field edge.

Blue alliance: x=0 to x=640 (left half)
Red alliance: x=640 to x=1280 (right half)
Full overhead height (482px at 0.67 crop ratio)

Sample strategically: ~50 per alliance across all 3 videos.
"""

import cv2
import csv
import random
from pathlib import Path

PROJECT = Path(__file__).parent.parent
VIDEO_DIR = PROJECT / "data" / "videos" / "2026"
OUTPUT_DIR = PROJECT / "data" / "alliance_crops"
VIDEOS = ["week0_f1m1.mp4", "week0_f1m2.mp4", "week0_f1m3.mp4"]

OVERHEAD_CROP_RATIO = 0.67
SAMPLES_PER_ALLIANCE = 50  # 50 blue + 50 red = 100 total


def extract_crops():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in ["blue", "red"]:
        (OUTPUT_DIR / d).mkdir(exist_ok=True)

    # Also create labeling symlink
    labeling_link = PROJECT / "data" / "labeling" / "alliance_crops"
    if not labeling_link.exists():
        labeling_link.symlink_to(OUTPUT_DIR)

    all_frames = []  # (video_name, frame_idx, total_frames)

    # First pass: count frames per video
    for vname in VIDEOS:
        vpath = VIDEO_DIR / vname
        cap = cv2.VideoCapture(str(vpath))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        all_frames.append((vname, total, fps))
        print(f"{vname}: {total} frames, {fps} fps, {total/fps:.0f}s")

    # Total frames across all videos
    total_all = sum(t for _, t, _ in all_frames)

    # Distribute samples proportionally across videos
    # Then sample evenly across each video's duration
    metadata = []
    crop_count = {"blue": 0, "red": 0}

    for vname, total, fps in all_frames:
        vpath = VIDEO_DIR / vname
        # How many samples from this video (proportional)
        n_samples = max(1, round(SAMPLES_PER_ALLIANCE * total / total_all))
        
        # Evenly space frame indices across the video
        step = total // (n_samples + 1)
        frame_indices = [step * (i + 1) for i in range(n_samples)]
        
        print(f"\n{vname}: extracting {n_samples} samples (step={step})")

        cap = cv2.VideoCapture(str(vpath))
        h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        crop_h = int(h_full * OVERHEAD_CROP_RATIO)
        mid_x = w_full // 2

        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

            overhead = frame[:crop_h, :]

            # Blue alliance: left half (hub to left edge)
            blue_crop = overhead[:, :mid_x]
            bname = f"{vname.replace('.mp4', '')}_frame_{fidx:06d}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / "blue" / bname), blue_crop)
            crop_count["blue"] += 1
            metadata.append({
                "video": vname, "frame": fidx, "alliance": "blue",
                "crop_x": 0, "crop_y": 0, "crop_w": mid_x, "crop_h": crop_h,
                "filename": f"blue/{bname}"
            })

            # Red alliance: right half (hub to right edge)
            red_crop = overhead[:, mid_x:]
            rname = f"{vname.replace('.mp4', '')}_frame_{fidx:06d}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / "red" / rname), red_crop)
            crop_count["red"] += 1
            metadata.append({
                "video": vname, "frame": fidx, "alliance": "red",
                "crop_x": mid_x, "crop_y": 0, "crop_w": mid_x, "crop_h": crop_h,
                "filename": f"red/{rname}"
            })

        cap.release()

    # Write metadata
    with open(OUTPUT_DIR / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nDone: {crop_count['blue']} blue + {crop_count['red']} red = {sum(crop_count.values())} crops")
    print(f"Crop dimensions: {mid_x}x{crop_h} per alliance half")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_crops()

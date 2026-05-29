#!/usr/bin/env python3
"""Extract candidate shot windows from a match video.

Phase 1 objective:
- prioritize likely made-shot moments into the center container
- generate short clips + keyframes for efficient human annotation

Detection heuristic (baseline):
- monitor frame-difference motion energy inside a user-provided center-goal ROI
- keep local peaks over threshold, with cooldown-based de-duplication

Outputs:
- clips/*.mp4 (candidate windows)
- keyframes/*.jpg (center frame per window)
- candidates.jsonl (event metadata)
- manifest.csv (summary table)
- labelstudio_tasks.json (frame tasks for Label Studio)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import cv2
import numpy as np


@dataclass
class CandidateEvent:
    candidate_id: str
    video_id: str
    frame_index: int
    timestamp_s: float
    motion_score: float
    start_frame: int
    end_frame: int
    clip_path: str
    keyframe_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate shot windows from video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--roi",
        required=True,
        help="Center goal ROI as normalized x1,y1,x2,y2 (0..1)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/shot_candidates/single_run",
        help="Output directory for clips and manifests",
    )
    parser.add_argument("--window-before", type=float, default=1.2, help="Seconds before event")
    parser.add_argument("--window-after", type=float, default=1.0, help="Seconds after event")
    parser.add_argument("--cooldown", type=float, default=0.35, help="Min seconds between candidates")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="Absolute motion score threshold; if unset uses median + std-multiplier * std",
    )
    parser.add_argument(
        "--std-multiplier",
        type=float,
        default=2.0,
        help="Threshold multiplier for dynamic thresholding",
    )
    parser.add_argument("--max-candidates", type=int, default=200, help="Cap on number of candidates")
    parser.add_argument(
        "--local-files-prefix",
        default="/data/local-files/?d=",
        help="Label Studio local-files prefix for generated task image paths",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path used to build Label Studio relative paths (defaults to inferred git/project root)",
    )
    return parser.parse_args()


def parse_roi(roi_str: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must contain 4 comma-separated values: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return x1, y1, x2, y2


def iter_motion_scores(
    video_path: Path,
    roi: tuple[float, float, float, float],
) -> tuple[list[float], float]:
    """Compute per-frame motion scores inside ROI."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x1, y1, x2, y2 = roi
    rx1, ry1 = int(x1 * width), int(y1 * height)
    rx2, ry2 = int(x2 * width), int(y2 * height)

    prev_gray = None
    scores: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi_frame = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            scores.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(np.mean(diff))
            scores.append(score)

        prev_gray = gray

    cap.release()
    return scores, fps


def detect_peaks(
    scores: list[float],
    fps: float,
    threshold: float,
    cooldown_s: float,
    max_candidates: int,
) -> list[int]:
    """Peak detection with local-max + cooldown."""

    if len(scores) < 3:
        return []

    cooldown_frames = max(1, int(round(cooldown_s * fps)))
    peaks: list[int] = []
    last_peak = -cooldown_frames

    for i in range(1, len(scores) - 1):
        if scores[i] < threshold:
            continue
        if scores[i] < scores[i - 1] or scores[i] < scores[i + 1]:
            continue
        if i - last_peak < cooldown_frames:
            if peaks and scores[i] > scores[peaks[-1]]:
                peaks[-1] = i
                last_peak = i
            continue
        peaks.append(i)
        last_peak = i

    peaks = sorted(peaks, key=lambda idx: scores[idx], reverse=True)[:max_candidates]
    return sorted(peaks)


def infer_repo_root(video_path: Path) -> Path:
    """Infer project root by looking for pyproject.toml."""

    for parent in [video_path.parent, *video_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def relative_for_labelstudio(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = path.resolve()
    return rel.as_posix()


def export_candidate_assets(
    video_path: Path,
    peaks: Iterable[int],
    scores: list[float],
    fps: float,
    output_dir: Path,
    window_before: float,
    window_after: float,
) -> list[CandidateEvent]:
    clips_dir = output_dir / "clips"
    keyframes_dir = output_dir / "keyframes"
    clips_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    windows: list[tuple[int, int, int]] = []
    before_frames = int(round(window_before * fps))
    after_frames = int(round(window_after * fps))

    for frame_idx in peaks:
        start_f = max(0, frame_idx - before_frames)
        end_f = min(total_frames - 1, frame_idx + after_frames)
        windows.append((frame_idx, start_f, end_f))

    # Read all frames once to simplify clip extraction (Phase 1 baseline, manageable scale)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    video_id = video_path.stem
    events: list[CandidateEvent] = []

    for idx, (frame_idx, start_f, end_f) in enumerate(windows, start=1):
        candidate_id = f"{video_id}_cand_{idx:04d}"
        clip_path = clips_dir / f"{candidate_id}.mp4"
        keyframe_path = keyframes_dir / f"{candidate_id}.jpg"

        # Write clip window
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        for i in range(start_f, end_f + 1):
            writer.write(frames[i])
        writer.release()

        # Write keyframe image (event center)
        cv2.imwrite(str(keyframe_path), frames[frame_idx])

        events.append(
            CandidateEvent(
                candidate_id=candidate_id,
                video_id=video_id,
                frame_index=frame_idx,
                timestamp_s=frame_idx / fps,
                motion_score=float(scores[frame_idx]),
                start_frame=start_f,
                end_frame=end_f,
                clip_path=str(clip_path),
                keyframe_path=str(keyframe_path),
            )
        )

    return events


def write_outputs(
    events: list[CandidateEvent],
    output_dir: Path,
    repo_root: Path,
    local_files_prefix: str,
) -> None:
    jsonl_path = output_dir / "candidates.jsonl"
    csv_path = output_dir / "manifest.csv"
    tasks_path = output_dir / "labelstudio_tasks.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.__dict__) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEvent.__annotations__.keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(event.__dict__)

    tasks = []
    for event in events:
        keyframe_rel = relative_for_labelstudio(Path(event.keyframe_path), repo_root)
        tasks.append(
            {
                "data": {
                    "image": f"{local_files_prefix}{keyframe_rel}",
                    "video_id": event.video_id,
                    "frame_index": event.frame_index,
                    "timestamp_s": round(event.timestamp_s, 3),
                    "candidate_id": event.candidate_id,
                    "clip_path": event.clip_path,
                    "motion_score": event.motion_score,
                }
            }
        )

    with tasks_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    roi = parse_roi(args.roi)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    scores, fps = iter_motion_scores(video_path, roi)
    score_array = np.array(scores, dtype=np.float32)

    if args.motion_threshold is not None:
        threshold = float(args.motion_threshold)
    else:
        med = float(median(scores))
        std = float(np.std(score_array))
        threshold = med + (args.std_multiplier * std)

    peaks = detect_peaks(
        scores=scores,
        fps=fps,
        threshold=threshold,
        cooldown_s=args.cooldown,
        max_candidates=args.max_candidates,
    )

    events = export_candidate_assets(
        video_path=video_path,
        peaks=peaks,
        scores=scores,
        fps=fps,
        output_dir=output_dir,
        window_before=args.window_before,
        window_after=args.window_after,
    )

    repo_root = Path(args.repo_root).resolve() if args.repo_root else infer_repo_root(video_path)
    write_outputs(events, output_dir, repo_root, args.local_files_prefix)

    print("=" * 70)
    print("Candidate shot extraction complete")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames analyzed: {len(scores)}")
    print(f"Threshold used: {threshold:.3f}")
    print(f"Candidate count: {len(events)}")
    print(f"Output dir: {output_dir}")
    print(f"Task file: {output_dir / 'labelstudio_tasks.json'}")


if __name__ == "__main__":
    main()

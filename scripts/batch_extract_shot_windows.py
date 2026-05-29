#!/usr/bin/env python3
"""Batch wrapper for candidate shot-window extraction.

Manifest CSV required columns:
- video: path to input video
- roi: normalized center-goal ROI as x1,y1,x2,y2

Optional columns:
- run_name
- motion_threshold
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "extract_candidate_shot_windows.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch candidate shot-window extraction")
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV with columns: video,roi[,run_name,motion_threshold]",
    )
    parser.add_argument(
        "--output-root",
        default="data/shot_candidates/batch_run",
        help="Root directory for per-video outputs",
    )
    parser.add_argument("--window-before", type=float, default=1.2)
    parser.add_argument("--window-after", type=float, default=1.0)
    parser.add_argument("--cooldown", type=float, default=0.35)
    parser.add_argument("--std-multiplier", type=float, default=2.0)
    parser.add_argument("--max-candidates", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Manifest is empty. Nothing to process.")
        return

    failures = 0
    for idx, row in enumerate(rows, start=1):
        video = (row.get("video") or "").strip()
        roi = (row.get("roi") or "").strip()

        if not video or not roi:
            print(f"[{idx}] Skipping row with missing video/roi: {row}")
            failures += 1
            continue

        video_path = Path(video).expanduser()
        run_name = (row.get("run_name") or video_path.stem).strip()
        run_output = output_root / run_name

        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--video",
            str(video_path),
            "--roi",
            roi,
            "--output-dir",
            str(run_output),
            "--window-before",
            str(args.window_before),
            "--window-after",
            str(args.window_after),
            "--cooldown",
            str(args.cooldown),
            "--std-multiplier",
            str(args.std_multiplier),
            "--max-candidates",
            str(args.max_candidates),
            "--repo-root",
            str(PROJECT_ROOT),
        ]

        motion_threshold = (row.get("motion_threshold") or "").strip()
        if motion_threshold:
            cmd.extend(["--motion-threshold", motion_threshold])

        print(f"[{idx}/{len(rows)}] Processing {video_path} -> {run_output}")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if proc.returncode != 0:
            failures += 1
            print(f"  ❌ failed with return code {proc.returncode}")
        else:
            print("  ✅ done")

    print("=" * 70)
    print(f"Batch extraction complete. rows={len(rows)}, failures={failures}")
    print(f"Output root: {output_root}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

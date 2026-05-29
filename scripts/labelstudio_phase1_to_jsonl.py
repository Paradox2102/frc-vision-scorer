#!/usr/bin/env python3
"""Convert Label Studio Phase 1 export JSON to normalized JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema.phase1_annotation import (
    BBox,
    FrameAnnotation,
    RegionAnnotation,
    ShotEventAnnotation,
    to_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Phase 1 Label Studio export to JSONL")
    parser.add_argument("--input", required=True, help="Label Studio export JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--annotation-index",
        type=int,
        default=0,
        help="Which annotation entry to use per task (default: first)",
    )
    return parser.parse_args()


def _first_or_none(items: Any) -> str | None:
    if not items:
        return None
    if isinstance(items, list):
        return str(items[0]) if items else None
    return str(items)


def _pct_to_unit(value: float) -> float:
    return float(value) / 100.0


def parse_task(task: dict[str, Any], annotation_index: int) -> FrameAnnotation | None:
    annotations = task.get("annotations") or []
    if not annotations:
        return None

    if annotation_index >= len(annotations):
        return None

    ann = annotations[annotation_index]
    results = ann.get("result", [])
    data = task.get("data", {})

    task_id = task.get("id", data.get("candidate_id", "unknown"))
    video_id = str(data.get("video_id", "unknown_video"))
    frame_index = int(data.get("frame_index", 0))
    timestamp_s = float(data.get("timestamp_s", 0.0))
    image_path = str(data.get("image", ""))

    region_by_id: dict[str, RegionAnnotation] = {}

    shot_event = ShotEventAnnotation()

    # Pass 1: collect rectangles
    for r in results:
        if r.get("type") != "rectanglelabels":
            continue
        value = r.get("value", {})
        labels = value.get("rectanglelabels", [])
        label = _first_or_none(labels)
        if label is None:
            continue

        region_id = str(r.get("id", f"region_{len(region_by_id)+1}"))
        region_by_id[region_id] = RegionAnnotation(
            region_id=region_id,
            label=label,  # type: ignore[arg-type]
            bbox=BBox(
                x=_pct_to_unit(float(value.get("x", 0.0))),
                y=_pct_to_unit(float(value.get("y", 0.0))),
                width=_pct_to_unit(float(value.get("width", 0.0))),
                height=_pct_to_unit(float(value.get("height", 0.0))),
            ),
        )

    # Pass 2: attach metadata
    for r in results:
        from_name = r.get("from_name")
        value = r.get("value", {})
        region_id = str(r.get("id")) if r.get("id") is not None else None

        # Region-level metadata
        if region_id and region_id in region_by_id:
            region = region_by_id[region_id]
            if from_name == "region_team_number":
                region.team_number = _first_or_none(value.get("text"))
            elif from_name == "region_alliance":
                alliance = _first_or_none(value.get("choices"))
                if alliance:
                    region.alliance = alliance  # type: ignore[assignment]
            elif from_name == "region_occlusion":
                occ = _first_or_none(value.get("choices"))
                if occ:
                    region.occlusion = occ  # type: ignore[assignment]

        # Frame-level metadata
        if from_name == "made_shot_event":
            label = _first_or_none(value.get("choices"))
            if label:
                shot_event.label = label  # type: ignore[assignment]
        elif from_name == "shot_team_number":
            shot_event.team_number = _first_or_none(value.get("text"))
        elif from_name == "shot_ball_track_id":
            shot_event.ball_track_id = _first_or_none(value.get("text"))
        elif from_name == "shot_confidence":
            conf = _first_or_none(value.get("choices"))
            if conf:
                shot_event.confidence = conf  # type: ignore[assignment]

    return FrameAnnotation(
        task_id=task_id,
        video_id=video_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        image_path=image_path,
        regions=list(region_by_id.values()),
        shot_event=shot_event,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    tasks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Expected Label Studio export to be a JSON list of tasks")

    records: list[FrameAnnotation] = []
    skipped = 0

    for task in tasks:
        parsed = parse_task(task, annotation_index=args.annotation_index)
        if parsed is None:
            skipped += 1
            continue
        records.append(parsed)

    to_jsonl(records, output_path)

    print("=" * 70)
    print("Label Studio Phase 1 export conversion complete")
    print("=" * 70)
    print(f"Input tasks: {len(tasks)}")
    print(f"Records written: {len(records)}")
    print(f"Skipped tasks: {skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

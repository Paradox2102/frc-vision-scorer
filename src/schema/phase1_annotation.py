"""Phase 1 annotation schema helpers for FRC Vision Scorer.

This module defines a normalized schema for frame-level annotation focused on:
- robot localization and tracking cues
- bumper number ROI + transcribed team number
- center goal ROI
- made-shot events into the center container only

The schema is intentionally simple and JSONL-friendly so it can be exported from
Label Studio and consumed by training / evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import json

RegionLabel = Literal["robot", "bumper_number_roi", "center_goal_roi", "fuel_ball"]
Alliance = Literal["red", "blue", "unknown"]
Occlusion = Literal["clear", "partial", "heavy", "unknown"]
ShotEventLabel = Literal["none", "made_shot_center"]


@dataclass
class BBox:
    """Normalized bounding box in xywh format (0..1)."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class RegionAnnotation:
    """Single region annotation on a frame."""

    region_id: str
    label: RegionLabel
    bbox: BBox
    team_number: str | None = None
    alliance: Alliance = "unknown"
    occlusion: Occlusion = "unknown"


@dataclass
class ShotEventAnnotation:
    """Frame-level shot event annotation."""

    label: ShotEventLabel = "none"
    team_number: str | None = None
    ball_track_id: str | None = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


@dataclass
class FrameAnnotation:
    """Normalized frame annotation record for training and evaluation."""

    task_id: int | str
    video_id: str
    frame_index: int
    timestamp_s: float
    image_path: str
    regions: list[RegionAnnotation] = field(default_factory=list)
    shot_event: ShotEventAnnotation = field(default_factory=ShotEventAnnotation)


@dataclass
class AnnotationDataset:
    """Collection of frame annotations."""

    version: str
    records: list[FrameAnnotation]


def _validate_unit_interval(value: float, field_name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{field_name} must be in [0,1], got {value}")


def validate_record(record: FrameAnnotation) -> None:
    """Minimal schema validation for a single frame record."""

    if record.frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {record.frame_index}")
    if record.timestamp_s < 0:
        raise ValueError(f"timestamp_s must be >= 0, got {record.timestamp_s}")

    for region in record.regions:
        _validate_unit_interval(region.bbox.x, "bbox.x")
        _validate_unit_interval(region.bbox.y, "bbox.y")
        _validate_unit_interval(region.bbox.width, "bbox.width")
        _validate_unit_interval(region.bbox.height, "bbox.height")


def to_jsonl(records: list[FrameAnnotation], output_path: str | Path) -> None:
    """Write normalized records to JSONL."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for r in records:
            validate_record(r)
            payload: dict[str, Any] = {
                "task_id": r.task_id,
                "video_id": r.video_id,
                "frame_index": r.frame_index,
                "timestamp_s": r.timestamp_s,
                "image_path": r.image_path,
                "regions": [
                    {
                        "region_id": region.region_id,
                        "label": region.label,
                        "bbox": {
                            "x": region.bbox.x,
                            "y": region.bbox.y,
                            "width": region.bbox.width,
                            "height": region.bbox.height,
                        },
                        "team_number": region.team_number,
                        "alliance": region.alliance,
                        "occlusion": region.occlusion,
                    }
                    for region in r.regions
                ],
                "shot_event": {
                    "label": r.shot_event.label,
                    "team_number": r.shot_event.team_number,
                    "ball_track_id": r.shot_event.ball_track_id,
                    "confidence": r.shot_event.confidence,
                },
            }
            f.write(json.dumps(payload) + "\n")


def from_jsonl(path: str | Path) -> AnnotationDataset:
    """Load normalized annotation JSONL."""

    records: list[FrameAnnotation] = []
    src = Path(path)

    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            data = json.loads(line)

            regions: list[RegionAnnotation] = []
            for region in data.get("regions", []):
                bbox_data = region["bbox"]
                regions.append(
                    RegionAnnotation(
                        region_id=str(region.get("region_id", "")),
                        label=region["label"],
                        bbox=BBox(
                            x=float(bbox_data["x"]),
                            y=float(bbox_data["y"]),
                            width=float(bbox_data["width"]),
                            height=float(bbox_data["height"]),
                        ),
                        team_number=region.get("team_number"),
                        alliance=region.get("alliance", "unknown"),
                        occlusion=region.get("occlusion", "unknown"),
                    )
                )

            shot = data.get("shot_event", {})
            record = FrameAnnotation(
                task_id=data["task_id"],
                video_id=data["video_id"],
                frame_index=int(data["frame_index"]),
                timestamp_s=float(data["timestamp_s"]),
                image_path=data["image_path"],
                regions=regions,
                shot_event=ShotEventAnnotation(
                    label=shot.get("label", "none"),
                    team_number=shot.get("team_number"),
                    ball_track_id=shot.get("ball_track_id"),
                    confidence=shot.get("confidence", "unknown"),
                ),
            )

            try:
                validate_record(record)
            except ValueError as exc:
                raise ValueError(f"Invalid record at line {line_no}: {exc}") from exc

            records.append(record)

    return AnnotationDataset(version="phase1.v1", records=records)

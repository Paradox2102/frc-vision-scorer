"""Export a YOLO-format dataset to Label Studio import JSON.

This converts YOLO txt labels into Label Studio *predictions* so you can
quickly correct them instead of drawing from scratch.

Assumptions:
- One class: fuel_ball (class id 0)
- Dataset structure:
    data/dataset_v1/images/{train,val}/*.jpg
    data/dataset_v1/labels/{train,val}/*.txt

Output:
- labeling/review_v1/tasks_with_predictions.json

Run:
  . .venv/bin/activate
  python scripts/export_yolo_to_labelstudio.py

Then in Label Studio:
- Enable local file serving (see README snippet printed by this script)
- Import the generated JSON.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_v1"
OUT_DIR = PROJECT_ROOT / "labeling" / "review_v1"
OUT_PATH = OUT_DIR / "tasks_with_predictions.json"

# Label Studio config names (must match your label config XML)
LS_IMAGE_FIELD = "image"
LS_FROM_NAME = "label"
LS_TO_NAME = "image"
LS_LABEL_VALUE = "fuel_ball"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


def _list_samples(split: str) -> list[Sample]:
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split

    samples: list[Sample] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            samples.append(Sample(img_path, lbl_path))
    return samples


def _yolo_to_ls_rect(xc: float, yc: float, w: float, h: float) -> dict:
    # YOLO coords are normalized [0..1] for center + width/height.
    x = (xc - w / 2.0) * 100.0
    y = (yc - h / 2.0) * 100.0
    width = w * 100.0
    height = h * 100.0

    # Clamp for sanity; Label Studio expects 0..100.
    x = max(0.0, min(100.0, x))
    y = max(0.0, min(100.0, y))
    width = max(0.0, min(100.0 - x, width))
    height = max(0.0, min(100.0 - y, height))

    return {
        "from_name": LS_FROM_NAME,
        "to_name": LS_TO_NAME,
        "type": "rectanglelabels",
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rectanglelabels": [LS_LABEL_VALUE],
        },
    }


def _local_files_url(path: Path) -> str:
    # Label Studio local file serving uses a URL like:
    #   /data/local-files/?d=<path-relative-to-doc-root>
    # We'll use path relative to PROJECT_ROOT.
    rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return f"/data/local-files/?d={rel.as_posix()}"


def build_tasks(samples: list[Sample], max_images: int | None = None, seed: int = 13) -> list[dict]:
    if max_images is not None and len(samples) > max_images:
        random.Random(seed).shuffle(samples)
        samples = samples[:max_images]

    tasks: list[dict] = []

    for s in samples:
        # Load image to ensure it's readable (also gives us size if needed later)
        img = cv2.imread(str(s.image_path))
        if img is None:
            continue

        results = []
        with s.label_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = parts
                if int(float(cls_id)) != 0:
                    continue
                results.append(_yolo_to_ls_rect(float(xc), float(yc), float(w), float(h)))

        task = {
            "data": {LS_IMAGE_FIELD: _local_files_url(s.image_path)},
            "predictions": [
                {
                    "model_version": "yolo_dataset_v1_autolabel",
                    "result": results,
                }
            ],
        }
        tasks.append(task)

    return tasks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = _list_samples("train") + _list_samples("val")
    tasks = build_tasks(samples)

    OUT_PATH.write_text(json.dumps(tasks, indent=2))

    print(f"Wrote {len(tasks)} tasks -> {OUT_PATH}")
    print()
    print("Label Studio setup notes:")
    print("1) Start Label Studio (from repo root):")
    print("   LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\")
    print(f"   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=\"{PROJECT_ROOT}\" \\")
    print("   labeling/.venv/bin/label-studio start -p 8080")
    print()
    print("2) Create a new project and set label config to:")
    print(f"   {PROJECT_ROOT / 'labeling' / 'label_config_fuel_ball.xml'}")
    print()
    print("3) Import tasks JSON:")
    print(f"   {OUT_PATH}")


if __name__ == "__main__":
    main()

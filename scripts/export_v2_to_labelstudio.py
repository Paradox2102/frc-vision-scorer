"""Export dataset_v2 YOLO labels to Label Studio JSON for human correction.

Prioritizes the 2026 frames since those are the ones we actually need
for the real competition detector.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_v2"
OUT_PATH = PROJECT_ROOT / "labeling" / "review_v2" / "tasks_with_predictions.json"


def _yolo_to_ls_rect(xc: float, yc: float, w: float, h: float) -> dict:
    x = (xc - w / 2.0) * 100.0
    y = (yc - h / 2.0) * 100.0
    width = w * 100.0
    height = h * 100.0
    x = max(0.0, min(100.0, x))
    y = max(0.0, min(100.0, y))
    width = max(0.0, min(100.0 - x, width))
    height = max(0.0, min(100.0 - y, height))
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": x, "y": y, "width": width, "height": height,
            "rectanglelabels": ["fuel_ball"],
        },
    }


def _local_files_url(path: Path) -> str:
    rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return f"/data/local-files/?d={rel.as_posix()}"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tasks = []

    # Process both train and val, prioritizing 2026 frames
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            results = []

            if lbl_path.exists():
                for line in lbl_path.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        _, xc, yc, w, h = parts
                        results.append(_yolo_to_ls_rect(
                            float(xc), float(yc), float(w), float(h)
                        ))

            # Tag with source info
            is_2026 = "2026" in img_path.name
            task = {
                "data": {
                    "image": _local_files_url(img_path),
                    "source": "2026" if is_2026 else "2017",
                },
                "predictions": [{
                    "model_version": "yolo_v1_hsv_merged",
                    "result": results,
                }],
            }
            tasks.append(task)

    # Sort: 2026 frames first (more important to correct)
    tasks.sort(key=lambda t: (0 if t["data"]["source"] == "2026" else 1))

    OUT_PATH.write_text(json.dumps(tasks, indent=2))
    n_2026 = sum(1 for t in tasks if t["data"]["source"] == "2026")
    n_2017 = sum(1 for t in tasks if t["data"]["source"] == "2017")
    print(f"Wrote {len(tasks)} tasks -> {OUT_PATH}")
    print(f"  2026 frames: {n_2026} (review first)")
    print(f"  2017 frames: {n_2017}")


if __name__ == "__main__":
    main()

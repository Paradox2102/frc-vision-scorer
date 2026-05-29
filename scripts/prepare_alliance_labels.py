#!/usr/bin/env python3
"""Generate Label Studio import for alliance-side crops with v5 pre-annotations."""

import json
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).parent.parent
CROP_DIR = PROJECT / "data" / "alliance_crops"
MODEL_PATH = PROJECT / "runs" / "train" / "frc_v5" / "weights" / "best.pt"

CLASS_NAMES = {0: "fuel_ball", 1: "robot", 2: "hub"}
LABEL_CLASSES = {"fuel_ball", "robot"}  # hub_opening is new — manual only


def main():
    model = YOLO(str(MODEL_PATH))
    tasks = []
    total_anns = 0

    for alliance in ["blue", "red"]:
        crops = sorted((CROP_DIR / alliance).glob("*.jpg"))
        print(f"Processing {len(crops)} {alliance} crops...")

        for i, crop_path in enumerate(crops):
            results = model.predict(str(crop_path), conf=0.25, verbose=False)

            annotations = []
            for r in results:
                if r.boxes is None:
                    continue
                h_img, w_img = r.orig_shape
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = CLASS_NAMES.get(cls_id, "unknown")
                    if cls_name not in LABEL_CLASSES:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    annotations.append({
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x1 / w_img) * 100,
                            "y": (y1 / h_img) * 100,
                            "width": ((x2 - x1) / w_img) * 100,
                            "height": ((y2 - y1) / h_img) * 100,
                            "rectanglelabels": [cls_name],
                            "rotation": 0,
                        },
                        "score": conf,
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": w_img,
                        "original_height": h_img,
                    })

            total_anns += len(annotations)

            # Path for Label Studio local files
            rel_path = f"data/labeling/alliance_crops/{alliance}/{crop_path.name}"

            task = {
                "data": {
                    "image": f"/data/local-files/?d={rel_path}",
                    "alliance": alliance,
                },
                "predictions": [{
                    "model_version": "frc_v5_preannotation",
                    "result": annotations,
                }] if annotations else [],
            }
            tasks.append(task)

    output = CROP_DIR / "labelstudio_import.json"
    with open(output, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\n{len(tasks)} tasks, {total_anns} pre-annotations")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()

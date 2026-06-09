#!/usr/bin/env python3
"""Compare bumper checkpoints on reviewed hard-negative crop directories."""
import argparse
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights, images, conf):
    model = YOLO(weights)
    stats = defaultdict(lambda: {"images": 0, "fired": 0, "detections": 0, "max_conf": 0.0})
    results = model.predict(
        source=[str(p) for p in images],
        conf=conf,
        stream=True,
        verbose=False,
    )
    for path, result in zip(images, results):
        label = path.parent.name.rstrip("0123456789")
        boxes = result.boxes
        n = len(boxes)
        confidences = boxes.conf.tolist() if n else []
        row = stats[label]
        row["images"] += 1
        row["fired"] += int(n > 0)
        row["detections"] += n
        row["max_conf"] = max(row["max_conf"], max(confidences, default=0.0))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hard_negatives/boxes")
    ap.add_argument("--weights", action="append", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    root = Path(args.root)
    images = sorted(root.glob("equipment_*/*/frame*.jpg"))
    if not images:
        raise SystemExit(f"No equipment crops found under {root}")

    for weights in args.weights:
        print(f"\n{weights}")
        stats = evaluate(weights, images, args.conf)
        totals = {"images": 0, "fired": 0, "detections": 0, "max_conf": 0.0}
        for label in sorted(stats):
            row = stats[label]
            print(
                f"  {label:8s} images={row['images']:3d} fired={row['fired']:3d} "
                f"detections={row['detections']:3d} max_conf={row['max_conf']:.3f}"
            )
            for key in ("images", "fired", "detections"):
                totals[key] += row[key]
            totals["max_conf"] = max(totals["max_conf"], row["max_conf"])
        print(
            f"  {'total':8s} images={totals['images']:3d} fired={totals['fired']:3d} "
            f"detections={totals['detections']:3d} max_conf={totals['max_conf']:.3f}"
        )


if __name__ == "__main__":
    main()

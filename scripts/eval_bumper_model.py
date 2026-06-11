#!/usr/bin/env python3
"""Evaluate the trained red/blue bumper model: per-class metrics + red<->blue confusion.

The headline risk for a color-defining detector is red/blue swaps, so we surface the
off-diagonal of the confusion matrix explicitly.

Usage:
    .venv/bin/python scripts/eval_bumper_model.py
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "bumper_benjamin" / "bumper_dataset.yaml"
BEST = PROJECT_ROOT / "runs" / "train" / "frc_bumpers" / "weights" / "best.pt"


def main():
    if not BEST.exists():
        raise SystemExit(f"No weights at {BEST} (train first)")
    model = YOLO(str(BEST))
    print(f"names={model.names}")
    metrics = model.val(data=str(DATASET_YAML), plots=True)
    ap50 = metrics.box.ap50          # per-class AP@50
    maps = metrics.box.maps          # per-class mAP@50-95
    p = metrics.box.p; r = metrics.box.r
    print("\nPer-class:")
    for i, name in model.names.items():
        print(f"  {i} {name:12s}  P={p[i]:.3f} R={r[i]:.3f} "
              f"mAP50={ap50[i]:.3f} mAP50-95={maps[i]:.3f}")
    print(f"\noverall mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    # Confusion matrix: rows/cols include a trailing background index.
    cm = metrics.confusion_matrix.matrix
    print("\nConfusion matrix (rows=true, cols=pred; last index=background):")
    print(cm)
    # With names {0: blue_bumper, 1: red_bumper}: cm[i, j] = true i predicted j.
    try:
        blue_as_red = cm[0, 1]
        red_as_blue = cm[1, 0]
        print(f"\nblue->red confusions: {blue_as_red}")
        print(f"red->blue confusions: {red_as_blue}")
    except Exception as e:
        print("confusion-cell read skipped:", e)
    print("\nAlso inspect runs/train/frc_bumpers/confusion_matrix_normalized.png "
          "for the authoritative orientation.")


if __name__ == "__main__":
    main()

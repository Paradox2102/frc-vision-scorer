#!/usr/bin/env python3
"""Evaluate the trained red/blue bumper model: per-class metrics + red<->blue confusion.

The headline risk for a color-defining detector is red/blue swaps, so we surface the
off-diagonal of the confusion matrix explicitly.

Usage:
    .venv/bin/python scripts/eval_bumper_model.py
    .venv/bin/python scripts/eval_bumper_model.py \\
        --weights runs/train/frc_bumpers_offpalette_v1/weights/best.pt \\
        --data data/bumper_hardneg_equipment/bumper_dataset.yaml --split test \\
        --name offpalette_v1_test

Relative --weights/--data/--project paths resolve against the project root. A dataset
yaml without a ``path:`` key has its split dirs resolved relative to the yaml's folder
by ultralytics.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "bumper_benjamin" / "bumper_dataset.yaml"
BEST = PROJECT_ROOT / "runs" / "train" / "frc_bumpers" / "weights" / "best.pt"


def _resolve(p) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else PROJECT_ROOT / q


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", default=str(BEST), help="model weights (default: %(default)s)")
    ap.add_argument("--data", default=str(DATASET_YAML), help="dataset yaml (default: %(default)s)")
    ap.add_argument("--split", default="val", choices=("val", "test", "train"),
                    help="dataset split to evaluate (default: val)")
    ap.add_argument("--project", default=None,
                    help="ultralytics output project dir for plots (default: ultralytics' own)")
    ap.add_argument("--name", default=None, help="ultralytics run name under --project")
    ap.add_argument("--device", default=None, help="ultralytics device (e.g. 0 or cpu)")
    a = ap.parse_args()

    weights, data = _resolve(a.weights), _resolve(a.data)
    if not weights.exists():
        raise SystemExit(f"No weights at {weights} (train first)")
    if weights.stat().st_size < 4096:
        raise SystemExit(f"{weights} looks like a Git LFS pointer (worktree?); "
                         "pass the main checkout's weights path")
    if not data.exists():
        raise SystemExit(f"No dataset yaml at {data}")
    model = YOLO(str(weights))
    print(f"weights={weights}\ndata={data} split={a.split}\nnames={model.names}")
    val_kw = {"data": str(data), "split": a.split, "plots": True}
    if a.project:
        val_kw["project"] = str(_resolve(a.project))
    if a.name:
        val_kw["name"] = a.name
    if a.device is not None:
        val_kw["device"] = a.device
    metrics = model.val(**val_kw)
    ap50 = metrics.box.ap50          # per-class AP@50
    maps = metrics.box.maps          # per-class mAP@50-95
    p = metrics.box.p; r = metrics.box.r
    print("\nPer-class:")
    # ap50/p/r are indexed by position among classes present in the split
    # (metrics.box.ap_class_index), maps by class id.
    present = list(metrics.box.ap_class_index)
    for i, name in model.names.items():
        if i not in present:
            print(f"  {i} {name:12s}  (no labels in this split)")
            continue
        k = present.index(i)
        print(f"  {i} {name:12s}  P={p[k]:.3f} R={r[k]:.3f} "
              f"mAP50={ap50[k]:.3f} mAP50-95={maps[i]:.3f}")
    print(f"\noverall mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    # Confusion matrix: rows/cols include a trailing background index.
    cm = metrics.confusion_matrix.matrix
    print("\nConfusion matrix (rows=pred, cols=true per ultralytics; last index=background):")
    print(cm)
    # With names {0: blue_bumper, 1: red_bumper}: ultralytics stores cm[pred, true].
    try:
        blue_as_red = cm[1, 0]
        red_as_blue = cm[0, 1]
        print(f"\nblue->red confusions (true blue, predicted red): {blue_as_red}")
        print(f"red->blue confusions (true red, predicted blue): {red_as_blue}")
    except Exception as e:
        print("confusion-cell read skipped:", e)
    print(f"\nAlso inspect {metrics.save_dir}/confusion_matrix_normalized.png "
          "for the authoritative orientation.")


if __name__ == "__main__":
    main()

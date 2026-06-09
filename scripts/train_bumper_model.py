#!/usr/bin/env python3
"""Train a red/blue bumper detector (YOLOv8m) on the Benjamin dataset.

Two classes: blue_bumper, red_bumper. Fuel + hub are intentionally NOT trained here
(fuel stays with the existing model; hub is static per-event). Output ->
runs/train/frc_bumpers/. RTX 3090 (24 GB).

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_bumper_model.py
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "bumper_benjamin"
DATASET_YAML = DATASET_DIR / "bumper_dataset.yaml"
RUNS = PROJECT_ROOT / "runs" / "train"

BASE_MODEL = "yolov8m.pt"
EPOCHS = 100
PATIENCE = 20
IMG_SIZE = 640
BATCH_SIZE = -1        # AutoBatch: use the full (now-exclusive) GPU (~60% VRAM target)
DEVICE = 0
WORKERS = 8


def _write_dataset_yaml():
    """Generate a clean config with an absolute `path` (avoids Roboflow's ../-relative
    quirk). Class order is the dataset's own (blue_bumper=0, red_bumper=1); the app's
    DetectionSchema is name-driven, so order doesn't matter downstream."""
    DATASET_YAML.write_text(
        f"path: {DATASET_DIR}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        "names:\n  0: blue_bumper\n  1: red_bumper\n\nnc: 2\n"
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to an existing dataset yaml (e.g. a hard-negative "
                         "augmented set). Default: regenerate + train on Benjamin.")
    ap.add_argument("--name", default="frc_bumpers", help="run name under runs/train/")
    ap.add_argument("--weights", default=BASE_MODEL,
                    help="initial checkpoint (default: yolov8m.pt)")
    args = ap.parse_args()

    if args.data:
        data_yaml = args.data
    else:
        if not (DATASET_DIR / "train" / "images").exists():
            raise SystemExit(f"Dataset not found under {DATASET_DIR} "
                             "(run scripts/download_bumper_datasets.py first)")
        _write_dataset_yaml()
        data_yaml = str(DATASET_YAML)

    model = YOLO(args.weights)
    model.train(
        data=data_yaml,
        epochs=EPOCHS, patience=PATIENCE, imgsz=IMG_SIZE, batch=BATCH_SIZE,
        device=DEVICE, workers=WORKERS,
        project=str(RUNS), name=args.name, exist_ok=True,
        pretrained=True, optimizer="auto", seed=42, deterministic=True,
        amp=True, close_mosaic=10, val=True, plots=True,
        # Color is class-defining (red vs blue) — keep hue jitter modest so the two
        # classes don't bleed into each other under augmentation.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.0,
    )
    best = RUNS / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    m = YOLO(str(best))
    print(f"Model names: {m.names}")
    metrics = m.val(data=data_yaml)
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

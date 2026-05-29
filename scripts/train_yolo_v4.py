#!/usr/bin/env python3
"""Train YOLO v4 on Week 0 Finals auto-annotated dataset."""
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "dataset_v4" / "dataset.yaml"
PRETRAIN_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "frc_v2" / "weights" / "best.pt"
BASE_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

# Use v2 weights if available, else base
weights = str(PRETRAIN_WEIGHTS) if PRETRAIN_WEIGHTS.exists() else str(BASE_WEIGHTS)
print(f"Starting from: {weights}")

model = YOLO(weights)
results = model.train(
    data=str(DATASET_YAML),
    epochs=50,
    patience=10,
    imgsz=640,  # 640 for Mac Mini speed (960 for DL rig)
    batch=8,
    device="mps",
    project=str(PROJECT_ROOT / "runs" / "train"),
    name="frc_v4",
    exist_ok=True,
    workers=4,
    verbose=True,
)
print(f"\nTraining complete. Best model: {results.save_dir}/weights/best.pt")

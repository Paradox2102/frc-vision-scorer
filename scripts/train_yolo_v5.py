#!/usr/bin/env python3
"""Train YOLOv8n on FRC v5 dataset.

This script trains a YOLOv8n model on the v5 dataset with:
- Generic robot detection (no red/blue distinction)
- fuel_ball and hub detection
- 100 epochs, patience 15
- Image size 640, batch 16
- Outputs to runs/train/frc_v5/
"""

from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "dataset_v5" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "train" / "frc_v5"

# Model configuration
BASE_MODEL = "yolov8n.pt"  # YOLOv8 nano (fastest)
EPOCHS = 100
PATIENCE = 15
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0  # GPU device (0 when using CUDA_VISIBLE_DEVICES=1)

# Training parameters
WORKERS = 8
CACHE = False  # Set to True if enough RAM (caches images for faster training)


def main():
    """Train YOLO model on FRC v5 dataset."""
    print("=" * 80)
    print("FRC Vision Scorer v5 - YOLO Training")
    print("=" * 80)
    
    # Validate dataset
    if not DATASET_YAML.exists():
        print(f"Error: Dataset not found at {DATASET_YAML}")
        print("Run scripts/export_v5_labels.py first to create the dataset.")
        return
    
    print(f"\nDataset: {DATASET_YAML}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: GPU {DEVICE}")
    
    # Load base model
    print(f"\nLoading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)
    
    # Train
    print("\nStarting training...")
    print("=" * 80)
    
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        cache=CACHE,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name="frc_v5",
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        save=True,
        save_period=-1,
        plots=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nBest weights: {OUTPUT_DIR / 'weights' / 'best.pt'}")
    print(f"Last weights: {OUTPUT_DIR / 'weights' / 'last.pt'}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    # Validate on test set
    print("\nRunning validation on best model...")
    best_model = YOLO(str(OUTPUT_DIR / "weights" / "best.pt"))
    metrics = best_model.val(data=str(DATASET_YAML))
    
    print("\nValidation metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

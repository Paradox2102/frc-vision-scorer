#!/usr/bin/env python3
"""Train YOLO26s hub-zone model for small object detection.

Uses YOLO26s (small variant) with ProgLoss/STAL features for improved
small object detection of fuel balls and robots near hubs.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data/hub_zone_dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs/train"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26s hub-zone model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_YAML),
        help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (adjust based on GPU memory)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="hub_zone_v1",
        help="Run name for output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device(s): 0, 1, 2, etc. or cpu"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights (recommended)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLO26s Hub-Zone Model Training")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   Data YAML: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Batch size: {args.batch}")
    print(f"   Device: {args.device}")
    print(f"   Pretrained: {args.pretrained}")
    print(f"   Output name: {args.name}")
    
    # Load YOLO26s model
    if args.pretrained:
        print(f"\n📦 Loading YOLO26s with pretrained weights...")
        model = YOLO("yolo26s.pt")
    else:
        print(f"\n📦 Loading YOLO26s from scratch...")
        model = YOLO("yolo26s.yaml")
    
    # Train
    print(f"\n🚀 Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        # YOLO26 features for small object detection
        # ProgLoss: progressive loss weighting for small objects
        # STAL: Small-Target Attention Layer
        # These are automatically enabled in YOLO26 architecture
        
        # Augmentation for small objects
        mosaic=1.0,       # Mosaic augmentation (mixes 4 images)
        mixup=0.1,        # Mixup augmentation (blends 2 images)
        copy_paste=0.1,   # Copy-paste augmentation (good for small objects)
        
        # Optimizer settings
        optimizer="AdamW",
        lr0=0.001,        # Initial learning rate
        lrf=0.01,         # Final learning rate (1% of initial)
        momentum=0.937,
        weight_decay=0.0005,
        
        # Additional settings
        patience=20,      # Early stopping patience
        save=True,
        save_period=10,   # Save checkpoint every 10 epochs
        plots=True,       # Generate training plots
        verbose=True
    )
    
    # Summary
    print(f"\n" + "=" * 80)
    print("✅ Training Complete")
    print("=" * 80)
    print(f"📁 Model saved to: {OUTPUT_DIR}/{args.name}/weights/best.pt")
    print(f"📊 Metrics:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print()


if __name__ == "__main__":
    main()

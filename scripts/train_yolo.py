#!/usr/bin/env python3
"""
Train YOLOv8 on FRC ball detection data.
Start with pre-labeled data, iterate with manual corrections.
"""

import os
from pathlib import Path
from ultralytics import YOLO


def train(
    dataset_yaml: str,
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "mps",  # mps for Apple Silicon, cuda for NVIDIA, cpu fallback
    project: str = "runs/train",
    name: str = "frc_v1",
):
    """Train YOLOv8 on FRC dataset."""
    
    # Load pretrained model
    model_name = f"yolov8{model_size}.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    print(f"Training on: {dataset_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,  # Don't flip vertically (field is always right-side up)
        fliplr=0.5,
        mosaic=0.8,
    )
    
    print(f"\nTraining complete!")
    print(f"  Best model: {project}/{name}/weights/best.pt")
    
    return results


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    dataset_yaml = str(base / "data" / "dataset_v1" / "data.yaml")
    
    if not os.path.exists(dataset_yaml):
        print(f"Dataset not found: {dataset_yaml}")
        exit(1)
    
    # Check if MPS is available
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU (no GPU detected)")
    
    train(
        dataset_yaml=dataset_yaml,
        model_size="n",  # Start small for quick iteration
        epochs=30,  # Enough to see if it's learning
        imgsz=640,
        batch=8,
        device=device,
        project=str(base / "runs" / "train"),
        name="frc_v1",
    )

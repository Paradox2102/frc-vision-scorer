"""Train YOLOv8n on dataset_v3 (v2 + human-corrected Label Studio labels).

Improvements over v2:
  - 1349 total images (was 1371)
  - 585 images with bounding boxes (human-corrected priority)
  - 764 negative samples
  - Human-corrected labels replace ML-generated labels
  - 85/15 train/val split

Training configuration for 2x RTX 3090 (24GB each):
  - YOLOv8n (small, fast)
  - Image size 960 (for small ball detection)
  - Batch size 16 (can go up to 32 if memory allows)
  - 100 epochs
  - Uses CUDA (device=0 for first GPU)
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data" / "dataset_v3" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "train"


def main():
    print("=" * 70)
    print("Training YOLOv8n on dataset_v3")
    print("=" * 70)
    print(f"Data: {DATA_YAML}")
    print(f"Output: {OUTPUT_DIR / 'frc_v3'}")
    print("=" * 70)
    
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,           # More epochs for better convergence
        batch=16,             # Conservative for 3090 - can try 32
        imgsz=960,            # Large size for small ball detection
        patience=20,          # Early stopping
        project=str(OUTPUT_DIR),
        name="frc_v3",
        device=0,             # Use first GPU (CUDA)
        workers=8,            # More workers for faster data loading
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        close_mosaic=15,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mixup=0.1,
        copy_paste=0.1,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Best model: {OUTPUT_DIR / 'frc_v3' / 'weights' / 'best.pt'}")
    print(f"Results: {OUTPUT_DIR / 'frc_v3'}")
    
    # Print key metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\nFinal Metrics:")
        if 'metrics/mAP50(B)' in metrics:
            print(f"  mAP50: {metrics['metrics/mAP50(B)']:.3f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"  mAP50-95: {metrics['metrics/mAP50-95(B)']:.3f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

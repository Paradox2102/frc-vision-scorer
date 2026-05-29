"""Train YOLOv8n on dataset_v2 (2026 + 2017, merged YOLO+HSV prelabels).

Improvements over v1:
  - 860 frames (was 186)
  - 15,849 annotations (was 1,053)
  - Mixed 2026 + 2017 footage
  - Larger image size (960) for small ball detection
  - More augmentation epochs
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data" / "dataset_v2" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "train"


def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        batch=8,
        imgsz=960,           # larger than v1 (640) for small ball detection
        patience=15,          # early stopping
        project=str(OUTPUT_DIR),
        name="frc_v2",
        device="mps",
        workers=4,
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

    print(f"\nTraining complete.")
    print(f"Best model: {OUTPUT_DIR / 'frc_v2' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

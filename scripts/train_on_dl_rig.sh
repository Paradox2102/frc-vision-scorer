#!/bin/bash
# Train YOLO v4 on DL rig (2x RTX 3090)
# Usage: Run from frc-vision-scorer project root
# 1. scp dataset to DL rig
# 2. SSH and run training

set -e

DL_RIG="dl-rig"
REMOTE_DIR="/home/wgilliam/frc-vision-scorer"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== FRC Vision Scorer - DL Rig Training ==="
echo "Local: $LOCAL_DIR"
echo "Remote: $DL_RIG:$REMOTE_DIR"

# Create remote directory structure
echo "Setting up remote directory..."
ssh $DL_RIG "mkdir -p $REMOTE_DIR/data/dataset_v4/{images/{train,val},labels/{train,val}} $REMOTE_DIR/runs/train $REMOTE_DIR/scripts"

# Sync dataset
echo "Syncing dataset_v4 to DL rig..."
rsync -avz --progress \
    "$LOCAL_DIR/data/dataset_v4/" \
    "$DL_RIG:$REMOTE_DIR/data/dataset_v4/"

# Sync pretrained weights
echo "Syncing pretrained weights..."
scp "$LOCAL_DIR/yolov8n.pt" "$DL_RIG:$REMOTE_DIR/"
if [ -f "$LOCAL_DIR/runs/train/frc_v2/weights/best.pt" ]; then
    ssh $DL_RIG "mkdir -p $REMOTE_DIR/runs/train/frc_v2/weights"
    scp "$LOCAL_DIR/runs/train/frc_v2/weights/best.pt" "$DL_RIG:$REMOTE_DIR/runs/train/frc_v2/weights/"
fi

# Create remote training script
echo "Creating remote training script..."
cat << 'REMOTE_TRAIN' | ssh $DL_RIG "cat > $REMOTE_DIR/scripts/train_v4.py"
#!/usr/bin/env python3
"""Train YOLO v4 on DL rig with 2x RTX 3090."""
from ultralytics import YOLO
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "dataset_v4" / "dataset.yaml"

# Update dataset.yaml paths for DL rig
with open(DATASET_YAML) as f:
    cfg = yaml.safe_load(f)
cfg["path"] = str(PROJECT_ROOT / "data" / "dataset_v4")
with open(DATASET_YAML, "w") as f:
    yaml.dump(cfg, f)

PRETRAIN = PROJECT_ROOT / "runs" / "train" / "frc_v2" / "weights" / "best.pt"
BASE = PROJECT_ROOT / "yolov8n.pt"
weights = str(PRETRAIN) if PRETRAIN.exists() else str(BASE)
print(f"Starting from: {weights}")

model = YOLO(weights)
results = model.train(
    data=str(DATASET_YAML),
    epochs=100,
    patience=15,
    imgsz=960,  # Higher resolution on GPU
    batch=32,   # Larger batch on 3090
    device=0,   # Single GPU
    project=str(PROJECT_ROOT / "runs" / "train"),
    name="frc_v4",
    exist_ok=True,
    workers=8,
    verbose=True,
    plots=True,
)
print(f"\nTraining complete!")
print(f"Best model: {results.save_dir}/weights/best.pt")
REMOTE_TRAIN

# Install dependencies if needed
echo "Ensuring dependencies..."
ssh $DL_RIG "cd $REMOTE_DIR && pip install ultralytics opencv-python-headless 2>/dev/null || true"

# Start training
echo ""
echo "Starting training on DL rig..."
echo "Monitor with: ssh $DL_RIG 'tail -f $REMOTE_DIR/runs/train/frc_v4/results.csv'"
ssh $DL_RIG "cd $REMOTE_DIR && nohup python scripts/train_v4.py > train_v4.log 2>&1 &"
echo "Training started in background. Check: ssh $DL_RIG 'tail $REMOTE_DIR/train_v4.log'"

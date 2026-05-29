# Training Status - YOLO v4

## Dataset Preparation: ✅ COMPLETE

- **Frames extracted**: 3,375 total
  - 1,038 overhead views (top 65%)
  - 1,038 alliance left views
  - 1,038 alliance right views
  - 261 full frame samples

- **Auto-annotation**: ✅ Complete
  - YOLO v2 model + HSV color detection
  - 127,300 total detections (37.72 per image)
  - 80/20 train/val split (2700 train, 675 val)

- **Dataset location**: `data/dataset_v4/`
  - Properly formatted for YOLO training
  - Labels: 4 classes (fuel_ball, robot_red, robot_blue, hub)

## Training: ⚠️ NEEDS DL RIG

Attempted training on Mac Mini M4 CPU:
- Speed: ~35-60 minutes per epoch
- Est. total time for 50 epochs: **30-60 hours** ❌

**Recommendation**: Train on DL rig (2x RTX 3090) instead
- Expected speed: ~3-5 minutes per epoch
- Est. total time: ~2.5-4 hours ✅

### To Train on DL Rig:

1. Package dataset:
```bash
cd ~/development/projects/frc-vision-scorer
tar -czf frc_v4_training.tar.gz \
    data/dataset_v4/ \
    scripts/train_yolo_v4.py \
    runs/train/frc_v2/weights/best.pt \
    docs/training_on_dl_rig.md
```

2. Copy to DL rig:
```bash
scp frc_v4_training.tar.gz dl-rig:~/
```

3. On DL rig:
```bash
ssh dl-rig
mkdir -p ~/frc-vision-scorer
cd ~/frc-vision-scorer
tar -xzf ~/frc_v4_training.tar.gz

# Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics torch torchvision opencv-python-headless

# Train with both GPUs
tmux new -s frc-train
.venv/bin/python scripts/train_yolo_v4.py \
    --pretrained \
    --epochs 50 \
    --patience 10 \
    --imgsz 960 \
    --batch 32 \
    --device 0
# Ctrl+B, D to detach
```

4. Monitor progress:
```bash
tmux attach -t frc-train
```

5. Copy results back:
```bash
cd ~/frc-vision-scorer
tar -czf frc_v4_results.tar.gz runs/train/frc_v4*
exit
```

6. On Mac Mini:
```bash
cd ~/development/projects/frc-vision-scorer
scp dl-rig:~/frc-vision-scorer/frc_v4_results.tar.gz .
tar -xzf frc_v4_results.tar.gz
```

## Next Steps

Once training is complete:
1. Evaluate model performance (compare v2 vs v4)
2. Test robot tracking pipeline (Task 4)
3. Test scoring attribution pipeline (Task 5)

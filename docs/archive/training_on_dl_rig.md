# Training YOLO v4 on DL Rig

If training on the Mac Mini is too slow, you can train on the DL rig with 2x RTX 3090 GPUs.

## Preparation

1. Package the dataset and scripts:
```bash
cd ~/development/projects/frc-vision-scorer
tar -czf frc_v4_dataset.tar.gz \
    data/dataset_v4/ \
    scripts/train_yolo_v4.py \
    runs/train/frc_v2/weights/best.pt
```

2. Copy to DL rig:
```bash
scp frc_v4_dataset.tar.gz dl-rig:~/frc-vision-scorer/
```

3. SSH to DL rig:
```bash
ssh dl-rig
```

## On DL Rig

1. Extract dataset:
```bash
cd ~/frc-vision-scorer
tar -xzf frc_v4_dataset.tar.gz
```

2. Create venv and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics torch torchvision opencv-python-headless
```

3. Train with transfer learning:
```bash
.venv/bin/python scripts/train_yolo_v4.py \
    --pretrained \
    --epochs 50 \
    --patience 10 \
    --imgsz 960 \
    --batch 32 \
    --device 0
```

Or train on both GPUs (DDP):
```bash
.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=2 \
    scripts/train_yolo_v4.py \
    --pretrained \
    --epochs 50 \
    --patience 10 \
    --imgsz 960 \
    --batch 64 \
    --device 0,1
```

4. Copy results back:
```bash
cd ~/frc-vision-scorer
tar -czf frc_v4_results.tar.gz runs/train/frc_v4/
exit
```

5. On Mac Mini:
```bash
cd ~/development/projects/frc-vision-scorer
scp dl-rig:~/frc-vision-scorer/frc_v4_results.tar.gz .
tar -xzf frc_v4_results.tar.gz
```

## Monitoring Training

Use `tmux` or `screen` to keep training running:
```bash
tmux new -s frc-train
.venv/bin/python scripts/train_yolo_v4.py --pretrained
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t frc-train
```

## Expected Performance

With 2x RTX 3090:
- ~50 images/sec training speed
- ~3-5 minutes per epoch (3375 images)
- ~150-250 minutes total for 50 epochs
- Early stopping likely kicks in around epoch 10-15

## Comparing Models

After training, compare v2 vs v4:
```bash
.venv/bin/python scripts/compare_models.py \
    runs/train/frc_v2/weights/best.pt \
    runs/train/frc_v4/weights/best.pt \
    --test-video data/videos/2026/week0_f1m1.mp4
```

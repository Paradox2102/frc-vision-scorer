# Hub-Zone YOLO26 Pipeline

## Overview

Per-robot scoring attribution system using hub-zone crop approach with YOLO26 small object detection.

**Why hub-zone crops?**
- Full 1280x720 field: balls are ~15px (too small for accurate detection)
- Hub-zone crops (400x400): balls are ~30-50px (much better SNR)
- Focused detection region = higher accuracy + faster inference

## Pipeline Architecture

```
Week 0 Videos (3 matches)
    ↓
1. Hub Detection (v5 model)
    ↓
2. Extract Crops (every 3rd frame)
    ↓ 10,358 total crops
3. Filter Interesting Crops (robots/balls present)
    ↓ ~1,000-3,000 crops
4. Pre-label with v5
    ↓
5. Label Studio (add hub_opening class)
    ↓
6. Train YOLO26s hub-zone model
    ↓
7. Hub-zone inference → per-robot scores
```

## Scripts

### 1. Extract Hub-Zone Crops
```bash
.venv/bin/python scripts/extract_hub_zone_crops.py
```

**Output:**
- `data/hub_zones/blue/` - Blue hub crops
- `data/hub_zones/red/` - Red hub crops
- `data/hub_zones/crop_metadata.csv` - Crop coordinates

**Process:**
- Uses v5 model to detect hubs in first 30 frames
- Averages hub positions across frames
- Extracts 400x400 crops centered on each hub
- Processes every 3rd frame (reduces dataset size)

### 2. Filter to Interesting Crops
```bash
.venv/bin/python scripts/filter_hub_zone_crops.py
```

**Output:**
- `data/hub_zones_filtered/blue/` - Filtered blue crops
- `data/hub_zones_filtered/red/` - Filtered red crops
- `data/hub_zones_filtered/filtered_metadata.csv` - Filtered crop list

**Process:**
- Runs v5 model on all crops
- Keeps crops with robots OR balls detected
- Reduces ~10K crops to ~1-3K for labeling

### 3. Prepare for Label Studio
```bash
.venv/bin/python scripts/prepare_hub_zone_labels.py
```

**Output:**
- `data/hub_zones_filtered/labelstudio_import.json` - Label Studio import file

**Process:**
- Generates Label Studio task JSON
- Includes v5 pre-annotations for `fuel_ball` and `robot`
- Ready for manual `hub_opening` labeling

### 4. Label in Label Studio

**Import:**
1. Open Label Studio: http://localhost:8080
2. Create new project: "FRC Hub Zones"
3. Import `data/hub_zones_filtered/labelstudio_import.json`

**Label Classes:**
- `fuel_ball` - Yellow foam balls (pre-labeled)
- `robot` - Competition robots (pre-labeled)
- `hub_opening` - **NEW:** The actual hub opening/target where balls enter

**Labeling Tips:**
- Review v5 pre-annotations (fuel_ball, robot) - correct if needed
- Add `hub_opening` bounding boxes around the scoring target
- Hub opening is typically circular, ~60-80px diameter in crops

**Export:**
- Export as YOLO format when done
- Organize into train/val/test splits:
  - `data/hub_zones_filtered/images/train/`
  - `data/hub_zones_filtered/images/val/`
  - `data/hub_zones_filtered/labels/train/`
  - `data/hub_zones_filtered/labels/val/`

### 5. Train YOLO26s Hub-Zone Model

**On DL Rig (recommended):**
```bash
# Copy dataset to DL rig
rsync -avz data/hub_zones_filtered/ dl-rig:~/frc-hub-zones/

# SSH to DL rig
ssh dl-rig

# Train (2x RTX 3090)
cd ~/frc-hub-zones
python train_hub_zone_model.py \
  --data hub_zone_dataset.yaml \
  --epochs 100 \
  --batch 32 \
  --device 0,1 \
  --pretrained

# Copy model back
scp -r runs/train/hub_zone_v1 ship:~/development/projects/frc-vision-scorer/runs/train/
```

**Locally (slower):**
```bash
.venv/bin/python scripts/train_hub_zone_model.py \
  --epochs 100 \
  --batch 16 \
  --pretrained
```

**Model:**
- YOLO26s (small variant, not nano)
- 640px training resolution
- ProgLoss + STAL features for small objects
- Augmentation: mosaic, mixup, copy-paste

**Expected Results:**
- mAP50 > 90% (fuel_ball, robot classes)
- mAP50 > 85% (hub_opening - new class)
- Training time: ~2-3 hours (DL rig), ~8-12 hours (local)

### 6. Run Hub-Zone Inference

**Test on Week 0 match:**
```bash
.venv/bin/python scripts/hub_zone_inference.py \
  data/videos/2026/week0_f1m1.mp4
```

**Output:**
- Per-robot score counts
- Ball attribution to robots
- Scoring event timeline

**Process:**
1. Detect hubs with v5 in first frames
2. Crop hub zones each frame
3. Run YOLO26 hub-zone model on crops
4. Track robots with ByteTrack
5. Detect balls entering hub opening
6. Attribute balls to nearest robot
7. Output per-robot scores

### 7. Web App Integration

**Add hub-zone model to web app:**

The web app integration is **pending** until the hub-zone model is trained.

Once trained, the model will be added as a new option in the web UI:
- Model dropdown: v2, v4, v5, **hub_zone_v1**
- When selected, switches to hub-zone inference pipeline
- Displays per-robot scores in real-time

## Key Files

```
scripts/
  extract_hub_zone_crops.py       # Step 1: Extract crops
  filter_hub_zone_crops.py        # Step 2: Filter crops
  prepare_hub_zone_labels.py      # Step 3: Label Studio prep
  train_hub_zone_model.py         # Step 5: Training
  hub_zone_inference.py           # Step 6: Inference

data/
  hub_zones/                      # All extracted crops
    blue/                         # Blue hub crops
    red/                          # Red hub crops
    crop_metadata.csv             # Crop coordinates
  
  hub_zones_filtered/             # Filtered crops
    blue/                         # Filtered blue crops
    red/                          # Filtered red crops
    filtered_metadata.csv         # Filtered metadata
    labelstudio_import.json       # Label Studio import
    images/                       # After labeling
      train/                      # Training images
      val/                        # Validation images
    labels/                       # After labeling
      train/                      # Training labels
      val/                        # Validation labels
  
  hub_zone_dataset.yaml           # YOLO dataset config

runs/
  train/
    hub_zone_v1/                  # Trained model
      weights/
        best.pt                   # Best model weights
```

## Dataset Statistics

**Extraction:**
- Videos: 3 (week0_f1m1, week0_f1m2, week0_f1m3)
- Frames per video: ~5,179
- Total crops: 10,358 (5,179 blue + 5,179 red)
- Crop size: 400x400px

**Filtering:**
- Input crops: 10,358
- Filtered crops: ~1,000-3,000 (estimated)
- Reduction: ~70-90%

**Classes:**
- `fuel_ball` - Pre-labeled by v5
- `robot` - Pre-labeled by v5
- `hub_opening` - **Manual labeling required**

## Model Performance Targets

**YOLO26s Hub-Zone Model:**
- mAP50 (fuel_ball): > 95%
- mAP50 (robot): > 90%
- mAP50 (hub_opening): > 85%
- Inference speed: ~60 FPS (M4 Pro)

**Attribution Accuracy:**
- High confidence: Ball within 100px of robot before scoring
- Medium confidence: Ball trajectory points to robot quadrant
- Low confidence: Ball appeared near hub with no clear attribution

## Next Steps for wg

1. **Wait for filtering to complete** (~5-10 more minutes)
2. **Run Label Studio prep:** `prepare_hub_zone_labels.py`
3. **Import into Label Studio** at localhost:8080
4. **Label hub_openings** on filtered crops (~1-3K images)
5. **Export as YOLO format**
6. **Copy to DL rig** and train YOLO26s model
7. **Test inference** on Week 0 videos
8. **Integrate into web app** (code ready, just needs trained model)

## Notes

- Hub detection by v5 is 98.6% mAP50 - very reliable
- Hub positions are stable across frames - averaging improves accuracy
- 400x400 crops are generous - capture full scoring zone + approach paths
- Every 3rd frame is sufficient - balls move slowly (~5 px/frame)
- Pre-annotations save ~70% of labeling time
- Hub opening is the critical new class - most important to label accurately

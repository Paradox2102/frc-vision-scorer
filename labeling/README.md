# Labeling workflow (fuel_ball)

## 0) What this is
We use Label Studio to *correct* noisy auto-labels from the YOLO v1 model + HSV color detection.
Corrected labels will be used to train a much better v3 detector.

## 1) Start Label Studio
From repo root:

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Users/ship/development/projects/frc-vision-scorer" \
labeling/.venv/bin/label-studio start labeling/ls_fuelball --init \
  -l labeling/label_config_fuel_ball.xml \
  --username ship --password fuelball \
  -p 8080 -b
```

Open: http://localhost:8080
Login:
- email: `ship`
- password: `fuelball`

## 2) Import tasks (with prelabels)

### Dataset v2 (current - 782 tasks)
Import: `labeling/review_v2/tasks_with_predictions.json`

Contains 782 frames (560 from 2026 footage, 222 from 2017 Steamworks) with 14,137 pre-labeled bounding boxes.

Sources:
- 2026 Kettering practice (overhead, 50-70 balls, small ~10-15px)
- 2026 team practice videos (closer angles, AprilTags on hubs)
- 2026 FRC 3255 full match
- 2026 Ri3D scoring demo
- 2026 IBOTS shooter test
- 2017 Steamworks matches (similar yellow balls)

### Dataset v1 (legacy - 186 tasks)
Import: `labeling/review_v1/tasks_with_predictions.json`

## 3) Labeling rules (quick)
- Only label: `fuel_ball`
- **Delete obvious false positives** (yellow field parts, banners, bumpers, yellow tape, etc.)
- **Tight boxes** around the visible ball (don't include shadows)
- If a ball is <~30% visible, skip it (consistency > completeness)
- **Add missed balls** — draw new boxes for balls the model missed

## 4) Priority frames
- 2026 frames first (filenames starting with `2026_`) — most valuable
- Frames with lots of balls visible
- Frames where auto-labels are clearly wrong

## 5) Keyboard shortcuts
- **Delete/Backspace** — remove selected box
- **Ctrl+Enter** — submit annotation
- **Arrow keys** — navigate between tasks

## 6) After labeling
Export annotations from Label Studio as JSON, then convert to YOLO format:

```bash
python scripts/labelstudio_export_to_yolo.py \
    --input labeling/export.json \
    --output data/dataset_v3/labels \
    --image-dir data/dataset_v2/images \
    --copy-images data/dataset_v3/images \
    --split 0.8
```

Then retrain:
```bash
python scripts/train_yolo_v2.py  # update data path to dataset_v3
```

---

## Phase 1 scoring annotation (robot + team ID + made-shot events)

For Phase 1 team scoring tasks, use:

- Config: `labeling/label_config_phase1_scoring.xml`
- Schema: `labeling/annotation_schema_phase1.json`
- Guide: `labeling/README_phase1_scoring.md`

# Phase 1 Pipeline Quickstart

## 1) Build candidate shot windows

Single video:

```bash
.venv/bin/python scripts/extract_candidate_shot_windows.py \
  --video data/videos/2026_cranberry_breakdown.mp4 \
  --roi 0.43,0.22,0.58,0.43 \
  --output-dir data/shot_candidates/cranberry_breakdown
```

Batch mode via manifest:

```bash
.venv/bin/python scripts/batch_extract_shot_windows.py \
  --manifest data/manifests/phase1_shot_videos.csv \
  --output-root data/shot_candidates/phase1_batch
```

## 2) Annotate in Label Studio

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Users/ship/development/projects/frc-vision-scorer" \
labeling/.venv/bin/label-studio start labeling/ls_phase1_scoring --init \
  -l labeling/label_config_phase1_scoring.xml \
  --username ship --password fuelball \
  -p 8080 -b
```

Import tasks generated at:

```text
data/shot_candidates/<run_name>/labelstudio_tasks.json
```

## 3) Convert Label Studio export to normalized JSONL

```bash
.venv/bin/python scripts/labelstudio_phase1_to_jsonl.py \
  --input labeling/phase1_export.json \
  --output data/annotations/phase1_train.jsonl
```

## 4) Train baseline components (next step placeholders)

Detector training command (placeholder, adapt to final dataset yaml):

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/dataset_phase1/data.yaml \
  epochs=100 imgsz=1280 batch=16
```

OCR baseline experiment (placeholder):

```bash
.venv/bin/python scripts/train_team_number_ocr.py \
  --annotations data/annotations/phase1_train.jsonl \
  --output models/ocr_phase1
```

Goal-entry event baseline (placeholder):

```bash
.venv/bin/python scripts/train_goal_entry_baseline.py \
  --events data/annotations/phase1_train.jsonl \
  --output models/goal_event_phase1
```

## 5) Evaluate per-team counts + event metrics

```bash
.venv/bin/python scripts/evaluate_phase1_scoring.py \
  --ground-truth data/eval/ground_truth_events.jsonl \
  --predictions data/eval/predicted_events.jsonl \
  --time-tolerance 0.6 \
  --output-json data/eval/reports/phase1_eval.json
```

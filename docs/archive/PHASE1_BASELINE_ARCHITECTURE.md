# Phase 1 Baseline Architecture (FRC Vision Scorer)

## Scope
Phase 1 baseline is constrained to three requirements:
1. per-team scored-ball counts
2. robot team-number ID
3. counting only **made shots into center container**

---

## End-to-end pipeline

```text
Video -> Candidate shot windows -> Label Studio annotation -> Model training
      -> Inference (detect+track+OCR+event) -> Event attribution -> Per-team counts
      -> Evaluation (MAE + precision/recall)
```

### Stage A — Candidate window mining
- Script: `scripts/extract_candidate_shot_windows.py`
- Input: full match video + center goal ROI
- Output: candidate clips/keyframes around likely goal-entry moments

### Stage B — Annotation
- Config: `labeling/label_config_phase1_scoring.xml`
- Schema: `labeling/annotation_schema_phase1.json`
- Required labels:
  - `robot`
  - `bumper_number_roi`
  - `center_goal_roi`
  - frame-level `made_shot_center`

### Stage C — Baseline models

#### 1) Robot detection + tracking
- Detector baseline: YOLOv8/YOLO11 (`robot` class)
- Tracker baseline: ByteTrack
- Output: stable `robot_track_id` per frame

#### 2) Team-number OCR
- Input: `bumper_number_roi` crops linked to robot tracks
- OCR baseline: PaddleOCR / EasyOCR (to be selected in implementation)
- Aggregation:
  - accumulate OCR votes per `robot_track_id`
  - majority vote with confidence weighting
  - optional temporal smoothing (HMM-style) for ID stability

#### 3) Goal-entry event detector
- Input signals:
  - ball detections/tracks near `center_goal_roi`
  - trajectory direction into goal ROI
  - optional frame-diff peak score (from candidate mining)
- Output event type:
  - `made_shot_center`
- Explicitly exclude misses and non-center targets.

#### 4) Shot attribution
- For each made-shot event:
  - select shooter candidate from nearest active robot track at launch-time window
  - prefer robot with recent outbound ball trajectory intersecting event track
  - map `robot_track_id -> team_number` via OCR vote table
- Emit event record:
  - `match_id`, `team_number`, `timestamp_s`, `event_label=made_shot_center`

---

## Data contracts

### Normalized frame annotation (training/eval intermediate)
Defined in `src/schema/phase1_annotation.py`.

Core fields:
- `video_id`, `frame_index`, `timestamp_s`, `image_path`
- `regions[]` with bbox + label + optional team/alliance metadata
- `shot_event` with `label`, `team_number`, and confidence

### Event output contract (inference)
JSONL line per event:

```json
{"match_id":"2026_example_qm1","team_number":"3255","timestamp_s":43.27,"event_label":"made_shot_center"}
```

---

## Baseline training/eval plan

1. Train detector(s) on mixed frame dataset (`robot`, `bumper_number_roi`, `center_goal_roi`, `fuel_ball`)
2. Train/plug OCR on bumper crops, calibrate confidence thresholds
3. Train goal-entry classifier/rules over candidate windows
4. Integrate attribution logic into event emitter
5. Evaluate with:
   - per-team count MAE
   - event precision / recall / F1

Evaluation scaffold script:
- `scripts/evaluate_phase1_scoring.py`

# Robot Identification Calibration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a per-match calibration step where users click on each robot in a sample frame, the system OCRs the bumper number, and the user confirms/corrects it — constrained to teams from the TBA match data.

**Architecture:** New calibration step inserted between scoreboard calibration and inference. Backend runs YOLO on the calibration frame to detect robots, crops each robot's bumper region, runs Tesseract OCR on the white text, and returns candidates. Frontend shows the frame with detected robots highlighted and dropdowns pre-populated with TBA match teams. The confirmed robot-to-team mapping is passed to InferenceJob to seed `robot_tracks` with known positions and team numbers.

**Tech Stack:** Python (FastAPI, OpenCV, pytesseract, ultralytics), vanilla JS frontend

---

### Task 1: Backend — Robot Detection Endpoint

**Files:**
- Modify: `web/app.py` — add `/api/detect-robots` endpoint

**What it does:**
- Accepts a base64 frame (reuses the already-grabbed calibration frame)
- Runs YOLO model on it to detect robots (class_id == 1)
- For each detected robot:
  - Crops the bumper region (bottom 35% of bounding box — where bumper numbers are)
  - Runs pytesseract OCR on the crop (white text on colored background)
  - Detects bumper color (reuses `detect_bumper_color`)
- Returns list of detected robots with: bounding box, alliance guess, OCR'd number, confidence
- Also accepts `match_teams` (the 6 team numbers) and fuzzy-matches OCR results to valid teams

### Task 2: Frontend — Robot ID Calibration UI

**Files:**
- Modify: `web/static/index.html` — add robot calibration section
- Modify: `web/static/app.js` — add robot calibration flow
- Modify: `web/static/style.css` — styling for robot cards

**What it does:**
- New section shown after scoreboard calibration, before inference starts
- Displays the calibration frame with robot bounding boxes drawn on canvas
- Below: a card per detected robot showing:
  - Cropped robot image
  - Alliance color badge
  - Dropdown of team numbers for that alliance (from TBA match data)
  - OCR suggestion pre-selected if it matches a valid team
- "Confirm & Start Inference" button
- "Skip" button (falls back to existing OCR-during-inference behavior)

### Task 3: Wire Robot IDs into Inference Pipeline

**Files:**
- Modify: `web/app.py` — accept robot mappings in inference start, seed robot_tracks

**What it does:**
- `/api/start-inference` accepts optional `robot_identifications` parameter
- If provided, seeds `InferenceJob.robot_tracks` with known positions and team numbers
- These pre-seeded robots get priority in `get_canonical_robot_id` — new detections near a known position inherit its team number
- The canonical robot limit (3 per alliance) counts pre-seeded robots

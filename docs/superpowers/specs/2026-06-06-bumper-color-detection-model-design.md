# Bumper-Color Detection Model — Design

**Date:** 2026-06-06
**Status:** Draft for review

## Summary

Replace the current generic robot detector with a model that detects **red bumpers**,
**blue bumpers**, and **fuel** directly. Alliance color becomes inherent to the
detection class instead of being inferred downstream by HSV bumper analysis. The
work splits into two sub-projects joined by a fixed contract:

- **SP-A — Build the model:** merge two Roboflow datasets into one unified schema and
  train a YOLOv8m model.
- **SP-B — Integrate the model:** load it, normalize its output, expand bumper boxes
  into robot boxes for SAM2 seeding, and take color from the class.

**Shared contract:** a `.pt` whose `model.names == {0: "red_bumper", 1: "blue_bumper",
2: "fuel"}`. With that fixed, the two sub-projects build and test independently.

## Motivation

The current detector (`runs/train/hub_zone_yolo26s/weights/best.pt`) detects three
classes — `fuel_ball` (0), `robot` (1), `hub_opening` (2) — and does **not** know
alliance color. Color is computed after detection by `classify_bumper_color`
(`web/field/bumper.py`), an HSV heuristic on a cropped bumper band. Robot paint, LEDs,
and mechanisms in the opposite color make that heuristic error-prone.

A model trained to detect red vs. blue bumpers directly:
- gives alliance color as a first-class, learned signal (more robust than HSV);
- yields a bumper box that is the ideal crop for VLM team-number reading;
- puts its detection box bottom right at the robot's ground-contact point, which is
  exactly what field-coordinate projection wants.

What it gives up — `hub_opening` detection — is already redundant: hub positions are
set statically per event through the existing calibration step
(`event_hub_positions`, `app.py:1675`; `set_hub_positions` API). So dropping the hub
class costs nothing for scoring.

## SP-A — Build the model

### Inputs
Two Roboflow Universe datasets (both downloaded via the `roboflow` SDK using the
`ROBOFLOW_API` key already present in the project `.env`):

1. **Benjamin** — `benjamins-workspace-h5vao/frc-bumpers-h0p1g` — red/blue bumpers
   (no fuel).
2. **Karthik** — `karthiks-workspace-gez2e/frc-bumpers-red-blue` — red/blue bumpers
   **and** fuel (smaller).

Exact class **names** and **id order** in each source are unknown until download and
must be read from each dataset's `data.yaml` — do not assume. The merge maps by
normalized name, not by source id.

### Unified schema
```
0: red_bumper
1: blue_bumper
2: fuel
```

### Merge step
A new script `scripts/merge_bumper_datasets.py`, modeled on the existing
`scripts/merge_roboflow_dataset.py` (which already remaps class ids, drops unmapped
classes, and reconciles differing train/val/test split layouts):

- Build a normalization table: lowercase + strip + collapse spaces/underscores, then
  map source names → unified ids. Expected aliases:
  - `{red_bumper, red bumper, red, redbumper}` → 0
  - `{blue_bumper, blue bumper, blue, bluebumper}` → 1
  - `{fuel, fuel_ball, ball, cargo}` → 2
- Any source class not in the table is **dropped** (its label lines removed); the
  script logs every dropped class name and count so unexpected classes surface loudly.
- Rewrite each label file's class ids to the unified schema; copy images.
- Reconcile splits: Benjamin and Karthik may use different split folder names
  (`valid` vs `val`, presence/absence of `test`). Output a clean `train/val/test`
  layout under `data/bumper_merged/` with a generated `dataset.yaml`.
- Print a summary table: per-source, per-split, per-class counts.

### Data-quality risk: unlabeled fuel
The Benjamin set is bumpers-only. If its images contain fuel balls that are **not**
labeled, training treats those pixels as background and degrades fuel detection.

**Mitigation (decided after inspecting the data):** after merge, sample Benjamin
images and check for visible-but-unlabeled fuel. YOLO's loss has no per-image way to
say "ignore fuel here," so if unlabeled fuel is present and material, pick one of:
- **(a)** pseudo-label fuel on Benjamin images with a Karthik-only fuel detector, then
  spot-review the auto-labels before merging; or
- **(b)** accept the cost if fuel is rare enough in Benjamin images to be noise.

The chosen option is recorded in the implementation plan once the data is seen. If
fuel is essentially absent from Benjamin images, no action is needed.

### Training
A new script `scripts/train_bumper_model.py`, modeled on `scripts/train_yolo_v5.py`:
- Base: `yolov8m.pt`.
- `data: data/bumper_merged/dataset.yaml`, `imgsz: 640`, `epochs: 100`,
  `patience: 20`, `batch: 16` (3090, 24 GB; raise if VRAM allows), `device: 0`.
- Output: `runs/train/frc_bumpers/`.

### Evaluation (acceptance criteria for SP-A)
- `ultralytics` validation: report overall mAP@50 and **per-class** mAP.
- **Red↔blue confusion** read directly from the normalized confusion matrix — this is
  the headline metric; red/blue swaps are the worst failure (they flip alliance).
- Sanity check: run the trained model on a handful of acceptance-harness frames
  (see `acceptance-harness-download` memory) and eyeball red/blue/fuel boxes.
- Gate to SP-B switchover: per-class mAP@50 for both bumper classes at least roughly
  comparable to the current robot-detection recall, and red↔blue confusion low
  (target < ~5% off-diagonal). Exact thresholds confirmed against the eval output;
  this is a judgment gate, not an automated pass/fail.

## SP-B — Integrate the model

### Model loading + schema auto-detection
- Add an env override for the detector path (e.g. `FRC_DETECT_MODEL`), defaulting to
  the current model until SP-A validates, then to `runs/train/frc_bumpers/weights/best.pt`.
- After `YOLO(...)` load (`app.py:481`), inspect `self.model.names` and build a small
  `DetectionSchema` describing, by class id: which ids are robots and their fixed color
  (`red_bumper`→"red", `blue_bumper`→"blue", legacy `robot`→None), which id is fuel,
  which (if any) is hub. This makes both the legacy and new models load unchanged.

### Detection normalization helper
Centralize the currently-scattered class filtering (`app.py:798`, `:1637`, `:1942`,
`:2820`, `:2880`, `:3332`) behind one helper that takes raw `sv.Detections`
+ the schema and returns:
- `robot_dets`: list of `(box, color_or_None, conf)`,
- `ball_boxes`,
- `hub_boxes` (empty for the new model).

Two related surfaces the planner must also touch:
- `app.py:3638` hardcodes `"model_classes": {0: "fuel_ball", 1: "robot",
  2: "hub_opening"}` in `/api/status`; drive it from the loaded schema so it doesn't
  report stale classes under the new model.
- `_detect_bumper_color_static` (`app.py:4251`, called at `:4881`) is a **second** HSV
  color path. It stays on the legacy HSV approach for now (not in SP-B's critical
  path), but the plan should note it explicitly so it isn't mistaken for dead code or
  accidentally rewired.

### Box semantics
- **Color, field position, and VLM crop** use the **bumper box** as-is. The box bottom
  is the ground-contact point; the bumper box is the team-number crop for the VLM.
  Field projection's code path is unchanged (`_bottom_center` → box `y2`), but note a
  small **modeling shift**: a bumper box's `y2` is the bumper's bottom edge, slightly
  above true wheel/floor contact, versus a full-robot box bottom. Expected to be
  negligible, but the SP-B integration test should confirm field positions don't drift
  meaningfully versus the legacy model on the same footage.
- **SAM2 seeding only** uses an **expanded** box. New `expand_bumper_to_robot(box,
  frame_shape, factor)` grows the box **upward** by `factor` (env
  `ROBOT_BOX_FROM_BUMPER_FACTOR`, default ~3.0 → robot height ≈ 3× bumper-box height),
  clamped to the frame top; width unchanged (bumpers define the footprint). Applied at
  the SAM2 prompt site (`app.py:2821`) and the periodic re-prompt `detect_fn`
  (`app.py:2881`). This preserves whole-robot SAM2 masks and the appearance-gallery
  signature.

### Color from class
- `to_field_detections` (`web/field/detect_bridge.py`) gains a per-box color input:
  accept `robot_dets` as `(box, color)` pairs; when `color` is provided use it,
  otherwise fall back to the injected `color_fn` (legacy HSV path). Keeps the function
  unit-testable and backward-compatible.
- With the new model, `classify_bumper_color` / `detect_bumper_color` are bypassed for
  the primary color. They remain in the tree for the legacy model and as documentation.

### Hubs
- `update_hub_positions` already prefers `event_hub_positions`. With no `hub_opening`
  detections it simply relies on the static positions. Add a startup check: if the
  loaded schema has no hub class **and** no `event_hub_positions` are set for the event,
  log a clear warning pointing at the calibration step.

### Out of scope for SP-B
- Multi-camera fusion color voting (`web/field/fusion.py`) keeps working unchanged; the
  per-detection color it receives is now the model's class color instead of HSV. No
  voting-logic change planned.
- Gallery, VLM reading, tracker, and scoring logic are untouched beyond receiving
  better color and whole-robot masks.

## Testing

**SP-A:** the merge script is verified by its printed per-class/per-split counts and a
spot-check of remapped label files; training/eval verified by the metrics above.

**SP-B (unit, `.venv/bin/python -m pytest` per `test-env-uses-venv` memory):**
- `DetectionSchema` correctly classifies both legacy (`robot/fuel_ball/hub_opening`)
  and new (`red_bumper/blue_bumper/fuel`) `names` dicts.
- Normalization helper buckets a synthetic `sv.Detections` into robot/ball/hub with
  correct colors.
- `expand_bumper_to_robot`: grows upward by the factor, clamps at frame top, leaves
  width and the bottom edge unchanged (so field position is invariant).
- `to_field_detections`: uses an explicit per-box color when given; falls back to
  `color_fn` when color is `None`.

**SP-B (integration):** run the existing acceptance harness on one match with the new
model + static hubs; confirm tracks seed, masks cover whole robots, alliance colors are
correct, and scoring still runs.

## Build order
SP-A first (it produces the contract artifact), then SP-B. SP-B's pure helpers
(`DetectionSchema`, `expand_bumper_to_robot`, `to_field_detections` refactor) can be
built and unit-tested in parallel against the fixed schema before weights exist.

# SP-B: Integrate the Bumper Model (AUGMENT) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

## Context: this is now an AUGMENT integration, not a replace

SP-A produced a **2-class bumper detector** (`runs/train/frc_bumpers/weights/best.pt`,
`names = {0: blue_bumper, 1: red_bumper}`, mAP50 0.977, **0 red↔blue swaps**). It detects
**robots + alliance color only** — no fuel, no hub. So we run **two models**:

- **Existing model** (`runs/train/hub_zone_yolo26s`, `self.model`) — keeps providing
  **fuel** (class 0) and **hub** (class 2), exactly as today. Its robot class (1) is
  ignored.
- **New bumper model** (`self.bumper_model`) — provides **robot detections + alliance
  color** (red_bumper→red, blue_bumper→blue), replacing the existing model's robot class
  **and** the HSV color path.

**Goal:** robots + color come from the bumper model (resolved through the existing fusion
vote); fuel/hub stay on the existing model; the HSV color path is deleted; bumper boxes
are expanded upward into robot boxes for SAM2 seeding so masks + appearance gallery stay
whole-robot.

**Net change from today:** only the **robot-detection + color** paths switch to the new
model. **Ball/hub paths are unchanged** (they keep reading the existing model's
detections — so the `class_id==0` ball filter and `class_id==2` hub filter stay as-is).
This is *smaller* than the earlier "route every class-id site through one schema" plan,
but adds a second model pass and a dual-source HUD/detection split.

**Tech Stack:** Python, `ultralytics`/`supervision`, `web/field/`. Tests:
`.venv/bin/python -m pytest` (base conda python lacks pytest — project memory).

**Spec:** `docs/superpowers/specs/2026-06-06-bumper-color-detection-model-design.md`
(note: the spec describes the single-3-class replace; this plan supersedes it with the
augment reality from SP-A's as-built outcome).

---

## Performance note

Running two YOLOv8m models on the overhead frame ≈ +6 ms/frame (the bumper model's
inference). Detection is not the pipeline bottleneck (SAM2 is), so this is acceptable.
If it ever matters, the bumper model can run only on the overhead crop (where robots are),
which is already the case.

## File Structure

- Create: `web/field/detect_schema.py` — `DetectionSchema` (describes the **bumper**
  model: red/blue → robot+color), `expand_bumper_to_robot`. Pure, no app import.
- Create: `tests/test_detect_schema.py`.
- Modify: `web/field/detect_bridge.py` — `to_field_detections` takes required `colors`,
  drops `color_fn`/`frame`.
- Modify: `tests/test_detect_bridge.py`.
- Modify: `web/field/bumper.py` — delete color logic; keep geometry.
- Modify: `web/app.py` — load `self.bumper_model` + schema; robot+color from it in
  `_detect_view`, the early pass, SAM2 seed/re-prompt, the slot-map, the HUD robot branch,
  and `_detect_robots_in_frame`; delete HSV; status endpoint. Ball/hub sites untouched.

---

## Task 1: `DetectionSchema` + `expand_bumper_to_robot` (pure, TDD)

**Files:** Create `web/field/detect_schema.py`; Test `tests/test_detect_schema.py`

Same helper as before — the bumper model's `names` yield robots-with-color and (for this
2-class model) **no** fuel/hub. `split()` returns `(robots[(box,color,conf)], [], [])`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_detect_schema.py
import numpy as np
from web.field.detect_schema import DetectionSchema, expand_bumper_to_robot


def test_schema_bumper_model_two_class():
    s = DetectionSchema.from_names({0: "blue_bumper", 1: "red_bumper"})
    assert s.robot_class_color == {0: "blue", 1: "red"}
    assert s.fuel_class is None and s.hub_class is None


class _FakeDets:
    def __init__(self, boxes, ids, confs):
        self.xyxy = np.array(boxes, float); self.class_id = np.array(ids, int)
        self.confidence = np.array(confs, float)
    def __len__(self): return len(self.class_id)


def test_split_bumper_only():
    s = DetectionSchema.from_names({0: "blue_bumper", 1: "red_bumper"})
    dets = _FakeDets([(0,0,10,10),(5,5,15,15)], [1,0], [0.9,0.8])
    robots, balls, hubs = s.split(dets)
    assert robots == [((0,0,10,10),"red",0.9), ((5,5,15,15),"blue",0.8)]
    assert balls == [] and hubs == []


def test_expand_grows_upward_only():
    assert expand_bumper_to_robot((100,400,160,420),(480,640,3),factor=3.0) == (100,360.0,160,420)


def test_expand_clamps_to_frame_top():
    assert expand_bumper_to_robot((0,30,50,60),(480,640,3),factor=5.0)[1] == 0.0
```

- [ ] **Step 2: Run → fail** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `web/field/detect_schema.py` (identical to the helper specced
  earlier — `from_names` maps `red_bumper`→"red"/`blue_bumper`→"blue"/`robot`→None,
  `fuel`/`fuel_ball`→fuel, `hub_opening`/`hub`→hub; `split()`; `expand_bumper_to_robot`
  grows the top by `factor`, clamps to 0, leaves width + bottom). Pure, name-driven.
- [ ] **Step 4: Run → pass** (4 tests).
- [ ] **Step 5: Commit** — `feat(detect): DetectionSchema + bumper->robot box expansion`.

---

## Task 2: `to_field_detections` takes class colors (TDD)

**Files:** Modify `web/field/detect_bridge.py:19-28`; Test `tests/test_detect_bridge.py`

Drop `color_fn`/`frame`; require a `colors` list parallel to `robot_boxes` (the HSV path
is being deleted, so there is no fallback colorizer).

- [ ] **Step 1–4:** rewrite the tests to call
  `to_field_detections(boxes, cam, field_dims, colors=[...])` (per-box color used; missing
  → `"unknown"`); run → fail; implement the new signature
  `to_field_detections(robot_boxes, calibration, field_dims, colors, margin=1.0)`
  (`color = colors[idx] if colors[idx] else "unknown"`); run → pass. (Full code as specced
  earlier in the pre-rewrite plan; `field_adjacent_persons` unchanged.)
- [ ] **Step 5: Commit** — `feat(detect): to_field_detections takes per-box class colors`.

---

## Task 3: Load the bumper model + schema

**Files:** Modify `web/app.py` (`:118` area, `:481` area)

- [ ] **Step 1: Env + path constants** near `MODEL_PATH` (`:118`):

```python
# Existing 3-class model stays for fuel + hub.
MODEL_PATH = PROJECT_ROOT / "runs/train/hub_zone_yolo26s/weights/best.pt"
# New bumper model supplies robots + alliance color (augment). Unset/missing -> fall
# back to the existing model's robot class + no class-color (legacy behavior).
BUMPER_MODEL_PATH = Path(os.environ.get(
    "FRC_BUMPER_MODEL", str(PROJECT_ROOT / "runs/train/frc_bumpers/weights/best.pt")))
ROBOT_BOX_FROM_BUMPER_FACTOR = float(os.environ.get("ROBOT_BOX_FROM_BUMPER_FACTOR", "3.0"))
```

- [ ] **Step 2: Load + schema** after `self.model = YOLO(str(MODEL_PATH))` (`:481`):

```python
from web.field.detect_schema import DetectionSchema, expand_bumper_to_robot
self.bumper_model = None
self.det_schema = None
if BUMPER_MODEL_PATH.exists():
    self.bumper_model = YOLO(str(BUMPER_MODEL_PATH))
    self.det_schema = DetectionSchema.from_names(self.bumper_model.names)
    print(f"[detect] bumper model={BUMPER_MODEL_PATH.name} "
          f"classes={self.bumper_model.names} robots={self.det_schema.robot_class_color}")
else:
    print(f"[detect] bumper model not found at {BUMPER_MODEL_PATH}; "
          f"falling back to existing model's robot class + HSV color")
```

Add `expand_bumper_to_robot` import at module level too (used in Task 5). When
`self.bumper_model is None`, every robot+color site below keeps today's behavior (guarded
by `if self.bumper_model is not None`). This keeps the app runnable without the weights.

- [ ] **Step 3: Smoke** — load both models in a one-liner; confirm
  `bumper.names == {0:'blue_bumper',1:'red_bumper'}`.
- [ ] **Step 4: Commit** — `feat(detect): load bumper model + schema (augment)`.

---

## Task 4: Robots + color from the bumper model (detection paths)

**Files:** Modify `web/app.py` `:781-820` (`_detect_view`), `:2926-2955` (early pass),
`:4865-4900` (`_detect_robots_in_frame`)

For each, robots+color come from the **bumper model**; **ball/hub logic is NOT touched**.

- [ ] **Step 1: `_detect_view` (`:781-820`)** — run the bumper model on the view frame
  for robots; drop the existing model's robot loop. (Ball/hub aren't produced here.)

```python
        if self.bumper_model is not None:
            bres = self.bumper_model(frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
            bdets = sv.Detections.from_ultralytics(bres)
            robots, _b, _h = self.det_schema.split(bdets)
        else:
            # legacy fallback: robots from the existing model + HSV color (color=None)
            robots = [(tuple(dets.xyxy[i]), None, float(dets.confidence[i]))
                      for i in range(len(dets)) if dets.class_id[i] == 1]
        robot_boxes, robot_colors = [], []
        for (box, color, _c) in robots:
            cx, cy = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0
            if any(mx<=cx<=mx+mw and my<=cy<=my+mh for (mx,my,mw,mh) in view.masks):
                continue
            robot_boxes.append(np.array(box)); robot_colors.append(color)
        field_dets = to_field_detections(robot_boxes, cal, self.field_dims, colors=robot_colors)
```

(When falling back, `robot_colors` is all `None` → `to_field_detections` stores
`"unknown"`; fusion's territory heuristic resolves it as today. With the bumper model,
real colors flow.)

- [ ] **Step 2: Early-OCR pre-pass (`:2933`)** — same swap: build `early_boxes` +
  `early_colors` from `self.bumper_model` via `schema.split` (fallback to existing model's
  class-1 when `bumper_model is None`); pass `colors=` to `to_field_detections`. Keep the
  mask filter; filter colors in lockstep.

- [ ] **Step 3: `_detect_robots_in_frame` (`:4865`)** — this standalone helper takes a
  `model` param. Pass it the bumper model from the caller (`/api/detect-robots`, `:4988`),
  build `DetectionSchema.from_names(model.names)`, use `schema.split` + class color.
  (Update the `:4988` call site to pass `job`/processor's `bumper_model` when available,
  else the existing model.)

- [ ] **Step 4: Ball + hub — confirm UNCHANGED.** `update_ball_trajectories` (`:1942`,
  `class_id==0`) and `update_hub_positions` (`:1637`, `class_id==2`) keep reading the
  **existing** model's `detections`. Do **not** modify them. (Hub is also static per
  event; leaving the existing detector running is harmless.) Add a one-line comment at each
  noting they intentionally consume the existing 3-class model, not the bumper model.

- [ ] **Step 5: Syntax check + commit** —
  `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`;
  `feat(detect): robots+color from bumper model in detection paths (augment)`.

---

## Task 5: SAM2 seed/re-prompt from the bumper model + box expansion

**Files:** Modify `web/app.py` `:2816-2822`, `:2870-2883`, `:2905`, `:3024-3069`

SAM2 robot prompts come from the **bumper model**, expanded upward into robot boxes; obj
ids `1..N` carry seed color; post-SAM2 color from that map (no HSV).

- [ ] **Step 1: Prompt frame (`:2818-2822`)** — add `view_prompt_colors` beside
  `view_prompt_boxes`; detect with `self.bumper_model`:

```python
                                    if is_prompt and self.bumper_model is not None:
                                        res = self.bumper_model(crop, conf=0.4, verbose=False)[0]
                                        robots, _b, _h = self.det_schema.split(
                                            sv.Detections.from_ultralytics(res))
                                        for (box, color, _c) in robots:
                                            rbox = expand_bumper_to_robot(box, crop.shape,
                                                        ROBOT_BOX_FROM_BUMPER_FACTOR)
                                            view_prompt_boxes[view.name].append(
                                                tuple(v*SAM2_SCALE for v in rbox))
                                            view_prompt_colors[view.name].append(color)
                                    elif is_prompt:  # legacy fallback: existing model, class 1
                                        res = self.model(crop, conf=0.4, verbose=False)[0]
                                        for i in range(len(res.boxes)):
                                            if int(res.boxes.cls[i]) == 1:
                                                box = res.boxes.xyxy[i].cpu().numpy()*SAM2_SCALE
                                                view_prompt_boxes[view.name].append(tuple(box))
                                                view_prompt_colors[view.name].append(None)
```

- [ ] **Step 2: Re-prompt `detect_fn` (`:2877-2882`)** — use the bumper model + expand
  (fallback to existing model class-1 unexpanded when `bumper_model is None`).
- [ ] **Step 3: Store obj-id→color map (`:2905`)** —
  `self.sam2_obj_color[view.name] = {i+1: c for i,c in enumerate(view_prompt_colors[view.name])}`.
- [ ] **Step 4: Post-SAM2 color (`:3043`)** — replace the `detect_bumper_color(...)` call
  with `color = self.sam2_obj_color.get(view.name, {}).get(obj_id) or "unknown"`. Fusion
  votes across views.
- [ ] **Step 5: Confirm fusion votes color** (read `web/field/fusion.py` `_merge` ~`:54`;
  no change expected — it already weights color votes and ignores `"unknown"`).
- [ ] **Step 6: Syntax + commit** — `feat(detect): SAM2 robot prompts from bumper model (expanded) + seed color`.

---

## Task 6: HUD draw branch — robots from bumper, balls/hubs from existing

**Files:** Modify `web/app.py` `:3280-3344`

The HUD loop iterates the **existing** model's `detections` and branches `class_id==0/1/2`.
Robots (class 1 of the existing model) should no longer be drawn from there — robots are
drawn from the tracked slots / bumper detections. Cleanest minimal change:

- [ ] **Step 1:** In the existing-`detections` loop, **skip the robot class** (it's now
  superseded): `elif class_id == 1: continue` (robots are drawn via the tracked-slot HUD,
  `_draw_bumper_hud`, already invoked for tracked slots). Keep `class_id==0` (ball) and the
  `else` (hub) branches drawing as today.
- [ ] **Step 2:** Robots are drawn from `detection_to_tracked` / track slots. Confirm the
  per-slot HUD (`_draw_bumper_hud`, fed by `track_by_slot`) still renders every tracked
  robot with its alliance color (now sourced from the bumper-model color via fusion).
  Read `:3140-3210` (slot-map build) — the slot map must be built from **bumper-model**
  robot detections now (see Task 4/Step on `:3187`/`:3204`).
- [ ] **Step 3: Slot map (`:3187`, `:3204`)** — these `class_id[di] != 1` filters run over
  `detections` (existing model). They must instead map **bumper-model** robot detections to
  slots (the boxes that actually seeded tracks). Build the slot map from the bumper
  detections used in `_detect_view`/SAM2, not the existing model's class-1 boxes. Read the
  surrounding code and thread the bumper robot boxes through; this is the fiddliest edit —
  verify HUD labels land on the right robots on a real clip (Task 8).
- [ ] **Step 4: Syntax + commit** — `feat(detect): HUD robots from bumper-model tracks; balls/hubs from existing model`.

---

## Task 7: Delete the HSV color path + status endpoint

**Files:** Modify `web/field/bumper.py`, `web/app.py` (`:60`, `:1691`, `:4251`, `:3638`)

- [ ] **Step 1:** Delete `classify_bumper_color`, `_color_masks`,
  `_largest_rectangular_band_ratio`, `BumperColor` from `web/field/bumper.py`. **Keep**
  `bumper_band_box`, `bumper_crop`, `_clamp_box` (HUD band + VLM crop).
- [ ] **Step 2:** Fix the import (`:60`) → `from web.field.bumper import bumper_band_box, bumper_crop`.
- [ ] **Step 3:** Delete `detect_bumper_color` (`:1691`) and `_detect_bumper_color_static`
  (`:4251`). Verify: `grep -n "detect_bumper_color\|classify_bumper_color" web/app.py`
  returns nothing (all callers rewired in Tasks 4–6).
- [ ] **Step 4: `/api/status` (`:3638`)** — report both models. Replace the hardcoded
  `model_classes` with the existing model's classes plus the bumper model's, e.g.
  `{"fuel_hub_model": {0:"fuel_ball",1:"robot",2:"hub_opening"}, "bumper_model": (job.bumper_model.names if a job with a bumper model exists else None)}` — derive from a
  job via `next(reversed(jobs.values()), None)`; fall back to the legacy dict.
- [ ] **Step 5: Run unit suites + syntax check + commit** —
  `pytest tests/test_detect_schema.py tests/test_detect_bridge.py -q`;
  `refactor(detect): delete HSV color path; color is the bumper class`.

---

## Task 8: Integration verification on real footage

**Files:** none (verification). Uses `runs/train/frc_bumpers/weights/best.pt` (present).

- [ ] **Step 1:** Unit suites green.
- [ ] **Step 2:** Run one match (`web/uploads/*.mp4`). Confirm startup logs the bumper
  model loaded; both models run.
- [ ] **Step 3: Acceptance points:**
  1. Robots detected + tracked from the bumper model (count comparable to legacy).
  2. **Alliance colors correct and stable** (bumper class via fusion) — fewer errors than
     the old HSV path.
  3. **SAM2 masks cover whole robots** (gallery crops whole-robot) — proves box expansion.
  4. **Fuel still detected** (existing model, unchanged) and ball trajectories work.
  5. Scoring runs (hub from existing model / static).
- [ ] **Step 4: Field-position drift check** vs legacy on the same clip.
- [ ] **Step 5:** Record results; if good, this is the new default (bumper model auto-loads
  from the default path). Commit any notes.

---

## Done when

- `detect_schema` + `detect_bridge` unit suites green; `grep` shows no HSV color path left.
- A real match: robots+color from the bumper model (stable, correct), whole-robot SAM2
  masks, fuel still detected by the existing model, scoring intact.
- App still runs with `FRC_BUMPER_MODEL` unset (falls back to legacy robot+HSV path).

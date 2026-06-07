# SP-B: Integrate the Bumper/Fuel Detection Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pipeline drive detection from a bumper model (`red_bumper`/`blue_bumper`/`fuel`), taking alliance color from the detection class, expanding bumper boxes into robot boxes for SAM2 seeding (so masks + appearance gallery stay whole-robot), and relying on static hub positions — while staying backward-compatible with the legacy `robot`/`fuel_ball`/`hub_opening` model.

**Architecture:** A `DetectionSchema` auto-detected from `model.names` describes which class ids are robots (and their fixed color), fuel, and hub. A small set of pure helpers (`DetectionSchema.split`, `expand_bumper_to_robot`) centralize the class-filtering logic currently scattered across `app.py`. The detection path passes per-box colors into `to_field_detections` (new optional `colors=` param); the SAM2 path seeds from expanded boxes and carries seed-time alliance per object id (HSV fallback for re-prompted tracks). Hubs come from `event_hub_positions` (existing calibration).

**Tech Stack:** Python, `ultralytics`/`supervision` detections, existing `web/field/` modules. Tests run with `.venv/bin/python -m pytest` (base conda python lacks pytest — project memory).

**Contract consumed (from SP-A):** `runs/train/frc_bumpers/weights/best.pt` with `names == {0: red_bumper, 1: blue_bumper, 2: fuel}`. **This plan's pure-helper tasks (1–4) do not need the weights and can be built first.**

**Spec:** `docs/superpowers/specs/2026-06-06-bumper-color-detection-model-design.md`

---

## File Structure

- Create: `web/field/detect_schema.py` — `DetectionSchema` (auto-detect from `names`, `.split(dets)`), `expand_bumper_to_robot(box, frame_shape, factor)`. Pure, no app import.
- Create: `tests/test_detect_schema.py` — unit tests for the above.
- Modify: `web/field/detect_bridge.py` — add optional `colors=` parallel list to `to_field_detections` (backward compatible).
- Modify: `tests/test_detect_bridge.py` — add a per-box-color test (existing tests must stay green).
- Modify: `web/app.py` — model load + schema; `_detect_view` color/filtering; SAM2 seed + re-prompt box expansion + per-obj color map + post-SAM2 coloring; `/api/status` classes; hub warning; untracked-robot HUD color; `FRC_DETECT_MODEL` env + default.

> `DetectionSchema` lives in `web/field/` (next to `detect_bridge.py`, `bumper.py`) — it is consumed by the field pipeline, not by training. It defines its own 3-line name normalizer rather than importing from `web/training/` so the two packages stay decoupled.

---

## Task 1: `DetectionSchema` (pure, TDD)

**Files:**
- Create: `web/field/detect_schema.py`
- Test: `tests/test_detect_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detect_schema.py
import numpy as np
from web.field.detect_schema import DetectionSchema, expand_bumper_to_robot


def test_schema_legacy_model():
    s = DetectionSchema.from_names({0: "fuel_ball", 1: "robot", 2: "hub_opening"})
    assert s.robot_class_color == {1: None}     # generic robot, color unknown
    assert s.fuel_class == 0
    assert s.hub_class == 2
    assert s.has_hub is True


def test_schema_bumper_model():
    s = DetectionSchema.from_names({0: "red_bumper", 1: "blue_bumper", 2: "fuel"})
    assert s.robot_class_color == {0: "red", 1: "blue"}
    assert s.fuel_class == 2
    assert s.hub_class is None
    assert s.has_hub is False


class _FakeDets:
    """Minimal supervision.Detections stand-in."""
    def __init__(self, boxes, class_ids, confs):
        self.xyxy = np.array(boxes, dtype=float)
        self.class_id = np.array(class_ids, dtype=int)
        self.confidence = np.array(confs, dtype=float)
    def __len__(self):
        return len(self.class_id)


def test_split_buckets_bumper_dets_with_color():
    s = DetectionSchema.from_names({0: "red_bumper", 1: "blue_bumper", 2: "fuel"})
    dets = _FakeDets(
        boxes=[(0, 0, 10, 10), (5, 5, 15, 15), (1, 1, 2, 2)],
        class_ids=[0, 1, 2],
        confs=[0.9, 0.8, 0.7],
    )
    robots, balls, hubs = s.split(dets)
    assert robots == [((0, 0, 10, 10), "red", 0.9), ((5, 5, 15, 15), "blue", 0.8)]
    assert balls == [(1, 1, 2, 2)]
    assert hubs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py -v`
Expected: FAIL (`ModuleNotFoundError: web.field.detect_schema`).

- [ ] **Step 3: Write the implementation**

```python
# web/field/detect_schema.py
"""Auto-detect a detector's class scheme from model.names so the pipeline works
with both the legacy (robot/fuel_ball/hub_opening) and the bumper
(red_bumper/blue_bumper/fuel) models without code changes. Pure — no app import."""
from __future__ import annotations
import re
from dataclasses import dataclass


def _norm(name: str) -> str:
    s = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


@dataclass
class DetectionSchema:
    robot_class_color: dict[int, str | None]  # class_id -> "red"/"blue"/None(generic)
    fuel_class: int | None
    hub_class: int | None
    names: dict[int, str]

    @classmethod
    def from_names(cls, names: dict[int, str]) -> "DetectionSchema":
        names = {int(k): v for k, v in names.items()}
        robot_color: dict[int, str | None] = {}
        fuel = hub = None
        for cid, raw in names.items():
            n = _norm(raw)
            if n == "red_bumper":
                robot_color[cid] = "red"
            elif n == "blue_bumper":
                robot_color[cid] = "blue"
            elif n == "robot":
                robot_color[cid] = None
            elif n in ("fuel", "fuel_ball"):
                fuel = cid
            elif n in ("hub_opening", "hub"):
                hub = cid
        return cls(robot_color, fuel, hub, names)

    @property
    def has_hub(self) -> bool:
        return self.hub_class is not None

    def split(self, dets):
        """Bucket an sv.Detections into (robots, balls, hubs).
        robots: list of (box, color_or_None, conf); balls/hubs: list of box."""
        robots, balls, hubs = [], [], []
        confs = getattr(dets, "confidence", None)
        for i in range(len(dets)):
            cid = int(dets.class_id[i])
            box = tuple(float(v) for v in dets.xyxy[i])
            conf = float(confs[i]) if confs is not None else 1.0
            if cid in self.robot_class_color:
                robots.append((box, self.robot_class_color[cid], conf))
            elif cid == self.fuel_class:
                balls.append(box)
            elif cid == self.hub_class:
                hubs.append(box)
        return robots, balls, hubs


def expand_bumper_to_robot(box, frame_shape, factor: float = 3.0):
    """Grow a bumper box UPWARD into an estimated whole-robot box for SAM2 seeding.
    Width and the bottom edge (ground-contact point) are unchanged; the top is
    raised by (factor-1)*height and clamped to the frame. frame_shape is (H, W, ...)."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    new_y1 = y2 - h * factor
    if frame_shape is not None:
        new_y1 = max(0.0, new_y1)
    else:
        new_y1 = max(0.0, new_y1)
    return (x1, new_y1, x2, y2)
```

- [ ] **Step 4: Add `expand_bumper_to_robot` tests**

Append to `tests/test_detect_schema.py`:

```python
def test_expand_grows_upward_only():
    # bumper box 20px tall at bottom of a 480px frame
    box = (100, 400, 160, 420)
    out = expand_bumper_to_robot(box, (480, 640, 3), factor=3.0)
    assert out[0] == 100 and out[2] == 160          # width unchanged
    assert out[3] == 420                            # bottom edge unchanged
    assert out[1] == 420 - 20 * 3.0                 # top raised by factor*height
    assert out[1] == 360


def test_expand_clamps_to_frame_top():
    box = (0, 30, 50, 60)   # height 30, factor 5 -> y1 would be 60-150 = -90
    out = expand_bumper_to_robot(box, (480, 640, 3), factor=5.0)
    assert out[1] == 0.0    # clamped
```

- [ ] **Step 5: Run all schema tests**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add web/field/detect_schema.py tests/test_detect_schema.py
git commit -m "feat(detect): DetectionSchema auto-detect + bumper->robot box expansion"
```

---

## Task 2: `to_field_detections` per-box color (TDD)

**Files:**
- Modify: `web/field/detect_bridge.py:19-28`
- Test: `tests/test_detect_bridge.py`

- [ ] **Step 1: Write the failing test (extend existing file)**

Append to `tests/test_detect_bridge.py`:

```python
def test_explicit_per_box_color_overrides_color_fn():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, 4.0, 0.0))
    box = (u - 30, v - 120, u + 30, v)
    # color_fn would say "blue"; explicit per-box color says "red" and must win.
    dets = to_field_detections([box], cam, None, (fm.length_m, fm.width_m),
                               color_fn=_color_stub, colors=["red"])
    assert len(dets) == 1 and dets[0].color == "red"


def test_none_per_box_color_falls_back_to_color_fn():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, 4.0, 0.0))
    box = (u - 30, v - 120, u + 30, v)
    dets = to_field_detections([box], cam, None, (fm.length_m, fm.width_m),
                               color_fn=_color_stub, colors=[None])
    assert dets[0].color == "blue"   # fell back to color_fn
```

- [ ] **Step 2: Run to verify new tests fail, old tests still pass**

Run: `.venv/bin/python -m pytest tests/test_detect_bridge.py -v`
Expected: the two new tests FAIL (`to_field_detections() got an unexpected keyword argument 'colors'`); the original three PASS.

- [ ] **Step 3: Implement the `colors=` param**

Modify `web/field/detect_bridge.py` `to_field_detections`:

```python
def to_field_detections(robot_boxes, calibration, frame, field_dims, color_fn,
                        margin=1.0, colors=None):
    out = []
    for idx, box in enumerate(robot_boxes):
        uv = _bottom_center(box)
        xy = calibration.image_to_field(uv)
        if xy is None or not _on_field(xy, field_dims, margin):
            continue
        explicit = colors[idx] if (colors is not None and idx < len(colors)) else None
        color = explicit if explicit else color_fn(frame, box)
        out.append(FieldDetection(pos=xy, color=color, box=tuple(box), score=1.0))
    return out
```

- [ ] **Step 4: Run all detect_bridge tests**

Run: `.venv/bin/python -m pytest tests/test_detect_bridge.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add web/field/detect_bridge.py tests/test_detect_bridge.py
git commit -m "feat(detect): optional per-box colors in to_field_detections (backward compatible)"
```

---

## Task 3: Load schema at model init

**Files:**
- Modify: `web/app.py` near `MODEL_PATH` (118) and model load (`:481`)

- [ ] **Step 1: Add the env-selectable model path + factor constant**

Near `MODEL_PATH` (`app.py:118`), add:

```python
# Detector selection. Default stays the legacy model until the bumper model
# (SP-A) is validated; set FRC_DETECT_MODEL to switch (see EVAL.md gate).
_DEFAULT_MODEL = PROJECT_ROOT / "runs/train/hub_zone_yolo26s/weights/best.pt"
MODEL_PATH = Path(os.environ.get("FRC_DETECT_MODEL", str(_DEFAULT_MODEL)))
# Bumper-box -> estimated robot-box vertical growth factor for SAM2 seeding.
ROBOT_BOX_FROM_BUMPER_FACTOR = float(os.environ.get("ROBOT_BOX_FROM_BUMPER_FACTOR", "3.0"))
```

(Replace the existing `MODEL_PATH = ...` line. `os` and `Path` are already imported.)

- [ ] **Step 2: Build the schema right after the model loads**

At `app.py:481` (`self.model = YOLO(str(MODEL_PATH))`), add immediately after:

```python
from web.field.detect_schema import DetectionSchema
self.det_schema = DetectionSchema.from_names(self.model.names)
print(f"[detect] model={MODEL_PATH.name} classes={self.model.names} "
      f"robots={self.det_schema.robot_class_color} hub={self.det_schema.has_hub}")
```

- [ ] **Step 3: Verify import + schema build (smoke)**

Run: `.venv/bin/python -c "from ultralytics import YOLO; from web.field.detect_schema import DetectionSchema; m=YOLO('runs/train/hub_zone_yolo26s/weights/best.pt'); print(DetectionSchema.from_names(m.names))"`
Expected: prints a `DetectionSchema` with `robot_class_color={1: None}`, `hub_class=2`.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(detect): env-selectable detector + DetectionSchema at load"
```

---

## Task 4: Wire the pre-SAM2 detection path (`_detect_view`)

**Files:**
- Modify: `web/app.py:796-810` (and the parallel call at `:2952`)

- [ ] **Step 1: Replace the hardcoded robot filter with schema split + per-box color**

In `_detect_view` (`app.py:796-810`), replace the `robot_boxes` loop and the `to_field_detections` call:

```python
        robots, _balls, _hubs = self.det_schema.split(dets)
        robot_boxes, robot_colors = [], []
        for (box, color, _conf) in robots:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            if any(mx <= cx <= mx + mw and my <= cy <= my + mh
                   for (mx, my, mw, mh) in view.masks):
                continue
            robot_boxes.append(np.array(box))
            robot_colors.append(color)
        field_dets = to_field_detections(robot_boxes, cal, frame, self.field_dims,
                                         color_fn=self.detect_bumper_color,
                                         colors=robot_colors)
```

> `colors` carries `None` for the legacy model (so HSV `color_fn` runs, unchanged behavior) and `"red"`/`"blue"` for the bumper model (HSV bypassed).

- [ ] **Step 2: Apply the same split at the other `to_field_detections` caller**

Inspect `app.py:2952` (the second `to_field_detections` call). Wherever its `robot_boxes` are built from `class_id == 1`, replace with `self.det_schema.split(...)` and pass `colors=`. (Read the surrounding ~30 lines first; mirror the Step 1 pattern.)

- [ ] **Step 3: Run the field/detect test suite**

Run: `.venv/bin/python -m pytest tests/test_detect_bridge.py tests/test_detect_schema.py -v`
Expected: PASS. (These cover the helpers; app wiring is verified in Task 7.)

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(detect): _detect_view uses DetectionSchema + class color"
```

---

## Task 5: Wire the SAM2 seed/re-prompt + post-SAM2 coloring

**Files:**
- Modify: `web/app.py:2816-2822` (prompt boxes), `:2870-2883` (detect_fn), `:3040-3069` (post-SAM2 color)

This is the box-expansion + color-carry task. SAM2 assigns obj ids `1..N` in prompt-box order, so the caller maps obj id → seed color directly; re-prompted tracks fall back to HSV.

- [ ] **Step 1: Seed prompts from expanded boxes + record seed colors**

Near where `view_prompt_boxes` is collected, add a parallel `view_prompt_colors` dict (same init site). Replace the prompt-frame detection block (`app.py:2818-2822`):

```python
                                    if is_prompt:
                                        res = self.model(crop, conf=0.4, verbose=False)[0]
                                        pdets = sv.Detections.from_ultralytics(res)
                                        robots, _b, _h = self.det_schema.split(pdets)
                                        for (box, color, _c) in robots:
                                            rbox = expand_bumper_to_robot(
                                                box, crop.shape, ROBOT_BOX_FROM_BUMPER_FACTOR)
                                            scaled = tuple(v * SAM2_SCALE for v in rbox)
                                            view_prompt_boxes[view.name].append(scaled)
                                            view_prompt_colors[view.name].append(color)
```

Add `from web.field.detect_schema import expand_bumper_to_robot` to the imports near the top of `app.py` (module level, beside the other `web.field` imports ~line 60).

> For the legacy model, `color` is `None` (no behavior change downstream); `expand_bumper_to_robot` on a whole-robot box grows it further upward, which is harmless for SAM2 seeding but unnecessary. Guard it: only expand when the schema is a bumper schema —
> `rbox = expand_bumper_to_robot(box, crop.shape, ROBOT_BOX_FROM_BUMPER_FACTOR) if color is not None else box`
> (color is not None ⇔ a colored bumper class ⇔ bumper model.)

- [ ] **Step 2: Expand boxes in the re-prompt `detect_fn` too**

Replace the `detect_fn` body (`app.py:2877-2882`):

```python
                                res = self.model(img, conf=0.4, verbose=False)[0]
                                rdets = sv.Detections.from_ultralytics(res)
                                robots, _b, _h = self.det_schema.split(rdets)
                                boxes = []
                                for (box, color, _c) in robots:
                                    boxes.append(expand_bumper_to_robot(box, img.shape,
                                                 ROBOT_BOX_FROM_BUMPER_FACTOR)
                                                 if color is not None else box)
                                return boxes
```

- [ ] **Step 3: Build the obj-id→color map and use it in post-SAM2 coloring**

Where `view_tracks` is stored (`app.py:2905`, `self.sam2_raw[view.name] = view_tracks`), also store the seed color map:

```python
                        # obj ids are 1..N in prompt order; map to seed alliance.
                        self.sam2_obj_color = getattr(self, "sam2_obj_color", {})
                        self.sam2_obj_color[view.name] = {
                            i + 1: c for i, c in enumerate(view_prompt_colors[view.name])
                        }
```

Then at the post-SAM2 coloring site (`app.py:3043`), replace:

```python
                            color = self.detect_bumper_color(view_frame, np.array([fx1, fy1, fx2, fy2]))
```

with:

```python
                            seed_color = self.sam2_obj_color.get(view.name, {}).get(obj_id)
                            color = seed_color or self.detect_bumper_color(
                                view_frame, np.array([fx1, fy1, fx2, fy2]))
```

> Read the loop around `app.py:3040-3069` to confirm the local variable names (`obj_id`, `view`, `fx1..fy2`). Adjust if they differ. Re-prompted tracks (obj id not in the seed map) keep the existing HSV path — no regression versus today.

- [ ] **Step 4: Quick syntax/import check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add web/app.py
git commit -m "feat(detect): SAM2 seeds from expanded robot boxes; seed-time alliance color"
```

---

## Task 6: Status endpoint, hub warning, untracked-robot HUD color

**Files:**
- Modify: `web/app.py:3638` (`/api/status` model_classes), hub-position warning, `:3326` (untracked HUD color)

- [ ] **Step 1: Drive `/api/status` model_classes from the schema**

At `app.py:3638`, replace the hardcoded dict:

```python
        "model_classes": {int(k): v for k, v in (job.det_schema.names.items()
                          if job and getattr(job, "det_schema", None) else
                          {0: "fuel_ball", 1: "robot", 2: "hub_opening"}.items())},
```

> Read the surrounding handler to get the correct reference to the running job/processor object (it may not be named `job`). Use whatever the handler already uses to reach the processor; fall back to the legacy dict when no job is active.

- [ ] **Step 2: Warn when a hubless schema has no static hubs**

In `update_hub_positions` (`app.py:1633`), at the top, add:

```python
        if not self.det_schema.has_hub and not event_hub_positions:
            if not getattr(self, "_warned_no_hub", False):
                print("[detect] WARNING: detector has no hub class and no static "
                      "event_hub_positions are set — set hubs via the calibration "
                      "step or scoring attribution will be degraded.")
                self._warned_no_hub = True
```

> Confirm `event_hub_positions` is in scope here (it is referenced as a global at `:1675`). If `self.det_schema` isn't reachable in this method, it is — `update_hub_positions` is a method on the processor.

- [ ] **Step 3: Untracked-robot HUD color via schema (cosmetic)**

The annotation block at `app.py:3326` calls `detect_bumper_color` for an untracked detection's display color. Leave the legacy path, but prefer the class color when available. Read ~`app.py:3300-3335`: the loop iterates `detections` by index `i` with `class_id`. Where it currently does `alliance = self.detect_bumper_color(...)`, prefer the schema's robot color:

```python
                            alliance = (self.det_schema.robot_class_color.get(int(class_id))
                                        or self.detect_bumper_color(overhead_frame, detections.xyxy[i]))
```

> This is display-only; if uncertain about the variable names, skip this sub-step — it does not affect tracking or scoring. Note in the commit if skipped.

- [ ] **Step 4: Note the second HSV path (no code change)**

`_detect_bumper_color_static` (`app.py:4251`, called at `:4881`) stays on HSV intentionally (not in the core detect→track path). Add a one-line comment at `:4251` so it isn't mistaken for dead code:

```python
def _detect_bumper_color_static(frame: np.ndarray, box) -> str:
    # Legacy HSV path; intentionally NOT schema-driven (used outside the main
    # detect->track loop). See SP-B spec "Detection normalization helper".
```

- [ ] **Step 5: Syntax check + commit**

Run: `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`

```bash
git add web/app.py
git commit -m "feat(detect): schema-driven status classes, hubless warning, HUD color"
```

---

## Task 7: Integration verification on real footage

**Files:** none (verification only)

Requires the SP-A artifact (`runs/train/frc_bumpers/weights/best.pt`) and static hub positions set for the test event via the calibration step.

- [ ] **Step 1: Confirm the full unit suite is green**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py tests/test_detect_bridge.py tests/test_class_remap.py -v`
Expected: all PASS.

- [ ] **Step 2: Run one match with the bumper model**

Set the env and run the app's analysis on a single match (use the acceptance-harness clip per project memory `acceptance-harness-download`; ensure the event has static hub positions via `/calibrate`):

```bash
FRC_DETECT_MODEL=runs/train/frc_bumpers/weights/best.pt \
  .venv/bin/python -m <the project's usual analysis entrypoint or the web app>
```

> Use the project's standard run path (see the `run` skill / how matches are normally analyzed). Confirm the startup log prints `[detect] model=best.pt classes={0: 'red_bumper', ...} hub=False`.

- [ ] **Step 3: Verify the four spec acceptance points**

Inspect the analyzed match (HUD/replay):
1. **Tracks seed** — robots are picked up and tracked (track count comparable to legacy).
2. **Whole-robot masks** — SAM2 masks cover whole robots, not just bumper bands (the appearance gallery crops look like whole robots). This confirms `expand_bumper_to_robot` worked.
3. **Alliance colors correct** — red robots tagged red, blue tagged blue, with fewer color errors than the HSV baseline.
4. **Scoring runs** — match scores compute with static hubs (no crash from the missing hub class).

- [ ] **Step 4: Field-position drift check (spec modeling-shift note)**

Spot-compare a few robots' field positions vs the legacy model on the same clip; confirm no meaningful drift from using the bumper-box bottom as the ground-contact point.

- [ ] **Step 5: Record results + flip the default (if GO)**

If acceptance passes and SP-A's `EVAL.md` is GO, change the default in `app.py` Step (Task 3 Step 1) so `_DEFAULT_MODEL` points at `runs/train/frc_bumpers/weights/best.pt` (keeping `FRC_DETECT_MODEL` as the override). Otherwise leave the legacy default and record what failed.

```bash
git add web/app.py
git commit -m "feat(detect): default to bumper model after acceptance (or: keep legacy, see notes)"
```

---

## Done when

- Unit suites for `detect_schema`, `detect_bridge`, `class_remap` are green.
- A real match analyzed with `FRC_DETECT_MODEL=...best.pt` seeds tracks, produces whole-robot masks, assigns correct alliance colors, and scores with static hubs.
- The legacy model still loads and behaves unchanged when `FRC_DETECT_MODEL` is unset (backward compatibility preserved).

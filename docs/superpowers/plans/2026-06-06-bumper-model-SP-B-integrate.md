# SP-B: Integrate the Bumper/Fuel Detection Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive detection from a bumper model (`red_bumper`/`blue_bumper`/`fuel`), taking alliance color from the detection class and resolving it through the existing multi-camera fusion vote; expand bumper boxes into robot boxes for SAM2 seeding (so masks + appearance gallery stay whole-robot); rely on static hubs; and **delete the HSV color path entirely**.

**Architecture:** A `DetectionSchema` auto-detected from `model.names` says which class ids are robots (and their fixed color), fuel, and hub. Every class-id filter in the codebase routes through `schema.split()` — this is **correctness-critical** because class 0 changes meaning (`fuel_ball` → `red_bumper`). Per-detection class color flows into `FieldDetection.color` and is reconciled by `web/field/fusion.py`'s existing color vote (no fusion change). SAM2 seeds from upward-expanded boxes and carries seed/re-prompt class color per object id. The HSV classifier and its three call sites are removed.

**Tech Stack:** Python, `ultralytics`/`supervision`, existing `web/field/` modules. Tests: `.venv/bin/python -m pytest` (base conda python lacks pytest — project memory).

**Contract consumed (from SP-A):** `runs/train/frc_bumpers/weights/best.pt`, `names == {0: red_bumper, 1: blue_bumper, 2: fuel}`. **Tasks 1–2 are pure helpers buildable now, before weights exist.**

**Spec:** `docs/superpowers/specs/2026-06-06-bumper-color-detection-model-design.md` (revised: full HSV removal, fusion-vote color, all class-id sites through schema).

---

## File Structure

- Create: `web/field/detect_schema.py` — `DetectionSchema` (+`split`), `expand_bumper_to_robot`. Pure, no app import.
- Create: `tests/test_detect_schema.py`.
- Modify: `web/field/detect_bridge.py` — `to_field_detections` takes required `colors`, drops `color_fn`/`frame`.
- Modify: `tests/test_detect_bridge.py` — update to the new signature.
- Modify: `web/field/bumper.py` — delete color logic (`classify_bumper_color`, `_color_masks`, `_largest_rectangular_band_ratio`, `BumperColor`); keep geometry (`bumper_band_box`, `bumper_crop`, `_clamp_box`).
- Modify: `web/app.py` — schema at load; route all class-id sites through schema; SAM2 seed/re-prompt expansion + color map; delete HSV methods; status/hub/HUD; `FRC_DETECT_MODEL` env + default flip.
- Modify: `web/static/index.html` — none required (no UI change in SP-B).

---

## Task 1: `DetectionSchema` + `expand_bumper_to_robot` (pure, TDD)

**Files:** Create `web/field/detect_schema.py`; Test `tests/test_detect_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_detect_schema.py
import numpy as np
from web.field.detect_schema import DetectionSchema, expand_bumper_to_robot


def test_schema_legacy_model():
    s = DetectionSchema.from_names({0: "fuel_ball", 1: "robot", 2: "hub_opening"})
    assert s.robot_class_color == {1: None}
    assert s.fuel_class == 0 and s.hub_class == 2 and s.has_hub is True


def test_schema_bumper_model():
    s = DetectionSchema.from_names({0: "red_bumper", 1: "blue_bumper", 2: "fuel"})
    assert s.robot_class_color == {0: "red", 1: "blue"}
    assert s.fuel_class == 2 and s.hub_class is None and s.has_hub is False


class _FakeDets:
    def __init__(self, boxes, class_ids, confs):
        self.xyxy = np.array(boxes, dtype=float)
        self.class_id = np.array(class_ids, dtype=int)
        self.confidence = np.array(confs, dtype=float)
    def __len__(self):
        return len(self.class_id)


def test_split_buckets_bumper_dets_with_color_and_fuel_is_class_2():
    # CORRECTNESS-CRITICAL: under the bumper schema class 0 is a RED ROBOT, and fuel is
    # class 2 — the opposite of the legacy schema. split() must bucket by schema, not id.
    s = DetectionSchema.from_names({0: "red_bumper", 1: "blue_bumper", 2: "fuel"})
    dets = _FakeDets([(0,0,10,10),(5,5,15,15),(1,1,2,2)], [0,1,2], [0.9,0.8,0.7])
    robots, balls, hubs = s.split(dets)
    assert robots == [((0,0,10,10),"red",0.9), ((5,5,15,15),"blue",0.8)]
    assert balls == [(1,1,2,2)]
    assert hubs == []


def test_split_legacy_schema_still_works():
    s = DetectionSchema.from_names({0: "fuel_ball", 1: "robot", 2: "hub_opening"})
    dets = _FakeDets([(0,0,9,9),(1,1,2,2),(3,3,4,4)], [1,0,2], [0.9,0.5,0.5])
    robots, balls, hubs = s.split(dets)
    assert robots == [((0,0,9,9), None, 0.9)]
    assert balls == [(1,1,2,2)] and hubs == [(3,3,4,4)]


def test_expand_grows_upward_only():
    out = expand_bumper_to_robot((100,400,160,420), (480,640,3), factor=3.0)
    assert out == (100, 360.0, 160, 420)   # width + bottom unchanged, top raised 3x height


def test_expand_clamps_to_frame_top():
    out = expand_bumper_to_robot((0,30,50,60), (480,640,3), factor=5.0)
    assert out[1] == 0.0
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py -v`
Expected: FAIL (`ModuleNotFoundError: web.field.detect_schema`).

- [ ] **Step 3: Implement**

```python
# web/field/detect_schema.py
"""Auto-detect a detector's class scheme from model.names so the pipeline is driven by
class meaning, not hardcoded ids. Pure — no app import."""
from __future__ import annotations
import re
from dataclasses import dataclass


def _norm(name: str) -> str:
    s = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


@dataclass
class DetectionSchema:
    robot_class_color: dict[int, str | None]   # class_id -> "red"/"blue"/None(generic)
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
        """Bucket sv.Detections into (robots, balls, hubs). robots: (box, color|None, conf)."""
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
    Width and bottom edge (ground contact) unchanged; top raised and clamped to frame."""
    x1, y1, x2, y2 = box
    new_y1 = max(0.0, y2 - (y2 - y1) * factor)
    return (x1, new_y1, x2, y2)
```

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py -v` → PASS (6 tests).

- [ ] **Step 5: Reconcile `_norm` with SP-A names** — confirm the trained model's `names` normalize to `red_bumper`/`blue_bumper`/`fuel`; if SP-A used different strings, extend `from_names` + add a test.

- [ ] **Step 6: Commit**

```bash
git add web/field/detect_schema.py tests/test_detect_schema.py
git commit -m "feat(detect): DetectionSchema (class->color/fuel/hub) + bumper->robot box expansion"
```

---

## Task 2: `to_field_detections` takes class colors (TDD)

**Files:** Modify `web/field/detect_bridge.py:19-28`; Test `tests/test_detect_bridge.py`

The HSV `color_fn` is being deleted, so `to_field_detections` no longer needs `color_fn` or `frame`; color is supplied per box.

- [ ] **Step 1: Rewrite the tests to the new signature**

Replace the body of `tests/test_detect_bridge.py` (keep imports + `truth_pinhole_camera`):

```python
import numpy as np
from web.field.field_model import FieldModel
from web.field.detect_bridge import to_field_detections, field_adjacent_persons
from tests._field_synth import truth_pinhole_camera


def test_box_maps_to_field_point_with_class_color():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    gx, gy = 8.0, 4.0
    u, v = cam.field_to_image((gx, gy, 0.0))
    box = (u - 30, v - 120, u + 30, v)
    dets = to_field_detections([box], cam, (fm.length_m, fm.width_m), colors=["blue"])
    assert len(dets) == 1
    assert np.hypot(dets[0].pos[0] - gx, dets[0].pos[1] - gy) < 0.10
    assert dets[0].color == "blue"


def test_per_box_colors_are_independent():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    boxes, colors = [], []
    for (gx, gy), c in [((6.0, 3.0), "red"), ((9.0, 5.0), "blue")]:
        u, v = cam.field_to_image((gx, gy, 0.0))
        boxes.append((u - 30, v - 120, u + 30, v)); colors.append(c)
    dets = to_field_detections(boxes, cam, (fm.length_m, fm.width_m), colors=colors)
    assert [d.color for d in dets] == ["red", "blue"]


def test_off_field_box_dropped():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((30.0, 4.0, 0.0))
    box = (u - 30, v - 120, u + 30, v)
    assert to_field_detections([box], cam, (fm.length_m, fm.width_m), colors=["red"]) == []


def test_field_adjacent_person_filter():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((8.0, -0.3, 0.0)); near = (u-25, v-150, u+25, v)
    uc, vc = cam.field_to_image((40.0, 4.0, 0.0)); crowd = (uc-25, vc-150, uc+25, vc)
    keep = field_adjacent_persons([near, crowd], cam, (fm.length_m, fm.width_m), margin=2.0)
    assert near in keep and crowd not in keep
```

- [ ] **Step 2: Run — verify fail**

Run: `.venv/bin/python -m pytest tests/test_detect_bridge.py -v`
Expected: FAIL (`to_field_detections()` still has the old `color_fn`/`frame` signature → `TypeError`/unexpected kwarg).

- [ ] **Step 3: Implement the new signature**

Replace `to_field_detections` in `web/field/detect_bridge.py`:

```python
def to_field_detections(robot_boxes, calibration, field_dims, colors, margin=1.0):
    """robot_boxes[i] is colored by colors[i] (the detection's class color: 'red'/'blue'
    /'unknown'/None). No HSV — color is a property of the detection."""
    out = []
    for idx, box in enumerate(robot_boxes):
        uv = _bottom_center(box)
        xy = calibration.image_to_field(uv)
        if xy is None or not _on_field(xy, field_dims, margin):
            continue
        color = colors[idx] if (colors is not None and idx < len(colors) and colors[idx]) else "unknown"
        out.append(FieldDetection(pos=xy, color=color, box=tuple(box), score=1.0))
    return out
```

(Drop the now-unused `frame` and `color_fn` params. `field_adjacent_persons` is unchanged.)

- [ ] **Step 4: Run — verify pass**

Run: `.venv/bin/python -m pytest tests/test_detect_bridge.py -v` → PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add web/field/detect_bridge.py tests/test_detect_bridge.py
git commit -m "feat(detect): to_field_detections takes per-box class colors (drop HSV color_fn)"
```

---

## Task 3: Load schema + env knobs at model init

**Files:** Modify `web/app.py` (`:118`, `:481`)

- [ ] **Step 1: Env-selectable model + factor**

Replace `MODEL_PATH = ...` (`app.py:118`):

```python
_DEFAULT_MODEL = PROJECT_ROOT / "runs/train/hub_zone_yolo26s/weights/best.pt"
# Flip _DEFAULT_MODEL to the bumper model in Task 8 after acceptance. FRC_DETECT_MODEL
# overrides at runtime.
MODEL_PATH = Path(os.environ.get("FRC_DETECT_MODEL", str(_DEFAULT_MODEL)))
ROBOT_BOX_FROM_BUMPER_FACTOR = float(os.environ.get("ROBOT_BOX_FROM_BUMPER_FACTOR", "3.0"))
```

- [ ] **Step 2: Build schema after load**

After `self.model = YOLO(str(MODEL_PATH))` (`app.py:481`):

```python
from web.field.detect_schema import DetectionSchema
self.det_schema = DetectionSchema.from_names(self.model.names)
print(f"[detect] model={MODEL_PATH.name} classes={self.model.names} "
      f"robots={self.det_schema.robot_class_color} hub={self.det_schema.has_hub}")
```

Add `from web.field.detect_schema import expand_bumper_to_robot` to the module imports (~`app.py:60`).

- [ ] **Step 3: Smoke check**

Run: `.venv/bin/python -c "from web.field.detect_schema import DetectionSchema; from ultralytics import YOLO; m=YOLO('runs/train/hub_zone_yolo26s/weights/best.pt'); print(DetectionSchema.from_names(m.names))"`
Expected: schema with `robot_class_color={1: None}`, `hub_class=2`.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(detect): env-selectable detector + DetectionSchema at load"
```

---

## Task 4: Route ALL class-id sites through the schema (correctness-critical)

**Files:** Modify `web/app.py` at `:796-810`, `:1942`, `:1637`, `:2933-2952`, `:4865-4890`

Every hardcoded `class_id == N` must become a schema call, because the new model's ids mean different things. Do each, then run a syntax check.

- [ ] **Step 1: `_detect_view` (`:796-810`)** — replace the `class_id==1` loop + `to_field_detections` call:

```python
        robots, _balls, _hubs = self.det_schema.split(dets)
        robot_boxes, robot_colors = [], []
        for (box, color, _conf) in robots:
            cx, cy = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0
            if any(mx <= cx <= mx+mw and my <= cy <= my+mh for (mx,my,mw,mh) in view.masks):
                continue
            robot_boxes.append(np.array(box)); robot_colors.append(color)
        field_dets = to_field_detections(robot_boxes, cal, self.field_dims, colors=robot_colors)
```

- [ ] **Step 2: Ball detector (`:1942`) — CORRECTNESS-CRITICAL**

`update_ball_trajectories` does `ball_mask = detections.class_id == 0`. Under the bumper schema class 0 is a **red robot**. Replace with the schema's fuel class:

```python
        if self.det_schema.fuel_class is None:
            return   # detector has no fuel class
        ball_mask = detections.class_id == self.det_schema.fuel_class
```

(Read the surrounding lines; keep the rest of the trajectory logic.)

- [ ] **Step 3: Hub detector (`:1637`)**

`hub_mask = detections.class_id == 2` → drive from schema; a hubless schema yields none:

```python
        if self.det_schema.hub_class is None:
            return
        hub_mask = detections.class_id == self.det_schema.hub_class
```

- [ ] **Step 4: Early-OCR pre-pass (`:2933-2952`)**

`early_boxes = [... if detections.class_id[i]==1]` + `to_field_detections(..., color_fn=...)` → use `schema.split` and pass colors:

```python
                robots, _b, _h = self.det_schema.split(detections)
                early_boxes, early_colors = [], []
                for (box, color, _c) in robots:
                    early_boxes.append(np.array(box)); early_colors.append(color)
                # (apply the existing main_view.masks filter to early_boxes/early_colors in lockstep)
                ...
                early_field_dets = to_field_detections(
                    early_boxes, self.calib, self.field_dims, colors=early_colors)
```

Read `:2926-2955` and keep the mask-filter behavior, filtering `early_colors` in lockstep with `early_boxes`.

- [ ] **Step 5: `/api/detect-robots` path (`_detect_robots_in_frame`, `:4865`)**

It receives `model` and does `class_id==1` + `_detect_bumper_color_static`. Build a schema from the passed model and use class color:

```python
def _detect_robots_in_frame(frame, model, match_teams, blue_teams, red_teams) -> list:
    from web.field.detect_schema import DetectionSchema
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    schema = DetectionSchema.from_names(model.names)
    robots_split, _b, _h = schema.split(detections)
    robots = []
    for (box, color, conf) in robots_split:
        x1, y1, x2, y2 = map(int, box)
        alliance = color or "unknown"
        candidate_teams = {"blue": blue_teams, "red": red_teams}.get(alliance, match_teams)
        ocr_result = _ocr_bumper_number(frame, box, candidate_teams)
        ...  # keep the rest of the per-robot dict construction
```

Read `:4865-4900` and preserve the result-dict shape; just replace the class filter + color source.

- [ ] **Step 6: Syntax check + commit**

Run: `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`

```bash
git add web/app.py
git commit -m "feat(detect): route all class-id sites through DetectionSchema (fuel=class2 fix)"
```

---

## Task 5: SAM2 seed/re-prompt expansion + class color → fusion vote

**Files:** Modify `web/app.py` `:2816-2822` (prompt), `:2870-2883` (detect_fn), `:2905` (store), `:3024-3069` (post-SAM2)

SAM2 obj ids are `1..N` in prompt order, so seed colors map directly. Color flows into each post-SAM2 `FieldDetection.color`; fusion votes across views (no fusion change).

- [ ] **Step 1: Prompt frame — expand boxes + record seed colors**

Add a `view_prompt_colors` dict beside `view_prompt_boxes` (same init site). Replace `:2818-2822`:

```python
                                    if is_prompt:
                                        res = self.model(crop, conf=0.4, verbose=False)[0]
                                        pdets = sv.Detections.from_ultralytics(res)
                                        robots, _b, _h = self.det_schema.split(pdets)
                                        for (box, color, _c) in robots:
                                            rbox = (expand_bumper_to_robot(box, crop.shape,
                                                        ROBOT_BOX_FROM_BUMPER_FACTOR)
                                                    if color is not None else box)
                                            view_prompt_boxes[view.name].append(
                                                tuple(v * SAM2_SCALE for v in rbox))
                                            view_prompt_colors[view.name].append(color)
```

- [ ] **Step 2: Re-prompt `detect_fn` — expand boxes**

Replace `:2877-2882`:

```python
                                res = self.model(img, conf=0.4, verbose=False)[0]
                                rdets = sv.Detections.from_ultralytics(res)
                                robots, _b, _h = self.det_schema.split(rdets)
                                return [
                                    (expand_bumper_to_robot(box, img.shape, ROBOT_BOX_FROM_BUMPER_FACTOR)
                                     if color is not None else box)
                                    for (box, color, _c) in robots
                                ]
```

- [ ] **Step 3: Store the obj-id → seed-color map**

Where `self.sam2_raw[view.name] = view_tracks` (`:2905`):

```python
                        self.sam2_obj_color = getattr(self, "sam2_obj_color", {})
                        self.sam2_obj_color[view.name] = {
                            i + 1: c for i, c in enumerate(view_prompt_colors[view.name])
                        }
```

- [ ] **Step 4: Post-SAM2 coloring (`:3043`) — class color, no HSV**

Replace `color = self.detect_bumper_color(view_frame, np.array([fx1,fy1,fx2,fy2]))` with:

```python
                            color = self.sam2_obj_color.get(view.name, {}).get(obj_id) or "unknown"
```

Re-prompted obj ids not in the seed map get `"unknown"`; fusion's territory fallback + other views resolve them. Read `:3024-3069` to confirm `obj_id`/`view`/`view_frame` names.

- [ ] **Step 5: Confirm fusion votes the color**

Read `web/field/fusion.py` (`_merge`, ~`:54`). Confirm per-detection `color` feeds the existing weighted color vote (no code change expected). If fusion ignores `"unknown"`, that's correct. Note findings in the commit message; do **not** add a fusion code change unless a test shows it's needed.

- [ ] **Step 6: Syntax check + commit**

Run: `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`

```bash
git add web/app.py
git commit -m "feat(detect): SAM2 seeds from expanded boxes; class color per track -> fusion vote"
```

---

## Task 6: Delete the HSV color path

**Files:** Modify `web/field/bumper.py`, `web/app.py` (`:60`, `:1691`, `:3326`, `:4251`, `:4881`)

- [ ] **Step 1: Delete color logic in `bumper.py`**

Remove `classify_bumper_color`, `_color_masks`, `_largest_rectangular_band_ratio`, and the `BumperColor` dataclass. **Keep** `_clamp_box`, `bumper_band_box`, `bumper_crop` (used by HUD + VLM crop). Remove now-unused imports.

- [ ] **Step 2: Fix the `bumper.py` import in app.py (`:60`)**

`from web.field.bumper import bumper_band_box, bumper_crop, classify_bumper_color`
→ `from web.field.bumper import bumper_band_box, bumper_crop`

- [ ] **Step 3: Delete `detect_bumper_color` method (`:1691`)** and `_detect_bumper_color_static` (`:4251`). Verify no remaining callers:

Run: `grep -n "detect_bumper_color\|classify_bumper_color\|_detect_bumper_color_static" web/app.py`
Expected after edits: **no matches** (all call sites rewired in Tasks 4/5 and Step 4 below).

- [ ] **Step 4: Untracked-robot HUD color (`:3326`)**

Replace `alliance = self.detect_bumper_color(overhead_frame, detections.xyxy[i])` with the schema's class color for that detection:

```python
                            alliance = self.det_schema.robot_class_color.get(int(class_id)) or "unknown"
```

(Read `:3300-3335` for the exact `class_id`/`i` names.)

- [ ] **Step 5: Run unit suites + syntax check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`
Run: `.venv/bin/python -m pytest tests/test_detect_schema.py tests/test_detect_bridge.py -q`
Expected: PASS, and the grep in Step 3 returns nothing.

- [ ] **Step 6: Commit**

```bash
git add web/field/bumper.py web/app.py
git commit -m "refactor(detect): delete HSV color path; color is now the detection class"
```

---

## Task 7: Status classes, hub warning, default flip

**Files:** Modify `web/app.py` (`:3638`, `:1633`, `:118`)

- [ ] **Step 1: `/api/status` model_classes from schema (`:3638`)**

`/api/status` is process-global (no job param); `jobs` is a module dict (`:346`). Replace the hardcoded dict:

```python
        "model_classes": (
            lambda _j: {int(k): v for k, v in _j.det_schema.names.items()}
            if (_j is not None and getattr(_j, "det_schema", None) is not None)
            else {0: "fuel_ball", 1: "robot", 2: "hub_opening"}
        )(next(reversed(jobs.values()), None) if jobs else None),
```

- [ ] **Step 2: Hubless-schema warning in `update_hub_positions` (`:1633`)**

```python
        if not self.det_schema.has_hub and not event_hub_positions:
            if not getattr(self, "_warned_no_hub", False):
                print("[detect] WARNING: detector has no hub class and no static "
                      "event_hub_positions set — set hubs in /calibrate or scoring degrades.")
                self._warned_no_hub = True
```

- [ ] **Step 3: Flip default model (only after SP-A acceptance GO)**

In Task 3 Step 1, change `_DEFAULT_MODEL` to `runs/train/frc_bumpers/weights/best.pt`. Do this only once SP-A's `EVAL.md` says GO and Task 8 passes; otherwise leave legacy default and note it.

- [ ] **Step 4: Syntax check + commit**

```bash
.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"
git add web/app.py
git commit -m "feat(detect): schema-driven status classes + hubless warning (+default flip if GO)"
```

---

## Task 8: Integration verification on real footage

**Files:** none (verification). Needs SP-A weights + static hubs for the test event.

- [ ] **Step 1: Unit suites green**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py tests/test_detect_bridge.py -v` → PASS.

- [ ] **Step 2: Run a match with the bumper model**

`FRC_DETECT_MODEL=runs/train/frc_bumpers/weights/best.pt` on one `web/uploads/*.mp4` (event has static hubs via `/calibrate`). Confirm the startup log shows `classes={0:'red_bumper',...} hub=False`.

- [ ] **Step 3: Verify acceptance points**

1. Tracks seed (count comparable to legacy).
2. **SAM2 masks cover whole robots** (gallery crops look whole-robot) — proves box expansion.
3. **Fuel still detected and NOT confused with red bumpers** — proves the class-2 fix (Task 4 Step 2).
4. Alliance colors correct and **stable across views** (fusion vote).
5. Scoring runs with static hubs (no crash from missing hub class).

- [ ] **Step 4: Field-position drift check** — spot-compare a few robots' field positions vs the legacy model on the same clip; confirm no meaningful drift from the bumper-box bottom.

- [ ] **Step 5: Record results; flip default if GO** (Task 7 Step 3), else document what failed.

```bash
git add web/app.py
git commit -m "feat(detect): default to bumper model after acceptance (or: keep legacy, see notes)"
```

---

## Done when

- `detect_schema` + `detect_bridge` unit suites green.
- `grep -n "classify_bumper_color\|detect_bumper_color" web/` returns nothing in the live path (HSV deleted).
- A real match analyzed with the bumper model seeds tracks, makes whole-robot masks, keeps fuel distinct from red bumpers, assigns stable correct colors via fusion, and scores with static hubs.

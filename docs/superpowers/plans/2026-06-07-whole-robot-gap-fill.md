# Whole-Robot Gap-Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover SAM2 tracking of robots whose bumpers are occluded by running the already-loaded whole-robot (hub_zone) model as a gap-fill pass alongside the bumper model at both SAM2 seeding points.

**Architecture:** The bumper model stays the high-trust source of alliance color. The whole-robot model contributes only boxes that don't overlap a bumper box; survivors enter as `color="unknown"` detections, which the existing `FieldTracker` already routes by field half. New logic is concentrated in pure, unit-tested helpers; `app.py` changes are thin wiring at two hook points.

**Tech Stack:** Python, Ultralytics YOLO, SAM2, OpenCV, pytest. Tests run with `.venv/bin/python -m pytest` (base conda python has no pytest).

**Spec:** `docs/superpowers/specs/2026-06-07-whole-robot-gap-fill-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `web/field/detect_schema.py` | Pure detector-output helpers (no torch). Add box-overlap, merge, and gap-fill-selection functions. | Modify |
| `web/field/detect_bridge.py` | Field projection + size filters. Extract the image-fraction size check into a reusable helper. | Modify |
| `web/app.py` | Config constants; `InferenceJob._detect_whole_robots` wrapper; wire Hook A (prompt seed) and Hook B (re-prompt). | Modify |
| `tests/test_detect_schema.py` | Unit tests for the new pure helpers. | Modify |
| `tests/test_detect_bridge.py` | Unit test for the extracted size helper; guard `_too_large` unchanged. | Modify |

All new decision logic lives in pure functions so it is testable without loading YOLO/SAM2. `app.py` only projects coordinates and calls these helpers.

---

## Task 1: Pure helpers — overlap, merge, gap-fill selection

**Files:**
- Modify: `web/field/detect_schema.py`
- Test: `tests/test_detect_schema.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detect_schema.py`:

```python
from web.field.detect_schema import (
    box_ios,
    merge_robot_detections,
    select_gap_fills,
)


def test_box_ios_full_containment_is_one():
    # smaller box fully inside larger -> IoS == 1.0
    assert box_ios((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0


def test_box_ios_disjoint_is_zero():
    assert box_ios((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_box_ios_partial_overlap():
    # overlap 5x10=50; smaller area = 10x10=100 -> 0.5
    assert box_ios((0, 0, 10, 10), (5, 0, 25, 10)) == 0.5


def test_merge_no_bumpers_keeps_all_as_unknown():
    fills = [(0, 0, 10, 10), (50, 50, 60, 60)]
    merged = merge_robot_detections([], fills, ios_thresh=0.5)
    assert merged == [
        ((0, 0, 10, 10), "unknown"),
        ((50, 50, 60, 60), "unknown"),
    ]


def test_merge_passes_bumper_entries_through_unchanged():
    bumpers = [((0, 0, 10, 30), "red"), ((40, 0, 50, 30), "blue")]
    merged = merge_robot_detections(bumpers, [], ios_thresh=0.5)
    assert merged == bumpers


def test_merge_suppresses_fill_overlapping_a_bumper():
    bumpers = [((0, 0, 20, 20), "red")]
    fills = [(0, 0, 10, 10),        # IoS 1.0 with the bumper -> dropped
             (100, 100, 120, 120)]  # disjoint -> kept as unknown
    merged = merge_robot_detections(bumpers, fills, ios_thresh=0.5)
    assert merged == [
        ((0, 0, 20, 20), "red"),
        ((100, 100, 120, 120), "unknown"),
    ]


def test_select_gap_fills_caps_per_half():
    # blue half already full (3 bumpers) -> no blue fill; red has room for one
    bumper_halves = ["blue", "blue", "blue", "red"]
    unknown_halves = ["blue", "red", "red"]
    # index 0 (blue) rejected; index 1 (red) accepted -> red now 2; index 2 (red) accepted -> red 3
    assert select_gap_fills(bumper_halves, unknown_halves, max_per_half=3) == [1, 2]


def test_select_gap_fills_skips_none_territory():
    # an off-field (None) unknown is never selected
    assert select_gap_fills([], [None, "blue"], max_per_half=3) == [1]


def test_select_gap_fills_stops_at_cap_within_unknowns():
    # no bumpers; 4 blue unknowns, cap 3 -> first three kept
    assert select_gap_fills([], ["blue"] * 4, max_per_half=3) == [0, 1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py -k "box_ios or merge or select_gap_fills" -v`
Expected: FAIL with `ImportError: cannot import name 'box_ios'` (etc.).

- [ ] **Step 3: Implement the helpers**

Append to `web/field/detect_schema.py` (after `expand_bumper_to_robot`):

```python
def box_ios(a, b):
    """Intersection-over-smaller of two xyxy boxes (0.0–1.0).

    IoS, not IoU: a small box fully inside a large one scores 1.0, which is what
    we want when asking "is this whole-robot box already covered by a bumper box?"
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def merge_robot_detections(bumper_robots, robot_boxes, ios_thresh):
    """Combine bumper-derived robot boxes with whole-robot gap-fill boxes.

    bumper_robots: list[(box_xyxy, color)] with color in {"red","blue"}.
    robot_boxes:   list[box_xyxy], whole-robot (colorless) boxes.
    Returns list[(box_xyxy, color)]: every bumper entry unchanged, followed by
    each whole-robot box NOT already covered by a bumper box (max IoS over all
    bumper boxes <= ios_thresh), tagged color="unknown".
    """
    bumper_boxes = [box for box, _color in bumper_robots]
    merged = list(bumper_robots)
    for fb in robot_boxes:
        if all(box_ios(fb, bb) <= ios_thresh for bb in bumper_boxes):
            merged.append((fb, "unknown"))
    return merged


def select_gap_fills(bumper_halves, unknown_halves, max_per_half=3):
    """Pick which unknown gap-fill boxes to seed, capping each field half.

    bumper_halves:  list of "blue"/"red" for the bumper boxes in this frame.
    unknown_halves: list of "blue"/"red"/None for the candidate fills (None =
                    off-field, never selected), in their original order.
    Returns the indices into unknown_halves to keep. A half accepts a fill only
    while its running count (bumpers + already-kept fills) is below max_per_half.
    """
    counts = {"blue": 0, "red": 0}
    for h in bumper_halves:
        if h in counts:
            counts[h] += 1
    keep = []
    for idx, h in enumerate(unknown_halves):
        if h in counts and counts[h] < max_per_half:
            counts[h] += 1
            keep.append(idx)
    return keep
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py -v`
Expected: PASS (all new tests plus the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add web/field/detect_schema.py tests/test_detect_schema.py
git commit -m "feat(detect): box IoS + bumper/whole-robot merge + per-half gap-fill selection"
```

---

## Task 2: Reusable image-fraction size guard

**Files:**
- Modify: `web/field/detect_bridge.py:55-70` (`_too_large`)
- Test: `tests/test_detect_bridge.py`

**Why:** `_too_large` (the size filter) only runs inside `to_field_detections`, which the SAM2 seeding path never calls. The gap-fill wrapper needs the calibration-independent image-fraction check on its own. Extract that branch so the wrapper and `_too_large` share one implementation.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detect_bridge.py`:

```python
from web.field.detect_bridge import box_exceeds_frac


def test_box_exceeds_frac_true_when_wide():
    # box width 700 / frame width 1000 = 0.7 > 0.5
    assert box_exceeds_frac((0, 0, 700, 50), (480, 1000, 3), 0.5) is True


def test_box_exceeds_frac_false_when_small():
    assert box_exceeds_frac((0, 0, 100, 100), (480, 1000, 3), 0.5) is False


def test_box_exceeds_frac_none_frame_is_false():
    assert box_exceeds_frac((0, 0, 999, 999), None, 0.5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_detect_bridge.py -k box_exceeds_frac -v`
Expected: FAIL with `ImportError: cannot import name 'box_exceeds_frac'`.

- [ ] **Step 3: Extract the helper and call it from `_too_large`**

In `web/field/detect_bridge.py`, add above `_too_large`:

```python
def box_exceeds_frac(box, frame_shape, max_box_frac):
    """True if box spans more than max_box_frac of the frame in either dimension.
    Calibration-independent backstop against oversized detections."""
    if frame_shape is None:
        return False
    fh, fw = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = box
    return bool(fw and fh and (
        (x2 - x1) / fw > max_box_frac or (y2 - y1) / fh > max_box_frac
    ))
```

Then replace the `frame_shape` branch inside `_too_large` (currently lines 65-69):

```python
    if box_exceeds_frac(box, frame_shape, max_box_frac):
        return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_detect_bridge.py -v`
Expected: PASS (new tests plus the existing `to_field_detections` / `_too_large` tests still pass — the refactor is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add web/field/detect_bridge.py tests/test_detect_bridge.py
git commit -m "refactor(detect): extract box_exceeds_frac size guard for reuse"
```

---

## Task 3: Config constants + whole-robot detect wrapper

**Files:**
- Modify: `web/app.py` (constants near line 483; import near line 55; `InferenceJob.__init__`; new method on `InferenceJob`)

No new test here — this is a thin wrapper around YOLO (covered by the integration test in Task 6). Keep it minimal.

- [ ] **Step 1: Add config constants**

In `web/app.py`, immediately after `DET_CONF_THRESHOLD = 0.35` (line 483):

```python
# --- Whole-robot gap-fill (recover bumper-occluded robots) -------------------
ROBOT_FILL_ENABLED = os.environ.get("ROBOT_FILL_ENABLED", "1") == "1"
# Independent of the bumper path's hardcoded conf=0.4; defaults to DET_CONF_THRESHOLD.
ROBOT_FILL_CONF = float(os.environ.get("ROBOT_FILL_CONF", str(DET_CONF_THRESHOLD)))
# A whole-robot box with IoS above this against any bumper box is already covered.
ROBOT_FILL_IOS = float(os.environ.get("ROBOT_FILL_IOS", "0.5"))
ROBOT_FILL_MAX_BOX_FRAC = float(os.environ.get("ROBOT_FILL_MAX_BOX_FRAC", "0.5"))
```

- [ ] **Step 2: Extend imports**

Update the `detect_schema` import (line 55) and the `detect_bridge` import to include the new helpers:

```python
from web.field.detect_schema import (
    DetectionSchema,
    expand_bumper_to_robot,
    merge_robot_detections,
    select_gap_fills,
)
```

Add `box_exceeds_frac` to the existing `from web.field.detect_bridge import (...)` block (locate it with `grep -n "from web.field.detect_bridge import" web/app.py`).

- [ ] **Step 3: Resolve the whole-robot class id once in `InferenceJob.__init__`**

After `self.model` is assigned (find with `grep -n "self.model = YOLO" web/app.py`), add:

```python
        # hub_zone model: the class whose schema color is None is the whole-robot
        # class (used for gap-filling bumper-occluded robots). Fallback to 1.
        try:
            _fill_schema = DetectionSchema.from_names(self.model.names)
            self.robot_fill_class = next(
                (cid for cid, color in _fill_schema.robot_class_color.items()
                 if color is None),
                1,
            )
        except Exception:
            self.robot_fill_class = 1
```

- [ ] **Step 4: Add the wrapper method on `InferenceJob`**

Place near the other detection helpers (e.g. just above `_detect_view`; find with `grep -n "def _detect_view" web/app.py`):

```python
    def _detect_whole_robots(self, frame):
        """Whole-robot (chassis) boxes from the hub_zone model, in `frame` coords.

        Used to gap-fill robots whose bumpers are occluded. Returns [] when
        disabled or unavailable. Applies an image-fraction size guard because the
        SAM2 seeding path does NOT run the `_too_large` filter.
        """
        if not ROBOT_FILL_ENABLED or self.model is None:
            return []
        res = self.model(frame, conf=ROBOT_FILL_CONF, verbose=False)[0]
        boxes = []
        for i in range(len(res.boxes)):
            if int(res.boxes.cls[i]) != self.robot_fill_class:
                continue
            box = tuple(float(v) for v in res.boxes.xyxy[i].cpu().numpy())
            if box_exceeds_frac(box, frame.shape, ROBOT_FILL_MAX_BOX_FRAC):
                continue
            boxes.append(box)
        return boxes
```

- [ ] **Step 5: Sanity-check the module imports**

Run: `.venv/bin/python -c "import web.app"`
Expected: no ImportError / NameError (model weights may emit load logs; that's fine).

- [ ] **Step 6: Commit**

```bash
git add web/app.py
git commit -m "feat(app): whole-robot gap-fill config + _detect_whole_robots wrapper"
```

---

## Task 4: Wire Hook A — SAM2 prompt-frame seeding

**Files:**
- Modify: `web/app.py:3147-3182` (the prompt-frame export/seed block)

This is the bumper-model branch of the prompt-frame seeding. It must: detect bumpers, expand to robot boxes, gap-fill with non-overlapping whole-robot boxes, cap fills per field half by projected territory, then append. The fallback (`elif is_prompt:`, no bumper model) is left unchanged.

- [ ] **Step 1: Make the top-crop offset available for projection**

Currently (lines 3147-3150):

```python
                                    top_crop_frac = (getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0) if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                                    if top_crop_frac > 0:
                                        top_crop_px = int(crop.shape[0] * top_crop_frac)
                                        crop = crop[top_crop_px:, :]
```

Replace with (compute `top_crop_px` unconditionally so it's in scope below):

```python
                                    top_crop_frac = (getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0) if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                                    top_crop_px = int(crop.shape[0] * top_crop_frac) if top_crop_frac > 0 else 0
                                    if top_crop_px > 0:
                                        crop = crop[top_crop_px:, :]
```

- [ ] **Step 2: Replace the bumper-model seeding branch with the gap-fill version**

Replace the `if is_prompt and self.bumper_model is not None:` block (lines 3159-3175) with:

```python
                                    if is_prompt and self.bumper_model is not None:
                                        res = self.bumper_model(
                                            crop, conf=0.4, verbose=False
                                        )[0]
                                        robots, _balls, _hubs = self.det_schema.split(
                                            sv.Detections.from_ultralytics(res)
                                        )
                                        bumper_robots = [
                                            (
                                                expand_bumper_to_robot(
                                                    box, crop.shape,
                                                    ROBOT_BOX_FROM_BUMPER_FACTOR,
                                                ),
                                                color,
                                            )
                                            for box, color, _c in robots
                                        ]
                                        # Gap-fill: whole-robot boxes not covered by a bumper.
                                        fills = self._detect_whole_robots(crop)
                                        merged = merge_robot_detections(
                                            bumper_robots, fills, ROBOT_FILL_IOS
                                        )

                                        def _territory(box):
                                            uv = ((box[0] + box[2]) / 2.0,
                                                  box[3] + top_crop_px)
                                            xy = view.calibration.image_to_field(uv)
                                            if xy is None or not _on_field(xy, self.field_dims):
                                                return None
                                            return ("blue"
                                                    if xy[0] < self.field_dims[0] / 2.0
                                                    else "red")

                                        bumper_entries = [(b, c) for (b, c) in merged
                                                          if c != "unknown"]
                                        unknown_entries = [(b, c) for (b, c) in merged
                                                           if c == "unknown"]
                                        bumper_halves = [
                                            h for h in (_territory(b) for b, _ in bumper_entries)
                                            if h is not None
                                        ]
                                        unknown_halves = [_territory(b)
                                                          for b, _ in unknown_entries]
                                        keep = select_gap_fills(
                                            bumper_halves, unknown_halves,
                                            MAX_ROBOTS_PER_ALLIANCE,
                                        )

                                        seeded = bumper_entries + [unknown_entries[j]
                                                                   for j in keep]
                                        for box, color in seeded:
                                            view_prompt_boxes[view.name].append(
                                                tuple(v * SAM2_SCALE for v in box)
                                            )
                                            view_prompt_colors[view.name].append(color)
```

(Leave the `elif is_prompt:` fallback block immediately following it unchanged.)

- [ ] **Step 3: Sanity-check imports/syntax**

Run: `.venv/bin/python -c "import web.app"`
Expected: no SyntaxError / NameError.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(app): Hook A — gap-fill bumper-occluded robots at SAM2 prompt seeding"
```

---

## Task 5: Wire Hook B — periodic re-prompt detect_fn

**Files:**
- Modify: `web/app.py:3234-3265` (`make_detect_fn`)

The re-prompt path delegates per-alliance capping to SAM2's mask-overlap gate + the tracker's 3-slot cap (no territory math here, per the spec). It just adds non-overlapping whole-robot boxes as `(box, "unknown")` tuples.

- [ ] **Step 1: Augment the bumper branch of detect_fn**

Replace the `if self.bumper_model is not None:` block inside `detect_fn` (lines 3241-3258) with:

```python
                                if self.bumper_model is not None:
                                    res = self.bumper_model(
                                        img, conf=0.4, verbose=False
                                    )[0]
                                    robots, _balls, _hubs = self.det_schema.split(
                                        sv.Detections.from_ultralytics(res)
                                    )
                                    bumper_robots = [
                                        (
                                            expand_bumper_to_robot(
                                                box, img.shape,
                                                ROBOT_BOX_FROM_BUMPER_FACTOR,
                                            ),
                                            color,
                                        )
                                        for box, color, _confidence in robots
                                    ]
                                    fills = self._detect_whole_robots(img)
                                    return merge_robot_detections(
                                        bumper_robots, fills, ROBOT_FILL_IOS
                                    )
```

(Leave the no-bumper fallback `return [...]` block below it unchanged — it already is the whole-robot model.)

- [ ] **Step 2: Sanity-check imports/syntax**

Run: `.venv/bin/python -c "import web.app"`
Expected: no SyntaxError / NameError.

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(app): Hook B — gap-fill bumper-occluded robots at SAM2 re-prompt"
```

---

## Task 6: Integration test + full suite

**Files:**
- Test: `tests/test_robot_gap_fill.py` (create)

Validate the end-to-end merge+cap behavior at the pure-logic seam (the seam Hook A/B depend on) without loading YOLO/SAM2, and confirm the kill switch path. Heavy model wiring is exercised by the existing acceptance harness (see manual step).

- [ ] **Step 1: Write the integration test**

Create `tests/test_robot_gap_fill.py`:

```python
"""Hook A/B composition: merge + per-half cap, the seam app.py wires together."""
from web.field.detect_schema import merge_robot_detections, select_gap_fills


def _territory(box, field_length, image_to_field):
    uv = ((box[0] + box[2]) / 2.0, box[3])
    xy = image_to_field(uv)
    if xy is None:
        return None
    return "blue" if xy[0] < field_length / 2.0 else "red"


def test_occluded_robot_recovered_as_unknown_fill():
    # One bumper-visible robot (red) and one chassis-only robot (no overlap).
    bumper_robots = [((0, 0, 40, 120), "red")]
    fills = [(500, 0, 540, 120)]  # disjoint from the bumper box
    merged = merge_robot_detections(bumper_robots, fills, ios_thresh=0.5)
    assert ((500, 0, 540, 120), "unknown") in merged
    assert len(merged) == 2


def test_fill_overlapping_bumper_not_double_seeded():
    bumper_robots = [((0, 0, 40, 120), "red")]
    fills = [(0, 0, 40, 120)]  # identical box -> IoS 1.0 -> suppressed
    merged = merge_robot_detections(bumper_robots, fills, ios_thresh=0.5)
    assert merged == [((0, 0, 40, 120), "red")]


def test_full_half_blocks_further_fills():
    # Simulate three confirmed blue + one blue chassis fill via a fake projection.
    field_length = 16.0

    def image_to_field(uv):
        # left of center -> blue half
        return (4.0, 4.0)

    bumpers = [(0, 0, 10, 30), (12, 0, 22, 30), (24, 0, 34, 30)]
    fill = (40, 0, 50, 30)
    bumper_halves = [_territory(b, field_length, image_to_field) for b in bumpers]
    unknown_halves = [_territory(fill, field_length, image_to_field)]
    assert select_gap_fills(bumper_halves, unknown_halves, max_per_half=3) == []


def test_room_in_half_admits_fill():
    field_length = 16.0

    def image_to_field(uv):
        return (12.0, 4.0)  # right of center -> red half

    bumpers = [(0, 0, 10, 30)]  # one red
    fill = (40, 0, 50, 30)
    bumper_halves = [_territory(b, field_length, image_to_field) for b in bumpers]
    unknown_halves = [_territory(fill, field_length, image_to_field)]
    assert select_gap_fills(bumper_halves, unknown_halves, max_per_half=3) == [0]
```

- [ ] **Step 2: Run the new test**

Run: `.venv/bin/python -m pytest tests/test_robot_gap_fill.py -v`
Expected: PASS (4 tests).

- [ ] **Step 3: Run the full detection/tracking test surface**

Run: `.venv/bin/python -m pytest tests/test_detect_schema.py tests/test_detect_bridge.py tests/test_robot_gap_fill.py tests/test_tracker.py -v`
Expected: PASS (no regressions in tracker or bridge).

- [ ] **Step 4: Commit**

```bash
git add tests/test_robot_gap_fill.py
git commit -m "test(gap-fill): merge + per-half cap composition for SAM2 seeding hooks"
```

- [ ] **Step 5: Manual acceptance (human-in-the-loop)**

Process a clip with a known bumper-occluded robot (one obscured by balls or another robot) through the normal pipeline and confirm:
1. The occluded robot now gets a track (previously dropped).
2. It appears as `"unknown"` alliance until OCR/gallery resolves it, then earns its color/number.
3. Toggle `ROBOT_FILL_ENABLED=0` and confirm seeding reverts to bumper-only (regression-safe kill switch).

Use the acceptance footage referenced in the project memory (`acceptance-harness-download`) if a fresh clip isn't handy. This step is verification, not a code change — record the observed before/after.

---

## Done When

- All unit + integration tests pass via `.venv/bin/python -m pytest`.
- `ROBOT_FILL_ENABLED=0` reproduces today's bumper-only seeding exactly.
- A bumper-occluded robot that previously went untracked now seeds a SAM2 object and a tracker slot, entering as `"unknown"` and remaining eligible for OCR/gallery identity resolution.

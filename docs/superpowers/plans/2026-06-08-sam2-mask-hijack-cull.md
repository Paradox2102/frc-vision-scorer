# Detector-Absence Cull for Hijacked SAM2 Masks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cull a SAM2 object whose mask has hijacked onto a static field structure (the trench), by detecting that no fresh detector box backs the mask (detector-absence) while the mask is stationary in field space — freeing its object slot so the real moving robot re-acquires.

**Architecture:** A second object-retirement trigger inside `Sam2FieldTracker.track`'s existing re-prompt loop. At each re-prompt boundary, each active object's latest mask is checked for corroboration against the freshly-detected boxes that the loop already produces; uncorroborated + stationary objects are culled (their masks stop being recorded, and they drop out of the active-roster cap). Stationarity is measured in field meters via an injected `project_fn` so a panning camera cannot disguise a static trench as moving.

**Tech Stack:** Pure-NumPy helpers in `web/field/sam2_tracker.py` (torch-free, unit-testable); the SAM2 wrapper's chunked re-prompt loop; `pytest` with the existing `RepromptStubPredictor`-style fakes (no GPU). Run tests with `.venv/bin/python -m pytest` (base conda python has no pytest).

**Spec:** `docs/superpowers/specs/2026-06-08-sam2-mask-hijack-cull-design.md`

---

## File Structure

- **`web/field/sam2_tracker.py`** (modify) — add two pure helpers (`mask_corroborated`, `is_stationary`) next to the existing pure helpers (`_ios`, `match_new_detections`); add four module-level config constants alongside the existing `MASK_*` knobs; add `cull_*` + `project_fn` params to `track()`; integrate corroboration/stationarity/cull into the re-prompt boundary block and a `culled_set` skip into `_record_chunk`.
- **`web/field/identity_diagnostics.py`** (modify) — register the new `SAM2_OBJECT_CULLED = "sam2.object_culled"` event on `DiagnosticEvents`. **Required:** `IdentityDiagnosticRecorder.record()` validates every event name against `ALL_EVENTS` (built from `DiagnosticEvents`) and *silently drops* (into `_dropped_logs`) any unregistered name. Without this registration the cull diagnostic is never recorded — in tests or production — and every cull assertion plus the Task 6 acceptance grep would find nothing.
- **`web/app.py`** (modify) — build a `project_fn` bound to each view's calibration and pass it into `sam2_tracker.track`. No new constants here (knobs live in `sam2_tracker.py`, matching that module's existing env-knob convention — a deliberate DRY deviation from the spec's "constants in app.py" wording).
- **`tests/field/test_sam2_tracker.py`** (modify) — unit tests for the two helpers; a `CullStubPredictor` fake and integration tests for the four cull scenarios + the capped-deadlock regression + the kill-switch.

Knob convention note: `sam2_tracker.py` already reads tunables from env at module scope (`MASK_MAX_ASPECT = float(os.environ.get(...))`, lines 26-28). The four `CULL_*` knobs follow that exact pattern, and `track()` exposes them as parameters defaulting to the module constants (mirroring how `new_obj_ios_thresh` is a param with a default). This keeps tests able to pass explicit values while production reads env.

---

## Task 1: Pure helper — `mask_corroborated`

Decides whether any fresh detector box backs an existing mask (the detector-absence signal, inverted). This is the per-object complement of the existing `match_new_detections`.

**Files:**
- Modify: `web/field/sam2_tracker.py` (add after `match_new_detections`, ~line 152)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing tests**

Add near the other pure-helper tests (top of file, after `test_match_new_detections_flags_unmatched`):

```python
def test_mask_corroborated_true_when_box_overlaps():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True            # the tracked mask
    overlapping = (6, 2, 10, 6)       # identical box -> IoS 1.0
    assert mask_corroborated(mask, [overlapping], (60, 20), ios_thresh=0.3) is True


def test_mask_corroborated_false_when_no_box_overlaps():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    far_box = (40, 2, 44, 6)          # nowhere near the mask
    assert mask_corroborated(mask, [far_box], (60, 20), ios_thresh=0.3) is False


def test_mask_corroborated_false_when_no_boxes():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    assert mask_corroborated(mask, [], (60, 20), ios_thresh=0.3) is False


def test_mask_corroborated_ignores_color_conflicting_box():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    overlapping = (6, 2, 10, 6)
    # A blue detection must NOT corroborate a red mask.
    assert mask_corroborated(
        mask, [overlapping], (60, 20), ios_thresh=0.3,
        mask_color="red", new_colors=["blue"],
    ) is False
    # Same-color (or colorless) detection still corroborates.
    assert mask_corroborated(
        mask, [overlapping], (60, 20), ios_thresh=0.3,
        mask_color="red", new_colors=["red"],
    ) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k mask_corroborated -v`
Expected: FAIL with `ImportError: cannot import name 'mask_corroborated'`.

- [ ] **Step 3: Implement the helper**

In `web/field/sam2_tracker.py`, immediately after `match_new_detections` (~line 152):

```python
def mask_corroborated(
    mask: np.ndarray,
    new_boxes: list[tuple[int, int, int, int]],
    wh: tuple[int, int],
    ios_thresh: float = 0.3,
    mask_color: Any | None = None,
    new_colors: list[Any] | None = None,
) -> bool:
    """Return True if any fresh detector box backs ``mask`` (IoS >= ios_thresh).

    The per-object complement of :func:`match_new_detections`: that function asks
    "which new boxes are NOT already tracked"; this asks "is this tracked mask
    still supported by a fresh detection". Used by the hijack cull as the
    detector-absence signal — a mask glued to a static structure (no bumper, no
    whole-robot detection) returns False here.

    Color-conflicting detections are ignored (a blue box cannot corroborate a red
    mask), matching the color gate in :func:`match_new_detections`.
    """
    if not new_boxes:
        return False
    w, h = wh
    for idx, (x1, y1, x2, y2) in enumerate(new_boxes):
        if mask_color in {"red", "blue"} and new_colors and idx < len(new_colors):
            n_col = new_colors[idx]
            if n_col in {"red", "blue"} and n_col != mask_color:
                continue
        box_mask = np.zeros((h, w), bool)
        box_mask[int(y1):int(y2), int(x1):int(x2)] = True
        if _ios(mask, box_mask) >= ios_thresh:
            return True
    return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k mask_corroborated -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): mask_corroborated helper (detector-absence signal)"
```

---

## Task 2: Pure helper — `is_stationary`

Field-space stationarity over a short window of positions; the safety gate that protects parked/moving robots from the cull.

**Files:**
- Modify: `web/field/sam2_tracker.py` (add after `mask_corroborated`)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_is_stationary_true_for_fixed_positions():
    from web.field.sam2_tracker import is_stationary
    window = [(10.0, 5.0), (10.1, 5.0), (9.95, 5.05)]
    assert is_stationary(window, max_span_m=0.5) is True


def test_is_stationary_false_for_moving_positions():
    from web.field.sam2_tracker import is_stationary
    window = [(10.0, 5.0), (11.0, 5.0), (12.0, 5.0)]  # 2 m span
    assert is_stationary(window, max_span_m=0.5) is False


def test_is_stationary_false_when_insufficient_known_positions():
    from web.field.sam2_tracker import is_stationary
    # Fewer than 2 known positions -> cannot confirm stationary -> conservative False
    assert is_stationary([(10.0, 5.0)], max_span_m=0.5) is False
    assert is_stationary([None, (10.0, 5.0)], max_span_m=0.5) is False
    assert is_stationary([None, None], max_span_m=0.5) is False


def test_is_stationary_ignores_none_entries():
    from web.field.sam2_tracker import is_stationary
    window = [(10.0, 5.0), None, (10.2, 5.1)]
    assert is_stationary(window, max_span_m=0.5) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k is_stationary -v`
Expected: FAIL with `ImportError: cannot import name 'is_stationary'`.

- [ ] **Step 3: Implement the helper**

```python
def is_stationary(positions, max_span_m: float) -> bool:
    """Return True if all known field positions lie within ``max_span_m``.

    ``positions`` is a list of (x, y) field-meter points, possibly containing
    ``None`` (unprojectable box). ``None`` entries are ignored. With fewer than
    two known positions we cannot confirm stationarity, so we return False
    (conservative — refuse to cull when motion is unmeasurable).
    """
    pts = [p for p in positions if p is not None]
    if len(pts) < 2:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    return span < max_span_m
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k is_stationary -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): is_stationary field-space window helper"
```

---

## Task 3: Config knobs + `track()` params + run-detector-when-capped decouple

The load-bearing behavioral change: today `detect_fn` only runs when `not capped` (`sam2_tracker.py:579`). The phantom hurts most when it fills the last active slot → capped → detector skipped → no corroboration data → the cull would deadlock. This task makes the detector run at every boundary when the cull is enabled (guarded so `CULL_HIJACK_ENABLED=0` fully reverts behavior and compute), **without yet adding any cull logic**.

**Files:**
- Modify: `web/field/sam2_tracker.py` (constants ~line 28; `track()` signature ~line 275-291; re-prompt loop ~line 555-698)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing test**

Add a fake that lets us count `detect_fn` calls while the roster is capped. Put it with the other stubs:

```python
class CullStubPredictor:
    """Reprompt stub with controllable per-object motion, for cull tests.

    Object ``oid`` lives at x in [10*oid, 10*oid+4). An oid listed in
    ``moving`` drifts +3 px in x per frame (so its projected position changes
    across re-prompt boundaries); all others are static. Honours
    start_frame_idx / max_frame_num_to_track like the real predictor.
    """

    H = 20
    W = 200
    NUM_FRAMES = 12

    def __init__(self, moving=()):
        self.moving = set(moving)
        self.add_calls: list[tuple[int, int]] = []

    def init_state(self, video_path: str, **kwargs) -> dict:
        return {"objs": {}}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        state["objs"][obj_id] = frame_idx
        self.add_calls.append((frame_idx, obj_id))

    def _rect_for(self, oid: int, f: int = 0) -> tuple[int, int, int, int]:
        x0 = oid * 10 + (3 * f if oid in self.moving else 0)
        return (x0, 2, x0 + 4, 6)

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None):
        start = 0 if start_frame_idx is None else start_frame_idx
        end = (self.NUM_FRAMES - 1 if max_frame_num_to_track is None
               else min(start + max_frame_num_to_track, self.NUM_FRAMES - 1))
        for f in range(start, end + 1):
            active = sorted(oid for oid, af in state["objs"].items() if af <= f)
            rects = [self._rect_for(oid, f) for oid in active]
            yield f, active, _make_mask_array(self.H, self.W, rects)

    def reset_state(self, state) -> None:
        pass


def test_detector_runs_at_capped_boundary_when_cull_enabled():
    """With the cull enabled, detect_fn is called even when the roster is capped
    (so corroboration data exists for the phantom)."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []  # no new robots; only here to be counted

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=2,          # capped from the start
        cull_enabled=True,
    )
    assert calls, "detect_fn must run at boundaries when cull is enabled even while capped"


def test_detector_skipped_at_capped_boundary_when_cull_disabled():
    """Kill switch: with the cull disabled, capped boundaries skip detect_fn
    exactly as before."""
    from web.field.sam2_tracker import Sam2FieldTracker

    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []

    tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=2,
        cull_enabled=False,
    )
    assert calls == [], "detect_fn must NOT run at capped boundaries when cull disabled"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "capped_boundary_when_cull" -v`
Expected: FAIL — `track()` got an unexpected keyword argument `cull_enabled`.

- [ ] **Step 3a: Add module constants**

In `web/field/sam2_tracker.py`, after the `MASK_*` knobs (~line 28):

```python
# Hijack-cull knobs (detector-absence + stationary). A SAM2 object whose mask has
# drifted onto a static field structure stays "visible" forever, so the existing
# absence-based retirement never fires. These drive a second retirement trigger:
# cull an object that goes CULL_UNCORROBORATED_REPROMPTS consecutive re-prompts
# with no fresh detector box backing its mask (IoS >= CULL_CORROB_IOS) while its
# field position stays within CULL_STATIONARY_M. See
# docs/superpowers/specs/2026-06-08-sam2-mask-hijack-cull-design.md.
CULL_HIJACK_ENABLED = os.environ.get("CULL_HIJACK_ENABLED", "1") not in ("0", "false", "False")
CULL_UNCORROBORATED_REPROMPTS = int(os.environ.get("CULL_UNCORROBORATED_REPROMPTS", "3"))
CULL_CORROB_IOS = float(os.environ.get("CULL_CORROB_IOS", "0.3"))
CULL_STATIONARY_M = float(os.environ.get("CULL_STATIONARY_M", "0.5"))
```

- [ ] **Step 3b: Add `track()` parameters**

In the `track()` signature (after `new_obj_ios_thresh`, ~line 288), add:

```python
        cull_enabled: bool = CULL_HIJACK_ENABLED,
        cull_uncorroborated_reprompts: int = CULL_UNCORROBORATED_REPROMPTS,
        cull_corrob_ios: float = CULL_CORROB_IOS,
        cull_stationary_m: float = CULL_STATIONARY_M,
        project_fn=None,
```

Add to the docstring Args a short line for each (project_fn: "Callable mapping a mask box (tracker pixel coords) to field meters (x, y), or None when unprojectable; used to measure stationarity in field space so camera pan can't disguise a static structure as moving. When None, falls back to pixel bottom-center (ego-motion-confounded; under-culls).").

- [ ] **Step 3c: Decouple the detector call from `capped`**

Replace the re-prompt boundary block. The current block (`sam2_tracker.py:563-696`, from `# Re-prompt at the chunk boundary:` down to just before `cursor = last_idx + 1`) is replaced by the structure below. This step implements only the decouple + shared `existing` build; the cull block is a no-op placeholder filled in Task 4.

```python
                    # Re-prompt at the chunk boundary. Recompute the active set
                    # (excluding anything culled), retiring absent objects.
                    active_objects = []
                    for oid in range(1, next_obj_id):
                        if oid in culled_set:
                            continue
                        lv = last_visible_frame.get(oid)
                        if lv is not None and (last_idx - lv) <= retirement_frames:
                            active_objects.append(oid)
                        else:
                            if oid not in retired_set and lv is not None:
                                retired_set.add(oid)
                                if diagnostics:
                                    diagnostics.record(
                                        "sam2.object_retired",
                                        details={"object_id": oid, "frame_idx": last_idx},
                                    )

                    capped = (len(active_objects) >= max_active_objects
                              or (next_obj_id - 1) >= max_lifetime_objects)

                    # Latest mask + color per active object — needed for both
                    # corroboration (cull) and new-object matching (seed).
                    existing = {}
                    existing_colors = {}
                    for oid in active_objects:
                        lv = last_visible_frame.get(oid)
                        if lv is not None and oid in results.get(lv, {}):
                            existing[oid] = results[lv][oid]["mask"]
                            if object_metadata is not None:
                                existing_colors[oid] = object_metadata.get(oid)

                    # Run the detector when we need to seed (not capped) OR need
                    # corroboration for the cull (cull_enabled). Guarding the extra
                    # capped-boundary call behind cull_enabled keeps the kill switch
                    # a full behavioral/compute revert.
                    run_detector = detect_fn is not None and (not capped or cull_enabled)
                    new_boxes, new_metadata = [], []
                    if run_detector:
                        frame_path = (
                            frame_paths[last_idx] if last_idx < len(frame_paths) else None
                        )
                        try:
                            detected = detect_fn(last_idx, frame_path) or []
                        except Exception:
                            detected = []
                        new_boxes, new_metadata = _normalize_detections(detected)
                        if diagnostics:
                            for box in new_boxes:
                                diagnostics.record(
                                    "sam2.reprompt_candidate",
                                    details={"frame_idx": last_idx, "box": list(box)},
                                )

                    # ---- Hijack cull (filled in Task 4) ----
                    # if cull_enabled and existing: ...

                    # ---- New-object seeding (unchanged logic, gated on capped) ----
                    if not capped and new_boxes:
                        if existing:
                            h, w = next(iter(existing.values())).shape
                        else:
                            w = max(int(b[2]) for b in new_boxes) + 1
                            h = max(int(b[3]) for b in new_boxes) + 1
                        int_boxes = [
                            (int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in new_boxes
                        ]
                        matched_indices = match_new_detections(
                            existing, int_boxes, (w, h),
                            ios_thresh=new_obj_ios_thresh,
                            existing_colors=existing_colors,
                            new_colors=new_metadata,
                        )
                        if diagnostics:
                            for idx, box_val in enumerate(new_boxes):
                                if idx not in matched_indices:
                                    diagnostics.record(
                                        "sam2.reprompt_rejected",
                                        reason="below_margin",
                                        details={"frame_idx": last_idx, "box": list(box_val)},
                                    )
                        for j in matched_indices:
                            # Re-check caps inside the loop (excluding culled).
                            active_objects = [
                                oid for oid in range(1, next_obj_id)
                                if oid not in culled_set
                                and last_visible_frame.get(oid) is not None
                                and (last_idx - last_visible_frame[oid]) <= retirement_frames
                            ]
                            cap_reached = (len(active_objects) >= max_active_objects
                                           or (next_obj_id - 1) >= max_lifetime_objects)
                            if cap_reached:
                                reason = ("roster_cap" if len(active_objects) >= max_active_objects
                                          else "lifetime_cap")
                                if diagnostics:
                                    diagnostics.record(
                                        "sam2.reprompt_rejected",
                                        reason=reason,
                                        details={"frame_idx": last_idx, "box": list(new_boxes[j])},
                                    )
                                continue
                            box_mask = np.zeros((h, w), bool)
                            bx1, by1, bx2, by2 = int_boxes[j]
                            box_mask[by1:by2, bx1:bx2] = True
                            best_ios = 0.0
                            for oid, existing_mask in existing.items():
                                if existing_colors and new_metadata:
                                    e_col = existing_colors.get(oid)
                                    n_col = new_metadata[j]
                                    if (e_col in {"red", "blue"} and n_col in {"red", "blue"}
                                            and e_col != n_col):
                                        continue
                                score = _ios(existing_mask, box_mask)
                                if score > best_ios:
                                    best_ios = score
                            predictor.add_new_points_or_box(
                                state,
                                frame_idx=last_idx,
                                obj_id=next_obj_id,
                                box=np.asarray(new_boxes[j], dtype=np.float32),
                            )
                            first_prompted_frame[next_obj_id] = last_idx
                            birth_detector_overlap[next_obj_id] = best_ios
                            if diagnostics:
                                diagnostics.record(
                                    "sam2.object_created",
                                    details={"object_id": next_obj_id, "frame_idx": last_idx,
                                             "box": list(new_boxes[j])},
                                )
                            if object_metadata is not None:
                                object_metadata[next_obj_id] = new_metadata[j]
                            next_obj_id += 1
                    elif capped and run_detector and new_boxes:
                        # Capped but we ran the detector for corroboration; preserve
                        # the rejection diagnostics the old `else` branch emitted.
                        if diagnostics:
                            reason = ("roster_cap" if len(active_objects) >= max_active_objects
                                      else "lifetime_cap")
                            for box_val in new_boxes:
                                diagnostics.record(
                                    "sam2.reprompt_rejected",
                                    reason=reason,
                                    details={"frame_idx": last_idx, "box": list(box_val)},
                                )
```

- [ ] **Step 3d: Declare `culled_set` before `_record_chunk`**

`_record_chunk` and the loop both reference `culled_set`. Add it next to `retired_set` (~line 355):

```python
        retired_set: set[int] = set()
        culled_set: set[int] = set()
        uncorroborated_reprompts: dict[int, int] = {}
        field_pos_window: dict[int, list] = {}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "capped_boundary_when_cull" -v`
Expected: 2 passed.

Then run the **whole** existing SAM2 test file to prove the loop rewrite preserved behavior:

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -v`
Expected: all prior tests still pass (re-prompt seeding, caps, bootstrap, diagnostics, structure).

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): run reprompt detector for corroboration even when capped (cull groundwork)"
```

---

## Task 4: Corroboration tracking, stationarity, and the cull decision

Fills the placeholder from Task 3. Wires the two pure helpers into per-object state, decides culls, stops recording culled masks, and drops culled objects from the active roster so re-acquisition can proceed.

**Files:**
- Modify: `web/field/identity_diagnostics.py` (register `SAM2_OBJECT_CULLED`)
- Modify: `web/field/sam2_tracker.py` (the `_record_chunk` skip ~line 491-503; the cull placeholder in the boundary block)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing integration tests**

```python
def _ident_diag():
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    return IdentityDiagnosticRecorder()


def _culled_ids(diagnostics):
    # Recorder stores events in the private `_events` list; each payload's event
    # name is under the key "event" (not "type"), and object_id stays inside
    # "details" (it is not one of record()'s standard hoisted keys).
    return {e["details"]["object_id"]
            for e in diagnostics._events
            if e.get("event") == "sam2.object_culled"}


def test_cull_removes_uncorroborated_stationary_object():
    """A static mask that no detection backs for K re-prompts is culled; a
    corroborated neighbour survives."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center

    stub = CullStubPredictor()                # objs 1 and 2 both static
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Always corroborate obj 1; never obj 2 (the phantom).
        return [(stub._rect_for(1), "blue")]

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=6,
        cull_enabled=True,
        cull_uncorroborated_reprompts=3,
        cull_corrob_ios=0.3,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box),
        diagnostics=diag,
    )

    assert 2 in _culled_ids(diag)
    # Obj 2's masks stop being recorded after the cull; obj 1 persists to the end.
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]
    # Diagnostics recorded the cull with the expected reason (reason lives inside
    # `details`, deliberately — see the implementation note in Step 3c).
    culled = [e for e in diag._events if e.get("event") == "sam2.object_culled"]
    assert any(e["details"]["object_id"] == 2
               and e["details"]["reason"] == "detector_absent_stationary"
               for e in culled)


def test_cull_spares_moving_object_even_if_uncorroborated():
    """A moving mask is never culled, even with no corroboration (mover = robot)."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center

    stub = CullStubPredictor(moving={2})      # obj 2 drifts every frame
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return []                              # corroborate nothing

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1, 0), stub._rect_for(2, 0)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        cull_enabled=True,
        cull_uncorroborated_reprompts=3,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box),
        diagnostics=diag,
    )

    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_cull_spares_corroborated_stationary_object():
    """A static mask that the detector keeps backing (a real parked robot) is not
    culled."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center

    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [stub._rect_for(2)]             # always backs obj 2

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=detect_fn,
        max_active_objects=6,
        cull_enabled=True,
        cull_uncorroborated_reprompts=3,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box),
        diagnostics=diag,
    )

    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_cull_frees_capped_slot_and_real_robot_reseeds():
    """The deadlock case: phantom fills the last slot; culling it lets the real
    moving robot (obj 3) seed into the freed slot."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center

    stub = CullStubPredictor(moving={3})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # obj 1 corroborated; obj 2 (phantom) never; obj 3 is a NEW robot box.
        return [(stub._rect_for(1), "blue"), (stub._rect_for(3, local_idx), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub",
        prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],  # 2 active = capped
        reprompt_every=2,
        detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2,                  # capped until the phantom is culled
        cull_enabled=True,
        cull_uncorroborated_reprompts=3,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box),
        diagnostics=diag,
    )

    assert 2 in _culled_ids(diag)              # phantom culled
    assert any(oid == 3 for (_f, oid) in stub.add_calls)  # real robot seeded after
    assert 3 in result[stub.NUM_FRAMES - 1]    # and tracked to the end
    assert 2 not in result[stub.NUM_FRAMES - 1]
```

Note: these tests read `diagnostics._events` (the recorder has no public accessor) and filter on the `"event"` key. This is confirmed against `web/field/identity_diagnostics.py:95,139`. The cull event must be registered first (Step 3a) or `record()` silently drops it and every assertion here fails.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "cull_removes or cull_spares or cull_frees" -v`
Expected: FAIL — no `sam2.object_culled` events (cull not implemented); obj 2 still present at the last frame; obj 3 never seeded in the capped test.

- [ ] **Step 3a: Register the `sam2.object_culled` diagnostic event**

In `web/field/identity_diagnostics.py`, add to the `DiagnosticEvents` class in the `# SAM2` group (after `SAM2_CHUNK_COMPLETE`, ~line 22):

```python
    SAM2_OBJECT_CULLED = "sam2.object_culled"
```

`ALL_EVENTS` is rebuilt from `DiagnosticEvents` at import (line 85), so this is all that's needed for `record("sam2.object_culled", ...)` to pass validation. Do **not** add a new `DiagnosticReasons` member — the cull's reason string is carried *inside* `details` (see Step 3c), so it bypasses the `ALL_REASONS` validator by design.

- [ ] **Step 3b: Skip culled objects in `_record_chunk`**

In `_record_chunk`, alongside the implausible-mask skip (`sam2_tracker.py:493`), add an early skip so a culled object's masks stop being recorded. Right after `oid_int = int(obj_id)` (~line 476):

```python
                    if oid_int in culled_set:
                        continue  # culled hijack mask — stop emitting it
```

- [ ] **Step 3c: Implement the cull block**

Replace the `# ---- Hijack cull (filled in Task 4) ----` placeholder with the following. Note `reason` is placed **inside** `details` (not as a `record()` kwarg) on purpose: a top-level `reason=` is validated against `ALL_REASONS` and would be dropped, whereas a value inside `details` is stored verbatim.

```python
                    if cull_enabled and existing:
                        h, w = next(iter(existing.values())).shape
                        for oid, mask in list(existing.items()):
                            corroborated = mask_corroborated(
                                mask, new_boxes, (w, h),
                                ios_thresh=cull_corrob_ios,
                                mask_color=(existing_colors.get(oid)
                                            if existing_colors else None),
                                new_colors=new_metadata,
                            )
                            if corroborated:
                                uncorroborated_reprompts[oid] = 0
                            else:
                                uncorroborated_reprompts[oid] = (
                                    uncorroborated_reprompts.get(oid, 0) + 1
                                )

                            box = results[last_visible_frame[oid]][oid]["box"]
                            pos = project_fn(box) if project_fn is not None else bottom_center(box)
                            window = field_pos_window.setdefault(oid, [])
                            window.append(pos)
                            if len(window) > cull_uncorroborated_reprompts:
                                del window[0]

                            if (uncorroborated_reprompts[oid] >= cull_uncorroborated_reprompts
                                    and is_stationary(window, cull_stationary_m)):
                                culled_set.add(oid)
                                if diagnostics:
                                    diagnostics.record(
                                        "sam2.object_culled",
                                        details={
                                            "object_id": oid,
                                            "frame_idx": last_idx,
                                            "reason": "detector_absent_stationary",
                                            "uncorroborated_reprompts":
                                                uncorroborated_reprompts[oid],
                                        },
                                    )

                        if culled_set:
                            active_objects = [o for o in active_objects if o not in culled_set]
                            for o in list(culled_set):
                                existing.pop(o, None)
                                existing_colors.pop(o, None)
                            capped = (len(active_objects) >= max_active_objects
                                      or (next_obj_id - 1) >= max_lifetime_objects)
```

This recomputes `capped` after culling so the **same boundary's** seeding block can use the freed slot. (Re-acquisition may also happen at the next boundary; both are fine.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "cull_removes or cull_spares or cull_frees" -v`
Expected: 4 passed.

Then the full file:

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py web/field/identity_diagnostics.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): cull hijacked masks (detector-absent + stationary)"
```

---

## Task 5: Wire `project_fn` into the production call in `app.py`

Give the cull a real field-space stationarity measurement bound to each view's calibration. Without this, `project_fn` defaults to `None` and the cull falls back to ego-motion-confounded pixel space.

**Files:**
- Modify: `web/app.py` (the `sam2_tracker.track(...)` call, ~line 4225-4238; mirror the result-projection math at `app.py:4451,4471-4472`)

- [ ] **Step 1: Read the downstream projection to mirror it**

Read `web/app.py:4448-4476`. The consumer converts a SAM2 result box `(x1,y1,x2,y2)` in 0.5× top-cropped coords to field via:
`fx = x/SAM2_SCALE`, `fy = y/SAM2_SCALE + sam2_y_offset`, `uv = ((fx1+fx2)/2, fy2)`, `xy = cal.image_to_field(uv)`, where `sam2_y_offset = int(view_frame.shape[0] * top_crop_frac)`.

`project_fn` must apply the same transform so stationarity is measured in the same field frame the rest of the pipeline uses. (Exact byte-match is not required — stationarity only needs a *stable, calibrated* mapping so a static structure reads constant and a mover reads changing; `image_to_field` is the part that removes camera pan.)

**Heads-up — two coexisting top-crop offset formulas.** The result consumer (app.py:4440) uses `sam2_y_offset = int(view_frame.shape[0] * top_crop_frac)`, while the `detect_fn` closure (app.py:4164-4170) uses `y_offset = int(full_h * top_crop_frac / (1 - top_crop_frac))`. They are not the same. Mirror the **consumer's** form (4440), since that is how the phantom's field position is actually computed downstream — matching it keeps `project_fn`'s motion measurement consistent with the field map the cull is trying to clean.

- [ ] **Step 2: Build and pass `project_fn` at the track call**

Immediately before the `sam2_tracker.track(...)` call (~line 4225), inside the `for view in self.views` loop, construct the projector. `SAM2_SCALE = 0.5` is already defined in this scope (line 3931); compute the per-view top-crop offset from the same source the consumer uses (the view's full-resolution frame height — use the value already available in this loop; if only the SAM2-scaled height is at hand, divide by `SAM2_SCALE`):

```python
                        _sam2_scale = 0.5
                        _top_crop = (
                            (getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0)
                            if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                        )
                        # Full-res view height for the top-crop y-offset, matching the
                        # result consumer (app.py:4440). Use the view's known full
                        # height in this scope.
                        _full_h = <view full-resolution frame height in this scope>
                        _y_off = int(_full_h * _top_crop) if _top_crop > 0 else 0
                        _cal = view.calibration

                        def _make_project_fn(cal, y_off, scale):
                            def _project(box):
                                x1, y1, x2, y2 = box
                                fx1, fx2 = x1 / scale, x2 / scale
                                fy2 = y2 / scale + y_off
                                return cal.image_to_field(((fx1 + fx2) / 2.0, fy2))
                            return _project

                        _project_fn = _make_project_fn(_cal, _y_off, _sam2_scale)
```

Then add `project_fn=_project_fn,` to the `sam2_tracker.track(...)` keyword arguments (alongside `diagnostics=self.identity_diagnostics`).

Implementation note: bind `cal`/`y_off`/`scale` via the `_make_project_fn` factory (not a bare closure over the loop variable) so the per-view binding is correct even though `track` runs synchronously here. `image_to_field` may return `None` (unprojectable) — that's expected and handled by `is_stationary`.

Resolving `<view full-resolution frame height in this scope>`: at the track call (~4225) `view_frame` from the consumer block (4440) is **not** in scope. Use the same source the `detect_fn` closure already uses for full height — `int(round(img.shape[0] / SAM2_SCALE))` where `img` is an exported SAM2 frame — i.e. read one exported frame from `tmp_dir_obj.name` (the SAM2 frames dir passed as `frames_dir`) and divide its height by `_sam2_scale`. If the export step already stored a full-resolution height on `view`, prefer reusing that. Compute `_full_h` once per view before building the projector.

- [ ] **Step 3: Verify the module imports and a smoke run**

Run: `.venv/bin/python -c "import ast,sys; ast.parse(open('web/app.py').read()); print('app.py parses')"`
Expected: `app.py parses`.

Run: `.venv/bin/python -m pytest tests/ -k "sam2 or fusion or detect_bridge or tracker" -q`
Expected: pass (no regression in the surrounding field pipeline).

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(app): pass field-space project_fn into SAM2 cull"
```

---

## Task 6: Full suite + acceptance verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass. If unrelated pre-existing failures appear, note them but do not fix in this plan.

- [ ] **Step 2: Acceptance run on the trench-hijack match**

The hijack was observed on `2026hop_qm72` (see memory `acceptance-harness-download` for footage + the app's Python processing API; CLI `yt-dlp` 403s — use the app's API + deno + H.264). Process that match with the cull enabled and confirm via diagnostics that the trench object is culled and the previously-lost robot re-acquires.

Run (adapt to the project's acceptance entrypoint):
`CULL_HIJACK_ENABLED=1 .venv/bin/python <acceptance/replay entrypoint for 2026hop_qm72>`

Verify in the identity diagnostics output:
- a `sam2.object_culled` event with `reason="detector_absent_stationary"` for the trench object's id, and
- the phantom no longer appears on the field map for the rest of the match, and
- the real robot that was hijacked is tracked after the cull (a track present at its moving field positions).

Compare against a baseline run with `CULL_HIJACK_ENABLED=0` to confirm the phantom persists there (regression guard for the feature's value).

- [ ] **Step 3: Update the memory note**

Mark `trench-object-false-positive` resolved (implemented), linking the spec and plan, and noting the acceptance result (culled? robot re-acquired?). Update `MEMORY.md` pointer text accordingly.

- [ ] **Step 4: Final commit (if any verification-driven tweaks were needed)**

```bash
git add -A
git commit -m "test(sam2): verify hijack cull on 2026hop_qm72 acceptance match"
```

---

## Notes for the implementer

- **Run tests with `.venv/bin/python -m pytest`** — the base conda python has no pytest (memory `test-env-uses-venv`).
- **The loop rewrite in Task 3 is the riskiest change.** Run the *entire* `tests/field/test_sam2_tracker.py` after Task 3 (not just the new tests) — the existing re-prompt/seed/cap/bootstrap/diagnostics tests are the regression guard that the rewrite preserved behavior.
- **`IdentityDiagnosticRecorder` event access** is already hardwired in the tests: events are in the private `_events` list, payload key `"event"`, and the `sam2.object_culled` name must be registered on `DiagnosticEvents` (Task 4 Step 3a) or `record()` silently drops it.
- **`bottom_center` is already exported** from `sam2_tracker.py` (line 85) — import it in tests as shown; it's the natural pixel-space projector for the `project_fn=None` fallback too.
- **DRY:** the new-object birth-evidence IoS computation (Task 3 seeding block) duplicates the IoS-vs-existing loop; it is carried over verbatim from the current code to keep the rewrite behavior-preserving. Do not refactor it in this plan.

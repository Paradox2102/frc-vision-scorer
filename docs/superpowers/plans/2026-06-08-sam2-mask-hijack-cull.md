# Detector-Absence Cull for Hijacked SAM2 Masks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cull a SAM2 object whose mask has hijacked onto a static field structure (the trench), by detecting that fresh detector boxes fail to back the mask across a window of re-prompts (detector-absence) while the mask is stationary in field space — freeing its object slot so the real moving robot re-acquires.

**Architecture:** A second object-retirement trigger inside `Sam2FieldTracker.track`'s existing re-prompt loop. At each re-prompt boundary, each active object's latest mask is checked for corroboration against the freshly-detected boxes the loop already produces. Corroboration outcomes accumulate in a **bounded sliding window** per object (not a reset-on-first-hit counter); an object that is uncorroborated on most of its recent window **and** stationary in field meters is culled (its masks stop being recorded; it drops out of the active-roster cap). The detector result is **tri-state** — a detector outage/unreadable-frame does not count as absence. Stationarity is measured via an injected `project_fn` so a panning camera cannot disguise a static trench as moving.

**Safety posture (important):** Detector-absence + stationarity cannot, by itself, distinguish a trench from a real robot parked in a persistent detector blind spot (Codex adversarial review finding #2). The mitigations are layered — it requires *both* the bumper and whole-robot detectors to miss; the sliding window tolerates intermittent detections; and **the feature ships disabled by default** (`CULL_HIJACK_ENABLED=0`) until the Task 6 acceptance + adversarial replay tests demonstrate no false culls, at which point the default flips on.

**Tech Stack:** Pure-NumPy helpers in `web/field/sam2_tracker.py` (torch-free, unit-testable); the SAM2 wrapper's chunked re-prompt loop; `pytest` with the existing `RepromptStubPredictor`-style fakes (no GPU). Run tests with `.venv/bin/python -m pytest` (base conda python has no pytest).

**Spec:** `docs/superpowers/specs/2026-06-08-sam2-mask-hijack-cull-design.md` (see its "Revision 2026-06-08 (post-adversarial-review)" section for the predicate change).

---

## File Structure

- **`web/field/sam2_tracker.py`** (modify) — add three pure helpers (`mask_corroborated`, `is_stationary`, `absent_enough`) next to the existing pure helpers; add module-level config constants alongside the existing `MASK_*` knobs; add `cull_*` + `project_fn` params to `track()`; integrate tri-state detector handling, sliding-window corroboration, stationarity, and the cull into the re-prompt boundary block, plus a `culled_set` skip into `_record_chunk`.
- **`web/field/identity_diagnostics.py`** (modify) — register `SAM2_OBJECT_CULLED = "sam2.object_culled"` on `DiagnosticEvents`. **Required:** `IdentityDiagnosticRecorder.record()` validates every event name against `ALL_EVENTS` (rebuilt from `DiagnosticEvents` at import) and *silently drops* (into `_dropped_logs`) any unregistered name. Without registration the cull diagnostic is never recorded — in tests or production.
- **`web/app.py`** (modify) — build a `project_fn` bound to each view's calibration and pass it into `sam2_tracker.track`; make the re-prompt `detect_fn` return `None` (not `[]`) on an unreadable frame so the tracker can tell "detector unavailable" from "detector ran, found nothing." No new constants here (knobs live in `sam2_tracker.py`).
- **`tests/field/test_sam2_tracker.py`** (modify) — unit tests for the three helpers; a `CullStubPredictor` fake and integration tests for: cull on persistent absence, spare moving, spare corroborated, **intermittent corroboration still culls**, **detector outage preserves evidence**, **lifetime-cap blocks reseed (documented limitation)**, freed-active-slot reseed, `None`-projection conservatism, and the kill-switch.

Knob convention note: `sam2_tracker.py` already reads tunables from env at module scope (`MASK_MAX_ASPECT = float(os.environ.get(...))`, lines 26-28). The `CULL_*` knobs follow that pattern, and `track()` exposes them as parameters defaulting to the module constants (mirroring `new_obj_ios_thresh`).

---

## Task 1: Pure helper — `mask_corroborated`

Decides whether any fresh detector box backs an existing mask (the per-boundary corroboration test). The per-object complement of `match_new_detections`.

**Files:**
- Modify: `web/field/sam2_tracker.py` (add after `match_new_detections`, ~line 152)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing tests**

Add near the other pure-helper tests (after `test_match_new_detections_flags_unmatched`):

```python
def test_mask_corroborated_true_when_box_overlaps():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    assert mask_corroborated(mask, [(6, 2, 10, 6)], (60, 20), ios_thresh=0.3) is True


def test_mask_corroborated_false_when_no_box_overlaps():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    assert mask_corroborated(mask, [(40, 2, 44, 6)], (60, 20), ios_thresh=0.3) is False


def test_mask_corroborated_false_when_no_boxes():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    assert mask_corroborated(mask, [], (60, 20), ios_thresh=0.3) is False


def test_mask_corroborated_ignores_color_conflicting_box():
    from web.field.sam2_tracker import mask_corroborated
    mask = np.zeros((20, 60), bool)
    mask[2:6, 6:10] = True
    box = (6, 2, 10, 6)
    assert mask_corroborated(mask, [box], (60, 20), ios_thresh=0.3,
                             mask_color="red", new_colors=["blue"]) is False
    assert mask_corroborated(mask, [box], (60, 20), ios_thresh=0.3,
                             mask_color="red", new_colors=["red"]) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k mask_corroborated -v`
Expected: FAIL — `ImportError: cannot import name 'mask_corroborated'`.

- [ ] **Step 3: Implement the helper**

After `match_new_detections` (~line 152):

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

    The per-object complement of :func:`match_new_detections`. Used as the
    per-boundary corroboration test for the hijack cull. Color-conflicting
    detections are ignored (a blue box cannot corroborate a red mask).
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
git commit -m "feat(sam2): mask_corroborated helper (per-boundary corroboration test)"
```

---

## Task 2: Pure helpers — `is_stationary` and `absent_enough`

`is_stationary` is the field-space safety gate. `absent_enough` is the sliding-window absence decision that replaces a reset-on-first-hit counter — so one intermittent overlapping detection (a robot crossing the trench, or a whole-robot false positive) cannot immortalize a hijacked mask (Codex finding #5).

**Files:**
- Modify: `web/field/sam2_tracker.py` (add after `mask_corroborated`)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_is_stationary_true_for_fixed_positions():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0), (10.1, 5.0), (9.95, 5.05)], 0.5) is True


def test_is_stationary_false_for_moving_positions():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0), (11.0, 5.0), (12.0, 5.0)], 0.5) is False


def test_is_stationary_false_when_insufficient_known_positions():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0)], 0.5) is False
    assert is_stationary([None, (10.0, 5.0)], 0.5) is False
    assert is_stationary([None, None], 0.5) is False


def test_is_stationary_ignores_none_entries():
    from web.field.sam2_tracker import is_stationary
    assert is_stationary([(10.0, 5.0), None, (10.2, 5.1)], 0.5) is True


def test_absent_enough_true_for_persistent_absence():
    from web.field.sam2_tracker import absent_enough
    # window of corroboration outcomes; False = uncorroborated (absent)
    assert absent_enough([False, False, False], k=3) is True
    assert absent_enough([False, False, False, False], k=3) is True


def test_absent_enough_true_despite_one_intermittent_hit():
    from web.field.sam2_tracker import absent_enough
    # 3 absences within a 4-wide window survive a single stray corroboration
    assert absent_enough([True, False, False, False], k=3) is True


def test_absent_enough_false_when_mostly_corroborated():
    from web.field.sam2_tracker import absent_enough
    assert absent_enough([True, True, True], k=3) is False
    assert absent_enough([True, True, False, True], k=3) is False


def test_absent_enough_false_before_window_fills():
    from web.field.sam2_tracker import absent_enough
    assert absent_enough([False, False], k=3) is False  # not yet k absences
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "is_stationary or absent_enough" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the helpers**

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


def absent_enough(history, k: int) -> bool:
    """Return True if a bounded corroboration ``history`` shows >= ``k`` absences.

    ``history`` is an iterable of booleans over the last N *available* re-prompt
    boundaries (True = the mask was corroborated by a fresh detection that
    boundary, False = absent). Counting absences within a bounded window — rather
    than resetting a counter to zero on any single corroboration — means one
    intermittent overlap (a robot crossing the trench, or a whole-robot false
    positive on it) cannot indefinitely preserve a hijacked mask. The caller sizes
    the window (maxlen) with slack over ``k`` so a genuinely-present robot,
    corroborated on most boundaries, never reaches ``k`` absences.
    """
    absent = sum(1 for c in history if not c)
    return absent >= k
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "is_stationary or absent_enough" -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): is_stationary + absent_enough (windowed cull gates)"
```

---

## Task 3: Config knobs + `track()` params + tri-state detector decouple

Two coupled behavioral changes, no cull logic yet:

1. **Run-detector-when-capped decouple.** Today `detect_fn` runs only when `not capped` (`sam2_tracker.py:579`). The phantom hurts most when it fills the last active slot → capped → detector skipped → no corroboration data → cull deadlocks. Make the detector run at every boundary when the cull is enabled.
2. **Tri-state detector result.** Distinguish "detector ran, found nothing" (`[]` → counts as absence) from "detector unavailable" (`detect_fn` raised, or returned `None` → does **not** count as absence, evidence preserved). Without this, a transient outage reads as universal absence and culls every stationary robot (Codex finding #1).

**Files:**
- Modify: `web/field/sam2_tracker.py` (constants ~line 28; `track()` signature ~275-291; state init ~355; re-prompt loop ~555-698)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing tests**

Add the shared fake and the decouple/outage tests:

```python
class CullStubPredictor:
    """Reprompt stub with controllable per-object motion, for cull tests.

    Object ``oid`` lives at x in [10*oid, 10*oid+4). An oid in ``moving`` drifts
    +3 px in x per frame (its projected position changes across boundaries);
    others are static. Honours start_frame_idx / max_frame_num_to_track.
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
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=2, cull_enabled=True,
    )
    assert calls, "detect_fn must run at capped boundaries when cull enabled"


def test_detector_skipped_at_capped_boundary_when_cull_disabled():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        max_active_objects=2, cull_enabled=False,
    )
    assert calls == [], "detect_fn must NOT run at capped boundaries when cull disabled"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "capped_boundary_when_cull" -v`
Expected: FAIL — `track()` got an unexpected keyword argument `cull_enabled`.

- [ ] **Step 3a: Add module constants**

After the `MASK_*` knobs (~line 28):

```python
# Hijack-cull knobs (detector-absence + stationary). A SAM2 object whose mask has
# drifted onto a static field structure stays "visible" forever, so the existing
# absence-based retirement never fires. These drive a second retirement trigger.
# See docs/superpowers/specs/2026-06-08-sam2-mask-hijack-cull-design.md.
#
# DEFAULT OFF: detector-absence + stationarity cannot distinguish a trench from a
# real robot parked in a persistent detector blind spot. Enable only after the
# acceptance + adversarial replay tests (plan Task 6) show no false culls.
CULL_HIJACK_ENABLED = os.environ.get("CULL_HIJACK_ENABLED", "0") not in ("0", "false", "False")
# Absences required *within the evidence window* before a stationary object is culled.
CULL_UNCORROBORATED_REPROMPTS = int(os.environ.get("CULL_UNCORROBORATED_REPROMPTS", "3"))
# Sliding-window length of corroboration outcomes kept per object. Must be >= K;
# the slack (W - K) is how many intermittent corroborations the cull tolerates
# without resetting suspicion.
CULL_EVIDENCE_WINDOW = int(os.environ.get("CULL_EVIDENCE_WINDOW", "5"))
# Min IoS of a fresh detection vs the mask to count as corroboration (lenient ->
# conservative culling).
CULL_CORROB_IOS = float(os.environ.get("CULL_CORROB_IOS", "0.3"))
# Max field-position span over the window to count as stationary.
CULL_STATIONARY_M = float(os.environ.get("CULL_STATIONARY_M", "0.5"))
```

- [ ] **Step 3b: Add `track()` parameters**

After `new_obj_ios_thresh` (~line 288):

```python
        cull_enabled: bool = CULL_HIJACK_ENABLED,
        cull_uncorroborated_reprompts: int = CULL_UNCORROBORATED_REPROMPTS,
        cull_evidence_window: int = CULL_EVIDENCE_WINDOW,
        cull_corrob_ios: float = CULL_CORROB_IOS,
        cull_stationary_m: float = CULL_STATIONARY_M,
        project_fn=None,
```

Docstring note for `project_fn`: "Callable mapping a mask box (tracker pixel coords) to field meters (x, y), or None when unprojectable; measures stationarity in field space so camera pan can't disguise a static structure as moving. When None, falls back to pixel bottom-center (ego-motion-confounded; under-culls)." And for the detector contract: "``detect_fn`` returning ``None`` (or raising) is treated as *detector unavailable* for that boundary — it does not contribute absence evidence. An empty list means the detector ran and found nothing (counts as absence)."

- [ ] **Step 3c: Declare per-object cull state before `_record_chunk`**

Next to `retired_set` (~line 355). Use a bounded deque for the evidence window:

```python
        from collections import deque
        retired_set: set[int] = set()
        culled_set: set[int] = set()
        corroboration_history: dict[int, deque] = {}
        field_pos_window: dict[int, list] = {}
```

- [ ] **Step 3d: Rewrite the boundary block (decouple + tri-state; cull placeholder)**

Replace the current boundary block (`sam2_tracker.py:563-696`, from `# Re-prompt at the chunk boundary:` to just before `cursor = last_idx + 1`):

```python
                    # Re-prompt at the chunk boundary. Recompute the active set
                    # (excluding culled), retiring absent objects.
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

                    # Latest mask + color per active object (for corroboration & seeding).
                    existing = {}
                    existing_colors = {}
                    for oid in active_objects:
                        lv = last_visible_frame.get(oid)
                        if lv is not None and oid in results.get(lv, {}):
                            existing[oid] = results[lv][oid]["mask"]
                            if object_metadata is not None:
                                existing_colors[oid] = object_metadata.get(oid)

                    # Tri-state detector. detector_available is False when the
                    # detector could not run (detect_fn raised or returned None);
                    # such boundaries contribute NO absence evidence. Run the
                    # detector when seeding (not capped) OR for corroboration
                    # (cull_enabled) — guarded so the kill switch fully reverts.
                    run_detector = detect_fn is not None and (not capped or cull_enabled)
                    detector_available = False
                    new_boxes, new_metadata = [], []
                    if run_detector:
                        frame_path = (
                            frame_paths[last_idx] if last_idx < len(frame_paths) else None
                        )
                        try:
                            detected = detect_fn(last_idx, frame_path)
                        except Exception:
                            detected = None
                        if detected is None:
                            detector_available = False
                        else:
                            detector_available = True
                            new_boxes, new_metadata = _normalize_detections(detected)
                            if diagnostics:
                                for box in new_boxes:
                                    diagnostics.record(
                                        "sam2.reprompt_candidate",
                                        details={"frame_idx": last_idx, "box": list(box)},
                                    )

                    # ---- Hijack cull (filled in Task 4) ----
                    # if cull_enabled and detector_available and existing: ...

                    # ---- New-object seeding (unchanged logic, gated on capped) ----
                    if not capped and detector_available and new_boxes:
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
                                state, frame_idx=last_idx, obj_id=next_obj_id,
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
                    elif capped and detector_available and new_boxes:
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

- [ ] **Step 4: Run the new tests, then the whole file**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "capped_boundary_when_cull" -v`
Expected: 2 passed.

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -v`
Expected: all prior tests still pass (the loop rewrite is behavior-preserving when `cull_enabled=False` and the detector is available).

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): tri-state reprompt detector + run-when-capped (cull groundwork)"
```

---

## Task 4: Sliding-window cull decision, stationarity, and cull effect

Fills the placeholder. Accumulates corroboration outcomes per object only on available boundaries, decides culls by `absent_enough` + `is_stationary`, stops recording culled masks, and drops culled objects from the active roster.

**Files:**
- Modify: `web/field/identity_diagnostics.py` (register `SAM2_OBJECT_CULLED`)
- Modify: `web/field/sam2_tracker.py` (the `_record_chunk` skip ~line 491-503; the cull placeholder)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing integration tests**

```python
def _ident_diag():
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    return IdentityDiagnosticRecorder()


def _culled_ids(diagnostics):
    # Recorder stores events in the private `_events` list; event name under key
    # "event"; object_id stays inside "details" (not a hoisted standard field).
    return {e["details"]["object_id"]
            for e in diagnostics._events
            if e.get("event") == "sam2.object_culled"}


def test_cull_removes_uncorroborated_stationary_object():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()                       # objs 1, 2 static
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue")]         # corroborate obj 1 only

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_corrob_ios=0.3, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]
    culled = [e for e in diag._events if e.get("event") == "sam2.object_culled"]
    assert any(e["details"]["object_id"] == 2
               and e["details"]["reason"] == "detector_absent_stationary"
               for e in culled)


def test_cull_spares_moving_object_even_if_uncorroborated():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={2})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1, 0), stub._rect_for(2, 0)],
        reprompt_every=2, detect_fn=lambda *_a: [], max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_cull_spares_corroborated_stationary_object():
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=lambda *_a: [stub._rect_for(2)],
        max_active_objects=6, cull_enabled=True, cull_uncorroborated_reprompts=3,
        cull_evidence_window=5, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 not in _culled_ids(diag)
    assert 2 in result[stub.NUM_FRAMES - 1]


def test_cull_survives_intermittent_corroboration():
    """A robot crossing the trench (or a periodic FP) corroborates obj 2 on an
    occasional boundary; the windowed evidence still culls it (Codex finding #5)."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    seen = {"n": 0}

    def detect_fn(local_idx, frame_path):
        seen["n"] += 1
        # Corroborate obj 2 on only the first boundary, never again.
        return [stub._rect_for(2)] if seen["n"] == 1 else []

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)


def test_detector_outage_does_not_cull():
    """detect_fn returning None (unavailable) contributes no absence evidence, so a
    total outage never culls a real stationary robot (Codex finding #1)."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=lambda *_a: None,  # detector unavailable every boundary
        max_active_objects=6, cull_enabled=True, cull_uncorroborated_reprompts=3,
        cull_evidence_window=5, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert _culled_ids(diag) == set()
    assert 1 in result[stub.NUM_FRAMES - 1] and 2 in result[stub.NUM_FRAMES - 1]


def test_cull_frees_active_slot_and_real_robot_reseeds():
    """Deadlock case: phantom fills the last active slot; culling it lets the real
    moving robot (obj 3) seed into the freed slot. max_lifetime_objects is generous
    so the ACTIVE cap is the binding one."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={3})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(3, local_idx), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)
    assert any(oid == 3 for (_f, oid) in stub.add_calls)
    assert 3 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]


def test_cull_does_not_reseed_when_lifetime_capped():
    """Documented limitation (Codex finding #3): culling frees the ACTIVE count but
    never the LIFETIME cap (next_obj_id only increments). When the lifetime cap is
    the binding one, the phantom is still culled but the real robot CANNOT re-seed.
    This test pins that behavior so it is explicit, not silent."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor(moving={3})
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        return [(stub._rect_for(1), "blue"), (stub._rect_for(3, local_idx), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn, object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=2,   # lifetime cap is binding
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag)                       # still culled
    assert all(oid != 3 for (_f, oid) in stub.add_calls)  # but NOT re-seeded
    # The blocked seed is logged so the limitation is observable, not silent.
    assert any(e.get("event") == "sam2.reprompt_rejected"
               and e.get("reason") == "lifetime_cap" for e in diag._events)


def test_none_projection_is_conservative():
    """If project_fn can't project the box (returns None), stationarity is
    unconfirmable and the object is not culled."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=lambda *_a: [], max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5, project_fn=lambda box: None, diagnostics=diag,
    )
    assert _culled_ids(diag) == set()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "cull_ or detector_outage or none_projection" -v`
Expected: FAIL — no `sam2.object_culled` events; obj 2 persists; obj 3 unseeded in the freeing test.

- [ ] **Step 3a: Register the diagnostic event**

In `web/field/identity_diagnostics.py`, in `DiagnosticEvents` `# SAM2` group (after `SAM2_CHUNK_COMPLETE`, ~line 22):

```python
    SAM2_OBJECT_CULLED = "sam2.object_culled"
```

`ALL_EVENTS` rebuilds from `DiagnosticEvents` at import (line 85), so this suffices for validation. Do **not** add a `DiagnosticReasons` member — the cull reason rides *inside* `details` (Step 3c), bypassing the `ALL_REASONS` validator.

- [ ] **Step 3b: Skip culled objects in `_record_chunk`**

Right after `oid_int = int(obj_id)` (~line 476), alongside the implausible-mask skip:

```python
                    if oid_int in culled_set:
                        continue  # culled hijack mask — stop emitting it
```

- [ ] **Step 3c: Implement the cull block**

Replace the `# ---- Hijack cull (filled in Task 4) ----` placeholder. `reason` is placed **inside** `details` deliberately (a top-level `reason=` is validated against `ALL_REASONS` and would be dropped):

```python
                    if cull_enabled and detector_available and existing:
                        h, w = next(iter(existing.values())).shape
                        for oid, mask in list(existing.items()):
                            corroborated = mask_corroborated(
                                mask, new_boxes, (w, h),
                                ios_thresh=cull_corrob_ios,
                                mask_color=(existing_colors.get(oid)
                                            if existing_colors else None),
                                new_colors=new_metadata,
                            )
                            hist = corroboration_history.setdefault(
                                oid, deque(maxlen=cull_evidence_window))
                            hist.append(bool(corroborated))

                            box = results[last_visible_frame[oid]][oid]["box"]
                            pos = project_fn(box) if project_fn is not None else bottom_center(box)
                            win = field_pos_window.setdefault(oid, [])
                            win.append(pos)
                            if len(win) > cull_evidence_window:
                                del win[0]

                            if (absent_enough(hist, cull_uncorroborated_reprompts)
                                    and is_stationary(win, cull_stationary_m)):
                                culled_set.add(oid)
                                if diagnostics:
                                    diagnostics.record(
                                        "sam2.object_culled",
                                        details={
                                            "object_id": oid,
                                            "frame_idx": last_idx,
                                            "reason": "detector_absent_stationary",
                                            "absences": sum(1 for c in hist if not c),
                                            "window": len(hist),
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

This recomputes `capped` after culling so the same boundary's seeding block can use a freed **active** slot. (It cannot relieve the lifetime cap — see `test_cull_does_not_reseed_when_lifetime_capped`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -k "cull_ or detector_outage or none_projection" -v`
Expected: all passed.

Run the full file:
Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py web/field/identity_diagnostics.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): windowed detector-absence cull for hijacked masks"
```

---

## Task 5: Wire `project_fn` into `app.py` (correct top-crop) + tri-state detect_fn

Give the cull a real field-space stationarity measurement bound to each view's calibration, and make the production `detect_fn` signal "unavailable" on unreadable frames.

**Files:**
- Modify: `web/app.py` (the `make_detect_fn` closure ~4131-4218; the `sam2_tracker.track(...)` call ~4225-4238; mirror the result-projection at `app.py:4451,4471-4472`)

- [ ] **Step 1: Read the projection + offset formulas to mirror them correctly**

Read `web/app.py:4448-4476` (the result consumer) and `4156-4179` (the detect_fn closure's own scaling).

Key fact (Codex finding #4): exported SAM2 frames are **already top-cropped**. The consumer's offset is `int(view_frame.shape[0] * top_crop_frac)` using the **original** (uncropped) height. From a cropped frame you only have `cropped_full_h = round(img.shape[0] / SAM2_SCALE)`; the original height is `cropped_full_h / (1 - top_crop_frac)`. Therefore the correct y-offset is:

`y_off = cropped_full_h * top_crop_frac / (1 - top_crop_frac)`

This is exactly the formula the `detect_fn` closure already uses at `app.py:4167` — and it is algebraically equal to the consumer's `original_h * top_crop_frac` (since `original_h = cropped_full_h / (1 - tc)`). The "two divergent formulas" are the same value expressed from different heights. `project_fn` must use the `4167` form because only the cropped height is available at the track-call site. **Do not** use `cropped_full_h * top_crop_frac` (off by a factor of `1/(1-tc)`, which at oblique corner-camera calibration can shift field motion across the 0.5 m stationarity threshold).

- [ ] **Step 2: Make the re-prompt `detect_fn` tri-state**

In `make_detect_fn`'s inner `detect_fn` (`app.py:4131-4136`), an unreadable frame currently returns `[]`. Change it to return `None` so the tracker treats it as *detector unavailable* (no absence evidence) rather than *found nothing*:

```python
                            def detect_fn(local_idx, frame_path):
                                if not frame_path:
                                    return None          # was: []
                                img = cv2.imread(frame_path)
                                if img is None:
                                    return None          # was: []
                                ...
```

Leave the successful path (returning the merged list, possibly empty) unchanged — an empty list still legitimately means "ran, found nothing."

- [ ] **Step 3: Build and pass `project_fn` at the track call**

Immediately before `sam2_tracker.track(...)` (~4225), inside `for view in self.views`. `SAM2_SCALE = 0.5` is already in scope (line 3931). Compute the cropped full-res height from one exported SAM2 frame (the same `frames_dir` passed to `track`), then the corrected offset:

```python
                        _sam2_scale = 0.5
                        _top_crop = (
                            (getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0)
                            if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                        )
                        # Cropped full-res height from an exported SAM2 frame, then
                        # recover the offset matching the consumer (app.py:4167 form).
                        _exported = sorted(
                            f for f in os.listdir(tmp_dir_obj.name)
                            if f.endswith(".jpg") or f.endswith(".jpeg")
                        )
                        _cropped_full_h = None
                        if _exported:
                            _img0 = cv2.imread(os.path.join(tmp_dir_obj.name, _exported[0]))
                            if _img0 is not None:
                                _cropped_full_h = int(round(_img0.shape[0] / _sam2_scale))
                        _y_off = (
                            int(_cropped_full_h * _top_crop / max(1.0 - _top_crop, 1e-6))
                            if (_cropped_full_h is not None and _top_crop > 0) else 0
                        )

                        def _make_project_fn(cal, y_off, scale):
                            def _project(box):
                                x1, y1, x2, y2 = box
                                fx1, fx2 = x1 / scale, x2 / scale
                                fy2 = y2 / scale + y_off
                                return cal.image_to_field(((fx1 + fx2) / 2.0, fy2))
                            return _project

                        _project_fn = _make_project_fn(view.calibration, _y_off, _sam2_scale)
```

Then add `project_fn=_project_fn,` to the `sam2_tracker.track(...)` kwargs (next to `diagnostics=self.identity_diagnostics`). Bind via the `_make_project_fn` factory (not a bare loop-variable closure). `image_to_field` may return `None` — handled by `is_stationary`.

- [ ] **Step 4: Add a projection-parity smoke check**

Because `project_fn` must match the downstream consumer's field mapping, add a lightweight assertion-style check (a small unit test in `tests/field/` or an inline `assert` exercised by an existing app-level test if one covers this path): for a synthetic `top_crop_frac > 0` and a stub calibration whose `image_to_field` is identity, `project_fn(box)` must equal the consumer's `((fx1+fx2)/2, fy2)` computed with `sam2_y_offset = int(original_h * top_crop_frac)` where `original_h = cropped_full_h/(1-top_crop_frac)`. If no existing test reaches this code, add `tests/field/test_sam2_project_fn.py` with a pure reconstruction of the two formulas asserting equality at `top_crop_frac ∈ {0.0, 0.2}`.

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('app.py parses')"`
Expected: `app.py parses`.

Run: `.venv/bin/python -m pytest tests/ -k "sam2 or fusion or detect_bridge or tracker or project_fn" -q`
Expected: pass.

```bash
git add web/app.py tests/field/test_sam2_project_fn.py
git commit -m "feat(app): tri-state detect_fn + correct-offset field project_fn for SAM2 cull"
```

---

## Task 6: Acceptance gate, adversarial replay, and enablement

The feature ships **disabled by default**. This task is the gate that must pass before flipping `CULL_HIJACK_ENABLED` on. **Do not flip the default to `"1"` unless every check below passes.**

**Files:** `web/field/sam2_tracker.py` (only to flip the default if the gate passes); memory note; possibly `docs/`.

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass. Note (do not fix here) any unrelated pre-existing failures.

- [ ] **Step 2: Acceptance run on the trench-hijack match (`2026hop_qm72`)**

See memory `acceptance-harness-download` for footage + the app's Python processing API (CLI `yt-dlp` 403s; use the app API + deno + H.264). Process the match twice:

- `CULL_HIJACK_ENABLED=1 ...` → confirm a `sam2.object_culled` event (`reason="detector_absent_stationary"`) fires for the trench object, the phantom disappears from the field map for the rest of the match, and the previously-lost robot is tracked at its moving positions afterward.
- `CULL_HIJACK_ENABLED=0 ...` → confirm the phantom persists (proves the feature's value).

- [ ] **Step 3: Adversarial replay/inspection for the residual-risk cases (Codex findings #1, #2, #5)**

On the same or representative match footage with the cull enabled, confirm via diagnostics that **no real robot is wrongly culled** in these cases:
- **Parked / temporarily-occluded robot:** a robot that stops (scoring/defending/climbing) and/or passes behind the hub is **not** culled (the whole-robot detector keeps corroborating it). This is the key check for finding #2 — if a parked real robot *is* culled, the detector-absence premise is too weak; do **not** enable by default, and record that escalation to an independent positive hijack signal (appearance discontinuity at the transfer frame) is required.
- **Detector outage:** confirmed by `test_detector_outage_does_not_cull`; spot-check that real GPU/inference hiccups in a run don't produce spurious culls.
- **Intermittent trench corroboration:** confirmed by `test_cull_survives_intermittent_corroboration`.

- [ ] **Step 4: Decide enablement**

If Steps 1-3 pass cleanly (phantom culled, robot re-acquired, zero false culls of real robots), flip the default:

```python
CULL_HIJACK_ENABLED = os.environ.get("CULL_HIJACK_ENABLED", "1") not in ("0", "false", "False")
```

and commit. **If any false cull occurred,** leave the default `"0"`, document the failing case, and stop — the predicate needs the deferred appearance-discontinuity positive signal before it is safe to enable. Surface this to the user rather than enabling.

- [ ] **Step 5: Update the memory note**

Update `trench-object-false-positive` to resolved/implemented, linking spec + plan, recording the acceptance result (culled? robot re-acquired? any false culls?) and the final default state of `CULL_HIJACK_ENABLED`. Update `MEMORY.md` pointer text.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "test(sam2): acceptance + adversarial verification of hijack cull on 2026hop_qm72"
```

---

## Notes for the implementer

- **Run tests with `.venv/bin/python -m pytest`** — base conda python has no pytest (memory `test-env-uses-venv`).
- **The Task 3 loop rewrite is the riskiest change.** After Task 3 run the *entire* `tests/field/test_sam2_tracker.py` — the existing re-prompt/seed/cap/bootstrap/diagnostics tests are the behavior-preservation guard.
- **Tri-state contract:** the cull only updates evidence and may cull on a boundary where `detector_available` is True. A `None`/exception boundary contributes nothing — this is the finding-#1 fix; keep it intact when editing.
- **Sliding window vs counter:** the cull uses `absent_enough(history, k)` over a bounded deque, **not** a reset-on-corroboration integer. Do not "simplify" it back to a counter — that reintroduces finding #5.
- **`IdentityDiagnosticRecorder`:** events are in the private `_events` list, payload key `"event"`; the `sam2.object_culled` name must be registered (Task 4 Step 3a) or `record()` silently drops it.
- **`bottom_center` is already exported** from `sam2_tracker.py` (line 85) — the natural pixel-space projector for the `project_fn=None` fallback and for tests.
- **Lifetime-cap limitation is intentional and tested** (`test_cull_does_not_reseed_when_lifetime_capped`). Do not add ID-reservation / predictor-rebuild machinery — `max_lifetime_objects` is 18 in a 6-robot match and cannot bind; the heavier fix is YAGNI. The blocked-reseed is logged, not silent.
- **DRY:** the seeding block's birth-evidence IoS loop is carried over verbatim to keep the rewrite behavior-preserving. Do not refactor it in this plan.

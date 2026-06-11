# SAM2 Backward Backfill for Late-Seeded Objects — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A robot first detected mid-match (e.g. emerging from behind the hub at t=30s) gets its track extended *backward* in time from its seed frame, bounded by a horizon, without creating duplicate observations where the robot already existed under an older ID.

**Architecture:** `Sam2FieldTracker.track()` today runs forward chunked propagation; only objects prompted at the bootstrap frame get a reverse pass (`propagate_backward`). Objects seeded at later re-prompt boundaries are forward-only. This plan adds a **post-pass**: after forward tracking completes, a *second predictor session* is opened on the same frames dir, each late-seeded object is prompted at its seed frame with its seed box, and `propagate_in_video(reverse=True)` runs from the latest seed frame backward. Backfilled masks are merged **per-object** (`results[f].setdefault(oid, ...)` — never frame-replacement, which would clobber other objects), restricted to each object's own pre-seed window, and dropped wherever they overlap an already-tracked mask (the robot usually existed there under an older ID — backfill must not create duplicates for fusion to dedupe).

**Why a second session:** SAM2 propagates ALL objects in a state together; you cannot reverse-propagate a subset of the main session's objects. A fresh state holding only the late objects sidesteps this. Cost: one extra `init_state` (JPEG decode of the frames dir) plus reverse propagation over the horizon — acceptable because it runs once per view, not per object.

**Tech Stack:** Python, numpy, SAM2 offline video predictor protocol (`init_state` / `add_new_points_or_box` / `propagate_in_video(reverse=True)` / `reset_state`), stub predictors for tests (no GPU).

**Default OFF** behind `SAM2_BACKFILL_ENABLED`, with per-job `backfill_enabled` Form param on `/api/infer` (same pattern as `reanchor_enabled`). A/B before any default flip — measured on 2026hop qm72 (robots emerge from hub occlusion late) and qm24.

**Key existing code to read first:**
- `web/field/sam2_tracker.py` — `track()`: the `propagate_backward` block (the existing reverse-pass pattern to imitate), `_propagate()` (how `reverse=True` is passed), `first_prompted_frame` / `prompt_idx` / `bootstrap_idx` bookkeeping, `plausible_robot_mask`, `_ios`.
- `tests/field/test_sam2_tracker.py` — `RepromptStubPredictor` (honors `start_frame_idx`/`max_frame_num_to_track`), `CullStubPredictor`, `_make_mask_array`, `_ident_diag`.
- `web/app.py` — `/api/infer` per-job override pattern (`reanchor_enabled`), the `sam2_tracker.track(...)` call site (kwargs-splat pattern for optional params).

**Important context for the engineer:**
- `results` is `{frame_idx: {obj_id: {"box","mask","evidence"}}}`. Frames the forward pass never emitted (before bootstrap when `propagate_backward=False`) may be missing entirely; `setdefault` both the frame dict and the object entry.
- Evidence dicts must carry `"type": "backfilled"` so downstream consumers / replay review can distinguish them. Grep `evidence` consumers first (`web/app.py` reads `info["box"]`/`info["mask"]`; nothing branches on type except diagnostics/replay).
- The tracker may run with a stub: never import torch at module level; the new code lives inside `track()` / a helper and uses the same `_maybe_autocast`/`inference_mode` wrapper via `_propagate`-style logic on the *backfill* state.
- TDD per @superpowers:test-driven-development. Run tests with `.venv/bin/python -m pytest` (base conda python has no pytest).

---

## File Structure

- Modify: `web/field/sam2_tracker.py`
  - New module knobs: `BACKFILL_ENABLED`, `BACKFILL_MAX_SECONDS_FRAMES` (frames, since the tracker has no fps), `BACKFILL_OVERLAP_IOS`.
  - New pure helper `merge_backfilled_frame(...)` (testable without a predictor).
  - New `track()` params: `backfill_enabled`, `backfill_max_frames`, `backfill_overlap_ios`; bookkeeping `seed_box[oid]`; post-pass after the forward loop.
- Modify: `web/field/identity_diagnostics.py` — register `SAM2_OBJECT_BACKFILLED = "sam2.object_backfilled"` (the recorder silently drops unknown event names — this bit us with `object_reanchored`).
- Modify: `web/app.py` — job attr `backfill_enabled`, `/api/infer` Form param, kwargs-splat at the `track()` call.
- Test: `tests/field/test_sam2_tracker.py` — `BackfillStubPredictor` + tests.

---

### Task 1: Pure merge helper with overlap suppression

**Files:**
- Modify: `web/field/sam2_tracker.py` (top tier, next to the other pure helpers)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_merge_backfilled_frame_adds_new_object_only():
    """Backfill merges per-object; it never replaces an existing entry."""
    from web.field.sam2_tracker import merge_backfilled_frame
    existing_mask = _rect_mask(0, 0, 5, 10)
    frame_entry = {1: {"box": (0, 0, 5, 10), "mask": existing_mask,
                       "evidence": {"type": "propagated"}}}
    new_mask = _rect_mask(6, 0, 9, 10)
    added = merge_backfilled_frame(
        frame_entry, obj_id=7, mask=new_mask, overlap_ios=0.3)
    assert added is True
    assert 7 in frame_entry and 1 in frame_entry
    assert frame_entry[7]["evidence"]["type"] == "backfilled"
    assert frame_entry[7]["box"] == (6, 0, 9, 10)


def test_merge_backfilled_frame_suppressed_by_overlap():
    """Where the robot already exists under an older id, backfill must NOT
    add a duplicate observation."""
    from web.field.sam2_tracker import merge_backfilled_frame
    existing_mask = _rect_mask(0, 0, 5, 10)
    frame_entry = {1: {"box": (0, 0, 5, 10), "mask": existing_mask,
                       "evidence": {"type": "propagated"}}}
    overlapping = _rect_mask(1, 1, 4, 9)  # inside obj 1's mask
    added = merge_backfilled_frame(
        frame_entry, obj_id=7, mask=overlapping, overlap_ios=0.3)
    assert added is False
    assert 7 not in frame_entry


def test_merge_backfilled_frame_never_overwrites_same_id():
    """A frame already holding this oid (forward pass) wins."""
    from web.field.sam2_tracker import merge_backfilled_frame
    fwd_mask = _rect_mask(0, 0, 5, 10)
    frame_entry = {7: {"box": (0, 0, 5, 10), "mask": fwd_mask,
                       "evidence": {"type": "propagated"}}}
    added = merge_backfilled_frame(
        frame_entry, obj_id=7, mask=_rect_mask(6, 0, 9, 10), overlap_ios=0.3)
    assert added is False
    assert frame_entry[7]["evidence"]["type"] == "propagated"


def test_merge_backfilled_frame_rejects_implausible_mask():
    """The mask-shape filter applies to backfilled masks too."""
    from web.field.sam2_tracker import merge_backfilled_frame
    # tall, thin, top-heavy tadpole: wide head rows at top, 1px stalk below
    m = np.zeros((60, 30), bool)
    m[0:8, 10:22] = True   # head
    m[8:60, 15:16] = True  # stalk
    frame_entry = {}
    added = merge_backfilled_frame(frame_entry, obj_id=7, mask=m,
                                   overlap_ios=0.3)
    assert added is False and frame_entry == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -q -k merge_backfilled`
Expected: 4 FAILED (ImportError: cannot import `merge_backfilled_frame`)

- [ ] **Step 3: Implement the helper** (in the pure-helpers tier of `sam2_tracker.py`, after `_ios`)

```python
def merge_backfilled_frame(frame_entry: dict, obj_id: int, mask: np.ndarray,
                           overlap_ios: float) -> bool:
    """Merge one backfilled mask into one frame's entry dict, in place.

    Returns True if added. Refuses when: the frame already has this obj_id
    (the forward pass is authoritative), the mask shape is implausible for a
    robot, or the mask overlaps ANY already-recorded mask at IoS >=
    ``overlap_ios`` — that robot already exists there under an older id, and
    a duplicate would force fusion to dedupe two ids for one robot.
    """
    if obj_id in frame_entry:
        return False
    box = mask_to_box(mask)
    if box is None or not plausible_robot_mask(mask):
        return False
    for other in frame_entry.values():
        if _ios(mask, other["mask"]) >= overlap_ios:
            return False
    frame_entry[obj_id] = {
        "box": box,
        "mask": mask,
        "evidence": {"type": "backfilled", "gap_len": 0,
                     "detector_overlap": 0.0},
    }
    return True
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -q -k merge_backfilled`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): pure merge helper for backward-backfill masks"
```

---

### Task 2: Seed-box bookkeeping + module knobs

**Files:**
- Modify: `web/field/sam2_tracker.py`

- [ ] **Step 1: Add module knobs** (next to `REANCHOR_*`)

```python
# Backward backfill: after the forward pass, late-seeded objects (first
# prompted at a re-prompt boundary, not the bootstrap frame) are re-prompted
# in a SECOND predictor session and propagated in reverse over a bounded
# horizon, recovering the track before its first detection. Merged
# per-object; overlap-suppressed (see merge_backfilled_frame). DEFAULT OFF
# pending A/B — the recovered frames mostly predate identity, so the win is
# position/coverage, not team confirmation.
BACKFILL_ENABLED = os.environ.get("SAM2_BACKFILL_ENABLED", "0") not in ("0", "false", "False")
# Horizon in TRACKED frames (the tracker has no fps; app converts seconds).
BACKFILL_MAX_FRAMES = int(os.environ.get("SAM2_BACKFILL_MAX_FRAMES", "150"))
# A backfilled mask overlapping any existing mask at this IoS is dropped.
BACKFILL_OVERLAP_IOS = float(os.environ.get("SAM2_BACKFILL_OVERLAP_IOS", "0.3"))
```

- [ ] **Step 2: Record seed boxes.** `track()` already records `first_prompted_frame[oid]`. Add `seed_box: dict[int, tuple] = {}` beside it, and populate at all three seeding sites (initial prompt loop, bootstrap loop, boundary seeding) — each site already has the box in hand; store `tuple(float(v) for v in box)`. Also add `track()` params:

```python
backfill_enabled: bool = BACKFILL_ENABLED,
backfill_max_frames: int = BACKFILL_MAX_FRAMES,
backfill_overlap_ios: float = BACKFILL_OVERLAP_IOS,
```

with docstring entries mirroring the knob comments.

Note: a re-anchored object intentionally KEEPS its original `seed_box` /
`first_prompted_frame` — backfill should recover the track before its first
appearance, and the original seed is that anchor.

- [ ] **Step 3: Run the existing suite (no behavior change expected)**

Run: `.venv/bin/python -m pytest tests/field/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add web/field/sam2_tracker.py
git commit -m "feat(sam2): seed-box bookkeeping + backfill knobs (no behavior change)"
```

---

### Task 3: Diagnostics event registration

**Files:**
- Modify: `web/field/identity_diagnostics.py` (the `DiagnosticEvents` class)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Failing test**

```python
def test_backfill_event_name_is_registered():
    """The recorder DROPS unknown event names silently (bit us with
    object_reanchored) — assert registration explicitly."""
    from web.field.identity_diagnostics import ALL_EVENTS
    assert "sam2.object_backfilled" in ALL_EVENTS
```

- [ ] **Step 2: Run to verify it fails**, then add `SAM2_OBJECT_BACKFILLED = "sam2.object_backfilled"` to `DiagnosticEvents` (next to `SAM2_OBJECT_REANCHORED`).

- [ ] **Step 3: Run to verify pass; commit**

```bash
git add web/field/identity_diagnostics.py tests/field/test_sam2_tracker.py
git commit -m "feat(diagnostics): register sam2.object_backfilled event"
```

---

### Task 4: The backfill post-pass in track()

**Files:**
- Modify: `web/field/sam2_tracker.py` (`track()`, after the chunked `while True:` loop, INSIDE the `try:` so `finally` still resets state)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Write the stub + failing tests**

The stub must model: a late object invisible before its seed frame in the
forward pass, but recoverable in reverse from a second session. Track
`init_state` call count to assert session isolation.

```python
class BackfillStubPredictor:
    """Obj 1 lives at x=10 for the whole video. A 'late' robot at x=50
    exists physically from frame 4 but the forward pass only sees it from
    its seed (the detector finds it at the boundary, frame 6). In a reverse
    session, an object prompted at frame >= 4 emits masks back to frame 4.
    Frames are 0..11; reprompt boundary every 3 frames.
    """
    H = 20
    W = 200
    NUM_FRAMES = 12
    LATE_EXISTS_FROM = 4

    def __init__(self):
        self.init_calls = 0
        self.reverse_calls: list[tuple] = []

    def init_state(self, video_path: str, **kwargs) -> dict:
        self.init_calls += 1
        return {"objs": {}, "session": self.init_calls}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None) -> None:
        state["objs"][obj_id] = (frame_idx, tuple(float(v) for v in box))

    def _rect(self, obj_id, sx):
        return (int(sx), 2, int(sx) + 4, 6)

    def propagate_in_video(self, state, start_frame_idx=None,
                           max_frame_num_to_track=None, reverse=False):
        if reverse:
            self.reverse_calls.append(
                (state["session"], start_frame_idx, max_frame_num_to_track))
            start = start_frame_idx
            stop = max(0, start - (max_frame_num_to_track or start))
            for f in range(start, stop - 1, -1):
                oids, rects = [], []
                for oid, (pf, box) in sorted(state["objs"].items()):
                    if f >= self.LATE_EXISTS_FROM:
                        oids.append(oid)
                        rects.append((int(box[0]), 2, int(box[0]) + 4, 6))
                yield f, oids, _make_mask_array(self.H, self.W, rects)
            return
        start = 0 if start_frame_idx is None else start_frame_idx
        end = (self.NUM_FRAMES - 1 if max_frame_num_to_track is None
               else min(start + max_frame_num_to_track, self.NUM_FRAMES - 1))
        for f in range(start, end + 1):
            oids, rects = [], []
            for oid, (pf, box) in sorted(state["objs"].items()):
                if f >= pf:
                    oids.append(oid)
                    rects.append((int(box[0]), 2, int(box[0]) + 4, 6))
            yield f, oids, _make_mask_array(self.H, self.W, rects)

    def reset_state(self, state) -> None:
        pass


def test_backfill_extends_late_object_backward():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = BackfillStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # Late robot only becomes detectable from frame 6 — so it is seeded
        # at the boundary at frame 7, NOT the first boundary (frame 3, which
        # precedes LATE_EXISTS_FROM=4 and would leave nothing to backfill).
        if local_idx >= 6:
            return [((50, 2, 54, 6), "red")]
        return []

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=3, detect_fn=detect_fn,
        object_metadata={1: "blue"},
        backfill_enabled=True, backfill_max_frames=10,
        diagnostics=diag,
    )
    # With reprompt_every=3 the chunks are 0-3, 4-7, 8-11 (each chunk yields
    # start..start+3), so boundaries land at 3, 7, 11. The late robot is
    # detectable from frame 6 -> seeded at boundary 7; backfill must recover
    # frames 4..6 (LATE_EXISTS_FROM=4 bounds the reverse pass).
    late_oid = 2
    seed_f = min(f for f, ent in result.items() if late_oid in ent
                 and ent[late_oid]["evidence"]["type"] != "backfilled")
    backfilled = [f for f, ent in result.items() if late_oid in ent
                  and ent[late_oid]["evidence"]["type"] == "backfilled"]
    assert backfilled, "expected backfilled frames before the seed"
    assert max(backfilled) < seed_f
    assert min(backfilled) >= BackfillStubPredictor.LATE_EXISTS_FROM
    assert stub.init_calls == 2, "backfill must use a second session"
    assert any(e.get("event") == "sam2.object_backfilled"
               for e in diag._events)


def test_backfill_does_not_duplicate_where_object_already_tracked():
    """If a backfilled mask lands on frames where obj 1 already covers that
    spot, the overlap suppression drops it (merge helper already unit-tested;
    this is the integration check). NOTE: the FORWARD pass intentionally has
    no overlap dedup (two seeded ids can overlap during contact), so the
    assertion is scoped to BACKFILLED entries only.
    """
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = BackfillStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        # Detector box ON TOP of obj 1: color-conflict prevents claiming,
        # so it seeds a new id (2) at obj 1's exact spot.
        if local_idx >= 6:
            return [((10, 2, 14, 6), "red")]
        return []

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=3, detect_fn=detect_fn,
        object_metadata={1: "blue"},
        backfill_enabled=True, backfill_max_frames=10,
    )
    # Obj 1's mask covers (10,2,14,6) on every pre-seed frame, so EVERY
    # candidate backfill mask for obj 2 overlaps it at IoS 1.0 and must be
    # suppressed: zero backfilled entries for obj 2.
    # Positive control FIRST: the backfill session must actually have run
    # (otherwise the empty list below passes vacuously pre-implementation).
    assert stub.init_calls == 2, "backfill must open a second session"
    backfilled_2 = [f for f, ent in result.items() if 2 in ent
                    and ent[2]["evidence"]["type"] == "backfilled"]
    assert backfilled_2 == [], \
        f"backfill must not duplicate an already-tracked robot: {backfilled_2}"


def test_backfill_disabled_is_noop():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = BackfillStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)

    def detect_fn(local_idx, frame_path):
        return [((50, 2, 54, 6), "red")]

    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[(10, 2, 14, 6)],
        reprompt_every=3, detect_fn=detect_fn,
        backfill_enabled=False,
    )
    assert stub.init_calls == 1
    assert not any(ent[oid]["evidence"]["type"] == "backfilled"
                   for ent in result.values() for oid in ent)
```

- [ ] **Step 2: Run to verify failures**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -q -k backfill`
Expected: `test_backfill_extends_late_object_backward` FAILS on
`assert backfilled` and `test_backfill_does_not_duplicate_where_object_already_tracked`
FAILS on its `init_calls == 2` positive control (`backfill_enabled` is
accepted but inert — Task 2 added the params without behavior).
`test_backfill_disabled_is_noop` ALREADY PASSES at this point; that is
expected, not a smell.

Also: pin `cull_enabled=False` in all three tests' `track()` calls. The
cull now defaults ON, and obj 1 is detector-uncorroborated in every test;
with default tunables the cull lands harmlessly after the last chunk, but
an env override (e.g. `CULL_UNCORROBORATED_REPROMPTS=1`) would cull obj 1
mid-video and erase the masks the suppression test depends on. The existing
suite pins `cull_enabled` explicitly; follow that convention. Also drop the
unused `BackfillStubPredictor._rect` helper (rects are built inline).

- [ ] **Step 3: Implement the post-pass.** After the `while True:` chunk loop (still inside `try:`), append:

```python
            # ---- Backward backfill post-pass (late-seeded objects) ----
            if backfill_enabled and reprompting:
                late = {
                    oid: (first_prompted_frame[oid], seed_box[oid])
                    for oid in seed_box
                    if first_prompted_frame.get(oid, bootstrap_idx) > bootstrap_idx
                    and oid not in culled_set
                }
                if late:
                    # The main state is no longer needed; free it before
                    # opening the backfill session.
                    if hasattr(predictor, "reset_state"):
                        predictor.reset_state(state)
                    state = predictor.init_state(
                        video_path=frames_dir,
                        offload_video_to_cpu=self.offload_video_to_cpu,
                        offload_state_to_cpu=self.offload_state_to_cpu,
                    )
                    for oid, (sf, sb) in late.items():
                        predictor.add_new_points_or_box(
                            state, frame_idx=sf, obj_id=oid,
                            box=np.asarray(sb, dtype=np.float32),
                        )
                    start = max(sf for sf, _sb in late.values())
                    earliest = max(0, min(sf for sf, _sb in late.values())
                                   - backfill_max_frames)
                    horizon = start - earliest
                    backfilled_frames: dict[int, int] = {}
                    for f_idx, obj_ids, mask_logits in _propagate(
                        start, horizon, reverse=True,
                    ):
                        f_idx_int = int(f_idx)
                        bool_arr = (mask_logits > 0).cpu().numpy()
                        entry = results.setdefault(f_idx_int, {})
                        for i, obj_id in enumerate(obj_ids):
                            oid_int = int(obj_id)
                            sf, _sb = late[oid_int]
                            # Only the object's OWN pre-seed window; its
                            # forward-pass frames are authoritative.
                            if not (sf - backfill_max_frames
                                    <= f_idx_int < sf):
                                continue
                            if bool_arr.ndim == 4:
                                mask2d = bool_arr[i, 0].astype(bool)
                            elif bool_arr.ndim == 3:
                                mask2d = bool_arr[i].astype(bool)
                            else:
                                mask2d = bool_arr.astype(bool)
                            if merge_backfilled_frame(
                                entry, oid_int, mask2d,
                                backfill_overlap_ios,
                            ):
                                backfilled_frames[oid_int] = \
                                    backfilled_frames.get(oid_int, 0) + 1
                    if diagnostics:
                        for oid_int, n in backfilled_frames.items():
                            diagnostics.increment("sam2_objects_backfilled")
                            diagnostics.record(
                                "sam2.object_backfilled",
                                details={"object_id": oid_int,
                                         "seed_frame": late[oid_int][0],
                                         "frames_recovered": n},
                            )
```

Notes for the implementer:
- `_propagate(start, horizon, reverse=True)` already exists and passes
  `start_frame_idx` / `max_frame_num_to_track` / `reverse` through — reuse it
  (it also handles the torch autocast/inference-mode wrapper and stub path).
- `state` is reassigned, so the `finally: predictor.reset_state(state)`
  cleans up whichever state is current. Verify the reassignment is visible
  to the `finally` (it is — same function scope).
- Real SAM2 emits ALL prompted objects on every processed frame; the window
  check (`sf - backfill_max_frames <= f < sf`) is what keeps each object
  inside its own horizon even though the reverse pass is shared.
- The seed frame itself (`f == sf`) ends up with no mask for the late object
  (the forward chunk ended before the seed prompt; the window excludes
  `f >= sf`). A one-frame hole — acceptable; note it in a code comment.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -q -k backfill`
Expected: all backfill tests pass

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/field/ tests/test_detect_bridge.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): backward backfill post-pass for late-seeded objects"
```

---

### Task 5: App wiring (per-job knob)

**Files:**
- Modify: `web/app.py` — three places, all following the existing `reanchor_enabled` pattern exactly:
  1. `InferenceJob.__init__`: `self.backfill_enabled: Optional[bool] = None` next to `self.reanchor_enabled`.
  2. The `sam2_tracker.track(...)` call: add to the kwargs-splat block
     `**({"backfill_enabled": self.backfill_enabled} if self.backfill_enabled is not None else {})`,
     and pass `backfill_max_frames=int(round(SAM2_BACKFILL_SECONDS * eff_fps))` — add module constant `SAM2_BACKFILL_SECONDS = float(os.environ.get("SAM2_BACKFILL_SECONDS", "15.0"))` near `REPROMPT_SECONDS`; `eff_fps` is already computed in that scope for `reprompt_every`.
  3. `/api/infer`: `backfill_enabled: Optional[bool] = Form(None)` + `if backfill_enabled is not None: job.backfill_enabled = bool(backfill_enabled)`.

- [ ] **Step 1: Make the edits**
- [ ] **Step 2: Compile + suite**

Run: `.venv/bin/python -m py_compile web/app.py && .venv/bin/python -m pytest tests/field/ -q`
Expected: compile OK, all pass

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(infer): per-job backfill_enabled knob"
```

---

### Task 6: A/B acceptance on qm72 + qm24

No code. Use the established harness (see `scripts/run_infer_job.py` docstring and the `sam2-export-scale-ab` memory):

- [ ] Start server fresh (`./web/start.sh`), `POST /api/event/load {"event_key": "2026hop"}`.
- [ ] qm72 control already exists (latest cull-on run); launch arm with `-F "cull_hijack_enabled=true" -F "backfill_enabled=true"` on `web/uploads/f7f78be7-….mp4`, `match_key=2026hop_qm72`, roster `--blue 8005 4174 1014 --red 1625 3035 1986`.
- [ ] Compare with `/tmp/ab_compare.py`-style script (or rebuild from `/api/replays/{id}` + diagnostics.jsonl):
  - `sam2_objects_backfilled` count and `frames_recovered` distribution.
  - Coverage: robot-observations per frame before each late object's seed window vs control.
  - **Guard metric:** fused-track count / identity churn must NOT rise (duplicates would show as extra fused ids), confirmed teams not worse, scores unchanged.
  - Wall-clock: expect +1 `init_state` per view (~15-20 s each) + reverse propagation (~horizon × per-frame cost).
- [ ] Repeat on qm24 (`web/uploads/d6ce1999-….mp4`, roster `--blue 8513 573 4174 --red 401 6328 3506`).
- [ ] Record verdict in auto-memory (`sam2-backfill-ab`), flip `SAM2_BACKFILL_ENABLED` default only if coverage gain is real and churn/teams are neutral.

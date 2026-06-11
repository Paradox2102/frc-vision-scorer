# SAM2 Decoupled Detector Ticks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the YOLO detector cascade more often than SAM2 re-seeds, so corroboration/cull/eviction evidence accumulates 2-4× faster and a candidate new robot must persist across consecutive detector ticks before it may seed — without re-introducing the seeding churn that killed the `REPROMPT_SECONDS=1.0` A/B.

**Architecture:** Today detection, corroboration, and seeding all happen once per chunk boundary (every `REPROMPT_SECONDS=2.0s`). The 1.0s-cadence A/B failed because halving the boundary interval bundled two effects: faster evidence AND doubled seeding opportunities (more churn, +25% cost from extra SAM2 chunk restarts). This plan unbundles them. Chunk boundaries (and therefore SAM2 restarts and seeding) stay at 2.0s. At each boundary, the detector additionally runs on K-1 evenly spaced *intermediate* frames of the chunk that just finished — retroactively, against the per-frame masks SAM2 already produced (`results[f]` exists for every tracked frame). Each tick appends one corroboration observation per object. Seeding gains a persistence gate: a detection at the boundary may only seed a NEW object if a tick detection matched it (same position, same color) at the previous tick too.

**Why retroactive ticks are sound:** corroboration asks "did the detector see a robot where this mask was?" — `results[f][oid]["mask"]` is the mask exactly at frame f, and `detect_fn(f, path)` decodes the source frame at f. No SAM2 state is touched; this is pure extra YOLO passes (cheap next to propagation — measured: one full-res YOLO pass per 2s of footage is noise in the profile).

**Evidence-window semantics (the subtle part):** `cull_evidence_window` / `cull_uncorroborated_reprompts` are counted in *observations*. With K ticks per boundary, K observations land per boundary, so fixed window sizes would shrink the cull/evict time horizon by K× — re-creating the cadence-coupling bug this plan exists to avoid. Window and threshold therefore scale by K at `track()` entry: `window = cull_evidence_window * ticks`, `k_absent = cull_uncorroborated_reprompts * ticks`. Time horizons stay constant by construction.

**Tech Stack:** Python, numpy, stub predictors (no GPU). Default behavior unchanged: `SAM2_DETECT_TICKS_PER_BOUNDARY=1`, `SAM2_SEED_PERSISTENCE_TICKS=1` reproduce today's exact behavior.

**Key existing code to read first:**
- `web/field/sam2_tracker.py` `track()`: the boundary block — detector invocation (tri-state contract: `None`/raise = unavailable, `[]` = ran-and-empty), corroboration bookkeeping loop (`mask_corroborated`, `corroboration_history`, `field_pos_window`), seeding loop, re-anchor block (v2 guards), `_record_chunk`.
- `tests/field/test_sam2_tracker.py`: `CullStubPredictor` (per-object motion control), `_ident_diag`, `_culled_ids`.
- The `reprompt-cadence-ab` memory: 1.0s cadence REJECTED — this plan must not recreate it.

**Important context for the engineer:**
- `detect_fn(local_idx, frame_path)` in app.py seeks the source video by frame index and keeps a cached `VideoCapture`; ascending tick order within a boundary keeps reads near-sequential. The tracker passes `frame_paths[idx]` when available; ticks must pass the tick frame's own path (`frame_paths[tick_idx]`), not the boundary's.
- Detector colors ride along as metadata: `detected` items are `(box, color)`; `_normalize_detections` splits them.
- TDD per @superpowers:test-driven-development; run tests with `.venv/bin/python -m pytest` (base conda python has no pytest).

---

## File Structure

- Modify: `web/field/sam2_tracker.py`
  - Module knobs `DETECT_TICKS_PER_BOUNDARY`, `SEED_PERSISTENCE_TICKS`; pure helpers `tick_frames(...)` and `detection_persisted(...)`; `track()` params + boundary-block changes.
- Modify: `web/field/identity_diagnostics.py` — register `NOT_PERSISTENT = "not_persistent"` in `DiagnosticReasons`. **This is load-bearing:** `record()` validates `reason` against the closed `ALL_REASONS` registry and the resulting ValueError is swallowed by the blanket except — an unregistered reason means the event is *silently dropped* (this trap already bit `sam2.object_reanchored` for event names).
- Modify: `web/app.py` — per-job Form params `detect_ticks`, `seed_persistence` (same splat pattern as `reanchor_enabled`).
- Test: `tests/field/test_sam2_tracker.py`.

No new files: the helpers are pure and belong with `mask_corroborated` in the tracker's pure tier; the test conventions live in the existing test module.

---

### Task 1: Pure helper — tick frame selection

**Files:**
- Modify: `web/field/sam2_tracker.py` (pure-helpers tier)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Failing tests**

```python
def test_tick_frames_single_tick_is_boundary_only():
    from web.field.sam2_tracker import tick_frames
    assert tick_frames(chunk_start=3, boundary=5, ticks=1) == [5]


def test_tick_frames_two_ticks_adds_midpoint():
    from web.field.sam2_tracker import tick_frames
    # chunk frames 3,4,5,6,7,8 with boundary 8 -> midpoint ~5, then 8
    assert tick_frames(chunk_start=3, boundary=8, ticks=2) == [5, 8]


def test_tick_frames_clamps_to_available_frames():
    from web.field.sam2_tracker import tick_frames
    # 2-frame chunk can't host 4 distinct ticks; dedupe, keep order, end at boundary
    out = tick_frames(chunk_start=4, boundary=5, ticks=4)
    assert out[-1] == 5 and out == sorted(set(out))
```

- [ ] **Step 2: Run to verify fail** (`-k tick_frames`, ImportError expected)

- [ ] **Step 3: Implement**

```python
def tick_frames(chunk_start: int, boundary: int, ticks: int) -> list[int]:
    """Frame indices for detector ticks across a finished chunk.

    Evenly spaced across (chunk_start, boundary], always ending at the
    boundary (whose detection is the only one allowed to seed). ticks=1
    reproduces the current behavior exactly: [boundary].
    """
    ticks = max(1, int(ticks))
    span = boundary - chunk_start + 1
    out = []
    for i in range(1, ticks + 1):
        f = chunk_start + round(i * span / ticks) - 1
        out.append(min(boundary, max(chunk_start, int(f))))
    return sorted(set(out))
```

- [ ] **Step 4: Run to verify pass; commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): tick_frames helper for sub-boundary detector cadence"
```

---

### Task 2: Pure helper — seed persistence gate

**Files:**
- Modify: `web/field/sam2_tracker.py` (pure-helpers tier)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Failing tests**

```python
def test_detection_persisted_match_by_overlap_and_color():
    from web.field.sam2_tracker import detection_persisted
    prev = [((10, 2, 14, 6), "red"), ((40, 2, 44, 6), "blue")]
    assert detection_persisted((11, 2, 15, 6), "red", prev, (200, 20)) is True


def test_detection_persisted_rejects_color_conflict():
    from web.field.sam2_tracker import detection_persisted
    prev = [((10, 2, 14, 6), "blue")]
    assert detection_persisted((11, 2, 15, 6), "red", prev, (200, 20)) is False


def test_detection_persisted_rejects_disjoint_position():
    from web.field.sam2_tracker import detection_persisted
    prev = [((100, 2, 104, 6), "red")]
    assert detection_persisted((10, 2, 14, 6), "red", prev, (200, 20)) is False


def test_detection_persisted_unknown_color_matches_any():
    from web.field.sam2_tracker import detection_persisted
    prev = [((10, 2, 14, 6), None)]
    assert detection_persisted((11, 2, 15, 6), "red", prev, (200, 20)) is True
```

- [ ] **Step 2: Run to verify fail**

- [ ] **Step 3: Implement** (reuse `_ios` by rasterising both boxes — same approach as `mask_corroborated`)

```python
def detection_persisted(box, color, prev_detections, wh,
                        ios_thresh: float = 0.3) -> bool:
    """True if ``box`` (a boundary seeding candidate) had a same-color
    (or colorless) detection overlapping it at the previous tick.

    The flicker-FP killer: a detection that exists at one instant and not
    ~1s earlier is noise (red clutter, ball pile); a real robot that just
    appeared persists into the next tick and seeds one boundary later — a
    bounded, deliberate acquisition delay.
    """
    w, h = wh
    bx = np.zeros((h, w), bool)
    x1, y1, x2, y2 = (int(v) for v in box)
    bx[y1:y2, x1:x2] = True
    for pb, pc in prev_detections:
        if (color in ("red", "blue") and pc in ("red", "blue")
                and pc != color):
            continue
        pm = np.zeros((h, w), bool)
        px1, py1, px2, py2 = (int(v) for v in pb)
        pm[py1:py2, px1:px2] = True
        if _ios(bx, pm) >= ios_thresh:
            return True
    return False
```

- [ ] **Step 4: Run to verify pass; commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): detection_persisted gate for flicker-FP seeds"
```

---

### Task 3: Wire ticks into the boundary block

**Files:**
- Modify: `web/field/sam2_tracker.py` (`track()`)
- Test: `tests/field/test_sam2_tracker.py`

- [ ] **Step 1: Failing tests**

```python
def test_detector_ticks_run_on_intermediate_frames():
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)],
        reprompt_every=4, detect_fn=detect_fn,
        detect_ticks_per_boundary=2,
    )
    # First chunk covers frames 0..4 (boundary 4). With ticks=2,
    # tick_frames(0, 4, 2) == [1, 4] (round() banker's-rounds 2.5 -> 2,
    # giving frame 1) — the exact midpoint doesn't matter, only that an
    # intermediate tick exists plus the boundary tick.
    assert 4 in calls
    assert any(c in (1, 2, 3) for c in calls), \
        "expected an intermediate tick inside the first chunk"


def test_tick_corroboration_accelerates_eviction_evidence():
    """With 2 ticks/boundary and windows scaled by ticks, an uncorroborated
    stationary object reaches the eviction bar in the same WALL-CLOCK time
    but with 2x the observations — and a corroborated object never does."""
    from web.field.sam2_tracker import Sam2FieldTracker, bottom_center
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()

    def detect_fn(local_idx, frame_path):
        # obj 1 corroborated at every tick; obj 2 never; new robot at 15.
        return [(stub._rect_for(1), "blue"), (stub._rect_for(15), "red")]

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2, detect_fn=detect_fn,
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20,
        cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,
        detect_ticks_per_boundary=2,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert 2 in _culled_ids(diag), "uncorroborated obj still evicted"
    assert 1 not in _culled_ids(diag), "corroborated obj must survive"


def test_seed_persistence_blocks_flicker_fp():
    """A detection present at the boundary tick but absent at the previous
    tick must NOT seed (persistence=2); one present at both ticks must."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    diag = _ident_diag()
    flicker = stub._rect_for(15)
    steady = stub._rect_for(12)

    def detect_fn(local_idx, frame_path):
        out = [(stub._rect_for(1), "blue"), (steady, "red")]
        # flicker only on exact boundary frames of reprompt_every=4 chunks
        if local_idx % 4 == 0:
            out.append((flicker, "red"))
        return out

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)],
        reprompt_every=4, detect_fn=detect_fn,
        object_metadata={1: "blue"},
        detect_ticks_per_boundary=2,
        seed_persistence_ticks=2,
        diagnostics=diag,
    )
    created_boxes = [tuple(int(v) for v in e["details"]["box"])
                     for e in diag._events
                     if e["event"] == "sam2.object_created"]
    assert steady in created_boxes, "persistent detection must seed"
    assert flicker not in created_boxes, "flicker FP must not seed"
    rejected = [e for e in diag._events
                if e["event"] == "sam2.reprompt_rejected"
                and e.get("reason") == "not_persistent"]
    assert rejected, "expected a not_persistent rejection diagnostic"


def test_default_single_tick_behavior_unchanged():
    """ticks=1, persistence=1 must reproduce the legacy call pattern:
    detect_fn called exactly once per boundary, at the boundary frame."""
    from web.field.sam2_tracker import Sam2FieldTracker
    stub = CullStubPredictor()
    tracker = Sam2FieldTracker(predictor=stub)
    calls = []

    def detect_fn(local_idx, frame_path):
        calls.append(local_idx)
        return []

    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1)],
        reprompt_every=4, detect_fn=detect_fn,
    )
    # CullStubPredictor has NUM_FRAMES=12 and _record_chunk(start, 4) yields
    # frames start..start+4 (5 frames/chunk), so boundaries land at exactly
    # [4, 9, 11]. With ticks=1 the detector runs once per boundary, at the
    # boundary frame, and nowhere else.
    assert calls == [4, 9, 11]
```

- [ ] **Step 2: Run to verify failures** (`-k "ticks or persistence or single_tick"`)

- [ ] **Step 3: Implement.** Changes inside `track()` (plus one registry line):

0. In `web/field/identity_diagnostics.py`, add to `DiagnosticReasons`:
   `NOT_PERSISTENT = "not_persistent"`. Without this, `record()` raises
   ValueError on the unknown reason, the blanket except swallows it, and
   the rejection diagnostic is silently dropped — the persistence test
   would fail with no error output anywhere.

1. New params + module knobs:

```python
DETECT_TICKS_PER_BOUNDARY = int(os.environ.get("SAM2_DETECT_TICKS_PER_BOUNDARY", "1"))
SEED_PERSISTENCE_TICKS = int(os.environ.get("SAM2_SEED_PERSISTENCE_TICKS", "1"))
```

```python
detect_ticks_per_boundary: int = DETECT_TICKS_PER_BOUNDARY,
seed_persistence_ticks: int = SEED_PERSISTENCE_TICKS,
```

2. At `track()` entry, scale the evidence windows so time horizons are
   cadence-invariant (document this in the docstring — it is the lesson of
   the failed 1.0s A/B):

```python
ticks = max(1, int(detect_ticks_per_boundary))
cull_evidence_window = cull_evidence_window * ticks
cull_uncorroborated_reprompts = cull_uncorroborated_reprompts * ticks
```

3. In the boundary block, replace the single detector invocation with a loop
   over `tick_frames(last_end_prev + 1, last_idx, ticks)` — note
   `last_end_prev` must be captured BEFORE `last_end = last_idx` (today the
   old value is overwritten; keep a `chunk_start = last_end + 1` local
   computed before the update). For each tick frame `tf`:
   - call `detect_fn(tf, frame_paths[tf] if tf < len(frame_paths) else None)`
     under the existing try/except tri-state contract (an erroring tick
     contributes no evidence; the error print/diagnostic stays once-per-track),
   - corroborate each active object against **`results.get(tf, {})`-masks**
     (the mask at that tick's own frame — NOT the boundary mask), appending
     to `corroboration_history` / `field_pos_window` exactly as the current
     per-boundary loop does (factor the current loop body into a local
     closure `def _corroborate_tick(tick_boxes, tick_meta, tick_frame): ...`
     rather than duplicating it). Inside the closure, `field_pos_window`
     receives the object's box AT THE TICK FRAME
     (`results[tick_frame][oid]["box"]` when present, else fall back to
     `results[last_visible_frame[oid]][oid]["box"]`) — the current code is
     boundary-relative and must be generalized. The SAME fallback applies to
     the corroboration MASK: when the object has no mask at the tick frame
     (active but occluded there), corroborate against its last-visible mask
     — this exactly mirrors today's `existing[oid]` semantics, preserving
     the "ticks=1 reproduces current behavior" guarantee (without the
     fallback, an occluded-at-boundary object would record extra absences
     in UNSCALED windows at ticks=1),
   - keep the LAST tick's (boundary frame's) `new_boxes`/`new_metadata` for
     seeding and re-anchoring, and the second-to-last tick's detections as
     `prev_tick_detections` for the persistence gate — store the
     NORMALIZED form `list(zip(boxes, metadata))` (the output of
     `_normalize_detections`), since `detection_persisted` unpacks
     `(box, color)` pairs and raw `detect_fn` output may be bare boxes.
     With `ticks == 1`, `prev_tick_detections` is the PREVIOUS boundary's
     detections (kept across loop iterations) so `seed_persistence_ticks=2`
     still works at legacy cadence.
   - Color backfill votes: count once per boundary (boundary tick only), not
     per tick — `COLOR_BACKFILL_VOTES` is calibrated in boundaries.

4. In the seeding loop, after the re-anchor block and before the zone gate:

```python
if seed_persistence_ticks > 1 and not detection_persisted(
        int_boxes[j],
        new_metadata[j] if j < len(new_metadata) else None,
        prev_tick_detections, (w, h)):
    if diagnostics:
        diagnostics.record(
            "sam2.reprompt_rejected", reason="not_persistent",
            details={"frame_idx": last_idx, "box": list(new_boxes[j])})
    continue
```

   (`reprompt_rejected` is an already-registered event NAME; the new
   `reason` string is NOT free-form — it must be registered per item 0
   above, or the recorder silently drops the event.)

5. Cull/eviction decision points themselves stay boundary-only (they already
   live in the boundary block); only their *evidence* arrives per-tick.

6. Edge semantics (one clause each, to prevent drift):
   - `detector_available` for the seeding/eviction path reflects the
     BOUNDARY tick's outcome specifically — do not OR availability across
     ticks (a mid-chunk tick succeeding does not make boundary seeding
     data available).
   - Persistence gate fails OPEN when the previous tick's detector was
     unavailable (raised/returned None): unavailability is not negative
     evidence under the tri-state contract. Fall back to the most recent
     AVAILABLE tick's detections; if none exists yet, let the seed through.
   - `sam2.reprompt_candidate` diagnostics stay BOUNDARY-ONLY (matching the
     color-backfill-votes decision) so ticks>1 doesn't inflate event volume
     toward the recorder's max_events cap.

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/field/ tests/test_detect_bridge.py -q`
Expected: all pass — pay special attention to the existing cull/eviction
tests, which must be untouched by the refactor (they run with ticks=1).

- [ ] **Step 5: Commit**

```bash
git add web/field/sam2_tracker.py tests/field/test_sam2_tracker.py
git commit -m "feat(sam2): sub-boundary detector ticks + seed persistence gate"
```

---

### Task 4: App wiring (per-job knobs)

**Files:**
- Modify: `web/app.py` — the established three-touch pattern:
  1. `InferenceJob.__init__`: `self.detect_ticks: Optional[int] = None`, `self.seed_persistence: Optional[int] = None`.
  2. `track()` call site: extend the kwargs-splat block with `detect_ticks_per_boundary` / `seed_persistence_ticks` when the job attrs are not None.
  3. `/api/infer`: `detect_ticks: Optional[int] = Form(None)` (clamp 1–8), `seed_persistence: Optional[int] = Form(None)` (clamp 1–3).

- [ ] **Step 1: Make the edits**
- [ ] **Step 2: `py_compile` + full suite**
- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(infer): per-job detect_ticks / seed_persistence knobs"
```

---

### Task 5: A/B acceptance

No code. Three arms on qm72 AND qm24 (control runs already exist for both — see memories `trench-object-false-positive` and `sam2-reanchor-v1-failed` for job ids and the runner invocation):

- [ ] Arm 1 (evidence-only): `-F "detect_ticks=2"` — expect: identical seeds, faster cull/evict evidence, `not_persistent` count = 0, wall-clock +K YOLO passes per boundary per view (measure!).
- [ ] Arm 2 (full): `-F "detect_ticks=2" -F "seed_persistence=2"` — expect: false seeds down (qm72 trench re-seed blips and qm24 clutter seeds gated), `object_created` count down, confirmed teams not worse, acquisition latency for REAL late robots +≤1 boundary (verify a known late robot — qm24's 3506 at the 2:06 crossing — still acquires).
- [ ] Compare: `object_created`, `reprompt_rejected(not_persistent)`, `sam2_objects_culled`, eviction count, confirmed teams, scores, wall-clock.
- [ ] Record verdict in auto-memory; only flip env defaults (`SAM2_DETECT_TICKS_PER_BOUNDARY=2`, `SAM2_SEED_PERSISTENCE_TICKS=2`) if Arm 2 is neutral-or-better on confirmed teams on BOTH matches and the known-late-robot check passes.

**Risk register for the reviewer/implementer:**
- The detect_fn `VideoCapture` cache assumes near-sequential seeks; tick frames ascend within a boundary, so reads stay sequential. Verify no pathological seek pattern in the timing logs (`[SAM2] detect_fn` errors / frame_export vs analysis timings).
- `results.get(tf)` can be missing an object that was occluded at the tick frame but visible at the boundary — that tick records an absence for it. This is correct behavior (the detector genuinely couldn't corroborate it there) but means windows must be generous; the K-scaling above preserves today's tolerance exactly.
- Persistence gating delays GENUINE new-robot acquisition by up to one boundary when the robot appears in the last sub-tick interval. Bounded and deliberate; the qm24 3506 check guards it.

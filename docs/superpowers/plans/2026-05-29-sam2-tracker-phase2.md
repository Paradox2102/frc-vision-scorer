# Phase 2 — SP-T.2: Re-point SP3 fusion to SAM2 tracklets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed the `FieldTracker` (Phase 1's sticky-id SlotMapper) fused, cross-camera SAM2 *tracklets* — each physical robot gets ONE stable fused `track_id` derived from its per-camera SAM2 object ids — without rebuilding SP3's fusion-evidence model.

**Architecture:** SAM2 emits per-camera object ids that collide across cameras (`main:1` and `corner_left:1` are different robots). SP3's `fuse_detections` already clusters per-camera field detections into one physical robot but **discards** the per-camera `track_id`. Phase 2 makes this additive-and-stateless-plus-one-registry: (1) `FusedObservation` additively carries `members` (the `(camera, track_id)` keys in its cluster) — `evidence` stays the exact 3-tuple existing consumers unpack; (2) a new pure, stateful `FusedTrackRegistry` maps a cluster's member-set → a stable integer fused id across frames (union-find, so a corner-only tracklet keeps its id when `main` later joins the cluster); (3) the fused id is stamped onto the merged `FieldDetection` and fed to `FieldTracker`, whose Phase-1 sticky binding turns it into a fixed slot. **Offline-first:** SAM2 is batch/offline, so — exactly like Phase 1 — this phase does NOT touch `app.py`'s live streaming loop. The deliverable is the pure, unit-tested fusion+registry core plus an offline multi-camera driver that validates the full chain on real footage (skip-clean when multi-camera calibration is absent).

**Why app.py is untouched:** `app.py`'s `process_video` runs SP3 fusion over *live per-frame YOLO* detections (anonymous, `track_id=None`) and behaves exactly as today. SAM2 is offline. Wiring SAM2 tracklets into the live loop is the later live milestone. The additive `FusedObservation` fields (defaults preserve the 3-tuple `evidence` contract) keep `app.py`'s existing fusion path byte-for-byte unchanged.

**Tech Stack:** Python 3.12 / `.venv`, NumPy, `web.field.fusion` (SP3), `web.field.tracker.FieldTracker` (Phase 1), `web.field.sam2_tracker.Sam2FieldTracker` (Phase 1), `web.field.camera_view.load_views` (SP3), pytest. RTX 3090 (driver only), offline/batch.

**Spec:** `docs/superpowers/specs/2026-05-29-tracking-identity-upgrade-design.md` (SP-T.2, lines 214–230). **Gate SP-T.0 PASSED; Phase 1 (SP-T.1) committed.**

**Phase 1 facts carried in:**
- `FieldDetection(pos, color, box, score=1.0, track_id=None)` is a **mutable** dataclass (`tracker.py:18`). A stable `track_id` binds to a slot on first sight and keeps it (sticky), via `FieldTracker._id_to_slot` (`tracker.py:149-202`).
- `Sam2FieldTracker.track(frames_dir, prompt_idx, prompt_boxes) -> {frame_idx: {obj_id: {"box": (x1,y1,x2,y2), "mask": ndarray}}}` — per-camera obj_ids are `1,2,3,…` and are **stable within that camera** across frames (`sam2_tracker.py:186-266`).
- `fuse_detections(per_camera, weights, radius_m=0.6) -> list[FusedObservation]`; `per_camera = {cam: [(FieldDetection, crop), ...]}`; `FusedObservation(detection, evidence)` where `evidence = [(camera, crop, weight), ...]` (`fusion.py`). Clustering is greedy single-link by field distance; one camera never merges with itself.
- `load_views(event_key) -> [CameraView]` returns only calibrated cameras, **main first** (`camera_view.py:27`). Corner cams are calibrated for event `2026hop` only (not `test`/qual1).
- Coord-space gotcha (from Phase 1 driver `scripts/sam2_field_track.py:36,187-200`): SAM2 masks are on 0.5×-scaled export frames; rescale box bottom-centers `÷0.5` (×2) back to full-res crop pixels BEFORE `cal.image_to_field(...)`.

---

## File Structure

- **Modify** `web/field/fusion.py` — additively add `members` and `track_id` fields to `FusedObservation`; `_merge` populates `members` from each cluster member's `(camera, det.track_id)`. `evidence` 3-tuple and the merge math are unchanged. One responsibility unchanged: "many cameras' detections → merged detection + pooled evidence", now also reporting which per-camera tracklets formed the cluster.
- **Create** `web/field/fused_track_id.py` — `FusedTrackRegistry`: pure, stateful, no torch/SAM2/cv2. Maps a cluster's member-set → a stable integer fused id across frames (union-find). One responsibility: "give each physical robot one stable id that survives camera comings-and-goings."
- **Create** `tests/field/test_fused_track_id.py` — unit tests for the registry (stability, corner-only→both merge, new-id allocation, id-bridging).
- **Modify** `tests/field/test_fusion.py` — add cases for `members` population (additive; existing cases stay green).
- **Create** `scripts/sam2_multicam_track.py` — offline driver: per-view `Sam2FieldTracker.track()` → per-camera field detections WITH per-camera track_ids → `fuse_detections` → `FusedTrackRegistry` → `FieldTracker.update`. Asserts fixed-6 invariants; reports continuity. **Skips cleanly** when fewer than 1 view is calibrated or footage is missing. Mirrors `scripts/sam2_field_track.py`.

---

## Task 2.1: `FusedObservation` carries cluster membership (TDD, additive)

Make fusion report which per-camera tracklets formed each cluster, without changing the existing `evidence` contract or merge math.

**Files:**
- Modify: `web/field/fusion.py`
- Test: `tests/field/test_fusion.py`

- [ ] **Step 1: Write failing tests** (append to `tests/field/test_fusion.py`)

```python
def test_members_single_camera():
    # per-camera FieldDetection carries its SAM2 obj id as track_id
    d = FieldDetection(pos=(1.0, 2.0), color="blue", box=(0, 0, 10, 10), track_id=1)
    per_cam = {"main": [(d, b"crop")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0}, radius_m=0.6)
    assert fused[0].members == [("main", 1)]
    assert fused[0].track_id is None  # registry assigns it later; default is None


def test_members_two_cameras_same_robot():
    dm = FieldDetection(pos=(5.0, 3.0), color="blue", box=(0, 0, 10, 10), track_id=2)
    dc = FieldDetection(pos=(5.2, 3.1), color="blue", box=(0, 0, 10, 10), track_id=1)
    per_cam = {"main": [(dm, b"m")], "corner_left": [(dc, b"c")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert len(fused) == 1
    assert set(fused[0].members) == {("main", 2), ("corner_left", 1)}
```

- [ ] **Step 2: Run → fail**

Run: `.venv/bin/python -m pytest tests/field/test_fusion.py -v`
Expected: FAIL — `AttributeError: 'FusedObservation' object has no attribute 'members'`.

- [ ] **Step 3: Implement** — make the additive change in `web/field/fusion.py`:

In the `FusedObservation` dataclass, add the two fields with defaults (defaults keep every existing constructor call and the `evidence` 3-tuple contract intact):

```python
from dataclasses import dataclass, field

@dataclass
class FusedObservation:
    detection: FieldDetection                 # what the tracker sees
    evidence: list                            # [(source_camera, bumper_crop, weight), ...]
    members: list = field(default_factory=list)   # [(camera, per_camera_track_id), ...]
    track_id: int | None = None               # stable fused id (assigned by FusedTrackRegistry)
```

In `_merge`, after building `evidence`, also build `members` from the cluster and pass it in:

```python
    members = [(cam, det.track_id) for (cam, det, crop, w) in cluster]
    return FusedObservation(
        detection=FieldDetection(pos=(float(avg[0]), float(avg[1])), color=color, box=box),
        evidence=evidence,
        members=members,
    )
```

(Leave `evidence`, the weighted-average position, color vote, and representative box exactly as they are.)

- [ ] **Step 4: Run → pass** (the whole fusion suite, proving the additive change broke nothing)

Run: `.venv/bin/python -m pytest tests/field/test_fusion.py -v`
Expected: PASS — all prior cases plus the two new ones.

- [ ] **Step 5: Commit**

```bash
git add web/field/fusion.py tests/field/test_fusion.py
git commit -m "feat(field): FusedObservation reports cluster member tracklets (additive)"
```

## Task 2.2: `FusedTrackRegistry` — stable fused ids across frames (TDD, pure)

A cluster's member-set identifies one physical robot. Because each camera's SAM2 id is already stable within that camera, a cluster keeps its identity as long as it shares **any** member key with a previously-seen cluster — even when members join (corner-only robot later seen by main) or drop out (main occluded). Union-find over member keys gives that.

**Files:**
- Create: `web/field/fused_track_id.py`
- Test: `tests/field/test_fused_track_id.py`

- [ ] **Step 1: Write failing tests** (`tests/field/test_fused_track_id.py`)

```python
from web.field.fused_track_id import FusedTrackRegistry
from web.field.fusion import FusedObservation
from web.field.tracker import FieldDetection


def _obs(members, pos=(0.0, 0.0)):
    det = FieldDetection(pos=pos, color="unknown", box=(0, 0, 1, 1))
    return FusedObservation(detection=det, evidence=[], members=list(members))


def test_distinct_clusters_get_distinct_ids():
    reg = FusedTrackRegistry()
    a = _obs([("main", 1)])
    b = _obs([("main", 2)])
    reg.assign([a, b])
    assert a.track_id != b.track_id
    assert {a.track_id, b.track_id} == {0, 1}


def test_same_members_keep_id_across_frames():
    reg = FusedTrackRegistry()
    f1 = _obs([("main", 1)])
    reg.assign([f1])
    first = f1.track_id
    f2 = _obs([("main", 1)])      # next frame, same tracklet
    reg.assign([f2])
    assert f2.track_id == first


def test_corner_only_then_joined_by_main_keeps_id():
    # The key SP-T.2 case: a robot first seen ONLY by a corner camera, then later
    # also by main, must keep the SAME fused id (so its slot is stable).
    reg = FusedTrackRegistry()
    f1 = _obs([("corner_left", 1)])
    reg.assign([f1])
    cid = f1.track_id
    f2 = _obs([("corner_left", 1), ("main", 3)])  # main joins the cluster
    reg.assign([f2])
    assert f2.track_id == cid


def test_id_stamped_onto_detection():
    # The fused id must land on the FieldDetection the tracker will consume.
    reg = FusedTrackRegistry()
    obs = _obs([("main", 1)])
    reg.assign([obs])
    assert obs.detection.track_id == obs.track_id


def test_new_member_set_gets_fresh_id():
    reg = FusedTrackRegistry()
    reg.assign([_obs([("main", 1)])])         # id 0
    reg.assign([_obs([("main", 1)])])         # still 0
    fresh = _obs([("corner_right", 9)])       # never seen
    reg.assign([_obs([("main", 1)]), fresh])
    assert fresh.track_id == 1


def test_bridging_two_prior_ids_collapses_to_one():
    # Two tracklets that were separate clusters become one cluster (e.g. SAM2 had
    # split a robot across cameras, now they're within radius). They must collapse
    # to a single fused id, not keep two.
    reg = FusedTrackRegistry()
    reg.assign([_obs([("main", 1)]), _obs([("corner_left", 1)])])  # ids 0 and 1
    merged = _obs([("main", 1), ("corner_left", 1)])
    reg.assign([merged])
    assert merged.track_id in (0, 1)
```

- [ ] **Step 2: Run → fail**

Run: `.venv/bin/python -m pytest tests/field/test_fused_track_id.py -v`
Expected: FAIL — `ModuleNotFoundError: web.field.fused_track_id`.

- [ ] **Step 3: Implement** `web/field/fused_track_id.py`:

```python
"""Stable cross-camera fused track ids for the SAM2 tracklet path.

SAM2 emits per-camera object ids that collide across cameras and restart per
prompt. SP3 fusion clusters per-camera detections of one physical robot; this
registry gives each such cluster ONE integer id that is stable across frames,
so the Phase-1 FieldTracker sticky binding turns it into a fixed slot.

A cluster is identified by its set of member keys ``(camera, per_camera_id)``.
Because each camera's id is stable within that camera, two clusters that share
ANY member key are the same physical robot — even across a member joining (a
corner-only robot later seen by main) or dropping out (main occluded). That is
exactly union-find over member keys: each member key remembers its fused id, and
a cluster that touches several known ids collapses them to one.

Pure: no torch / SAM2 / cv2 imports."""
from __future__ import annotations


class FusedTrackRegistry:
    def __init__(self) -> None:
        # member key (camera, per_camera_id) -> fused id
        self._key_to_id: dict[tuple, int] = {}
        self._next_id = 0

    def assign(self, observations: list) -> None:
        """Stamp a stable ``track_id`` onto each FusedObservation (and its
        ``detection``), mutating in place. Call once per frame, in order."""
        for obs in observations:
            keys = [tuple(m) for m in obs.members]
            # Fused ids already known for any member key in this cluster.
            known = {self._key_to_id[k] for k in keys if k in self._key_to_id}
            if known:
                fused_id = min(known)
                # Collapse any bridged ids: every key currently pointing at one of
                # the `known` ids is repointed to the winner.
                if len(known) > 1:
                    losers = known - {fused_id}
                    for k, v in list(self._key_to_id.items()):
                        if v in losers:
                            self._key_to_id[k] = fused_id
            else:
                fused_id = self._next_id
                self._next_id += 1
            # Register every member key of this cluster to the chosen id.
            for k in keys:
                self._key_to_id[k] = fused_id
            obs.track_id = fused_id
            obs.detection.track_id = fused_id
```

- [ ] **Step 4: Run → pass**

Run: `.venv/bin/python -m pytest tests/field/test_fused_track_id.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add web/field/fused_track_id.py tests/field/test_fused_track_id.py
git commit -m "feat(field): FusedTrackRegistry — stable cross-camera fused track ids"
```

## Task 2.3: Fused tracklets → SlotMapper end-to-end (TDD, pure)

Prove the spec's two SP-T.2 acceptance behaviors without a GPU: **two views of one robot resolve to one slot**, and **a corner-only (main-occluded) tracklet keeps its slot**. This is the integration of Task 2.1 + 2.2 + Phase-1 `FieldTracker`, driven by hand-built per-camera tracklets.

**Files:**
- Test: `tests/field/test_tracklet_fusion_to_slot.py`

- [ ] **Step 1: Write failing tests** (`tests/field/test_tracklet_fusion_to_slot.py`)

```python
"""SP-T.2 end-to-end (pure): per-camera SAM2 tracklets -> fuse -> stable fused id
-> FieldTracker sticky slot. No GPU; tracklets are hand-built field detections."""
from web.field.fusion import fuse_detections
from web.field.fused_track_id import FusedTrackRegistry
from web.field.tracker import FieldDetection, FieldTracker

FIELD_LEN = 16.0
WEIGHTS = {"main": 1.0, "corner_left": 2.5, "corner_right": 2.5}


def _det(x, y, track_id, color="blue", box=(0, 0, 10, 10)):
    return FieldDetection(pos=(x, y), color=color, box=box, track_id=track_id)


def _step(tracker, reg, per_cam, dt=0.1):
    fused = fuse_detections(per_cam, weights=WEIGHTS, radius_m=0.6)
    reg.assign(fused)                       # stamps obs.detection.track_id
    dets = [obs.detection for obs in fused]
    return tracker.update(dets, {}, dt)


def test_two_views_one_robot_one_slot():
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    # one blue robot at ~(3,3), seen by main (id 1) and corner_left (id 1)
    per_cam = {"main": [(_det(3.0, 3.0, 1), b"m")],
               "corner_left": [(_det(3.2, 3.1, 1), b"c")]}
    outs = _step(tracker, reg, per_cam)
    assert len(outs) == 1                   # exactly one slot active
    slot = outs[0].slot_id
    # next frame, robot moved a bit; same per-camera ids -> same slot
    per_cam2 = {"main": [(_det(3.3, 3.0, 1), b"m")],
                "corner_left": [(_det(3.5, 3.1, 1), b"c")]}
    outs2 = _step(tracker, reg, per_cam2)
    assert [o.slot_id for o in outs2] == [slot]


def test_corner_only_when_main_occluded_keeps_slot():
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    # frame 1: both cameras see the robot
    outs1 = _step(tracker, reg, {
        "main": [(_det(4.0, 2.0, 1), b"m")],
        "corner_left": [(_det(4.1, 2.0, 1), b"c")],
    })
    slot = outs1[0].slot_id
    # frame 2: main loses it (occluded) — only corner_left sees it; SAME slot
    outs2 = _step(tracker, reg, {
        "main": [],
        "corner_left": [(_det(4.2, 2.0, 1), b"c")],
    })
    assert slot in [o.slot_id for o in outs2]
    # frame 3: main re-acquires (SAM2 id may persist as 1) — still the same slot
    outs3 = _step(tracker, reg, {
        "main": [(_det(4.3, 2.0, 1), b"m")],
        "corner_left": [(_det(4.4, 2.0, 1), b"c")],
    })
    assert slot in [o.slot_id for o in outs3]


def test_two_distinct_robots_two_slots():
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    outs = _step(tracker, reg, {
        "main": [(_det(2.0, 2.0, 1), b"a"), (_det(6.0, 5.0, 2), b"b")],
    })
    assert len({o.slot_id for o in outs}) == 2
```

- [ ] **Step 2: Run → fail** (red first — confirms the test imports the new module and exercises the chain)

Run: `.venv/bin/python -m pytest tests/field/test_tracklet_fusion_to_slot.py -v`
Expected: PASS once Tasks 2.1–2.2 are in. If RED, it points at a real gap (e.g. fused id not stamped onto `detection`) — fix the gap, do not weaken the test.

> Note: this task is pure integration of already-implemented units, so the test may go green immediately. That is acceptable here (the units were each TDD'd). The value is locking the SP-T.2 acceptance behaviors as regression tests. If green on first run, briefly confirm by temporarily breaking `reg.assign` (e.g. not stamping `detection.track_id`) and re-running to see it fail, then revert.

- [ ] **Step 3: (only if RED) fix the gap** in `fused_track_id.py` / `fusion.py` so the chain stamps stable ids correctly. No new production code is expected beyond Tasks 2.1–2.2.

- [ ] **Step 4: Run → pass, plus the full field suite** (regression guardrail — proves Phase 1 + SP3 + app.py path untouched)

Run: `.venv/bin/python -m pytest tests/field/ tests/test_radar.py tests/test_alliance_consistency.py -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/field/test_tracklet_fusion_to_slot.py
git commit -m "test(field): SP-T.2 fused-tracklet-to-slot acceptance (two-view->one slot; corner-only->slot)"
```

## Task 2.4: Offline multi-camera driver + continuity check (validation)

End-to-end on real footage: per-view SAM2 → fused stable tracklets → `FieldTracker`. This is an integration experiment (not TDD), judged by invariants + eyeballed continuity, mirroring Phase 1's `scripts/sam2_field_track.py`. It **skips cleanly** when the event has no calibrated views or footage is missing (corner cams are calibrated for `2026hop` only; `test`/qual1 has main-only — a main-only run still exercises the new fusion+registry path with a single camera).

**Files:**
- Create: `scripts/sam2_multicam_track.py`

- [ ] **Step 1: Scaffold** — args `--event` (default `test`), `--labels` (segment sidecar, default the qual1 one), `--video`. Path setup mirrors `scripts/sam2_field_track.py:19-31`. Load the segment sidecar (`start_frame/end_frame/prompt_frame/stride`). `views = load_views(event)`; if `len(views) == 0` → print + `sys.exit(0)` (skip-clean). For each view, load its `CameraView.calibration`; build `FieldModel` from `views[0].calibration.field_season`.

- [ ] **Step 2: Per-view SAM2 export + track.** For each `view` in `views`:
  - Export the segment frames cropped to that view (`view.extract(frame)`) at `SAM2_SCALE = 0.5` into a per-view `tempfile.TemporaryDirectory()` (numbered JPEGs), exactly like `sam2_field_track.py:107-140`.
  - YOLO-detect robot boxes (`class_id == 1`, `conf=0.4`) on that view's **0.5× prompt-frame crop** to get `prompt_boxes_scaled`.
  - `Sam2FieldTracker().track(frames_dir=tmp, prompt_idx=prompt_local_idx, prompt_boxes=prompt_boxes_scaled)` → `raw[view.name] = {local_idx: {obj_id: {"box","mask"}}}`.
  - Clean up that view's tempdir after its propagation finishes.

- [ ] **Step 3: Per-frame fuse + assign + track.** Build one `FusedTrackRegistry()` and one `FieldTracker(fm.length_m)`. For each local frame index (union across views), for each view:
  - For each `obj_id` tracklet that frame: rescale box `÷SAM2_SCALE` to full-res crop pixels, bottom-center `uv = ((fx1+fx2)/2, fy2)`, project `xy = view.calibration.image_to_field(uv)`; skip if `None`/off-field.
  - Append `(FieldDetection(pos=xy, color="unknown", box=fullres_box, score=1.0, track_id=obj_id), crop=None)` to `per_camera[view.name]`.
  - Then `fused = fuse_detections(per_camera, weights={v.name: (2.5 if v.name!="main" else 1.0) for v in views})`; `reg.assign(fused)`; `outs = tracker.update([o.detection for o in fused], {}, dt=stride/fps)`; record `frames[gf] = {t.slot_id: bottom_center(t.box) for t in outs if t.state=="visible" and t.box}`.

- [ ] **Step 4: Invariants + continuity.** Assert fixed-6 / ≤3-per-alliance exactly like `sam2_field_track.py:274-305` (slots `<6`; blue=slots 0-2, red=3-5). Print `continuity_stats(frames)` (`from web.field.spike_metrics import continuity_stats`) and the per-view SAM2 ms/frame + active camera list. Print how many fused observations had multi-camera `members` (cross-camera merges actually happening) when ≥2 views are active.

- [ ] **Step 5: Run it.**

Run: `.venv/bin/python scripts/sam2_multicam_track.py --event test`
Expected: with `test` (main-only) — runs single-camera through the new fusion+registry path, invariants hold, continuity printed. (For a true multi-cam run, an event with calibrated corners + matching footage is needed; `2026hop` is calibrated but requires its own footage — document if unavailable and rely on Task 2.3's pure two-view test for the cross-camera behavior.)

- [ ] **Step 6: Commit**

```bash
git add scripts/sam2_multicam_track.py
git commit -m "feat(sam2): offline multi-camera driver — per-view SAM2 -> fused stable ids -> SlotMapper"
```

---

## Notes for the executor

- Test env: `.venv/bin/python -m pytest` (base conda python lacks pytest). See [[test-env-uses-venv]].
- Branch `feat/sam2-tracking-identity-upgrade`. Never push to `origin` (ohmeow-ship); team remote is `paradox`. Don't push unless asked.
- **Do NOT modify `web/app.py` in this phase.** The additive `FusedObservation` fields (`members`, `track_id`, both defaulted) keep app.py's live SP3 fusion path unchanged; the regression run in Task 2.3 Step 4 is the guardrail.
- Tasks 2.1–2.3 are pure and GPU-free (the testable core SP-T.2 requires). Task 2.4 is the GPU validation driver and is allowed to skip-clean.
- **Reuse SP3 unchanged** beyond the additive `members`/`track_id` fields: the clustering, weighted-average position, color vote, and `evidence` 3-tuple are not modified — honoring the spec's "reuse the fusion-evidence model unchanged."
- Reference skills: @superpowers:test-driven-development, @superpowers:verification-before-completion.

## Acceptance summary (SP-T.2)

- `fuse_detections` reports cluster membership additively; existing fusion tests + app.py path unaffected.
- `FusedTrackRegistry` assigns one stable integer id per physical robot, stable across frames and across a member joining/leaving the cluster (corner-only → both cameras keeps its id).
- Pure end-to-end test proves: two views of one robot → one slot; corner-only (main-occluded) tracklet → keeps its slot; two distinct robots → two slots.
- Full `tests/field/` + radar + alliance suites green (proves Phase 1 and the live app.py fusion path are untouched).
- Offline multi-camera driver runs the full chain on real footage (or skips cleanly), asserting fixed-6 invariants and reporting continuity.

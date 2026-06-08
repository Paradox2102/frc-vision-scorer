# Backward Track Confirmation — Slice 1 (Whole-Match Evidence Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect per-track whole-match evidence (motion trajectory, robot-confirmation log, person-overlap) during the analysis loop, with zero change to current output — the foundation slices 2–4 consume.

**Architecture:** A new pure module `web/field/track_evidence.py` defines `TrackEvidence` (per-track accumulator) and `MatchEvidence` (the registry, one per job). `InferenceJob` owns a `MatchEvidence` and a per-frame slot→track binding log, updated once per processed frame from data already computed in the loop. No pruning, no classification, no output change — that is slices 2–4.

**Tech stack:** Python 3.12, numpy, pytest (`.venv/bin/python -m pytest`). Reuses `web.field.detect_schema.box_ios` for per-frame person overlap.

**Spec:** `docs/superpowers/specs/2026-06-07-backward-track-confirmation-design.md`

---

## File Structure

- **Create** `web/field/track_evidence.py` — pure data layer. `TrackEvidence` dataclass + `net_displacement_m`/`person_overlap_frac`; `MatchEvidence` registry with `observe`/confirmation setters. No torch, no app imports.
- **Create** `tests/field/test_track_evidence.py` — unit tests for the pure module.
- **Modify** `web/app.py` — `InferenceJob.__init__`: add `self._match_evidence` + `self._binding_log` + `PERSON_OVERLAP_IOS` knob. Analysis loop: per-frame `observe(...)` + binding-log append. Confirmation sites (OCR/VLM/gallery/color): record on evidence.

DRY note: per-frame box/person overlap reuses `box_ios` (do **not** write a new IoU). YAGNI: slice 1 introduces only `PERSON_OVERLAP_IOS`; `MOTION_CONFIRM_M`/`PERSON_OVERLAP_FRAC` are the classifier's (slice 2) — do not add them here.

---

## Task 1: `TrackEvidence` dataclass + robust net displacement

**Files:**
- Create: `web/field/track_evidence.py`
- Test: `tests/field/test_track_evidence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/field/test_track_evidence.py
import math
from web.field.track_evidence import TrackEvidence


def _ev(positions):
    e = TrackEvidence(track_id=1)
    for f, x, y in positions:
        e.add_position(f, x, y)
    return e


def test_net_displacement_zero_for_jittery_static_track():
    # 14 m-range structure: stays at (10.5, 7.6) with +-0.2 m per-frame jitter
    import itertools
    jit = itertools.cycle([0.2, -0.15, 0.1, -0.2, 0.05])
    pos = [(f, 10.5 + next(jit), 7.6 + next(jit)) for f in range(40)]
    assert _ev(pos).net_displacement_m(window=5) < 0.5


def test_net_displacement_catches_real_drive():
    # robot drives ~5 m across the field over its life
    pos = [(f, 2.0 + 0.13 * f, 4.0) for f in range(40)]  # 0..~5 m
    assert _ev(pos).net_displacement_m(window=5) > 3.0


def test_net_displacement_robust_to_single_outlier():
    pos = [(f, 3.0, 3.0) for f in range(20)]
    pos[10] = (10, 30.0, 30.0)  # one wild outlier frame
    assert _ev(pos).net_displacement_m(window=5) < 0.5


def test_net_displacement_too_few_samples_is_zero():
    assert _ev([(0, 1.0, 1.0), (1, 2.0, 2.0)]).net_displacement_m(window=5) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/field/test_track_evidence.py -q`
Expected: FAIL — `ModuleNotFoundError: web.field.track_evidence`.

- [ ] **Step 3: Write minimal implementation**

```python
# web/field/track_evidence.py
"""Per-track whole-match evidence (motion, confirmations, person overlap).

Pure data layer — no torch, no app imports — so it is unit-testable in isolation
and reusable by the classifier (slice 2) and backward pass (slice 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


@dataclass
class TrackEvidence:
    track_id: int
    frames_seen: list = field(default_factory=list)        # processed-frame indices
    field_positions: list = field(default_factory=list)    # (frame, x, y) meters
    person_overlap_frames: int = 0
    ever_team_read: bool = False
    ever_appearance_match: bool = False
    ever_bumper_color: bool = False
    first_confident_frame: "int | None" = None

    def add_position(self, frame: int, x: float, y: float) -> None:
        self.field_positions.append((int(frame), float(x), float(y)))

    def net_displacement_m(self, window: int = 5) -> float:
        """Robust whole-match displacement: median (x, y) over consecutive windows
        of `window` samples, then the max pairwise distance between window medians.
        Per-window median rejects frame jitter and single outliers; max-pairwise
        catches movement anywhere in the match (even out-and-back). 0.0 if fewer
        than two full windows of positioned samples."""
        pts = self.field_positions
        if len(pts) < 2 * window:
            return 0.0
        centers = []
        for i in range(0, len(pts) - window + 1, window):
            chunk = pts[i:i + window]
            centers.append((_median([p[1] for p in chunk]),
                            _median([p[2] for p in chunk])))
        best = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = ((centers[i][0] - centers[j][0]) ** 2 +
                     (centers[i][1] - centers[j][1]) ** 2) ** 0.5
                if d > best:
                    best = d
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/field/test_track_evidence.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add web/field/track_evidence.py tests/field/test_track_evidence.py
git commit -m "feat(evidence): TrackEvidence with robust whole-match net displacement"
```

---

## Task 2: Person-overlap fraction + confirmation recording

**Files:**
- Modify: `web/field/track_evidence.py`
- Test: `tests/field/test_track_evidence.py`

- [ ] **Step 1: Write the failing test**

```python
def test_person_overlap_frac():
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=(f < 4))  # 4 of 10
    assert abs(e.person_overlap_frac() - 0.4) < 1e-9


def test_person_overlap_frac_no_frames_is_zero():
    assert TrackEvidence(track_id=1).person_overlap_frac() == 0.0


def test_first_confident_frame_is_earliest():
    e = TrackEvidence(track_id=1)
    e.observe_frame(5, overlapped_person=False)
    e.confirm(7, team_read=True)
    e.confirm(6, appearance=True)   # earlier confirmation arrives later in code
    assert e.ever_team_read and e.ever_appearance_match
    assert e.first_confident_frame == 6


def test_confirm_bumper_color_is_not_confident_frame():
    # color alone is weak evidence: recorded, but does NOT set first_confident_frame
    e = TrackEvidence(track_id=1)
    e.confirm(3, bumper_color=True)
    assert e.ever_bumper_color is True
    assert e.first_confident_frame is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/field/test_track_evidence.py -k "overlap or confident or bumper_color" -q`
Expected: FAIL — `observe_frame`/`confirm`/`person_overlap_frac` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `TrackEvidence`:

```python
    def observe_frame(self, frame: int, overlapped_person: bool) -> None:
        self.frames_seen.append(int(frame))
        if overlapped_person:
            self.person_overlap_frames += 1

    def person_overlap_frac(self) -> float:
        n = len(self.frames_seen)
        return (self.person_overlap_frames / n) if n else 0.0

    def _mark_confident(self, frame: int) -> None:
        if self.first_confident_frame is None or frame < self.first_confident_frame:
            self.first_confident_frame = int(frame)

    def confirm(self, frame: int, *, team_read: bool = False,
                appearance: bool = False, bumper_color: bool = False) -> None:
        # team_read / appearance are strong robot evidence -> set first_confident_frame.
        # bumper_color is weak (red/blue field structure fires it) -> recorded only.
        if team_read:
            self.ever_team_read = True
            self._mark_confident(frame)
        if appearance:
            self.ever_appearance_match = True
            self._mark_confident(frame)
        if bumper_color:
            self.ever_bumper_color = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/field/test_track_evidence.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add web/field/track_evidence.py tests/field/test_track_evidence.py
git commit -m "feat(evidence): person-overlap fraction and confirmation recording"
```

---

## Task 3: `MatchEvidence` registry

**Files:**
- Modify: `web/field/track_evidence.py`
- Test: `tests/field/test_track_evidence.py`

- [ ] **Step 1: Write the failing test**

```python
from web.field.track_evidence import MatchEvidence


def test_match_evidence_aggregates_per_track():
    m = MatchEvidence()
    m.observe(track_id=1, frame=0, field_xy=(2.0, 4.0), overlapped_person=False)
    m.observe(track_id=1, frame=1, field_xy=(2.6, 4.0), overlapped_person=False)
    m.observe(track_id=2, frame=0, field_xy=(8.0, 4.0), overlapped_person=True)
    assert set(m.by_track) == {1, 2}
    assert len(m.by_track[1].field_positions) == 2
    assert m.by_track[2].person_overlap_frames == 1


def test_match_evidence_tolerates_missing_position():
    m = MatchEvidence()
    m.observe(track_id=1, frame=0, field_xy=None, overlapped_person=False)
    assert m.by_track[1].frames_seen == [0]
    assert m.by_track[1].field_positions == []  # coasting frame: no motion sample


def test_match_evidence_confirm_routes_to_track():
    m = MatchEvidence()
    m.observe(track_id=3, frame=2, field_xy=(5.0, 5.0), overlapped_person=False)
    m.confirm(track_id=3, frame=2, team_read=True)
    assert m.by_track[3].ever_team_read
    assert m.by_track[3].first_confident_frame == 2


def test_match_evidence_confirm_before_observe_autocreates():
    m = MatchEvidence()
    m.confirm(track_id=9, frame=1, appearance=True)  # confirmation can precede observe
    assert m.by_track[9].ever_appearance_match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/field/test_track_evidence.py -k match_evidence -q`
Expected: FAIL — `MatchEvidence` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
class MatchEvidence:
    """Per-job registry of TrackEvidence keyed by fused track_id."""

    def __init__(self) -> None:
        self.by_track: dict = {}

    def _track(self, track_id: int) -> TrackEvidence:
        te = self.by_track.get(track_id)
        if te is None:
            te = TrackEvidence(track_id=int(track_id))
            self.by_track[track_id] = te
        return te

    def observe(self, track_id, frame, field_xy, *, overlapped_person) -> None:
        te = self._track(track_id)
        te.observe_frame(frame, overlapped_person)
        if field_xy is not None:
            te.add_position(frame, field_xy[0], field_xy[1])

    def confirm(self, track_id, frame, *, team_read=False, appearance=False,
                bumper_color=False) -> None:
        self._track(track_id).confirm(
            frame, team_read=team_read, appearance=appearance,
            bumper_color=bumper_color)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/field/test_track_evidence.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add web/field/track_evidence.py tests/field/test_track_evidence.py
git commit -m "feat(evidence): MatchEvidence registry"
```

---

## Task 4: Wire `MatchEvidence` into the analysis loop (no output change)

**Files:**
- Modify: `web/app.py` — `InferenceJob.__init__` (near line 749, the `self.video_fps`/`self.frame_skip` block); analysis loop after `track_outs = self.field_tracker.update(...)` (line ~4403).
- Test: `tests/field/test_track_evidence_integration.py` (new) + existing suites unchanged.

> **Implementer note:** line numbers drift — locate by the anchors quoted below, not by absolute number. Wrap all evidence calls so they can never raise into the loop (mirror `IdentityDiagnosticRecorder`'s swallow-and-continue posture).

- [ ] **Step 1: Add init state + knob**

In `InferenceJob.__init__`, near `self.video_fps = 0`:

```python
        self._match_evidence = MatchEvidence()
        self._binding_log: list[dict] = []   # [{"frame_num": int, "bindings": {track_id: slot_id}}]
        self._evidence_frame = 0             # current processed frame, set at loop TOP
```

> Why a separate `_evidence_frame`: `self._current_frame` is assigned at the *end* of
> the loop body (`app.py:4787`), so at the read/appearance confirmation sites it still
> holds the *previous* processed frame (and `0` on the first iteration). Recording
> confirmations against it would put `first_confident_frame` off by one frame (or wildly
> wrong on frame 0) — the precise field slice 3 keys on. `_evidence_frame`, set at the
> top of the loop, is the correct current frame at every tap.

Add import at top of `app.py` with the other `web.field` imports:

```python
from web.field.track_evidence import MatchEvidence
```

Add the knob near the other `DET_*` env constants:

```python
PERSON_OVERLAP_IOS = float(os.environ.get("PERSON_OVERLAP_IOS", "0.3"))
```

- [ ] **Step 2: Add a testable per-frame evidence helper, called from the loop**

Add this method to `InferenceJob` (keeps the loop thin and lets Task-4 Step 4 unit-test
the wiring without the full mock stack). It works entirely on **post-fusion** data, so
all evidence is keyed by the fused `track_id` (B1 fix — fusion happens at `app.py:4395`,
*before* this runs). Bumper color is folded in here from `obs.detection.color`, **not**
from the per-camera `obj_id` loop at ~4337.

```python
    def _record_frame_evidence(self, fused, frame_num, main_person_boxes):
        """Slice-1 whole-match evidence. No behavior change; consumed by the
        backward-confirmation classifier (slice 2+). All keys are fused track_ids.
        Position is the RAW fused measurement (det.pos): a coasting track is absent
        from `fused`, so still frames contribute no motion sample — intended, so
        slice 3 sees real movement, not Kalman-predicted drift."""
        for obs in fused:
            tid = obs.track_id
            if tid is None:
                continue
            det = obs.detection
            fxy = det.pos if det is not None else None
            box = det.box if det is not None else None
            color = det.color if det is not None else None
            overlapped = bool(box is not None and main_person_boxes and any(
                box_ios(tuple(box), tuple(pb)) >= PERSON_OVERLAP_IOS
                for pb in main_person_boxes))
            self._match_evidence.observe(tid, frame_num, fxy, overlapped_person=overlapped)
            if color in ("red", "blue"):
                self._match_evidence.confirm(tid, frame_num, bumper_color=True)
        # binding log: field tracker maps fused track_id -> slot_id
        self._binding_log.append({
            "frame_num": frame_num,
            "bindings": dict(self.field_tracker._id_to_slot),  # {track_id: slot_id}
        })
```

Set `self._evidence_frame` at the **top** of the loop body, as soon as `frame_num` is
known (so every confirmation tap downstream reads the correct current frame):

```python
                self._evidence_frame = frame_num
```

Then call the helper just after `track_outs = self.field_tracker.update(field_dets, self._view_flags, dt)`:

```python
                try:
                    self._record_frame_evidence(
                        fused, frame_num, person_boxes_by_cam.get("main", []))
                except Exception:
                    pass
```

Add `box_ios` to the `web.field.detect_schema` import line in `app.py` if not already imported.

- [ ] **Step 3: Tap the read/appearance confirmation sites**

Bumper *color* is already recorded in Step 2. Here add the three **strong** confirmations
(read / appearance), each keyed by the fused `track_id` for the confirmed slot via the
`_id_to_slot` inverse. Helper on `InferenceJob`:

```python
    def _track_id_for_slot(self, slot_id):
        # _id_to_slot is {track_id: slot_id}; invert. Returns None when the slot is
        # not currently bound to a fused track (confirmation then drops — acceptable
        # for slice 1; these sites normally run on frames where the slot IS bound).
        return next((tid for tid, s in self.field_tracker._id_to_slot.items()
                     if s == slot_id), None)
```

At each site, use `self._evidence_frame` as the confirmation frame (the correct current
processed frame — see the `__init__` note), wrapped in `try/except Exception: pass`:

- VLM read (anchor: `app.py:1514`, inside `read_fused_bumper_numbers` called at ~4523):
  `tid = self._track_id_for_slot(slot_id)` → if not None: `self._match_evidence.confirm(tid, self._evidence_frame, team_read=True)`
- Gallery match (anchor: `app.py:1296`): same, `appearance=True`.

> The OCR consensus site at `app.py:948` lives in `ocr_robot_bumpers`, which currently
> has **no caller** (dead path) — do NOT tap it; the live strong-read path is the VLM
> site at 1514. If `ocr_robot_bumpers` is revived later, add the `team_read=True` tap then.

- [ ] **Step 4: Write the integration test (exercises the real wiring)**

```python
# tests/field/test_track_evidence_integration.py
from types import SimpleNamespace
from web.field.track_evidence import MatchEvidence


def _job():
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)        # bypass heavy __init__ deps
    job._match_evidence = MatchEvidence()
    job._binding_log = []
    job.field_tracker = SimpleNamespace(_id_to_slot={1: 0, 2: 3})
    return job


def _obs(track_id, pos, box, color):
    det = SimpleNamespace(pos=pos, box=box, color=color)
    return SimpleNamespace(track_id=track_id, detection=det)


def test_record_frame_evidence_populates_and_keys_by_fused_id():
    job = _job()
    person = [(95, 50, 130, 200)]  # overlaps track 2's box below
    fused = [
        _obs(1, (2.0, 4.0), (10, 10, 40, 120), "red"),     # robot, no person overlap
        _obs(2, (8.0, 4.0), (100, 60, 140, 190), None),    # overlaps the ref box
    ]
    job._record_frame_evidence(fused, frame_num=7, main_person_boxes=person)
    assert set(job._match_evidence.by_track) == {1, 2}
    assert job._match_evidence.by_track[1].ever_bumper_color is True   # color folded in
    assert job._match_evidence.by_track[1].field_positions == [(7, 2.0, 4.0)]
    assert job._match_evidence.by_track[2].person_overlap_frames == 1
    assert job._binding_log == [{"frame_num": 7, "bindings": {1: 0, 2: 3}}]


def test_record_frame_evidence_skips_none_track_id():
    job = _job()
    job._record_frame_evidence([_obs(None, (1.0, 1.0), (0, 0, 5, 5), None)],
                               frame_num=0, main_person_boxes=[])
    assert job._match_evidence.by_track == {}


def test_track_id_for_slot_inverts_binding():
    job = _job()
    assert job._track_id_for_slot(0) == 1
    assert job._track_id_for_slot(3) == 2
    assert job._track_id_for_slot(5) is None   # unbound slot -> None (drops cleanly)
```

> This drives the actual `_record_frame_evidence`/`_track_id_for_slot` code with
> synthetic fused observations + a person box, locking the fused-id keying (B1),
> color folding, overlap, and binding-log direction — without the full SAM2 mock stack.

- [ ] **Step 5: Run targeted + regression suites**

Run: `.venv/bin/python -m pytest tests/field/test_track_evidence.py tests/field/test_track_evidence_integration.py tests/field/test_sam2_streaming.py -q`
Expected: track_evidence + integration PASS; `test_sam2_streaming` shows **no new failure** vs. its pre-change result (it has a pre-existing unrelated state; compare, don't assume green).

Run the broader guard: `.venv/bin/python -m pytest tests/field tests/test_detect_bridge.py -q`
Expected: no NEW failures introduced by this slice.

- [ ] **Step 6: Verify no output change**

Confirm `_match_evidence`/`_binding_log` are never written to `meta.json` in this slice (grep `app.py` and `replay_recorder.py` — they must not appear). Output bytes for a fixture run are unchanged.

- [ ] **Step 7: Commit**

```bash
git add web/app.py tests/field/test_track_evidence_integration.py
git commit -m "feat(evidence): collect whole-match track evidence in analysis loop"
```

---

## Done criteria (slice 1)

- `web/field/track_evidence.py` exists, pure, 12+ unit tests green.
- `InferenceJob` populates `self._match_evidence` and `self._binding_log` each frame.
- Robot-confirmation sites (OCR/VLM/gallery/color) record onto evidence.
- **No change** to `meta.json`, HUD, scoring, or any existing test's pass/fail vs. baseline.
- Slice 2 (classifier + retroactive cleanup) can consume `self._match_evidence` + `self._binding_log` with no further plumbing.

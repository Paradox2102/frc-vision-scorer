# Tracking & Identity Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Kalman field tracker's frame-to-frame association with SAM2 mask propagation (gated by a spike that proves it cuts identity swaps), then rebuild identity on the resulting stable tracks.

**Architecture:** A validation spike first compares SAM2 mask-propagation tracking against the existing field-space Kalman `FieldTracker` on a hard segment of real footage. Only if SAM2 wins (≥50% fewer identity swaps, acceptable latency) does it become the tracker; the Kalman tracker is then demoted to a thin field-space `SlotMapper`, multi-camera fusion is re-pointed to per-camera SAM2 tracklets, and identity is rebuilt (`bumper_number` class + IoS matching, consecutive-agreement lock, appearance embeddings). If the spike fails, fall back to *augment* (SAM2 IDs as an association cue) without retiring the Kalman tracker.

**Tech Stack:** Python 3.12 / `.venv`, `facebookresearch/sam2` (`SAM2VideoPredictor`), `ultralytics` YOLO (existing `hub_zone_yolo26s`), `supervision` 0.27, OpenCV, NumPy, pytest. RTX 3090 (CUDA, torch 2.10), offline/batch.

**Spec:** `docs/superpowers/specs/2026-05-29-tracking-identity-upgrade-design.md`

---

## File Structure

**Phase 0 — Spike (SP-T.0), fully detailed below:**
- Create `scripts/sam2_spike.py` — integration harness: runs SAM2 + `FieldTracker` over a clip, renders a side-by-side video, prints the metrics table. Mirrors the standalone pattern in `scripts/verify_field_tracking.py`.
- Create `web/field/spike_metrics.py` — pure, testable metric functions (ID-switch counting from anchors, continuity stats). No SAM2/torch imports, so it unit-tests fast.
- Create `tests/field/test_spike_metrics.py` — unit tests for the metric functions.
- Create `data/videos/qual37.spike_labels.json` — hand-authored ground-truth sidecar (segment frame range + swap/occlusion anchor events).

**Phases 1–4 — post-gate (task-level only; expanded after the spike):**
- Create `web/field/sam2_tracker.py` — production SAM2 tracker wrapper (prompt + propagate + periodic re-prompt), emits `(track_id, box, mask)` per camera.
- Modify `web/field/tracker.py` — `FieldTracker` → thin `SlotMapper` (keep slots/alliance/coast/zones; drop distance min-cost matching).
- Modify `web/app.py` + SP3 fusion modules — re-point fusion to SAM2 tracklets; wire identity.
- Detector training data + config — add `bumper_number` class.
- Create `web/field/number_match.py` — IoS number↔robot matching.
- Modify identity logic — consecutive-agreement lock (replace cumulative-vote lock).
- `web/field/` re-id — appearance embeddings (builds on `docs/superpowers/plans/2026-05-27-embedding-reid-robot-tracking.md`).

---

## Phase 0 — SP-T.0: SAM2 Tracking Spike (the gate)

This phase is the immediate work. It must complete and pass the gate before any Phase 1+ code is written.

### Task 0: SAM2 environment setup

**Files:**
- Modify: `requirements.txt` (or the project's dep manifest) — add `sam2`
- Create: `checkpoints/` (gitignored) — SAM2 weights

- [ ] **Step 1: Install SAM2 into the venv**

Run:
```bash
cd /home/nick/frc-vision-scorer
.venv/bin/pip install "git+https://github.com/facebookresearch/sam2.git"
```
Expected: installs `sam2` and its deps without touching torch (torch 2.10+cu126 already present).

- [ ] **Step 2: Download the checkpoint**

Run:
```bash
mkdir -p checkpoints && cd checkpoints
curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```
Expected: `checkpoints/sam2.1_hiera_large.pt` (~900 MB). (Fallback `sam2.1_hiera_small.pt` / `_base_plus.pt` if latency forces it.)

- [ ] **Step 3: Verify import + CUDA**

Run:
```bash
.venv/bin/python -c "import torch, sam2; from sam2.build_sam import build_sam2_video_predictor; print('sam2 OK, cuda', torch.cuda.is_available())"
```
Expected: `sam2 OK, cuda True`

- [ ] **Step 4: Add checkpoints to .gitignore**

Add `checkpoints/` to `.gitignore`. Do NOT commit the weights.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore(spike): add sam2 dependency for tracking spike"
```

### Task 1: Spike metric functions (pure, TDD)

The metrics must be testable without SAM2/torch. Identity swaps are counted from a small set of hand-labeled **anchors** — `(frame, point_uv, true_label)` — that pin a known physical robot to a screen location at sampled frames. A tracker "switches" when the track covering an anchor's location changes its assigned `true_label` between consecutive anchors for that label.

**Files:**
- Create: `web/field/spike_metrics.py`
- Test: `tests/field/test_spike_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/field/test_spike_metrics.py
from web.field.spike_metrics import count_id_switches, continuity_stats


def test_count_id_switches_detects_a_swap():
    # Two tracks, two labels. Anchors pin label "A" and "B" at known points.
    # tracker_frames[frame] = {track_id: (x, y)}
    tracker_frames = {
        0: {1: (0.0, 0.0), 2: (10.0, 0.0)},
        1: {1: (1.0, 0.0), 2: (9.0, 0.0)},
        2: {1: (9.0, 0.0), 2: (1.0, 0.0)},  # tracks crossed but kept ids
    }
    # Ground truth: label A stays near x=0 then moves to x=9 by frame 2 (it is track 1);
    # label B is the opposite. anchors: (frame, (x,y), label)
    anchors = [
        (0, (0.0, 0.0), "A"), (0, (10.0, 0.0), "B"),
        (2, (9.0, 0.0), "A"), (2, (1.0, 0.0), "B"),
    ]
    # Track 1 holds A throughout (0,0)->(9,0); track 2 holds B. No switch.
    assert count_id_switches(tracker_frames, anchors) == 0


def test_count_id_switches_counts_a_real_swap():
    tracker_frames = {
        0: {1: (0.0, 0.0), 2: (10.0, 0.0)},
        2: {1: (1.0, 0.0), 2: (9.0, 0.0)},  # tracks stayed put; the robots crossed under them
    }
    anchors = [
        (0, (0.0, 0.0), "A"), (0, (10.0, 0.0), "B"),
        (2, (9.0, 0.0), "A"), (2, (1.0, 0.0), "B"),  # A is now on the RIGHT, B on the LEFT
    ]
    # Counting is per-label. Robot A's nearest track flips 1->2 (one switch) and
    # robot B's flips 2->1 (a second switch). A single physical crossing-swap is
    # counted once per involved robot, so the total is 2. Both trackers are scored
    # identically, so the ratio between them stays meaningful.
    assert count_id_switches(tracker_frames, anchors) == 2


def test_continuity_stats_counts_births_and_deaths():
    tracker_frames = {
        0: {1: (0, 0)},
        1: {1: (0, 0), 2: (5, 5)},  # 2 born
        2: {2: (5, 5)},             # 1 died
    }
    s = continuity_stats(tracker_frames)
    assert s["n_tracks"] == 2
    assert s["births"] == 2 and s["deaths"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/field/test_spike_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: web.field.spike_metrics`

- [ ] **Step 3: Write minimal implementation**

```python
# web/field/spike_metrics.py
"""Pure metric functions for the SAM2 tracking spike. No torch/SAM2 imports —
unit-testable in isolation. Identity swaps are counted against hand-labeled
anchors that pin a known physical robot to a screen point at sampled frames."""
from __future__ import annotations
import math


def _nearest_track(tracks: dict, point) -> int | None:
    best, bd = None, math.inf
    for tid, (x, y) in tracks.items():
        d = (x - point[0]) ** 2 + (y - point[1]) ** 2
        if d < bd:
            best, bd = tid, d
    return best


def count_id_switches(tracker_frames: dict, anchors: list) -> int:
    """anchors: list of (frame, point, label). For each label, walk its anchors in
    frame order; the track assigned to the label is the track nearest that anchor's
    point on that frame. Each time that track id changes, count one switch.

    Counting is per-label: one physical crossing that swaps two robots produces two
    switches (one per robot). That is fine — both trackers are scored with the same
    function, so the relative comparison (the gate's ≥50% reduction) is valid."""
    by_label: dict[str, list] = {}
    for frame, point, label in sorted(anchors, key=lambda a: a[0]):
        by_label.setdefault(label, []).append((frame, point))
    switches = 0
    for label, seq in by_label.items():
        prev_tid = None
        for frame, point in seq:
            tracks = tracker_frames.get(frame, {})
            tid = _nearest_track(tracks, point)
            if prev_tid is not None and tid is not None and tid != prev_tid:
                switches += 1
            if tid is not None:
                prev_tid = tid
    return switches


def continuity_stats(tracker_frames: dict) -> dict:
    """Track births/deaths and lifetime stats from per-frame {track_id: pos}."""
    frames = sorted(tracker_frames)
    seen, births, deaths, lifetimes = set(), 0, 0, {}
    prev_ids: set = set()
    for f in frames:
        ids = set(tracker_frames[f])
        births += len(ids - prev_ids)
        deaths += len(prev_ids - ids)
        for tid in ids:
            lifetimes[tid] = lifetimes.get(tid, 0) + 1
        seen |= ids
        prev_ids = ids
    deaths += len(prev_ids)  # tracks alive at the end "die" at clip end
    return {
        "n_tracks": len(seen),
        "births": births,
        "deaths": deaths,
        "mean_frames_per_track": (sum(lifetimes.values()) / len(lifetimes)) if lifetimes else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/field/test_spike_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add web/field/spike_metrics.py tests/field/test_spike_metrics.py
git commit -m "feat(spike): pure id-switch + continuity metrics for tracking comparison"
```

### Task 2: Author the ground-truth label sidecar

**Files:**
- Create: `data/videos/qual37.spike_labels.json`

- [ ] **Step 1: Pick the hard segment**

Scrub `data/videos/qual37.mp4` for a ~15–30 s window containing a hub-zone scrum and at least one same-alliance crossing. Record the frame range. Pick a `prompt_frame` near the segment start where all robots to be tracked are visible (un-occluded), so SAM2 can be prompted cleanly.

Helper to scrub:
```bash
.venv/bin/python -c "import cv2;c=cv2.VideoCapture('data/videos/qual37.mp4');print('fps',c.get(5),'frames',int(c.get(7)))"
```

- [ ] **Step 2: Write the sidecar**

```json
{
  "video": "data/videos/qual37.mp4",
  "segment": {"start_frame": 0, "end_frame": 0, "prompt_frame": 0, "stride": 2},
  "anchors": [
    {"frame": 0, "uv": [0, 0], "label": "blue_A"}
  ],
  "occlusion_events": [
    {"frame": 0, "label": "blue_A", "note": "passes behind hub"}
  ]
}
```
Fill `segment` and add ~2 anchors per robot at the segment start, mid (during the scrum), and end — pinning each physical robot to its screen `uv` at those frames. `uv` is in main-crop pixels (same crop the harness uses).

This anchor/occlusion sidecar shape is what the harness actually reads; it supersedes the rougher `{frame, robot_a_slot, robot_b_slot, kind}` event shape sketched in the spec.

- [ ] **Step 3: Commit**

```bash
git add data/videos/qual37.spike_labels.json
git commit -m "test(spike): ground-truth swap/occlusion labels for qual37 hard segment"
```

### Task 3: Spike harness — SAM2 vs FieldTracker

This is an integration experiment, not TDD. Correctness is judged by the rendered side-by-side video and the printed metrics. Build incrementally and eyeball each stage.

**Files:**
- Create: `scripts/sam2_spike.py`

Reference APIs (verified):
- Calibration: `from web.field import store as field_store; cal = field_store.load("test", "main")`
- Field model + dims: `from web.field.field_model import FieldModel; fm = FieldModel.load(cal.field_season)`
- Detector + bridge: `from web.app import MODEL_PATH, CALIB_MAIN_CROP_FRAC, _detect_bumper_color_static`; `from web.field.detect_bridge import to_field_detections`
- Kalman baseline: `from web.field.tracker import FieldTracker`
- Projection: `cal.image_to_field((u, v))`
- SAM2: `from sam2.build_sam import build_sam2_video_predictor`

- [ ] **Step 1: Scaffold — load clip, calibration, segment sidecar**

Mirror the header of `scripts/verify_field_tracking.py` (path setup, `field_store.load("test", "main")`, `FieldModel.load`, skip-cleanly if calibration absent). Load `data/videos/qual37.spike_labels.json`. Crop each frame to `fr[: int(h * CALIB_MAIN_CROP_FRAC), :]` (same crop the calibration was solved on).

- [ ] **Step 2: Baseline pass — run FieldTracker over the segment**

For each frame in the segment (by `stride`): YOLO detect (`conf=0.4`, robot `class_id == 1`), `to_field_detections(...)`, `trk.update(dets, {}, dt=stride/fps)`. Record `baseline_frames[frame] = {slot_id: pos}` and keep each track's `box` for rendering.

- [ ] **Step 3: SAM2 pass — prompt at prompt_frame, propagate over the segment**

```python
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
                                        "checkpoints/sam2.1_hiera_large.pt", device="cuda")
# init_state accepts a frames directory; export the segment crop (same
# CALIB_MAIN_CROP_FRAC crop the calibration was solved on — no masks needed for the
# single-prompt gate) into a tempfile.TemporaryDirectory() so SAM2 sees exactly the
# calibrated view and the JPEGs are cleaned up automatically.
#   with tempfile.TemporaryDirectory() as TMP_FRAME_DIR: export frames; init_state(...)
state = predictor.init_state(video_path=TMP_FRAME_DIR)
# prompt with detector boxes on the (segment-local) prompt frame
import torch
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for obj_id, box in enumerate(prompt_boxes, start=1):
        predictor.add_new_points_or_box(state, frame_idx=PROMPT_LOCAL_IDX,
                                        obj_id=obj_id, box=box)
    for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        masks = (mask_logits > 0).cpu().numpy()
        # mask -> box -> bottom-center -> cal.image_to_field
        ...
        sam2_frames[seg_to_global(f_idx)] = {oid: field_xy for oid, field_xy in ...}
```
Note: for the gate, a **single prompt at a clean frame** is sufficient to test SAM2's core occlusion-robustness. Periodic re-prompting is deferred to Phase 1 (SP-T.1).

- [ ] **Step 4: Compute + print metrics for both**

```python
from web.field.spike_metrics import count_id_switches, continuity_stats
anchors = [(a["frame"], tuple(a["uv"]), a["label"]) for a in labels["anchors"]]
# anchors are in image uv; project to field once so they live in the same space as tracks
anchors_field = [(f, cal.image_to_field(uv), lab) for f, uv, lab in anchors]
print("FieldTracker swaps:", count_id_switches(baseline_frames, anchors_field), continuity_stats(baseline_frames))
print("SAM2 swaps:        ", count_id_switches(sam2_frames, anchors_field), continuity_stats(sam2_frames))
print("SAM2 latency ms/frame:", sam2_ms_per_frame)
```

- [ ] **Step 5: Render side-by-side video for eyeballing**

Left = frame with FieldTracker slot boxes+ids; right = frame with SAM2 masks+ids. Write `data/videos/qual37_spike_compare.mp4`. Visually confirm the swap counts match what you see.

- [ ] **Step 6: Run the spike**

Run: `.venv/bin/python scripts/sam2_spike.py`
Expected: prints both metric lines + latency; writes the comparison video.

- [ ] **Step 7: Commit the harness**

```bash
git add scripts/sam2_spike.py
git commit -m "feat(spike): SAM2-vs-FieldTracker comparison harness on qual37 segment"
```

### Task 4: Evaluate the gate and record the decision

- [ ] **Step 1: Compute swap reduction**

`reduction = (baseline_swaps - sam2_swaps) / max(1, baseline_swaps)`.

- [ ] **Step 2: Record the result in the spec**

Append a short "SP-T.0 Result" note to the spec: baseline swaps, SAM2 swaps, reduction %, ms/frame, and the GATE verdict.
- **PASS** (reduction ≥ 0.5 AND latency acceptable for batch) → proceed to Phase 1 (replace).
- **FAIL** → switch the plan to the *augment* fallback (SAM2 IDs as a cue into the Kalman matching); Phase 1 below is replaced accordingly. **Stop and surface to the human** with the numbers before proceeding.

- [ ] **Step 3: Commit the decision**

```bash
git add docs/superpowers/specs/2026-05-29-tracking-identity-upgrade-design.md
git commit -m "docs(spike): record SP-T.0 gate result and verdict"
```

---

## 🚦 GATE — do not proceed past here until Task 4 records a PASS

If FAIL: re-plan around the *augment* fallback (the spike harness and metrics are reused as-is for the augmented tracker). The phases below assume PASS.

---

## Phase 1 — SP-T.1: SAM2-as-tracker + SlotMapper (task-level; expand after gate)

To be broken into bite-sized TDD steps once Phase 0 confirms SAM2 behavior (mask quality on distant robots, re-prompt cadence observed in the spike).

- **Task 1.1 — `web/field/sam2_tracker.py`.** Production wrapper: `prompt(frame, boxes)`, `propagate(frame) -> list[Track]`, `reprompt(frame, boxes)`, `reset()`. Periodic re-prompt: re-run detector every N frames, IoS-match to existing masks, prompt unmatched. Unit-test with a **stub predictor** returning canned masks — verify mask→box, re-prompt matching, ID stability. (No GPU in tests.)
- **Task 1.2 — Demote `FieldTracker` → `SlotMapper` (`web/field/tracker.py`).** Keep 6 slots, alliance-by-side, coast budget, static-zone seeding gate, `update(...) -> list[TrackOut]` contract. Drop distance min-cost permutation matching; bind a stable upstream id to a slot on first sight. New tests: stable-id-keeps-slot across coast; new-id-in-static-zone not seeded; coast-budget semantics under stable ids (per spec). Update `tests/test_radar.py` / existing tracker tests as needed.
- **Task 1.3 — Wire into the offline path** behind a flag so the Kalman path stays runnable for comparison. Update `scripts/verify_field_tracking.py` to optionally drive the SAM2 tracker → SlotMapper.

## Phase 2 — SP-T.2: Re-point SP3 fusion (task-level)

- **Task 2.1 — Run `sam2_tracker` per sub-view** (main + corner_left + corner_right via `composite_layout` / `camera_view.extract`). Project each view's mask bottom-centers to field.
- **Task 2.2 — Fuse tracklets** using SP3's existing additive-evidence cross-camera association (field-proximity), feeding the SlotMapper. Reuse the fusion-evidence model unchanged; swap detection source for SAM2 tracklets.
- **Task 2.3 — Extend fusion tests**: two-view same field position → one slot; corner-only (main-occluded) tracklet → its slot. Use `2026hop` calibration (all three cameras present).

## Phase 3 — SP4: Identity rebuild (task-level)

- **Task 3.1 — `bumper_number` detection class.** Label data (parallel long pole), add to dataset YAML, retrain. Out-of-band from code TDD; track as its own effort.
- **Task 3.2 — `web/field/number_match.py` (IoS).** Rasterize number boxes to masks (`sv.xyxy_to_mask`), match to robot masks via `sv.mask_iou_batch(..., overlap_metric=sv.OverlapMetric.IOS)` ≥ 0.9. Output number crop keyed to track id. TDD: containment, ambiguous overlap, no-match.
- **Task 3.3 — Consecutive-agreement lock.** Replace cumulative-vote lock with lock-after-N-consecutive + `VLM_DIRECT_CONTRADICTION_CONF` unlock. TDD state machine incl. wrong-lock recovery. Targets `reread-audit-deadlock`.
- **Task 3.4 — Appearance-embedding re-id.** Per-track SigLIP/DINOv2 embedding to re-link a lost track before treating a new id as new. Builds on `2026-05-27-embedding-reid-robot-tracking.md`. TDD: match/no-match on stored embeddings.

---

## Notes for the executor

- Test env: use `.venv/bin/python -m pytest` (base conda python lacks pytest).
- Never push to `origin` (ohmeow-ship); the team remote is `paradox`. Don't push unless asked.
- All work is on branch `feat/sam2-tracking-identity-upgrade`.
- Phase 0 is the only fully-specified phase. Phases 1–4 are deliberately task-level and MUST be expanded to bite-sized TDD steps (re-invoke writing-plans) after the gate, informed by what the spike reveals.

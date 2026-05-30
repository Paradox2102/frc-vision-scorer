# Phase 1 — SAM2-as-tracker + SlotMapper (offline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a production offline SAM2 mask-propagation tracker and make `FieldTracker` consume its stable track ids (sticky id→slot binding), without touching the live `app.py` streaming loop.

**Architecture:** Offline-first (the spec defers live). `web/field/sam2_tracker.py` propagates SAM2 masks over a clip and emits stable per-track boxes. `FieldDetection` gains an optional `track_id`; `FieldTracker.update` binds a stable id to a slot on first sight and keeps it (no distance matching needed for that detection), falling back to today's distance min-cost matching when `track_id is None`. This is **backward-compatible**: `app.py` passes anonymous fused detections (no track_id) and behaves exactly as today; the new offline driver passes track_ids and gets SAM2-stable slots. An offline driver script validates the whole chain on real footage.

**Why app.py is untouched:** `app.py` runs a per-frame streaming loop (`process_video`); `SAM2VideoPredictor` is batch/offline. Wiring SAM2 into the live loop needs the streaming SAM2 fork and is part of the later live milestone, not this phase.

**Tech Stack:** Python 3.12 / `.venv`, `sam2` `SAM2VideoPredictor` (pinned), `supervision` 0.27, OpenCV, NumPy, pytest. RTX 3090, offline.

**Spec:** `docs/superpowers/specs/2026-05-29-tracking-identity-upgrade-design.md` (SP-T.1). **Gate SP-T.0 PASSED.**

**Spike learnings carried in:** SAM2 API = `build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt")`, `init_state(video_path=frames_dir)`, `add_new_points_or_box(state, frame_idx, obj_id, box=)`, `propagate_in_video(state)` → `(f_idx, obj_ids, mask_logits)`; masks via `(mask_logits>0).cpu().numpy()` then `sv.mask_to_xyxy`; 0.5× crop export is fine; ~151 ms/frame on GPU; `_C` ext uncompiled (non-fatal); robots occluded at the prompt frame need a re-prompt to be picked up.

---

## File Structure

- Create `web/field/sam2_tracker.py` — offline SAM2 tracker + pure helpers (mask→box, IoS new-detection matching). Reuses the SAM2 flow proven in `scripts/sam2_spike.py`.
- Create `tests/field/test_sam2_tracker.py` — unit tests for the pure helpers + the tracker wrapper driven by a **stub predictor** (no GPU/torch).
- Modify `web/field/tracker.py` — `FieldDetection` gains optional `track_id`; `FieldTracker._assign` honors stable ids (sticky bind) when present. Backward-compatible.
- Modify `tests/field/` (existing tracker tests) + add cases for stable-id binding.
- Create `scripts/sam2_field_track.py` — offline driver: SAM2 tracker → FieldTracker(track_ids) over a clip; asserts fixed-6 invariants; reports continuity vs the anonymous path.

---

## Task 1.1: `sam2_tracker.py` — pure helpers (TDD, additive, no app.py impact)

Start with the pure, GPU-free helpers so they're unit-tested fast.

**Files:** Create `web/field/sam2_tracker.py`, `tests/field/test_sam2_tracker.py`.

- [ ] **Step 1: Write failing tests for the pure helpers**

```python
# tests/field/test_sam2_tracker.py
import numpy as np
from web.field.sam2_tracker import mask_to_box, bottom_center, match_new_detections


def test_mask_to_box_tight_bbox():
    m = np.zeros((10, 10), bool); m[2:5, 3:8] = True
    assert mask_to_box(m) == (3, 2, 8, 5)  # x1,y1,x2,y2 (exclusive max+1)


def test_mask_to_box_empty_returns_none():
    assert mask_to_box(np.zeros((4, 4), bool)) is None


def test_bottom_center():
    assert bottom_center((3, 2, 8, 5)) == (5.5, 5.0)


def test_match_new_detections_flags_unmatched():
    # existing masks cover left half; a new det on the right is unmatched (new robot)
    existing = {1: _rect_mask(0, 0, 5, 10)}
    new_boxes = [(1, 1, 4, 9), (6, 1, 9, 9)]   # first overlaps id1, second doesn't
    unmatched = match_new_detections(existing, new_boxes, (10, 10), ios_thresh=0.5)
    assert unmatched == [1]  # index of the second box only


def _rect_mask(x1, y1, x2, y2, wh=(10, 10)):
    m = np.zeros((wh[1], wh[0]), bool); m[y1:y2, x1:x2] = True; return m
```

- [ ] **Step 2: Run → fail** (`.venv/bin/python -m pytest tests/field/test_sam2_tracker.py -v`)

- [ ] **Step 3: Implement the pure helpers** in `web/field/sam2_tracker.py`:
  - `mask_to_box(mask)->tuple|None`: nonzero bbox (x1,y1,x2,y2), None if empty. Use NumPy `np.where` (torch-free for tests). **Convention: exclusive max** — `x2 = xs.max()+1`, `y2 = ys.max()+1` (matches the test `(3,2,8,5)` for `m[2:5,3:8]`). Note this differs from the spike's inclusive `_mask_to_box`; pick exclusive here and keep `bottom_center` consistent with it.
  - `bottom_center(box)->tuple`: `((x1+x2)/2, y2)` — `y2` is the exclusive-max bottom from `mask_to_box`.
  - `match_new_detections(existing_masks, new_boxes, wh, ios_thresh)->list[int]`: rasterize each new box to a mask (`np.zeros` + slice), compute Intersection-over-Smaller vs each existing mask, return indices of boxes whose best IoS < thresh (i.e. genuinely new robots to prompt). Keep IoS as a small pure helper.

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit** (`feat(sam2): pure mask/box/IoS helpers for the SAM2 tracker`).

## Task 1.2: `Sam2FieldTracker` wrapper (stub-tested, additive)

**Files:** extend `web/field/sam2_tracker.py`, `tests/field/test_sam2_tracker.py`.

- [ ] **Step 1: Failing test with a stub predictor.** Define a `StubPredictor` exposing `init_state`, `add_new_points_or_box`, `propagate_in_video` returning canned `(f_idx, obj_ids, mask_logits)` for 3 frames / 2 objects. Assert `Sam2FieldTracker.track(...)` returns `{frame_idx: {obj_id: {"box":..., "mask":...}}}` with stable ids across frames and correct boxes.

- [ ] **Step 2-4: Implement `Sam2FieldTracker`** generalizing `scripts/sam2_spike.py`'s SAM2 pass: ctor takes a predictor factory (so the stub injects), `track(frames_dir, prompt_idx, prompt_boxes)` prompts then propagates, converting each mask to a box. Re-prompting: expose `reprompt_boxes` hook but keep the default single-prompt (occlusion-recovery re-prompt is a documented follow-up — the offline predictor's mid-stream re-prompt needs care; flag as DONE_WITH_CONCERNS if attempted). Run tests → pass.

- [ ] **Step 5: Commit** (`feat(sam2): offline Sam2FieldTracker wrapper (stub-tested)`).

## Task 1.3: `FieldDetection.track_id` + sticky id→slot binding (TDD, backward-compatible)

**Files:** Modify `web/field/tracker.py`; extend the existing tracker tests.

Integration facts: `FieldDetection(pos, color, box, score)` (`tracker.py:18`); `FieldTracker.update(detections, occlusion, dt)`; `_assign` builds `last_det_to_slot[orig_idx]=slot_id`; `app.py` passes anonymous fused detections (no track_id) at `app.py:2340`.

- [ ] **Step 1: Failing tests** — (a) a detection with a stable `track_id` binds to a slot and KEEPS that slot on later frames even when a closer detection of another id appears (no distance steal); (b) detections with `track_id=None` behave exactly as today (regression test mirroring existing behavior).

- [ ] **Step 2-4:** Add `track_id: int|None = None` to `FieldDetection`. In `_assign`, before distance matching, bind any detection whose `track_id` is already mapped to its existing slot (sticky), and bind a never-seen `track_id` to a free slot (respecting alliance side + static-zone gate); only `track_id is None` detections go through the existing distance min-cost path. **Critical: remove the sticky-bound detections from the paired list BEFORE allocating the `np.full` cost matrix**, or they'd appear as both sticky and distance candidates and corrupt the index→slot mapping. For each sticky detection still set `last_det_to_slot[orig_idx]=slot_id` so the VLM-binding contract holds. Maintain a `self._id_to_slot` map. Preserve `last_det_to_slot` + `TrackOut`/`update` contract exactly. Run full `tests/field/` + `tests/test_radar.py` + `tests/test_alliance_consistency.py` → green (proves app.py path unaffected).

- [ ] **Step 5: Commit** (`feat(field): optional stable track_id with sticky slot binding (back-compat)`).

## Task 1.4: offline driver + continuity check

**Files:** Create `scripts/sam2_field_track.py`.

- [ ] End-to-end: `Sam2FieldTracker` over a clip → per-frame field detections WITH track_ids (bottom-center → `image_to_field`) → `FieldTracker.update` → assert fixed-6 / 3-per-alliance invariants hold; print continuity vs the anonymous path. Run on the qual1 window; eyeball the radar/boxes. Commit.
- **Coordinate-space gotcha:** SAM2 masks are computed on the 0.5×-scaled export frames (see the spike). Rescale mask bottom-centers back to full-res crop pixels (÷0.5 → ×2) BEFORE `cal.image_to_field(...)`, since the calibration is in full-res main-crop space. Mirror the spike's rescale at `scripts/sam2_spike.py` (~lines 366–371).

---

## Notes
- `.venv/bin/python -m pytest`. Branch `feat/sam2-tracking-identity-upgrade`. Never push to `origin` (use `paradox`); don't push unless asked.
- **Do NOT modify `web/app.py` in this phase** — the live streaming integration is the later live milestone.
- Tasks 1.1–1.2 are additive (zero app risk). Task 1.3 touches shared `tracker.py` — the regression tests (full `tests/field/` + radar + alliance) are the guardrail that `app.py`'s path is unchanged.

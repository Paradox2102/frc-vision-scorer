# Field-Space Tracking + Alliance (SP2) — Design

**Date:** 2026-05-29
**Status:** Approved (pending spec review)
**Part of:** "Field-coordinate rework" — Sub-project 2 of 5
**Builds on:** SP1 field-coordinate foundation (`web/field/` calibration, projection, radar)
and the tag-anchored calibration (`docs/superpowers/specs/2026-05-29-tag-anchored-calibration-design.md`).

## Problem

The live pipeline tracks robots in **pixel space** (`InferenceJob`: ByteTrack →
nearest-pixel canonical-ID merge → appearance embeddings → VLM team votes). It loses
and swaps identities when robots cross, cluster, or are occluded — the failures the two
2026-05-27 reid specs tried to patch. SP1 gave us a calibrated map from image pixels to
metric field coordinates. SP2 replaces pixel-space tracking with **field-space
tracking**: a robot's field position is physical and continuous, so crossings and
occlusions become tractable with a motion model and explicit occlusion geometry.

## Scope

**In scope (SP2):**
- A fixed **3v3** field-space tracker: 6 stable tracks (3 blue + 3 red), constant-
  velocity motion model, per-alliance association, occlusion-aware coasting.
- Alliance per track from **bumper color** (sticky), not field side.
- **Occlusion model** (hub + person + robot-robot) feeding coasting and VLM gating.
- **Replace** the canonical-ID core of `InferenceJob` with the field tracker; wire the
  live **radar**; **re-bind** existing VLM bumper reads onto stable tracks.

**Out of scope:**
- Clean team-number identity rebuild (SP4) — VLM reads are only *re-bound* to tracks.
- Multi-camera fusion (SP3) — main camera only.
- Corner-camera calibration, scoring, analytics, photoreal radar (SP3/SP5).
- Person tracking/identity — people are per-frame occluder boxes only.

## Decisions (from brainstorming)

- **Replace the live pixel pipeline now** (not standalone): the field tracker becomes
  the identity source in `InferenceJob`; the two 2026-05-27 reid specs are superseded.
- **Alliance from bumper color** (the existing `_detect_bumper_color_static`); field
  side is only a sanity hint.
- **Fixed 3v3 hard constraint**: exactly 6 tracks, never >3 per alliance.
- **Re-bind VLM** team reads to tracks (fixed-6 ⇒ duplicate team numbers impossible).
- **Occlusion folded into SP2**: hubs (static, predictive ray-cast) + people (dynamic) +
  **robot-robot** (blocks bumper reads).

## Coordinate consistency

Calibration was solved in the main-cam crop frame (origin (0,0); SP1 used the top
`CALIB_MAIN_CROP_FRAC=0.60`). The live `InferenceJob` crops `OVERHEAD_CROP_RATIO=0.67`.
Both are top-aligned from (0,0), so a detection's `(u,v)` is the same coordinate in both
frames and `image_to_field` is valid across the image plane. Detections in the
0.60–0.67 strip extrapolate the calibration slightly (acceptable; noted as a minor
accuracy caveat). **A calibration must exist for the event/camera** or field tracking
cannot run.

## Architecture

```
frame
 └─ crop main-cam region
     ├─ custom YOLO (robot) ─► boxes ─► bottom-center → image_to_field → (x,y)
     │                                   bumper color  → alliance hint
     │                                                         │
     ├─ stock COCO YOLO (person) ─► person boxes ─► feet→ground (field-adjacent only)
     │                                                         │
     │                              hub prisms (from field model, once)
     │                                                         ▼
     │                                          occlusion resolver (camera pose +
     │                                          tracks + persons + hub prisms)
     │                                            → per-track view_occluded / bumper_occluded
     │                                                         │
     └────────── FieldDetections + occlusion flags ───────────► FieldTracker (fixed 3v3,
                                                                 CV-Kalman, Hungarian)
                                                                  → 6 stable tracks
                                                                     ├─ HUD labels (alliance+slot, team)
                                                                     ├─ Radar dots (alliance-tinted)
                                                                     └─ VLM re-bind (gated by bumper_occluded)
```

### Module structure

```
web/field/
  tracker.py     # FieldTracker — pure (no cv2/torch). Fixed 3v3 CV-Kalman + Hungarian.
  occlusion.py   # pure geometry: hub prisms, ray-cast, depth-order, box overlap.
  detect_bridge.py  # YOLO boxes + calibration + frame -> FieldDetection list (uses cv2).
```
`tracker.py` and `occlusion.py` are pure/numpy (fast, deterministic tests).
`detect_bridge.py` is the thin cv2-touching adapter. Integration lives in `InferenceJob`.

## Components

### 1. `FieldTracker` (`web/field/tracker.py`) — pure

- **Input per step:** `detections: list[FieldDetection]`, `occlusion: dict[slot_id ->
  {view_occluded: bool}]` (bumper flags are consumed downstream, not by the tracker),
  `dt: float`.
- **`FieldDetection`:** `{pos: (x, y), color: "blue"|"red"|"unknown", box: (x1,y1,x2,y2),
  score: float}`.
- **State:** 6 `Track {slot_id, alliance (sticky), kf (x,y,vx,vy), last_seen, hits,
  coasting, missed}`. Slots 0–2 blue, 3–5 red.
- **Motion model:** constant-velocity Kalman filter in meters; process noise tuned for
  robot dynamics; `dt` from frame timestamps (FRAME_SKIP-aware).
- **Step:**
  1. **Predict** every track.
  2. **Per alliance**, gate detections of that color (and `unknown`) to the 3 tracks by
     Mahalanobis/Euclidean distance; **Hungarian-assign** ≤3 detections to 3 tracks.
  3. **Update** matched tracks (KF correct). Reset `missed`, `coasting=False`.
  4. **Unmatched track:** `missed += 1`. Coast on prediction. Allowed-miss budget is
     `MAX_COAST_OCCLUDED` when `view_occluded` else `MAX_COAST_OPEN` (occluded tracks
     coast far longer and **cannot be re-seeded/stolen**).
  5. **Unmatched detection:** seed it only into an *empty or expired* same-alliance slot
     (a slot whose `missed` exceeded its budget and is **not** occluded).
- **Alliance stickiness:** set at slot seed; per-frame detection color is only an
  association hint (cost penalty on mismatch), never flips a slot.
- **Invariants:** never >6 tracks; never >3 per alliance; `slot_id` stable across
  crossings/occlusions.
- **Output:** 6 tracks: `{slot_id, alliance, pos, vel, box?, state: visible|coasting}`.

### 2. `occlusion.py` — pure geometry

Principle: **an occluder lies between the camera and the robot.** Camera center =
`-R^T t` from the calibration.

- `hub_prisms(field_model) -> list[Prism]`: cluster the field tags into the two hub
  rings (k=2 on tag (x,y), or split by `alliance_side`); each prism footprint = convex
  hull of a cluster's (x,y); height = max cluster tag z (+ small margin).
- `occluded_by_hub(cam, prism, track_xy, robot_h) -> bool`: does the segment
  camera_center → (track_xy, robot_h) intersect the prism volume? Uses the track's
  **predicted** position → predictive.
- `camera_distance(cam, ground_xy) -> float`: distance from camera center to a ground
  point (depth ordering).
- `box_overlap(a, b) -> float`; `bumper_region(box) -> box` (lower band of a robot box).
- `resolve_occlusion(cam, tracks, person_boxes, hub_prisms, robot_h) ->
  dict[slot_id -> {view_occluded, bumper_occluded, cause}]`:
  - **view_occluded** if a hub prism is hit, OR a nearer occluder's box (person, or
    another track) covers most of the track's expected box.
  - **bumper_occluded** if a nearer occluder's box covers the track's **bumper region**
    (covers the robot-robot "blocks the number" case; also set by hub/person partial
    overlap of the bumper band).
  - Occluders are depth-ordered by `camera_distance` of their ground points; only
    occluders **nearer than the track** count.

### 3. `detect_bridge.py`

- `to_field_detections(robot_boxes, calibration, frame) -> list[FieldDetection]`:
  bottom-center `((x1+x2)/2, y2)` → `image_to_field`; bumper color via
  `_detect_bumper_color_static`. Drop detections returning `None` (above horizon) or
  landing off-field (bounds + margin).
- `field_adjacent_persons(person_boxes, calibration, field_dims, margin) -> list[box]`:
  keep only persons whose feet ground-project onto/near the carpet within `margin` of
  the field perimeter; drop crowd/stands (feet off-field or above horizon).

### 4. `InferenceJob` integration

- **Startup:** load `field_store.load(event, camera)`; if absent → clear error, field
  tracking disabled (we require a calibration). Build `hub_prisms` once. Load a stock
  COCO YOLO (`yolo26n.pt`/`yolov8n.pt`) for `person`.
- **Per frame:** custom YOLO robots → `to_field_detections`; COCO YOLO → persons →
  `field_adjacent_persons`; `resolve_occlusion` over current tracks + persons + hub
  prisms; `FieldTracker.update(detections, {slot: view_occluded}, dt)` → 6 tracks.
  `slot_id` is the canonical identity.
- **VLM re-bind:** `vision_identify_robots`/`ocr_robot_bumpers` key votes by `slot_id`;
  **skip/discount reads for tracks with `bumper_occluded`** (prevents reading an
  occluder's number onto the rear robot). Fixed-6 makes duplicate team numbers
  impossible.
- **Output:** HUD labels by alliance+slot and team (votes); `Radar.render` of the 6
  track positions, alliance-tinted; existing diagnostics keyed by slot_id.
- **Remove:** `get_canonical_robot_id`, `bytetrack_to_canonical`, embedding re-id
  (`embedder`/`reid` use in the loop), appearance alliance clustering, alliance capping
  (fixed-6 supersedes it), the side-based alliance heuristic. **Drop ByteTrack** (the
  field tracker owns continuity). **Keep:** custom YOLO, hub logic, bumper color, VLM,
  diagnostics.

## Data flow

`frame → YOLO(robot)+YOLO(person) → field detections + field-adjacent persons →
resolve_occlusion(camera, tracks, persons, hubs) → FieldTracker.update → 6 tracks →
{HUD, radar, VLM re-bind gated by bumper_occluded}`.

## Error handling

- No calibration → explicit error; field tracking does not start.
- `image_to_field` None / off-field → drop detection.
- Color unknown → still tracked by position; alliance from the slot.
- <6 visible (occlusion/off-frame) → coast; never fabricate a detection.
- >3 of one color (false positive/misread) → Hungarian keeps the best 3.
- All `bumper_occluded` reads skipped → team stays at its last confident vote.

## Testing

- **`FieldTracker`** (synthetic, pure): straight tracks; crossing tracks keep slot_ids;
  occlusion coast + recover (long coast when `view_occluded`); color-misread robustness;
  fixed-6 / ≤3-per-alliance invariants; re-seed only after expiry and only when not
  occluded.
- **`occlusion.py`** (synthetic, pure): robot path behind a hub prism flagged
  `view_occluded` over the correct interval; person/robot box over a **farther** track
  → `bumper_occluded` (and `view_occluded` when coverage is large), over a **nearer**
  track → neither; crowd person (off-field feet) ignored; depth ordering correct.
- **`detect_bridge`**: box→field via a synthetic calibration; off-field/None dropped;
  field-adjacent person filter keeps perimeter people, drops stands.
- **Integration**: run `InferenceJob` on the test video + saved calibration over a
  window; assert exactly 6 tracks, stable slot_ids, radar renders, no duplicate team
  numbers, and a known robot-behind-robot pass suppresses the rear VLM read.

## Build order (for the plan)

1. `FieldTracker` + tests (pure; no app changes).
2. `occlusion.py` + tests (pure).
3. `detect_bridge.py` + tests.
4. `InferenceJob` integration: replace canonical-ID core, wire radar, re-bind VLM with
   `bumper_occluded` gating, add COCO person YOLO. Remove superseded reid/ByteTrack code.

Each step is independently verifiable; 1–3 land with no risk to the live pipeline.

## YAGNI

- No IMM/MHT (CV-Kalman suffices for this vantage); no re-id embeddings; no multi-camera;
  no clean identity rebuild (SP4); no person tracking/identity (per-frame boxes only);
  no full-mesh occlusion (SP5) — hub prisms from tag rings are enough.

## Risks & mitigations

- **Freeing ByteTrack** removes a working path → keep changes confined to `InferenceJob`,
  land tracker/occlusion/bridge first (verified in isolation), integrate last; the
  calibration-required guard prevents silent breakage.
- **Second YOLO compute** → SP1 already accepts offline-slow; run person detection at a
  reduced cadence if needed.
- **Bumper-color noise** → sticky per-slot alliance + association cost (not a hard flip);
  majority over a track's history.
- **Robot-robot depth ordering** relies on ground projection of both robots → both feet
  on carpet; reasonable for this high vantage.

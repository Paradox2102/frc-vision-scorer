# Field-Coordinate Foundation — Design

**Date:** 2026-05-28
**Status:** Approved (pending spec review)
**Component:** new `web/field/` package (calibration, projection, radar) + FastAPI calibration endpoints + calibration UI
**Part of:** "Field-coordinate rework" — Sub-project 1 of 5 (see [Roadmap](#roadmap-where-this-fits))

## Problem

The tracker performs poorly across every axis at once: ID switches / track
fragmentation, wrong team numbers, wrong alliance, and robots lost during hub
scrums. The root cause is structural — **everything lives in raw overhead-crop
pixel space.** Continuity is fought with appearance re-ID, alliance is decided by
brittle heuristics (x-position, then a hard-coded frame-1600 color switch), and
all of it is entangled in a 3,300-line `web/app.py`.

The basketball-tracking approach the team wants to adopt
([Roboflow notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb))
supplies the missing layer: project every detection onto a **known field map**.
Once detections are field `(x, y)` coordinates, three of the four failure modes
become byproducts of geometry rather than things we fight:

- **Continuity** — a robot is a persistent field-space object with a motion model;
  it survives occlusion via its predicted position instead of relying on
  appearance.
- **Alliance** — which half of the field a robot is on. No color heuristic.
- **Occlusion / off-screen** — field position carries across cameras, so "lost in
  the main camera" is not "lost."

The FRC setup has an asset most sports-tracking pipelines lack: **AprilTags (36h11)
at published field coordinates**, which make camera calibration largely automatic.

## Scope

This spec covers **Sub-project 1 only: the field-coordinate foundation for the
main camera.** It is the base every later sub-project stands on.

### In scope

- A `field_model` that loads the season's WPILib `AprilTagFieldLayout` JSON and a
  small curated set of 3D landmark points; exposes field dimensions and an
  alliance-side test.
- AprilTag (36h11) detection on a frame.
- Full camera calibration with the **fisheye** model: intrinsics + distortion
  (`cv2.fisheye.calibrate`) and pose (`cv2.solvePnP` on undistorted points),
  using AprilTag correspondences plus a manual-click fallback.
- The two projections every later sub-project depends on: `image_to_field` (ground
  plane, z=0) and `field_to_image`.
- Reprojection-error validation, a field-line overlay on the video for eyeballing,
  and an acceptance gate on RMS error.
- Persistence of the calibration (intrinsics reusable, pose re-solvable).
- A diagnostic **radar**: a top-down field that plots the *current frame's* robot
  detections as field `(x, y)` dots, tinted by field side.

### Out of scope (later sub-projects)

| # | Sub-project | Delivers |
|---|---|---|
| **2** | Field-space tracking + alliance | 6 stable tracks with a motion model; alliance by field side; survive occlusion (main cam) |
| **3** | Multi-camera fusion | Calibrate both corner cameras into the shared field space; fuse detections; cross-camera association |
| **4** | Identity rebuild | Greenfield team-number assignment riding on stable field tracks |
| **5** | Polish / scoring / analytics | Photoreal radar, scoring-from-position, heatmaps, speed/distance |

SP1's radar shows **raw per-frame detection dots only** — no track persistence and
no team numbers. Side-tinting is a free geometric hint, not identity.

## Context: input & cameras

- **Input is a single composited video** (main wide shot on top, two corner
  cameras side-by-side along the bottom, one shared scoreboard/clock). The three
  views are therefore perfectly time-synced; multi-camera work (SP3) is "crop three
  fixed sub-regions," not frame-level synchronization.
- **Cameras are static** during a match and usually across an event, but differ
  slightly per venue. Calibration is a one-time solve per camera setup.
- **The main lens is strongly fisheye** (visible barrel curvature of the field
  rail), which is why the fisheye distortion model is required rather than a planar
  homography on raw pixels.
- SP1 operates on the **main-camera crop region only.**

## Goals

- Calibrate a full fisheye camera model for the main camera, automatically from
  AprilTags where possible and via a manual fallback where not.
- Provide accurate, tested `image_to_field` / `field_to_image` projections.
- Make calibration quality *visible and gated*: RMS reprojection error plus a
  field-line overlay and a radar preview.
- Establish a clean, isolated `web/field/` package — the seam where the rework
  begins paying down the `app.py` monolith — with a small, stable public surface
  for SP2–SP4.

## Non-goals

- No track persistence, motion model, team identity, or scoring (SP2/SP4/SP5).
- No corner-camera calibration or fusion (SP3).
- No performance optimization. **Correctness first, speed later** — calibration may
  process many frames and run slowly; that is acceptable for SP1.
- The full 3D field mesh is *not* a runtime dependency; we extract coordinates from
  it offline into small data files.

## Coordinate frame

Adopt the **WPILib field convention**: meters, origin at the corner by the Blue
alliance area, +x down the long axis toward the Red alliance area, +y across, +z up,
carpet at z=0. The published `AprilTagFieldLayout` drops in directly.

**Season is confirmed: 2026 REBUILT.** Canonical layout is
`data/field/2026-rebuilt-welded.json` (WPILib `allwpilib@v2026.2.2`, 32 tags, full
3D pose). **The frame cross-check has already passed:** the CAD engineering drawings
are in inches and reconcile to the JSON meters by an exact ×0.0254 scale with a
shared origin/axes (verified on tag 1 and on field dimensions —
length 16.541 m / 651.22 in, width 8.069 m / 317.69 in). See
`data/field/FIELD_REFERENCE.md` for the full reconciliation and the transcribed CAD
tag table (the documented manual-fallback source).

**Tag size caveat:** the WPILib JSON carries tag *poses* but not tag *size*. The
official FRC 36h11 tag edge length must be supplied as a constant before tag-corner
3D coordinates can be derived from pose (confirm from the 2026 manual; likely
0.1651 m / 6.5 in — do not hardcode unverified).

**Main-camera vantage** is from behind/above the scoring table looking across the
field — well-suited to ground-plane projection.

## Architecture

### Pipeline (SP1)

```
Composited frame
  └─ crop main-camera sub-region
       │   intrinsics K,D  ── once per camera (multi-viewpoint calib OR imported) ──┐
       │                                                                            │
       └─ [per event]  detect AprilTags ──► correspondences ──┐                     │
       │                 (+ manual clicks) ───────────────────┤                     ▼
       │                                                       ▼          solvePnP(undistorted)
       │                                              (K,D from above) ──► rvec, tvec
       │                                                                  │
       │                                                       CameraCalibration (persisted)
       │
       └─ [per frame]  YOLO robot detect → bottom-center pixel
                        └─ CameraCalibration.image_to_field(uv) → (x, y)
                              └─ radar.render → top-down dots (tinted by side)
```

### Module structure

A new isolated package; deliberately **not** grafted into `web/app.py`.

```
web/field/
  __init__.py
  field_model.py      # load WPILib AprilTagFieldLayout JSON + curated landmarks_3d;
                      #   field dimensions; tag_corners_3d(id)->(4,3);
                      #   alliance_side((x,y))->"blue"|"red"
  apriltag_detect.py  # detect 36h11 tags -> {tag_id: (4,2) corner px}  (pupil-apriltags)
  calibration.py      # Calibrator.solve(correspondences, image_size) -> CameraCalibration
                      #   CameraCalibration.image_to_field(uv) -> (x, y)
                      #   CameraCalibration.field_to_image(xyz) -> uv
                      #   .rms_error ; .to_json() / .from_json()
  radar.py            # render top-down field (schematic from dimensions) + plot (x,y) dots
  store.py            # persist/load calibration per camera+event (JSON under web/uploads/)
data/field/
  2026-rebuilt-welded.json  # WPILib AprilTagFieldLayout (downloaded, canonical)
  2026_landmarks.json       # curated 3D landmark points extracted from the 3D model
  FIELD_REFERENCE.md        # verified frame/units reconciliation + CAD tag table
```

### Public surface (stable contract for SP2–SP4)

- `FieldModel.load(season) -> FieldModel`
- `FieldModel.tag_corners_3d(tag_id) -> np.ndarray  # (4, 3)`
- `FieldModel.alliance_side(point_xy) -> "blue" | "red"`
- `FieldModel.dimensions`
- `Calibrator.solve(correspondences, image_size, model="fisheye") -> CameraCalibration`
- `CameraCalibration.image_to_field(uv) -> (x, y)`
- `CameraCalibration.field_to_image(point_xyz) -> uv`  # point_xyz is a 3-vector
- `CameraCalibration.rms_error -> float`
- `CameraCalibration.to_json() / from_json()`
- `Radar.render(points_with_side) -> image`

## Calibration pipeline (detail)

Because the camera is static, calibration is a one-time solve per setup, and we
accumulate evidence across frames to do it well.

**Step A — Gather correspondences.** Detect AprilTags across a short window
(~30–60 frames). The camera does not move, but ball occlusion shifts frame to
frame, maximizing distinct tags and image coverage. Each tag → 4 image corners;
`field_model` supplies the 4 matching 3D corners. Manual-fallback clicks add
correspondences (a clicked tag corner, or a point placed on a top-down field
template → `(x, y, 0)`).

**Step B — Obtain intrinsics + distortion (once per camera, NOT per event).**
Fisheye intrinsics (`K` + `D`) **cannot be recovered from a single static
viewpoint**: `cv2.fisheye.calibrate` needs multiple genuinely distinct viewpoints to
separate focal length from distortion, and a fixed match camera over many frames is
still *one* viewpoint (occlusion only changes which tags are visible, not the
vantage). So intrinsics are acquired separately and treated as a per-camera-model
constant — via a one-time **multi-viewpoint** calibration (footage where the camera
actually moves, e.g. pre-match panning or a dedicated capture) fed to
`cv2.fisheye.calibrate`, **or imported** known lens values. Saved once and reused.

**Step C — Solve pose from tags (the per-event auto step).** Given the saved
intrinsics, undistort the detected tag corners with `cv2.fisheye.undistortPoints`,
then `cv2.solvePnP` against their known 3D coords → `rvec, tvec`. Pose is exactly
what changes between venues and solves robustly from a single frame's tag
detections. Static camera ⇒ refine / average over the window for stability.

**Step D — Projections.**
- `field_to_image(x,y,z)` → `cv2.fisheye.projectPoints`.
- `image_to_field(u,v)` → undistort pixel to a camera-frame ray, rotate into world
  coords via `R = Rodrigues(rvec)` and camera center `C = -Rᵀ·tvec`, then
  **intersect the ray with the carpet plane z=0** → `(x, y)`. Fed each robot's
  **bottom-center** pixel (ground contact) to minimize height parallax.

**Step E — Validate & gate.** Reproject all known tag corners; report **RMS
reprojection error (px)**; overlay the field perimeter + key lines on the sample
frame. Accept only if RMS is under the configured threshold; otherwise surface the
error and prompt for manual correspondences.

**Provisional thresholds** (confirm/tune during the plan, expose as config):
- Accept if RMS ≤ **5 px**; warn at 5–10 px; reject above **10 px**.
- Synthetic-test accuracy target: recovered field position within **10 cm** of
  ground truth (including edge points near the field walls).
- Minimum to attempt an **intrinsics** solve: **≥ 4 distinct tags** and **≥ 12
  correspondence points** spanning **≥ 3 of the 4 image quadrants**.
- Minimum to attempt a **pose** solve: **≥ 4 non-collinear** points.
- Below these, fail loudly (see degeneracy guards) rather than returning a
  low-confidence calibration.

### Why split intrinsics from pose

Intrinsics (the lens) barely change between events; pose (where the camera sits)
changes whenever the camera is bumped. Saving them separately means recalibrating a
nudged camera at a new venue is just re-running Step C against the *saved*
intrinsics — far fewer points needed, matching the intuition that a tag's
position is reliable even when its ID is unreadable.

## Using the team's 3D field model

The model is a **coordinate source, not a runtime dependency**:

1. **Augment `field_model`** with exact dimensions and a curated set of named 3D
   landmark points (field corners, tape-line intersections, hub/goal structure
   corners) → `data/field/<season>_landmarks.json`. These feed the manual fallback
   (click a hub corner when tags are unreadable) and validation (project the field
   wireframe onto the frame).
2. **Cross-check** the WPILib tag frame against the exact dimensions to confirm
   origin/axes agree before trusting auto-calibration.

The full mesh stays out of the pipeline (a future SP5 asset: occlusion reasoning, a
photoreal radar).

## Calibration UI, persistence & API

### Persistence (`store.py`)

One record per camera per event:
`{K, D, rvec, tvec, image_size, crop_region, field_season, rms_error, created_at}`,
JSON under `web/uploads/` (alongside existing event/diagnostic artifacts).
Intrinsics (`K, D`) are stored reusably so a nudged camera only re-solves pose.

### API (FastAPI — mirrors existing `/api/ocr-regions`, `/api/hub-positions`)

- `POST /api/calibration/auto` — grab sample frames, detect tags, solve; return RMS
  + overlay preview.
- `POST /api/calibration/manual` — submit clicked point↔field correspondences;
  re-solve **pose** against the saved intrinsics (`K, D`). Requires intrinsics to
  already exist (from a prior `auto` solve, or imported); if none are persisted,
  return a clear error instructing the user to run `auto` first.
- `GET  /api/calibration` — current calibration + status.
- `GET  /api/calibration/preview` — field-line overlay + radar image.

### UI (extends the existing "mark on a sample frame" setup flow)

A calibration panel showing the sample frame with detected tags highlighted, the
reprojected field lines overlaid, the live RMS error, and a click-to-add-point mode
for the manual fallback — with the radar beside it so projected robot dots can be
checked against reality immediately.

## Testing strategy

TDD per project skills — the synthetic test defines "correct" and is written first.

- **Synthetic round-trip (accuracy, no live video):** choose a known `K, D, R, t`;
  project the real field tag corners via `cv2.fisheye.projectPoints` to synthetic
  pixels; run calibration; assert recovered field positions are within tolerance
  (target: a few cm) and `field_to_image ∘ image_to_field ≈ identity`. This is the
  closed-loop accuracy guarantee — an overlay can hide systematic error; this
  cannot.
- **`field_model` units:** WPILib JSON loads; `tag_corners_3d` geometry;
  `alliance_side` boundary cases; cross-check against exact dimensions.
- **Degeneracy guards:** too-few-tags / collinear-points fail loudly with a clear
  error rather than returning a garbage calibration.
- **Real-frame smoke test:** one saved sample frame; assert tag detection finds the
  expected IDs and RMS is under the gate.

## Dependencies

- `pupil-apriltags` (36h11 detection) — new.
- `opencv-python` (`cv2.fisheye`, `solvePnP`) — already present.
- `numpy` — already present.
- `data/field/2026-rebuilt-welded.json` — WPILib layout, **already downloaded**.
- FRC 36h11 tag edge-length constant — **to confirm** from the 2026 manual.
- Curated `data/field/2026_landmarks.json` extracted from the team's 3D field
  model — to build (offline, from the model / `FIELD_REFERENCE.md`).

## Risks & mitigations

- **Tags too small/blurred to decode in a given venue feed** → manual-click fallback
  for pose; intrinsics can come from any frame/event with readable tags (lens is
  constant) and be reused.
- **Fisheye intrinsics cannot be derived from a static viewpoint** (the core
  geometric constraint) → intrinsics come from a one-time *multi-viewpoint*
  calibration (footage with real camera motion) or are imported, then reused; the
  per-event auto step solves **pose only** from tags. Do not attempt to recover
  `K, D` from a single match frame.
- **Acquiring real intrinsics for a camera we don't physically control** → use
  footage where the camera moves (pre-match panning, between-match repositioning) as
  the multi-viewpoint source, or import known lens values; gate pose solves on RMS
  and a minimum tag/coverage count.
- **Fisheye edge accuracy** (robots near the walls) → the fisheye model is chosen
  precisely for this; the synthetic test must include edge points and assert their
  tolerance.
- **Robot height parallax** → use bottom-center (ground contact); a known-height
  3D-point refinement is deferred to a later sub-project.

## Roadmap: where this fits

Sub-project 1 of 5. Build order 1 → 2 → 3 → 4 (each verifiable on its own), with 5
as future polish:

1. **Field-coordinate foundation (this spec)** — main cam: calibration + projection
   + diagnostic radar.
2. **Field-space tracking + alliance** — motion-model tracker; 6 stable tracks;
   alliance by side.
3. **Multi-camera fusion** — both corner cameras into the shared field space.
4. **Identity rebuild** — greenfield team-number assignment on stable field tracks.
5. **Polish / scoring / analytics** — photoreal radar, scoring-from-position,
   heatmaps.

Each sub-project gets its own spec → plan → implementation cycle.

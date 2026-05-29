# Tag-Anchored Field Calibration — Design

**Date:** 2026-05-29
**Status:** Approved (pending spec review)
**Supersedes (in UI):** ground-plane homography calibration page

## Problem

The current field-calibration page fits a ground-plane homography from hand-clicked
ground points (field corners, line intersections). On the real broadcast feed this
fails badly:

- The saved calibration is `mode: homography` with **RMS ≈ 475 px** — unusable.
- A homography is a plane→plane map needing ≥4 **ground** correspondences, but the
  only ground features (field corners) are off-screen — two of the four corners are
  cut off by the broadcast framing.

We need a calibration anchored to features that ARE visible: the AprilTags scattered
across the field interior.

## Key geometric facts (verified)

- The 2026 REBUILT field has 32 tags at **three height levels (0.55 / 0.89 / 1.12 m)**,
  spread across the full field (x 0.01–16.53, y 0.64–7.42). They are **not** on the
  ground plane, and they are **non-coplanar** in z.
- Because tags are elevated and non-coplanar, their pixel positions cannot feed a
  ground-plane homography (parallax), but they are exactly the right input for a full
  **camera resection** (intrinsics + distortion + pose).
- There are **no saved intrinsics** for the broadcast camera, and a single static feed
  cannot be multi-view fisheye-calibrated.
- **Empirical de-risking** (`scripts/calibrate_single_view_experiment.py` on real
  footage): single-view `cv2.calibrateCamera` over the non-coplanar tag corners
  converges to **RMS 0.86 px** (fx≈1194, fy≈1223, k1=−0.25, k2; principal point fixed
  at image center; pose solved). The detector auto-finds **only 4 tags** in the main
  crop — which is precisely why hand-labeling additional visible tags matters.
- Corner-order permutation `(0,1,2,3)` (identity) wins the reprojection search — the
  pupil_apriltags corner order aligns directly with `FieldModel.tag_corners_3d`. This
  resolves the standing 2D↔3D corner-order TODO.

> Note on "image_to_field offsets": ray-casting a tag's image center to z=0 lands
> 1–3 m past the tag's known (x,y). This is **expected parallax** (the tag is ~1 m up),
> not calibration error. Robots sit near the floor, so their pixels ray-cast correctly.
> The 0.86 px reprojection RMS is the real accuracy signal.

## Approach

Replace the homography page with **tag-anchored full-camera calibration**:

1. Grab a frame, cropped to the main-camera region (isolates one camera from the
   3-camera composite).
2. Auto-detect AprilTags and draw them on the frame.
3. Let the user hand-label additional visible-but-unreadable tags: click the tag in
   the frame, then click the matching tag marker on a top-down field map.
4. Solve a single-view pinhole + radial camera model (`cv2.calibrateCamera`) from all
   correspondences (auto tag corners + manual tag centers).
5. Save it so the existing `image_to_field` ray-casts pixels to the floor (z=0).
6. Show a reprojection overlay + color-coded RMS for verification.

### Why a camera model, not a homography

Tags are elevated and non-coplanar → enables a full camera resection, and the
resulting calibration ray-casts any pixel to the ground plane (z=0) for robot
tracking. A homography cannot consume elevated tag pixels without already knowing the
camera geometry (chicken-and-egg).

## Components

### 1. Solver — `web/field/calibration.py`

**New `Calibrator.solve_camera_from_tags(correspondences, image_size, field_season)`**

- `correspondences: {tag_id: (img_pts (N,2), obj_pts (N,3))}`. Auto-detected tags
  contribute their 4 corners (image corners ↔ `FieldModel.tag_corners_3d`). Manually
  labeled tags contribute 1 center point (clicked pixel ↔ `tag_translation`).
- Stack all correspondences into a single `cv2.calibrateCamera` view.
- Flags: `CALIB_USE_INTRINSIC_GUESS | CALIB_FIX_PRINCIPAL_POINT | CALIB_ZERO_TANGENT_DIST | CALIB_FIX_K3`.
- Initial K: `cx, cy` = image center; `fx = fy ≈ image_width` (reasonable broadcast
  guess). Returns `K, D (5,), rvec, tvec`.
- Returns a `CameraCalibration(mode="pinhole", K, D, rvec, tvec, image_size,
  field_season, rms_error)`.
- Raise `CalibrationError` with a clear message when total points are too few or
  `cv2.calibrateCamera` fails / the geometry is degenerate.

**New `mode="pinhole"` on `CameraCalibration`**

The existing `mode="camera"` uses the **fisheye** model (`cv2.fisheye.projectPoints` /
`cv2.fisheye.undistortPoints`, D shape 4). Single-view calibration produces a
**standard pinhole + radial** model (D shape 5) and must use `cv2.projectPoints` /
`cv2.undistortPoints`. Rather than overload the fisheye branch, add a distinct mode:

- `field_to_image`: pinhole branch → `cv2.projectPoints(obj, rvec, tvec, K, D)`.
- `image_to_field`: pinhole branch → `cv2.undistortPoints(pt, K, D, P=K)` → back out
  the normalized ray → intersect z=0. The ray-cast-to-ground math is identical to the
  camera branch and is shared (only the undistort call differs).
- `to_json` / `from_json` already serialize D as a flat list — works for 5 elements.
- The fisheye `mode="camera"` path is left untouched.

### 2. API — `web/app.py`

- **`POST /api/calibration/frame`** (extend): keep the crop to the main-camera region
  (`CALIB_MAIN_CROP_FRAC`) — it is essential to isolate one camera from the 3-camera
  composite. Add `detected_tags: {id: [[u,v]×4]}` to the response, filtered to tags
  present in the field model. Detection runs on the cropped frame so pixel coords match
  what the user sees and clicks.
- **`GET /api/calibration/tag-map`** (new): returns `{tags: {id: {x, y, z}},
  field: {length, width}}` for the season. The frontend renders the top-down map
  client-side so click hit-testing is precise.
- **`POST /api/calibration/solve-tags`** (new): body `{event, camera, image_size,
  auto_tags: {id: [[u,v]×4]}, manual_tags: [{id, u, v}]}`. Builds correspondences
  (auto → 4 corners via `tag_corners_3d`; manual → center via `tag_translation`),
  calls `solve_camera_from_tags`, saves via `field_store.save`, and returns
  `{status, rms_error, n_tags, n_points, overlay}` where `overlay` is a base64 JPEG of
  the same frame with the reprojected field perimeter + 2 m grid + all tag positions
  (green = used, orange = other).
- The homography solver and `/api/calibration/solve-homography` endpoint remain in the
  codebase but are no longer called by the page.

### 3. Frontend — rewrite `web/static/calibrate.html`

Two canvases side by side:

- **Left — broadcast frame**: drawn on grab; auto-detected tags rendered as green quads
  with ID labels.
- **Right — top-down field map**: every field tag drawn as a labeled marker
  (green = detected/used, grey = available). **Rendered rotated 180°** (flip both field
  axes: `screen_x ∝ L − field_x`, `screen_y ∝ W − field_y`) so the map's orientation
  roughly matches the camera's view, making click-to-click matching intuitive. The same
  transform is used for click hit-testing. Underlying tag (x,y) coordinates are
  unchanged — the flip is render-only.

**Manual labeling flow**: click a tag in the frame → a pending marker highlights →
click the matching tag marker on the top-down map → the pair is added to the
correspondence table. Repeat per tag. Manual rows have a delete button.

**Correspondence table**: lists auto tags (id, source=auto) and manual tags
(id, pixel, source=manual).

**Solve**: posts to `/api/calibration/solve-tags`; shows color-coded RMS and the
returned overlay image.

### Coordinate consistency

Detect, display, click, and solve all operate in the cropped main-cam frame coordinate
space (origin 0,0; size = crop w×h). The principal-point guess is the crop center.

### Corner order

Use identity `(0,1,2,3)` mapping between detector corners and
`FieldModel.tag_corners_3d`, validated by the 0.86 px solve. The `calibrateCamera` RMS
surfaces any regression.

## Error handling

- Require a minimum total of ~8 points (e.g. 2 auto tags, or a mix of auto + manual) —
  the camera model has ~10 free parameters (fx, fy, k1, k2, + 6 pose). Fewer → 400 with
  a clear "need more tags" message.
- `cv2.calibrateCamera` failure or degenerate (coplanar / collinear) geometry → 400
  with the reason. High RMS is surfaced (color-coded) in the UI rather than hidden.

## Testing

- **Unit** (`solve_camera_from_tags`): synthetic camera via `tests/_field_synth.py` —
  project known tag corners, solve, assert low RMS and recovered pose close to truth.
  Include a mixed corners + centers case.
- **Unit** (pinhole mode): `field_to_image ∘ image_to_field` round-trip on ground
  points (z=0) returns the input within tolerance.
- **API**: `/api/calibration/solve-tags` with synthetic correspondences returns ok +
  low RMS; too-few-points returns 400.
- **Manual / integration**: run the page on real footage; confirm RMS and that the
  overlay's field outline and tag reprojections line up.

## YAGNI cuts

- Single grabbed frame (no multi-frame tag accumulation in the UI — can be added later).
- No automatic tag-ID guessing for manual clicks (the user picks from the map).
- Homography solver code retained but removed from the UI.

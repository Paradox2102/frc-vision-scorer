# Tag-Anchored Field Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken ground-homography calibration page with a tag-anchored single-view camera resection: detect AprilTags, hand-label additional visible tags against a 180°-rotated top-down map, solve `cv2.calibrateCamera`, and save a calibration that ray-casts pixels to the floor.

**Architecture:** Tags sit at 0.55–1.12 m height and are non-coplanar, so their pixels feed a full camera model (intrinsics + radial distortion + pose), not a ground homography. A new `mode="pinhole"` on `CameraCalibration` does standard `cv2.projectPoints` / `cv2.undistortPoints` (the existing `"camera"` mode is fisheye and is left untouched). A new solver `solve_camera_from_tags` runs single-view `cv2.calibrateCamera` over auto tag corners + manual tag centers. New/extended API endpoints feed a rewritten two-canvas calibration page.

**Tech Stack:** Python, OpenCV (`cv2.calibrateCamera`, `cv2.projectPoints`, `cv2.undistortPoints`), FastAPI, `pupil_apriltags`, vanilla HTML/JS canvas.

**Spec:** `docs/superpowers/specs/2026-05-29-tag-anchored-calibration-design.md`

**REQUIRED SUB-SKILL for every task:** @superpowers:test-driven-development

**Test runner (this repo):** the base conda python lacks pytest — always use `.venv/bin/python -m pytest`.

---

## File Structure

- **Modify** `web/field/calibration.py` — add `mode="pinhole"` projection branch (shared ground-cast helper), add `Calibrator.solve_camera_from_tags`, add `MIN_CALIB_POINTS`.
- **Modify** `tests/_field_synth.py` — add a *pinhole* truth camera + `project_tag_centers` helper (the existing helpers are fisheye and can't validate a pinhole solver).
- **Create** `tests/test_calibration_pinhole.py` — pinhole round-trip + `solve_camera_from_tags` tests.
- **Modify** `web/app.py` — extend `/api/calibration/frame` (return detected tags), add `/api/calibration/tag-map` and `/api/calibration/solve-tags`, extract a shared `_draw_field_overlay` helper (used by both solve-homography and solve-tags).
- **Modify** `tests/test_calibration_api.py` — tests for tag-map, frame-detected-tags, solve-tags.
- **Rewrite** `web/static/calibrate.html` — two-canvas tag-labeling UI with 180°-rotated top-down map.
- **Create** `scripts/verify_tag_calibration.py` — end-to-end backend check on real footage.

---

## Task 1: `mode="pinhole"` projection on `CameraCalibration`

**Files:**
- Modify: `web/field/calibration.py` (`field_to_image`, `image_to_field`; add `_cast_norm_ray_to_ground`)
- Create: `tests/test_calibration_pinhole.py`

- [ ] **Step 1: Write the failing round-trip test**

Create `tests/test_calibration_pinhole.py`:

```python
import numpy as np
import pytest
from web.field.calibration import CameraCalibration, Calibrator, CalibrationError
from web.field.field_model import FieldModel
from tests._field_synth import (look_at, truth_pinhole_camera,
                                 project_tag_corners, in_frame_points)


def test_pinhole_field_image_roundtrip_on_ground():
    """field_to_image then image_to_field returns the original ground point."""
    K = np.array([[1200., 0., 960.], [0., 1200., 324.], [0., 0., 1.]])
    D = np.array([-0.25, -0.08, 0.0, 0.0, 0.0])           # k1,k2,p1,p2,k3
    rvec, tvec = look_at((8.27, -3.0, 6.0), (8.27, 4.03, 0.0))
    cam = CameraCalibration(mode="pinhole", K=K, D=D, rvec=rvec, tvec=tvec,
                            image_size=(1920, 648),
                            field_season="2026-rebuilt-welded")
    for x, y in [(3., 3.), (8.27, 4.), (13., 5.)]:
        u, v = cam.field_to_image((x, y, 0.0))
        xs, ys = cam.image_to_field((u, v))
        assert np.hypot(xs - x, ys - y) < 1e-6
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_pinhole.py::test_pinhole_field_image_roundtrip_on_ground -v`
Expected: FAIL — `truth_pinhole_camera` import error (added in Task 2) AND/OR pinhole branch missing. (If the import blocks collection, temporarily drop `truth_pinhole_camera, project_tag_corners, in_frame_points` from the import line for this step; re-add in Task 2.)

> Tip: to keep this step self-contained, import only `look_at` for now; the other names are used by Task 2 tests.

- [ ] **Step 3: Implement the pinhole branch + shared ground-cast helper**

In `web/field/calibration.py`, add a private helper to `CameraCalibration` (place after `image_to_field`):

```python
    def _cast_norm_ray_to_ground(self, norm) -> "tuple | None":
        """Normalized camera ray (x_n, y_n) -> field (x, y) where it hits z=0."""
        ray_cam = np.array([norm[0], norm[1], 1.0])
        R, _ = cv2.Rodrigues(self.rvec)
        cam_center = (-R.T @ self.tvec).reshape(3)
        ray_world = R.T @ ray_cam
        denom = ray_world[2]
        if abs(denom) < 1e-9:
            return None
        s = -cam_center[2] / denom
        if s <= 0:
            return None
        world = cam_center + s * ray_world
        return (float(world[0]), float(world[1]))
```

Also update the `mode` field comment in the dataclass (calibration.py:26) from
`# "camera" or "homography"` to `# "camera" (fisheye), "homography", or "pinhole"`.

In `field_to_image`, add the pinhole branch **before** the fisheye fall-through (after the homography branch):

```python
        if self.mode == "pinhole":
            obj = np.array([[point_xyz]], dtype=np.float64)
            D = np.asarray(self.D, float).reshape(-1, 1)
            img, _ = cv2.projectPoints(obj, self.rvec, self.tvec, self.K, D)
            u, v = img.reshape(2)
            return (float(u), float(v))
```

In `image_to_field`, add the pinhole branch (after the homography branch, before the fisheye code), and refactor the existing fisheye tail to reuse the helper:

```python
        if self.mode == "pinhole":
            pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
            D = np.asarray(self.D, float).reshape(-1, 1)
            norm = cv2.undistortPoints(pt, self.K, D).reshape(2)
            return self._cast_norm_ray_to_ground(norm)
        # fisheye camera mode:
        pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
        norm = cv2.fisheye.undistortPoints(pt, self.K, self.D).reshape(2)
        return self._cast_norm_ray_to_ground(norm)
```

(Delete the now-duplicated ray-cast lines that previously lived inline in the fisheye branch.)

- [ ] **Step 4: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_pinhole.py::test_pinhole_field_image_roundtrip_on_ground -v`
Expected: PASS (after temporarily importing only `look_at`).

- [ ] **Step 5: Confirm no regression in the fisheye path**

Run: `.venv/bin/python -m pytest tests/test_calibration_projection.py tests/test_calibration_solve.py -q`
Expected: PASS (the shared helper must not change fisheye behavior).

- [ ] **Step 6: Commit**

```bash
git add web/field/calibration.py tests/test_calibration_pinhole.py
git commit -m "feat(field): pinhole projection mode for single-view calibration"
```

---

## Task 2: `solve_camera_from_tags` + pinhole synthetic helper

**Files:**
- Modify: `tests/_field_synth.py`
- Modify: `web/field/calibration.py` (`Calibrator.solve_camera_from_tags`, `MIN_CALIB_POINTS`)
- Modify: `tests/test_calibration_pinhole.py`

- [ ] **Step 1: Add the pinhole synthetic helper**

Append to `tests/_field_synth.py`:

```python
# Pinhole truth camera for single-view calibration tests. Principal point MUST be
# at image center (cx=w/2, cy=h/2) because solve_camera_from_tags FIXES it there.
TRUTH_K_PINHOLE = np.array([[1200., 0., 960.], [0., 1200., 540.], [0., 0., 1.]])
TRUTH_D_PINHOLE = np.array([-0.25, -0.08, 0.0, 0.0, 0.0])   # k1,k2,p1,p2,k3 (mild radial)


def truth_pinhole_camera(eye=(8.27, -3.0, 6.0), target=(8.27, 4.03, 0.0),
                         K=TRUTH_K_PINHOLE, D=TRUTH_D_PINHOLE,
                         image_size=(1920, 1080)):
    """Pinhole camera above/behind the scoring table looking across the field.
    cx/cy are image-centered to match the solver's fixed principal point."""
    rvec, tvec = look_at(eye, target)
    return CameraCalibration(mode="pinhole", K=K.copy(), D=D.copy(),
                             rvec=rvec, tvec=tvec, image_size=image_size,
                             crop_region=(0, 0, image_size[0], image_size[1]),
                             field_season="2026-rebuilt-welded")
```

(The solver tests below build center correspondences inline, so no separate
`project_tag_centers` helper is needed.)

Now restore the full import line in `tests/test_calibration_pinhole.py` (it imports `truth_pinhole_camera, project_tag_corners, in_frame_points`).

- [ ] **Step 2: Write the failing solver tests**

Append to `tests/test_calibration_pinhole.py`:

```python
def test_solve_camera_from_tags_recovers_pinhole_camera():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)            # 4 corners per in-frame tag
    assert len(corr) >= 4
    solved = Calibrator.solve_camera_from_tags(
        corr, image_size=cam.image_size, field_season="2026-rebuilt-welded")
    assert solved.mode == "pinhole"
    assert solved.rms_error < 1.0                  # px
    assert solved.crop_region == (0, 0, cam.image_size[0], cam.image_size[1])
    for x, y in in_frame_points(cam, [(3., 3.), (8.27, 4.), (13., 5.), (6., 6.)]):
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = solved.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10     # 10 cm on the ground plane


def test_solve_camera_from_tags_mixed_corners_and_centers():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corners = project_tag_corners(cam, fm)
    mixed = {}
    for i, tid in enumerate(sorted(corners)):
        if i % 2 == 0:
            mixed[tid] = corners[tid]                       # 4 corners
        else:
            c3d = fm.tag_corners_3d(tid).mean(axis=0)
            uv = cam.field_to_image(tuple(c3d))
            mixed[tid] = (np.array([uv]), np.array([c3d]))  # 1 center
    solved = Calibrator.solve_camera_from_tags(
        mixed, image_size=cam.image_size, field_season="2026-rebuilt-welded")
    assert solved.rms_error < 2.0


def test_solve_camera_from_tags_too_few_points_raises():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    tid = sorted(fm.tags)[0]
    c3d = fm.tag_corners_3d(tid).mean(axis=0)
    one = {tid: (np.array([cam.field_to_image(tuple(c3d))]), np.array([c3d]))}
    with pytest.raises(CalibrationError):
        Calibrator.solve_camera_from_tags(
            one, image_size=cam.image_size, field_season="2026-rebuilt-welded")
```

- [ ] **Step 3: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_calibration_pinhole.py -k solve_camera -v`
Expected: FAIL — `AttributeError: ... has no attribute 'solve_camera_from_tags'`.

- [ ] **Step 4: Implement the solver**

In `web/field/calibration.py`, add a module constant near `MIN_POSE_TAGS`:

```python
MIN_CALIB_POINTS = 8       # single-view calibrateCamera: ~10 free params
```

Add to class `Calibrator`:

```python
    @staticmethod
    def solve_camera_from_tags(correspondences: dict, image_size: tuple,
                               field_season: str) -> "CameraCalibration":
        """Single-view full-camera resection from AprilTag correspondences.

        correspondences: {key: (img_pts (N,2), obj_pts (N,3))} — N=4 for auto-detected
        tag corners, N=1 for a manually-labeled tag center. Estimates a pinhole model
        (fx, fy + radial k1,k2) and pose via cv2.calibrateCamera over the non-coplanar
        tag points; principal point is fixed at the image center, tangential distortion
        and k3 are zeroed. Returns a mode='pinhole' calibration."""
        img_list, obj_list = [], []
        for (i2d, o3d) in correspondences.values():
            img_list.append(np.asarray(i2d, np.float64).reshape(-1, 2))
            obj_list.append(np.asarray(o3d, np.float64).reshape(-1, 3))
        if not img_list:
            raise CalibrationError("no correspondences supplied")
        img_all = np.vstack(img_list).astype(np.float32)
        obj_all = np.vstack(obj_list).astype(np.float32)
        n = len(img_all)
        if n < MIN_CALIB_POINTS:
            raise CalibrationError(
                f"need >= {MIN_CALIB_POINTS} points for single-view calibration, got "
                f"{n} (auto tags give 4 points each, manual tags 1 each)")
        w, h = int(image_size[0]), int(image_size[1])
        K0 = np.array([[float(w), 0., w / 2.0],
                       [0., float(w), h / 2.0],
                       [0., 0., 1.]])
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
                 cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
        try:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                [obj_all.reshape(-1, 1, 3)], [img_all.reshape(-1, 1, 2)],
                (w, h), K0.copy(), np.zeros(5), flags=flags)
        except cv2.error as e:
            raise CalibrationError(f"calibrateCamera failed: {str(e)[:120]}")
        return CameraCalibration(
            mode="pinhole", K=np.asarray(K, float),
            D=np.asarray(D, float).reshape(-1),
            rvec=np.asarray(rvecs[0], float).reshape(3, 1),
            tvec=np.asarray(tvecs[0], float).reshape(3, 1),
            image_size=(w, h), crop_region=(0, 0, w, h),
            field_season=field_season, rms_error=float(rms))
```

- [ ] **Step 5: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_pinhole.py -v`
Expected: PASS (all four tests).

- [ ] **Step 6: Commit**

```bash
git add web/field/calibration.py tests/_field_synth.py tests/test_calibration_pinhole.py
git commit -m "feat(field): solve_camera_from_tags single-view resection"
```

---

## Task 3: `/frame` returns detected tags + new `/tag-map` endpoint

**Files:**
- Modify: `web/app.py` (`calibration_frame` ~2384-2423; add `calibration_tag_map`)
- Modify: `tests/test_calibration_api.py`

- [ ] **Step 1: Write the failing API tests**

Append to `tests/test_calibration_api.py`:

```python
def test_tag_map_endpoint():
    r = _client().get("/api/calibration/tag-map")
    assert r.status_code == 200
    j = r.json()
    assert j["field"]["length"] > 0 and j["field"]["width"] > 0
    assert len(j["tags"]) >= 4
    some = next(iter(j["tags"].values()))
    assert {"x", "y", "z"} <= set(some)


def test_calibration_frame_includes_detected_tags():
    import os
    if not os.path.exists("data/test_footage/qual1_first_champs_2026.mp4"):
        import pytest; pytest.skip("test video not present")
    r = _client().post("/api/calibration/frame", json={"timestamp": 75})
    assert r.status_code == 200
    j = r.json()
    assert "detected_tags" in j        # dict; may be empty but key present
    for corners in j["detected_tags"].values():
        assert len(corners) == 4       # 4 (u,v) corners per tag
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_calibration_api.py -k "tag_map or detected_tags" -v`
Expected: FAIL — 404 for tag-map; `detected_tags` key missing.

- [ ] **Step 3: Add the `/tag-map` endpoint**

In `web/app.py`, after `calibration_landmarks` (~2484), add:

```python
@app.get("/api/calibration/tag-map")
async def calibration_tag_map(season: str = "2026-rebuilt-welded"):
    """Top-down field tag layout for the calibration UI: tag (x,y,z) + field dims."""
    fm = FieldModel.load(season)
    tags = {str(tid): {"x": float(t["t"][0]), "y": float(t["t"][1]),
                       "z": float(t["t"][2])}
            for tid, t in fm.tags.items()}
    return {"tags": tags, "field": {"length": fm.length_m, "width": fm.width_m}}
```

- [ ] **Step 4: Extend `/frame` to return detected tags**

In `calibration_frame`, replace the final encode/return block so detection runs on the crop:

```python
    main = frame[: int(h * CALIB_MAIN_CROP_FRAC), :]
    fm = FieldModel.load("2026-rebuilt-welded")
    det = detect_tags(main)
    detected = {str(tid): det[tid].tolist() for tid in det if tid in fm.tags}
    ok2, buf = cv2.imencode(".jpg", main, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok2:
        return JSONResponse(status_code=500, content={"error": "jpeg encode failed"})
    b64 = base64.b64encode(buf.tobytes()).decode()
    return {"frame": b64, "width": int(main.shape[1]), "height": int(main.shape[0]),
            "detected_tags": detected}
```

- [ ] **Step 5: Run to confirm pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_api.py -k "tag_map or detected_tags or frame" -v`
Expected: PASS (frame test may SKIP if video absent — that's fine).

- [ ] **Step 6: Commit**

```bash
git add web/app.py tests/test_calibration_api.py
git commit -m "feat(app): tag-map endpoint + detected tags in frame grab"
```

---

## Task 4: shared overlay helper + `/solve-tags` endpoint

**Files:**
- Modify: `web/app.py` (extract `_draw_field_overlay`; refactor `calibration_solve_homography`; add `calibration_solve_tags`)
- Modify: `tests/test_calibration_api.py`

- [ ] **Step 1: Write the failing solve-tags tests**

Append to `tests/test_calibration_api.py`:

```python
def test_solve_tags_recovers_and_saves(tmp_path, monkeypatch):
    import numpy as np
    from web.field import store as field_store
    from web.field.field_model import FieldModel
    from tests._field_synth import truth_pinhole_camera, project_tag_corners
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)

    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()                       # image_size (1920,1080)
    corr = project_tag_corners(cam, fm)
    auto = {str(tid): corr[tid][0].tolist() for tid in corr}

    r = _client().post("/api/calibration/solve-tags", json={
        "event": "evt_tags", "camera": "main", "season": "2026-rebuilt-welded",
        "image_size": list(cam.image_size), "auto_tags": auto, "manual_tags": []})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["rms_error"] < 2.0
    assert j["n_tags"] >= 4
    saved = field_store.load("evt_tags", "main")
    assert saved is not None and saved.mode == "pinhole"


def test_solve_tags_too_few_points(tmp_path, monkeypatch):
    from web.field import store as field_store
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)
    r = _client().post("/api/calibration/solve-tags", json={
        "image_size": [1920, 648], "auto_tags": {},
        "manual_tags": [{"id": 1, "u": 10, "v": 10}]})
    assert r.status_code == 400
    assert "point" in r.json().get("error", "").lower()
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_calibration_api.py -k solve_tags -v`
Expected: FAIL — 404 (endpoint missing).

- [ ] **Step 3: Extract the shared overlay helper**

In `web/app.py`, add above `calibration_solve_homography` (~2426):

```python
def _draw_field_overlay(frame, cal, fm):
    """Draw the reprojected field perimeter + 2 m grid using cal.field_to_image.
    Returns fp(x,y)->(int,int) so callers can add their own markers."""
    L, W = fm.length_m, fm.width_m
    def fp(x, y):
        u, v = cal.field_to_image((x, y, 0.0))
        return (int(u), int(v))
    cv2.polylines(frame, [np.array([fp(0, 0), fp(L, 0), fp(L, W), fp(0, W)],
                                   np.int32)], True, (0, 255, 0), 2)
    x = 0.0
    while x <= L + 1e-6:
        cv2.line(frame, fp(x, 0), fp(x, W), (0, 180, 0), 1); x += 2.0
    y = 0.0
    while y <= W + 1e-6:
        cv2.line(frame, fp(0, y), fp(L, y), (0, 180, 0), 1); y += 2.0
    return fp
```

Refactor the overlay block inside `calibration_solve_homography` to use it (replace the manual perimeter/grid drawing with `fp = _draw_field_overlay(main, cal, fm)`, keeping the clicked-point circles that follow).

- [ ] **Step 4: Add the `/solve-tags` endpoint**

Add after `calibration_solve_homography`:

```python
@app.post("/api/calibration/solve-tags")
async def calibration_solve_tags(request: Request):
    """Body: {event?, camera?, season?, image_size:[w,h], timestamp?,
    auto_tags:{id:[[u,v]x4]}, manual_tags:[{id,u,v}]}.
    Solves a single-view pinhole camera model from AprilTag correspondences
    (auto -> 4 corners, manual -> center), saves it, and returns RMS + overlay."""
    body = await request.json()
    event = body.get("event", "test"); camera = body.get("camera", "main")
    season = body.get("season", "2026-rebuilt-welded")
    image_size = tuple(body.get("image_size", [1920, 648]))
    fm = FieldModel.load(season)
    auto = body.get("auto_tags", {}) or {}
    manual = body.get("manual_tags", []) or []
    corr = {}
    for sid, corners in auto.items():
        tid = int(sid)
        if tid in fm.tags:
            corr[f"a{tid}"] = (np.array(corners, float).reshape(-1, 2),
                               fm.tag_corners_3d(tid))
    for m in manual:
        tid = int(m["id"])
        if tid in fm.tags:
            center3d = fm.tag_corners_3d(tid).mean(axis=0).reshape(1, 3)
            corr[f"m{tid}"] = (np.array([[float(m["u"]), float(m["v"])]], float),
                               center3d)
    try:
        cal = Calibrator.solve_camera_from_tags(corr, image_size=image_size,
                                                field_season=season)
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    field_store.save(event, camera, cal)

    used = {int(s) for s in auto.keys()} | {int(m["id"]) for m in manual}
    n_points = sum(len(c[0]) for c in corr.values())
    overlay_b64 = None
    ts = float(body.get("timestamp", 75))
    cap = cv2.VideoCapture(str(CALIB_TEST_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    okf, frame = cap.read(); cap.release()
    if okf and frame is not None:
        main = frame[: int(frame.shape[0] * CALIB_MAIN_CROP_FRAC), :]
        _draw_field_overlay(main, cal, fm)
        mh, mw = main.shape[:2]
        for tid in fm.tags:
            t3 = fm.tag_translation(tid)
            u, v = cal.field_to_image((float(t3[0]), float(t3[1]), float(t3[2])))
            if 0 <= u < mw and 0 <= v < mh:
                color = (0, 255, 0) if tid in used else (0, 165, 255)
                cv2.circle(main, (int(u), int(v)), 5, color, -1)
                cv2.putText(main, str(tid), (int(u) + 4, int(v) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        okj, buf = cv2.imencode(".jpg", main, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if okj:
            overlay_b64 = base64.b64encode(buf.tobytes()).decode()
    return {"status": "ok", "rms_error": cal.rms_error, "n_tags": len(corr),
            "n_points": n_points, "overlay": overlay_b64}
```

- [ ] **Step 5: Run to confirm pass + no regression**

Run: `.venv/bin/python -m pytest tests/test_calibration_api.py -v`
Expected: PASS (solve-homography tests still pass after the refactor; solve-tags tests pass).

- [ ] **Step 6: Commit**

```bash
git add web/app.py tests/test_calibration_api.py
git commit -m "feat(app): solve-tags endpoint + shared field-overlay helper"
```

---

## Task 5: Rewrite the calibration page (two-canvas tag labeling)

**Files:**
- Rewrite: `web/static/calibrate.html`

No automated test (static HTML/JS); verified manually in Task 6 and in the browser.

- [ ] **Step 1: Replace `web/static/calibrate.html` with the tag-labeling UI**

Write the full file:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Field Calibration (AprilTags)</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background:#1a1a2e; color:#e0e0e0; font-family:'Segoe UI',sans-serif;
         font-size:14px; padding:16px; }
  h1 { font-size:1.2em; color:#a0c4ff; margin-bottom:6px; }
  .instructions { color:#b0b8c8; margin-bottom:14px; font-size:0.9em;
                  max-width:1100px; line-height:1.5; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center;
              margin-bottom:14px; }
  .controls label { color:#a0b0c0; }
  input[type="number"], input[type="text"] { background:#16213e; border:1px solid #334;
    color:#e0e0e0; padding:5px 8px; border-radius:4px; font-size:13px; }
  input[type="number"] { width:80px; } input[type="text"] { width:100px; }
  button { background:#0f3460; color:#a0c4ff; border:1px solid #1a5080; padding:6px 14px;
    border-radius:4px; cursor:pointer; font-size:13px; transition:background .15s; }
  button:hover { background:#1a5080; }
  button.primary { background:#155030; color:#80ffb4; border-color:#1e7040; }
  button.primary:hover { background:#1e7040; }
  .canvases { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:14px; }
  .canvas-col h2 { font-size:.95em; color:#a0c4ff; margin-bottom:6px; }
  .canvas-wrap { position:relative; border:1px solid #334; display:inline-block;
                 background:#0a0a14; }
  #frameCanvas { cursor:crosshair; display:block; max-width:760px; }
  #mapCanvas   { cursor:pointer;   display:block; }
  .points-section { margin-bottom:14px; }
  .points-section h2 { font-size:1em; color:#a0c4ff; margin-bottom:8px; }
  table { border-collapse:collapse; width:100%; max-width:760px; background:#16213e; }
  th { background:#0f3460; color:#a0c4ff; padding:6px 10px; text-align:left;
       font-size:12px; white-space:nowrap; }
  td { border-bottom:1px solid #222840; padding:5px 8px; vertical-align:middle; }
  tr:hover td { background:#1a2540; }
  .del-btn { background:#3a0a0a; color:#ff8080; border-color:#6a1010; padding:3px 8px;
             font-size:12px; }
  .del-btn:hover { background:#5a1010; }
  .src-auto { color:#80ffb4; } .src-manual { color:#ffd060; }
  .solve-row { display:flex; align-items:center; gap:14px; margin-bottom:14px;
               flex-wrap:wrap; }
  #statusLine { font-size:13px; padding:4px 10px; border-radius:4px; background:#111;
                min-width:240px; }
  .rms-good{color:#60ff90;} .rms-warn{color:#ffd060;} .rms-bad{color:#ff6060;}
  .rms-info{color:#a0c4ff;} .rms-err{color:#ff8080;}
  .overlay-section h2 { font-size:1em; color:#a0c4ff; margin-bottom:8px; }
  #overlayImg { max-width:100%; border:1px solid #334; display:none; background:#0a0a14; }
  .hint { color:#7a86a0; font-size:12px; margin-top:4px; }
</style>
</head>
<body>

<h1>Field Calibration — AprilTags</h1>
<p class="instructions">
  Grab a frame. Detected tags are outlined in <span style="color:#80ffb4">green</span>.
  For a tag you can see but that wasn't detected: <b>click it on the left frame</b>, then
  <b>click the matching tag on the right map</b>. The map is rotated 180° to roughly match
  the camera's view. Add several spread-out tags, then Solve. Need &ge; 8 points
  (each detected tag = 4 points, each manual tag = 1).
</p>

<div class="controls">
  <label>Timestamp (s): <input type="number" id="timestamp" value="75" min="0" step="1"></label>
  <button onclick="grabFrame()">Grab frame</button>
  <label>Event: <input type="text" id="event" value="test"></label>
  <label>Camera: <input type="text" id="camera" value="main"></label>
</div>

<div class="canvases">
  <div class="canvas-col">
    <h2>Broadcast frame</h2>
    <div class="canvas-wrap"><canvas id="frameCanvas"></canvas></div>
    <div class="hint" id="frameHint">Click a tag here, then its marker on the map →</div>
  </div>
  <div class="canvas-col">
    <h2>Top-down tag map (rotated 180°)</h2>
    <div class="canvas-wrap"><canvas id="mapCanvas"></canvas></div>
    <div class="hint">Green = used · grey = available</div>
  </div>
</div>

<div class="points-section">
  <h2>Correspondences</h2>
  <table>
    <thead><tr><th>Tag</th><th>Source</th><th>Pixel (u,v)</th><th>Field (x,y)</th><th></th></tr></thead>
    <tbody id="rowsBody"><tr><td colspan="5" style="color:#556;font-style:italic;padding:12px 10px;">
      Grab a frame to begin.</td></tr></tbody>
  </table>
</div>

<div class="solve-row">
  <button class="primary" onclick="solve()">Solve &amp; preview</button>
  <span id="statusLine" class="rms-info">—</span>
</div>

<div class="overlay-section">
  <h2>Overlay (reprojected field + tags)</h2>
  <p class="hint" id="noOverlayMsg">Overlay appears here after a successful solve.</p>
  <img id="overlayImg" src="" alt="">
</div>

<script>
// ---- State ----
let tagMap = {};          // {id: {x,y,z}}
let field = {length: 16.541, width: 8.069};
let baseImg = null, frameW = 0, frameH = 0, fScale = 1.0;
let autoTags = {};        // {id: [[u,v]x4]}
let manualTags = [];      // [{id, u, v}]
let pendingPixel = null;  // {u, v} awaiting a map click
let mapScale = 1, mapW = 0, mapH = 0;
const MAP_PAD = 16;

(async function init(){ await fetchTagMap(); drawMap(); await grabFrame(); })();

async function fetchTagMap(){
  try {
    const r = await fetch('/api/calibration/tag-map');
    const j = await r.json();
    tagMap = j.tags || {}; field = j.field || field;
  } catch(e){ setStatus('Could not load tag map: '+e.message,'rms-err'); }
}

// ---- 180-rotated map transform (flip both field axes) ----
function fieldToMap(x, y){
  return [MAP_PAD + (field.length - x) * mapScale,
          MAP_PAD + (field.width  - y) * mapScale];
}
function drawMap(){
  const canvas = document.getElementById('mapCanvas');
  const targetW = 520;
  mapScale = targetW / field.length;
  mapW = targetW; mapH = field.width * mapScale;
  canvas.width = mapW + 2*MAP_PAD; canvas.height = mapH + 2*MAP_PAD;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0a0a14'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#445'; ctx.lineWidth = 1;
  ctx.strokeRect(...fieldToMap(field.length,field.width),
                 mapW * (0), 0); // placeholder; draw rect explicitly below
  // field rectangle (corners 0,0 and L,W under the flip)
  const c1 = fieldToMap(0,0), c2 = fieldToMap(field.length, field.width);
  ctx.strokeRect(Math.min(c1[0],c2[0]), Math.min(c1[1],c2[1]),
                 Math.abs(c2[0]-c1[0]), Math.abs(c2[1]-c1[1]));
  // midline
  const m1 = fieldToMap(field.length/2, 0), m2 = fieldToMap(field.length/2, field.width);
  ctx.beginPath(); ctx.moveTo(m1[0],m1[1]); ctx.lineTo(m2[0],m2[1]);
  ctx.strokeStyle='#334'; ctx.stroke();
  // tags
  const used = usedIds();
  for (const [id, t] of Object.entries(tagMap)){
    const [mx,my] = fieldToMap(t.x, t.y);
    const isUsed = used.has(parseInt(id));
    ctx.beginPath(); ctx.arc(mx,my,7,0,2*Math.PI);
    ctx.fillStyle = isUsed ? 'rgba(96,255,144,0.9)' : 'rgba(150,160,180,0.55)';
    ctx.fill(); ctx.strokeStyle='#0a0a14'; ctx.lineWidth=1; ctx.stroke();
    ctx.font='10px monospace'; ctx.fillStyle = isUsed ? '#d8ffe8' : '#aab';
    ctx.fillText(id, mx+8, my+3);
  }
  if (pendingPixel){
    ctx.fillStyle='#ffd060'; ctx.font='12px sans-serif';
    ctx.fillText('pick the tag for the pending frame click', MAP_PAD, 12);
  }
}
function usedIds(){
  const s = new Set(Object.keys(autoTags).map(Number));
  manualTags.forEach(m => s.add(m.id));
  return s;
}

// ---- Frame ----
async function grabFrame(){
  const ts = parseFloat(document.getElementById('timestamp').value) || 75;
  if (manualTags.length && !confirm('Grabbing a new frame clears manual tags. Continue?')) return;
  setStatus('Grabbing frame…','rms-info');
  try {
    const r = await fetch('/api/calibration/frame', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({timestamp:ts})});
    const j = await r.json();
    if (j.error){ setStatus('Frame error: '+j.error,'rms-err'); return; }
    autoTags = j.detected_tags || {}; manualTags = []; pendingPixel = null;
    const img = new Image();
    img.onload = function(){
      frameW = img.naturalWidth; frameH = img.naturalHeight;
      const canvas = document.getElementById('frameCanvas');
      const maxW = 760; fScale = Math.min(1, maxW / frameW);
      canvas.width = Math.round(frameW*fScale); canvas.height = Math.round(frameH*fScale);
      baseImg = img; redrawFrame(); drawMap(); renderRows();
      setStatus(`Frame grabbed. ${Object.keys(autoTags).length} tags auto-detected.`,'rms-info');
    };
    img.src = 'data:image/jpeg;base64,' + j.frame;
  } catch(e){ setStatus('Fetch error: '+e.message,'rms-err'); }
}

function redrawFrame(){
  const canvas = document.getElementById('frameCanvas');
  const ctx = canvas.getContext('2d');
  if (!baseImg) return;
  ctx.drawImage(baseImg,0,0,canvas.width,canvas.height);
  // auto tag quads
  ctx.lineWidth = 2;
  for (const [id, corners] of Object.entries(autoTags)){
    ctx.strokeStyle = 'rgba(96,255,144,0.95)'; ctx.beginPath();
    corners.forEach((c,i) => { const x=c[0]*fScale, y=c[1]*fScale;
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.closePath(); ctx.stroke();
    const cx = corners.reduce((a,c)=>a+c[0],0)/4*fScale;
    const cy = corners.reduce((a,c)=>a+c[1],0)/4*fScale;
    ctx.fillStyle='#80ffb4'; ctx.font='bold 12px monospace'; ctx.fillText(id, cx-6, cy+4);
  }
  // manual markers
  manualTags.forEach(m => {
    const x=m.u*fScale, y=m.v*fScale;
    ctx.beginPath(); ctx.arc(x,y,5,0,2*Math.PI);
    ctx.fillStyle='rgba(255,208,96,0.95)'; ctx.fill(); ctx.strokeStyle='#fff'; ctx.stroke();
    ctx.fillStyle='#ffd060'; ctx.font='bold 12px monospace'; ctx.fillText(m.id, x+7, y-6);
  });
  // pending
  if (pendingPixel){
    const x=pendingPixel.u*fScale, y=pendingPixel.v*fScale;
    ctx.beginPath(); ctx.arc(x,y,7,0,2*Math.PI);
    ctx.strokeStyle='#ffd060'; ctx.lineWidth=2; ctx.stroke();
  }
}

document.getElementById('frameCanvas').addEventListener('click', function(e){
  if (!baseImg) return;
  const rect = this.getBoundingClientRect();
  pendingPixel = {u: Math.round((e.clientX-rect.left)/fScale),
                  v: Math.round((e.clientY-rect.top)/fScale)};
  document.getElementById('frameHint').textContent =
    'Pending click set — now click the matching tag on the map →';
  redrawFrame(); drawMap();
});

document.getElementById('mapCanvas').addEventListener('click', function(e){
  const rect = this.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  // nearest tag marker within 14px
  let best=null, bestD=14*14;
  for (const [id,t] of Object.entries(tagMap)){
    const [tx,ty]=fieldToMap(t.x,t.y); const d=(tx-mx)**2+(ty-my)**2;
    if (d<bestD){ bestD=d; best=parseInt(id); }
  }
  if (best===null) return;
  if (!pendingPixel){ setStatus('Click a tag on the frame first.','rms-warn'); return; }
  if (best in autoTags){ setStatus(`Tag ${best} already auto-detected.`,'rms-warn');
    pendingPixel=null; redrawFrame(); drawMap(); return; }
  manualTags = manualTags.filter(m=>m.id!==best);
  manualTags.push({id:best, u:pendingPixel.u, v:pendingPixel.v});
  pendingPixel=null;
  document.getElementById('frameHint').textContent='Click a tag here, then its marker on the map →';
  redrawFrame(); drawMap(); renderRows();
  setStatus(`Added manual tag ${best}.`,'rms-info');
});

function renderRows(){
  const tb = document.getElementById('rowsBody'); tb.replaceChildren();
  const rows = [];
  Object.keys(autoTags).map(Number).sort((a,b)=>a-b).forEach(id => rows.push({id,src:'auto'}));
  manualTags.slice().sort((a,b)=>a.id-b.id).forEach(m => rows.push({id:m.id,src:'manual',u:m.u,v:m.v}));
  if (!rows.length){ const tr=document.createElement('tr'); const td=document.createElement('td');
    td.colSpan=5; td.textContent='No tags yet.'; td.style.cssText='color:#556;font-style:italic;padding:12px 10px;';
    tr.appendChild(td); tb.appendChild(tr); return; }
  rows.forEach(r => {
    const t = tagMap[r.id] || {x:'?',y:'?'};
    const tr = document.createElement('tr');
    const tdId=document.createElement('td'); tdId.textContent=r.id; tr.appendChild(tdId);
    const tdSrc=document.createElement('td'); tdSrc.textContent=r.src;
    tdSrc.className = r.src==='auto'?'src-auto':'src-manual'; tr.appendChild(tdSrc);
    const tdPx=document.createElement('td'); tdPx.style.fontFamily='monospace';
    tdPx.textContent = r.src==='manual' ? `(${r.u}, ${r.v})` : '4 corners'; tr.appendChild(tdPx);
    const tdXY=document.createElement('td'); tdXY.style.fontFamily='monospace';
    tdXY.textContent = `(${(+t.x).toFixed(2)}, ${(+t.y).toFixed(2)})`; tr.appendChild(tdXY);
    const tdDel=document.createElement('td');
    if (r.src==='manual'){ const b=document.createElement('button'); b.textContent='✕';
      b.className='del-btn'; b.onclick=()=>{ manualTags=manualTags.filter(m=>m.id!==r.id);
        redrawFrame(); drawMap(); renderRows(); }; tdDel.appendChild(b); }
    tr.appendChild(tdDel); tb.appendChild(tr);
  });
}

async function solve(){
  const nPoints = Object.keys(autoTags).length*4 + manualTags.length;
  if (nPoints < 8){ setStatus(`Need ≥ 8 points; have ${nPoints}. Add more tags.`,'rms-err'); return; }
  const event = document.getElementById('event').value.trim() || 'test';
  const camera = document.getElementById('camera').value.trim() || 'main';
  const ts = parseFloat(document.getElementById('timestamp').value) || 75;
  setStatus('Solving…','rms-info');
  try {
    const r = await fetch('/api/calibration/solve-tags',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({event,camera,timestamp:ts,image_size:[frameW,frameH],
        auto_tags:autoTags, manual_tags:manualTags})});
    const j = await r.json();
    if (j.error){ setStatus('Solve error: '+j.error,'rms-err'); return; }
    const rms = parseFloat(j.rms_error);
    let cls='rms-good'; if (rms>=20) cls='rms-bad'; else if (rms>=8) cls='rms-warn';
    setStatus(`RMS = ${rms.toFixed(2)} px  (${j.n_tags} tags, ${j.n_points} pts)`, cls);
    if (j.overlay){ const img=document.getElementById('overlayImg');
      img.src='data:image/jpeg;base64,'+j.overlay; img.style.display='block';
      document.getElementById('noOverlayMsg').style.display='none'; }
  } catch(e){ setStatus('Fetch error: '+e.message,'rms-err'); }
}

function setStatus(msg,cls){ const el=document.getElementById('statusLine');
  el.textContent=msg; el.className=cls||'rms-info'; }
</script>
</body>
</html>
```

> Note: in `drawMap()` the placeholder `strokeRect` line (the one with the comment "placeholder; draw rect explicitly below") is dead — delete it; the explicit `c1/c2` rectangle replaces it. Left in the listing only to flag it; remove during implementation.

- [ ] **Step 2: Clean up the dead placeholder line**

Remove the placeholder `ctx.strokeRect(...fieldToMap(field.length,field.width), mapW * (0), 0);` line noted above so only the explicit field rectangle is drawn.

- [ ] **Step 3: Lint-check the page loads (no automated assert)**

Run: `.venv/bin/python -c "import pathlib,html.parser as h; p=pathlib.Path('web/static/calibrate.html'); s=p.read_text(); assert 'solve-tags' in s and 'tag-map' in s and 'fieldToMap' in s; print('ok', len(s))"`
Expected: prints `ok <len>`.

- [ ] **Step 4: Commit**

```bash
git add web/static/calibrate.html
git commit -m "feat(ui): tag-anchored calibration page (auto-detect + click-to-map manual tags)"
```

---

## Task 6: End-to-end verification on real footage

**Files:**
- Create: `scripts/verify_tag_calibration.py`

- [ ] **Step 1: Write the verification script**

Create `scripts/verify_tag_calibration.py`:

```python
"""End-to-end check of the tag-anchored calibration backend on real footage.
Grabs a frame (with auto-detected tags), solves from those tags, prints RMS, and
writes the overlay to data/test_footage/tag_calib_overlay.jpg.
Run: .venv/bin/python scripts/verify_tag_calibration.py"""
import base64, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from web.app import app

VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
if not VIDEO.exists():
    print("test video not present — skipping"); sys.exit(0)

c = TestClient(app)
fr = c.post("/api/calibration/frame", json={"timestamp": 75}).json()
auto = fr.get("detected_tags", {})
print(f"auto-detected tags: {sorted(int(k) for k in auto)}  size={fr['width']}x{fr['height']}")
r = c.post("/api/calibration/solve-tags", json={
    "event": "verify", "camera": "main", "timestamp": 75,
    "image_size": [fr["width"], fr["height"]], "auto_tags": auto, "manual_tags": []})
j = r.json()
if j.get("error"):
    print("SOLVE ERROR:", j["error"]); sys.exit(1)
print(f"RMS = {j['rms_error']:.2f} px  ({j['n_tags']} tags, {j['n_points']} pts)")
if j.get("overlay"):
    out = ROOT / "data/test_footage/tag_calib_overlay.jpg"
    out.write_bytes(base64.b64decode(j["overlay"]))
    print("overlay written:", out)
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/verify_tag_calibration.py`
Expected: prints the auto-detected tag IDs (≈ `[6, 8, 17, 27]`), a finite RMS, and writes the overlay. With only the 4 auto tags the field-wide fit may be loose (that's expected — the UI exists to add more); the RMS should be a small number of px, not hundreds.

- [ ] **Step 3: Visually inspect the overlay**

Open `data/test_footage/tag_calib_overlay.jpg`; the green tag markers should land on the actual tags. (In the running app, adding several hand-labeled tags across the field tightens the perimeter fit.)

- [ ] **Step 4: Full calibration suite green**

Run: `.venv/bin/python -m pytest tests/ -k calibration -q`
Expected: PASS (skips allowed for video-dependent tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_tag_calibration.py
git commit -m "test(field): end-to-end tag-calibration verification script"
```

---

## Notes & gotchas

- **Principal point is fixed at image center.** The synthetic pinhole truth camera MUST use `cx=w/2, cy=h/2` or the solver (which fixes the principal point) can't reach <1 px RMS — this is why `TRUTH_K_PINHOLE` is image-centered.
- **Fisheye path untouched.** `mode="camera"` still uses `cv2.fisheye.*`; only the shared `_cast_norm_ray_to_ground` helper is factored out — Task 1 Step 5 guards against regression.
- **Coordinate space.** Everything (detect, display, click, solve) is in the cropped main-cam frame (origin 0,0). `solve_camera_from_tags` records `crop_region=(0,0,w,h)`; downstream `image_to_field` callers must pass cropped-frame pixels.
- **180° map flip is feed-specific** to this broadcast vantage (per spec); it's render-only and does not change stored coordinates.
- **Homography code retained.** `solve_ground_homography` and `/api/calibration/solve-homography` stay (and now share `_draw_field_overlay`); the page simply no longer calls them.

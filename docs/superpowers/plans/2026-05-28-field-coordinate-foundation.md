# Field-Coordinate Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate a full fisheye camera model for the main camera from AprilTags (with a manual fallback) and project per-frame robot detections onto a top-down field radar that demonstrably matches reality.

**Architecture:** A new isolated `web/field/` package owns field geometry: `field_model` (WPILib tag layout + dimensions + alliance side), `apriltag_detect` (36h11 detection), `calibration` (fisheye intrinsics via `cv2.fisheye.calibrate` + pose via `cv2.solvePnP`, plus `image_to_field`/`field_to_image`), `radar` (top-down render), and `store` (JSON persistence). FastAPI endpoints and a calibration page in `web/static/` drive it. All geometry is validated by a closed-loop **synthetic-camera** test before any live video is involved.

**Tech Stack:** Python 3.12, OpenCV (`cv2.fisheye`, `cv2.solvePnP`, `cv2.aruco` for test tag rendering), `pupil-apriltags`, NumPy, FastAPI, pytest (run via `.venv/bin/python -m pytest`).

**Spec:** `docs/superpowers/specs/2026-05-28-field-coordinate-foundation-design.md`

---

## Conventions (read before starting)

- **Test runner:** `.venv/bin/python -m pytest` (the base conda python has no pytest). Tests live flat in `tests/` as `test_*.py` (see `tests/conftest.py`).
- **Coordinate frame:** WPILib — meters; origin at the Blue-side corner; +x toward Red; +y across; +z up; carpet z=0. Field is 16.541 m × 8.069 m. Alliance midline at x = 8.2705 m.
- **Canonical data:** `data/field/2026-rebuilt-welded.json` (32 tags, full pose). See `data/field/FIELD_REFERENCE.md`.
- **Tag size constant:** the FRC 36h11 tag edge length is **not** in the JSON. Use a single module constant `TAG_SIZE_M` (provisional `0.1651` / 6.5 in — marked TODO to confirm from the 2026 manual). All synthetic tests are *self-consistent* and do not depend on this value being correct; only real-frame accuracy does.
- **WPILib tag frame:** a tag's pose rotates a tag whose default (identity) outward normal is field **+x**; the tag plane spans the tag-local **y** (left) and **z** (up) axes. A tag's 4 corners in tag-local coords are therefore `(0, ±s/2, ±s/2)`, rotated by the pose quaternion and translated to the pose position.
- **Corner ordering is the #1 correctness risk.** The 3D corner order from `field_model.tag_corners_3d()` MUST match the 2D corner order returned by `pupil_apriltags` for `solvePnP` to be correct. Task 3 determines the detector's order empirically; Task 2 matches it; **Task 5's pose solve (exact, runnable) is the authoritative validator** — if `image_to_field` is systematically mirrored/rotated, the corner order is wrong (fix the order, never the tolerance).
- **Robot detections:** supervision `Detections`; robot class id = `1` (`detections.class_id == 1`, boxes from `detections.xyxy`). The localization point is the **bottom-center** of the robot bbox: `((x1+x2)/2, y2)`.
- **Main-camera crop:** SP1 operates on the existing overhead crop (top `OVERHEAD_CROP_RATIO=0.67` of the frame; `web/app.py:108`). The crop region is stored with the calibration.
- **Commits:** one per task step group as shown. End commit messages with the Co-Authored-By trailer used in this repo. Work happens on branch `feature/field-coordinate-foundation`.

---

## File Structure

**Create:**
- `web/field/__init__.py` — package marker + public exports.
- `web/field/field_model.py` — load WPILib layout; dimensions; `tag_corners_3d`; `alliance_side`.
- `web/field/apriltag_detect.py` — detect 36h11 tags → `{id: (4,2) corners}`.
- `web/field/calibration.py` — `Calibrator.solve(...)`, `CameraCalibration` (projections, rms, (de)serialize).
- `web/field/radar.py` — top-down field render + point plotting.
- `web/field/store.py` — persist/load calibration JSON per camera+event.
- `tests/_field_synth.py` (shared synthetic-camera helper), `tests/test_field_model.py`, `tests/test_field_corners.py`, `tests/test_apriltag_detect.py`, `tests/test_calibration_projection.py`, `tests/test_calibration_solve.py`, `tests/test_calibration_intrinsics.py`, `tests/test_calibration_store.py`, `tests/test_radar.py`, `tests/test_calibration_api.py`
- `data/field/2026_landmarks.json` — curated manual-fallback landmarks (stub in this SP; populated from the 3D model later).

**Modify:**
- `pyproject.toml` — add `pupil-apriltags` dependency.
- `web/app.py` — add `/api/calibration/*` endpoints; wire per-frame robot field projection into the radar/diagnostic.
- `web/static/index.html`, `web/static/app.js`, `web/static/style.css` — calibration panel + radar preview.

---

## Task 0: Dependencies & package scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `web/field/__init__.py`

- [ ] **Step 1: Add dependency and install**

Add `"pupil-apriltags>=1.0.4"` to `pyproject.toml` `[project].dependencies`, then:

Run: `.venv/bin/pip install pupil-apriltags`
Expected: installs cleanly.

- [ ] **Step 2: Create package marker**

```python
# web/field/__init__.py
"""Field-coordinate foundation: calibration, projection, and radar."""
```

- [ ] **Step 3: Verify imports**

Run: `.venv/bin/python -c "import pupil_apriltags, cv2; assert hasattr(cv2,'fisheye'); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml web/field/__init__.py
git commit -m "chore(field): add pupil-apriltags + scaffold web/field package"
```

---

## Task 1: FieldModel — load layout, dimensions, alliance side

**Files:**
- Create: `web/field/field_model.py`
- Test: `tests/test_field_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_field_model.py
import math
import numpy as np
from web.field.field_model import FieldModel, TAG_SIZE_M

def test_loads_2026_layout():
    fm = FieldModel.load("2026-rebuilt-welded")
    assert len(fm.tags) == 32
    assert math.isclose(fm.length_m, 16.541, abs_tol=1e-3)
    assert math.isclose(fm.width_m, 8.069, abs_tol=1e-3)

def test_tag_pose_matches_known_value():
    fm = FieldModel.load("2026-rebuilt-welded")
    t = fm.tag_translation(1)
    np.testing.assert_allclose(t, [11.8779798, 7.4247756, 0.889], atol=1e-6)

def test_alliance_side_split_at_midline():
    fm = FieldModel.load("2026-rebuilt-welded")
    assert fm.alliance_side((1.0, 4.0)) == "blue"   # near x=0 (blue corner)
    assert fm.alliance_side((15.0, 4.0)) == "red"    # near far wall
    assert fm.alliance_side((fm.length_m / 2 - 0.01, 4.0)) == "blue"
    assert fm.alliance_side((fm.length_m / 2 + 0.01, 4.0)) == "red"

def test_unknown_tag_raises():
    fm = FieldModel.load("2026-rebuilt-welded")
    import pytest
    with pytest.raises(KeyError):
        fm.tag_translation(999)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_field_model.py -v`
Expected: FAIL (`ModuleNotFoundError: web.field.field_model`).

- [ ] **Step 3: Implement `field_model.py`**

```python
# web/field/field_model.py
"""Season field model: WPILib AprilTag layout + dimensions + alliance geometry."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# FRC 36h11 tag edge length (meters). TODO: confirm from 2026 game manual.
TAG_SIZE_M = 0.1651  # 6.5 in (provisional)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "field"


@dataclass
class FieldModel:
    tags: dict[int, dict]   # id -> {"t": (3,) array, "q": (w,x,y,z)}
    length_m: float
    width_m: float

    @classmethod
    def load(cls, season: str) -> "FieldModel":
        path = DATA_DIR / f"{season}.json"
        data = json.loads(path.read_text())
        tags = {}
        for tag in data["tags"]:
            tr = tag["pose"]["translation"]
            q = tag["pose"]["rotation"]["quaternion"]
            tags[int(tag["ID"])] = {
                "t": np.array([tr["x"], tr["y"], tr["z"]], dtype=float),
                "q": np.array([q["W"], q["X"], q["Y"], q["Z"]], dtype=float),
            }
        return cls(tags=tags, length_m=float(data["field"]["length"]),
                   width_m=float(data["field"]["width"]))

    def tag_translation(self, tag_id: int) -> np.ndarray:
        return self.tags[tag_id]["t"]

    def tag_quaternion(self, tag_id: int) -> np.ndarray:
        return self.tags[tag_id]["q"]

    def alliance_side(self, point_xy) -> str:
        return "blue" if point_xy[0] < self.length_m / 2 else "red"
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_field_model.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add web/field/field_model.py tests/test_field_model.py
git commit -m "feat(field): FieldModel loads 2026 layout, dims, alliance side"
```

---

## Task 2: Tag corner 3D geometry

Compute the four 3D corner coordinates of a tag from its pose + `TAG_SIZE_M`, in an order to be matched to the detector in Task 3. Uses the WPILib tag-frame convention (normal = +x, plane spans local y/z).

**Files:**
- Modify: `web/field/field_model.py`
- Test: `tests/test_field_corners.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_field_corners.py
import numpy as np
from web.field.field_model import FieldModel, TAG_SIZE_M

def test_corners_centered_on_translation():
    fm = FieldModel.load("2026-rebuilt-welded")
    c = fm.tag_corners_3d(1)            # (4,3)
    assert c.shape == (4, 3)
    np.testing.assert_allclose(c.mean(axis=0), fm.tag_translation(1), atol=1e-6)

def test_corners_lie_in_vertical_plane_of_size_s():
    # Each edge length equals TAG_SIZE_M; corners are coplanar.
    fm = FieldModel.load("2026-rebuilt-welded")
    c = fm.tag_corners_3d(1)
    edges = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    for e in edges:
        assert abs(e - TAG_SIZE_M) < 1e-6

def test_identity_rotation_spans_local_y_z():
    # A synthetic tag with identity quaternion: corners vary only in y and z.
    fm = FieldModel.load("2026-rebuilt-welded")
    fm.tags[900] = {"t": np.array([5.0, 4.0, 1.0]),
                    "q": np.array([1.0, 0.0, 0.0, 0.0])}
    c = fm.tag_corners_3d(900)
    assert np.allclose(c[:, 0], 5.0, atol=1e-9)        # x constant (normal axis)
    assert set(np.round(c[:, 1] - 4.0, 4)) == {-TAG_SIZE_M/2, TAG_SIZE_M/2}
    assert set(np.round(c[:, 2] - 1.0, 4)) == {-TAG_SIZE_M/2, TAG_SIZE_M/2}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_field_corners.py -v`
Expected: FAIL (`AttributeError: tag_corners_3d`).

- [ ] **Step 3: Implement quaternion rotation + corners**

Add to `field_model.py`:

```python
def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])

# method on FieldModel:
def tag_corners_3d(self, tag_id: int) -> np.ndarray:
    """4 corner positions (4,3) in field meters. Local corner order
    (provisional, reconciled with the detector in Task 3):
    [-y-z, +y-z, +y+z, -y+z] with normal along local +x."""
    s = TAG_SIZE_M / 2.0
    local = np.array([
        [0.0, -s, -s],
        [0.0,  s, -s],
        [0.0,  s,  s],
        [0.0, -s,  s],
    ])
    R = _quat_to_matrix(self.tag_quaternion(tag_id))
    return (R @ local.T).T + self.tag_translation(tag_id)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_field_corners.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/field_model.py tests/test_field_corners.py
git commit -m "feat(field): tag_corners_3d from pose + tag size (WPILib frame)"
```

---

## Task 3: AprilTag detection + corner-order reconciliation

Detect 36h11 tags and, critically, pin down the detector's corner ordering so Task 2's 3D order matches. Test images are rendered deterministically with `cv2.aruco` (DICT_APRILTAG_36h11) — no external fixtures.

**Files:**
- Create: `web/field/apriltag_detect.py`
- Test: `tests/test_apriltag_detect.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_apriltag_detect.py
import cv2
import numpy as np
from web.field.apriltag_detect import detect_tags

def _render_tag(tag_id: int, px: int = 240, canvas: int = 480) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(d, tag_id, px)
    img = np.full((canvas, canvas), 255, np.uint8)
    off = (canvas - px) // 2
    img[off:off+px, off:off+px] = marker
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def test_detects_rendered_tag():
    img = _render_tag(5)
    dets = detect_tags(img)
    assert 5 in dets
    assert dets[5].shape == (4, 2)

def test_corner_order_is_ccw_from_bottom_left():
    # Document & lock the detector's corner order against a centered, axis-aligned tag.
    img = _render_tag(5, px=240, canvas=480)
    corners = detect_tags(img)[5]
    cx, cy = corners.mean(axis=0)
    quad = [(1 if x > cx else 0, 1 if y > cy else 0) for x, y in corners]
    # Assert the observed order (fill in from Step 4 run); this test PINS it.
    assert quad == [(0, 1), (1, 1), (1, 0), (0, 0)]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_apriltag_detect.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement detector**

```python
# web/field/apriltag_detect.py
"""Detect 36h11 AprilTags -> {tag_id: (4,2) image corners}."""
from __future__ import annotations
import cv2
import numpy as np
from pupil_apriltags import Detector

_detector = Detector(families="tag36h11")

def detect_tags(frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    out: dict[int, np.ndarray] = {}
    for d in _detector.detect(gray):
        out[int(d.tag_id)] = np.asarray(d.corners, dtype=float)  # (4,2)
    return out
```

- [ ] **Step 4: Run, read the real corner order, reconcile**

Run: `.venv/bin/python -m pytest tests/test_apriltag_detect.py::test_detects_rendered_tag -v`
Expected: PASS.

Run the order probe:
`.venv/bin/python -c "from tests.test_apriltag_detect import _render_tag; from web.field.apriltag_detect import detect_tags; c=detect_tags(_render_tag(5))[5]; cx,cy=c.mean(0); print([(1 if x>cx else 0,1 if y>cy else 0) for x,y in c])"`

- Update `test_corner_order_is_ccw_from_bottom_left`'s expected `quad` to the printed value so it PINS the real order.
- Ensure `FieldModel.tag_corners_3d` local-corner order corresponds to the same geometric order (image +x ≈ field "tag local +y or -y", image +y down ≈ field "tag local -z"). Add a comment in `field_model.py` mapping detector order → local order; adjust the `local` array order if Task 5's solve shows a systematic flip (Task 5 is the real check).

- [ ] **Step 5: Run full file to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_apriltag_detect.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add web/field/apriltag_detect.py tests/test_apriltag_detect.py
git commit -m "feat(field): 36h11 detection + pinned corner ordering"
```

---

## Task 4: CameraCalibration projections (known camera)

Implement `field_to_image` and `image_to_field` for a calibration with known `K, D, rvec, tvec`. This is the closed-loop accuracy core — tested with a hand-set synthetic camera before any solving.

**Files:**
- Create: `web/field/calibration.py`
- Test: `tests/test_calibration_projection.py`

- [ ] **Step 1: Write failing tests**

First create a shared synthetic-camera helper (used by Tasks 4, 5, 5b, 7). It builds a *realistic* camera with a `look_at` so test points genuinely fall in-frame — no magic extrinsics.

```python
# tests/_field_synth.py
"""Shared synthetic fisheye camera for field-calibration tests."""
import cv2
import numpy as np
from web.field.calibration import CameraCalibration

IMAGE_SIZE = (1920, 1080)
TRUTH_K = np.array([[900., 0., 960.], [0., 900., 540.], [0., 0., 1.]])
TRUTH_D = np.array([-0.05, 0.01, 0.0, 0.0])      # mild fisheye

def look_at(eye, target, world_up=(0., 0., 1.)):
    """OpenCV-convention extrinsics: camera +z = forward, +x = right, +y = down."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(np.asarray(world_up, float), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])                      # world -> camera
    rvec, _ = cv2.Rodrigues(R)
    tvec = (-R @ eye).reshape(3, 1)
    return rvec, tvec

def truth_camera(eye=(8.27, -3.0, 6.0), target=(8.27, 4.03, 0.0),
                 K=TRUTH_K, D=TRUTH_D):
    """Camera behind/above the scoring table (y<0) looking across the field."""
    rvec, tvec = look_at(eye, target)
    return CameraCalibration(K=K.copy(), D=D.copy(), rvec=rvec, tvec=tvec,
                             image_size=IMAGE_SIZE, crop_region=None,
                             field_season="2026-rebuilt-welded")

def project_tag_corners(cam, fm):
    """{tag_id: (corners_2d (4,2), corners_3d (4,3))} for tags fully in-frame."""
    w, h = cam.image_size
    corr = {}
    for tid in fm.tags:
        c3d = fm.tag_corners_3d(tid)
        uv = np.array([cam.field_to_image(p) for p in c3d])
        if np.all((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)):
            corr[tid] = (uv, c3d)
    return corr

def in_frame_points(cam, candidates):
    w, h = cam.image_size
    keep = []
    for x, y in candidates:
        u, v = cam.field_to_image((x, y, 0.0))
        if 0 <= u < w and 0 <= v < h:
            keep.append((x, y))
    return keep
```

```python
# tests/test_calibration_projection.py
import numpy as np
from tests._field_synth import truth_camera, in_frame_points

def test_field_to_image_to_field_roundtrip():
    cam = truth_camera()
    pts = in_frame_points(cam, [(2.0, 2.0), (8.27, 4.0), (14.0, 2.0), (14.0, 6.0)])
    assert len(pts) >= 3                                  # honest: these are imaged
    for x, y in pts:
        uv = cam.field_to_image((x, y, 0.0))
        xy = cam.image_to_field(uv)
        assert np.linalg.norm(np.array(xy) - np.array([x, y])) < 0.01  # 1 cm

def test_image_to_field_lands_on_ground_plane():
    cam = truth_camera()
    uv = cam.field_to_image((6.0, 3.0, 0.0))
    x, y = cam.image_to_field(uv)
    assert abs(x - 6.0) < 0.01 and abs(y - 3.0) < 0.01
```

> The round-trip is pure math and holds regardless of image bounds; `in_frame_points` keeps the asserted points honestly inside the frame. Do not weaken the 1 cm tolerance.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_projection.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `CameraCalibration` projections**

```python
# web/field/calibration.py
"""Fisheye camera calibration + field<->image projection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class CameraCalibration:
    K: np.ndarray            # (3,3)
    D: np.ndarray            # (4,) fisheye
    rvec: np.ndarray         # (3,1)
    tvec: np.ndarray         # (3,1)
    image_size: tuple        # (w, h)
    crop_region: Optional[tuple]  # (x, y, w, h) of main-cam crop, or None
    field_season: str
    rms_error: float = 0.0

    def field_to_image(self, point_xyz) -> tuple:
        obj = np.array([[point_xyz]], dtype=np.float64)  # (1,1,3)
        img, _ = cv2.fisheye.projectPoints(obj, self.rvec, self.tvec, self.K, self.D)
        u, v = img.reshape(2)
        return (float(u), float(v))

    def image_to_field(self, uv) -> tuple:
        pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
        # Undistort to normalized camera-frame ray (z=1).
        norm = cv2.fisheye.undistortPoints(pt, self.K, self.D).reshape(2)
        ray_cam = np.array([norm[0], norm[1], 1.0])
        R, _ = cv2.Rodrigues(self.rvec)
        cam_center = (-R.T @ self.tvec).reshape(3)
        ray_world = R.T @ ray_cam
        # Intersect with ground plane z=0.
        s = -cam_center[2] / ray_world[2]
        world = cam_center + s * ray_world
        return (float(world[0]), float(world[1]))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_projection.py -v`
Expected: PASS. If round-trip fails, fix the projection math (do NOT relax the 1 cm tolerance).

- [ ] **Step 5: Commit**

```bash
git add web/field/calibration.py tests/test_calibration_projection.py
git commit -m "feat(field): fisheye field<->image projection (synthetic round-trip)"
```

---

## Task 5: Calibrator.solve_pose — auto-solve camera POSE from tags (headline accuracy)

The per-event auto step and the headline accuracy guarantee: given known intrinsics, project every in-frame tag corner through a synthetic ground-truth camera, solve **pose** from those correspondences, and assert the recovered camera localizes robots within 10 cm. (Intrinsics are NOT solved here — see Task 5b for why; pose is what changes per venue and solves exactly from a single frame.) This pose solve is also the real validator of the Task 2/3 corner ordering.

**Files:**
- Modify: `web/field/calibration.py`
- Test: `tests/test_calibration_solve.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibration_solve.py
import numpy as np
import pytest
from web.field.calibration import Calibrator, CalibrationError
from web.field.field_model import FieldModel
from tests._field_synth import truth_camera, project_tag_corners, in_frame_points

def test_solve_pose_recovers_camera_within_10cm():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_camera()
    corr = project_tag_corners(cam, fm)
    assert len(corr) >= 2
    solved = Calibrator.solve_pose(corr, K=cam.K, D=cam.D,
                                   image_size=(1920, 1080),
                                   field_season="2026-rebuilt-welded")
    assert solved.rms_error < 1.0  # px
    pts = in_frame_points(cam, [(2., 2.), (8.27, 4.), (14., 2.), (14., 6.)])
    assert len(pts) >= 3
    for x, y in pts:
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = solved.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10  # 10 cm

def test_too_few_tags_raises():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_camera()
    corr = project_tag_corners(cam, fm)
    one = dict(list(corr.items())[:1])  # a single tag (4 coplanar pts) is not enough
    with pytest.raises(CalibrationError):
        Calibrator.solve_pose(one, K=cam.K, D=cam.D, image_size=(1920, 1080),
                              field_season="2026-rebuilt-welded")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_solve.py -v`
Expected: FAIL (`ImportError: Calibrator`).

- [ ] **Step 3: Implement `Calibrator.solve_pose` + `CalibrationError`**

Add to `calibration.py`:

```python
class CalibrationError(Exception):
    pass


MIN_POSE_TAGS = 2          # >=2 tags so points are non-coplanar across the field


class Calibrator:
    @staticmethod
    def solve_pose(correspondences: dict, K, D, image_size: tuple,
                   field_season: str) -> "CameraCalibration":
        """Solve camera POSE against KNOWN intrinsics K, D (the per-event auto step).
        correspondences: {tag_id: (corners_2d (4,2), corners_3d (4,3))}."""
        if len(correspondences) < MIN_POSE_TAGS:
            raise CalibrationError(
                f"need >= {MIN_POSE_TAGS} tags for pose, got {len(correspondences)}")
        K = np.asarray(K, float)
        D = np.asarray(D, float).reshape(-1)
        obj = np.vstack([c3d for (_, c3d) in correspondences.values()]).astype(np.float64)
        img = np.vstack([c2d for (c2d, _) in correspondences.values()]).astype(np.float64)
        # Undistort to a pinhole frame (intrinsics K), then PnP.
        undist = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D, P=K)
        ok, rvec, tvec = cv2.solvePnP(obj, undist.reshape(-1, 1, 2), K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise CalibrationError("solvePnP failed")
        cal = CameraCalibration(K=K, D=D, rvec=rvec, tvec=tvec,
                                image_size=image_size, crop_region=None,
                                field_season=field_season)
        cal.rms_error = _reprojection_rms(cal, correspondences)
        return cal


def _reprojection_rms(cal: "CameraCalibration", correspondences: dict) -> float:
    errs = []
    for (c2d, c3d) in correspondences.values():
        for (u, v), p in zip(c2d, c3d):
            pu, pv = cal.field_to_image(tuple(p))
            errs.append((pu - u) ** 2 + (pv - v) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_solve.py -v`
Expected: PASS. If `image_to_field` is systematically off (e.g. mirrored/rotated), the tag corner order from `tag_corners_3d` (Task 2) does not match the detector order (Task 3) — fix the `local` corner order in `tag_corners_3d`, **not** the tolerance. This pose solve is exact, so it is the authoritative corner-order check.

- [ ] **Step 5: Commit**

```bash
git add web/field/calibration.py tests/test_calibration_solve.py
git commit -m "feat(field): Calibrator.solve_pose (PnP against known intrinsics) <10cm"
```

---

## Task 5b: Calibrator.calibrate_intrinsics — multi-viewpoint fisheye intrinsics

Fisheye intrinsics (`K, D`) **cannot** be recovered from a single static viewpoint — `cv2.fisheye.calibrate` needs multiple genuinely distinct vantages to separate focal length from distortion. This is a one-time, per-camera utility (fed real footage where the camera moves, or used to import/derive intrinsics offline); the per-event flow then uses `solve_pose` (Task 5). The synthetic test proves multi-viewpoint recovery works and documents the requirement.

**Files:**
- Modify: `web/field/calibration.py`
- Test: `tests/test_calibration_intrinsics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibration_intrinsics.py
import numpy as np
import pytest
from web.field.calibration import Calibrator, CalibrationError
from web.field.field_model import FieldModel
from tests._field_synth import truth_camera, TRUTH_K, IMAGE_SIZE

def _view_from(eye, fm):
    cam = truth_camera(eye=eye)
    w, h = IMAGE_SIZE
    obj, img = [], []
    for tid in fm.tags:
        c3d = fm.tag_corners_3d(tid)
        uv = np.array([cam.field_to_image(p) for p in c3d])
        if np.all((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)):
            obj.append(c3d); img.append(uv)
    return (np.vstack(obj), np.vstack(img)) if obj else None

def test_intrinsics_recovered_from_multiple_viewpoints():
    fm = FieldModel.load("2026-rebuilt-welded")
    eyes = [(8.27,-3,6), (4,-2,5), (12,-2,5), (8.27,-5,8), (2,-1,4), (14,-1,4)]
    views = [v for v in (_view_from(e, fm) for e in eyes) if v is not None]
    assert len(views) >= 4
    K, D = Calibrator.calibrate_intrinsics(views, IMAGE_SIZE)
    assert abs(K[0,0] - TRUTH_K[0,0]) < 30 and abs(K[1,1] - TRUTH_K[1,1]) < 30  # ~3%
    assert abs(K[0,2] - TRUTH_K[0,2]) < 20 and abs(K[1,2] - TRUTH_K[1,2]) < 20

def test_intrinsics_needs_enough_viewpoints():
    fm = FieldModel.load("2026-rebuilt-welded")
    one = [_view_from((8.27,-3,6), fm)]
    with pytest.raises(CalibrationError):
        Calibrator.calibrate_intrinsics(one, IMAGE_SIZE)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_intrinsics.py -v`
Expected: FAIL (`AttributeError: calibrate_intrinsics`).

- [ ] **Step 3: Implement `calibrate_intrinsics`**

Add to the `Calibrator` class in `calibration.py`:

```python
    MIN_INTRINSIC_VIEWS = 4

    @staticmethod
    def calibrate_intrinsics(views: list, image_size: tuple):
        """views: list of (obj_pts (N,3), img_pts (N,2)) from DISTINCT viewpoints.
        Returns (K, D). Requires multiple genuinely different camera vantages —
        a single static viewpoint is degenerate for fisheye intrinsics."""
        if len(views) < Calibrator.MIN_INTRINSIC_VIEWS:
            raise CalibrationError(
                f"fisheye intrinsics need >= {Calibrator.MIN_INTRINSIC_VIEWS} "
                f"distinct viewpoints, got {len(views)}")
        obj_arr = [o.reshape(-1, 1, 3).astype(np.float64) for (o, _) in views]
        img_arr = [i.reshape(-1, 1, 2).astype(np.float64) for (_, i) in views]
        K = np.zeros((3, 3)); D = np.zeros((4, 1))
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
        cv2.fisheye.calibrate(
            obj_arr, img_arr, image_size, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9))
        return K, D.reshape(-1)
```

> If `cv2.fisheye.calibrate` raises `CALIB_CHECK_COND`-related errors on real (noisy) data, drop that implicit check by keeping `CALIB_RECOMPUTE_EXTRINSIC` and adding `cv2.fisheye.CALIB_FIX_SKEW`; for stubborn real footage, fall back to importing known intrinsics. The synthetic (noise-free) test should converge cleanly.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_intrinsics.py -v`
Expected: PASS (intrinsics within tolerance; single-viewpoint raises). If recovery is loose, add more/again-more-varied viewpoints to `eyes` — do not loosen tolerances beyond ~3%.

- [ ] **Step 5: Commit**

```bash
git add web/field/calibration.py tests/test_calibration_intrinsics.py
git commit -m "feat(field): multi-viewpoint fisheye intrinsics calibration"
```

---

## Task 6: Serialization + persistence store

**Files:**
- Modify: `web/field/calibration.py` (add `to_json`/`from_json`)
- Create: `web/field/store.py`
- Test: `tests/test_calibration_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibration_store.py
import numpy as np
from web.field.calibration import CameraCalibration
from web.field import store

def _cal():
    return CameraCalibration(
        K=np.eye(3), D=np.zeros(4), rvec=np.zeros((3,1)), tvec=np.ones((3,1)),
        image_size=(1920,1080), crop_region=(0,0,1920,724),
        field_season="2026-rebuilt-welded", rms_error=2.5)

def test_json_roundtrip():
    c = _cal()
    c2 = CameraCalibration.from_json(c.to_json())
    np.testing.assert_allclose(c2.K, c.K)
    np.testing.assert_allclose(c2.tvec, c.tvec)
    assert c2.image_size == (1920,1080)
    assert c2.crop_region == (0,0,1920,724)
    assert c2.rms_error == 2.5

def test_store_save_load(tmp_path):
    store.UPLOAD_DIR = tmp_path
    store.save("evt1", "main", _cal())
    loaded = store.load("evt1", "main")
    assert loaded is not None
    np.testing.assert_allclose(loaded.tvec, np.ones((3,1)))
    assert store.load("evt1", "missing") is None

def test_load_intrinsics_only(tmp_path):
    store.UPLOAD_DIR = tmp_path
    store.save("evt1", "main", _cal())
    K, D = store.load_intrinsics("evt1", "main")
    np.testing.assert_allclose(K, np.eye(3))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_store.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `to_json`/`from_json` + `store.py`**

```python
# add to calibration.py
def to_json(self) -> dict:
    return {
        "K": self.K.tolist(), "D": np.asarray(self.D).reshape(-1).tolist(),
        "rvec": self.rvec.reshape(-1).tolist(), "tvec": self.tvec.reshape(-1).tolist(),
        "image_size": list(self.image_size),
        "crop_region": list(self.crop_region) if self.crop_region else None,
        "field_season": self.field_season, "rms_error": self.rms_error,
    }

@classmethod
def from_json(cls, d: dict) -> "CameraCalibration":
    return cls(
        K=np.array(d["K"], float), D=np.array(d["D"], float),
        rvec=np.array(d["rvec"], float).reshape(3,1),
        tvec=np.array(d["tvec"], float).reshape(3,1),
        image_size=tuple(d["image_size"]),
        crop_region=tuple(d["crop_region"]) if d["crop_region"] else None,
        field_season=d["field_season"], rms_error=d.get("rms_error", 0.0))
```

```python
# web/field/store.py
"""Persist camera calibrations as JSON under web/uploads/."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from web.field.calibration import CameraCalibration

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"

def _path(event_key: str, camera: str) -> Path:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"calibration_{safe}.json"

def save(event_key: str, camera: str, cal: CameraCalibration) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _path(event_key, camera).write_text(json.dumps(cal.to_json(), indent=2))

def load(event_key: str, camera: str) -> CameraCalibration | None:
    p = _path(event_key, camera)
    if not p.exists():
        return None
    return CameraCalibration.from_json(json.loads(p.read_text()))

def load_intrinsics(event_key: str, camera: str):
    cal = load(event_key, camera)
    return (None, None) if cal is None else (cal.K, np.asarray(cal.D).reshape(-1))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/calibration.py web/field/store.py tests/test_calibration_store.py
git commit -m "feat(field): calibration (de)serialization + JSON store"
```

---

## Task 7: Manual fallback — pose from sparse clicked points

The manual path uses the *same* `solve_pose`, but its correspondences come from a handful of human-clicked points (each bound to a known 3D field coordinate) rather than auto-detected tag corners. This test proves pose recovers from sparse, mixed points — the realistic blurred-tag fallback. (`solve_pose` accepts any `{key: (pts_2d, pts_3d)}` mapping, not only 4-corner tags.)

**Files:**
- Test: `tests/test_calibration_solve.py` (extend)

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_calibration_solve.py
def test_manual_pose_from_sparse_clicked_points():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_camera()
    # Simulate ~8 clicked points: tag centers (known 3D) projected to pixels.
    items = {}
    for i, tid in enumerate(t for t in fm.tags if t <= 28):
        c3d = fm.tag_corners_3d(tid).mean(axis=0)          # tag center (3,)
        uv = cam.field_to_image(tuple(c3d))
        w, h = cam.image_size
        if 0 <= uv[0] < w and 0 <= uv[1] < h:
            items[f"pt{i}"] = (np.array([uv]), np.array([c3d]))   # (1,2),(1,3)
        if len(items) >= 8:
            break
    pose = Calibrator.solve_pose(items, K=cam.K, D=cam.D,
                                 image_size=(1920, 1080),
                                 field_season="2026-rebuilt-welded")
    for x, y in in_frame_points(cam, [(3., 3.), (13., 5.)]):
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = pose.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_solve.py::test_manual_pose_from_sparse_clicked_points -v`
Expected: PASS — confirms `solve_pose` works on sparse non-tag-corner correspondences (the manual fallback). If it fails because points are coplanar/few, ensure ≥6 well-spread points.

- [ ] **Step 3: Commit**

```bash
git add tests/test_calibration_solve.py
git commit -m "test(field): manual pose from sparse clicked points"
```

---

## Task 8: Radar render

**Files:**
- Create: `web/field/radar.py`
- Test: `tests/test_radar.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_radar.py
import numpy as np
from web.field.radar import Radar

def test_render_returns_image_of_configured_size():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    img = r.render([])
    assert img.shape == (round(8.069*40), round(16.541*40), 3)

def test_point_maps_to_expected_pixel():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    # field (0,0) -> top-left-ish origin mapping; check a known interior point.
    px = r.field_to_pixel((8.2705, 4.0345))  # field center
    assert abs(px[0] - 16.541*40/2) <= 1
    assert abs(px[1] - 8.069*40/2) <= 1

def test_side_tint_blue_left_red_right():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    img = r.render([((2.0, 4.0), "blue"), (14.0, 4.0) if False else ((14.0, 4.0), "red")])
    assert img is not None  # smoke: renders without error
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_radar.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `radar.py`**

```python
# web/field/radar.py
"""Top-down field radar: render field + plot (x,y) detections."""
from __future__ import annotations
import cv2
import numpy as np

_TINT = {"blue": (255, 120, 0), "red": (0, 0, 255), None: (180, 180, 180)}

class Radar:
    def __init__(self, field_length: float, field_width: float, px_per_m: float = 40):
        self.L, self.W, self.s = field_length, field_width, px_per_m
        self.w = round(field_length * px_per_m)
        self.h = round(field_width * px_per_m)

    def field_to_pixel(self, xy) -> tuple:
        return (xy[0] * self.s, xy[1] * self.s)

    def render(self, points) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 30, np.uint8)
        cv2.line(img, (self.w // 2, 0), (self.w // 2, self.h), (90, 90, 90), 1)
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (90, 90, 90), 1)
        for xy, side in points:
            u, v = self.field_to_pixel(xy)
            cv2.circle(img, (int(u), int(v)), 6, _TINT.get(side), -1)
        return img
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_radar.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/radar.py tests/test_radar.py
git commit -m "feat(field): top-down radar render + field->pixel mapping"
```

---

## Task 9: FastAPI calibration endpoints

Mirror the existing `/api/hub-positions` + `/api/grab-frame` patterns (`web/app.py:2175`, `:2197`). Persist via `web.field.store`.

**Files:**
- Modify: `web/app.py` (add endpoints near the other setup endpoints, ~line 2197)
- Test: `tests/test_calibration_api.py`

- [ ] **Step 1: Write failing tests** (use FastAPI `TestClient`)

```python
# tests/test_calibration_api.py
from fastapi.testclient import TestClient
import importlib

def _client():
    app_mod = importlib.import_module("web.app")
    return TestClient(app_mod.app)

def test_get_calibration_empty():
    r = _client().get("/api/calibration", params={"event": "test_evt", "camera": "main"})
    assert r.status_code == 200
    assert r.json()["has_calibration"] in (False, True)

def test_manual_without_intrinsics_returns_clear_error():
    r = _client().post("/api/calibration/manual", json={
        "event": "no_such_evt", "camera": "main", "correspondences": []})
    assert r.status_code == 400
    assert "auto" in r.json().get("error", "").lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_api.py -v`
Expected: FAIL (404 / endpoint missing).

- [ ] **Step 3: Implement endpoints**

Add to `web/app.py`. **Imports must be fully qualified** — `web/app.py` imports its own modules as `from web import reid`, `from web.embedder import ...`, `from web.diagnostic import ...` (verified: there are no bare-name sibling imports). Use the same form:

```python
from web.field import store as field_store
from web.field.field_model import FieldModel
from web.field.calibration import Calibrator, CalibrationError
from web.field.apriltag_detect import detect_tags
```

```python
@app.get("/api/calibration")
async def get_calibration(event: str, camera: str = "main"):
    cal = field_store.load(event, camera)
    return {"has_calibration": cal is not None,
            "rms_error": cal.rms_error if cal else None}

@app.post("/api/calibration/manual")
async def calibration_manual(request: Request):
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    K, D = field_store.load_intrinsics(event, camera)
    if K is None:
        return JSONResponse(status_code=400, content={
            "error": "No saved intrinsics — run auto calibration first."})
    # ... build correspondences {key:(pts_2d, pts_3d)} from clicked points +
    #     FieldModel; call Calibrator.solve_pose(corr, K, D, image_size, season);
    #     field_store.save; return {"rms_error": cal.rms_error}.
    return {"status": "ok"}

@app.post("/api/calibration/auto")
async def calibration_auto(request: Request):
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    K, D = field_store.load_intrinsics(event, camera)
    if K is None:
        return JSONResponse(status_code=400, content={
            "error": "No intrinsics — import or run multi-viewpoint "
                     "intrinsics calibration (Task 5b) first."})
    # ... grab N frames (reuse grab-frame mechanics), detect_tags across them,
    #     build correspondences via FieldModel.tag_corners_3d (match detector
    #     corner order), Calibrator.solve_pose(corr, K, D, ...), field_store.save,
    #     return {rms_error, n_tags, overlay_b64}.
    return {"status": "ok"}
```

> The `auto` endpoint solves **pose** against existing intrinsics (per Task 5 — fisheye intrinsics are not derivable from one static viewpoint). Intrinsics come from Task 5b (multi-viewpoint) or an import step. Keep the *solve* logic in `web/field/` (already tested); these endpoints only marshal frames/points in and JSON out.

> The `auto`/`manual` bodies are the integration glue; keep the *solve* logic in `web/field/` (already tested). These endpoints only marshal frames/points in and JSON out. Keep them thin.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_api.py -v`
Expected: PASS (the two wiring tests). Full auto/manual behavior is verified manually in Task 10.

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_calibration_api.py
git commit -m "feat(app): calibration API endpoints (get/auto/manual)"
```

---

## Task 10: Calibration UI + radar preview (manual verification)

**Files:**
- Modify: `web/static/index.html`, `web/static/app.js`, `web/static/style.css`

- [ ] **Step 1: Add a "Field Calibration" panel** to the setup flow: sample-frame display, "Auto-calibrate" button (calls `/api/calibration/auto`), detected-tag highlights + reprojected field-line overlay, live RMS readout, a click-to-add-point mode posting to `/api/calibration/manual`, and the radar image beside it.

- [ ] **Step 2: Manual verification checklist** (run the app, document results in the PR):

Run: `.venv/bin/python web/app.py` (serves on `:8004`), then in the browser:
- Load a sample frame; click Auto-calibrate; confirm RMS ≤ 5 px and the overlaid field lines visually hug the real field edges.
- Confirm projected robot dots on the radar land where the robots actually are (spot-check 2–3 robots).
- Force the manual path (e.g. a frame with no decodable tags); click ≥4 field points; confirm pose re-solves and the overlay realigns.

- [ ] **Step 3: Commit**

```bash
git add web/static/
git commit -m "feat(ui): field calibration panel + radar preview"
```

---

## Task 11: Wire per-frame robot field projection into the radar/diagnostic

**Files:**
- Modify: `web/app.py` (in the detection loop where `robot_mask = detections.class_id == 1`, ~`:1753`)

- [ ] **Step 1: Load calibration at job start**

When an inference job starts, `field_store.load(event_key, "main")`; if present, build a `Radar` from the `FieldModel` dimensions; if absent, skip projection (log once).

- [ ] **Step 2: Project robot bottom-centers each frame**

For each robot detection box `(x1,y1,x2,y2)`, compute bottom-center `((x1+x2)/2, y2)`, map crop→full-frame coords if needed, `image_to_field`, and `alliance_side` for the tint. Collect `[((x,y), side), ...]`.

- [ ] **Step 3: Emit radar frame to the existing diagnostic/WebSocket channel**

Reuse the diagnostic WebSocket (`_diagnostic_ws_send`, `web/app.py:808`) to send a `radar` message (base64 of `Radar.render(points)`), best-effort (never block inference). No track persistence — raw per-frame dots only (SP1 scope).

- [ ] **Step 4: Manual verification**

Run a short clip with a saved calibration; confirm the live radar shows moving dots roughly matching robot positions and correct field sides.

- [ ] **Step 5: Commit**

```bash
git add web/app.py
git commit -m "feat(app): project per-frame robot detections onto live radar"
```

---

## Final verification

- [ ] Run the whole field suite:

Run: `.venv/bin/python -m pytest tests/test_field_model.py tests/test_field_corners.py tests/test_apriltag_detect.py tests/test_calibration_projection.py tests/test_calibration_solve.py tests/test_calibration_intrinsics.py tests/test_calibration_store.py tests/test_radar.py tests/test_calibration_api.py -v`
Expected: all PASS.

- [ ] Confirm the synthetic pose-solve accuracy test (`test_solve_pose_recovers_camera_within_10cm`) passes — this is the closed-loop guarantee that the per-event geometry is correct — and `test_intrinsics_recovered_from_multiple_viewpoints` (proves multi-viewpoint intrinsics work).
- [ ] Confirm no regression in the existing suite:

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: no new failures (pre-existing skips for missing sample images are fine).

## Deferred to later sub-projects (do NOT build here)

Track persistence / motion model (SP2), corner cameras + fusion (SP3), team-number identity (SP4), scoring/analytics (SP5). The real-world `TAG_SIZE_M` value and the curated `2026_landmarks.json` are confirmed/populated when real footage is first calibrated.

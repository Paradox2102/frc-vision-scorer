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
- **Corner ordering is the #1 correctness risk.** The 3D corner order from `field_model.tag_corners_3d()` MUST match the 2D corner order returned by `pupil_apriltags` for `solvePnP` to be correct. Task 3 determines the detector's order empirically; Task 2 matches it.
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
- `tests/test_field_model.py`, `tests/test_field_corners.py`, `tests/test_apriltag_detect.py`, `tests/test_calibration_projection.py`, `tests/test_calibration_solve.py`, `tests/test_calibration_store.py`, `tests/test_radar.py`, `tests/test_calibration_api.py`
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

```python
# tests/test_calibration_projection.py
import cv2
import numpy as np
from web.field.calibration import CameraCalibration

def _synthetic_camera():
    # Camera behind/above the scoring table looking across the field.
    K = np.array([[900., 0., 960.], [0., 900., 540.], [0., 0., 1.]])
    D = np.array([-0.05, 0.01, 0.0, 0.0])          # mild fisheye
    # Place camera ~5 m above, ~4 m back from a field point, looking down-forward.
    rvec, _ = cv2.Rodrigues(cv2.Rodrigues(np.array([[1,0,0],[0,-1,0],[0,0,-1]], float))[0])
    tvec = np.array([[-8.0], [-4.0], [9.0]])
    return CameraCalibration(K=K, D=D, rvec=rvec.reshape(3,1), tvec=tvec,
                             image_size=(1920, 1080), crop_region=None,
                             field_season="2026-rebuilt-welded")

def test_field_to_image_to_field_roundtrip():
    cam = _synthetic_camera()
    pts = [(2.0, 2.0), (8.27, 4.0), (14.0, 1.0), (14.0, 7.0)]  # incl. edges
    for x, y in pts:
        uv = cam.field_to_image((x, y, 0.0))
        xy = cam.image_to_field(uv)
        assert np.linalg.norm(np.array(xy) - np.array([x, y])) < 0.01  # 1 cm

def test_image_to_field_lands_on_ground_plane():
    cam = _synthetic_camera()
    uv = cam.field_to_image((6.0, 3.0, 0.0))
    x, y = cam.image_to_field(uv)
    assert abs(x - 6.0) < 0.01 and abs(y - 3.0) < 0.01
```

> Note: the worker may need to adjust the synthetic extrinsics in Step 1 so the test points project within the image; the *assertions* (round-trip < 1 cm) must not be weakened.

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

## Task 5: Calibrator.solve — recover the camera from correspondences

The headline accuracy guarantee: build a synthetic ground-truth camera, project every real tag corner through it, solve from those correspondences, and assert the recovered camera localizes robots within 10 cm (including field-edge points).

**Files:**
- Modify: `web/field/calibration.py`
- Test: `tests/test_calibration_solve.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibration_solve.py
import cv2
import numpy as np
import pytest
from web.field.calibration import CameraCalibration, Calibrator, CalibrationError
from web.field.field_model import FieldModel

def _truth():
    K = np.array([[900., 0., 960.], [0., 900., 540.], [0., 0., 1.]])
    D = np.array([-0.05, 0.01, 0.0, 0.0])
    R = np.array([[1,0,0],[0,-1,0],[0,0,-1]], float)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([[-8.0], [-4.0], [9.0]])
    return CameraCalibration(K, D, rvec, tvec, (1920,1080), None, "2026-rebuilt-welded")

def _project_all_tag_corners(cam, fm):
    """Build correspondences {tag_id: (corners_2d (4,2), corners_3d (4,3))} for
    tags whose projection lands inside the image."""
    corr = {}
    for tid in fm.tags:
        c3d = fm.tag_corners_3d(tid)
        uv = np.array([cam.field_to_image(p) for p in c3d])
        if np.all((uv[:,0] >= 0) & (uv[:,0] < 1920) & (uv[:,1] >= 0) & (uv[:,1] < 1080)):
            corr[tid] = (uv, c3d)
    return corr

def test_solve_recovers_camera_within_10cm():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = _truth()
    corr = _project_all_tag_corners(cam, fm)
    assert len(corr) >= 4
    solved = Calibrator.solve(corr, image_size=(1920,1080),
                              field_season="2026-rebuilt-welded")
    assert solved.rms_error < 1.0  # px
    for x, y in [(2.,2.), (8.27,4.), (14.,1.), (14.,7.)]:
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = solved.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10  # 10 cm

def test_too_few_tags_raises():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = _truth()
    corr = _project_all_tag_corners(cam, fm)
    few = dict(list(corr.items())[:1])
    with pytest.raises(CalibrationError):
        Calibrator.solve(few, image_size=(1920,1080), field_season="2026-rebuilt-welded")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_solve.py -v`
Expected: FAIL (`ImportError: Calibrator`).

- [ ] **Step 3: Implement `Calibrator` + `CalibrationError`**

Add to `calibration.py`:

```python
class CalibrationError(Exception):
    pass


MIN_TAGS = 4
MIN_POINTS = 12


class Calibrator:
    @staticmethod
    def solve(correspondences: dict, image_size: tuple,
              field_season: str, K: np.ndarray | None = None,
              D: np.ndarray | None = None) -> "CameraCalibration":
        """correspondences: {tag_id: (corners_2d (4,2), corners_3d (4,3))}.
        If K/D are provided, solve pose only (manual fallback); else full solve."""
        obj_pts = [c3d.astype(np.float64) for (_, c3d) in correspondences.values()]
        img_pts = [c2d.astype(np.float64) for (c2d, _) in correspondences.values()]
        n_pts = sum(len(p) for p in img_pts)

        if K is None or D is None:
            if len(correspondences) < MIN_TAGS or n_pts < MIN_POINTS:
                raise CalibrationError(
                    f"need >= {MIN_TAGS} tags / {MIN_POINTS} points, "
                    f"got {len(correspondences)} / {n_pts}")
            K = np.zeros((3, 3)); D = np.zeros((4, 1))
            obj_arr = [p.reshape(-1, 1, 3) for p in obj_pts]
            img_arr = [p.reshape(-1, 1, 2) for p in img_pts]
            flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                     cv2.fisheye.CALIB_FIX_SKEW)
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                obj_arr, img_arr, image_size, K, D, flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))

        # Pose via solvePnP on undistorted points (pinhole after undistortion).
        all_obj = np.vstack(obj_pts)
        all_img = np.vstack(img_pts).reshape(-1, 1, 2)
        undist = cv2.fisheye.undistortPoints(all_img, K, D, P=K).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            all_obj, undist, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise CalibrationError("solvePnP failed")

        cal = CameraCalibration(K=np.asarray(K), D=np.asarray(D).reshape(-1),
                                rvec=rvec, tvec=tvec, image_size=image_size,
                                crop_region=None, field_season=field_season)
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
Expected: PASS. If `image_to_field` is systematically off, the tag corner order (Task 2/3) is likely flipped — fix the `local` corner order in `tag_corners_3d`, not the tolerance.

- [ ] **Step 5: Commit**

```bash
git add web/field/calibration.py tests/test_calibration_solve.py
git commit -m "feat(field): Calibrator.solve (fisheye intrinsics + PnP pose) <10cm"
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

## Task 7: Manual fallback — pose-only solve against saved intrinsics

**Files:**
- Test: `tests/test_calibration_solve.py` (extend)

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_calibration_solve.py
def test_manual_pose_solve_with_given_intrinsics():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = _truth()
    corr = _project_all_tag_corners(cam, fm)
    # Use a handful of manually-clicked-style points; reuse known K, D.
    pose_only = Calibrator.solve(corr, image_size=(1920,1080),
                                 field_season="2026-rebuilt-welded",
                                 K=cam.K, D=cam.D)
    for x, y in [(3.,3.), (13.,5.)]:
        uv = cam.field_to_image((x, y, 0.0))
        xs, ys = pose_only.image_to_field(uv)
        assert np.hypot(xs - x, ys - y) < 0.10
```

- [ ] **Step 2: Run to verify it fails (or passes if already supported)**

Run: `.venv/bin/python -m pytest tests/test_calibration_solve.py::test_manual_pose_solve_with_given_intrinsics -v`
Expected: PASS if Task 5's `K/D` branch is correct (this test confirms the manual path). If it fails, fix the given-intrinsics branch.

- [ ] **Step 3: Commit**

```bash
git add tests/test_calibration_solve.py
git commit -m "test(field): manual pose-only solve against saved intrinsics"
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

Add to `web/app.py` (import `from field import store as field_store`, `from field.field_model import FieldModel`, `from field.calibration import Calibrator, CalibrationError`, `from field.apriltag_detect import detect_tags` — match the existing import style, which imports sibling modules like `database`, `reid` by bare name):

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
    # ... build correspondences from clicked points + FieldModel, solve pose-only ...
    # (worker: assemble {tag_id:(c2d,c3d)} or point/field pairs; call Calibrator.solve
    #  with K, D; save via field_store.save; return rms_error.)
    return {"status": "ok"}

@app.post("/api/calibration/auto")
async def calibration_auto(request: Request):
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    # ... grab N frames (reuse grab-frame mechanics), detect_tags across them,
    #     build correspondences via FieldModel.tag_corners_3d, Calibrator.solve,
    #     field_store.save, return {rms_error, n_tags, overlay_b64}.
    return {"status": "ok"}
```

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

Run: `.venv/bin/python -m pytest tests/test_field_model.py tests/test_field_corners.py tests/test_apriltag_detect.py tests/test_calibration_projection.py tests/test_calibration_solve.py tests/test_calibration_store.py tests/test_radar.py tests/test_calibration_api.py -v`
Expected: all PASS.

- [ ] Confirm the synthetic-solve accuracy test (`test_solve_recovers_camera_within_10cm`) passes — this is the closed-loop guarantee that the geometry is correct.
- [ ] Confirm no regression in the existing suite:

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: no new failures (pre-existing skips for missing sample images are fine).

## Deferred to later sub-projects (do NOT build here)

Track persistence / motion model (SP2), corner cameras + fusion (SP3), team-number identity (SP4), scoring/analytics (SP5). The real-world `TAG_SIZE_M` value and the curated `2026_landmarks.json` are confirmed/populated when real footage is first calibrated.

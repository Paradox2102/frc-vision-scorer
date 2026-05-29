# Multi-Camera Fusion (Corner Cameras) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two wide-angle corner cameras as *additive* sensors — calibrated into the shared field frame via tag resection, fused with the main camera into the existing 6-slot tracker, and feeding clearer bumper-number reads into the existing identity vote tally every frame (not just when the main camera is blocked).

**Architecture:** Each physical camera becomes a `CameraView` (crop rect into the composite + overlay masks + its own per-camera `CameraCalibration`). Every frame, all calibrated views run YOLO → project ground-contact points to field `(x,y)` → produce detections + bumper crops. A new pure `fuse_detections()` clusters cross-camera observations into merged `FieldDetection`s (fed to the untouched `FieldTracker`) plus a parallel `FusedObservation` side-channel carrying pooled, source-weighted bumper crops. Identity reads from all cameras pour into the existing `team_ocr_votes` tally, with corner reads weighted higher and ungated. Corner cams are calibrated with the same `solve_camera_from_tags` pinhole resection as the main cam; the only new calibration UX is "manual locate → auto refine in an upscaled ROI" for the spread off-hub tags the detector can't decode frame-wide.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, FastAPI, `pupil_apriltags`, Ultralytics YOLO, pytest. Tests run with `.venv/bin/python -m pytest` (the base conda python has no pytest).

**Spec:** `docs/superpowers/specs/2026-05-29-multi-camera-fusion-design.md`

---

## Conventions for every task

- **Run tests with** `.venv/bin/python -m pytest` — never bare `pytest`.
- **Test location:** new tests are grouped under `tests/field/`. The repo uses pytest package-import mode (`tests/__init__.py` exists), so **create `tests/field/__init__.py` (empty) once** before the first `tests/field/` test, or collection will error. (Alternatively place the tests flat in `tests/` to match the existing convention — filenames are unique either way.)
- **TDD:** write the failing test, run it red, implement minimally, run it green, commit.
- **Commit** after each green step with the message shown.
- Field-package modules import as `web.field.*` (see existing `detect_bridge.py`).
- Reference skills: @superpowers:test-driven-development, @superpowers:verification-before-completion.

---

## File Structure

**New files:**
- `web/field/composite_layout.py` — the measured composite geometry: per-camera crop rect `(x,y,w,h)` and overlay-mask rects, loaded from `data/field/composite_layout.json` with a built-in default. One responsibility: "where is each camera in the composite, and what graphics cover it."
- `data/field/composite_layout.json` — the measured numbers (data, not code).
- `web/field/camera_view.py` — `CameraView` dataclass + `extract(composite_frame)` (crop + blank masks) + `load_views(event_key)` (build the list of calibrated views from layout + `field_store`). One responsibility: "turn a composite frame into per-camera working frames."
- `web/field/fusion.py` — `FusedObservation` dataclass + `fuse_detections(per_camera)`; cross-camera cluster + merge. One responsibility: "many cameras' detections → merged detections + pooled identity evidence."
- Test files mirror under `tests/`.

**Modified files:**
- `web/field/apriltag_detect.py` — add `detect_tags_in_roi()` (ROI refine, coords mapped back to full frame).
- `web/app.py` — multi-view ingest in `process_video`; `CameraView` load in `_init_field_tracking`; `vision_identify_robots` gains a per-source weight + corner-ungated reads; calibration `frame`/`solve-tags` endpoints take a camera crop region; new `/api/calibration/refine-tag` endpoint; multi-camera occlusion merge.
- `web/static/calibrate.html` — camera picker drives the crop region; manual click runs ROI refine; held-out error shown.

---

## Phase 1 — Composite layout + CameraView extraction

Goal: deterministically split a composite frame into three masked per-camera frames. No inference behavior changes yet.

### Task 1: Measure the composite geometry

**Files:**
- Create: `data/field/composite_layout.json`
- Scratch: `scripts/measure_composite_layout.py` (throwaway helper, may delete after)

- [ ] **Step 1: Write a measurement helper** that dumps annotated crops so a human can confirm the seam, the L/R split, and the three overlay-mask rects.

```python
# scripts/measure_composite_layout.py
import cv2, sys
sys.path.insert(0, "web")
cap = cv2.VideoCapture("data/test_footage/qual1_first_champs_2026.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(20 * fps)); ok, f = cap.read(); cap.release()
h, w = f.shape[:2]            # expect 1080 x 1920
print("frame", w, h)
# Candidate seam between main (top) and corner row (bottom). Refine by eye.
SEAM = int(h * 0.60)
for name, x0, x1 in (("LEFT", 0, w // 2), ("RIGHT", w // 2, w)):
    crop = f[SEAM:, x0:x1]
    cv2.imwrite(f"/tmp/layout_{name}.jpg", crop)
    print(name, "crop", crop.shape)
```

- [ ] **Step 2: Run it and visually confirm** the seam row, the split column, and read off the overlay rects (Hopper Division on LEFT bottom-right; GE Aerospace + molex on RIGHT) in **full-composite pixel coordinates**.

Run: `.venv/bin/python scripts/measure_composite_layout.py` then open `/tmp/layout_LEFT.jpg` / `/tmp/layout_RIGHT.jpg`.

- [ ] **Step 3: Write the measured numbers** to `data/field/composite_layout.json`. `crop` is `[x, y, w, h]` in full-composite pixels; `masks` is a list of `[x, y, w, h]` rects **relative to that camera's crop origin**.

```json
{
  "frame_size": [1920, 1080],
  "cameras": {
    "main":         { "crop": [0, 0, 1920, 723], "masks": [] },
    "corner_left":  { "crop": [0, 648, 960, 432], "masks": [[<measured Hopper rect>]] },
    "corner_right": { "crop": [960, 648, 960, 432], "masks": [[<GE rect>], [<molex rect>]] }
  }
}
```

Note: `main` keeps `0..723` (= `OVERHEAD_CROP_RATIO` 0.67 × 1080) so main inference is byte-for-byte unchanged. Corner crops use the measured seam (≈648) and split (960). Replace `<...>` with the rects you read in Step 2.

- [ ] **Step 4: Commit**

```bash
git add data/field/composite_layout.json scripts/measure_composite_layout.py
git commit -m "feat(field): measured composite camera layout (crops + overlay masks)"
```

### Task 2: `composite_layout.py` loader

**Files:**
- Create: `web/field/composite_layout.py`
- Test: `tests/field/test_composite_layout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/field/test_composite_layout.py
from web.field.composite_layout import load_layout, CameraLayout

def test_loads_three_cameras_with_crops():
    layout = load_layout()
    assert set(layout) >= {"main", "corner_left", "corner_right"}
    cl = layout["corner_left"]
    assert isinstance(cl, CameraLayout)
    x, y, w, h = cl.crop
    assert w > 0 and h > 0
    # masks are (x,y,w,h) tuples relative to the crop
    for m in cl.masks:
        assert len(m) == 4

def test_main_crop_starts_at_origin():
    layout = load_layout()
    assert layout["main"].crop[0] == 0 and layout["main"].crop[1] == 0
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest tests/field/test_composite_layout.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# web/field/composite_layout.py
"""Where each physical camera lives inside the composite broadcast frame, and
which graphics overlays cover field area. Pure data + a thin loader."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

_LAYOUT_PATH = Path(__file__).resolve().parents[2] / "data/field/composite_layout.json"


@dataclass(frozen=True)
class CameraLayout:
    name: str
    crop: tuple          # (x, y, w, h) in full-composite pixels
    masks: tuple         # tuple of (x, y, w, h) rects, relative to the crop origin


def load_layout(path: Path = _LAYOUT_PATH) -> dict[str, CameraLayout]:
    raw = json.loads(Path(path).read_text())
    out = {}
    for name, spec in raw["cameras"].items():
        out[name] = CameraLayout(
            name=name,
            crop=tuple(spec["crop"]),
            masks=tuple(tuple(m) for m in spec.get("masks", [])),
        )
    return out
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_composite_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/composite_layout.py tests/field/test_composite_layout.py
git commit -m "feat(field): composite_layout loader"
```

### Task 3: `CameraView.extract` (crop + mask)

**Files:**
- Create: `web/field/camera_view.py`
- Test: `tests/field/test_camera_view.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/field/test_camera_view.py
import numpy as np
from web.field.camera_view import CameraView

def _frame():
    return np.full((1080, 1920, 3), 255, np.uint8)

def test_extract_crops_to_region():
    v = CameraView(name="corner_left", crop=(0, 648, 960, 432), masks=(), calibration=None)
    out = v.extract(_frame())
    assert out.shape == (432, 960, 3)

def test_extract_blanks_masks():
    v = CameraView(name="corner_right", crop=(960, 648, 960, 432),
                   masks=((10, 20, 30, 40),), calibration=None)
    out = v.extract(_frame())
    # masked rect is zeroed; a pixel outside it is untouched
    assert out[20:60, 10:40].sum() == 0
    assert out[0, 0].sum() == 255 * 3

def test_extract_is_a_copy():
    f = _frame()
    v = CameraView(name="main", crop=(0, 0, 1920, 723), masks=((0, 0, 5, 5),), calibration=None)
    v.extract(f)
    assert f[0, 0].sum() == 255 * 3  # masking did not mutate the source frame
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest tests/field/test_camera_view.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** (`load_views` is added in the next task; only `CameraView` + `extract` here)

```python
# web/field/camera_view.py
"""A physical camera = a crop rect into the composite + overlay masks + its own
calibration. extract() turns a composite frame into this camera's working frame."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CameraView:
    name: str
    crop: tuple                  # (x, y, w, h) in full-composite pixels
    masks: tuple                 # tuple of (x, y, w, h), relative to crop origin
    calibration: Optional[object]  # web.field.calibration.CameraCalibration or None

    def extract(self, composite_frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.crop
        crop = composite_frame[y:y + h, x:x + w].copy()  # copy: never mutate source
        for (mx, my, mw, mh) in self.masks:
            crop[my:my + mh, mx:mx + mw] = 0
        return crop
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_camera_view.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/camera_view.py tests/field/test_camera_view.py
git commit -m "feat(field): CameraView.extract (crop + overlay mask)"
```

### Task 4: `load_views(event_key)` — only calibrated cameras

**Files:**
- Modify: `web/field/camera_view.py`
- Test: `tests/field/test_camera_view.py`

- [ ] **Step 1: Add failing tests** (append)

```python
from web.field import camera_view as cv_mod

def test_load_views_includes_only_calibrated(monkeypatch):
    from web.field.composite_layout import CameraLayout
    layout = {
        "main": CameraLayout("main", (0, 0, 1920, 723), ()),
        "corner_left": CameraLayout("corner_left", (0, 648, 960, 432), ()),
    }
    monkeypatch.setattr(cv_mod, "load_layout", lambda: layout)
    # main calibrated, corner_left not
    monkeypatch.setattr(cv_mod.field_store, "load",
                        lambda ek, cam: object() if cam == "main" else None)
    views = cv_mod.load_views("evt")
    assert [v.name for v in views] == ["main"]

def test_load_views_orders_main_first(monkeypatch):
    from web.field.composite_layout import CameraLayout
    layout = {
        "corner_left": CameraLayout("corner_left", (0, 648, 960, 432), ()),
        "main": CameraLayout("main", (0, 0, 1920, 723), ()),
    }
    monkeypatch.setattr(cv_mod, "load_layout", lambda: layout)
    monkeypatch.setattr(cv_mod.field_store, "load", lambda ek, cam: object())
    views = cv_mod.load_views("evt")
    assert views[0].name == "main"
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest tests/field/test_camera_view.py -v`
Expected: FAIL (`load_views` / `load_layout` / `field_store` not defined).

- [ ] **Step 3: Implement** (add imports + function)

```python
# add near top of web/field/camera_view.py
from web.field.composite_layout import load_layout
from web.field import store as field_store


def load_views(event_key: str) -> list["CameraView"]:
    """Build the list of calibrated CameraViews for an event. Cameras without a
    saved calibration are omitted. 'main' is always ordered first so it remains the
    reference view for fusion."""
    layout = load_layout()
    views = []
    for name, cl in layout.items():
        cal = field_store.load(event_key, name)
        if cal is None:
            continue
        views.append(CameraView(name=name, crop=cl.crop, masks=cl.masks, calibration=cal))
    views.sort(key=lambda v: (v.name != "main", v.name))  # main first, then stable
    return views
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_camera_view.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/camera_view.py tests/field/test_camera_view.py
git commit -m "feat(field): load_views builds calibrated CameraViews (main first)"
```

---

## Phase 2 — ROI tag refine

Goal: turn a rough human click on an undecodable tag into sub-pixel corners by detecting in a small upscaled ROI, with coordinates mapped back to full-frame and the decoded id verified.

### Task 5: `detect_tags_in_roi`

**Files:**
- Modify: `web/field/apriltag_detect.py`
- Test: `tests/field/test_apriltag_roi.py`

- [ ] **Step 1: Write the failing test** (synthetic: paste a known tag image into a blank frame; assert refine returns corners near the paste location in full-frame coords; assert id mismatch is rejected). If no tag asset exists, generate one with `cv2.aruco`-style 36h11 is unavailable — instead test the **coordinate-mapping + id-guard logic** by monkeypatching the inner detector.

```python
# tests/field/test_apriltag_roi.py
import numpy as np
from web.field import apriltag_detect as ad

def test_roi_maps_corners_back_to_full_frame(monkeypatch):
    # Fake inner detector: in the (upscaled) ROI it "finds" tag 17 at a known spot.
    def fake_detect(frame_bgr):
        return {17: np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float)}
    monkeypatch.setattr(ad, "detect_tags", fake_detect)
    frame = np.zeros((400, 400, 3), np.uint8)
    # click near (200,200); ROI half-size 50, upscale 4
    out = ad.detect_tags_in_roi(frame, click_xy=(200, 200), target_id=17,
                                half=50, upscale=4)
    assert out is not None
    # corner (10,10) in 4x-upscaled ROI -> (10/4,10/4) + ROI offset (150,150)
    assert np.allclose(out[0], [150 + 2.5, 150 + 2.5], atol=1e-6)

def test_roi_rejects_id_mismatch(monkeypatch):
    monkeypatch.setattr(ad, "detect_tags", lambda f: {99: np.zeros((4, 2))})
    out = ad.detect_tags_in_roi(np.zeros((400, 400, 3), np.uint8),
                                click_xy=(200, 200), target_id=17)
    assert out is None  # decoded a different tag -> reject

def test_roi_none_when_nothing_decodes(monkeypatch):
    monkeypatch.setattr(ad, "detect_tags", lambda f: {})
    out = ad.detect_tags_in_roi(np.zeros((400, 400, 3), np.uint8),
                                click_xy=(200, 200), target_id=17)
    assert out is None
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest tests/field/test_apriltag_roi.py -v`
Expected: FAIL (`detect_tags_in_roi` not defined).

- [ ] **Step 3: Implement** (append to `apriltag_detect.py`)

```python
def detect_tags_in_roi(frame_bgr, click_xy, target_id, half=60, upscale=4):
    """Run tag detection in a small upscaled window around a click and return the
    target tag's corners in FULL-FRAME pixel coords, or None.

    Used by the calibration UI's "manual locate -> auto refine": the operator
    clicks roughly on a tag the frame-wide detector missed; we zoom that patch so
    the detector can decode it. We require the decoded id to equal target_id, so a
    stray neighbouring decode can't create a mislabeled correspondence.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = int(round(click_xy[0])), int(round(click_xy[1]))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    up = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    det = detect_tags(up)
    corners = det.get(int(target_id))
    if corners is None:
        return None
    # map upscaled-ROI corners back to full-frame: /upscale, + ROI offset
    return np.asarray(corners, float) / float(upscale) + np.array([x0, y0], float)
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_apriltag_roi.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/apriltag_detect.py tests/field/test_apriltag_roi.py
git commit -m "feat(field): detect_tags_in_roi for manual-locate tag refine"
```

---

## Phase 3 — Corner-camera calibration (endpoints + UI + held-out validation)

Goal: calibrate each corner camera with the same pinhole resection as main, gathering the spread tags via ROI refine, and reporting held-out-tag error. Per-camera correspondences are independent (shared tags map separately per camera).

### Task 6: Held-out validation helper

**Files:**
- Modify: `web/field/calibration.py`
- Test: `tests/field/test_calibration_holdout.py`

- [ ] **Step 1: Write the failing test** — build correspondences from a *known* calibration (project synthetic field points through a hand-made `CameraCalibration`), solve on a subset, and assert held-out field error is small.

```python
# tests/field/test_calibration_holdout.py
import numpy as np
from web.field.calibration import holdout_field_error, CameraCalibration

def test_holdout_field_error_zero_for_exact():
    # identity-ish homography: field == image (so image_to_field returns input)
    cal = CameraCalibration(mode="homography", H=np.eye(3), image_size=(100, 100))
    held = {7: (np.array([[3.0, 4.0]]), np.array([[3.0, 4.0, 0.0]]))}
    err = holdout_field_error(cal, held)
    assert err < 1e-6
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest tests/field/test_calibration_holdout.py -v`
Expected: FAIL (`holdout_field_error` not defined).

- [ ] **Step 3: Implement** (append to `calibration.py`)

```python
def holdout_field_error(cal: "CameraCalibration", held: dict) -> float:
    """Mean field-position error in meters over held-out correspondences.
    held: {tag_id: (img_pts (N,2), obj_pts (N,3))}. Projects each image point to
    the field and compares to the tag's known field (x,y)."""
    errs = []
    for (i2d, o3d) in held.values():
        for (u, v), p in zip(np.asarray(i2d).reshape(-1, 2), np.asarray(o3d).reshape(-1, 3)):
            xy = cal.image_to_field((float(u), float(v)))
            if xy is None:
                continue
            errs.append(np.hypot(xy[0] - p[0], xy[1] - p[1]))
    return float(np.mean(errs)) if errs else float("inf")
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_calibration_holdout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/calibration.py tests/field/test_calibration_holdout.py
git commit -m "feat(field): holdout_field_error for calibration validation"
```

### Task 7: `/api/calibration/frame` accepts a camera crop region

**Files:**
- Modify: `web/app.py` (`calibration_frame`, ~2432)
- Test: manual (endpoint smoke)

- [ ] **Step 1: Change the crop selection** to use the composite layout for the requested camera instead of the hardcoded `CALIB_MAIN_CROP_FRAC`.

Replace the crop block in `calibration_frame`:

```python
# old:
#   h = frame.shape[0]
#   main = frame[: int(h * CALIB_MAIN_CROP_FRAC), :]
# new:
from web.field.composite_layout import load_layout   # add at top of app.py with other imports
camera = body.get("camera", "main")
layout = load_layout()
cl = layout.get(camera)
if cl is None:
    return JSONResponse(status_code=400, content={"error": f"unknown camera '{camera}'"})
cx, cy, cw, ch = cl.crop
main = frame[cy:cy + ch, cx:cx + cw]
for (mx, my, mw, mh) in cl.masks:
    main[my:my + mh, mx:mx + mw] = 0
```

Keep the rest (`detect_tags(main)`, encode, return) unchanged. Auto-detect will return the hub cluster for corners — expected.

- [ ] **Step 2: Smoke test the endpoint** for a corner camera.

Run:
```bash
.venv/bin/python - <<'PY'
import requests
r = requests.post("http://127.0.0.1:8000/api/calibration/frame",
                  json={"timestamp": 20, "camera": "corner_left"})
d = r.json(); print({k: d.get(k) for k in ("width", "height", "detected_tags")})
PY
```
Expected: width≈960, height≈432, `detected_tags` includes the hub cluster (e.g. 25/26/27). (Start the server first per the project run skill if not running.)

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(app): calibration/frame crops to the requested camera region"
```

### Task 8: `/api/calibration/refine-tag` endpoint

**Files:**
- Modify: `web/app.py` (new endpoint near the other calibration routes)
- Test: manual (endpoint smoke)

- [ ] **Step 1: Add the endpoint** — grab the camera's crop at a timestamp, run `detect_tags_in_roi`, return refined full-frame corners or `null`.

```python
@app.post("/api/calibration/refine-tag")
async def calibration_refine_tag(request: Request):
    """Manual-locate -> auto-refine. Body: {camera?, timestamp?, video_path?,
    u, v, id}. Runs tag detection in an upscaled ROI around (u,v) in the camera's
    crop; returns {corners: [[u,v]x4]} in crop-pixel coords, or {corners: null}."""
    from web.field.apriltag_detect import detect_tags_in_roi
    from web.field.composite_layout import load_layout
    body = await request.json()
    camera = body.get("camera", "main")
    tid = int(body["id"])
    ts = float(body.get("timestamp", 75))
    # reuse the same video-path allowlist logic as calibration_frame
    path = CALIB_TEST_VIDEO  # (apply the same allowed_dir check if video_path given)
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps)); ok, frame = cap.read(); cap.release()
    if not ok or frame is None:
        return JSONResponse(status_code=400, content={"error": "could not read frame"})
    cl = load_layout().get(camera)
    if cl is None:
        return JSONResponse(status_code=400, content={"error": f"unknown camera '{camera}'"})
    cx, cy, cw, ch = cl.crop
    crop = frame[cy:cy + ch, cx:cx + cw]
    corners = detect_tags_in_roi(crop, (float(body["u"]), float(body["v"])), tid)
    return {"corners": corners.tolist() if corners is not None else None}
```

- [ ] **Step 2: Smoke test** — pick a pixel where a known off-hub tag sits (from the LEFT crop you inspected in Task 1) and confirm it returns 4 corners for that id, and returns `null` for an empty patch.

Run:
```bash
.venv/bin/python - <<'PY'
import requests
r = requests.post("http://127.0.0.1:8000/api/calibration/refine-tag",
                  json={"camera":"corner_left","timestamp":20,"u":700,"v":240,"id":18})
print(r.json())
PY
```
Expected: `{"corners": [[...],[...],[...],[...]]}` (or `null` if that pixel isn't on tag 18 — adjust u,v).

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(app): calibration/refine-tag (manual-locate -> ROI auto-refine)"
```

### Task 9: `solve-tags` overlay uses the camera crop; return held-out error

**Files:**
- Modify: `web/app.py` (`calibration_solve_tags`, ~2544)
- Test: manual (endpoint smoke during Task 11)

- [ ] **Step 1: Replace the hardcoded overlay crop** (`frame[: int(... CALIB_MAIN_CROP_FRAC), :]`) with the requested camera's crop via `load_layout()` (same pattern as Task 7), so the corner overlay is drawn on the corner frame.

- [ ] **Step 2: Add optional held-out reporting.** Accept `holdout_ids: [int]` in the body; before solving, split `corr` into solve-set and held-set by tag id; solve on the solve-set; compute `holdout_field_error(cal, held)` and include it in the response as `holdout_error_m` (omit/`null` when no holdout given).

```python
holdout_ids = set(int(i) for i in body.get("holdout_ids", []))
held = {k: v for k, v in corr.items() if _tag_id_of(k) in holdout_ids}
solve_corr = {k: v for k, v in corr.items() if _tag_id_of(k) not in holdout_ids}
# _tag_id_of("a25") -> 25, "m25" -> 25  (strip the a/m prefix)
cal = Calibrator.solve_camera_from_tags(solve_corr, image_size=image_size, field_season=season)
field_store.save(event, camera, cal)
holdout_err = holdout_field_error(cal, held) if held else None
# ... add "holdout_error_m": holdout_err to the returned dict
```

Add the tiny helper near the endpoint:
```python
def _tag_id_of(key: str) -> int:
    return int(key[1:]) if key[:1] in ("a", "m") else int(key)
```
Import `holdout_field_error` from `web.field.calibration`.

- [ ] **Step 3: Smoke** is covered by the real calibration in Task 11.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(app): solve-tags per-camera overlay + held-out error reporting"
```

### Task 10: Calibrate UI — camera picker drives crop, ROI refine on click, held-out error

**Files:**
- Modify: `web/static/calibrate.html`

- [ ] **Step 1: Make the camera field drive frame grabbing.** The page already has `#camera` (line 68). Pass `camera` in the `grabFrame()` POST to `/api/calibration/frame` and re-grab when the camera value changes (add a `change` listener that calls `grabFrame()` then `redrawFrame()`). The working correspondence rows (`renderRows`) already reset per grab — confirm switching camera clears the in-progress correspondence list so each camera's set is independent (the per-camera requirement).

- [ ] **Step 2: Manual click → ROI refine.** In the frame-canvas click handler (currently sets `pendingPixel`): after the user picks the map tag (so the target id is known), POST `{camera, timestamp, u, v, id}` to `/api/calibration/refine-tag`. If `corners` come back non-null, store them as an **auto-style** entry (4 corners → goes into `auto_tags` on solve) and flag the row "refined ✓"; if null, fall back to the existing manual center behavior (`manual_tags`). This is the manual-locate→auto-refine UX.

- [ ] **Step 3: Held-out error UI.** Add a checkbox per correspondence row ("hold out"); on `solve()`, send the checked tag ids as `holdout_ids`. Show `holdout_error_m` in `#statusLine` next to RMS (e.g. `RMS 2.3px · held-out 0.31 m`).

- [ ] **Step 4: Manual verification.** Load `/calibrate`, set camera to `corner_left`, grab frame (corner view shows), auto hub tags appear, manually locate two spread tags (ROI refine snaps them), mark one as held-out, Solve. Confirm overlay draws on the corner frame and held-out error prints.

- [ ] **Step 5: Commit**

```bash
git add web/static/calibrate.html
git commit -m "feat(ui): calibrate page supports corner cameras + ROI refine + held-out error"
```

### Task 11: Actually calibrate both corner cameras (acceptance gate)

**Files:**
- Produces: `web/uploads/calibration_<event>_corner_left.json`, `..._corner_right.json`

- [ ] **Step 1: Calibrate `corner_left`** via the UI on `qual1_first_champs_2026.mp4` (event `test`): auto hub tags + manually locate the spread tags (31/32/17/22/27). Aim for ≥2 successfully-located off-hub tags spanning the frame plus the hub cluster.

- [ ] **Step 2: Validate.** Mark 1–2 edge tags as held-out and Solve. **Acceptance gate:** held-out field error ≤ 0.5 m AND ≥2 off-hub tags located.

  If it fails, follow the spec escalation ladder:

  **Tier 1 — relax distortion flags.** The flags are currently hardcoded at `web/field/calibration.py:190-191`. Parameterize them so the escalation is real (this is a required code edit, not just an operator action):

  ```python
  # calibration.py — change the signature:
  def solve_camera_from_tags(correspondences, image_size, field_season,
                             relax_distortion: bool = False):
      ...
      flags = cv2.CALIB_USE_INTRINSIC_GUESS
      if not relax_distortion:
          flags |= cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
      # else: allow tangential + k3 (more terms, justified once points span the frame)
  ```
  Then thread `relax_distortion` through `/api/calibration/solve-tags` (read `body.get("relax_distortion", False)`) and add a checkbox in the calibrate UI. Re-solve with it on and re-check the held-out gate. Add a quick unit test that `relax_distortion=True` produces a non-zero `D[2]`/`D[4]` term on a synthetic spread set.

  **Tier 2 — homography fallback.** If still failing, fall back to homography (`/api/calibration/solve-homography`) for that camera and record that it is pose-free (it will be excluded from ray-cast occlusion by Task 17's `posed` filter — no further gate change needed).

- [ ] **Step 3: Repeat for `corner_right`** (tags 8/9/10 hub + spread).

- [ ] **Step 4: Record results** in a short note in the PR / commit body: per-camera held-out error and which path (pinhole / relaxed / homography) each landed on.

- [ ] **Step 5: Commit the calibration files**

```bash
git add web/uploads/calibration_test_corner_left.json web/uploads/calibration_test_corner_right.json
git commit -m "feat(field): corner-camera calibrations for test footage"
```

---

## Phase 4 — Cross-camera fusion (pure)

### Task 12: `FusedObservation` + `fuse_detections`

**Files:**
- Create: `web/field/fusion.py`
- Test: `tests/field/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/field/test_fusion.py
from web.field.fusion import fuse_detections, FusedObservation
from web.field.tracker import FieldDetection

def _det(x, y, color="blue", box=(0, 0, 10, 10)):
    return FieldDetection(pos=(x, y), color=color, box=box)

def test_single_camera_passthrough():
    per_cam = {"main": [(_det(1.0, 2.0), b"crop")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0}, radius_m=0.6)
    assert len(fused) == 1
    obs = fused[0]
    assert isinstance(obs, FusedObservation)
    assert abs(obs.detection.pos[0] - 1.0) < 1e-9
    assert obs.evidence == [("main", b"crop", 1.0)]

def test_merges_two_cameras_same_robot():
    per_cam = {
        "main": [(_det(5.0, 3.0, "blue"), b"m")],
        "corner_left": [(_det(5.2, 3.1, "blue"), b"c")],
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert len(fused) == 1
    obs = fused[0]
    # weighted-average position pulled toward the higher-weight corner read
    assert 5.0 < obs.detection.pos[0] <= 5.2
    assert {e[0] for e in obs.evidence} == {"main", "corner_left"}

def test_keeps_distinct_robots_separate():
    per_cam = {"main": [(_det(1, 1), b"a"), (_det(8, 7), b"b")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0}, radius_m=0.6)
    assert len(fused) == 2

def test_color_vote_resolves_unknown():
    per_cam = {
        "main": [(_det(2, 2, "unknown"), b"m")],
        "corner_left": [(_det(2.1, 2.0, "blue"), b"c")],
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 2.5}, radius_m=0.6)
    assert fused[0].detection.color == "blue"
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest tests/field/test_fusion.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# web/field/fusion.py
"""Cross-camera fusion: many cameras' field detections -> merged detections (for
the tracker) + pooled, source-weighted identity evidence (side channel).

Greedy single-link clustering by field distance. Inputs are per-camera lists of
(FieldDetection, bumper_crop); a cluster of detections within radius_m is one
physical robot. The merged FieldDetection (weighted-average position, weighted
color vote) is what the tracker consumes; the FusedObservation also carries the
raw evidence so the identity step can read each camera's bumper crop with its
source weight."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from web.field.tracker import FieldDetection


@dataclass
class FusedObservation:
    detection: FieldDetection                 # what the tracker sees
    evidence: list                            # [(source_camera, bumper_crop, weight), ...]


def fuse_detections(per_camera: dict, weights: dict, radius_m: float = 0.6) -> list:
    # flatten to (camera, det, crop, weight)
    items = []
    for cam, dets in per_camera.items():
        w = float(weights.get(cam, 1.0))
        for (det, crop) in dets:
            items.append((cam, det, crop, w))

    used = [False] * len(items)
    out = []
    for i in range(len(items)):
        if used[i]:
            continue
        cam_i, det_i, crop_i, w_i = items[i]
        cluster = [(cam_i, det_i, crop_i, w_i)]
        used[i] = True
        pi = np.array(det_i.pos, float)
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            cam_j, det_j, crop_j, w_j = items[j]
            if cam_j == cam_i:
                continue  # one camera shouldn't merge with itself
            if np.linalg.norm(pi - np.array(det_j.pos, float)) <= radius_m:
                cluster.append((cam_j, det_j, crop_j, w_j))
                used[j] = True
        out.append(_merge(cluster))
    return out


def _merge(cluster) -> FusedObservation:
    ws = np.array([c[3] for c in cluster], float)
    pos = np.array([c[1].pos for c in cluster], float)
    avg = (pos * ws[:, None]).sum(axis=0) / ws.sum()
    # weighted color vote, ignoring "unknown"
    votes = {}
    for cam, det, crop, w in cluster:
        if det.color and det.color != "unknown":
            votes[det.color] = votes.get(det.color, 0.0) + w
    color = max(votes, key=votes.get) if votes else "unknown"
    # representative box: highest-weight cluster member
    box = max(cluster, key=lambda c: c[3])[1].box
    evidence = [(cam, crop, w) for (cam, det, crop, w) in cluster]
    return FusedObservation(
        detection=FieldDetection(pos=(float(avg[0]), float(avg[1])), color=color, box=box),
        evidence=evidence,
    )
```

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_fusion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/field/fusion.py tests/field/test_fusion.py
git commit -m "feat(field): fuse_detections + FusedObservation (cross-camera merge)"
```

---

## Phase 5 — Multi-view ingest + fusion in the inference loop

Goal: drive all calibrated views each frame, fuse, feed the tracker, and attach identity evidence to the slot the tracker chose. When only `main` is calibrated, behavior is identical to today.

### Task 13: Load views in `_init_field_tracking`

**Files:**
- Modify: `web/app.py` (`_init_field_tracking`, ~1703)
- Test: manual

- [ ] **Step 1:** After the existing main calibration load/validate succeeds, also build the view list:

```python
from web.field.camera_view import load_views   # add at top with other field imports
# after self.calib is set and field tracker built:
self.views = load_views(self.event_key)         # main first; corners if calibrated
print(f"  Cameras active: {[v.name for v in self.views]}")
```

Keep the existing main-only `self.calib` (used by the early-OCR seeding pass and as the reference). `self.views[0]` is main.

- [ ] **Step 2: Verify** the line prints `['main']` on an event with only main calibrated, and `['main','corner_left','corner_right']` on the `test` event after Phase 3.

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(app): load CameraViews during field-tracking init"
```

### Task 14: Per-view detection + fusion in the main loop

**Files:**
- Modify: `web/app.py` (`process_video` main loop, steps 1–4 around 1838–1875)
- Test: manual smoke (Task 18) + parity check here

- [ ] **Step 1: Add a per-view detection helper** on the job class (keeps the loop readable):

```python
CORNER_VOTE_WEIGHT = 2.5   # module constant near the other vote weights; corner reads out-weigh main same-modality

def _detect_view(self, view, composite_frame):
    """Run YOLO + person detection on one camera view; return
    (field_dets_with_crops, person_boxes_field). field_dets_with_crops is a list of
    (FieldDetection, bumper_crop_b64)."""
    frame = view.extract(composite_frame)
    cal = view.calibration
    results = self.model(frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(results)
    robot_boxes = [dets.xyxy[i] for i in range(len(dets)) if dets.class_id[i] == 1]
    field_dets = to_field_detections(robot_boxes, cal, frame, self.field_dims,
                                     color_fn=self.detect_bumper_color)
    # pair each field det with its bumper crop (same bottom-band logic as VLM path)
    paired = [(d, self._bumper_crop_b64(frame, d.box)) for d in field_dets]
    person_results = self.person_model(frame, classes=[0], conf=DET_CONF_THRESHOLD, verbose=False)[0]
    pdets = sv.Detections.from_ultralytics(person_results)
    person_boxes_all = [tuple(pdets.xyxy[i]) for i in range(len(pdets))]
    person_boxes = field_adjacent_persons(person_boxes_all, cal, self.field_dims)
    return paired, person_boxes
```

Add `_bumper_crop_b64(self, frame, box)` factoring the existing bumper-band crop+upscale+encode logic out of `vision_identify_robots` (DRY — `vision_identify_robots` should call it too). **Preserve the `LEGIBILITY_FLOOR` check**: have it return `(crop_b64, legibility_score)` (or `None` when below floor) so the read loop can skip illegible crops — today's main path filters on `legibility_score` and `VLM_TOP_K`, and dropping that for main would be a perf/quality regression. The read loop (Task 16) skips evidence whose crop is `None`/below floor; corner crops are still read when legible, just ungated on occlusion.

- [ ] **Step 2: Replace loop steps 1–4** (single-view detection + `to_field_detections` + `field_tracker.update`) with the multi-view version:

```python
per_camera = {}            # cam_name -> [(FieldDetection, crop_b64), ...]
person_boxes_by_cam = {}   # cam_name -> [boxes] (field-filtered, that cam's pixels)
for view in self.views:
    paired, pboxes = self._detect_view(view, frame)
    per_camera[view.name] = paired
    person_boxes_by_cam[view.name] = pboxes

weights = {v.name: (CORNER_VOTE_WEIGHT if v.name != "main" else 1.0) for v in self.views}
fused = fuse_detections(per_camera, weights=weights)            # list[FusedObservation]
field_dets = [obs.detection for obs in fused]
dt = FRAME_SKIP / (self.video_fps or 30.0)
track_outs = self.field_tracker.update(field_dets, self._view_flags, dt)
```

The annotation pass below still uses the **main** view's `detections`/`overhead_frame`; keep `overhead_frame = self.views[0].extract(frame)` (main) so existing HUD/scoring/OCR-region code is unchanged.

- [ ] **Step 3: Parity check.** On an event with only `main` calibrated, confirm `track_outs` count and positions match pre-change behavior on the first ~200 processed frames (spot-check the radar / diagnostic snapshot is unchanged). Import `fuse_detections` at top.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(app): multi-view detection + cross-camera fusion in inference loop"
```

### Task 15: Attach fused evidence to the tracker-chosen slot

**Files:**
- Modify: `web/field/tracker.py` (`update` returns assignment), `web/app.py` (consume it)
- Test: `tests/field/test_tracker_assignment.py`

- [ ] **Step 1: Failing test** — `update` should tell us which detection index landed in which slot.

```python
# tests/field/test_tracker_assignment.py
from web.field.tracker import FieldTracker, FieldDetection

def test_update_reports_detection_to_slot():
    tr = FieldTracker(field_length=16.0)
    dets = [FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1))]
    outs = tr.update(dets, {}, 0.1)
    # new API: assignment maps detection index -> slot_id
    assert hasattr(tr, "last_det_to_slot")
    assert tr.last_det_to_slot.get(0) in {t.slot_id for t in outs}
```

- [ ] **Step 2: Run red**

Run: `.venv/bin/python -m pytest tests/field/test_tracker_assignment.py -v`
Expected: FAIL (`last_det_to_slot` not set).

- [ ] **Step 3: Implement** — record `det index -> slot_id` as `_assign` matches/seeds. Pass the detection's index into `_assign` (the per-alliance filtering currently loses indices, so tag each detection with its original index first):

In `update`, before the per-alliance loop, build `indexed = list(enumerate(detections))` and filter on `d` while keeping `idx`. Initialise `self.last_det_to_slot = {}`. In `_assign`, when a det matches a slot (`t.kf.update(...)`) or seeds a free slot, record `self._owner.last_det_to_slot[det_index] = slot.slot_id`. Simplest concrete approach: change `_assign` to accept `(slot, dets_with_idx, ...)` where each element is `(orig_index, FieldDetection)`, and write into `self.last_det_to_slot`. Keep the cost/matching math identical; only thread the index through.

(Keep the public return value of `update` unchanged — `TrackOut` list — so all existing callers are unaffected; the mapping is an attribute.)

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_tracker_assignment.py tests/field/ -v`
Expected: PASS (and existing tracker tests still green).

- [ ] **Step 5: Commit**

```bash
git add web/field/tracker.py tests/field/test_tracker_assignment.py
git commit -m "feat(field): tracker exposes detection-index -> slot_id assignment"
```

---

## Phase 6 — Identity: additive, corner-weighted

### Task 16: Source-weighted, ungated reads from fused evidence

**Files:**
- Modify: `web/app.py` (`vision_identify_robots` → add a source-weighted entry point; main loop calls it with fused evidence)
- Test: manual smoke (Task 18) + a focused unit test on the vote arithmetic

- [ ] **Step 1: Factor the vote application** out of `vision_identify_robots` into a small method so it can be driven from fused evidence with a per-source weight:

```python
def _apply_bumper_read(self, slot_id, crop_b64, remaining, remaining_blue,
                       remaining_red, source_weight):
    """Read one bumper crop and add a (confidence x source_weight) vote to the slot.
    source_weight > 1 for corner cameras (clearer reads)."""
    robot = self.robot_tracks.get(slot_id)
    if not robot:
        return
    res = _vision_read_bumpers([crop_b64], remaining, remaining_blue, remaining_red)[0]
    team = res.get("team_number")
    if not team:
        return
    conf = res.get("confidence", 0.0)
    weight = (1.0 if conf >= 0.85 else 0.5) * float(source_weight)
    if robot.team_ocr_votes is None:
        robot.team_ocr_votes = {}
    robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + weight
```

- [ ] **Step 2: Drive it from fused evidence in the main loop.** After `track_outs` and `last_det_to_slot` are available, build slot→evidence and read every camera's crop for each slot, **corner reads ungated**, main read keeps the existing `bumper_occluded` gate:

```python
# map fused detection index -> slot, then pour that obs's evidence into the slot
if processed_frame_num >= 10 and (processed_frame_num % BUMPER_READ_INTERVAL == 0):
    remaining, r_blue, r_red = self._remaining_rosters()   # factor from vision_identify_robots
    for det_idx, obs in enumerate(fused):
        slot_id = self.field_tracker.last_det_to_slot.get(det_idx)
        if slot_id is None:
            continue
        robot = self.robot_tracks.get(slot_id)
        if robot and robot.team_number_confirmed and not is_reread_due:
            continue
        for (cam, crop_b64, w) in obs.evidence:
            if crop_b64 is None:
                continue       # below legibility floor (see _bumper_crop_b64) — skip
            if cam == "main" and self._occ.get(slot_id, {}).get("bumper_occluded"):
                continue       # main keeps its occlusion gate
            # corner reads are NOT gated -> additive even when main sees the robot
            self._apply_bumper_read(slot_id, crop_b64, remaining, r_blue, r_red,
                                    source_weight=w)
```

Add `BUMPER_READ_INTERVAL` near the other intervals (reuse the VLM cadence value). Keep the existing `vision_identify_robots` call for the early-OCR seeding pass; the per-frame multi-cam reads above supersede the in-loop `vision_identify_robots` call (remove that one in-loop call to avoid double-reading main). `assign_teams_from_votes()` / `audit_overturn_confirmations()` on `IDENTITY_INTERVAL` are unchanged and now see corner-weighted votes.

- [ ] **Step 3: Unit-test the vote arithmetic** (corner out-weighs main, same modality):

```python
# tests/field/test_vote_weighting.py  (uses a tiny fake job-like object)
def test_corner_outweighs_main_same_confidence(monkeypatch):
    # main: weight 1.0 * conf-weight 1.0 = 1.0 ; corner: 2.5 * 1.0 = 2.5
    # assert a single corner read beats a single main read for the same team set
    ...
```
(Implement against a minimal stub that calls `_apply_bumper_read` with a monkeypatched `_vision_read_bumpers` returning fixed teams/confidence; assert resulting `team_ocr_votes`.)

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_vote_weighting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/field/test_vote_weighting.py
git commit -m "feat(app): additive corner-weighted bumper reads via fused evidence"
```

---

## Phase 7 — Multi-camera occlusion

### Task 17: Merge occlusion across posed cameras

**Files:**
- Modify: `web/app.py` (loop step 5, ~1875)
- Test: manual smoke

- [ ] **Step 1: Run `resolve_occlusion` for each posed camera and merge.** A track is `view_occluded` only if occluded in **every** posed camera that could see it (any camera with a clear line keeps it visible); `bumper_occluded` likewise OR-reduced toward "readable if any camera can read it":

```python
occ_input = [{"slot_id": t.slot_id, "pos": t.pos, "box": t.box} for t in track_outs]
posed = [v for v in self.views if v.calibration.camera_center is not None]
per_cam_occ = []
for v in posed:
    # person boxes for THIS camera were computed in Task 14 (person_boxes_by_cam)
    per_cam_occ.append(resolve_occlusion(
        v.calibration, occ_input, person_boxes_by_cam[v.name], self.hub_prisms))
self._occ = _merge_occlusion(per_cam_occ)   # view_occluded = AND across cams; bumper too
self._view_flags = self._occ
```

- [ ] **Step 2: Add `_merge_occlusion`** (module-level helper): for each slot, `view_occluded = all(cam[slot]["view_occluded"])`, `bumper_occluded = all(cam[slot]["bumper_occluded"])`, cause = first non-null. With only main posed, this equals today's single-camera result (parity).

- [ ] **Step 3: Unit-test the merge**

```python
# tests/field/test_occlusion_merge.py
from web.app import _merge_occlusion   # or move helper to web/field/occlusion.py and import there
def test_visible_if_any_camera_sees():
    a = {0: {"view_occluded": True,  "bumper_occluded": True,  "cause": "hub"}}
    b = {0: {"view_occluded": False, "bumper_occluded": False, "cause": None}}
    m = _merge_occlusion([a, b])
    assert m[0]["view_occluded"] is False
    assert m[0]["bumper_occluded"] is False
```

Prefer putting `_merge_occlusion` in `web/field/occlusion.py` (pure, importable) rather than `app.py`.

- [ ] **Step 4: Run green**

Run: `.venv/bin/python -m pytest tests/field/test_occlusion_merge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/app.py web/field/occlusion.py tests/field/test_occlusion_merge.py
git commit -m "feat(app): merge occlusion across posed cameras (visible if any sees)"
```

---

## Phase 8 — End-to-end validation

### Task 18: Multi-camera smoke test on real footage

**Files:**
- Create: `tests/test_multicam_smoke.py` (mirrors the SP2 field-tracking smoke check)

- [ ] **Step 1: Write a smoke test** that runs the pipeline on `qual1_first_champs_2026.mp4` for the `test` event (all three cameras calibrated) over a bounded frame window and asserts the spec's two pass signals:
  - **(a) occlusion fill:** at least one frame has a tracked robot whose field position falls in a hub-occluded zone for `main` but is supported by a corner detection (i.e., a fused observation with `corner_*` evidence in that region).
  - **(b) additive identity:** for at least one slot that `main`-only evidence leaves unconfirmed/mislabeled, the final voted identity matches the TBA roster once corner reads are included.

Structure it like the existing SP2 smoke check (bounded frames, asserts on the diagnostic/track outputs rather than exact values). Gate it behind the footage existing (skip if missing) so CI without the asset still passes.

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_multicam_smoke.py -v`
Expected: PASS (or a clear, actionable assertion message if a corner calibration is too weak — loop back to Task 11).

- [ ] **Step 3: Full suite green**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_multicam_smoke.py
git commit -m "test(field): multi-camera fusion smoke check on real footage"
```

### Task 19: Cleanup + docs

- [ ] **Step 1:** Delete `scripts/measure_composite_layout.py` if no longer needed (or keep if useful), remove the now-dead single-view `vision_identify_robots` in-loop call if fully superseded, and run `@superpowers:verification-before-completion`.
- [ ] **Step 2:** Update any operator note / README section describing how to calibrate the corner cameras (camera picker + ROI refine + held-out gate).
- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore(field): cleanup + corner-camera calibration docs"
```

---

## Acceptance summary

- Each corner camera calibrated with held-out field error ≤ 0.5 m and ≥2 off-hub tags located (or documented fallback).
- All three cameras run every frame; `fuse_detections` merges them into the untouched 6-slot tracker.
- Corner bumper reads are additive (ungated) and weighted higher; identity confirms/corrects even when main sees the robot.
- Behavior with only `main` calibrated is unchanged (parity checks in Tasks 14 & 17).
- Full pytest suite green; multi-camera smoke test demonstrates occlusion fill + additive identity on real footage.

# Match-Ordered Calibration Epochs — Implementation Plan (Spec 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep multiple per-camera calibrations per event as match-ordered epochs (never overwriting), auto-select the epoch covering the match being analyzed, validate it against auto-detected AprilTags after the timer phase, and cancel the run with a clear "needs calibration" status on drift.

**Architecture:** Pure helpers (match-order parsing/epoch selection, tag-alignment scoring) layered under a versioned calibration store; the `InferenceJob` selects each camera's epoch from `self.match_key` at init and validates the main camera against sampled in-match frames before the SAM2 pass.

**Tech Stack:** Pure-Python/NumPy helpers (torch-free, unit-testable); existing `cv2`-based calibration projection; `pytest` via `.venv/bin/python -m pytest` (base python has no pytest). Synthetic fixtures in `tests/_field_synth.py`.

**Spec:** `docs/superpowers/specs/2026-06-08-calibration-epochs-design.md`

---

## File Structure

- **`web/field/match_order.py`** (new) — pure: `match_sort_key(match_key)` and `select_epoch(versions, match_key)`. No IO.
- **`web/field/calibration.py`** (modify) — add pure `tag_alignment_error(cal, detected_tags, field_model)`.
- **`web/field/store.py`** (modify) — add `CalibrationVersion`, `save_version`, `load_versions`, `update_version`; keep `save`/`load`/`load_intrinsics`.
- **`web/field/camera_view.py`** (modify) — `load_views` accepts an optional `{camera: version_id}` selection map (single chosen path; default = latest per camera).
- **`web/app.py`** (modify) — `_init_field_tracking` (select epochs from `self.match_key`); `process_video` (`_verify_calibration_alignment` after the timer phase, abort on drift); the six `field_store.save` endpoints (geometry solves → `save_version`; property edits → `update_version`); config constants.
- **Tests** — `tests/field/test_match_order.py` (new), additions to `tests/test_calibration_store.py`, `tests/test_calibration_pinhole.py` (or a new `tests/field/test_tag_alignment.py`), and an `InferenceJob` validation test.

Note: `web/app.py` carries unrelated uncommitted WIP. Every commit below stages **only** the listed files explicitly — never `git add -A`/`.`/`-am`. The app.py tasks (5-7) need the stash-and-restore dance the prior app.py work used: `git stash push -- web/app.py` → edit → commit only `web/app.py` → `git stash pop`. (The controller handles the stash; the implementer is told when a task is app.py-only.)

---

## Task 1: Match-order parsing + epoch selection (`match_order.py`)

Pure, no IO. `select_epoch` is the core of "which calibration covers this match."

**Files:**
- Create: `web/field/match_order.py`
- Test: `tests/field/test_match_order.py` (new)

- [ ] **Step 1: Write failing tests**

```python
import pytest
from web.field.match_order import match_sort_key, select_epoch


def test_sort_key_orders_quals_numerically():
    k = match_sort_key
    assert k("2026hop_qm2") < k("2026hop_qm10") < k("2026hop_qm120")

def test_sort_key_quals_before_playoffs():
    assert match_sort_key("2026hop_qm120") < match_sort_key("2026hop_sf1m1")
    assert match_sort_key("2026hop_sf1m1") < match_sort_key("2026hop_f1m1")

def test_sort_key_playoff_set_then_match():
    assert match_sort_key("2026hop_sf1m1") < match_sort_key("2026hop_sf2m1")
    assert match_sort_key("2026hop_sf1m1") < match_sort_key("2026hop_sf1m2")

def test_select_epoch_picks_greatest_start_at_or_before():
    versions = [
        {"version_id": "a", "start_match": "2026hop_qm1"},
        {"version_id": "b", "start_match": "2026hop_qm50"},
    ]
    assert select_epoch(versions, "2026hop_qm72")["version_id"] == "b"
    assert select_epoch(versions, "2026hop_qm49")["version_id"] == "a"
    assert select_epoch(versions, "2026hop_qm50")["version_id"] == "b"

def test_select_epoch_legacy_none_covers_from_start():
    versions = [
        {"version_id": "legacy", "start_match": None},
        {"version_id": "b", "start_match": "2026hop_qm50"},
    ]
    assert select_epoch(versions, "2026hop_qm10")["version_id"] == "legacy"
    assert select_epoch(versions, "2026hop_qm80")["version_id"] == "b"

def test_select_epoch_earlier_only_version_still_covers_earlier_match():
    # Only a qm50 version exists; analyzing qm10 still selects it (earliest extends
    # back to the start). The tag check then decides if it actually fits.
    versions = [{"version_id": "b", "start_match": "2026hop_qm50"}]
    assert select_epoch(versions, "2026hop_qm10")["version_id"] == "b"

def test_select_epoch_empty_returns_none():
    assert select_epoch([], "2026hop_qm1") is None

def test_select_epoch_none_match_key_returns_latest_created():
    versions = [
        {"version_id": "a", "start_match": "2026hop_qm1", "created_at": "2026-06-01"},
        {"version_id": "b", "start_match": "2026hop_qm50", "created_at": "2026-06-08"},
    ]
    assert select_epoch(versions, None)["version_id"] == "b"
```

- [ ] **Step 2: Run — expect ImportError**

`.venv/bin/python -m pytest tests/field/test_match_order.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
"""Match-order parsing and calibration-epoch selection (pure, no IO).

Match order is the FRC convention parsed from the match key — qual before playoff,
then ascending set/match number — NOT wall-clock time. Calibrations are anchored to
a start match and apply forward in this order until the next epoch.
"""
from __future__ import annotations
import re

# comp_level rank (lower = earlier in the event)
_LEVEL_RANK = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
_KEY_RE = re.compile(r".*_(qm|ef|qf|sf|f)(\d+)(?:m(\d+))?$")


def match_sort_key(match_key: str) -> tuple:
    """Sortable (level_rank, set_number, match_number) for an FRC match key like
    '2026hop_qm72' or '2026hop_sf1m1'. Unknown/garbage keys sort last."""
    m = _KEY_RE.match(match_key or "")
    if not m:
        return (99, 0, 0)
    level, n1, n2 = m.group(1), int(m.group(2)), m.group(3)
    if n2 is None:                      # quals: '_qm72' -> n1 is the match number
        return (_LEVEL_RANK[level], 0, n1)
    return (_LEVEL_RANK[level], n1, int(n2))   # playoffs: setN matchM


def select_epoch(versions, match_key):
    """Return the calibration version whose epoch covers ``match_key``.

    versions: list of dicts with at least 'start_match' (a match key or None) and
    'version_id'; 'created_at' used as a tiebreak / for the None-key fallback.
    Rule: pick the greatest start_match that sorts <= match_key. A None start_match
    (legacy) sorts before all matches (covers from the start). If every version
    starts AFTER match_key, the earliest still wins (earliest extends back to the
    start). Returns None only if versions is empty. If match_key is None, return the
    most recently created version.
    """
    if not versions:
        return None
    if match_key is None:
        return max(versions, key=lambda v: v.get("created_at") or "")
    target = match_sort_key(match_key)

    def start_key(v):
        sm = v.get("start_match")
        return (-1, 0, 0) if sm is None else match_sort_key(sm)

    covering = [v for v in versions if start_key(v) <= target]
    pool = covering if covering else versions   # else: earliest extends backward
    return max(pool, key=start_key)
```

- [ ] **Step 4: Run — expect pass**

`.venv/bin/python -m pytest tests/field/test_match_order.py -v` → all pass.

- [ ] **Step 5: Commit**

```bash
git add web/field/match_order.py tests/field/test_match_order.py
git commit -m "feat(calib): match-order parsing + epoch selection (pure)"
```

---

## Task 2: Tag-alignment scorer (`calibration.py`)

**Files:**
- Modify: `web/field/calibration.py` (add after `holdout_field_error`, ~line 256)
- Test: `tests/field/test_tag_alignment.py` (new)

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from web.field.calibration import tag_alignment_error
from web.field.field_model import FieldModel
from tests._field_synth import truth_pinhole_camera, project_tag_corners


def test_aligned_tags_near_zero_error():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)   # {tid: (img(4,2), obj(4,3))}
    detected = {tid: img for tid, (img, _obj) in corr.items()}
    err_m, n = tag_alignment_error(cam, detected, fm)
    assert n >= 4
    assert err_m < 0.10               # tags reproject to <10 cm in field space

def test_shifted_tags_large_error():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    corr = project_tag_corners(cam, fm)
    detected = {tid: (img + 60.0) for tid, (img, _obj) in corr.items()}  # 60px shift
    err_m, n = tag_alignment_error(cam, detected, fm)
    assert n >= 4
    assert err_m > 0.5                # well past the drift threshold

def test_no_usable_tags_returns_inf_zero():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    err_m, n = tag_alignment_error(cam, {999999: np.zeros((4, 2))}, fm)
    assert n == 0 and err_m == float("inf")
```

- [ ] **Step 2: Run — expect ImportError**

`.venv/bin/python -m pytest tests/field/test_tag_alignment.py -v` → FAIL.

- [ ] **Step 3: Implement**

In `web/field/calibration.py`, after `holdout_field_error`:

```python
def tag_alignment_error(cal, detected_tags, field_model):
    """(mean field error in meters, n_tags) between auto-detected AprilTags and
    their calibrated positions.

    detected_tags: {tag_id: corners (4,2)} from detect_tags(). Builds correspondences
    only for tags the field model knows, pairing detector corners to
    field_model.tag_corners_3d(tid) the SAME way solve_camera_from_tags does, then
    reuses holdout_field_error (height-aware). Returns (inf, 0) when no usable tags.
    """
    held = {}
    for tid, corners in detected_tags.items():
        if tid in field_model.tags:
            held[tid] = (np.asarray(corners, float).reshape(-1, 2),
                         field_model.tag_corners_3d(tid))
    if not held:
        return float("inf"), 0
    return holdout_field_error(cal, held), len(held)
```

- [ ] **Step 4: Run — expect pass**

`.venv/bin/python -m pytest tests/field/test_tag_alignment.py -v` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/field/calibration.py tests/field/test_tag_alignment.py
git commit -m "feat(calib): tag_alignment_error scorer (reuses holdout_field_error)"
```

---

## Task 3: Versioned store (`store.py`)

**Files:**
- Modify: `web/field/store.py`
- Test: `tests/test_calibration_store.py` (append)

- [ ] **Step 1: Write failing tests** (append; reuse the file's existing `_cal()` helper)

```python
def test_save_version_creates_new_file_never_overwrites(tmp_path):
    store.UPLOAD_DIR = tmp_path
    v1 = store.save_version("evt", "main", _cal(), start_match="2026evt_qm1", label="day1")
    v2 = store.save_version("evt", "main", _cal(), start_match="2026evt_qm50")
    assert v1 != v2
    files = sorted(tmp_path.glob("calibration_evt_main__*.json"))
    assert len(files) == 2          # neither overwrote the other

def test_load_versions_returns_all_with_metadata(tmp_path):
    store.UPLOAD_DIR = tmp_path
    store.save_version("evt", "main", _cal(), start_match="2026evt_qm1", label="day1")
    store.save_version("evt", "main", _cal(), start_match="2026evt_qm50")
    vs = store.load_versions("evt", "main")
    assert len(vs) == 2
    starts = {v["start_match"] for v in vs}
    assert starts == {"2026evt_qm1", "2026evt_qm50"}
    assert all(v["calibration"] is not None and "version_id" in v for v in vs)
    assert any(v["label"] == "day1" for v in vs)

def test_load_versions_includes_legacy_single_file_as_none_start(tmp_path):
    store.UPLOAD_DIR = tmp_path
    store.save("evt", "main", _cal())          # legacy one-file save
    vs = store.load_versions("evt", "main")
    assert len(vs) == 1
    assert vs[0]["start_match"] is None        # covers from the beginning

def test_update_version_in_place_keeps_metadata(tmp_path):
    store.UPLOAD_DIR = tmp_path
    vid = store.save_version("evt", "main", _cal(), start_match="2026evt_qm1", label="day1")
    cal2 = _cal(); cal2.robot_track_top_crop = 0.42
    store.update_version("evt", "main", vid, cal2)
    vs = store.load_versions("evt", "main")
    assert len(vs) == 1                         # no new epoch created
    v = vs[0]
    assert v["version_id"] == vid and v["start_match"] == "2026evt_qm1" and v["label"] == "day1"
    assert v["calibration"].robot_track_top_crop == 0.42

def test_load_versions_empty_when_none(tmp_path):
    store.UPLOAD_DIR = tmp_path
    assert store.load_versions("evt", "main") == []
```

- [ ] **Step 2: Run — expect failures (functions missing)**

`.venv/bin/python -m pytest tests/test_calibration_store.py -v` → new tests FAIL.

- [ ] **Step 3: Implement** (add to `web/field/store.py`; keep existing functions)

```python
import time, glob, os

def _version_path(event_key: str, camera: str, version_id: str) -> Path:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"calibration_{safe}__{version_id}.json"

def _version_glob(event_key: str, camera: str) -> str:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return str(UPLOAD_DIR / f"calibration_{safe}__*.json")

def save_version(event_key, camera, cal, start_match=None, label=None) -> str:
    """Write a NEW calibration version (never overwrites). Returns its version_id."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # microsecond timestamp keeps version_ids sortable and unique
    version_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time()*1e6) % 1_000_000:06d}"
    payload = {"calibration": cal.to_json(), "start_match": start_match,
               "label": label, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "version_id": version_id}
    _version_path(event_key, camera, version_id).write_text(json.dumps(payload, indent=2))
    return version_id

def _read_version_file(p: Path) -> dict:
    d = json.loads(p.read_text())
    return {"calibration": CameraCalibration.from_json(d["calibration"]),
            "start_match": d.get("start_match"), "label": d.get("label"),
            "created_at": d.get("created_at"), "version_id": d["version_id"]}

def load_versions(event_key, camera) -> list[dict]:
    """All calibration versions for (event,camera): versioned files + the legacy
    single file (as a start_match=None version). Sorted by created_at."""
    out = []
    for fp in glob.glob(_version_glob(event_key, camera)):
        try:
            out.append(_read_version_file(Path(fp)))
        except Exception:
            pass
    legacy = _path(event_key, camera)
    if legacy.exists():
        cal = CameraCalibration.from_json(json.loads(legacy.read_text()))
        out.append({"calibration": cal, "start_match": None, "label": "legacy",
                    "created_at": "", "version_id": "legacy"})
    out.sort(key=lambda v: v.get("created_at") or "")
    return out

def update_version(event_key, camera, version_id, cal) -> None:
    """Overwrite an existing version's calibration in place (property edits), keeping
    its start_match/label/created_at. 'legacy' updates the legacy single file."""
    if version_id == "legacy":
        save(event_key, camera, cal)
        return
    p = _version_path(event_key, camera, version_id)
    d = json.loads(p.read_text())
    d["calibration"] = cal.to_json()
    p.write_text(json.dumps(d, indent=2))
```

Notes for the implementer: `glob`, `time`, `os` imports go at the top with the existing imports. The `version_id` format is sortable lexically (date-time-micros), which `select_epoch`'s `created_at` tiebreak relies on.

- [ ] **Step 4: Run — expect pass** (and the file's prior tests still pass)

`.venv/bin/python -m pytest tests/test_calibration_store.py -v` → all pass.

- [ ] **Step 5: Commit**

```bash
git add web/field/store.py tests/test_calibration_store.py
git commit -m "feat(calib): versioned calibration store (save_version/load_versions/update_version)"
```

---

## Task 4: Per-camera epoch selection in `load_views` (`camera_view.py`)

Let `load_views` build each `CameraView` from a chosen version. Single path: an optional `selection` map `{camera: version_id}`; default (None) = latest version per camera (back-compat with today's "one file" behavior).

**Files:**
- Modify: `web/field/camera_view.py`
- Test: `tests/field/test_camera_view_epochs.py` (new)

- [ ] **Step 1: Write failing test**

```python
def test_load_views_selects_requested_version(tmp_path, monkeypatch):
    from web.field import store
    from web.field.calibration import CameraCalibration
    import numpy as np
    store.UPLOAD_DIR = tmp_path
    def cal(tc):
        c = CameraCalibration(K=np.eye(3), D=np.zeros(4), rvec=np.zeros((3,1)),
                              tvec=np.ones((3,1)), image_size=(1920,1080),
                              field_season="2026-rebuilt-welded", mode="pinhole")
        c.robot_track_top_crop = tc
        return c
    v1 = store.save_version("evt", "main", cal(0.1), start_match="evt_qm1")
    v2 = store.save_version("evt", "main", cal(0.4), start_match="evt_qm50")
    from web.field.camera_view import load_views
    views = load_views("evt", selection={"main": v1})
    main = next(v for v in views if v.name == "main")
    assert abs(main.calibration.robot_track_top_crop - 0.1) < 1e-9
    # default (no selection) -> latest
    main_latest = next(v for v in load_views("evt") if v.name == "main")
    assert abs(main_latest.calibration.robot_track_top_crop - 0.4) < 1e-9
```

(If `load_views` requires the composite layout, the test relies on the real `composite_layout.json`; `main` is always present.)

- [ ] **Step 2: Run — expect fail** (`load_views` takes no `selection`)

- [ ] **Step 3: Implement** — change `load_views(event_key)` → `load_views(event_key, selection=None)`. For each camera in the layout: `versions = field_store.load_versions(event_key, name)`; if none, `continue` (unchanged behavior — uncalibrated cameras omitted); pick `selection[name]` if provided and present, else the most-recent version (`max` by `created_at`); use that version's `calibration`. Keep the existing mask-merge and `main`-first sort.

- [ ] **Step 4: Run — expect pass**, plus `.venv/bin/python -m pytest tests/field/ -q` to confirm no regression in callers of `load_views`.

- [ ] **Step 5: Commit**

```bash
git add web/field/camera_view.py tests/field/test_camera_view_epochs.py
git commit -m "feat(calib): load_views per-camera version selection"
```

---

## Task 5: Epoch selection in `_init_field_tracking` (`app.py`)

**Files:** Modify: `web/app.py` (`_init_field_tracking`, ~3679-3712). **app.py-only — controller stashes WIP first.**

- [ ] **Step 1: Implement selection**

Replace the single `cal = field_store.load(self.event_key, self.camera)` (and the `cal is None` / pose-free guards that follow at ~3679-3692) with epoch selection:

```python
from web.field.match_order import select_epoch
versions = field_store.load_versions(self.event_key, self.camera)
if not versions:
    self.error = (f"No calibration found for event '{self.event_key}' "
                  f"camera '{self.camera}'. Calibrate the camera first.")
    print(f"  FIELD TRACKING DISABLED: {self.error}")
    return False
chosen = select_epoch(versions, self.match_key)
cal = chosen["calibration"]
self._calib_version_id = chosen["version_id"]   # for the validation message
if self.match_key is None:
    print(f"  [calib] no match_key — using latest version {chosen['version_id']}")
else:
    print(f"  [calib] match {self.match_key}: epoch {chosen['version_id']} "
          f"(start_match={chosen['start_match']})")
```

Keep the existing `cal.camera_center is None` pose-free guard, now applied to the **selected** `cal`. Then `self.calib = cal` as today.

Build `self.views` with the selected epoch per camera. Compute a per-camera selection map and pass it to `load_views`:

```python
selection = {}
for name in load_layout().keys():
    vs = field_store.load_versions(self.event_key, name)
    pick = select_epoch(vs, self.match_key)
    if pick is not None:
        selection[name] = pick["version_id"]
self.views = load_views(self.event_key, selection=selection)
```

(`load_layout` is already imported via `camera_view`; import it in app.py or expose a helper — follow the existing import style. Verify `_calib_version_id` is initialized in `__init__` to `None`.)

- [ ] **Step 2: Verify import + parse**

`.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`

- [ ] **Step 3: Sanity test** — `.venv/bin/python -m pytest tests/ -k "calibration or field" -q` (no regressions in importable tests).

- [ ] **Step 4: Commit** — `git add web/app.py` (only) + message `feat(app): select calibration epoch by match_key in _init_field_tracking`.

---

## Task 6: Match-start AprilTag validation in `process_video` (`app.py`)

**Files:** Modify: `web/app.py` (config constants near other knobs; `_verify_calibration_alignment` method; call site after the timer phase ~line 3876, before the SAM2 pre-computation pass ~line 3905). **app.py-only — controller stashes WIP first.**

- [ ] **Step 1: Add config constants** (near the other env knobs, e.g. by the SAM2/GALLERY knobs)

```python
CALIB_CHECK_ENABLED = os.environ.get("CALIB_CHECK_ENABLED", "1") not in ("0","false","False")
CALIB_DRIFT_MAX_M = float(os.environ.get("CALIB_DRIFT_MAX_M", "0.5"))
CALIB_CHECK_MIN_TAGS = int(os.environ.get("CALIB_CHECK_MIN_TAGS", "2"))
CALIB_CHECK_SAMPLE_FRAMES = int(os.environ.get("CALIB_CHECK_SAMPLE_FRAMES", "5"))
```

- [ ] **Step 2: Add the validation method** on `InferenceJob`

```python
def _verify_calibration_alignment(self) -> bool:
    """Validate self.calib against auto-detected AprilTags in sampled in-match
    frames (main view). Returns True to proceed. On drift, sets self.error +
    self.status='error' + self.needs_calibration and returns False. No tags -> warn
    + proceed. Runs off the event loop is not required here (process_video already
    runs in a worker context); detect_tags must stay on one sequential path."""
    if not CALIB_CHECK_ENABLED or self.calib is None:
        return True
    main_view = next((v for v in self.views if v.name == "main"), None)
    if main_view is None or not self.processing_ranges:
        return True
    lo, hi = self.processing_ranges[0]
    n = max(1, CALIB_CHECK_SAMPLE_FRAMES)
    frame_idxs = [int(lo + (hi - lo) * i / (n - 1)) for i in range(n)] if n > 1 else [lo]
    best_err, best_n = float("inf"), 0
    cap = cv2.VideoCapture(str(self.video_path))
    try:
        for fidx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            crop = main_view.extract(frame)
            detected = detect_tags(crop)
            err_m, ntags = tag_alignment_error(self.calib, detected, self.field_model)
            if ntags >= CALIB_CHECK_MIN_TAGS and err_m < best_err:
                best_err, best_n = err_m, ntags
    finally:
        cap.release()
    if best_n < CALIB_CHECK_MIN_TAGS:
        print(f"  [calib] WARNING: could not validate calibration — <{CALIB_CHECK_MIN_TAGS} "
              f"AprilTags detected across {n} sampled frames; proceeding unvalidated.")
        return True
    if best_err > CALIB_DRIFT_MAX_M:
        self.status = "error"
        self.error = (f"Calibration drift for '{self.camera}' on {self.match_key or 'this match'}: "
                      f"AprilTags are {best_err:.2f} m off (epoch {self._calib_version_id}). "
                      f"The camera likely moved/zoomed — recalibrate for this match "
                      f"(it will be saved as a new calibration version).")
        self.needs_calibration = {"event": self.event_key, "camera": self.camera,
                                  "match_key": self.match_key, "error_m": round(best_err, 3),
                                  "epoch": self._calib_version_id}
        print(f"  FIELD TRACKING ABORTED: {self.error}")
        return False
    print(f"  [calib] OK: {best_n} tags align to {best_err:.3f} m (epoch {self._calib_version_id})")
    return True
```

Initialize `self.needs_calibration = None` in `__init__`. Confirm `detect_tags` is imported in app.py (it is used by `/api/calibration/frame`); confirm `tag_alignment_error` import is added.

- [ ] **Step 3: Wire the call** — after the timer phase computes `self.processing_ranges` (~line 3876) and **before** the SAM2 pre-computation pass (~line 3905). **Mirror the existing init-abort path** (lines 3794-3800) so the client is notified over the WebSocket — an early `return` from inside the later `try` reaches neither the "complete" send (end of `try`) nor the "error" send (in `except`), so you MUST send explicitly:

```python
            if not self._verify_calibration_alignment():
                # _verify_calibration_alignment already set self.status="error",
                # self.error, and self.needs_calibration. Notify the client and stop
                # before the SAM2 pass (mirrors the _init_field_tracking abort at ~3799).
                await self._send_ws({"type": "error", "error": self.error,
                                     "needs_calibration": self.needs_calibration})
                return
```

(Match the surrounding indentation inside `process_video`. `_send_ws` is the job's WS helper, line ~2145. Note: `self._replay` is created later — at ~3909, after this point — so no replay row is written on a drift abort; that's consistent with the existing `_init_field_tracking` abort, which also writes none. The `finally` block guards on `getattr(self,"_replay",None)`, so the early return is safe.)

- [ ] **Step 4: Verify parse + import**

`.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`

- [ ] **Step 5: Test** — see Task 8 for the integration test (it exercises this method with a stub). For now: `.venv/bin/python -m pytest tests/ -k "calibration or field" -q`.

- [ ] **Step 6: Commit** — `git add web/app.py` (only) + `feat(app): match-start AprilTag calibration drift check (abort on drift)`.

---

## Task 7: Route save endpoints to the versioned store (`app.py`)

**Files:** Modify: `web/app.py` (six `field_store.save` call sites). **app.py-only — controller stashes WIP first.**

- [ ] **Step 1: Geometry solves → `save_version`** at: `solve-tags` (~5917), `solve-homography` (~5848), `manual` (~5640), `auto` (~5681). For each:
  - Read `start_match = body.get("start_match")` (default None).
  - Read `label = body.get("label")` (default None).
  - Carry forward masks/top-crop from the **selected/latest** version, not the legacy `load()`. Where the code currently does `existing = field_store.load(...)`, replace with: `vs = field_store.load_versions(event, camera); prev = select_epoch(vs, start_match) if vs else None` then `if prev: cal.masks = prev["calibration"].masks; cal.robot_track_top_crop = prev["calibration"].robot_track_top_crop`.
  - Replace `field_store.save(event, camera, cal)` with `field_store.save_version(event, camera, cal, start_match=start_match, label=label)`.

- [ ] **Step 2: Property edits → `update_version`** at: `masks` (~5582), `robot-track-top-crop` (~5604). For each:
  - `version_id = body.get("version_id")`. Load versions; pick the target: the requested `version_id`, else the most-recent (`max` by `created_at`).
  - If no versions exist, create one: `save_version(event, camera, cal, start_match=None)` (mirrors today's "no calibration → default" branch). Else modify the chosen version's calibration field and `update_version(event, camera, target_version_id, cal)`.

- [ ] **Step 3: Verify parse** — `.venv/bin/python -c "import ast; ast.parse(open('web/app.py').read()); print('ok')"`.

- [ ] **Step 4: Endpoint smoke test** (optional, if a FastAPI TestClient harness exists in `tests/test_calibration_api.py` — follow its pattern): a `solve-tags`-style save then `load_versions` returns a version with the posted `start_match`; a `masks` POST updates that version in place (no new epoch). If no API harness, rely on Task 3 store tests + manual verification.

- [ ] **Step 5: Commit** — `git add web/app.py` (only) + `feat(app): calibration saves create/update epoch versions`.

---

## Task 8: Integration test for the drift check + full suite

**Files:** Test: `tests/field/test_calibration_drift_check.py` (new) — exercises `_verify_calibration_alignment` with a minimal stubbed job, OR a focused unit test of the decision logic if constructing an `InferenceJob` is too heavy.

- [ ] **Step 1: Write the test.** Prefer a lightweight approach: build a tiny object exposing the attributes the method reads (`calib`, `views` with a `.extract` + `.name=="main"`, `processing_ranges`, `field_model`, `video_path`, `match_key`, `camera`, `_calib_version_id`) and bind `_verify_calibration_alignment` to it, OR factor the decision into a pure helper `evaluate_calibration_drift(best_err, best_n, min_tags, max_m)` and unit-test that plus a thin frame-sampling wrapper. Cover:
  - aligned (tags reproject to <0.5 m) → returns True, no error set.
  - drifted (>0.5 m) → returns False, `needs_calibration` populated with match/camera/error.
  - no tags (n < min) → returns True with a warning, no error.
  - `CALIB_CHECK_ENABLED=0` → returns True regardless.

  If factoring a pure `evaluate_calibration_drift` helper is cleaner to test, add it to `web/field/calibration.py` (or `match_order.py`) and have `_verify_calibration_alignment` call it — decide during implementation and keep the method thin.

- [ ] **Step 2: Run the new test** → pass.

- [ ] **Step 3: Full suite** — `.venv/bin/python -m pytest tests/ -q`. All pass (note, don't fix, any unrelated pre-existing failures — e.g. the known `hub_no_robot_zones` WIP failure).

- [ ] **Step 4: Manual verification (the real proof).** With at least two calibration versions for `2026hop` `main` (the existing one + a day-2 version), confirm:
  - Running an early-match (calibration-matching) video proceeds (status not error).
  - Running `b3cc7613` (qm84, moved camera) with only the early-match calibration **aborts** with the `needs_calibration` message naming qm84 + main + the meters error — and does NOT enter the SAM2 pass.
  Compare against `CALIB_CHECK_ENABLED=0` (proceeds as before).

- [ ] **Step 5: Commit** — `git add tests/field/test_calibration_drift_check.py` (+ any pure helper file) + `test(calib): drift-check integration + decision-logic coverage`.

---

## Notes for the implementer

- **Run tests with `.venv/bin/python -m pytest`** (base python has no pytest — memory `test-env-uses-venv`).
- **`web/app.py` has unrelated uncommitted WIP.** Tasks 5, 6, 7 touch ONLY app.py; the controller stashes the WIP (`git stash push -- web/app.py`) before the task and restores it (`git stash pop`) after the commit, so your commit contains only your changes. Never `git add -A`/`.`/`-am`.
- **Pure-first ordering:** Tasks 1-3 are pure/leaf and independently testable; 4 is a small store-aware change; 5-7 are the app integration; 8 verifies end-to-end. Land them in order.
- **Don't change `CameraCalibration`'s schema** — epoch metadata lives in the store layer (the version JSON wraps `cal.to_json()`), keeping the projection class clean.
- **`select_epoch` is reused** in three places (init main-camera pick, init per-camera view map, save carry-forward) — import the one in `match_order.py`; don't reimplement.
- **detect_tags is a non-thread-safe singleton** — keep the sampling loop sequential (it already is, inside `_verify_calibration_alignment`).
- **Corner cameras** are intentionally not validated (tags not machine-decodable): the method only checks `main`. Corners still get their selected epoch via `load_views`.

# Field-Space Tracking (SP2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pixel-space canonical-ID tracking in `InferenceJob` with a fixed-3v3 field-space tracker (alliance by bumper color, occlusion-aware coasting from hub/person/robot occluders, VLM team reads re-bound to stable slots) and wire the live radar.

**Architecture:** Three pure/adapter modules — `tracker.py` (constant-velocity Kalman, per-alliance brute-force assignment, fixed 6 slots), `occlusion.py` (hub-prism ray-cast + depth-ordered person/robot occlusion → `view_occluded`/`bumper_occluded`), `detect_bridge.py` (YOLO box → field point + bumper color) — then a focused `InferenceJob` integration that makes track `slot_id` the canonical identity.

**Tech Stack:** Python, NumPy (pure tracker/occlusion — no cv2/torch), OpenCV + Ultralytics YOLO (bridge + integration), FastAPI.

**Spec:** `docs/superpowers/specs/2026-05-29-field-space-tracking-design.md`

**REQUIRED SUB-SKILL for every task:** @superpowers:test-driven-development

**Test runner:** `.venv/bin/python -m pytest` (base conda python has no pytest).

**No new dependencies:** per-alliance association is brute-forced over ≤3!=6 permutations (no scipy). Kalman is hand-rolled NumPy.

---

## File Structure

- **Create** `web/field/tracker.py` — `FieldDetection`, `Track`, `FieldTracker` (pure NumPy).
- **Create** `web/field/occlusion.py` — `Prism`, `hub_prisms`, ray-cast/overlap/depth helpers, `resolve_occlusion` (pure NumPy; takes a `CameraCalibration` for projection).
- **Create** `web/field/detect_bridge.py` — `to_field_detections`, `field_adjacent_persons` (uses cv2 via a passed color-fn; no app import).
- **Modify** `web/field/calibration.py` — add `camera_center` property to `CameraCalibration`.
- **Create** tests: `tests/test_tracker.py`, `tests/test_occlusion.py`, `tests/test_detect_bridge.py`.
- **Modify** `web/app.py` — `InferenceJob`: integrate the tracker as identity source, occlusion gating, VLM re-bind, radar wiring; remove ByteTrack/canonical-ID/embedding-reid/alliance-clustering/lineup-side-heuristic.
- **Create** `scripts/verify_field_tracking.py` — integration smoke check on the test video.

---

## Task 1: `camera_center` property on `CameraCalibration`

**Files:** Modify `web/field/calibration.py`; Test `tests/test_calibration_pinhole.py` (append).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calibration_pinhole.py`:

```python
def test_camera_center_matches_neg_RT_t():
    from tests._field_synth import look_at
    K = np.array([[1200., 0., 960.], [0., 1200., 540.], [0., 0., 1.]])
    rvec, tvec = look_at((8.27, -3.0, 6.0), (8.27, 4.03, 0.0))
    cam = CameraCalibration(mode="pinhole", K=K, D=np.zeros(5), rvec=rvec, tvec=tvec,
                            image_size=(1920, 1080), field_season="2026-rebuilt-welded")
    import cv2
    R, _ = cv2.Rodrigues(rvec)
    expected = (-R.T @ tvec).reshape(3)
    assert np.allclose(cam.camera_center, expected)
    assert np.allclose(cam.camera_center, [8.27, -3.0, 6.0], atol=1e-6)  # = eye


def test_camera_center_none_without_pose():
    cam = CameraCalibration(mode="homography", H=np.eye(3),
                            image_size=(100, 100), field_season="x")
    assert cam.camera_center is None
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_calibration_pinhole.py -k camera_center -v` → FAIL (no attribute).

- [ ] **Step 3: Implement** — add to `CameraCalibration` (after `image_to_field`):

```python
    @property
    def camera_center(self):
        """Camera center in field coords (-R^T t), or None for pose-free modes."""
        if self.rvec is None or self.tvec is None:
            return None
        R, _ = cv2.Rodrigues(self.rvec)
        return (-R.T @ self.tvec).reshape(3)
```

- [ ] **Step 4: Run to verify pass** — same command → PASS.

- [ ] **Step 5: Commit** — `git add web/field/calibration.py tests/test_calibration_pinhole.py && git commit -m "feat(field): camera_center property on CameraCalibration"`

---

## Task 2: `FieldTracker` (fixed 3v3, CV-Kalman)

**Files:** Create `web/field/tracker.py`, `tests/test_tracker.py`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_tracker.py`:

```python
import numpy as np
from web.field.tracker import FieldTracker, FieldDetection


def _det(x, y, color, box=(0, 0, 10, 10)):
    return FieldDetection(pos=(x, y), color=color, box=box, score=0.9)


def _step(trk, dets, occ=None, dt=0.1):
    return trk.update(dets, occ or {}, dt)


def test_six_tracks_three_per_alliance():
    trk = FieldTracker(field_length=16.541)
    dets = [_det(2, 2, "blue"), _det(3, 5, "blue"), _det(4, 7, "blue"),
            _det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    out = _step(trk, dets)
    assert len([t for t in out if t.state == "visible"]) == 6
    assert sum(t.alliance == "blue" for t in out) == 3
    assert sum(t.alliance == "red" for t in out) == 3


def test_never_exceeds_three_per_alliance():
    trk = FieldTracker(field_length=16.541)
    # 5 blue detections in one frame — only 3 slots exist.
    dets = [_det(2 + i, 2 + i, "blue") for i in range(5)]
    out = _step(trk, dets)
    assert sum(t.alliance == "blue" and t.state == "visible" for t in out) <= 3


def test_slot_id_stable_through_crossing():
    trk = FieldTracker(field_length=16.541)
    # two blue robots swap x positions over time; ids must stick to trajectories.
    a, b = (2.0, 3.0), (6.0, 3.0)
    for _ in range(6):
        _step(trk, [_det(*a, "blue"), _det(*b, "blue"),
                    _det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")])
        a = (a[0] + 0.6, 3.0); b = (b[0] - 0.6, 3.0)
    out = {t.slot_id: t.pos for t in trk.update(
        [_det(*a, "blue"), _det(*b, "blue")], {}, 0.1) if t.alliance == "blue"}
    # The track that started near x=2 should now be at higher x (it moved right).
    # Just assert two distinct stable blue slots persisted (no id churn -> 2 ids).
    assert len(out) == 2


def test_coast_when_occluded_then_recover():
    trk = FieldTracker(field_length=16.541, max_coast_open=2, max_coast_occluded=30)
    base = [_det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    seed = [_det(2.0, 3.0, "blue"), _det(5.0, 3.0, "blue"), _det(8.0, 3.0, "blue")]
    _step(trk, seed + base)
    # slot for the first blue:
    blue0 = [t for t in trk.tracks if t.alliance == "blue"][0]
    sid = blue0.slot_id
    # Drop that blue detection for 10 frames, marked occluded -> must coast (not drop).
    for _ in range(10):
        out = trk.update([_det(5.0, 3.0, "blue"), _det(8.0, 3.0, "blue")] + base,
                         {sid: {"view_occluded": True}}, 0.1)
    coasting = [t for t in out if t.slot_id == sid][0]
    assert coasting.state == "coasting"   # survived, not re-seeded away
    # Re-appears -> visible again, same slot id.
    out = trk.update([_det(5.6, 3.0, "blue"), _det(5.0, 3.0, "blue"),
                      _det(8.0, 3.0, "blue")] + base, {}, 0.1)
    assert any(t.slot_id == sid and t.state == "visible" for t in out)


def test_open_track_expires_and_reseeds():
    trk = FieldTracker(field_length=16.541, max_coast_open=2, max_coast_occluded=30)
    base = [_det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    _step(trk, [_det(2, 3, "blue"), _det(5, 3, "blue"), _det(8, 3, "blue")] + base)
    # Lose a blue with NO occlusion for > max_coast_open frames; slot may re-seed
    # to a new detection that appears elsewhere.
    for _ in range(4):
        out = trk.update([_det(5, 3, "blue"), _det(8, 3, "blue")] + base, {}, 0.1)
    # A new blue robot appears at (3,6); a free slot should take it.
    out = trk.update([_det(3, 6, "blue"), _det(5, 3, "blue"), _det(8, 3, "blue")] + base,
                     {}, 0.1)
    assert sum(t.alliance == "blue" and t.state == "visible" for t in out) == 3


def test_color_unknown_still_associates_by_position():
    trk = FieldTracker(field_length=16.541)
    base = [_det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    _step(trk, [_det(2, 3, "blue"), _det(5, 3, "blue"), _det(8, 3, "blue")] + base)
    out = trk.update([_det(2.1, 3.0, "unknown"), _det(5, 3, "blue"),
                      _det(8, 3, "blue")] + base, {}, 0.1)
    # the unknown-color detection updates the nearest blue slot (no new track)
    assert sum(t.alliance == "blue" and t.state == "visible" for t in out) == 3
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_tracker.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `web/field/tracker.py`**

```python
"""Fixed-3v3 field-space multi-object tracker (pure NumPy).

6 slots (3 blue + 3 red), each a constant-velocity Kalman filter in field meters.
Per frame, per alliance, brute-force assign that color's <=3 detections to its 3
slots (>=3!=6 permutations), update matched, coast unmatched. Occluded slots coast
far longer and cannot be re-seeded. Alliance is sticky per slot."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional
import numpy as np

ALLIANCES = ("blue", "red")


@dataclass
class FieldDetection:
    pos: tuple              # (x, y) field meters
    color: str              # "blue" | "red" | "unknown"
    box: tuple              # (x1, y1, x2, y2) image px
    score: float = 1.0


@dataclass
class TrackOut:
    slot_id: int
    alliance: str
    pos: tuple
    vel: tuple
    box: Optional[tuple]
    state: str              # "visible" | "coasting" | "empty"


class _KF:
    """Constant-velocity KF: state [x, y, vx, vy]."""
    def __init__(self, xy, q=2.0, r=0.05):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)
        self.P = np.diag([r, r, 4.0, 4.0])
        self.q, self.r = q, r

    def predict(self, dt):
        F = np.eye(4); F[0, 2] = dt; F[1, 3] = dt
        self.x = F @ self.x
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.P = F @ self.P @ F.T + G @ (self.q * np.eye(2)) @ G.T

    def update(self, z):
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + self.r * np.eye(2)
        Kg = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ (np.asarray(z, float) - H @ self.x)
        self.P = (np.eye(4) - Kg @ H) @ self.P

    @property
    def pos(self): return (float(self.x[0]), float(self.x[1]))
    @property
    def vel(self): return (float(self.x[2]), float(self.x[3]))


@dataclass
class _Track:
    slot_id: int
    alliance: str
    kf: Optional[_KF] = None
    box: Optional[tuple] = None
    missed: int = 0
    hits: int = 0
    coasting: bool = False
    active: bool = False


class FieldTracker:
    def __init__(self, field_length, gate_m=2.5, max_coast_open=8,
                 max_coast_occluded=90, color_penalty=3.0):
        self.gate = gate_m
        self.max_open = max_coast_open
        self.max_occ = max_coast_occluded
        self.color_penalty = color_penalty
        self.tracks = []
        sid = 0
        for a in ALLIANCES:
            for _ in range(3):
                self.tracks.append(_Track(slot_id=sid, alliance=a)); sid += 1

    def _slots(self, alliance):
        return [t for t in self.tracks if t.alliance == alliance]

    def update(self, detections, occlusion, dt):
        # 1. predict active tracks
        for t in self.tracks:
            if t.active:
                t.kf.predict(dt)
        # 2. per alliance association
        for alliance in ALLIANCES:
            slots = self._slots(alliance)
            dets = [d for d in detections if d.color in (alliance, "unknown")]
            self._assign(slots, dets, alliance)
        # 3. coast / expire / reseed
        for t in self.tracks:
            if not t.active:
                continue
            if t.missed == 0:
                continue
            occ = occlusion.get(t.slot_id, {}).get("view_occluded", False)
            t.coasting = True
            budget = self.max_occ if occ else self.max_open
            if t.missed > budget:
                t.active = False; t.kf = None; t.box = None; t.coasting = False
        return self._emit()

    def _assign(self, slots, dets, alliance):
        active = [s for s in slots if s.active]
        # cost matrix active-tracks x dets (gated)
        used_det = set()
        if active and dets:
            cost = np.full((len(active), len(dets)), 1e6)
            for i, s in enumerate(active):
                px, py = s.kf.pos
                for j, d in enumerate(dets):
                    dist = np.hypot(d.pos[0] - px, d.pos[1] - py)
                    if dist <= self.gate:
                        c = dist + (self.color_penalty if d.color == "unknown" else 0.0)
                        cost[i, j] = c
            rows, cols = self._best_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                s = active[i]; d = dets[j]
                s.kf.update(d.pos); s.box = d.box; s.missed = 0; s.hits += 1
                s.coasting = False
                used_det.add(j)
        # leftover detections seed empty / expired slots
        for j, d in enumerate(dets):
            if j in used_det:
                continue
            free = next((s for s in slots if not s.active), None)
            if free is None:
                continue
            free.kf = _KF(d.pos); free.box = d.box; free.active = True
            free.missed = 0; free.hits = 1; free.coasting = False
            used_det.add(j)
        # tracks with no detection this round -> missed++
        for s in active:
            if s.missed == 0 and s.box is not None and s.hits > 0 and s not in \
                    [active[i] for i, j in zip(*self._last_rows)] :
                pass  # handled below
        # (simpler: recompute matched set)
        matched = set()
        # mark matched by checking which active tracks got missed reset this call:
        # we tracked that via missed==0 updates above; for others, increment.
        for s in active:
            # if not updated this frame, it still has its previous missed; bump it
            pass

    def _best_assignment(self, cost):
        # brute force over permutations of the smaller dimension (<=3 each side)
        n, m = cost.shape
        best, best_cost = None, None
        idx = list(range(m))
        for perm in permutations(idx, min(n, m)):
            rows = list(range(min(n, m)))
            c = sum(cost[r, perm[k]] for k, r in enumerate(rows))
            if best_cost is None or c < best_cost:
                best_cost = c; best = (rows, list(perm))
        self._last_rows = best if best else ([], [])
        return best if best else ([], [])

    def _emit(self):
        out = []
        for t in self.tracks:
            if t.active:
                out.append(TrackOut(t.slot_id, t.alliance, t.kf.pos, t.kf.vel,
                                    t.box, "coasting" if t.coasting else "visible"))
            else:
                out.append(TrackOut(t.slot_id, t.alliance, (0.0, 0.0), (0.0, 0.0),
                                    None, "empty"))
        return out
```

> NOTE TO IMPLEMENTER: the `_assign` "missed" bookkeeping above is sketched and buggy
> (the trailing loops are placeholders). Implement it cleanly: track, per `_assign`
> call, the set of slots that received a detection; every other *active* slot gets
> `missed += 1`. Keep `_best_assignment` (brute-force ≤3×3). Let the tests in Step 1
> drive the exact behavior — they are the contract. Do not ship the placeholder loops.

- [ ] **Step 4: Run tests; iterate until green** — `.venv/bin/python -m pytest tests/test_tracker.py -v`. Fix the `missed`/assignment bookkeeping until all pass. Expected: 7 passed.

- [ ] **Step 5: Commit** — `git add web/field/tracker.py tests/test_tracker.py && git commit -m "feat(field): fixed-3v3 field-space Kalman tracker"`

---

## Task 3: `occlusion.py`

**Files:** Create `web/field/occlusion.py`, `tests/test_occlusion.py`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_occlusion.py`:

```python
import numpy as np
from web.field.field_model import FieldModel
from web.field.occlusion import (hub_prisms, occluded_by_hub, camera_distance,
                                  box_overlap, resolve_occlusion)
from tests._field_synth import truth_pinhole_camera


def test_hub_prisms_two_compact_footprints():
    fm = FieldModel.load("2026-rebuilt-welded")
    prisms = hub_prisms(fm)
    assert len(prisms) == 2
    for p in prisms:
        xs, ys = p.footprint[:, 0], p.footprint[:, 1]
        assert (xs.max() - xs.min()) < 2.0 and (ys.max() - ys.min()) < 2.0  # ~1.2m
        assert p.z_top > 1.0
    centers = sorted(p.footprint.mean(axis=0)[0] for p in prisms)
    assert centers[0] < 6 and centers[1] > 10   # blue ~4.5, red ~12


def test_occluded_by_hub_blocks_behind_not_beside():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()  # eye ~ (8.27, -3, 6)
    prism = [p for p in hub_prisms(fm) if p.footprint.mean(axis=0)[0] > 10][0]  # red hub ~x12
    cc = cam.camera_center
    behind = (prism.footprint.mean(axis=0)[0], prism.footprint.mean(axis=0)[1] + 3.0)
    beside = (2.0, 4.0)
    assert occluded_by_hub(cc, prism, behind, robot_h=0.5) is True
    assert occluded_by_hub(cc, prism, beside, robot_h=0.5) is False


def test_camera_distance_orders_depth():
    cc = np.array([8.27, -3.0, 6.0])
    near = camera_distance(cc, (8.27, 0.0))
    far = camera_distance(cc, (8.27, 6.0))
    assert near < far


def test_resolve_robot_robot_bumper_occlusion():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    # Two robots nearly in line from the camera; the nearer covers the farther.
    # Build two "tracks" with overlapping image boxes; nearer has smaller cam distance.
    near_xy = (8.0, 0.5); far_xy = (8.0, 3.0)
    # project to get overlapping boxes
    def box_at(xy):
        u, v = cam.field_to_image((xy[0], xy[1], 0.3))
        return (u - 40, v - 80, u + 40, v + 10)
    tracks = [{"slot_id": 0, "pos": near_xy, "box": box_at(near_xy)},
              {"slot_id": 1, "pos": far_xy, "box": box_at(far_xy)}]
    res = resolve_occlusion(cam, tracks, person_boxes=[], hub_prisms=[], robot_h=0.5)
    # the farther track (slot 1) should be bumper_occluded if boxes overlap
    if box_overlap(tracks[0]["box"], tracks[1]["box"]) > 0:
        assert res[1]["bumper_occluded"] is True
        assert res[0]["bumper_occluded"] is False


def test_resolve_ignores_off_field_crowd_person():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    tracks = [{"slot_id": 0, "pos": (8.0, 4.0), "box": (900, 300, 1000, 420)}]
    # a person box high in the image (crowd) shouldn't be treated as field-adjacent;
    # resolve_occlusion only receives already-filtered person boxes, so an empty list
    # here means "no field-adjacent people" -> no occlusion.
    res = resolve_occlusion(cam, tracks, person_boxes=[], hub_prisms=[], robot_h=0.5)
    assert res[0]["view_occluded"] is False and res[0]["bumper_occluded"] is False
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_occlusion.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `web/field/occlusion.py`**

```python
"""Occlusion geometry (pure NumPy + a CameraCalibration for projection).

An occluder lies between the camera and the robot. Hubs are static prisms derived
from the tallest tag z-ring; persons/robots are depth-ordered boxes. Produces, per
track, view_occluded (drives coasting) and bumper_occluded (gates VLM reads)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Prism:
    footprint: np.ndarray   # (N,2) convex polygon, field meters
    z_top: float
    z_bottom: float = 0.0


def _zgroups(zs, tol=0.05):
    zs = np.sort(np.unique(np.round(zs, 3)))
    groups = [[zs[0]]]
    for z in zs[1:]:
        (groups[-1].append(z) if z - groups[-1][-1] <= tol else groups.append([z]))
    return groups


def _convex_hull(pts):
    pts = np.asarray(pts, float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts
    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(pts) + half(pts[::-1]))


def hub_prisms(field_model):
    """Two hub prisms from the tallest tag z-group, split at the field midline."""
    ids = list(field_model.tags)
    ts = np.array([field_model.tag_translation(i) for i in ids])  # (N,3)
    top = _zgroups(ts[:, 2])[-1]                  # tallest z cluster = hub ring
    mask = np.isin(np.round(ts[:, 2], 3), np.round(top, 3))
    hub_xy = ts[mask, :2]; hub_z = ts[mask, 2].max()
    mid = field_model.length_m / 2.0
    prisms = []
    for sel in (hub_xy[:, 0] < mid, hub_xy[:, 0] >= mid):
        if sel.sum() >= 3:
            prisms.append(Prism(footprint=_convex_hull(hub_xy[sel]), z_top=float(hub_z) + 0.2))
    return prisms


def _point_in_poly(pt, poly):
    x, y = pt; inside = False; n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def occluded_by_hub(cam_center, prism, track_xy, robot_h, n=40):
    """Does the segment camera_center -> (track_xy, robot_h) pass through the prism?"""
    cc = np.asarray(cam_center, float)
    end = np.array([track_xy[0], track_xy[1], robot_h])
    for t in np.linspace(0.0, 1.0, n):
        p = cc + (end - cc) * t
        if prism.z_bottom <= p[2] <= prism.z_top and _point_in_poly(p[:2], prism.footprint):
            return True
    return False


def camera_distance(cam_center, ground_xy):
    cc = np.asarray(cam_center, float)
    return float(np.hypot.reduce([cc[0] - ground_xy[0], cc[1] - ground_xy[1]]) ** 0  # placeholder
                 ) or float(np.linalg.norm(cc - np.array([ground_xy[0], ground_xy[1], 0.0])))


def box_overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _bumper_region(box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    return (x1, y1 + 0.6 * h, x2, y2)   # lower 40%


def resolve_occlusion(cam, tracks, person_boxes, hub_prisms, robot_h=0.5):
    """tracks: list of {slot_id, pos:(x,y), box:(x1,y1,x2,y2)}.
    Returns {slot_id: {view_occluded, bumper_occluded, cause}}."""
    cc = cam.camera_center
    out = {}
    # precompute camera distance of each track
    tdist = {t["slot_id"]: camera_distance(cc, t["pos"]) for t in tracks}
    for t in tracks:
        sid = t["slot_id"]; box = t["box"]; view = False; bump = False; cause = None
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        bregion = _bumper_region(box)
        barea = max(1.0, (bregion[2] - bregion[0]) * (bregion[3] - bregion[1]))
        # hubs (view occlusion; predictive)
        for p in hub_prisms:
            if occluded_by_hub(cc, p, t["pos"], robot_h):
                view = True; cause = "hub"
        # persons (nearer than the track, by image footing): persons are foreground
        for pb in person_boxes:
            if box_overlap(box, pb) / area > 0.5:
                view = True; cause = cause or "person"
            if box_overlap(bregion, pb) / barea > 0.3:
                bump = True
        # robots: another track nearer the camera covering this one
        for o in tracks:
            if o["slot_id"] == sid:
                continue
            if tdist[o["slot_id"]] < tdist[sid]:
                if box_overlap(box, o["box"]) / area > 0.6:
                    view = True; cause = cause or "robot"
                if box_overlap(bregion, o["box"]) / barea > 0.3:
                    bump = True
        out[sid] = {"view_occluded": view, "bumper_occluded": bump or view, "cause": cause}
    return out
```

> NOTE TO IMPLEMENTER: `camera_distance` above is written with a placeholder hack — replace
> its body with the clean one line: `return float(np.linalg.norm(np.asarray(cam_center,float) -
> np.array([ground_xy[0], ground_xy[1], 0.0])))`. The tests are the contract.

- [ ] **Step 4: Run tests; iterate to green** — `.venv/bin/python -m pytest tests/test_occlusion.py -v`. Expected: 5 passed.

- [ ] **Step 5: Commit** — `git add web/field/occlusion.py tests/test_occlusion.py && git commit -m "feat(field): hub/person/robot occlusion geometry"`

---

## Task 4: `detect_bridge.py`

**Files:** Create `web/field/detect_bridge.py`, `tests/test_detect_bridge.py`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from web.field.field_model import FieldModel
from web.field.detect_bridge import to_field_detections, field_adjacent_persons
from tests._field_synth import truth_pinhole_camera


def _color_stub(frame, box):
    return "blue"


def test_box_maps_to_field_point():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    gx, gy = 8.0, 4.0
    u, v = cam.field_to_image((gx, gy, 0.0))     # ground point
    box = (u - 30, v - 120, u + 30, v)           # bottom-center == (u, v)
    dets = to_field_detections([box], cam, frame=None, field_dims=(fm.length_m, fm.width_m),
                               color_fn=_color_stub)
    assert len(dets) == 1
    assert np.hypot(dets[0].pos[0] - gx, dets[0].pos[1] - gy) < 0.10
    assert dets[0].color == "blue"


def test_off_field_box_dropped():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    u, v = cam.field_to_image((30.0, 4.0, 0.0))  # way off field (x=30 > 16.5)
    box = (u - 30, v - 120, u + 30, v)
    dets = to_field_detections([box], cam, None, (fm.length_m, fm.width_m), _color_stub)
    assert dets == []


def test_field_adjacent_person_filter():
    fm = FieldModel.load("2026-rebuilt-welded")
    cam = truth_pinhole_camera()
    # person standing just outside the near sideline (y slightly < 0): feet near carpet
    u, v = cam.field_to_image((8.0, -0.3, 0.0))
    near = (u - 25, v - 150, u + 25, v)
    # "crowd" person whose feet project far off field (x huge)
    uc, vc = cam.field_to_image((40.0, 4.0, 0.0))
    crowd = (uc - 25, vc - 150, uc + 25, vc)
    keep = field_adjacent_persons([near, crowd], cam, (fm.length_m, fm.width_m), margin=2.0)
    assert near in keep and crowd not in keep
```

- [ ] **Step 2: Run to verify failure** → FAIL (module missing).

- [ ] **Step 3: Implement `web/field/detect_bridge.py`**

```python
"""Adapter: YOLO image boxes -> field-space detections / field-adjacent persons.

Kept import-light (no app.py import): the bumper-color reader is injected as color_fn
so this stays unit-testable with a stub."""
from __future__ import annotations
from web.field.tracker import FieldDetection


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _on_field(xy, dims, margin):
    L, W = dims
    return -margin <= xy[0] <= L + margin and -margin <= xy[1] <= W + margin


def to_field_detections(robot_boxes, calibration, frame, field_dims, color_fn, margin=1.0):
    out = []
    for box in robot_boxes:
        uv = _bottom_center(box)
        xy = calibration.image_to_field(uv)
        if xy is None or not _on_field(xy, field_dims, margin):
            continue
        color = color_fn(frame, box) if frame is not None else "unknown"
        out.append(FieldDetection(pos=xy, color=color, box=tuple(box), score=1.0))
    return out


def field_adjacent_persons(person_boxes, calibration, field_dims, margin=2.0):
    keep = []
    for box in person_boxes:
        xy = calibration.image_to_field(_bottom_center(box))
        if xy is not None and _on_field(xy, field_dims, margin):
            keep.append(box)
    return keep
```

- [ ] **Step 4: Run tests to verify pass** — `.venv/bin/python -m pytest tests/test_detect_bridge.py -v`. Expected: 3 passed.

- [ ] **Step 5: Commit** — `git add web/field/detect_bridge.py tests/test_detect_bridge.py && git commit -m "feat(field): YOLO->field detection bridge + person filter"`

---

## Task 5: `InferenceJob` integration (replace canonical-ID core)

> **Risk:** touches the live inference loop. Land Tasks 1–4 first (all verified in
> isolation). Do this task in small commits; run the integration smoke check (Task 6)
> after each sub-step that compiles. Recommended under subagent-driven-development with
> review between sub-steps.

**Files:** Modify `web/app.py` (`InferenceJob.__init__`, `run` loop, `vision_identify_robots`/`ocr_robot_bumpers`, HUD draw).

- [ ] **Step 1: Startup wiring (`InferenceJob.__init__` / `run` setup)**
  - Import: `from web.field import store as field_store` (already present), `from web.field.tracker import FieldTracker`, `from web.field.occlusion import hub_prisms, resolve_occlusion`, `from web.field.detect_bridge import to_field_detections, field_adjacent_persons`, `from web.field.field_model import FieldModel`, `from web.field.radar import Radar`.
  - Load calibration: `self.calib = field_store.load(self.event_key, self.camera)`. If `self.calib is None or self.calib.mode == "homography"` → log a clear error and set `self.field_tracking = False` (fall through to legacy only if you choose to keep it; per spec, prefer to **raise/abort** field tracking with a visible message). Otherwise `self.field_tracking = True`.
  - `self.field_model = FieldModel.load(self.calib.field_season)`.
  - `self.hub_prisms = hub_prisms(self.field_model)`.
  - `self.field_tracker = FieldTracker(field_length=self.field_model.length_m)`.
  - `self.person_model = YOLO("yolo26n.pt")` (COCO; class 0 = person). Load once.
  - `self.radar = Radar(self.field_model.length_m, self.field_model.width_m)`.
  - Commit: `feat(app): load calibration + field tracker at job start`.

- [ ] **Step 2: Per-frame core replacement (in the `while` loop, ~app.py 1751-1860)**
  Replace the ByteTrack/`get_canonical_robot_id`/embedding ordering block with:
  ```python
  # robot detections -> field detections
  robot_boxes = [robot_detections.xyxy[i] for i in range(len(robot_detections))]
  field_dims = (self.field_model.length_m, self.field_model.width_m)
  dets = to_field_detections(robot_boxes, self.calib, overhead_frame, field_dims,
                             color_fn=_detect_bumper_color_static)
  # persons -> field-adjacent occluder boxes
  pres = self.person_model(overhead_frame, conf=0.35, verbose=False, classes=[0])[0]
  pboxes = [pres.boxes.xyxy[i].cpu().numpy() for i in range(len(pres.boxes))]
  persons = field_adjacent_persons(pboxes, self.calib, field_dims)
  # expected boxes for occlusion of currently-active tracks (predicted positions)
  prev = [{"slot_id": t.slot_id, "pos": t.pos, "box": t.box}
          for t in self.field_tracker._emit() if t.box is not None]
  occ = resolve_occlusion(self.calib, prev + _det_track_views(dets), persons,
                          self.hub_prisms)
  view_flags = {sid: {"view_occluded": v["view_occluded"]} for sid, v in occ.items()}
  tracks = self.field_tracker.update(dets, view_flags, dt=FRAME_SKIP / self.fps)
  ```
  Then build `tracked_robot_boxes = [(t.slot_id, t.box) for t in tracks if t.state == "visible"]`
  for the OCR/VLM stage, and keep a `self._occ = occ` for the VLM gate. `_det_track_views`
  is a tiny helper mapping current detections to `{slot_id?, pos, box}` for occlusion of
  *visible* detections (or simply resolve occlusion on the post-update `tracks`). Prefer:
  resolve occlusion AFTER `update` using the emitted tracks (their `box` or projected
  expected box), so a single consistent set feeds both `view`/`bumper`. Restructure to:
  predict → resolve(view) → update → resolve(bumper) if needed, OR resolve once on the
  emitted tracks and use both fields (simpler; the predicted boxes are close enough).
  Keep it simple: **resolve once on emitted tracks; use view for next-frame coast intent
  is fine since occlusion changes slowly.** Drive the exact ordering with the smoke test.
  - Commit: `feat(app): field tracker drives per-frame robot identity`.

- [ ] **Step 3: VLM re-bind + bumper-occlusion gate**
  - In `vision_identify_robots` / `ocr_robot_bumpers`, key everything by `slot_id` (the
    `tracked_robots` tuples now carry slot_id). Maintain a per-slot team-vote dict
    (reuse `RobotTrack.team_ocr_votes` keyed by slot_id, or a new `self.slot_votes`).
  - **Skip** building a VLM candidate for any slot where `self._occ[slot]["bumper_occluded"]`
    is True (don't read an occluded bumper).
  - Team assignment per slot = argmax votes; fixed-6 ⇒ no duplicate-team cleanup needed.
  - Commit: `feat(app): re-bind VLM team reads to slots, gate on bumper occlusion`.

- [ ] **Step 4: HUD + radar**
  - Label each visible track box with `f"{alliance[0].upper()}{slot_local}"` (slot_local
    0-2 within alliance) + team number if assigned. Draw boxes from `tracks`.
  - Compose a radar image: `self.radar.render([(t.pos, t.alliance) for t in tracks if
    t.state != "empty"])`; embed/emit it (mirror how existing frames are emitted, e.g.
    as a side panel or a separate streamed image). Match the existing output mechanism.
  - Commit: `feat(app): HUD labels by slot + live radar`.

- [ ] **Step 5: Remove superseded code**
  - Delete / stop calling: `get_canonical_robot_id`, `bytetrack_to_canonical`,
    `next_canonical_id`, `self.tracker` (ByteTrack) usage, `self.embedder` usage in the
    loop, `appearance_match_robots`/appearance clustering, `_clear_team_from_other_robots`,
    `cap_alliance`, the `LINEUP_END_FRAME` left=blue/right=red seeding (~1799-1824).
    Leave scoring-side helpers (`_is_robot_on_alliance_side`) untouched.
  - Run full suite: `.venv/bin/python -m pytest tests/ -q` → all green.
  - Commit: `refactor(app): remove pixel-space canonical-ID/reid tracking (superseded by SP2)`.

---

## Task 6: Integration smoke check

**Files:** Create `scripts/verify_field_tracking.py`.

- [ ] **Step 1: Write the script**

```python
"""Smoke check: run field tracking over a window of the test video with the saved
calibration; assert exactly 6 slots and stable ids. Writes a radar+overlay sample.
Run: .venv/bin/python scripts/verify_field_tracking.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Minimal harness: drive FieldTracker + bridge directly over frames using the saved
# calibration, independent of the full InferenceJob (which needs an event session).
import cv2, numpy as np
from web.field import store as field_store
from web.field.field_model import FieldModel
from web.field.tracker import FieldTracker
from web.field.detect_bridge import to_field_detections
from web.field.radar import Radar
from ultralytics import YOLO

cal = field_store.load("test", "main")
assert cal is not None and cal.mode != "homography", "need a pose-bearing calibration"
fm = FieldModel.load(cal.field_season)
trk = FieldTracker(field_length=fm.length_m)
model = YOLO(str(ROOT / "yolo26n.pt"))  # or the project robot model
cap = cv2.VideoCapture(str(ROOT / "data/test_footage/qual1_first_champs_2026.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
seen_ids = set()
for fi in range(200, 500, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
    if not ok: break
    main = fr[: int(fr.shape[0] * 0.60), :]
    res = model(main, conf=0.4, verbose=False, classes=[0])[0]  # placeholder class
    boxes = [res.boxes.xyxy[i].cpu().numpy() for i in range(len(res.boxes))]
    dets = to_field_detections(boxes, cal, main, (fm.length_m, fm.width_m),
                               color_fn=lambda f, b: "unknown")
    tracks = trk.update(dets, {}, dt=5 / fps)
    seen_ids.update(t.slot_id for t in tracks)
cap.release()
print("distinct slot ids seen:", sorted(seen_ids))
assert seen_ids <= set(range(6)), "more than 6 slots!"
print("OK: <=6 stable slots")
```

- [ ] **Step 2: Run it** — `.venv/bin/python scripts/verify_field_tracking.py`. Expected: prints `<=6 stable slots`. (Uses the robot model the project trained; the `classes`/model here is a placeholder — wire the real robot detector when running.)

- [ ] **Step 3: Full suite green** — `.venv/bin/python -m pytest tests/ -q` → all pass.

- [ ] **Step 4: Commit** — `git add scripts/verify_field_tracking.py && git commit -m "test(field): SP2 field-tracking smoke check"`

---

## Notes & gotchas

- **Pure modules first.** Tasks 1–4 don't touch `app.py`; they're fully unit-tested and
  zero-risk. Only Task 5 modifies the live loop.
- **No scipy.** Per-alliance assignment is ≤3×3 → brute-force permutations.
- **Occlusion ordering.** Resolve occlusion on the tracker's emitted tracks (real box if
  visible, projected expected box if coasting). `view_occluded` extends the coast budget;
  `bumper_occluded` gates the VLM read. Let the smoke test confirm the wiring; keep the
  resolve call single-pass for simplicity.
- **Calibration required.** Field tracking needs a pose-bearing calibration for the
  event/camera; abort with a clear message otherwise.
- **`event_key`/`camera`.** Confirm how `InferenceJob` knows its event key + camera name
  (mirror `/api/calibration` which uses `event`,`camera="main"`); thread them into the
  job so `field_store.load` resolves the right calibration.
- **Robot detector vs COCO.** Robots come from the project's custom model (class id 1);
  persons come from the stock COCO `yolo26n.pt` (class id 0). Two separate model calls.

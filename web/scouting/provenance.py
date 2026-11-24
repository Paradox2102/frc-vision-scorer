"""Where each exported robot position came from: which cameras saw it, how its
team label was decided, and the camera calibrations that placed it.

Three sources, all optional (an old replay may lack any of them):

* ``draw.jsonl.gz`` in the replay dir: the HUD's per-frame draw log. A main
  entry with ``ghost: true`` is the tracker coasting (a prediction, nobody saw
  it); a corner entry with ``slot: null`` is a detection the tracker did not
  associate. So the cameras that saw slot S at frame F are "main" (a non-ghost
  main entry with slot S) plus every corner whose list has an entry with slot
  S. ``slot_cameras`` turns that into one bitmask per (frame, slot).
* DB table ``match_robot_identity_segment``: per-slot label provenance from
  the window-labels stage (``window_digits`` / ``window_lineup`` read the
  number on MAIN-camera tiles; ``window_fill`` / ``window_elim`` propagate a
  label without a read; ``window_none`` = unlabelled). Older replays carry the
  live reader's ``read`` / ``stitched`` / ``none`` / NULL, camera unrecorded.
* The calibration version files (web/field/store.py) named by the replay's
  summary: ``calibration_version`` (main) and ``corner_calibration``.
  ``camera_detail`` turns one version into intrinsics, lens distortion amount
  and the camera pose in field terms.
"""
from __future__ import annotations

import functools
import gzip
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Bit i of a ``cams`` mask = CAMERA_BITS[i]. Other camera names a replay's
# draw log uses (offseason layouts) get the next bits, in sorted order.
CAMERA_BITS = ("main", "corner_left", "corner_right")

# Sources whose label came from reading the robot's number, and on which
# camera: window_labels reads digits / lineup only from main-camera crops.
READ_CAMERA = {"window_digits": "main", "window_lineup": "main"}


# ----------------------------------------------------------------- draw log
@dataclass(frozen=True)
class SlotCameras:
    """One replay's camera bitmasks. ``legend`` gives the bit order (CAMERA_BITS
    first). ``frames`` is the sorted array of frames the draw log holds;
    ``masks[slot]`` is a uint8/uint16 array aligned to it (0 = no camera saw
    the slot in that frame: coasting, or not on the field)."""
    legend: tuple
    frames: np.ndarray
    masks: dict

    def mask(self, frame: int, slot) -> Optional[int]:
        """The bitmask for (frame, slot); None when the draw log has no line
        for that frame (unknown, not "unseen")."""
        i = int(np.searchsorted(self.frames, frame))
        if i >= len(self.frames) or int(self.frames[i]) != int(frame):
            return None
        arr = self.masks.get(slot)
        return int(arr[i]) if arr is not None else 0


def draw_log_path(replay_dir) -> Optional[Path]:
    if replay_dir is None:
        return None
    p = Path(replay_dir) / "draw.jsonl.gz"
    return p if p.exists() else None


def slot_cameras(replay_dir) -> Optional[SlotCameras]:
    """Per (frame, slot) camera bitmask from the replay's draw log, or None when
    there is none. Cached on the file's path + mtime + size."""
    p = draw_log_path(replay_dir)
    if p is None:
        return None
    st = p.stat()
    return _slot_cameras_cached(str(p), st.st_mtime, st.st_size)


_DECODER = json.JSONDecoder()


def _draw_line(line: str) -> dict:
    """frame_num / main / corners of one draw-log line. Most of a line is the
    radar trails and the base64 thumbnail, so only those three values are
    decoded (``raw_decode`` at each top-level key: an unescaped ``"main":``
    can't occur inside a JSON string); anything odd falls back to json.loads."""
    try:
        out = {}
        for key in ("frame_num", "main", "corners"):
            i = line.find(f'"{key}":')
            if i < 0:
                continue
            j = i + len(key) + 3
            while line[j] in " \t":
                j += 1
            out[key], _ = _DECODER.raw_decode(line, j)
        if "frame_num" in out:
            return out
    except (ValueError, IndexError):
        pass
    return json.loads(line)


@functools.lru_cache(maxsize=512)
def _slot_cameras_cached(path: str, mtime: float, size: int) -> Optional[SlotCameras]:
    rows: list = []                       # (frame, {slot: set(camera names)})
    names: set = set()
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    d = _draw_line(line)
                    fn = int(d["frame_num"])
                except (ValueError, KeyError, TypeError):
                    continue
                seen: dict = {}
                for e in d.get("main") or []:
                    if e.get("slot") and not e.get("ghost"):
                        seen.setdefault(e["slot"], set()).add("main")
                for cam, lst in (d.get("corners") or {}).items():
                    names.add(cam)
                    for e in lst or []:
                        if e.get("slot"):
                            seen.setdefault(e["slot"], set()).add(cam)
                rows.append((fn, seen))
    except (OSError, EOFError, gzip.BadGzipFile):
        if not rows:
            return None
    legend = CAMERA_BITS + tuple(sorted(names - set(CAMERA_BITS)))
    bit = {n: 1 << i for i, n in enumerate(legend)}
    rows.sort(key=lambda r: r[0])
    frames = np.array([r[0] for r in rows], dtype=np.int64)
    dtype = np.uint8 if len(legend) <= 8 else np.uint32
    slots = {s for _, seen in rows for s in seen}
    masks = {s: np.zeros(len(rows), dtype=dtype) for s in slots}
    for i, (_, seen) in enumerate(rows):
        for s, cams in seen.items():
            masks[s][i] = sum(bit[c] for c in cams)
    return SlotCameras(legend, frames, masks)


def event_legend(legends) -> list:
    """One bit order for many replays: CAMERA_BITS, then every other name, sorted."""
    extra = {n for lg in legends if lg for n in lg} - set(CAMERA_BITS)
    return list(CAMERA_BITS) + sorted(extra)


def remap(mask: Optional[int], src: tuple, dst: list) -> Optional[int]:
    """A mask built on legend ``src`` re-expressed on legend ``dst``."""
    if mask is None or tuple(src) == tuple(dst[:len(src)]):
        return mask
    out = 0
    for i, name in enumerate(src):
        if mask >> i & 1:
            out |= 1 << dst.index(name)
    return out


# ----------------------------------------------------------------- identity
def identity_segments(conn, job_id: str) -> list:
    """The window-labels identity segments of one replay (job), sorted by slot
    then start frame. [] when the table or the job's rows are missing."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(match_robot_identity_segment)")}
        if not cols:
            return []
        src = "source" if "source" in cols else "NULL"
        rows = conn.execute(
            f"SELECT slot, slot_label, start_frame, end_frame, team_number, confirmed, score, "
            f"{src} FROM match_robot_identity_segment WHERE job_id = ? "
            "ORDER BY slot, start_frame", (str(job_id),)).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for slot, label, f0, f1, team, conf, score, source in rows:
        try:
            team = int(str(team).strip()) if team not in (None, "") else None
        except ValueError:
            team = None
        out.append({"slot": label or str(slot), "start_frame": int(f0), "end_frame": int(f1),
                    "team": team, "confirmed": bool(conf),
                    "score": float(score) if score is not None else None, "source": source})
    return out


def collapse_segments(segs: list) -> list:
    """Merge neighbouring segments of one slot that share source and team (the
    window stage writes one row per window). Score = frame-weighted mean,
    confirmed only if every merged row was."""
    out: list = []
    for s in sorted(segs, key=lambda s: (s["slot"], s["start_frame"])):
        prev = out[-1] if out else None
        if (prev and prev["slot"] == s["slot"] and prev["source"] == s["source"]
                and prev["team"] == s["team"]):
            w0 = prev["_w"]
            w1 = max(1, s["end_frame"] - s["start_frame"] + 1)
            if prev["score"] is not None and s["score"] is not None:
                prev["score"] = (prev["score"] * w0 + s["score"] * w1) / (w0 + w1)
            else:
                prev["score"] = prev["score"] if s["score"] is None else s["score"]
            prev["_w"] = w0 + w1
            prev["end_frame"] = max(prev["end_frame"], s["end_frame"])
            prev["confirmed"] = prev["confirmed"] and s["confirmed"]
        else:
            out.append({**s, "_w": max(1, s["end_frame"] - s["start_frame"] + 1)})
    for s in out:
        s.pop("_w", None)
    return out


# ----------------------------------------------------------------- calibration
def match_cameras(summary: dict) -> dict:
    """{camera: {version_id, status, reason}} for the calibrations a replay used:
    main from ``calibration_version``, corners from ``corner_calibration``."""
    out = {}
    if summary.get("calibration_version") is not None:
        out["main"] = {"version_id": summary.get("calibration_version"),
                       "status": "main", "reason": None}
    for cam, c in sorted((summary.get("corner_calibration") or {}).items()):
        c = c if isinstance(c, dict) else {}
        out[cam] = {"version_id": c.get("version_id"), "status": c.get("status"),
                    "reason": c.get("reason")}
    return out


def _round(v, nd=4):
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


def camera_pose(rvec, tvec) -> dict:
    """The camera's pose in field terms from an OpenCV (rvec, tvec):
    p_cam = R·X_field + t, X_field in WPILib metres (z up), camera axes
    x right, y down, z forward (the optical axis).

    * ``position_m`` = -Rᵀt, the camera centre in field metres.
    * ``yaw_deg`` = heading of the optical axis f = Rᵀ·[0, 0, 1] in the field
      XY plane: atan2(f_y, f_x). 0 = looking toward +x (the red wall), 90 =
      toward +y; counter-clockwise seen from above is positive. (-180, 180].
    * ``pitch_deg`` = elevation of the optical axis: atan2(f_z, |f_xy|).
      Negative = looking down. [-90, 90].
    * ``roll_deg`` = rotation of the camera about its optical axis. 0 = the
      image's up direction (camera -y) is as close to world up (+z) as it can
      be, i.e. the horizon is level in the image. Positive = the camera body is
      turned clockwise as seen from behind it, looking along the optical axis
      (so the scene appears turned counter-clockwise in the image).
      Formally, with u0 = the unit component of +z orthogonal to f and
      u = Rᵀ·[0, -1, 0], roll = atan2((u0 × u)·f, u0·u). For a camera looking
      straight up or down (u0 undefined) +x stands in for +z.

    Given yaw ψ, pitch θ, roll φ the camera rebuilds as f = (cosθcosψ,
    cosθsinψ, sinθ), u0 as above, u = u0·cosφ + (f × u0)·sinφ, and R has rows
    [f × u, -u, f] (camera x, y, z axes in field coordinates).
    """
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    t = np.asarray(tvec, float).reshape(3)
    c = -R.T @ t
    f = R.T @ np.array([0.0, 0.0, 1.0])
    u = R.T @ np.array([0.0, -1.0, 0.0])
    yaw = math.degrees(math.atan2(f[1], f[0]))
    pitch = math.degrees(math.atan2(f[2], math.hypot(f[0], f[1])))
    ref = np.array([0.0, 0.0, 1.0]) if math.hypot(f[0], f[1]) > 1e-9 else np.array([1.0, 0, 0])
    u0 = ref - (ref @ f) * f
    u0 /= np.linalg.norm(u0)
    roll = math.degrees(math.atan2(float(np.cross(u0, u) @ f), float(u0 @ u)))
    return {"position_m": [_round(v, 4) for v in c],
            "yaw_deg": _round(yaw, 3), "pitch_deg": _round(pitch, 3), "roll_deg": _round(roll, 3),
            "rotation_matrix": [[_round(v, 8) for v in row] for row in R],
            "rvec": [_round(v, 8) for v in np.asarray(rvec, float).reshape(-1)],
            "tvec": [_round(v, 8) for v in t]}


def pose_from_angles(position, yaw_deg, pitch_deg, roll_deg) -> tuple:
    """(rvec, tvec) for a camera at ``position`` (field metres) with the
    yaw/pitch/roll of ``camera_pose`` (the inverse of it; used by tests and
    for building synthetic cameras)."""
    import cv2
    ps, th, ph = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    f = np.array([math.cos(th) * math.cos(ps), math.cos(th) * math.sin(ps), math.sin(th)])
    ref = np.array([0.0, 0.0, 1.0]) if math.hypot(f[0], f[1]) > 1e-9 else np.array([1.0, 0, 0])
    u0 = ref - (ref @ f) * f
    u0 /= np.linalg.norm(u0)
    u = u0 * math.cos(ph) + np.cross(f, u0) * math.sin(ph)
    R = np.stack([np.cross(f, u), -u, f])
    t = -R @ np.asarray(position, float).reshape(3)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3, 1), t.reshape(3, 1)


def _undistort_shift(K, D, mode: str, pts) -> Optional[float]:
    """max |distorted pixel − ideal pinhole pixel| over ``pts``, or None when
    any point does not undistort cleanly (non-finite, or the undistorted point
    does not distort back to within 0.5 px: past a wide lens's fold the
    iterative solve lands on a wrong root)."""
    import cv2
    K = np.asarray(K, float)
    D = np.asarray(D, float).reshape(-1)
    src = np.asarray(pts, float).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    try:
        if mode == "camera":
            und = cv2.fisheye.undistortPoints(src, K, D[:4].reshape(4, 1), R=np.eye(3), P=K,
                                              criteria=crit)
            norm = cv2.undistortPoints(und, K, None)       # pixel -> normalised, no lens
            back = cv2.fisheye.distortPoints(norm, K, D[:4].reshape(4, 1))
        else:
            und = cv2.undistortPointsIter(src, K, D.reshape(-1, 1), np.eye(3), K, crit)
            norm = cv2.undistortPoints(und, K, None).reshape(-1, 2)
            obj = np.hstack([norm, np.ones((len(norm), 1))]).reshape(-1, 1, 3)
            back, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, D.reshape(-1, 1))
    except cv2.error:
        return None
    und, back = und.reshape(-1, 2), np.asarray(back, float).reshape(-1, 2)
    src = src.reshape(-1, 2)
    if not (np.all(np.isfinite(und)) and np.all(np.isfinite(back))):
        return None
    if float(np.max(np.linalg.norm(back - src, axis=1))) > 0.5:
        return None
    return float(np.max(np.linalg.norm(und - src, axis=1)))


def distortion_shift(cal) -> dict:
    """How far the lens moves pixels: ``edge_shift_px`` at the image's mid-left
    and mid-right edges, ``corner_shift_px`` at its four corners (each the max
    over those points of |distorted − ideal pinhole pixel|; None when it can't
    be computed)."""
    if cal.K is None or cal.D is None or cal.mode == "homography":
        return {"edge_shift_px": None, "corner_shift_px": None}
    w, h = (float(v) for v in cal.image_size)
    if w <= 0 or h <= 0:
        return {"edge_shift_px": None, "corner_shift_px": None}
    edges = [(0.0, (h - 1) / 2), (w - 1, (h - 1) / 2)]
    corners = [(0.0, 0.0), (w - 1, 0.0), (0.0, h - 1), (w - 1, h - 1)]
    return {"edge_shift_px": _round(_undistort_shift(cal.K, cal.D, cal.mode, edges), 2),
            "corner_shift_px": _round(_undistort_shift(cal.K, cal.D, cal.mode, corners), 2)}


def calibration_detail(cal) -> dict:
    """CameraCalibration -> the export's per-version camera detail (no ids)."""
    K = None if cal.K is None else np.asarray(cal.K, float)
    D = None if cal.D is None else [_round(v, 8) for v in np.asarray(cal.D, float).reshape(-1)]
    model = {"camera": "fisheye_kb4", "pinhole": "opencv_pinhole"}.get(cal.mode)
    pose = None
    if cal.mode != "homography" and cal.rvec is not None and cal.tvec is not None:
        pose = camera_pose(cal.rvec, cal.tvec)
    return {
        "mode": cal.mode, "field_season": cal.field_season or None,
        "image_size": [int(v) for v in cal.image_size] if cal.image_size else None,
        "crop_region": [int(round(v)) for v in cal.crop_region] if cal.crop_region else None,
        "rms_error_px": _round(cal.rms_error, 3),
        "intrinsics": None if K is None else {
            "fx": _round(K[0, 0], 3), "fy": _round(K[1, 1], 3),
            "cx": _round(K[0, 2], 3), "cy": _round(K[1, 2], 3),
            "K": [[_round(v, 6) for v in row] for row in K]},
        "distortion": {"model": model if D is not None else None, "coefficients": D,
                       **distortion_shift(cal)},
        "pose": pose,
        "homography": (None if cal.mode != "homography" or cal.H is None
                       else [[_round(v, 10) for v in row] for row in np.asarray(cal.H, float)]),
    }


def _version_file(event_key: str, camera: str, version_id: str) -> Path:
    from web.field import store
    if version_id == "legacy":
        return store._path(event_key, camera)
    return store._version_path(event_key, camera, version_id)


def camera_detail(event_key: str, camera: str, version_id: str) -> dict:
    """One calibration version as the export's CAMERA_DETAIL. A version file
    that can't be read gives {version_id, error} instead of raising."""
    p = _version_file(event_key, camera, str(version_id))
    try:
        st = p.stat()
        return dict(_camera_detail_cached(str(p), st.st_mtime, str(version_id)))
    except Exception as e:                       # missing / unreadable / malformed
        return {"version_id": version_id, "error": f"{type(e).__name__}: {e}"[:300]}


@functools.lru_cache(maxsize=256)
def _camera_detail_cached(path: str, mtime: float, version_id: str) -> dict:
    from web.field.calibration import CameraCalibration
    d = json.loads(Path(path).read_text())
    if version_id == "legacy":                   # the legacy file is a bare calibration
        d = {"calibration": d, "version_id": "legacy", "label": "legacy"}
    cal = CameraCalibration.from_json(d["calibration"])
    return {"version_id": d.get("version_id", version_id), "created_at": d.get("created_at"),
            "label": d.get("label"), "start_match": d.get("start_match"),
            "rejected": bool(d.get("rejected")), **calibration_detail(cal)}

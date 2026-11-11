"""Render a replay's HUD video after finalize, with the FINAL labels
(docs/superpowers/specs/2026-11-10-hud-post-render-design.md sections 5-6).

    python -m web.hud_render <replay_id> [--db PATH]

Inputs: the replay row (source video, meta.json), ``meta.summary.hud_static``,
the per-frame draw list ``draw.jsonl.gz`` recorded live (hud_post_render knob),
or, for a replay without one, an approximate draw list built from
``meta.frames`` (back-fill), and the slot corrections in the DB. The source is
decoded exactly as the analysis loop does (one seek to the first processing
frame, then sequential reads), every recorded frame's main view is checked
against its recorded thumbnail (hud_draw.main_thumb), one frame is emitted
per ``meta.frames`` entry, and the result replaces ``hud.mp4`` atomically (a
live-encoded original is kept once as ``hud.live.mp4``). Stats go to ``render.json`` next to the video.

Import-light on purpose: never imports web.app (no torch, no models).
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
import zlib
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from web import hud_draw, hud_labels

OVERHEAD_CROP_RATIO = 0.67          # = web.app.OVERHEAD_CROP_RATIO (pinned by a test)
DEFAULT_FIELD_SEASON = "2026-rebuilt-welded"
# A recorded main_thumb (hud_draw) is aligned when no thumbnail pixel differs
# by more than this. The same decode gives 0. A 1-frame offset is not caught
# on every frame: on real 60 fps broadcast files with duplicated frames (~30 fps
# content) it stays <= 2 on ~50% of frames. Detection still works because that
# far exceeds the 2% abort budget (MISALIGNED_FRAC); on a synthetic near-static
# video every offset frame differs by 10+ (tests/test_hud_draw.py).
THUMB_MAX_DIFF = 2
MISALIGNED_FRAC = 0.02              # more mismatches than this share aborts
APPROX_NOTE = "approx boxes"
# Back-fill robot volume (spec section 6): 0.95 m x 0.95 m x 0.5 m.
ROBOT_HALF_M = 0.475
ROBOT_HEIGHT_M = 0.5
ROBOT_CENTRE_OFFSET_M = 0.475       # = window_crops.ROBOT_CENTRE_OFFSET_M
TRAIL_LEN = 16
LOCK_FILE = "render.lock"
RENDER_VIDEO = "hud.render.mp4"
RENDER_THUMB = "thumb.render.jpg"
LIVE_VIDEO = "hud.live.mp4"
STATS_FILE = "render.json"


class RenderError(Exception):
    """A render that cannot finish; the message is the render_error."""


# How long one render may run (the app's worker and this module's own
# deadline): 0.5 s x frame_count clamped to [600 s, 3600 s]; 1800 s when the
# count is missing, non-numeric, non-finite or <= 0.
RENDER_TIMEOUT_PER_FRAME_S = 0.5    # ~10x a measured render's per-frame cost
RENDER_TIMEOUT_MIN_S = 600.0
RENDER_TIMEOUT_MAX_S = 3600.0
RENDER_TIMEOUT_DEFAULT_S = 1800.0


def render_timeout_s(row) -> float:
    """The render deadline for a replay row (see the constants above)."""
    try:
        n = float((row or {}).get("frame_count"))
    except (TypeError, ValueError):
        return RENDER_TIMEOUT_DEFAULT_S
    if not math.isfinite(n) or n <= 0:
        return RENDER_TIMEOUT_DEFAULT_S
    return min(RENDER_TIMEOUT_MAX_S, max(RENDER_TIMEOUT_MIN_S, RENDER_TIMEOUT_PER_FRAME_S * n))


class _Deadline:
    """SIGALRM (ITIMER_REAL) that raises RenderError("timeout after N s") in
    the main thread. A no-op without a deadline or off the main thread. The
    render's own cleanup (temp file) and run()'s lock release still run."""

    def __init__(self, seconds: Optional[float]) -> None:
        self.seconds = seconds
        self._armed = False
        self._old = None

    def _fire(self, signum, frame):
        raise RenderError(f"timeout after {self.seconds:g} s")

    def __enter__(self):
        if self.seconds is None or threading.current_thread() is not threading.main_thread():
            return self
        self._old = signal.signal(signal.SIGALRM, self._fire)
        signal.setitimer(signal.ITIMER_REAL, max(float(self.seconds), 1e-3))
        self._armed = True
        return self

    def __exit__(self, *exc):
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
            self._armed = False
        return False


def load_draw_list(path) -> dict:
    """{replay frame index: line}. A truncated file (job killed mid-write)
    keeps the lines read before the damage."""
    out = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except ValueError:
                    break
                out[int(line["i"])] = line
    except (EOFError, OSError, zlib.error):
        pass
    return out


def _main_calibration(event_key, camera, version_id, match_key, warnings: list):
    """The main CameraCalibration of ``version_id``; when that version is gone
    (or unknown), the epoch covering ``match_key``, with a warning."""
    if not event_key:
        return None
    from web.field import store as field_store
    from web.field.match_order import select_epoch
    versions = field_store.load_versions(event_key, camera or "main")
    chosen = next((v for v in versions if v["version_id"] == version_id), None) \
        if version_id else None
    if chosen is None:
        chosen = select_epoch(versions, match_key)
        if chosen is not None:
            warnings.append(f"calibration version {version_id!r} not found: using epoch "
                            f"{chosen['version_id']} for {match_key}")
    return chosen["calibration"] if chosen else None


def _corner_views(event_key, match_key) -> list:
    """Corner views as the live job built them (calibrated cameras only),
    for a replay without hud_static: [{name, crop, masks}]."""
    if not event_key:
        return []
    from web.field import store as field_store
    from web.field.camera_view import load_views
    from web.field.composite_layout import load_layout
    from web.field.match_order import select_epoch
    selection = {}
    for name in load_layout(event_key=event_key):
        pick = select_epoch(field_store.load_versions(event_key, name), match_key)
        if pick is not None:
            selection[name] = pick["version_id"]
    return [{"name": v.name, "crop": [int(c) for c in v.crop],
             "masks": [[int(c) for c in m] for m in v.masks]}
            for v in load_views(event_key, selection=selection) if v.name != "main"]


def _ground_center_offset(summary: dict) -> float:
    """The job's ground_center_offset_m (summary.config holds non-defaults
    only), else the knob default."""
    v = (summary.get("config") or {}).get("ground_center_offset_m")
    if v is None:
        from web.job_config import KNOBS
        v = next(k.default for k in KNOBS if k.name == "ground_center_offset_m")
    return float(v)


def backfill_box(cal, field_pos, gco: float) -> Optional[list]:
    """[x1, y1, x2, y2]: the image bounding box of a 0.95 x 0.95 x 0.5 m robot
    volume centred on push_from_camera(cal, field_pos, 0.475 - gco), as
    window_crops.target centres its crops. None when any corner does not
    project to a finite point."""
    from web.field.calibration import push_from_camera
    try:
        offset = max(0.0, ROBOT_CENTRE_OFFSET_M - float(gco or 0.0))
        cx, cy = push_from_camera(cal, (float(field_pos[0]), float(field_pos[1])), offset)
        pts = [cal.field_to_image((cx + dx, cy + dy, z))
               for dx in (-ROBOT_HALF_M, ROBOT_HALF_M)
               for dy in (-ROBOT_HALF_M, ROBOT_HALF_M)
               for z in (0.0, ROBOT_HEIGHT_M)]
    except Exception:
        return None
    a = np.asarray(pts, float)
    if a.shape != (8, 2) or not np.all(np.isfinite(a)):
        return None
    return [float(a[:, 0].min()), float(a[:, 1].min()),
            float(a[:, 0].max()), float(a[:, 1].max())]


def build_backfill_lines(frames: list, cal, gco: float, view_wh) -> dict:
    """Approximate draw list from ``meta.frames[].robots[]`` (spec section 6):
    projected main boxes (no mask, no score), radar dots with 16-position
    trails, no corner boxes, hubs, hub boxes or decode hashes."""
    w, h = view_wh
    trails: dict = {}
    out = {}
    for idx, f in enumerate(frames):
        i = int(f.get("i", idx))
        main, radar, seen = [], [], set()
        for r in f.get("robots") or []:
            slot, pos = r.get("slot"), r.get("field_pos")
            if slot is None or pos is None:
                continue
            seen.add(slot)
            trail = trails.setdefault(slot, deque(maxlen=TRAIL_LEN))
            trail.append((float(pos[0]), float(pos[1])))
            radar.append({"slot": slot, "alliance": r.get("alliance"),
                          "pos": [float(pos[0]), float(pos[1])], "state": "visible",
                          "corner_only": False, "trail": [list(p) for p in trail]})
            box = backfill_box(cal, pos, gco)
            if box is not None and box[2] > 0 and box[3] > 0 and box[0] < w and box[1] < h:
                main.append({"slot": slot, "slot_id": hud_draw.slot_id_from_label(slot),
                             "alliance": r.get("alliance"), "box": box, "ghost": False,
                             "score": None, "mask_poly": None})
        for s in list(trails):
            if s not in seen:
                del trails[s]
        out[i] = {"i": i, "frame_num": f.get("frame_num"), "main": main, "corners": {},
                  "hubs": [], "hub_boxes": [], "radar": radar}
    return out


def draw_frame(img: np.ndarray, line: Optional[dict], rows_by_slot: dict,
               corr_by_label: dict, ctx: SimpleNamespace) -> np.ndarray:
    """One composed HUD frame from a decoded source frame, its draw-list line
    (None = static layers only) and the final labels."""
    line = line or {}
    flags = ctx.flags
    annotated = img[:ctx.overhead_height, :].copy()
    if ctx.wire_cal is not None:
        try:
            hud_draw.draw_field_wireframe(annotated, ctx.wire_cal, ctx.field_model)
        except Exception:
            pass
    for b in line.get("hub_boxes") or []:
        hud_draw.draw_hub_box(annotated, b[:4])
    if flags.ball_boxes:
        for b in line.get("ball_boxes") or []:
            hud_draw.draw_ball_box(annotated, b[:4])
    for e in line.get("main") or []:
        kind, text = hud_labels.entry_text(e, rows_by_slot, corr_by_label)
        if kind == "phantom":
            ctx.phantoms += 1
            hud_draw.draw_phantom(annotated, e["box"], text)
        elif kind == "ghost":
            if flags.coasting_ghost:
                hud_draw.draw_ghost(annotated, e["box"], e.get("alliance"), text)
        else:
            sid = e.get("slot_id")
            sid = hud_draw.slot_id_from_label(e.get("slot")) if sid is None else int(sid)
            hud_draw.draw_robot(annotated, e["box"], e.get("alliance"), text, flags,
                                slot_id=sid,
                                mask=hud_draw.poly_mask(e.get("mask_poly"), annotated.shape))
    for hb in line.get("hubs") or []:
        hud_draw.draw_hub(annotated, hb["x"], hb["y"], hb.get("alliance"))
    if ctx.ocr_regions:
        hud_draw.draw_ocr_regions(annotated, ctx.ocr_regions)

    from web.field.radar import RadarTrack
    tracks = []
    for r in line.get("radar") or []:
        has_row, label = hud_labels.radar_text(r, rows_by_slot, corr_by_label)
        tracks.append(RadarTrack(
            pos=tuple(r["pos"]), alliance=r.get("alliance") if has_row else None,
            team=label, slot_id=hud_draw.slot_id_from_label(r.get("slot")),
            state=r.get("state") or "visible", corner_only=bool(r.get("corner_only")),
            trail=tuple(tuple(p) for p in r.get("trail") or ())))
    radar_img = ctx.radar.render(tracks)

    panels = []
    corners = line.get("corners") or {}
    for v in ctx.views:
        entries = []
        for e in corners.get(v["name"]) or []:
            kind, text = hud_labels.entry_text(e, rows_by_slot, corr_by_label)
            entries.append({"box": e["box"], "alliance": e.get("alliance"), "label": text,
                            "phantom": kind == "phantom",
                            "slot_id": (hud_draw.slot_id_from_label(e.get("slot"))
                                        if kind == "robot" else None)})
            ctx.phantoms += kind == "phantom"
        panel = hud_draw.extract_view(img, v["crop"], v.get("masks") or ())
        panels.append(hud_draw.camera_panel(v["name"], panel, entries, flags,
                                            max_width=ctx.panel_max_width))
    out = hud_draw.compose(annotated, radar_img, panels)
    if ctx.approx:
        hud_draw.draw_note(out, APPROX_NOTE)
    return out


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _video_frame_count(path: Path) -> int:
    """CAP_PROP_FRAME_COUNT of an encoded file (0 when it does not open)."""
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    finally:
        cap.release()


def render_replay(replay_id: str, db_path=None) -> dict:
    """Render ``replay_id`` and replace its hud.mp4; returns the render stats
    (also written to render.json). Raises RenderError with the reason."""
    from web import database
    from web.field.field_model import FieldModel
    from web.field.radar import Radar
    from web.replay_recorder import ReplayRecorder

    t0 = time.perf_counter()
    row = database.get_replay_sync(replay_id, db_path)
    if row is None:
        raise RenderError("replay not found")
    meta_path = Path(row.get("meta_path") or "")
    if not meta_path.is_file():
        raise RenderError(f"meta.json missing: {meta_path}")
    out_dir = meta_path.parent
    meta = json.loads(meta_path.read_text())
    summary, frames = meta.get("summary") or {}, meta.get("frames") or []
    if not frames:
        raise RenderError("replay has no frames")
    src = row.get("source_video_path")
    if not src or not Path(src).is_file():
        raise RenderError(f"source video missing: {src}")
    corr_rows = database.get_slot_corrections_sync(replay_id, db_path)
    corrections = hud_labels.effective_corrections(corr_rows)
    corr_by_label = {hud_draw.slot_label(s): c for s, c in corrections.items()}

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RenderError(f"cannot open source video: {src}")
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warnings: list = []
    static = summary.get("hud_static")
    draw_path = out_dir / "draw.jsonl.gz"
    if static and draw_path.is_file():
        approx = False
        lines = load_draw_list(draw_path)
        flags = hud_draw.flags_from_dict(static.get("flags"))
        overhead_height = int(static.get("overhead_height") or int(vh * OVERHEAD_CROP_RATIO))
        views = static.get("views") or []
        ocr_regions = static.get("ocr_regions") or {}
        refs = static.get("reference_paths") or []
        season = static.get("field_season")
        wire_cal = None
        if static.get("field_wireframe_overlay"):
            wire_cal = _main_calibration(row.get("event_key") or static.get("event_key"),
                                         static.get("camera"), static.get("calibration_version"),
                                         row.get("match_key"), warnings)
            if wire_cal is None:
                warnings.append("field wireframe skipped: main calibration not found")
    else:
        approx = True
        flags = hud_draw.draw_flags()
        overhead_height = int(vh * OVERHEAD_CROP_RATIO)
        cal = _main_calibration(row.get("event_key"), "main", summary.get("calibration_version"),
                                row.get("match_key"), warnings)
        if cal is None:
            cap.release()
            raise RenderError("no draw list, and no main calibration to project boxes with")
        lines = build_backfill_lines(frames, cal, _ground_center_offset(summary),
                                     (vw, overhead_height))
        views = _corner_views(row.get("event_key"), row.get("match_key"))
        ocr_regions, refs, season, wire_cal = {}, [], getattr(cal, "field_season", None), None
    field_model = FieldModel.load(season or DEFAULT_FIELD_SEASON)
    radar = Radar(field_model.length_m, field_model.width_m, field_model=field_model)
    if refs:
        radar.set_reference_paths([SimpleNamespace(**r) for r in refs])
    ctx = SimpleNamespace(flags=flags, overhead_height=overhead_height, views=views,
                          ocr_regions=ocr_regions, wire_cal=wire_cal, field_model=field_model,
                          radar=radar, approx=approx, phantoms=0,
                          panel_max_width=int((static or {}).get("panel_max_width")
                                              or hud_draw.PANEL_MAX_WIDTH))

    ranges = summary.get("processing_ranges") or []
    if ranges:
        first = int(ranges[0][0])
    else:
        first = 0
        warnings.append("no processing_ranges: decoding from frame 0")
    hashed = sum(1 for ln in lines.values() if ln.get("main_thumb"))
    allowed = int(MISALIGNED_FRAC * hashed)
    checked = mismatches = missing = 0
    fps = float(summary.get("fps") or row.get("fps") or 8.0)
    rec = ReplayRecorder(video_name=RENDER_VIDEO)
    tmp_video = out_dir / RENDER_VIDEO
    thumb = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        fn = first
        for idx, mf in enumerate(frames):
            target = mf.get("frame_num")
            if target is None or int(target) < fn:
                raise RenderError(f"decode misaligned: meta frame {idx} has frame_num {target} "
                                  f"but the decoder is at {fn}")
            target = int(target)
            while fn < target:
                if not cap.grab():
                    raise RenderError(f"source video ended at frame {fn}, before {target}")
                fn += 1
            ok, img = cap.read()
            if not ok:
                raise RenderError(f"source video ended at frame {fn}")
            fn += 1
            line = lines.get(int(mf.get("i", idx)))
            if line is not None and line.get("frame_num") not in (None, target):
                line = None
            if line is None:
                missing += 1
            elif line.get("main_thumb"):
                checked += 1
                if hud_draw.thumb_diff(hud_draw.main_thumb(img[:overhead_height, :]),
                                       line["main_thumb"]) > THUMB_MAX_DIFF:
                    mismatches += 1
                    if mismatches > allowed:
                        raise RenderError(f"decode misaligned: {mismatches} of {hashed} "
                                          f"frames fail the main-view thumbnail check")
            rows_by_slot = {r.get("slot"): r for r in mf.get("robots") or []}
            composed = draw_frame(img, line, rows_by_slot, corr_by_label, ctx)
            if not rec.is_open:
                rec.open_lazy(composed, fps, out_dir)
            rec.add_frame(composed, {})
            if idx == len(frames) // 2:
                thumb = composed.copy()
        rc = rec.close_video()
        if rc not in (None, 0):
            raise RenderError(f"encoder exited {rc}")
        if not tmp_video.is_file() or tmp_video.stat().st_size == 0:
            raise RenderError("encoder produced no video")
        encoded = _video_frame_count(tmp_video)
        if encoded != len(frames):
            raise RenderError(f"encoded video has {encoded} frames, expected {len(frames)}")
    except BaseException:
        cap.release()
        rec.close_video()
        tmp_video.unlink(missing_ok=True)
        raise
    cap.release()

    final = out_dir / "hud.mp4"
    live = out_dir / LIVE_VIDEO
    stats_path = out_dir / STATS_FILE
    kept_live = False
    if final.exists() and not live.exists() and not stats_path.exists():
        try:
            os.link(final, live)            # keep the live-encoded original, once
        except OSError:
            shutil.copy2(final, live)
        kept_live = True
    os.replace(tmp_video, final)
    if thumb is not None:
        if thumb.shape[:2] != (rec.height, rec.width):
            thumb = cv2.resize(thumb, (rec.width, rec.height))
        tmp_thumb = out_dir / RENDER_THUMB
        cv2.imwrite(str(tmp_thumb), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        os.replace(tmp_thumb, out_dir / "thumb.jpg")

    relabelled = hud_labels.relabel_counts(frames, corrections, hud_draw.slot_label)
    stats = {
        "replay_id": replay_id,
        "rendered_at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - t0, 2),
        "frames": len(frames),
        "source": "approx" if approx else "draw_list",
        "decode_checked": checked, "decode_mismatches": mismatches,
        "draw_lines_missing": missing if not approx else 0,
        "phantom_boxes": ctx.phantoms,
        "corrections": {lab: {"old": corr_by_label[lab][0], "new": corr_by_label[lab][1],
                              "frames_relabelled": n} for lab, n in relabelled.items()},
        "corrections_note": None if corr_rows else "no recorded corrections",
        "kept_live_video": kept_live,
        "width": rec.width, "height": rec.height, "codec": rec.codec or None,
        "video_path": str(final), "thumb_path": str(out_dir / "thumb.jpg"),
        "warnings": warnings,
    }
    _write_json_atomic(stats_path, stats)
    return stats


class RenderLock:
    """An open, flock(LOCK_EX)-held ``render.lock``. The kernel drops the
    flock when the holder dies, so a lock file left behind is no obstacle;
    the pid written into it is only for messages."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path, self.fd = path, fd


def _read_pid(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return -1


def lock_holder(out_dir) -> Optional[int]:
    """The pid holding ``<out_dir>/render.lock`` (-1 when it holds the lock
    but has not written its pid yet), or None when no process holds it.
    Probes with a non-blocking shared flock, never creating the file; a probe
    can make an acquirer racing it in that same instant see "locked"."""
    p = Path(out_dir) / LOCK_FILE
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read_pid(p)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def acquire_lock(out_dir) -> Optional[RenderLock]:
    """Take ``<out_dir>/render.lock`` (flock LOCK_EX|LOCK_NB, held until
    release_lock or this process dies) and write this pid into it. None when
    another holder has it."""
    p = Path(out_dir) / LOCK_FILE
    for _ in range(5):
        fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        try:
            same = os.fstat(fd).st_ino == os.stat(p).st_ino
        except FileNotFoundError:
            same = False
        if not same:                # a releaser unlinked it under us: lock the new file
            os.close(fd)
            continue
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        return RenderLock(p, fd)
    return None


def release_lock(lock: Optional[RenderLock]) -> None:
    """Remove the lock file (while still holding it) and drop the flock."""
    if lock is None:
        return
    try:
        lock.path.unlink(missing_ok=True)
    finally:
        os.close(lock.fd)


def _remove_leftovers(out_dir: Path) -> None:
    """A killed render's temp files (only ever touched under the lock)."""
    for name in (RENDER_VIDEO, RENDER_THUMB, STATS_FILE + ".tmp"):
        (out_dir / name).unlink(missing_ok=True)


def run(replay_id: str, db_path=None, deadline_s: Optional[float] = None) -> tuple:
    """Lock, render and record the outcome in the DB (render_status
    rendering -> rendered | render_failed + render_error, and the new video
    columns). Returns (exit code, result): 0 rendered (result = stats), 1
    failed, 3 locked by another live process (status left alone).
    ``deadline_s``: the render fails as "timeout after N s" past it (main
    thread only), so a renderer that outlives the server still ends."""
    from web import database
    row = database.get_replay_sync(replay_id, db_path)
    if row is None:
        return 1, {"replay_id": replay_id, "error": "replay not found"}

    def failed(msg):
        database.set_render_status_sync(replay_id, "render_failed", msg, db_path=db_path)
        return 1, {"replay_id": replay_id, "error": msg}

    if not row.get("meta_path"):
        return failed("meta.json missing")
    out_dir = Path(row["meta_path"]).parent
    if not out_dir.is_dir():
        return failed("replay directory missing")
    lock = acquire_lock(out_dir)
    if lock is None:
        return 3, {"replay_id": replay_id, "error": f"locked by pid {lock_holder(out_dir)}"}
    try:
        _remove_leftovers(out_dir)
        database.set_render_status_sync(replay_id, "rendering", None, db_path=db_path)
        try:
            with _Deadline(deadline_s):
                stats = render_replay(replay_id, db_path)
        except Exception as e:
            return failed(str(e) if isinstance(e, RenderError) else f"{type(e).__name__}: {e}")
        database.set_render_status_sync(
            replay_id, "rendered", None, db_path=db_path, codec=stats["codec"],
            width=stats["width"], height=stats["height"], video_path=stats["video_path"],
            thumb_path=stats["thumb_path"])
        return 0, stats
    finally:
        release_lock(lock)


def read_stats(out_dir) -> Optional[dict]:
    """The last render's stats (render.json), or None."""
    try:
        return json.loads((Path(out_dir) / STATS_FILE).read_text())
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render a replay's HUD video with the final labels.")
    ap.add_argument("replay_id")
    ap.add_argument("--db", default=None, help="SQLite DB (default: web.database.DB_PATH)")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="fail the render past this many seconds (default: render_timeout_s "
                         "of the replay, i.e. 0.5 s/frame clamped to 600-3600 s)")
    args = ap.parse_args(argv)
    try:
        os.nice(10)                 # yield CPU to the next job's analysis loop
    except OSError:
        pass
    deadline = args.deadline_s
    if deadline is None:
        from web import database
        deadline = render_timeout_s(database.get_replay_sync(args.replay_id, args.db))
    code, result = run(args.replay_id, args.db, deadline_s=deadline)
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    sys.exit(main())

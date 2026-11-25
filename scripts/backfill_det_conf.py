"""Backfill per-point ``det_conf`` into replays made before 2906f791.

Commit 2906f791 added ``det_conf`` (the YOLO box confidence of the detection
behind a tracked point) to replay meta.json ``frames[].robots[]`` and to
``summary.autonomous_paths`` samples. Older replays lack it. This script
recovers it by re-running ONLY the robot detector -- no SAM, readers or
identity -- and joining its boxes onto the boxes the live job drew:

* Replay: the one the scouting export uses per match
  (``web.scouting.loader.candidate_replays`` + ``pick_replay``).
* Detector: a lean ``InferenceJob`` (built with ``__new__``, only the YOLO
  models + config + field model set) running the live
  ``make_frame_detect_fn`` itself, so the cascade, field filter and per-half
  cap are the live code. Models load once per process.
* Views: ``load_views`` with the replay's own calibration versions
  (``summary.calibration_version``, ``summary.corner_calibration[cam]
  .version_id``); dropped corners (version None) are skipped.
* Frames: the live export prep -- one sequential decode from the first
  processed frame, ``view.extract``, calibrated top crop, INTER_AREA resize
  by ``sam3_export_scale``, JPEG q95 written to a temp file and read back.
  Only (frame, camera) pairs whose draw-log line has a slot box are run.
* Matching: detections back to view full-res coords (``/scale``, ``+y_off``,
  the detect fn's own formula); each draw-log slot box
  (``draw.jsonl.gz``: non-ghost ``main`` entries, ``corners`` entries with a
  slot) takes the detection with best IoU >= ``--iou``. Per (frame, slot)
  the cameras are combined like fusion does: live ``_merge`` carries the
  det_conf of the MAX-WEIGHT member (main 1.0, a corner CORNER_VOTE_WEIGHT x
  its range falloff). ``--cam-pick weight`` (default) reproduces that from
  each camera's box; ``--cam-pick max`` takes the max det_conf over cameras
  (simpler, biased high). Both are approximations: the draw log's corner
  slot is the nearest track within 0.85 m, not the fused cluster.
* Only points the draw log's radar marks ``visible`` get a value (live sets
  det_conf only for visible tracks); coasting points stay None.

Writes (unless --dry-run / --compare): atomic replace of meta.json, a
one-time backup ``meta.json.pre-detconf`` and ``summary.det_conf_backfill``
provenance. ``tracker_prob`` / ``state`` stay absent on old replays (the
loader reads a missing key as None).

Replays made ON 2906f791 itself (before the evidence-loop fix in web/app.py)
carry live det_conf only in autonomous_paths, and it is wrong: every
detection of a view-frame got the LAST object's evidence (leaked loop
variable ``info``). They count as "has det_conf", so re-run them with
``--force --match ...``.

Usage:
  .venv/bin/python scripts/backfill_det_conf.py --event 2026cur --dry-run --match 2026cur_qm8
  .venv/bin/python scripts/backfill_det_conf.py --event 2026cascmp --compare --match 2026cascmp_qm85
  .venv/bin/python scripts/backfill_det_conf.py --event 2026cur            # write all
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPT_VERSION = 1
BACKUP_SUFFIX = ".pre-detconf"
MAIN = "main"
# Unknown (branch-only) knobs that changed the CORNER detector: a backfill
# with the main-line detector would not reproduce those corner boxes.
CORNER_DETECTOR_KNOBS = ("corner_robot_model",)


# ------------------------------------------------------------------ pure parts
def export_to_full(box, scale: float, y_off: int) -> tuple:
    """Export-scale (top-cropped) box -> view full-res coords; the exact
    formula make_frame_detect_fn / evidence_build use."""
    return (box[0] / scale, box[1] / scale + y_off,
            box[2] / scale, box[3] / scale + y_off)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_box(box, dets, iou_min: float) -> tuple:
    """(det_conf | None, best_iou, matched) of the detection that best
    overlaps ``box``. ``dets`` = [(full_box, det_conf | None)]. A match whose
    detection has no confidence (motion / union box) returns (None, iou, True)."""
    best_i, best = -1, 0.0
    for i, (db, _c) in enumerate(dets):
        v = box_iou(box, db)
        if v > best:
            best_i, best = i, v
    if best_i < 0 or best < iou_min:
        return None, best, False
    return dets[best_i][1], best, True


def combine_cams(cands, mode: str = "weight") -> tuple:
    """Fuse one (frame, slot)'s per-camera det_confs. ``cands`` =
    [(cam, conf | None, weight)]. ``weight``: fusion's rule -- the det_conf
    of the max-weight member with a det_conf (``max((w, conf))``, ties to the
    larger conf); ``max``: the max det_conf. Returns (conf | None, cam)."""
    have = [(w, c, cam) for cam, c, w in cands if c is not None and w > 0]
    if not have:
        return None, None
    if mode == "max":
        c, cam = max((c, cam) for _w, c, cam in have)
        return c, cam
    w, c, cam = max(have)
    return c, cam


_DECODER = json.JSONDecoder()


def parse_draw_line(line: str, keys=("frame_num", "main", "corners", "radar")) -> dict:
    """Decode only ``keys`` of one draw-log line (the base64 thumbnail is most
    of it); falls back to json.loads (web.scouting.provenance._draw_line)."""
    try:
        out = {}
        for key in keys:
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


def draw_targets(lines) -> dict:
    """{frame: {"boxes": {cam: [(slot, box)]}, "state": {slot: state} | None}}
    from draw-log lines: non-ghost main entries + corner entries with a slot.
    Frames with no slot box are omitted."""
    out = {}
    for line in lines:
        try:
            d = parse_draw_line(line) if isinstance(line, str) else line
            fn = int(d["frame_num"])
        except (ValueError, KeyError, TypeError):
            continue
        boxes = defaultdict(list)
        for e in d.get("main") or []:
            if e.get("slot") and not e.get("ghost") and e.get("box"):
                boxes[MAIN].append((e["slot"], tuple(float(v) for v in e["box"][:4])))
        for cam, lst in (d.get("corners") or {}).items():
            for e in lst or []:
                if e.get("slot") and e.get("box"):
                    boxes[cam].append((e["slot"], tuple(float(v) for v in e["box"][:4])))
        if not boxes:
            continue
        radar = d.get("radar")
        state = ({r["slot"]: r.get("state") for r in radar if r.get("slot")}
                 if isinstance(radar, list) else None)
        out[fn] = {"boxes": dict(boxes), "state": state}
    return out


def read_draw_targets(path) -> dict:
    with gzip.open(path, "rt") as f:
        return draw_targets(f)


def slot_values(per_cam: dict, targets: dict, mode: str) -> dict:
    """{(frame, slot): (conf | None, cam | None)} from
    ``per_cam[(frame, slot)] = [(cam, conf, weight)]``, keeping only slots the
    draw log's radar marks visible (when it has a radar)."""
    out = {}
    for (fn, slot), cands in per_cam.items():
        state = (targets.get(fn) or {}).get("state")
        if state is not None and state.get(slot) != "visible":
            out[(fn, slot)] = (None, None)
            continue
        out[(fn, slot)] = combine_cams(cands, mode)
    return out


def _r4(v):
    return None if v is None else round(float(v), 4)


def has_det_conf(meta: dict) -> bool:
    """Any non-null det_conf in frames[].robots[] or autonomous_paths samples
    (replays made between 2906f791 and the finalize carry-over fix have it
    ONLY in autonomous_paths: the finalize re-resolve rebuilt the robot rows)."""
    for fr in meta.get("frames") or []:
        for rb in fr.get("robots") or []:
            if rb.get("det_conf") is not None:
                return True
    for samples in ((meta.get("summary") or {}).get("autonomous_paths") or {}).values():
        for s in samples or []:
            if s.get("det_conf") is not None:
                return True
    return False


def skip_reason(meta: dict, force: bool) -> Optional[str]:
    """Why a replay is skipped (None: process it)."""
    if not meta or not meta.get("frames"):
        return "no frames"
    if has_det_conf(meta) and not force:
        bf = (meta.get("summary") or {}).get("det_conf_backfill")
        return ("already backfilled (use --force)" if bf
                else "has live det_conf (new-code replay; use --force)")
    return None


def patch_meta(meta: dict, values: dict) -> tuple:
    """Write ``det_conf`` into meta in place: every ``frames[].robots[]``
    record (value for its (frame, slot), else None) and every
    ``summary.autonomous_paths`` sample (by frame_num; interpolated samples
    stay None). ``values`` = {(frame, slot): conf | None}. Returns
    (n_robot_records, n_filled, n_auto_samples, n_auto_filled)."""
    n_rec = n_fill = 0
    for fr in meta.get("frames") or []:
        fn = fr.get("frame_num")
        for rb in fr.get("robots") or []:
            n_rec += 1
            v = _r4(values.get((fn, rb.get("slot"))))
            rb["det_conf"] = v
            n_fill += v is not None
    n_ap = n_ap_fill = 0
    paths = (meta.get("summary") or {}).get("autonomous_paths") or {}
    for slot, samples in paths.items():
        for s in samples or []:
            n_ap += 1
            v = None if s.get("interpolated") else _r4(
                values.get((s.get("frame_num"), slot)))
            s["det_conf"] = v
            n_ap_fill += v is not None
    return n_rec, n_fill, n_ap, n_ap_fill


def write_meta_atomic(path: Path, meta: dict) -> None:
    """Backup once (meta.json.pre-detconf), then temp file + os.replace."""
    path = Path(path)
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        tmpb = backup.with_name(backup.name + ".tmp")
        tmpb.write_bytes(path.read_bytes())
        os.replace(tmpb, backup)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, path)


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def live_points(meta: dict, source: str = "frames"):
    """(frame, slot, live det_conf, state | None) for every live point:
    ``frames`` = frames[].robots[]; ``auto_paths`` = summary.autonomous_paths
    samples (visible-only by construction; interpolated samples skipped)."""
    if source == "auto_paths":
        paths = (meta.get("summary") or {}).get("autonomous_paths") or {}
        for slot, samples in paths.items():
            for s in samples or []:
                if not s.get("interpolated"):
                    yield s.get("frame_num"), slot, s.get("det_conf"), "visible"
        return
    for fr in meta.get("frames") or []:
        for rb in fr.get("robots") or []:
            yield fr.get("frame_num"), rb.get("slot"), rb.get("det_conf"), rb.get("state")


def compare_stats(meta: dict, values: dict, cams_of: dict, source: str = "frames",
                  frame_confs: Optional[dict] = None) -> dict:
    """Agreement of backfill ``values`` {(frame, slot): (conf, cam)} with the
    live ``det_conf`` already in ``meta`` (``source``: see live_points).
    Breakdown by the set of cameras with a draw box for the point
    (``cams_of``) and by the camera the backfill picked. With
    ``frame_confs`` ({frame: {4-dp confs of every backfill detection that
    frame}}), ``live_in_frame_dets`` counts live values reproduced by SOME
    detection of the frame -- detector/frame-alignment fidelity independent
    of which box the live value was attached to."""
    def blank():
        return {"n": 0, "live": 0, "bf_on_live": 0, "bf_where_live_null": 0,
                "live_in_frame_dets": 0, "diffs": []}
    tot = blank()
    by_set = defaultdict(blank)
    by_pick = defaultdict(blank)
    visible_live_null = 0
    for fn, slot, live, state in live_points(meta, source):
        bf, cam = values.get((fn, slot), (None, None))
        bf = _r4(bf)
        keys = [tot, by_set["+".join(sorted(cams_of.get((fn, slot), ()))) or "-"],
                by_pick[cam or "-"]]
        for b in keys:
            b["n"] += 1
            if live is not None:
                b["live"] += 1
                if frame_confs is not None and round(float(live), 4) in frame_confs.get(fn, ()):
                    b["live_in_frame_dets"] += 1
                if bf is not None:
                    b["bf_on_live"] += 1
                    b["diffs"].append(abs(bf - live))
            elif bf is not None:
                b["bf_where_live_null"] += 1
        if live is None and state == "visible":
            visible_live_null += 1

    def summ(b):
        d = b.pop("diffs")
        b["fill_on_live"] = round(b["bf_on_live"] / b["live"], 4) if b["live"] else None
        b["diff_median"] = _pct(d, 0.5)
        b["diff_p90"] = _pct(d, 0.9)
        b["diff_max"] = max(d) if d else None
        b["within_0.01"] = round(sum(x <= 0.01 + 1e-9 for x in d) / len(d), 4) if d else None
        return b
    return {"total": summ(tot),
            "by_cam_set": {k: summ(v) for k, v in sorted(by_set.items())},
            "by_picked_cam": {k: summ(v) for k, v in sorted(by_pick.items())},
            "live_visible_but_null": visible_live_null}


def view_selection(summary: dict, layout_names, epoch_pick) -> tuple:
    """({cam: version_id} for load_views, [cams to skip]) from a replay
    summary. Main: ``calibration_version``; corners: ``corner_calibration``
    version_id (None / missing entry with a status -> skipped). Cameras the
    summary does not mention fall back to ``epoch_pick(cam)`` (the live
    select_epoch), as the job itself did."""
    sel, skip = {}, []
    cc = summary.get("corner_calibration") or {}
    for name in layout_names:
        if name == MAIN:
            vid = summary.get("calibration_version") or epoch_pick(name)
        elif name in cc:
            vid = (cc[name] or {}).get("version_id")
            if vid is None:
                skip.append(name)
                continue
        else:
            vid = epoch_pick(name)
        if vid is not None:
            sel[name] = vid
    return sel, skip


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("inf")


def wait_for_memory(min_gb: float, log=print, poll_s: float = 5.0) -> float:
    """Block while MemAvailable < min_gb; returns seconds waited."""
    waited = 0.0
    warned = False
    while min_gb > 0 and mem_available_gb() < min_gb:
        if not warned:
            log(f"  [mem] MemAvailable {mem_available_gb():.2f} GB < {min_gb} GB; waiting")
            warned = True
        time.sleep(poll_s)
        waited += poll_s
    return waited


# ------------------------------------------------------------- detector rig
class _DeviceYOLO:
    """Wraps a YOLO model so every call runs on ``device`` (live calls pass
    no device: ultralytics' default, cuda:0 when available)."""

    def __init__(self, model, device):
        self._m, self._device = model, device

    def __call__(self, *a, **kw):
        kw.setdefault("device", self._device)
        return self._m(*a, **kw)

    def __getattr__(self, k):
        return getattr(self._m, k)


class DetectorRig:
    """The three YOLO models, loaded once, and a lean InferenceJob per replay."""

    def __init__(self, device: Optional[str] = None):
        import web.app as app
        from ultralytics import YOLO
        from web.field.detect_schema import DetectionSchema
        self.app = app
        wrap = (lambda m: _DeviceYOLO(m, device)) if device else (lambda m: m)
        self.model = wrap(YOLO(str(app.MODEL_PATH)))
        self.bumper_model = None
        self.det_schema = None
        if app.BUMPER_MODEL_PATH.exists():
            bm = YOLO(str(app.BUMPER_MODEL_PATH))
            self.det_schema = DetectionSchema.from_names(bm.names)
            self.bumper_model = wrap(bm)
        self.whole_robot_class_id = 1
        if self.det_schema is not None:
            for cid, col in self.det_schema.robot_class_color.items():
                if col is None:
                    self.whole_robot_class_id = cid
                    break
        self.person_model = wrap(YOLO(str(app.PROJECT_ROOT / "yolo26n.pt")))
        self.models = {"robot": str(app.MODEL_PATH),
                       "bumper": str(app.BUMPER_MODEL_PATH),
                       "person": "yolo26n.pt"}

    def make_job(self, config_overrides: dict, field_season):
        from web.job_config import JobConfig
        from web.field.field_model import FieldModel
        job = self.app.InferenceJob.__new__(self.app.InferenceJob)
        job.config = JobConfig(config_overrides)
        job.model = self.model
        job.bumper_model = self.bumper_model
        job.det_schema = self.det_schema
        job.whole_robot_class_id = self.whole_robot_class_id
        job.person_model = self.person_model
        job.sam3_export_scale = job.config.sam3_export_scale
        job.field_model = FieldModel.load(field_season)
        job.field_dims = (job.field_model.length_m, job.field_model.width_m)
        # make_frame_detect_fn records telemetry; a no-op sink.
        job.detection_archive = types.SimpleNamespace(record=lambda *a, **k: None)
        return job


def known_overrides(config: dict) -> tuple:
    """(overrides JobConfig accepts, unknown knob names dropped)."""
    from web.job_config import KNOBS
    names = {k.name for k in KNOBS}
    cfg = dict(config or {})
    unknown = sorted(k for k in cfg if k not in names)
    return {k: v for k, v in cfg.items() if k in names}, unknown


def top_crop_px(view, view_h: int, enabled: bool) -> int:
    frac = (getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0) if enabled else 0.0
    return int(view_h * frac) if frac > 0 else 0


def detect_match(rig: DetectorRig, job, views, video_path: str, targets: dict,
                 start_frame: Optional[int], iou_min: float, min_avail_gb: float,
                 log=print, mem_check_every: int = 100) -> tuple:
    """Run the live per-frame detector over every (frame, cam) the draw log
    has slot boxes for. Returns (per_cam, cams_of, stats):
    per_cam[(frame, slot)] = [(cam, conf|None, fusion weight)],
    cams_of[(frame, slot)] = {cams with a draw box}."""
    import cv2
    from web.field.calibration import push_from_camera
    from web.field.detect_bridge import corner_detection_weight
    app = rig.app
    scale = float(job.sam3_export_scale)
    by_name = {v.name: v for v in views}
    plans = {v.name: types.SimpleNamespace(source_frames=[0]) for v in views}
    fns = {}
    frames = sorted(targets)
    needed = set(frames)
    per_cam = defaultdict(list)
    cams_of = defaultdict(set)
    frame_confs = defaultdict(set)
    st = {"frames": 0, "calls": 0, "boxes": 0, "matched": 0, "matched_conf": 0,
          "missing_cam": defaultdict(int), "decode_short": 0, "mem_wait_s": 0.0,
          "detect_s": 0.0, "decode_s": 0.0}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    tmp = tempfile.TemporaryDirectory(prefix="detconf_")
    try:
        first = start_frame if start_frame is not None and start_frame <= frames[0] else frames[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        fpos = first
        last = frames[-1]
        while fpos <= last:
            t0 = time.perf_counter()
            if fpos in needed:
                ok, fr = cap.read()
            else:
                ok = cap.grab()
                fr = None
            st["decode_s"] += time.perf_counter() - t0
            if not ok:
                st["decode_short"] = sum(1 for f in frames if f >= fpos)
                log(f"  [decode] video ended at frame {fpos}; "
                    f"{st['decode_short']} draw frames unread")
                break
            if fpos in needed:
                st["frames"] += 1
                if st["frames"] % mem_check_every == 0:
                    st["mem_wait_s"] += wait_for_memory(min_avail_gb, log)
                for cam, entries in targets[fpos]["boxes"].items():
                    view = by_name.get(cam)
                    if view is None:
                        st["missing_cam"][cam] += len(entries)
                        continue
                    t1 = time.perf_counter()
                    crop = view.extract(fr)
                    y_off = top_crop_px(view, crop.shape[0], app.ROBOT_TRACK_TOP_CROP_ENABLED)
                    if y_off > 0:
                        crop = crop[y_off:, :]
                    small = cv2.resize(crop, (int(crop.shape[1] * scale),
                                              int(crop.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA)
                    path = os.path.join(tmp.name, f"{cam}.jpg")
                    cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    key = (cam, y_off)
                    if key not in fns:
                        fns[key] = job.make_frame_detect_fn(view, plans[cam], scale, y_off)
                    plans[cam].source_frames[0] = fpos
                    dets = fns[key](0, path)
                    st["detect_s"] += time.perf_counter() - t1
                    st["calls"] += 1
                    full = [(export_to_full(d[0], scale, y_off),
                             d[3] if len(d) > 3 else None) for d in dets]
                    frame_confs[fpos].update(round(float(c), 4) for _b, c in full
                                             if c is not None)
                    for slot, box in entries:
                        st["boxes"] += 1
                        conf, _iou, matched = match_box(box, full, iou_min)
                        st["matched"] += matched
                        st["matched_conf"] += conf is not None
                        w = 1.0
                        if cam != MAIN:
                            xy = None
                            try:
                                cal = view.calibration
                                uv = ((box[0] + box[2]) / 2.0, box[3])
                                xy = push_from_camera(cal, cal.image_to_field(uv),
                                                      job.config.ground_center_offset_m)
                            except Exception:
                                xy = None
                            w = app.CORNER_VOTE_WEIGHT * corner_detection_weight(
                                cam, xy, job.field_dims)
                        per_cam[(fpos, slot)].append((cam, conf, w))
                        cams_of[(fpos, slot)].add(cam)
            fpos += 1
    finally:
        cap.release()
        tmp.cleanup()
    st["missing_cam"] = dict(st["missing_cam"])
    st["frame_confs"] = frame_confs
    return per_cam, cams_of, st


# --------------------------------------------------------------------- driver
def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def pick_replays(conn, event: str, matches=None) -> list:
    """[(match_key, row, meta, meta_path)] -- the export's replay per match."""
    from web.scouting.loader import candidate_replays, pick_replay, _meta_path
    cand = candidate_replays(conn, event)
    keys = sorted(cand) if not matches else [m for m in matches]
    out = []
    for mk in keys:
        rows = cand.get(mk)
        if not rows:
            print(f"{mk}: no complete replay")
            continue
        row, meta, _legacy = pick_replay(rows)
        if row is None:
            print(f"{mk}: no readable replay")
            continue
        out.append((mk, row, meta, _meta_path(row)))
    return out


def gpu_peak() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    except Exception:
        pass
    return {}


def rss_peak_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def process_match(rig, event, mk, row, meta, meta_path, args) -> dict:
    from web.field.camera_view import load_views
    from web.field.composite_layout import load_layout
    from web.field import store as field_store
    from web.field.match_order import select_epoch
    t0 = time.perf_counter()
    summary = meta.get("summary") or {}
    replay_dir = Path(meta_path).parent
    draw = replay_dir / "draw.jsonl.gz"
    if not draw.exists():
        return {"status": "skip", "reason": "no draw.jsonl.gz"}
    video = row.get("source_video_path")
    if not video or not Path(video).exists():
        return {"status": "skip", "reason": f"no source video ({video})"}
    targets = read_draw_targets(draw)
    if not targets:
        return {"status": "skip", "reason": "draw log has no slot boxes"}

    overrides, unknown = known_overrides(summary.get("config"))
    layout_names = list(load_layout(event_key=event).keys())

    def epoch_pick(name):
        pick = select_epoch(field_store.load_versions(event, name), mk)
        return pick["version_id"] if pick else None
    sel, skip = view_selection(summary, layout_names, epoch_pick)
    if any(k in unknown for k in CORNER_DETECTOR_KNOBS):
        skip += [n for n in sel if n != MAIN and n not in skip]
        sel = {k: v for k, v in sel.items() if k == MAIN}
    views = [v for v in load_views(event, selection=sel) if v.name in sel]
    main_view = next((v for v in views if v.name == MAIN), None)
    if main_view is None or main_view.calibration is None:
        return {"status": "skip", "reason": "no main calibration"}
    job = rig.make_job(overrides, main_view.calibration.field_season)
    ranges = summary.get("processing_ranges") or []
    start = int(ranges[0][0]) if ranges and ranges[0] else None

    per_cam, cams_of, st = detect_match(
        rig, job, views, video, targets, start, args.iou, args.min_avail_gb)
    frame_confs = st.pop("frame_confs")
    values = slot_values(per_cam, targets, args.cam_pick)
    secs = round(time.perf_counter() - t0, 1)
    res = {"status": "ok", "seconds": secs, "views": [v.name for v in views],
           "skipped_cams": skip, "unknown_knobs": unknown, "stats": st,
           "rss_peak_gb": rss_peak_gb(), **gpu_peak()}
    if args.compare:
        res["compare"] = {}
        for mode in (args.cam_pick, "max" if args.cam_pick == "weight" else "weight"):
            vals = values if mode == args.cam_pick else slot_values(per_cam, targets, mode)
            for source in ("frames", "auto_paths"):
                c = compare_stats(meta, vals, cams_of, source, frame_confs)
                if c["total"]["live"]:
                    res["compare"][f"{source}/{mode}"] = c
        return res
    flat = {k: v[0] for k, v in values.items()}
    n_rec, n_fill, n_ap, n_ap_fill = patch_meta(meta, flat)
    res.update(n_robot_records=n_rec, n_filled=n_fill, n_auto_samples=n_ap,
               n_auto_filled=n_ap_fill)
    if args.dry_run:
        return res
    meta.setdefault("summary", {})["det_conf_backfill"] = {
        "script_version": SCRIPT_VERSION, "git_commit": git_commit(),
        "iou": args.iou, "cam_pick": args.cam_pick,
        "n_robot_records": n_rec, "n_filled": n_fill,
        "n_auto_samples": n_ap, "n_auto_filled": n_ap_fill,
        "cams": [v.name for v in views], "skipped_cams": skip,
        "unknown_knobs": unknown, "models": rig.models,
        "seconds": secs, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_meta_atomic(Path(meta_path), meta)
    res["written"] = str(meta_path)
    return res


def _fmt_cmp(name, b) -> str:
    def f(v, nd=3):
        return "-" if v is None else f"{v:.{nd}f}"
    return (f"    {name:<28} n={b['n']:>6} live={b['live']:>6} "
            f"fill_on_live={f(b['fill_on_live'])} bf_where_live_null={b['bf_where_live_null']:>5} "
            f"live_in_frame_dets={b['live_in_frame_dets']:>5} "
            f"|d| med={f(b['diff_median'], 4)} p90={f(b['diff_p90'], 4)} "
            f"max={f(b['diff_max'], 4)} within.01={f(b['within_0.01'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event", required=True)
    ap.add_argument("--match", action="append", help="match key (repeatable)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run replays that already have a non-null det_conf")
    ap.add_argument("--dry-run", action="store_true", help="compute, write nothing")
    ap.add_argument("--compare", action="store_true",
                    help="new-code replay: compare backfill vs live det_conf, write nothing")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--cam-pick", choices=("weight", "max"), default="weight")
    ap.add_argument("--min-avail-gb", type=float, default=2.5)
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: ultralytics')")
    ap.add_argument("--json-out", default=None, help="write per-match results here")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    try:
        os.nice(10)
    except OSError:
        pass
    from web.scouting.loader import default_db_path
    conn = sqlite3.connect(str(args.db or default_db_path()))
    todo = pick_replays(conn, args.event, args.match)
    rig = None
    results = {}
    done = 0
    for mk, row, meta, meta_path in todo:
        if args.limit is not None and done >= args.limit:
            break
        reason = None if args.compare else skip_reason(meta, args.force)
        if args.compare and not has_det_conf(meta):
            reason = "no live det_conf to compare against"
        if reason:
            print(f"{mk}: skip ({reason})")
            results[mk] = {"status": "skip", "reason": reason}
            continue
        wait_for_memory(args.min_avail_gb)
        if rig is None:
            rig = DetectorRig(args.device)
        try:
            res = process_match(rig, args.event, mk, row, meta, meta_path, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        results[mk] = res
        done += res.get("status") == "ok"
        if res.get("status") != "ok":
            print(f"{mk}: {res['status']} ({res.get('reason')})")
            continue
        st = res["stats"]
        line = (f"{mk}: {res['seconds']}s frames={st['frames']} calls={st['calls']} "
                f"boxes={st['boxes']} matched={st['matched']} "
                f"with_conf={st['matched_conf']} views={res['views']}"
                f" rss_peak={res['rss_peak_gb']}GB"
                f" vram_reserved={res.get('max_reserved_gb')}GB")
        if "n_filled" in res:
            line += (f" filled={res['n_filled']}/{res['n_robot_records']}"
                     f" auto={res['n_auto_filled']}/{res['n_auto_samples']}")
        if res["skipped_cams"] or res["unknown_knobs"] or st["missing_cam"]:
            line += (f" skipped_cams={res['skipped_cams']} unknown_knobs="
                     f"{res['unknown_knobs']} missing_cam={st['missing_cam']}")
        line += " [dry-run]" if args.dry_run else (" [compare]" if args.compare
                                                   else " WRITTEN")
        print(line, flush=True)
        if args.compare:
            for name, c in res["compare"].items():
                print(f"  {name}  (live visible-but-null: {c['live_visible_but_null']})")
                print(_fmt_cmp("TOTAL", c["total"]))
                for k, b in c["by_cam_set"].items():
                    print(_fmt_cmp(f"cams={k}", b))
                for k, b in c["by_picked_cam"].items():
                    print(_fmt_cmp(f"picked={k}", b))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

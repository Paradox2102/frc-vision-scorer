#!/usr/bin/env python
"""Replay auditor: estimate the mis-ID rate of a stored replay's team labels
with no hand annotation.

A stratified sample of (frame, slot) moments is drawn from the replay; for each
one a crop of the robot at the slot's tracked spot is cut at native resolution
from EVERY camera view that sees it (main + both corners), deinterlaced,
upscaled and tick-marked. A blind judge (Claude Code subagents following
data/eval/audit/JUDGE_BRIEF.md, or Jev via the ``jev`` subcommand) names the
team or abstains. Mis-ID estimate = judge-confident disagreements with the
pipeline label / judge-confident labelled samples, weighted by slot-time.

Subcommands
-----------
    sample  --replay <id> [--n 150 --seed 0]
            -> data/eval/audit/<match_key>_<replay8>/{audit.json, samples.jsonl,
               items.jsonl, items/<sample_id>/{item.json, main.jpg, ...}}
    jev     --audit <dir> [--mode multi|views] [--out answers_jev.jsonl]
            Jev as the judge (CALLS THE JEV API, ~1 cent per 150 items).
    score   --audit <dir> --answers <file.jsonl> [--min-conf 0.7]
            Judge coverage, agreement, estimated mis-ID (and seconds), per slot /
            team; against hand GT when the match has it (2026cur_qm8), the
            calibration table.

Answers file (JSONL), one row per item:
    {"sample_id": "s012", "team": "2637" | "none" | "unsure" | "no_robot",
     "confidence": 0.0-1.0 | "high" | "medium" | "low"}
``none``: a robot is there and its digits are legible but they are not a listed
team; ``unsure``: a robot is there, team not determinable; ``no_robot``: no robot
at the marked spot.

Geometry is the window-label stage's (web/field/window_crops.py): the slot's
field_pos is pushed half a robot away from the MAIN camera (the pipeline tracks
the near bumper edge), the footprint centre at 0.10 m is projected into each
view with the calibration version in force when the job ran
(scripts/build_track_truth.job_views), and the crop side is 150 px + 0.5 x the
projected width of a 0.95 m bumper in that view.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

AUDIT_ROOT = REPO / "data" / "eval" / "audit"
SCORE_SHIFT_M = 0.475            # the GT scorer's toward-camera correction (relabel_replay)
GT_RADIUS_M = 1.0                # score_match's default match radius
OUT_PX = 512                     # every view crop is written at OUT_PX x OUT_PX
MIN_BUMPER_PX = 16.0             # projected 0.95 m bumper narrower than this: view skipped
MAX_OFF_AXIS_DEG = 75.0          # target further off a camera's axis than this: skipped
MIN_DEPTH_M = 0.3                # target must be at least this far in front of the camera
VIEW_ORDER = ("main", "corner_left", "corner_right")
# Corner views: the corner calibrations (2026-06-30) are only good near each
# camera. Measured on qm8/qm13 against the job's own corner detections, the
# projected spot is within 1 bumper width of a detection for <10% of targets
# whose projected bumper is < 40 px and only ~50-75% of those >= 100 px (main:
# 80%). So a corner view is used only when its projected bumper is at least
# CORNER_MIN_BUMPER_PX and ONE detection of the job (same or adjacent processed
# frame) lies within SNAP_R_BW bumper widths of the projection (the runner-up
# at least SNAP_MARGIN times farther); the crop is then centred on that
# detection. Detection colour is not used (it would leak the slot's alliance).
CORNER_MIN_BUMPER_PX = 60.0
SNAP_R_BW = 1.5
SNAP_MARGIN = 1.5
CONF_WORDS = {"high": 0.9, "medium": 0.7, "med": 0.7, "low": 0.4}
ESCAPES = ("none", "unsure", "no_robot")
DEFAULT_MIN_CONF = 0.7
UNLABELLED_FRAC = 0.10
AUTO_MIN_SHARE = 0.20


# --------------------------------------------------------------------------- #
# Pure: sampling
# --------------------------------------------------------------------------- #

def period_of(frame_num: int, periods: list) -> Optional[str]:
    """'auto' / 'teleop' (the summary period whose processing range holds the
    frame), or None outside every processed period."""
    for p in periods or []:
        a = p.get("processing_start_frame", p.get("start_frame"))
        b = p.get("processing_end_frame", p.get("end_frame"))
        if a is not None and b is not None and a <= frame_num <= b:
            return str(p.get("label") or "match")
    return None


def slot_units(frames: list, periods: list) -> list:
    """Every (frame, slot) moment with a field position inside a processed
    period. A slot label repeated within one frame gets a ``#2`` suffix (as in
    gt_scoring.replay_from_meta); ``dup`` is that ordinal (1-based)."""
    out = []
    for i, f in enumerate(frames):
        fn = f.get("frame_num")
        if fn is None:
            continue
        per = period_of(int(fn), periods) if periods else "match"
        if per is None:
            continue
        seen: Counter = Counter()
        for r in f.get("robots") or []:
            pos = r.get("field_pos")
            if not r.get("slot") or not pos or len(pos) < 2 or pos[0] is None or pos[1] is None:
                continue
            seen[r["slot"]] += 1
            k = seen[r["slot"]]
            team = r.get("team_number")
            out.append({"idx": i, "frame": int(fn), "slot": r["slot"], "dup": k,
                        "slot_key": r["slot"] if k == 1 else f"{r['slot']}#{k}",
                        "alliance": r.get("alliance"),
                        "team": None if team in (None, "") else str(team),
                        "xy": (float(pos[0]), float(pos[1])), "period": per})
    return out


def stratum_of(u: dict) -> str:
    """Labelled units stratify by (period, slot); unlabelled ones by period only."""
    if u["team"] is None:
        return f"unlabelled|{u['period']}"
    return f"{u['period']}|{u['slot']}"


def allocate(sizes: dict, n: int, min_each: int = 1) -> dict:
    """Split ``n`` over strata proportionally to ``sizes`` (largest remainder),
    each non-empty stratum getting at least ``min(min_each, size)`` and none
    more than its size."""
    keys = [k for k in sorted(sizes) if sizes[k] > 0]
    if not keys or n <= 0:
        return {k: 0 for k in sizes}
    n = min(n, sum(sizes[k] for k in keys))
    out = {k: min(min_each, sizes[k]) for k in keys}
    if sum(out.values()) >= n:                     # n too small for the floor
        order = sorted(keys, key=lambda k: (-sizes[k], k))
        out = {k: 0 for k in keys}
        for k in order[:n]:
            out[k] = 1
        return {**{k: 0 for k in sizes}, **out}
    while sum(out.values()) < n:
        left = n - sum(out.values())
        open_ = [k for k in keys if out[k] < sizes[k]]
        tot = sum(sizes[k] for k in open_)
        quota = {k: left * sizes[k] / tot for k in open_}
        add = {k: min(int(quota[k]), sizes[k] - out[k]) for k in open_}
        if sum(add.values()) == 0:               # hand out by largest remainder
            k = max(open_, key=lambda k: (quota[k] - int(quota[k]), sizes[k], k))
            add[k] = 1
        for k, a in add.items():
            out[k] += a
    return {**{k: 0 for k in sizes}, **out}


def systematic(seq: list, k: int, rng: random.Random) -> list:
    """``k`` items spread evenly over ``seq`` (already time-ordered): one per
    equal-width bin, at a random offset shared by all bins."""
    n = len(seq)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(seq)
    step = n / k
    off = rng.uniform(0, step)
    return [seq[min(n - 1, int(off + j * step))] for j in range(k)]


def plan_sample(units: list, n: int, seed: int = 0, unlabelled_frac: float = UNLABELLED_FRAC,
                auto_min_share: float = AUTO_MIN_SHARE) -> tuple[list, dict]:
    """Stratified sample. Returns (samples, strata) where strata[key] = {"N":
    units in the stratum, "n": sampled}. Unlabelled units get
    ``unlabelled_frac`` of n (by period, proportional). Labelled units: auto gets
    at least ``auto_min_share`` of the labelled draw, then every (period, slot)
    stratum a proportional share with a floor of 1; systematic over time within
    each stratum. Sample ids are assigned in a shuffled order, so they say
    nothing about stratum or time."""
    rng = random.Random(seed)
    by: dict = defaultdict(list)
    for u in units:
        by[stratum_of(u)].append(u)
    for v in by.values():
        v.sort(key=lambda u: (u["frame"], u["slot_key"]))
    lab_keys = [k for k in by if not k.startswith("unlabelled|")]
    unl_keys = [k for k in by if k.startswith("unlabelled|")]
    n_unl = min(sum(len(by[k]) for k in unl_keys), int(round(n * unlabelled_frac)))
    n_lab = n - n_unl
    per_period: dict = defaultdict(int)
    for k in lab_keys:
        per_period[k.split("|")[0]] += len(by[k])
    period_n = allocate(dict(per_period), n_lab, min_each=1)
    if "auto" in period_n and len(period_n) > 1 and per_period["auto"]:
        want = min(per_period["auto"], max(period_n["auto"], int(round(auto_min_share * n_lab))))
        rest = {p: c for p, c in per_period.items() if p != "auto"}
        period_n = {"auto": want, **allocate(rest, n_lab - want, min_each=1)}
    alloc: dict = {}
    for p, m in period_n.items():
        alloc.update(allocate({k: len(by[k]) for k in lab_keys if k.split("|")[0] == p}, m))
    alloc.update(allocate({k: len(by[k]) for k in unl_keys}, n_unl, min_each=1))
    picks = []
    for k in sorted(alloc):
        for u in systematic(by[k], alloc[k], rng):
            picks.append({**u, "stratum": k})
    rng.shuffle(picks)
    for j, s in enumerate(picks):
        s["sample_id"] = f"s{j:03d}"
    strata = {k: {"N": len(by[k]), "n": alloc.get(k, 0)} for k in sorted(by)}
    return picks, strata


# --------------------------------------------------------------------------- #
# Pure: answers and scoring math
# --------------------------------------------------------------------------- #

def parse_conf(c) -> float:
    if c is None:
        return 0.0
    if isinstance(c, str):
        s = c.strip().lower()
        if s in CONF_WORDS:
            return CONF_WORDS[s]
        try:
            c = float(s)
        except ValueError:
            return 0.0
    c = float(c)
    if c > 1.0:                                   # 0-100 scale
        c /= 100.0
    return max(0.0, min(1.0, c))


def parse_answer(row: dict, roster: set) -> dict:
    """-> {"kind": "team"|"none"|"unsure"|"no_robot"|"invalid", "team", "conf"}.
    A team outside the roster counts as ``none`` (legible, not listed)."""
    t = row.get("team")
    t = None if t is None else str(t).strip()
    conf = parse_conf(row.get("confidence"))
    if t is None or t == "" or t.lower() in ("unsure", "unknown", "null"):
        return {"kind": "unsure", "team": None, "conf": conf}
    tl = t.lower()
    if tl in ("no_robot", "norobot", "no robot"):
        return {"kind": "no_robot", "team": None, "conf": conf}
    if tl in ("none", "other"):
        return {"kind": "none", "team": None, "conf": conf}
    if tl.startswith("frc"):
        t = t[3:]
    if t in roster:
        return {"kind": "team", "team": t, "conf": conf}
    return {"kind": "none", "team": None, "conf": conf}


def sample_weights(samples: list, strata: dict, frame_period_s: float) -> dict:
    """sample_id -> slot-seconds it stands for: N_stratum / n_present x period.
    ``samples`` are the judgeable ones (an item with no usable view drops out
    and its stratum's weight is spread over the rest)."""
    present = Counter(s["stratum"] for s in samples)
    return {s["sample_id"]: strata[s["stratum"]]["N"] / present[s["stratum"]] * frame_period_s
            for s in samples}


def ratio(pairs: list) -> Optional[float]:
    """Weighted rate from [(weight, is_hit)]; None when the weight sum is 0."""
    den = sum(w for w, _ in pairs)
    return None if den <= 0 else sum(w for w, h in pairs if h) / den


def misid_estimate(recs: list, min_conf: float) -> dict:
    """``recs``: labelled samples with "w", "pipe" (pipeline team), "ans"
    (parse_answer) and "stratum". Rates are slot-time weighted."""
    judged = [r for r in recs if r.get("ans")]
    conf = [r for r in judged if r["ans"]["kind"] == "team" and r["ans"]["conf"] >= min_conf]
    phantom = [r for r in judged if r["ans"]["kind"] == "no_robot" and r["ans"]["conf"] >= min_conf]
    out = {
        "n_items": len(recs), "n_judged": len(judged), "n_confident": len(conf),
        "n_no_robot": len(phantom),
        "coverage": ratio([(r["w"], r in conf) for r in judged]),
        "coverage_unw": (len(conf) / len(judged)) if judged else None,
        "misid": ratio([(r["w"], r["ans"]["team"] != r["pipe"]) for r in conf]),
        "misid_unw": (sum(r["ans"]["team"] != r["pipe"] for r in conf) / len(conf)) if conf else None,
        "n_disagree": sum(r["ans"]["team"] != r["pipe"] for r in conf),
        "no_robot_rate": ratio([(r["w"], r in phantom) for r in judged]),
        "labelled_s": sum(r["w"] for r in recs),
    }
    out["misid_s"] = None if out["misid"] is None else out["misid"] * out["labelled_s"]
    return out


def bootstrap_ci(recs: list, min_conf: float, reps: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Optional[tuple]:
    """Stratified bootstrap CI of ``misid_estimate(...)['misid']``: resample
    items with replacement within each stratum, keep each item's weight."""
    by: dict = defaultdict(list)
    for r in recs:
        by[r["stratum"]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        boot = []
        for v in by.values():
            boot += [v[rng.randrange(len(v))] for _ in v]
        m = misid_estimate(boot, min_conf)["misid"]
        if m is not None:
            vals.append(m)
    if len(vals) < reps // 2:
        return None
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


def breakdown(recs: list, key: str, min_conf: float) -> dict:
    """{value of r[key]: (confident n, disagreements, weighted wrong rate)}."""
    by: dict = defaultdict(list)
    for r in recs:
        if r.get("ans") and r["ans"]["kind"] == "team" and r["ans"]["conf"] >= min_conf:
            by[r[key]].append(r)
    return {k: (len(v), sum(r["ans"]["team"] != r["pipe"] for r in v),
                ratio([(r["w"], r["ans"]["team"] != r["pipe"]) for r in v]))
            for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))}


def gt_match(slot_key: str, pipe_xy: dict, gt_xy: dict, radius: float = GT_RADIUS_M):
    """Which GT robot the pipeline slot ``slot_key`` is on at one moment.
    ``pipe_xy``: {slot_key: (x, y)} for every pipeline slot then (shifted like
    the GT scorer); ``gt_xy``: {team: (x, y)} for every GT robot with a
    position. One Hungarian assignment (cost = distance, gated at ``radius``),
    so two slots can't both claim one robot. Returns (team or None, distance
    to the nearest GT robot or None)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if slot_key not in pipe_xy or not gt_xy:
        return None, None
    slots, teams = list(pipe_xy), list(gt_xy)
    D = np.array([[math.hypot(pipe_xy[s][0] - gt_xy[t][0], pipe_xy[s][1] - gt_xy[t][1])
                   for t in teams] for s in slots])
    near = float(D[slots.index(slot_key)].min())
    C = np.where(D <= radius, D, 1e6)
    rows, cols = linear_sum_assignment(C)
    for r, c in zip(rows, cols):
        if slots[r] == slot_key and D[r, c] <= radius:
            return teams[c], near
    return None, near


def calibration_table(recs: list, bins=(0.0, 0.5, 0.7, 0.9, 1.01)) -> list:
    """Judge team answers on GT-matched samples, by confidence bin:
    [(lo, hi, n, n_right)]."""
    out = []
    for lo, hi in zip(bins, bins[1:]):
        v = [r for r in recs if r.get("gt") and r.get("ans") and r["ans"]["kind"] == "team"
             and lo <= r["ans"]["conf"] < hi]
        out.append((lo, hi, len(v), sum(r["ans"]["team"] == r["gt"] for r in v)))
    return out


# --------------------------------------------------------------------------- #
# Geometry and crops
# --------------------------------------------------------------------------- #

def centre_xy(cal_main, field_pos, ground_center_offset_m: float = 0.0):
    """Footprint centre: field_pos pushed the rest of half a robot away from
    the main camera (window_crops.target)."""
    from web.field.calibration import push_from_camera
    from web.field.window_crops import ROBOT_CENTRE_OFFSET_M
    off = max(0.0, ROBOT_CENTRE_OFFSET_M - float(ground_center_offset_m or 0.0))
    return push_from_camera(cal_main, (float(field_pos[0]), float(field_pos[1])), off)


def _in_front(cal, xyz) -> bool:
    """The point is in front of the camera and within MAX_OFF_AXIS_DEG of its
    axis (fisheye corner cameras project points behind or far to the side to
    finite but meaningless pixels)."""
    import cv2
    import numpy as np
    if getattr(cal, "mode", None) == "homography" or getattr(cal, "rvec", None) is None:
        return True
    R, _ = cv2.Rodrigues(np.asarray(cal.rvec, float))
    pc = R @ np.asarray(xyz, float) + np.asarray(cal.tvec, float).reshape(3)
    if pc[2] < MIN_DEPTH_M:
        return False
    return math.degrees(math.atan2(math.hypot(pc[0], pc[1]), pc[2])) <= MAX_OFF_AXIS_DEG


def view_target(cal, crop: tuple, xy) -> Optional[dict]:
    """Where the footprint centre ``xy`` lands in one view: {u, v (view px),
    bumper_px, side, ok, why}. ``ok`` is False when it can't be cut usefully."""
    import numpy as np
    from web.field.window_crops import (CROP_BASE_PX, CROP_MAX_PX, CROP_MIN_PX,
                                        CROP_WIDTH_GAIN, HALF_BUMPER_M, TARGET_Z_M, in_view)
    w, h = crop[2], crop[3]
    if xy is None:
        return None
    if not _in_front(cal, (xy[0], xy[1], TARGET_Z_M)):
        return {"ok": False, "why": "behind/off-axis"}
    try:
        u, v = cal.field_to_image((xy[0], xy[1], TARGET_Z_M))
        spans = []
        for dx, dy in ((HALF_BUMPER_M, 0.0), (0.0, HALF_BUMPER_M)):
            a = cal.field_to_image((xy[0] - dx, xy[1] - dy, TARGET_Z_M))
            b = cal.field_to_image((xy[0] + dx, xy[1] + dy, TARGET_Z_M))
            spans.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
    except Exception:
        return {"ok": False, "why": "projection failed"}
    if not np.all(np.isfinite([u, v] + spans)):
        return {"ok": False, "why": "projection failed"}
    bw = max(spans)
    side = int(round(min(CROP_MAX_PX, max(CROP_MIN_PX, CROP_BASE_PX + CROP_WIDTH_GAIN * bw))))
    t = {"u": round(float(u), 1), "v": round(float(v), 1), "bumper_px": round(bw, 1),
         "side": side, "ok": True, "why": None}
    if not in_view(u, v, (w, h)):
        t.update(ok=False, why="outside view")
    elif bw < MIN_BUMPER_PX:
        t.update(ok=False, why=f"bumper {bw:.0f}px < {MIN_BUMPER_PX:.0f}px")
    return t


def snap_to_detection(t: dict, dets: list, r_bw: float = SNAP_R_BW,
                      margin: float = SNAP_MARGIN) -> Optional[tuple]:
    """``dets``: [(cx, cy)] detection box centres in the view. The one centre
    within ``r_bw`` bumper widths of the projected target, if it is unique
    (runner-up >= ``margin`` x farther); else None. Returns (cx, cy, dist_bw)."""
    if not dets or not t or not t.get("bumper_px"):
        return None
    bw = float(t["bumper_px"])
    ds = sorted((math.hypot(cx - t["u"], cy - t["v"]) / bw, cx, cy) for cx, cy in dets)
    d0, cx, cy = ds[0]
    if d0 > r_bw:
        return None
    if len(ds) > 1 and ds[1][0] < margin * max(d0, 0.25):
        return None
    return cx, cy, d0


def load_detections(job_id: str) -> dict:
    """{view: {source_frame: [(cx, cy)]}} from the job's detections file
    (view-local pixels), or {} when the job wrote none."""
    p = REPO / "web" / "uploads" / f"detections_{job_id}.jsonl"
    out: dict = defaultdict(dict)
    if not p.exists():
        return {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["view"]][int(d["source_frame"])] = [
            ((b["box"][0] + b["box"][2]) / 2.0, (b["box"][1] + b["box"][3]) / 2.0)
            for b in d.get("dets") or [] if b.get("box")]
    return dict(out)


def dets_near(view_dets: dict, frame: int, max_df: int) -> Optional[list]:
    """Detections of the processed frame nearest ``frame`` within ``max_df``."""
    if not view_dets:
        return None
    if frame in view_dets:
        return view_dets[frame]
    best = min(view_dets, key=lambda f: (abs(f - frame), f))
    return view_dets[best] if abs(best - frame) <= max_df else None


def corner_gate(t: dict, dets: Optional[list]) -> dict:
    """Apply the corner-view rule (see CORNER_MIN_BUMPER_PX) to one target."""
    t = dict(t)
    if t["bumper_px"] < CORNER_MIN_BUMPER_PX:
        t.update(ok=False, why=f"corner bumper < {CORNER_MIN_BUMPER_PX:.0f}px")
        return t
    snap = snap_to_detection(t, dets or [])
    if snap is None:
        t.update(ok=False, why="no unique detection near projection")
        return t
    t.update(proj_u=t["u"], proj_v=t["v"], u=round(snap[0], 1), v=round(snap[1], 1),
             snapped_bw=round(snap[2], 2))
    return t


def cut_square(view_img, u: float, v: float, side: int):
    """Square ``side`` crop around (u, v) placed like window_crops.wide_tile
    (target at 50% across, 62% down), clamped inside the view. Returns
    (crop, x0, y0, s)."""
    from web.field.window_crops import TARGET_X, TARGET_Y
    h, w = view_img.shape[:2]
    s = int(max(2, min(side, w, h)))
    x0 = max(0, min(w - s, int(u - s * TARGET_X)))
    y0 = max(0, min(h - s, int(v - s * TARGET_Y)))
    return view_img[y0:y0 + s, x0:x0 + s].copy(), x0, y0, s


def draw_ticks(img, tx: float, ty: float, size: int = 22) -> None:
    """White, black-edged triangles on the four edges pointing at (tx, ty)."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    tx, ty = min(max(int(tx), 0), w - 1), min(max(int(ty), 0), h - 1)
    half = size // 2
    for px, py, dx, dy in ((tx, 0, 0, 1), (tx, h - 1, 0, -1), (0, ty, 1, 0), (w - 1, ty, -1, 0)):
        tri = np.array([(px + dx * size, py + dy * size), (px - dy * half, py - dx * half),
                        (px + dy * half, py + dx * half)], np.int32)
        cv2.fillConvexPoly(img, tri, (255, 255, 255))
        cv2.polylines(img, [tri], True, (0, 0, 0), 2)


def render_view(view_img, t: dict) -> tuple:
    """Cut, deinterlace (web/bumper_register.deinterlace_crop), upscale to
    OUT_PX and tick-mark one view crop. Returns (image, meta)."""
    import cv2
    from web.bumper_register import comb_ratio, deinterlace_crop
    crop, x0, y0, s = cut_square(view_img, t["u"], t["v"], t["side"])
    comb = comb_ratio(crop)
    di = deinterlace_crop(crop)
    out = cv2.resize(di, (OUT_PX, OUT_PX), interpolation=cv2.INTER_CUBIC)
    k = OUT_PX / float(s)
    tx, ty = (t["u"] - x0) * k, (t["v"] - y0) * k
    draw_ticks(out, tx, ty)
    return out, {"x0": x0, "y0": y0, "side_px": s, "scale": round(k, 3),
                 "tick": [round(tx), round(ty)], "comb": round(comb, 3),
                 "deinterlaced": di is not crop}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_replay_inputs(replay_id: str, meta_path: Optional[str] = None) -> dict:
    """Replay row, meta (``meta_path`` overrides the replay's own meta.json,
    e.g. an offline relabel: labels from it, positions are the same), video,
    views, roster and fps."""
    from scripts.build_track_truth import job_views
    from scripts.relabel_replay import job_roster
    from scripts.score_ground_truth import REPLAYS_DIR, _probe_fps, _resolve, replay_row
    row = replay_row(replay_id)
    if meta_path:
        meta_p = Path(meta_path)
    else:
        meta_p = REPLAYS_DIR / row["id"] / "meta.json"
        if not meta_p.exists():
            meta_p = _resolve(row.get("meta_path"))
    if meta_p is None or not meta_p.exists():
        raise SystemExit(f"meta.json missing for replay {row['id']}")
    video = _resolve(row.get("source_video_path"))
    meta = json.loads(meta_p.read_text())
    mk = row["match_key"]
    views = job_views(mk.split("_")[0], mk, row["created_at"])
    red, blue = job_roster(mk)
    fps = _probe_fps(video) if video else None
    return {"row": row, "meta": meta, "video": video, "views": views, "red": red,
            "blue": blue, "fps": fps or 60.0}


def frame_period_s(frames: list, fps: float) -> float:
    from web.field.window_labels import frame_period_s as fp
    return fp(frames, fps) or 0.2


def audit_dir_for(row: dict) -> Path:
    return AUDIT_ROOT / f"{row['match_key']}_{row['id'][:8]}"


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list) -> None:
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #

def cmd_sample(args) -> None:
    import cv2
    from web.field.window_crops import iter_view_frames
    inp = load_replay_inputs(args.replay, getattr(args, "meta", None))
    row, meta, views = inp["row"], inp["meta"], inp["views"]
    if inp["video"] is None:
        raise SystemExit(f"source video missing for replay {row['id']}")
    frames = meta["frames"]
    periods = (meta.get("summary") or {}).get("periods") or []
    units = slot_units(frames, periods)
    samples, strata = plan_sample(units, args.n, args.seed, args.unlabelled_frac,
                                  args.auto_min_share)
    out_dir = Path(args.out) if args.out else audit_dir_for(row)
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    cal_main = next(c for n, _, c in views if n == "main")
    period_s = frame_period_s(frames, inp["fps"])
    step = int(round(period_s * inp["fps"]))
    dets = load_detections(row["id"])
    if not dets:
        print("WARNING: no detections file for this job: corner views are skipped")
    print(f"replay {row['id'][:8]} {row['match_key']}  {len(units)} slot-frames "
          f"({sum(u['team'] is not None for u in units)} labelled), {len(samples)} samples "
          f"-> {out_dir}")

    want: dict = defaultdict(list)
    for s in samples:
        xy = centre_xy(cal_main, s["xy"], args.ground_center_offset)
        s["centre_xy"] = [round(xy[0], 3), round(xy[1], 3)] if xy else None
        s["targets"] = {}
        for name, crop, cal in views:
            t = view_target(cal, crop, xy)
            if t and t["ok"] and name != "main":
                t = corner_gate(t, dets_near(dets.get(name) or {}, s["frame"], step))
            if t:
                s["targets"][name] = t
        want[s["frame"]].append(s)
    W = max(c[0] + c[2] for _, c, _ in views)
    H = max(c[1] + c[3] for _, c, _ in views)
    t0 = time.time()
    n_dec = 0
    for fnum, img in iter_view_frames(inp["video"], list(want), (0, 0, W, H)):
        n_dec += 1
        for s in want[fnum]:
            s["views"], s["view_meta"] = [], {}
            d = out_dir / "items" / s["sample_id"]
            d.mkdir(parents=True, exist_ok=True)
            for name, (x, y, w, h), _cal in views:
                t = s["targets"].get(name)
                if not t or not t["ok"]:
                    continue
                im, vm = render_view(img[y:y + h, x:x + w], t)
                cv2.imwrite(str(d / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                s["views"].append(name)
                s["view_meta"][name] = vm
    print(f"decoded {n_dec}/{len(want)} frames in {time.time() - t0:.0f}s")

    roster = {"red": inp["red"], "blue": inp["blue"]}
    recs, items = [], []
    for s in sorted(samples, key=lambda s: s["sample_id"]):
        s.setdefault("views", [])
        s["views"].sort(key=VIEW_ORDER.index)
        rec = {"sample_id": s["sample_id"], "replay": row["id"], "match_key": row["match_key"],
               "frame": s["frame"], "frame_idx": s["idx"], "t_s": round(s["frame"] / inp["fps"], 2),
               "period": s["period"], "slot": s["slot"], "slot_key": s["slot_key"],
               "dup": s["dup"], "alliance": s["alliance"], "pipeline_label": s["team"],
               "stratum": s["stratum"], "field_pos": [round(c, 3) for c in s["xy"]],
               "centre_xy": s["centre_xy"], "views": s["views"],
               "targets": s["targets"], "view_meta": s.get("view_meta", {})}
        d = out_dir / "items" / s["sample_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "item.json").write_text(json.dumps(rec, indent=1))
        recs.append(rec)
        if s["views"]:
            items.append({"sample_id": s["sample_id"],      # paths relative to the audit dir
                          "images": {n: f"items/{s['sample_id']}/{n}.jpg" for n in s["views"]},
                          "roster": roster})
    write_jsonl(out_dir / "samples.jsonl", recs)
    write_jsonl(out_dir / "items.jsonl", items)
    audit = {"replay": row["id"], "match_key": row["match_key"],
             "replay_created_at": row["created_at"], "video": inp["video"].name,
             "meta": getattr(args, "meta", None),
             "fps": inp["fps"], "frame_period_s": period_s, "n_requested": args.n,
             "seed": args.seed, "unlabelled_frac": args.unlabelled_frac,
             "auto_min_share": args.auto_min_share,
             "ground_center_offset_m": args.ground_center_offset, "roster": roster,
             "strata": strata,
             "labelled_slot_s": round(sum(u["team"] is not None for u in units) * period_s, 1),
             "unlabelled_slot_s": round(sum(u["team"] is None for u in units) * period_s, 1),
             "views": {n: list(c) for n, c, _ in views},
             "out_px": OUT_PX, "min_bumper_px": MIN_BUMPER_PX,
             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=1))
    vc = Counter(len(r["views"]) for r in recs)
    pv = Counter(v for r in recs for v in r["views"])
    why = Counter(f"{n}: {t['why']}" for r in recs for n, t in r["targets"].items() if not t["ok"])
    print(f"items {len(items)}/{len(recs)} with >=1 view; views per item {dict(sorted(vc.items()))}; "
          f"per view {dict(pv)}")
    print("skipped views: " + ", ".join(f"{k} x{v}" for k, v in why.most_common()))


# --------------------------------------------------------------------------- #
# jev
# --------------------------------------------------------------------------- #

def jev_multi_payload(model_key: str, imgs: dict, red: list, blue: list) -> tuple:
    """One Jev request with every view crop of one moment (``imgs``: view name
    -> JPEG bytes). Same option set and wording as the window-label digit
    request (web/jev_reader.build_jev_window_payload)."""
    from web import jev_reader as jr
    state, assets, sizes = {}, {}, []
    names = {"main": "main_view", "corner_left": "left_corner_view",
             "corner_right": "right_corner_view"}
    for n, jpg in imgs.items():
        ref = f"{names.get(n, n)}_image"
        a, wh = jr.jev_asset(jpg)
        state[names.get(n, n)] = ref
        assets[ref] = a
        sizes.append(wh)
    views_txt = ", ".join(f"`{names.get(n, n)}`" for n in imgs)
    context = (
        f"{views_txt} {'are crops' if len(imgs) > 1 else 'is a crop'} cut at the SAME moment "
        "from different cameras filming an FRC (FIRST Robotics Competition) match, around the "
        "spot where a tracker believes one robot is; every crop shows that same spot from its "
        "own camera (the side-camera spots can be off by up to a robot's width). "
        + jr.WINDOW_MARK + jr.WINDOW_BUMPER + jr._window_teams_line(red, blue))
    criteria = {t: f"Team {t} (red bumper)" for t in red}
    criteria.update({t: f"Team {t} (blue bumper)" for t in blue})
    criteria["not_robot"] = jr.WINDOW_NOT_ROBOT
    criteria.update({e: jr.JEV_ESCAPE_TEXT[e] for e in ("unreadable", "printed", "other")})
    q = ("Which option matches the white team-number digits on the bumper of the robot at the "
         "marked spot, using every crop? Choose an escape option unless you can actually see a "
         "listed team's digits on that robot's bumper in at least one crop.")
    state["context"] = context
    return ({"model": jr.JEV_MODELS[model_key]["model"], "state": state, "assets": assets,
             "questions": {"team": {"type": "choice", "instructions": q,
                                    "criteria": criteria}}}, sizes)


def jev_to_answer(probs: Optional[dict], roster: list) -> dict:
    """Jev option probabilities -> answer row fields (team, confidence)."""
    if not probs:
        return {"team": "unsure", "confidence": 0.0, "error": True}
    top = max(probs, key=lambda k: (probs[k], k))
    p = float(probs[top])
    if top in roster:
        team = top
    elif top in ("not_robot", "printed"):
        team = "no_robot"
    elif top == "other":
        team = "none"
    else:
        team = "unsure"
    return {"team": team, "confidence": round(p, 4)}


def fuse_views(per_view: dict, roster: list) -> dict:
    """Per-view probability dicts -> one answer: the view whose top option is
    most confident wins, unless two views confidently (>= 0.5) name different
    teams (then unsure)."""
    answers = {n: jev_to_answer(p, roster) for n, p in per_view.items() if p}
    if not answers:
        return {"team": "unsure", "confidence": 0.0, "error": True}
    teams = {a["team"] for a in answers.values() if a["team"] in roster and a["confidence"] >= 0.5}
    if len(teams) > 1:
        return {"team": "unsure", "confidence": 0.0, "conflict": sorted(teams)}
    team_ans = [a for a in answers.values() if a["team"] in roster]
    if team_ans:
        return dict(max(team_ans, key=lambda a: a["confidence"]))
    return dict(max(answers.values(), key=lambda a: a["confidence"]))


def cmd_jev(args) -> None:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from web import jev_reader as jr
    audit_dir = Path(args.audit)
    items = read_jsonl(audit_dir / "items.jsonl")
    if args.limit:
        items = items[:args.limit]
    session = jr.JevSession(jr.DEFAULT_MODEL, timeout_s=30.0, retries=1)
    if not session.available:
        raise SystemExit(f"Jev unavailable: {session.disabled_reason}")
    specs, owners = [], []
    for it in items:
        red, blue = it["roster"]["red"], it["roster"]["blue"]
        imgs = {n: (audit_dir / p).read_bytes() for n, p in it["images"].items()}
        if args.mode == "multi":
            specs.append(jev_multi_payload(jr.DEFAULT_MODEL, imgs, red, blue))
            owners.append((it["sample_id"], None))
        else:
            for n, jpg in imgs.items():
                specs.append(jr.build_jev_window_payload(jr.DEFAULT_MODEL, jpg, "wide", red, blue))
                owners.append((it["sample_id"], n))
    t0 = time.time()
    res = []
    for i in range(0, len(specs), 32):
        res += session.ask(specs[i:i + 32], "team")
    per: dict = defaultdict(dict)
    errors = 0
    for (sid, view), r in zip(owners, res):
        if "probs" not in r:
            errors += 1
        per[sid][view] = r.get("probs")
    out = []
    for it in items:
        sid = it["sample_id"]
        roster = it["roster"]["red"] + it["roster"]["blue"]
        p = per.get(sid, {})
        a = jev_to_answer(p.get(None), roster) if args.mode == "multi" else fuse_views(p, roster)
        row = {"sample_id": sid, **a, "judge": f"jev-{args.mode}"}
        row["probs"] = ({k: round(v, 4) for k, v in (p.get(None) or {}).items()}
                        if args.mode == "multi" else
                        {n: {k: round(v, 4) for k, v in (q or {}).items()} for n, q in p.items()})
        out.append(row)
    out_p = Path(args.out) if args.out else audit_dir / f"answers_jev_{args.mode}.jsonl"
    write_jsonl(out_p, out)
    st = session.stats()
    print(f"jev {args.mode}: {len(specs)} requests, {errors} errors, {time.time() - t0:.0f}s, "
          f"${float(st.get('cost_usd', 0.0)):.4f} -> {out_p}")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #

def gt_lookup(audit: dict, recs: list):
    """{sample_id: (gt team or None, nearest GT distance)} for a match with
    hand GT, else None. Pipeline positions are shifted SCORE_SHIFT_M away from
    the main camera like the GT scorer; GT positions are gt_scoring's."""
    import numpy as np
    from scripts.score_ground_truth import (REPLAYS_DIR, _probe_fps, _resolve, _shift_meta,
                                            replay_row)
    from web.field.ground_truth import GT_DIR, list_gt, video_fingerprint
    from web.field.gt_scoring import gt_position_at, gt_track, replay_from_meta
    mk = audit["match_key"]
    docs = [d for d in list_gt(mk, GT_DIR) if isinstance(d, dict) and d.get("team")
            and "keyframes" in d]
    if not docs:
        return None
    row = replay_row(audit["replay"])
    meta = json.loads((REPLAYS_DIR / row["id"] / "meta.json").read_text())
    shifted = _shift_meta(meta, mk.split("_")[0], mk, SCORE_SHIFT_M)
    video = _resolve(row.get("source_video_path"))
    rt = replay_from_meta(shifted, src_fps=_probe_fps(video) if video else None,
                          fingerprint=video_fingerprint(video) if video else None)
    tracks = [gt_track(d, rt) for d in docs]
    T = np.array([float(r["frame"]) for r in recs])
    gpos = {tr.team: gt_position_at(tr, T) for tr in tracks}
    out = {}
    for j, r in enumerate(recs):
        f = shifted["frames"][r["frame_idx"]]
        seen: Counter = Counter()
        pipe = {}
        for rr in f.get("robots") or []:
            pos = rr.get("field_pos")
            if not rr.get("slot") or not pos or pos[0] is None:
                continue
            seen[rr["slot"]] += 1
            k = rr["slot"] if seen[rr["slot"]] == 1 else f"{rr['slot']}#{seen[rr['slot']]}"
            pipe[k] = (float(pos[0]), float(pos[1]))
        g = {t: tuple(p[j]) for t, p in gpos.items() if not np.isnan(p[j][0])}
        out[r["sample_id"]] = gt_match(r["slot_key"], pipe, g)
    return out


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def cmd_score(args) -> None:
    audit_dir = Path(args.audit)
    audit = json.loads((audit_dir / "audit.json").read_text())
    samples = read_jsonl(audit_dir / "samples.jsonl")
    roster = set(audit["roster"]["red"]) | set(audit["roster"]["blue"])
    alliance_of = {t: a for a in ("red", "blue") for t in audit["roster"][a]}
    judgeable = [s for s in samples if s["views"]]
    w = sample_weights(judgeable, audit["strata"], audit["frame_period_s"])
    answers = {}
    for a in read_jsonl(Path(args.answers)):
        answers[a["sample_id"]] = parse_answer(a, roster)
    recs = [{**s, "w": w[s["sample_id"]], "pipe": s["pipeline_label"],
             "ans": answers.get(s["sample_id"])} for s in judgeable]
    lab = [r for r in recs if r["pipe"] is not None]
    unl = [r for r in recs if r["pipe"] is None]
    mc = args.min_conf
    print(f"audit {audit_dir.name}  replay {audit['replay'][:8]}  answers {Path(args.answers).name}"
          f"  min_conf {mc}")
    print(f"items {len(recs)} judgeable of {len(samples)} sampled; answered {len(answers)}; "
          f"labelled {len(lab)}, unlabelled {len(unl)}")
    kinds = Counter((r["ans"] or {}).get("kind", "missing") for r in recs)
    print("judge answers: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))

    est = misid_estimate(lab, mc)
    ci = bootstrap_ci(lab, mc)
    conf = [r for r in lab if r["ans"] and r["ans"]["kind"] == "team" and r["ans"]["conf"] >= mc]
    dis = [r for r in conf if r["ans"]["team"] != r["pipe"]]
    cross = sum(alliance_of.get(r["ans"]["team"]) != alliance_of.get(r["pipe"]) for r in dis)
    print(f"\nLABELLED  judge coverage {_pct(est['coverage'])} (time-weighted; "
          f"{est['n_confident']}/{est['n_judged']} items)  agreement "
          f"{_pct(None if est['misid'] is None else 1 - est['misid'])}")
    print(f"  estimated mis-ID {_pct(est['misid'])} "
          f"[95% CI {_pct(ci[0]) if ci else '-'} - {_pct(ci[1]) if ci else '-'}] "
          f"(unweighted {_pct(est['misid_unw'])}, {est['n_disagree']} disagreements: "
          f"{cross} cross-alliance / {len(dis) - cross} same)")
    if est["misid_s"] is not None:
        print(f"  = ~{est['misid_s']:.0f} slot-seconds mis-IDed of {est['labelled_s']:.0f} "
              f"labelled slot-seconds (coverage-extrapolated)")
    print(f"  judge says no robot at the spot: {_pct(est['no_robot_rate'])} of labelled time "
          f"({est['n_no_robot']} items)")
    for key, name in (("slot", "slot"), ("pipe", "pipeline team"), ("period", "period")):
        print(f"  by {name}: " + "  ".join(
            f"{k} {d}/{n} ({_pct(r)})" for k, (n, d, r) in breakdown(lab, key, mc).items()))
    if unl:
        ok = [r for r in unl if r["ans"]]
        nameable = [r for r in ok if r["ans"]["kind"] == "team" and r["ans"]["conf"] >= mc]
        nor = [r for r in ok if r["ans"]["kind"] == "no_robot" and r["ans"]["conf"] >= mc]
        tot = audit.get("unlabelled_slot_s") or 0
        lab_s = audit.get("labelled_slot_s") or 0
        share = ratio([(r["w"], r in nameable) for r in ok])
        print(f"\nUNLABELLED ({len(ok)} judged; {tot:.0f} of {tot + lab_s:.0f} slot-s unlabelled): "
              f"judge names a team on {_pct(share)}, "
              f"no robot on {_pct(ratio([(r['w'], r in nor) for r in ok]))}")
        if share is not None and tot + lab_s:
            print(f"  -> label coverage {_pct(lab_s / (tot + lab_s))} of slot-time; a further "
                  f"~{share * tot:.0f} slot-s show a robot the judge could name")

    if getattr(args, "json", None):
        tot = audit.get("unlabelled_slot_s") or 0
        lab_s = audit.get("labelled_slot_s") or 0
        Path(args.json).write_text(json.dumps({
            "audit": audit_dir.name, "replay": audit["replay"], "match_key": audit["match_key"],
            "meta": audit.get("meta"), "answers": str(args.answers), "min_conf": mc,
            "misid": est["misid"], "ci": list(ci) if ci else None, "misid_s": est["misid_s"],
            "judge_coverage": est["coverage"], "n_confident": est["n_confident"],
            "labelled_slot_s": lab_s, "unlabelled_slot_s": tot,
            "label_coverage": (lab_s / (tot + lab_s)) if tot + lab_s else None}, indent=1))
        print(f"score json -> {args.json}")

    gt = None if args.no_gt else gt_lookup(audit, recs)
    if gt is None:
        if args.dump:
            write_jsonl(Path(args.dump), [_dump_row(r, None) for r in recs])
        return
    for r in recs:
        r["gt"], r["gt_dist"] = gt.get(r["sample_id"], (None, None))
    glab = [r for r in lab if r["gt"]]
    print(f"\nGROUND TRUTH ({audit['match_key']}): {len(glab)}/{len(lab)} labelled items on a GT "
          f"robot within {GT_RADIUS_M} m (the rest: unannotated robot, phantom slot, or GT gap)")
    pipe_wrong = ratio([(r["w"], r["pipe"] != r["gt"]) for r in glab])
    print(f"  pipeline vs GT, all GT-matched labelled items: wrong {_pct(pipe_wrong)} "
          f"({sum(r['pipe'] != r['gt'] for r in glab)}/{len(glab)})")
    gconf = [r for r in glab if r["ans"] and r["ans"]["kind"] == "team" and r["ans"]["conf"] >= mc]
    if gconf:
        jr_ = sum(r["ans"]["team"] == r["gt"] for r in gconf)
        print(f"  judge vs GT (confident): right {jr_}/{len(gconf)} = {_pct(jr_ / len(gconf))}")
        a_est = ratio([(r["w"], r["ans"]["team"] != r["pipe"]) for r in gconf])
        g_est = ratio([(r["w"], r["pipe"] != r["gt"]) for r in gconf])
        print(f"  same items (GT-matched + judge-confident, n={len(gconf)}): audit mis-ID "
              f"{_pct(a_est)} vs GT mis-ID {_pct(g_est)}")
        fa = sum(r["ans"]["team"] != r["pipe"] and r["pipe"] == r["gt"] for r in gconf)
        miss = sum(r["ans"]["team"] == r["pipe"] and r["pipe"] != r["gt"] for r in gconf)
        both = sum(r["ans"]["team"] != r["pipe"] and r["pipe"] != r["gt"] for r in gconf)
        print(f"    judge flags a label GT says is right: {fa}; judge agrees with a label GT says "
              f"is wrong: {miss}; both say wrong: {both}")
        print(f"  coverage bias: GT mis-ID on judge-confident items {_pct(g_est)} vs on all "
              f"GT-matched items {_pct(pipe_wrong)}")
    gab = [r for r in glab if r["ans"] and r["ans"]["kind"] != "team"]
    if gab:
        print(f"  judge abstains on {len(gab)} GT-matched items: "
              + ", ".join(f"{k} {v}" for k, v in Counter(r["ans"]["kind"] for r in gab).items())
              + f"; pipeline wrong on {sum(r['pipe'] != r['gt'] for r in gab)} of them")
    nogt_np = [r for r in lab if not r["gt"] and r["ans"] and r["ans"]["kind"] == "no_robot"]
    print(f"  judge 'no_robot' on labelled items with no GT robot within {GT_RADIUS_M} m: "
          f"{len(nogt_np)} / {sum(1 for r in lab if not r['gt'])}")
    print("  calibration (judge team answers on GT-matched items, labelled + unlabelled):")
    print("    conf bin      n  right  P(right)")
    for lo, hi, n, k in calibration_table([r for r in recs if r["gt"]]):
        print(f"    [{lo:.2f},{min(hi, 1.0):.2f})  {n:4d}  {k:5d}  {_pct(k / n) if n else '-'}")
    if args.dump:
        write_jsonl(Path(args.dump), [_dump_row(r, r.get("gt")) for r in recs])
        print(f"per-item rows -> {args.dump}")


def _dump_row(r: dict, gt) -> dict:
    return {"sample_id": r["sample_id"], "frame": r["frame"], "t_s": r["t_s"], "slot": r["slot"],
            "period": r["period"], "views": r["views"], "pipe": r["pipe"], "gt": gt,
            "gt_dist": None if r.get("gt_dist") is None else round(r["gt_dist"], 2),
            "judge": (r["ans"] or {}).get("team") or (r["ans"] or {}).get("kind"),
            "conf": (r["ans"] or {}).get("conf"), "w": round(r["w"], 3)}


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw a sample and cut the item crops")
    s.add_argument("--replay", required=True, help="replay id or unique prefix")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--unlabelled-frac", type=float, default=UNLABELLED_FRAC)
    s.add_argument("--auto-min-share", type=float, default=AUTO_MIN_SHARE)
    s.add_argument("--ground-center-offset", type=float, default=0.0,
                   help="the job's ground_center_offset_m (0 unless the job set it)")
    s.add_argument("--out", default=None, help="audit dir (default data/eval/audit/<mk>_<id8>)")
    s.add_argument("--meta", default=None,
                   help="take the labels from this meta.json (e.g. an event_relabel output); "
                        "positions, video and views stay the replay's")
    j = sub.add_parser("jev", help="Jev as the judge (calls the Jev API)")
    j.add_argument("--audit", required=True)
    j.add_argument("--mode", choices=("multi", "views"), default="multi",
                   help="multi: one request with every view; views: one request per view, fused")
    j.add_argument("--out", default=None)
    j.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("score", help="score a judge answers file")
    c.add_argument("--audit", required=True)
    c.add_argument("--answers", required=True)
    c.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)
    c.add_argument("--no-gt", action="store_true")
    c.add_argument("--dump", default=None, help="write per-item rows (JSONL) here")
    c.add_argument("--json", default=None, help="write the estimates (JSON) here")
    args = ap.parse_args()
    {"sample": cmd_sample, "jev": cmd_jev, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()

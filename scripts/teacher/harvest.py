#!/usr/bin/env python3
"""P1 crop harvester for Claude-as-teacher distillation (design §3).

Per match: TBA roster -> video (local test_footage, prior download, or YouTube
download) -> sequential decode at --det-fps (12) -> bumper detection on every
sample (on one stretched field when the video is woven), samples with < 2
bumpers skipped (pre/post-match, crowd) -> whole-robot detection on kept
samples -> per detector, same-class duplicate / nested-box suppression and a
ByteTrack-style tracker (two-stage high / low conf association, constant-
velocity prediction, centre-distance fallback for small boxes, 1 s lost
buffer, no length cap) -> crops every >= 0.5 s per tracklet, near-duplicates of
the previous kept crop skipped, <= 6 spread in time -> bumper->robot link by
containment -> second decode cutting only the chosen crops -> deinterlaced JPEG
crops + manifest.jsonl + contact sheet + summary.json (tracklet stats).

v2 (Nov 10): v1 = 2 fps greedy IoU tracklets <= 3 s, bumper tracklets ~2 dets,
57-72% singletons. No calibration needed, so any season works.

Usage (from the repo or a worktree; data lands in the MAIN checkout):
  .venv/bin/python -m scripts.teacher.harvest --event 2025cur --matches qm1-qm15
  .venv/bin/python -m scripts.teacher.harvest --event 2026hop          # default scope
  .venv/bin/python -m scripts.teacher.harvest --video data/test_footage/2026cur_qm22.mp4 \\
      --match-key 2026cur_qm22
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# main-checkout resolution + the bumper detector path are shared with the runtime S1 reader
from web.students.paths import BUMPER_WEIGHTS, MAIN, main_checkout  # noqa: E402,F401

ROBOT_WEIGHTS = MAIN / "runs/train/hub_zone_yolo26s_2026cur_v2/weights/best.pt"
ROBOT_CLASS = 1                     # hub_zone model: 0 fuel_ball, 1 robot, 2 hub_opening
DEFAULT_OUT_ROOT = MAIN / "data/teacher/harvest_v2"   # v1 (2 fps greedy) stays in harvest/
VIDEO_ROOT = MAIN / "data/prior_years"
LOCAL_FOOTAGE = MAIN / "data/test_footage"
YTDLP_PYTHON = Path(os.environ.get(
    "TEACHER_YTDLP_PYTHON",
    "/tmp/claude-1000/-home-nick-AI-frc-vision-scorer/"
    "44602460-a583-4ab7-becb-4686b1d2276a/scratchpad/ytvenv/bin/python"))

BUMPER_CONF = 0.40
ROBOT_CONF = 0.35
MIN_BUMPERS_PER_FRAME = 2
IOU_THR = 0.3
MAX_GAP = 1                         # samples a tracklet may skip (v1 greedy tracker)
MAX_LEN_S = None                    # v1 capped greedy tracklets at 3 s; v2 has no cap
DET_FPS = 12.0                      # v2 detection / association rate (6 fps left 2026cur
                                    # bumper singletons at 34-38%: fast robots in the close
                                    # corner views move > a box width between samples)
NEW_TRACK_MARGIN = 0.10             # a track STARTS only at crop conf + this (0.4-0.5 one-shot
                                    # fires were mostly partial / field-structure boxes)
TRACK_LOW = 0.10                    # detector conf floor fed to the tracker's 2nd stage
DUP_IOU = 0.7                       # same-class same-frame duplicate suppression
NESTED_IOA = 0.8                    # ... and boxes >= 80% inside a higher-conf same-class box
TRACK_BUFFER_S = 1.0                # lost tracks re-findable for this long
CROP_MIN_DT = 0.5                   # kept crops of a tracklet >= 0.5 s apart
CROP_MIN_MOVE = 0.10                # ... and not near-duplicates (moved < 10% width
CROP_MIN_SCALE = 0.10               #     AND size changed < 10%)
CROPS_PER_TRACKLET = 6
MIN_SIDE_PX = 20
PAD_BUMPER = 0.25
PAD_ROBOT = 0.10
JPEG_Q = 92

MANIFEST_KEYS = (
    "event", "year", "match_key", "frame", "t_sec", "video_fps", "detector", "cls",
    "conf", "box", "px_w", "px_h", "tracklet_id", "linked_tracklet_id",
    "alliance_colour", "comb_ratio_before", "deinterlaced", "path", "roster",
)


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested)
# --------------------------------------------------------------------------- #

def parse_matches(spec: str) -> List[str]:
    """'qm1-qm15' | 'qm1-15' | 'qm3,qm7' | 'qm1-qm3,qm9' -> ['qm1', ...]."""
    out: List[str] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        m = re.fullmatch(r"([a-z]+)(\d+)\s*-\s*(?:([a-z]+))?(\d+)", part)
        if m:
            lvl, a, lvl2, b = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            if lvl2 and lvl2 != lvl:
                raise ValueError(f"mixed comp levels in range {part!r}")
            if b < a:
                raise ValueError(f"descending range {part!r}")
            out.extend(f"{lvl}{i}" for i in range(a, b + 1))
            continue
        if re.fullmatch(r"[a-z]+\d+(m\d+)?", part):
            out.append(part)
            continue
        raise ValueError(f"bad match spec {part!r}")
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Det:
    sample: int          # index in the sampled sequence (skipped frames count)
    frame: int           # video frame index
    t: float             # seconds
    box: Tuple[float, float, float, float]
    conf: float
    cls: str
    det_id: int = -1     # index into the per-match detection store (crop pixels)


@dataclass
class Tracklet:
    tid: str
    detector: str
    dets: List[Det] = field(default_factory=list)

    @property
    def cls(self) -> str:
        counts: Dict[str, int] = {}
        for d in self.dets:
            counts[d.cls] = counts.get(d.cls, 0) + 1
        return max(sorted(counts), key=lambda c: counts[c])


def build_tracklets(dets: Iterable[Det], detector: str, prefix: str,
                    iou_thr: float = IOU_THR, max_gap: int = MAX_GAP,
                    max_len_s: Optional[float] = MAX_LEN_S, same_cls: bool = True) -> List[Tracklet]:
    """v1 greedy frame-to-frame IoU tracklets (kept for ``--tracker greedy`` comparisons). A tracklet extends to a detection in
    a later sample if at most ``max_gap`` samples were skipped, IoU with its last
    box >= ``iou_thr`` (and same class when ``same_cls``), and the result spans
    <= ``max_len_s``; pairs are taken highest IoU first. Unmatched detections
    start new tracklets."""
    by_sample: Dict[int, List[Det]] = {}
    for d in dets:
        by_sample.setdefault(d.sample, []).append(d)
    tracks: List[Tracklet] = []
    active: List[Tracklet] = []
    for s in sorted(by_sample):
        cur = by_sample[s]
        active = [tr for tr in active if s - tr.dets[-1].sample <= max_gap + 1]
        pairs = []
        for ti, tr in enumerate(active):
            last = tr.dets[-1]
            for di, d in enumerate(cur):
                if same_cls and d.cls != last.cls:
                    continue
                if max_len_s is not None and d.t - tr.dets[0].t > max_len_s + 1e-9:
                    continue
                v = iou(last.box, d.box)
                if v >= iou_thr:
                    pairs.append((v, ti, di))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        used_t, used_d = set(), set()
        for v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            active[ti].dets.append(cur[di])
        for di, d in enumerate(cur):
            if di not in used_d:
                tr = Tracklet(tid=f"{prefix}{len(tracks)}", detector=detector, dets=[d])
                tracks.append(tr)
                active.append(tr)
    return tracks


def ioa(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of box ``a``'s area inside box ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def suppress_duplicates(dets: Sequence[Det], iou_thr: float = DUP_IOU,
                        nested_ioa: Optional[float] = NESTED_IOA) -> List[Det]:
    """Drop same-class boxes overlapping a higher-confidence box of that class
    in the same frame by IoU > ``iou_thr``, or lying inside it (>= ``nested_ioa``
    of their area; partial sub-boxes of one bumper). Duplicate fires would
    otherwise start their own one-detection tracklets."""
    keep: List[Det] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(k.cls != d.cls or (iou(k.box, d.box) <= iou_thr and
                                  (nested_ioa is None or ioa(d.box, k.box) < nested_ioa))
               for k in keep):
            keep.append(d)
    return keep


@dataclass
class TrackParams:
    """ByteTrack-style association, all in SAMPLE units (``det_fps``)."""
    high: float = 0.40          # first association (= crop conf for the detector)
    new: Optional[float] = None  # min conf to START a track (None = high)
    low: float = TRACK_LOW      # second association floor (recovers flickery low-conf fires)
    buffer: int = 12            # samples a lost track may be re-found after (1 s at 12 fps)
    match_iou: float = 0.2      # 1st stage: accept cost <= 1 - match_iou
    low_iou: float = 0.5        # 2nd stage (low-conf dets, tracked-only)
    centre_fallback: float = 0.75   # zero/low-IoU match if centre within this x box width
    size_ratio: float = 1.6     # ... and w, h within this factor
    cls_penalty: float = 0.15   # cost added when classes differ (class-agnostic otherwise)
    max_pred_dt: int = 2        # constant-velocity prediction capped at this many samples


class _Track:
    __slots__ = ("dets", "vel", "last", "hits")

    def __init__(self, d: Det):
        self.dets: List[Det] = [d]
        self.vel = (0.0, 0.0)
        self.last = d.sample
        self.hits = 1

    def predict(self, sample: int, max_dt: int) -> Tuple[float, float, float, float]:
        b = self.dets[-1].box
        dt = min(sample - self.last, max_dt)
        dx, dy = self.vel[0] * dt, self.vel[1] * dt
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    def add(self, d: Det) -> None:
        p = self.dets[-1]
        dt = max(1, d.sample - p.sample)
        v = (((d.box[0] + d.box[2]) - (p.box[0] + p.box[2])) / 2 / dt,
             ((d.box[1] + d.box[3]) - (p.box[1] + p.box[3])) / 2 / dt)
        self.vel = v if self.hits == 1 else (0.5 * self.vel[0] + 0.5 * v[0], 0.5 * self.vel[1] + 0.5 * v[1])
        self.dets.append(d)
        self.last = d.sample
        self.hits += 1


def assoc_cost(pred: Sequence[float], det_box: Sequence[float], same_cls: bool,
               p: TrackParams) -> float:
    """1 - IoU against the predicted box; for small / fast boxes with little
    overlap, a centre-distance fallback in [0.55, 1 - match_iou] when the centre
    is within ``centre_fallback`` x box width and the sizes agree."""
    c = 1.0 - iou(pred, det_box)
    pw, ph = box_wh(pred)
    dw, dh = box_wh(det_box)
    if p.centre_fallback > 0 and min(pw, ph, dw, dh) > 0:
        ok_size = max(pw / dw, dw / pw, ph / dh, dh / ph) <= p.size_ratio
        dist = np.hypot((pred[0] + pred[2] - det_box[0] - det_box[2]) / 2,
                        (pred[1] + pred[3] - det_box[1] - det_box[3]) / 2)
        lim = p.centre_fallback * max(pw, dw)
        if ok_size and dist <= lim:
            c = min(c, 0.55 + (1.0 - p.match_iou - 0.55) * dist / lim)
    return c if same_cls else c + p.cls_penalty


def _hungarian(cost: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment
    if cost.size == 0:
        return []
    big = 1e6
    c = np.where(cost <= thr, cost, big)
    rows, cols = linear_sum_assignment(c)
    return [(int(r), int(k)) for r, k in zip(rows, cols) if c[r, k] < big]


def bytetrack(samples: Sequence[Tuple[int, List[Det]]], detector: str, prefix: str,
              params: Optional[TrackParams] = None, dedup_iou: Optional[float] = DUP_IOU,
              nested_ioa: Optional[float] = NESTED_IOA) -> List[Tracklet]:
    """ByteTrack-style offline association over sampled frames.

    ``samples`` = (sample index, detections) in order, INCLUDING samples with no
    detections (they age tracks). Per sample: same-class duplicates suppressed;
    stage 1 = high-conf dets vs every live track (tracked or lost <= ``buffer``
    samples) on IoU against a constant-velocity prediction, with a centre-distance
    fallback for small boxes; stage 2 = low-conf dets vs tracks still unmatched
    that were seen last sample, plain IoU; unmatched high-conf dets start tracks;
    unmatched low-conf dets are dropped. A track seen once and missed on the next
    sample is closed (one-shot fires don't linger). Tracklet ids are assigned in
    order of first appearance."""
    p = params or TrackParams()
    live: List[_Track] = []
    done: List[_Track] = []
    for s, raw in samples:
        dets = suppress_duplicates(raw, dedup_iou, nested_ioa) if dedup_iou is not None else list(raw)
        keep = []
        for t in live:
            gap = s - t.last
            if gap > p.buffer + 1 or (t.hits == 1 and gap > 2):
                done.append(t)
            else:
                keep.append(t)
        live = keep
        high = [d for d in dets if d.conf >= p.high]
        low = [d for d in dets if p.low <= d.conf < p.high]
        preds = [t.predict(s, p.max_pred_dt) for t in live]
        cost = np.array([[assoc_cost(pr, d.box, t.dets[-1].cls == d.cls, p) for d in high]
                         for t, pr in zip(live, preds)], dtype=np.float64).reshape(len(live), len(high))
        m1 = _hungarian(cost, 1.0 - p.match_iou)
        used_t = {ti for ti, _ in m1}
        used_d = {di for _, di in m1}
        for ti, di in m1:
            live[ti].add(high[di])
        rem = [ti for ti in range(len(live)) if ti not in used_t and live[ti].last == s - 1]
        if rem and low:
            cost2 = np.array([[1.0 - iou(preds[ti], d.box) + (0 if live[ti].dets[-1].cls == d.cls
                                                              else p.cls_penalty) for d in low]
                              for ti in rem], dtype=np.float64)
            for ri, di in _hungarian(cost2, 1.0 - p.low_iou):
                live[rem[ri]].add(low[di])
        new_thr = p.high if p.new is None else p.new
        for di, d in enumerate(high):
            if di not in used_d and d.conf >= new_thr:
                live.append(_Track(d))
    done.extend(live)
    done.sort(key=lambda t: (t.dets[0].sample, t.dets[0].box[0], t.dets[0].box[1]))
    return [Tracklet(tid=f"{prefix}{i}", detector=detector, dets=t.dets) for i, t in enumerate(done)]


def tracklet_stats(tracks: Sequence[Tracklet]) -> dict:
    n = [len(t.dets) for t in tracks]
    if not n:
        return {"tracklets": 0}
    dur = [t.dets[-1].t - t.dets[0].t for t in tracks]
    return {"tracklets": len(n), "dets": int(sum(n)), "mean_dets": round(float(np.mean(n)), 2),
            "median_dets": float(np.median(n)), "singleton_frac": round(sum(x == 1 for x in n) / len(n), 3),
            "ge5_frac": round(sum(x >= 5 for x in n) / len(n), 3),
            "dets_in_ge5_frac": round(sum(x for x in n if x >= 5) / sum(n), 3),
            "mean_dur_s": round(float(np.mean(dur)), 2), "max_dur_s": round(float(max(dur)), 1)}


def select_crops_v2(tr: Tracklet, k: int = CROPS_PER_TRACKLET, min_side: float = MIN_SIDE_PX,
                    min_conf: float = 0.0, min_dt: float = CROP_MIN_DT,
                    min_move: float = CROP_MIN_MOVE, min_scale: float = CROP_MIN_SCALE) -> List[Det]:
    """Crop candidates walked in time: a detection (short side >= ``min_side``,
    conf >= ``min_conf``) is a candidate if >= ``min_dt`` s after the previous
    candidate and NOT a near-duplicate of it (centre moved < ``min_move`` x box
    width AND width and height changed < ``min_scale``). Then <= ``k`` spread in time."""
    cands: List[Det] = []
    for d in sorted(tr.dets, key=lambda d: d.t):
        w, h = box_wh(d.box)
        if min(w, h) < min_side or d.conf < min_conf:
            continue
        if cands:
            q = cands[-1]
            if d.t - q.t < min_dt - 1e-9:
                continue
            qw, qh = box_wh(q.box)
            move = np.hypot((d.box[0] + d.box[2] - q.box[0] - q.box[2]) / 2,
                            (d.box[1] + d.box[3] - q.box[1] - q.box[3]) / 2) / max(qw, 1e-6)
            scale = max(abs(w - qw) / max(qw, 1e-6), abs(h - qh) / max(qh, 1e-6))
            if move < min_move and scale < min_scale:
                continue
        cands.append(d)
    return [cands[i] for i in spread_pick(len(cands), k)]


def spread_pick(n: int, k: int) -> List[int]:
    """``min(n, k)`` indices into ``range(n)`` spread evenly (ends included)."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


def box_wh(box: Sequence[float]) -> Tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def select_crops(tr: Tracklet, k: int = CROPS_PER_TRACKLET,
                 min_side: float = MIN_SIDE_PX) -> List[Det]:
    """Up to ``k`` detections of the tracklet, spread evenly in time, skipping
    boxes whose short side is < ``min_side`` px."""
    ok = [d for d in sorted(tr.dets, key=lambda d: d.t) if min(box_wh(d.box)) >= min_side]
    return [ok[i] for i in spread_pick(len(ok), k)]


def bumper_in_robot(bbox: Sequence[float], rbox: Sequence[float], lower_frac: float = 0.6) -> bool:
    """Bumper box centre inside the robot box's lower ``lower_frac``."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    top = rbox[3] - lower_frac * (rbox[3] - rbox[1])
    return rbox[0] <= cx <= rbox[2] and top <= cy <= rbox[3]


def link_bumpers_to_robots(bumpers: Sequence[Tracklet], robots: Sequence[Tracklet]
                           ) -> Dict[str, Optional[str]]:
    """bumper tid -> robot tid whose box contains the bumper centre (lower 60%)
    on the most shared samples, or None. Ties -> earlier robot tracklet."""
    robot_at: Dict[int, List[Tuple[int, Det]]] = {}
    for ri, rt in enumerate(robots):
        for d in rt.dets:
            robot_at.setdefault(d.sample, []).append((ri, d))
    out: Dict[str, Optional[str]] = {}
    for bt in bumpers:
        counts: Dict[int, int] = {}
        for bd in bt.dets:
            hit = set()
            for ri, rd in robot_at.get(bd.sample, []):
                if ri not in hit and bumper_in_robot(bd.box, rd.box):
                    hit.add(ri)
                    counts[ri] = counts.get(ri, 0) + 1
        if counts:
            best = max(sorted(counts), key=lambda ri: counts[ri])
            out[bt.tid] = robots[best].tid
        else:
            out[bt.tid] = None
    return out


def colour_of(cls_name: str) -> Optional[str]:
    c = cls_name.lower()
    return "red" if c.startswith("red") else "blue" if c.startswith("blue") else None


def robot_links(bumpers: Sequence[Tracklet], b2r: Dict[str, Optional[str]]
                ) -> Dict[str, Tuple[Optional[str], Optional[str], List[str]]]:
    """robot tid -> (best bumper tid, majority alliance colour, all linked bumper
    tids). Best = the linked bumper tracklet with most detections."""
    by_robot: Dict[str, List[Tracklet]] = {}
    for bt in bumpers:
        r = b2r.get(bt.tid)
        if r is not None:
            by_robot.setdefault(r, []).append(bt)
    out = {}
    for r, bts in by_robot.items():
        best = max(bts, key=lambda b: len(b.dets))  # first wins ties
        votes: Dict[str, int] = {}
        for b in bts:
            col = colour_of(b.cls)
            if col:
                votes[col] = votes.get(col, 0) + len(b.dets)
        colour = None
        if votes:
            top = max(votes.values())
            winners = [c for c, v in votes.items() if v == top]
            colour = winners[0] if len(winners) == 1 else None
        out[r] = (best.tid, colour, [b.tid for b in bts])
    return out


def pad_box(box: Sequence[float], frac: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw, bh = box_wh(box)
    x0 = int(max(0, np.floor(box[0] - frac * bw)))
    y0 = int(max(0, np.floor(box[1] - frac * bh)))
    x1 = int(min(w, np.ceil(box[2] + frac * bw)))
    y1 = int(min(h, np.ceil(box[3] + frac * bh)))
    return x0, y0, x1, y1


def parse_roster(match: Optional[dict]) -> Optional[Dict[str, List[str]]]:
    if not match:
        return None
    al = match.get("alliances") or {}
    return {c: [k[3:] if k.startswith("frc") else k for k in (al.get(c) or {}).get("team_keys", [])]
            for c in ("red", "blue")}


def manifest_row(*, event: str, match_key: str, det: Det, video_fps: float, detector: str,
                 tracklet_id: str, linked_tracklet_id: Optional[str],
                 alliance_colour: Optional[str], comb_ratio_before: float, deinterlaced: bool,
                 path: str, roster: Optional[Dict[str, List[str]]], **extra) -> dict:
    w, h = box_wh(det.box)
    row = {
        "event": event,
        "year": int(event[:4]),
        "match_key": match_key,
        "frame": int(det.frame),
        "t_sec": round(float(det.t), 3),
        "video_fps": round(float(video_fps), 3),
        "detector": detector,
        "cls": det.cls,
        "conf": round(float(det.conf), 4),
        "box": [round(float(v), 1) for v in det.box],
        "px_w": int(round(w)),
        "px_h": int(round(h)),
        "tracklet_id": tracklet_id,
        "linked_tracklet_id": linked_tracklet_id,
        "alliance_colour": alliance_colour,
        "comb_ratio_before": round(float(comb_ratio_before), 4),
        "deinterlaced": bool(deinterlaced),
        "path": path,
        "roster": roster,
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# I/O: TBA, video, detection, writing
# --------------------------------------------------------------------------- #

def load_env() -> None:
    try:
        from dotenv import load_dotenv
        for p in (MAIN / ".env", REPO / ".env"):
            if p.exists():
                load_dotenv(p, override=False)
    except ImportError:
        pass


def tba_match(match_key: str) -> Optional[dict]:
    load_env()
    from web import tba
    try:
        return asyncio.run(tba.get_match(match_key))
    except Exception as e:  # network / 404 -> roster unknown
        print(f"[harvest] TBA lookup failed for {match_key}: {type(e).__name__}", flush=True)
        return None


def youtube_keys(match: Optional[dict]) -> List[str]:
    return [v["key"] for v in (match or {}).get("videos") or [] if v.get("type") == "youtube" and v.get("key")]


def acquire_video(match_key: str, event: str, match: Optional[dict],
                  video_root: Path = VIDEO_ROOT) -> Tuple[Optional[Path], bool]:
    """(path, downloaded_now). Prefers local test_footage, then an earlier
    download, then YouTube via the scratch yt-dlp venv."""
    for p in (LOCAL_FOOTAGE / f"{match_key}.mp4", video_root / event / f"{match_key}.mp4"):
        if p.exists() and p.stat().st_size > 0:
            return p, False
    keys = youtube_keys(match)
    if not keys:
        print(f"[harvest] {match_key}: no YouTube video on TBA", flush=True)
        return None, False
    if not YTDLP_PYTHON.exists():
        print(f"[harvest] yt-dlp venv missing at {YTDLP_PYTHON}", flush=True)
        return None, False
    out = video_root / event / f"{match_key}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    for key in keys:
        cmd = [str(YTDLP_PYTHON), "-I", str(REPO / "scripts/download_match.py"),
               "--url", f"https://youtu.be/{key}", "--out", str(out)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print(f"[harvest] {match_key}: download timed out ({key})", flush=True)
            continue
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out, True
        print(f"[harvest] {match_key}: download failed ({key}): {r.stderr.strip()[-300:]}", flush=True)
    return None, False


class Detectors:
    """Bumper + whole-robot YOLO models, loaded on first use (a run whose
    detections all come from --det-cache never touches the GPU)."""

    def __init__(self, imgsz: int = 640):
        self.imgsz = imgsz
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from ultralytics import YOLO
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.bumper = YOLO(str(BUMPER_WEIGHTS))
        self.robot = YOLO(str(ROBOT_WEIGHTS))
        self.bnames = self.bumper.names
        self.rnames = self.robot.names
        self._loaded = True

    def _run(self, model, frames, conf, classes=None):
        res = model(frames, conf=conf, imgsz=self.imgsz, device=self.device,
                    classes=classes, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            out.append(list(zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy().tolist(),
                                b.cls.cpu().numpy().astype(int).tolist())))
        return out

    def bumpers(self, frames, conf: float = BUMPER_CONF):
        self._load()
        return self._run(self.bumper, frames, conf)

    def robots(self, frames, conf: float = ROBOT_CONF):
        self._load()
        return self._run(self.robot, frames, conf, classes=[ROBOT_CLASS])


def iter_samples(video: Path, fps_out: float, limit: Optional[int] = None):
    """Yield (sample_idx, frame_idx, t_sec, frame) at ``fps_out`` by sequential
    decode (grab every frame, retrieve only sampled ones)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = fps / fps_out
    next_t, fi, si = 0.0, 0, 0
    try:
        while True:
            if not cap.grab():
                break
            if fi >= next_t - 1e-6:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield si, fi, fi / fps, frame
                si += 1
                next_t = si * step
                if limit is not None and si >= limit:
                    break
            fi += 1
    finally:
        cap.release()


def parse_idet(text: str) -> Dict[str, int]:
    """Counts from ffmpeg idet's 'Multi frame detection' line (last one wins)."""
    out = {"tff": 0, "bff": 0, "progressive": 0, "undetermined": 0}
    for line in text.splitlines():
        m = re.search(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
                      r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)", line)
        if m:
            out = dict(zip(("tff", "bff", "progressive", "undetermined"), map(int, m.groups())))
    return out


def interlaced_from_idet(counts: Dict[str, int], min_frac: float = 0.3) -> bool:
    n = sum(counts.values())
    return n > 0 and (counts["tff"] + counts["bff"]) / n >= min_frac


def detect_interlaced(video: Path, duration_s: float, n_frames: int = 300
                      ) -> Tuple[Optional[bool], Dict[str, int]]:
    """Video-level interlace check with ffmpeg ``idet`` on ``n_frames`` from a
    third of the way in. Crop-level comb_ratio alone misfires on progressive
    video (fences / thin horizontal edges score > 0.9 on ~22% of 2025 crops),
    so crops are deinterlaced only when the VIDEO is woven. None = unknown."""
    if not shutil.which("ffmpeg"):
        return None, {}
    ss = max(0.0, duration_s / 3)
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-ss", f"{ss:.1f}", "-i", str(video),
                            "-frames:v", str(n_frames), "-vf", "idet", "-an", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=300)
    except Exception:
        return None, {}
    counts = parse_idet(r.stderr)
    if not sum(counts.values()):
        return None, counts
    return interlaced_from_idet(counts), counts


def video_fps(video: Path) -> Tuple[float, int, int, int]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = (cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
           int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return out


def contact_sheet(tiles: List[Tuple[np.ndarray, str]], cols: int = 8, tw: int = 160,
                  th: int = 180, title: Optional[str] = None) -> np.ndarray:
    import cv2
    cells = []
    for img, lab in tiles:
        s = min(tw / max(1, img.shape[1]), (th - 20) / max(1, img.shape[0]))
        t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        c = np.zeros((th, tw, 3), np.uint8)
        c[:t.shape[0], :t.shape[1]] = t
        cv2.putText(c, lab, (2, th - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(c)
    while not cells or len(cells) % cols:
        cells.append(np.zeros((th, tw, 3), np.uint8))
    sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    if title:
        bar = np.zeros((24, sheet.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        sheet = np.vstack([bar, sheet])
    return sheet


def field_frame(frame: np.ndarray) -> np.ndarray:
    """One field (even rows) stretched back to full height: removes interlace
    combing on movers before DETECTION (at imgsz 640 the detector sees ~360
    rows of a 1080p frame, so dropping to a 540-row field costs ~nothing)."""
    import cv2
    h, w = frame.shape[:2]
    return cv2.resize(frame[::2], (w, h), interpolation=cv2.INTER_LINEAR)


def detect_pass(video: Path, detectors: Detectors, fps_out: float, limit_frames: Optional[int],
                batch: int, conf_floor: float, field_detect: bool = False) -> Tuple[List[Tuple[int, int, float]], Dict[int, List[Det]],
                                                        Dict[int, List[Det]], set]:
    """Sequential decode at ``fps_out``; bumper detection on every sample, robot
    detection only on kept samples (>= MIN_BUMPERS_PER_FRAME bumpers at
    BUMPER_CONF). Detections down to ``conf_floor`` are returned (the tracker's
    low stage uses them; crops only come from >= the detector's crop conf).
    ``field_detect``: detect on ``field_frame`` (woven video); boxes stay in
    full-frame coordinates and crops are still cut from the original frame.
    Returns (samples [(si, fi, t)], bumper dets by si, robot dets by si, kept si)."""
    samples: List[Tuple[int, int, float]] = []
    bd: Dict[int, List[Det]] = {}
    rd: Dict[int, List[Det]] = {}
    kept: set = set()

    def flush(buf):
        if not buf:
            return
        bres = detectors.bumpers([f for _, _, _, f in buf], conf=min(conf_floor, BUMPER_CONF))
        keep = []
        for (si, fi, t, frame), bb in zip(buf, bres):
            if sum(c >= BUMPER_CONF for _, c, _ in bb) >= MIN_BUMPERS_PER_FRAME:
                keep.append((si, fi, t, frame))
                kept.add(si)
                bd[si] = [Det(si, fi, t, tuple(box), conf, detectors.bnames[c]) for box, conf, c in bb]
        if not keep:
            return
        rres = detectors.robots([k[3] for k in keep], conf=min(conf_floor, ROBOT_CONF))
        for (si, fi, t, _), rr in zip(keep, rres):
            rd[si] = [Det(si, fi, t, tuple(box), conf, detectors.rnames[c]) for box, conf, c in rr]

    buf = []
    for si, fi, t, frame in iter_samples(video, fps_out, limit_frames):
        samples.append((si, fi, t))
        buf.append((si, fi, t, field_frame(frame) if field_detect else frame))
        if len(buf) >= batch:
            flush(buf)
            buf = []
    flush(buf)
    return samples, bd, rd, kept


def crop_pass(video: Path, wanted: Dict[int, List[Tuple[int, Tuple[float, ...], float]]]
              ) -> Dict[int, np.ndarray]:
    """Second sequential decode cutting only the chosen crops.
    ``wanted``: frame index -> [(crop id, box, pad frac)] -> {crop id: raw crop}."""
    import cv2
    out: Dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(video))
    last = max(wanted)
    fi = 0
    try:
        while fi <= last and cap.grab():
            if fi in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    for cid, box, pad in wanted[fi]:
                        x0, y0, x1, y1 = pad_box(box, pad, frame.shape[1], frame.shape[0])
                        if x1 > x0 and y1 > y0:
                            out[cid] = frame[y0:y1, x0:x1].copy()
            fi += 1
    finally:
        cap.release()
    return out


def harvest_match(*, match_key: str, video: Path, out_root: Path, detectors: Detectors,
                  roster: Optional[Dict[str, List[str]]], fps_out: float = DET_FPS,
                  limit_frames: Optional[int] = None, batch: int = 8,
                  deinterlace: str = "auto", tracker: str = "bytetrack",
                  det_cache: Optional[Path] = None, det_deinterlace: str = "auto") -> dict:
    import cv2
    import pickle
    from web.bumper_register import comb_ratio, deinterlace_crop

    event = match_key.split("_", 1)[0]
    t0 = time.time()
    vfps, nframes, W, H = video_fps(video)
    interlaced, idet = detect_interlaced(video, nframes / vfps if vfps else 0.0)
    field_detect = {"always": True, "never": False}.get(det_deinterlace, bool(interlaced))
    cache = (det_cache / f"{match_key}_{fps_out:g}fps{'_fd' if field_detect else ''}.pkl"
             if det_cache else None)
    if cache is not None and cache.exists():
        samples, bd, rd, kept = pickle.loads(cache.read_bytes())
    else:
        samples, bd, rd, kept = detect_pass(video, detectors, fps_out, limit_frames, batch,
                                            TRACK_LOW if tracker == "bytetrack" else BUMPER_CONF,
                                            field_detect=field_detect)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps((samples, bd, rd, kept)))
    t_detect = time.time() - t0

    t1 = time.time()
    buffer = max(1, int(round(TRACK_BUFFER_S * fps_out)))
    if tracker == "bytetrack":
        # Every sample is fed (skipped / empty ones age the tracks).
        btracks = bytetrack([(si, bd.get(si, [])) for si, _, _ in samples], "bumper", "b",
                            TrackParams(high=BUMPER_CONF, new=BUMPER_CONF + NEW_TRACK_MARGIN,
                                        buffer=buffer))
        rtracks = bytetrack([(si, rd.get(si, [])) for si, _, _ in samples], "robot", "r",
                            TrackParams(high=ROBOT_CONF, new=ROBOT_CONF + NEW_TRACK_MARGIN,
                                        buffer=buffer))
    else:  # v1 reproduction (comparison only): greedy IoU tracklets capped at 3 s
        btracks = build_tracklets([d for si in sorted(bd) for d in bd[si] if d.conf >= BUMPER_CONF],
                                  "bumper", "b", max_len_s=3.0)
        rtracks = build_tracklets([d for si in sorted(rd) for d in rd[si] if d.conf >= ROBOT_CONF],
                                  "robot", "r", max_len_s=3.0)
    b2r = link_bumpers_to_robots(btracks, rtracks)
    rlinks = robot_links(btracks, b2r)
    t_track = time.time() - t1

    picks_by_tr: Dict[str, List[Det]] = {}
    wanted: Dict[int, List[Tuple[int, Tuple[float, ...], float]]] = {}
    n_wanted = 0
    for tracks, detector, pad, cconf in ((btracks, "bumper", PAD_BUMPER, BUMPER_CONF),
                                         (rtracks, "robot", PAD_ROBOT, ROBOT_CONF)):
        for tr in tracks:
            picks = (select_crops_v2(tr, min_conf=cconf) if tracker == "bytetrack"
                     else select_crops(tr))
            picks_by_tr[tr.tid] = picks
            for d in picks:
                d.det_id = n_wanted
                n_wanted += 1
                wanted.setdefault(d.frame, []).append((d.det_id, d.box, pad))
    store = crop_pass(video, wanted)

    mdir = out_root / event / match_key
    cdir = mdir / "crops"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    sheet_b: List[Tuple[np.ndarray, str]] = []
    sheet_r: List[Tuple[np.ndarray, str]] = []
    n_deint = 0
    # auto: woven video -> deinterlace_crop (its own comb gate decides per crop);
    # progressive video -> untouched; unknown (no ffmpeg) -> per-crop comb gate.
    do_deint = {"always": True, "never": False}.get(deinterlace, interlaced is not False)
    for tracks, detector in ((btracks, "bumper"), (rtracks, "robot")):
        for tr in tracks:
            picks = [d for d in picks_by_tr[tr.tid] if d.det_id in store]
            if detector == "bumper":
                linked, colour, extra = b2r.get(tr.tid), colour_of(tr.cls), {}
            else:
                best, colour, all_b = rlinks.get(tr.tid, (None, None, []))
                linked, extra = best, {"linked_bumper_tracklets": all_b}
            t_start, t_end = tr.dets[0].t, tr.dets[-1].t
            for j, d in enumerate(picks):
                raw = store[d.det_id]
                cr = comb_ratio(raw)
                img = deinterlace_crop(raw) if do_deint else raw
                deint = img is not raw
                n_deint += deint
                rel = Path(event) / match_key / "crops" / f"{tr.tid}_f{d.frame}.jpg"
                cv2.imwrite(str(out_root / rel), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                rows.append(manifest_row(
                    event=event, match_key=match_key, det=d, video_fps=vfps, detector=detector,
                    tracklet_id=tr.tid, linked_tracklet_id=linked, alliance_colour=colour,
                    comb_ratio_before=cr, deinterlaced=deint, path=str(rel), roster=roster,
                    video_interlaced=interlaced,
                    tracklet_len=len(tr.dets), tracklet_start_t=round(t_start, 3),
                    tracklet_end_t=round(t_end, 3), tracklet_dur_s=round(t_end - t_start, 3),
                    tracklet_n_crops=len(picks), det_fps=fps_out, tracker=tracker,
                    field_detect=field_detect, **extra))
                if j == len(picks) // 2:  # one tile per tracklet (its middle crop)
                    lab = f"{tr.tid} {colour or '-'} {int(min(box_wh(d.box)))}px n{len(tr.dets)}"
                    if detector == "robot":
                        lab += f" <{linked or '-'}"
                    (sheet_b if detector == "bumper" else sheet_r).append((img, lab))

    # contact sheet: up to 48 bumper + 24 robot tracklets spread over the match
    tiles = [sheet_b[i] for i in spread_pick(len(sheet_b), 48)]
    rt = [sheet_r[i] for i in spread_pick(len(sheet_r), 24)]
    if roster:
        title = f"{match_key}  red {' '.join(roster['red'])}  blue {' '.join(roster['blue'])}"
    else:
        title = f"{match_key}  roster unknown"
    sheet = np.vstack([contact_sheet(tiles, title=title + "  | bumpers"),
                       contact_sheet(rt, title="robots (<linked bumper)")])
    cv2.imwrite(str(mdir / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    tmp = mdir / "manifest.jsonl.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_b_rows = sum(r["detector"] == "bumper" for r in rows)
    linked_b = [t for t in btracks if b2r.get(t.tid)]
    summary = {
        "match_key": match_key, "video": str(video), "video_fps": vfps, "video_frames": nframes,
        "width": W, "height": H, "samples": len(samples), "samples_kept": len(kept),
        "det_fps": fps_out, "tracker": tracker, "track_buffer_samples": buffer,
        "field_detect": field_detect,
        "bumper_dets": sum(d.conf >= BUMPER_CONF for v in bd.values() for d in v),
        "robot_dets": sum(d.conf >= ROBOT_CONF for v in rd.values() for d in v),
        "bumper_tracklets": len(btracks), "robot_tracklets": len(rtracks),
        "bumper_track_stats": tracklet_stats(btracks), "robot_track_stats": tracklet_stats(rtracks),
        "bumper_crops": n_b_rows, "robot_crops": len(rows) - n_b_rows,
        "bumper_tracklets_with_crops": sum(1 for t in btracks if picks_by_tr.get(t.tid)),
        "bumper_tracklets_linked": len(linked_b),
        "linked_fraction": round(len(linked_b) / len(btracks), 3) if btracks else None,
        "robot_tracklets_linked": len(rlinks),
        "deinterlaced_crops": n_deint, "video_interlaced": interlaced, "idet": idet,
        "roster": roster,
        "detect_sec": round(t_detect, 1), "track_sec": round(t_track, 1),
        "total_sec": round(time.time() - t0, 1),
        "crop_bytes": sum(p.stat().st_size for p in cdir.glob("*.jpg")),
    }
    with open(mdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(tmp, mdir / "manifest.jsonl")  # manifest last = match done (idempotency marker)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", help="TBA event key, e.g. 2025cur")
    ap.add_argument("--matches", help="qm1-qm15 | qm3,qm7 (default: the event's scope in events.py)")
    ap.add_argument("--video", type=Path, help="local video (with --match-key) instead of downloading")
    ap.add_argument("--match-key", help="e.g. 2026cur_qm22 (with --video)")
    ap.add_argument("--det-fps", "--fps", dest="det_fps", type=float, default=DET_FPS,
                    help="detection / association rate (v1 used 2)")
    ap.add_argument("--tracker", choices=("bytetrack", "greedy"), default="bytetrack",
                    help="greedy = v1 reproduction (with --det-fps 2 --det-deinterlace never)")
    ap.add_argument("--det-cache", type=Path,
                    help="cache raw detections here (re-runs skip the GPU pass; tuning)")
    ap.add_argument("--keep-video", action="store_true", help="keep videos downloaded by this run")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    ap.add_argument("--limit-frames", type=int, help="stop after N sampled frames (smoke runs)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--force", action="store_true", help="re-harvest matches with a manifest")
    ap.add_argument("--deinterlace", choices=("auto", "always", "never"), default="auto",
                    help="auto: deinterlace crops only when ffmpeg idet finds a woven video")
    ap.add_argument("--det-deinterlace", choices=("auto", "always", "never"), default="auto",
                    help="auto: detect on one field (stretched) when the video is woven")
    args = ap.parse_args(argv)

    if args.video:
        if not args.match_key:
            ap.error("--video needs --match-key")
        jobs = [(args.match_key, args.video)]
    elif args.event:
        if args.matches:
            ms = parse_matches(args.matches)
        else:
            from scripts.teacher.events import EVENTS
            if args.event not in EVENTS:
                ap.error(f"{args.event} has no default scope; pass --matches")
            ms = EVENTS[args.event]
        jobs = [(f"{args.event}_{m}", None) for m in ms]
    else:
        ap.error("need --event or --video/--match-key")

    detectors = None
    results = []
    for match_key, video in jobs:
        event = match_key.split("_", 1)[0]
        mdir = args.out_root / event / match_key
        if (mdir / "manifest.jsonl").exists() and not args.force:
            print(f"[harvest] {match_key}: manifest exists, skipping (--force to redo)", flush=True)
            continue
        t0 = time.time()
        match = tba_match(match_key)
        roster = parse_roster(match)
        downloaded = False
        if video is None:
            video, downloaded = acquire_video(match_key, event, match, args.video_root)
            if video is None:
                results.append({"match_key": match_key, "error": "no video"})
                continue
        t_dl = time.time() - t0
        video_mb = video.stat().st_size / 1e6
        if detectors is None:
            detectors = Detectors(imgsz=args.imgsz)
        try:
            s = harvest_match(match_key=match_key, video=video, out_root=args.out_root,
                              detectors=detectors, roster=roster, fps_out=args.det_fps,
                              limit_frames=args.limit_frames, batch=args.batch,
                              deinterlace=args.deinterlace, tracker=args.tracker,
                              det_cache=args.det_cache, det_deinterlace=args.det_deinterlace)
        finally:
            if downloaded and not args.keep_video:
                video.unlink(missing_ok=True)
        s.update({"download_sec": round(t_dl, 1), "downloaded": downloaded,
                  "video_mb": round(video_mb, 1)})
        with open(mdir / "summary.json", "w") as f:
            json.dump(s, f, indent=1)
        results.append(s)
        bs = s["bumper_track_stats"]
        print(f"[harvest] {match_key}: {s['bumper_crops']} bumper / {s['robot_crops']} robot crops, "
              f"{s['bumper_tracklets']}/{s['robot_tracklets']} tracklets (bumper mean "
              f"{bs.get('mean_dets')} / median {bs.get('median_dets')} dets, singletons "
              f"{bs.get('singleton_frac')}), linked "
              f"{s['linked_fraction']}, kept {s['samples_kept']}/{s['samples']} samples, "
              f"{s['crop_bytes'] / 1e6:.1f} MB, dl {t_dl:.0f}s + harvest {s['total_sec']:.0f}s",
              flush=True)
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"[harvest] {len(errs)} match(es) failed: {[r['match_key'] for r in errs]}", flush=True)
    return 1 if errs and len(errs) == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())

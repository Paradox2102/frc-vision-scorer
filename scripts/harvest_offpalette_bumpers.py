#!/usr/bin/env python3
"""Harvest fully-labelled MAIN-view frames for off-palette (teal / navy) bumpers.

The production bumper model (frc_bumpers_equipment_hardneg) misses teal (4322)
and near-black navy (6800) bumpers. This script builds a YOLO dataset of MAIN
broadcast views (rows 0..722 of a 1080p frame, saved at native 1920x723) in
which EVERY bumper is labelled, so the fine-tune never learns "real bumper =
background".

Per sampled frame (2 Hz, only while the broadcast match clock is ticking):
  1. robot detector (pipeline's hub_zone model, class 'robot', conf 0.20 cached;
     >= 0.35 robots MUST get a bumper) and the current bumper model (conf 0.15)
     on the main view, CPU only.
  2. each robot box (inside the per-match field ROI) needs exactly one bumper:
     a model bumper whose centre is in the robot's lower 60% (touching
     same-class pieces merged; class verified against pixel colour), else the
     TEAL pixel labeller (largest bumper-shaped teal component in the robot's
     lower band). Teal labels get 4322's alliance from TBA. A teal component
     also replaces a looser model box on the same robot ("refined_teal").
     There is NO navy pixel labeller: 6800's near-black bumper cannot be
     separated from its black frame / the blue-grey carpet by colour, so navy
     labels only come from model boxes that span >= 70% of the robot width.
  3. any robot left without a bumper, a class/colour doubt, a partial model
     box, a mid-confidence (0.30-0.60) bumper detection with no robot, or an
     unlabelled moving bumper-sized teal blob in the field -> the whole frame
     is rejected. Recurring (static) robot-less detections are field-equipment
     false positives and are left unlabelled.
Detections are cached per match (data/bumper_offpalette/cache/) so the
labeller can be re-tuned without re-running YOLO.

CPU only (CUDA_VISIBLE_DEVICES is cleared before torch loads).

Usage:
  .venv/bin/python scripts/harvest_offpalette_bumpers.py            # all train matches
  .venv/bin/python scripts/harvest_offpalette_bumpers.py --matches 2026cur_qm63 --jobs 1
  .venv/bin/python scripts/harvest_offpalette_bumpers.py --finalize # yaml + STATS only
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never touch the GPU
# keep each worker's decode/BLAS threads bounded: several workers share a CPU
# that is also feeding GPU pipeline jobs (oversubscription made a run 4x slower)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import asyncio
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[1]
ROBOT_MODEL = ROOT / "runs/train/hub_zone_yolo26s_2026cur_v2/weights/best.pt"
BUMPER_MODEL = ROOT / "runs/train/frc_bumpers_equipment_hardneg/weights/best.pt"
OLD_DATA = ROOT / "data/bumper_hardneg_equipment"
OUT = ROOT / "data/bumper_offpalette/v1"
CACHE = ROOT / "data/bumper_offpalette/cache"
FRAMES = CACHE / "frames"     # 2 Hz main-view JPEGs (q95), one decode per video
MAIN_ROWS = 723
NAMES = {0: "blue_bumper", 1: "red_bumper"}
CLS = {"blue": 0, "red": 1}

TRAIN = ["2026cabl_qm18", "2026cabl_qm26", "2026cabl_qm44",
         "2026cur_qm22", "2026cur_qm39", "2026cur_qm52"]
VALID = ["2026cabl_qm65", "2026cur_qm63"]
HELD_OUT = {"2026cabl_qm5", "2026cabl_qm12", "2026cur_qm103", "2026cur_qm112", "2026cur_qm85"}
# off-palette teams per event: colour family -> team key
OFFPALETTE_TEAMS = {"2026cabl": {"teal": "frc4322", "navy": "frc498"},
                    "2026cur": {"navy": "frc6800"}}

ROBOT_CONF = 0.35          # production DET_CONF_THRESHOLD
BUMP_CONF_LOW = 0.15       # harvest: catch weak off-palette detections
BUMP_CONF_PROD = 0.35
NOROBOT_KEEP_CONF = 0.60   # robot-less bumper this confident -> label it
NOROBOT_IGNORE_CONF = 0.30  # below this a robot-less bumper is ignored (left as background)
SAMPLE_HZ = 2.0            # candidate sampling rate; selection keeps >= MIN_GAP_S apart
MIN_GAP_S = 0.5
MAX_PER_MATCH = 120


# --------------------------------------------------------------------------- colour
def hsv_masks(hsv: np.ndarray) -> dict:
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    # teal: widened after looking at 4322 masks (front face is less saturated
    # than the bumper top); the white numerals are filled by closing
    teal = (h >= 78) & (h <= 102) & (s > 30) & (v > 45)
    navy = (h >= 100) & (h <= 130) & (s > 60) & (v >= 6) & (v <= 75)
    royal = (h >= 100) & (h <= 125) & (s > 120) & (v > 90)
    bluish = (h >= 80) & (h <= 130) & (s > 50) & (v > 6)
    # red: relaxed for dark maroon / behind-polycarbonate bumpers (3128 at cur:
    # S~76 V~48); H <= 5 keeps orange robot frames (4322 is orange, H 6-20) out
    red = ((h <= 5) | (h >= 168)) & (s > 55) & (v > 30)
    return {"teal": teal, "navy": navy, "royal": royal, "bluish": bluish, "red": red}


def box_colour(img: np.ndarray, box) -> dict:
    x1, y1, x2, y2 = [int(round(c)) for c in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return {k: 0.0 for k in ("teal", "navy", "royal", "bluish", "red", "blue_medH", "blue_p90V")}
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    m = hsv_masks(hsv)
    out = {k: float(v.mean()) for k, v in m.items()}
    bl = hsv[m["bluish"]]
    out["blue_medH"] = float(np.median(bl[:, 0])) if len(bl) >= 15 else -1.0
    out["blue_p90V"] = float(np.percentile(bl[:, 2], 90)) if len(bl) >= 15 else -1.0
    return out


def colour_family(fr: dict) -> str:
    """teal / navy / royal / red / unknown for a final bumper box.
    Measured on model boxes (2026cur_qm63, 2026cabl_qm26): royal bumpers have
    bluish-pixel p90 V >= 120 and median H 105-120; 6800's navy p90 V < 80;
    4322's teal median H 95-96."""
    if fr["red"] >= 0.10 and fr["red"] > fr["bluish"]:
        return "red"
    if fr["bluish"] < 0.10 or fr["blue_medH"] < 0:
        return "unknown"
    if fr["blue_medH"] <= 101 and fr["teal"] >= 0.10:
        return "teal"
    if fr["blue_p90V"] < NAVY_V[0]:
        return "navy"
    return "royal"


# navy threshold on bluish-pixel p90 V; set per match by set_navy_threshold()
# because broadcast exposure differs a lot between matches (cur qm39/qm52 are
# dark enough that royal 571/294 bumpers fell under a fixed threshold)
NAVY_V = [95.0]


def set_navy_threshold(match_key: str, frames: list) -> float:
    vals = []
    for f in frames[::4]:
        bs = [b for b in f["bumpers"] if b[5] == CLS["blue"] and b[4] >= 0.6]
        if not bs:
            continue
        img = cv2.imread(str(frame_path(match_key, f["idx"])))
        if img is None:
            continue
        for b in bs:
            v = box_colour(img, b[:4])["blue_p90V"]
            if v > 0:
                vals.append(v)
    NAVY_V[0] = 0.6 * float(np.median(vals)) if len(vals) >= 20 else 95.0
    return NAVY_V[0]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def frac_inside(inner, outer) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / a if a > 0 else 0.0


# --------------------------------------------------------------------------- pixel labeller
def pixel_bumper(img: np.ndarray, rbox, family: str):
    """Largest bumper-shaped teal/navy component in the robot box's lower band.
    Returns (box, info) or (None, reason)."""
    rx1, ry1, rx2, ry2 = rbox
    rw, rh = rx2 - rx1, ry2 - ry1
    H, W = img.shape[:2]
    # search band: robot width + 15% each side (to see if a blob runs beyond the
    # robot = wall strip / field element), lower 65% of the box + a little below
    sx1, sx2 = int(max(0, rx1 - 0.15 * rw)), int(min(W, rx2 + 0.15 * rw))
    sy1, sy2 = int(max(0, ry1 + 0.35 * rh)), int(min(H, ry2 + 0.06 * rh))
    if sx2 - sx1 < 8 or sy2 - sy1 < 6:
        return None, "tiny"
    hsv = cv2.cvtColor(img[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2HSV)
    raw = hsv_masks(hsv)[family].astype(np.uint8)
    kx = max(3, int(0.08 * rw) | 1)
    ky = max(3, int(0.06 * rh) | 1)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    best, why = None, "no_component"
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.02 * rw * rh:
            continue
        bx1, by1, bx2, by2 = sx1 + x, sy1 + y, sx1 + x + w, sy1 + y + h
        if w < 0.40 * rw:
            why = "narrow"; continue
        if w / max(h, 1) < 1.2:
            why = "not_wide"; continue
        if w / max(h, 1) > 7.0 or h < 0.12 * rh:
            why = "thin_strip"; continue      # polycarbonate wall edge / field line
        if h > 0.65 * rh:
            why = "tall"; continue
        if by2 < ry1 + 0.70 * rh:
            why = "not_bottom"; continue
        if bx1 < rx1 - 0.10 * rw or bx2 > rx2 + 0.10 * rw:
            why = "runs_past_robot"; continue
        fill = area / float(w * h)
        if fill < 0.35:
            why = "sparse"; continue
        if best is None or area > best[1]:
            # tighten to the raw (un-closed) mask pixels of this component
            comp = (lab == i) & (raw > 0)
            ys, xs = np.nonzero(comp)
            if len(xs) < 10:
                continue
            tb = (sx1 + xs.min(), sy1 + ys.min(), sx1 + xs.max() + 1, sy1 + ys.max() + 1)
            best = (tb, area, fill)
    if best is None:
        # a ball / robot part can split the bumper: union the aligned pieces
        pieces = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 0.01 * rw * rh or h > 0.65 * rh:
                continue
            bx1, by1, bx2, by2 = sx1 + x, sy1 + y, sx1 + x + w, sy1 + y + h
            if bx1 < rx1 - 0.10 * rw or bx2 > rx2 + 0.10 * rw or by2 < ry1 + 0.65 * rh:
                continue
            pieces.append((bx1, by1, bx2, by2, area))
        if len(pieces) < 2:
            return None, why
        pieces.sort(key=lambda p: -p[4])
        big = pieces[0]
        grp = [big] + [p for p in pieces[1:]
                       if min(p[3], big[3]) - max(p[1], big[1]) >= 0.4 * min(p[3] - p[1], big[3] - big[1])]
        if len(grp) < 2:
            return None, why
        ux1, uy1 = min(p[0] for p in grp), min(p[1] for p in grp)
        ux2, uy2 = max(p[2] for p in grp), max(p[3] for p in grp)
        uw, uh = ux2 - ux1, uy2 - uy1
        area = sum(p[4] for p in grp)
        if uw < 0.45 * rw or uw / max(uh, 1) < 1.2 or uh > 0.65 * rh or area / float(uw * uh) < 0.3:
            return None, why + "+union_failed"
        return (float(ux1), float(uy1), float(ux2), float(uy2)), {"fill": round(area / float(uw * uh), 2), "union": len(grp)}
    return tuple(float(c) for c in best[0]), {"fill": round(best[2], 2)}


def global_blobs(img: np.ndarray, family: str, roi, min_w: float, max_w: float, static=None):
    """Bumper-sized teal/navy blobs anywhere in the field ROI (for the
    'unlabelled off-palette bumper' frame veto)."""
    x1, y1, x2, y2 = roi
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    raw = hsv_masks(hsv)[family].astype(np.uint8)
    k = max(3, int(min_w * 0.15) | 1)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, k // 2 | 1))))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_w or w > max_w or w / max(h, 1) < 1.5 or area / float(w * h) < 0.4:
            continue
        if static is not None:
            st = static[y1 + y:y1 + y + h, x1 + x:x1 + x + w]
            if st.mean() > 0.3:
                continue
        out.append((x1 + x, y1 + y, x1 + x + w, y1 + y + h))
    return out


# --------------------------------------------------------------------------- detection cache
def extract_frames(match_key: str, video: Path) -> dict:
    """Decode the video ONCE; keep every 2 Hz sample's main view as a q95 JPEG
    (the dataset's own format) so every later stage reads small files instead
    of re-decoding 13k 1080p frames."""
    d = FRAMES / match_key
    idx_f = d / "index.json"
    if idx_f.exists():
        return json.loads(idx_f.read_text())
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = fps / SAMPLE_HZ
    want = set(int(round(i * step)) for i in range(int(n / step) + 1))
    got = []
    idx = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx not in want:
            continue
        ok, f = cap.retrieve()
        if ok:
            cv2.imwrite(str(d / f"f{idx:06d}.jpg"), f[:MAIN_ROWS], [cv2.IMWRITE_JPEG_QUALITY, 95])
            got.append(idx)
    cap.release()
    info = {"match": match_key, "video": str(video), "fps": fps, "idxs": got}
    idx_f.write_text(json.dumps(info))
    return info


def frame_path(match_key: str, idx: int) -> Path:
    return FRAMES / match_key / f"f{idx:06d}.jpg"


def detect_match(match_key: str, video: Path, threads: int = 3) -> dict:
    cache_f = CACHE / f"{match_key}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    info = extract_frames(match_key, video)
    fps = info["fps"]
    # only run YOLO inside the ticking-clock window (+-3 s)
    clock = [(i / fps, clock_patch(cv2.imread(str(frame_path(match_key, i))))) for i in info["idxs"]]
    cw = clock_window(clock)
    idxs = [i for i in info["idxs"] if cw is None or cw[0] - 3 <= i / fps <= cw[1] + 3]
    import torch
    from ultralytics import YOLO
    rob, bm = YOLO(str(ROBOT_MODEL)), YOLO(str(BUMPER_MODEL))
    if idxs:
        warm = cv2.imread(str(frame_path(match_key, idxs[0])))
        rob(warm, device="cpu", verbose=False)
        bm(warm, device="cpu", verbose=False)
    # AFTER the first predict: ultralytics' select_device("cpu") resets torch
    # to 8 threads, which made parallel workers spin-wait (10x slower)
    torch.set_num_threads(threads)
    frames = []
    for idx in idxs:
        m = cv2.imread(str(frame_path(match_key, idx)))
        r = rob(m, conf=0.20, device="cpu", verbose=False)[0]
        b = bm(m, conf=BUMP_CONF_LOW, device="cpu", verbose=False)[0]
        robots = [[*map(float, xy), float(c)] for xy, c, k in
                  zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
                  if r.names[int(k)] == "robot"]
        bumps = [[*map(float, xy), float(c), int(k)] for xy, c, k in
                 zip(b.boxes.xyxy.tolist(), b.boxes.conf.tolist(), b.boxes.cls.tolist())]
        frames.append({"idx": idx, "t": idx / fps, "robots": robots, "bumpers": bumps})
    data = {"match": match_key, "video": str(video), "fps": fps, "frames": frames}
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_f.write_text(json.dumps(data))
    return data


def match_period(frames: list, max_gap: float = 10.0) -> tuple:
    """Longest run of samples with >= 4 bumpers (conf >= 0.35), gaps <= max_gap s
    bridged. Pre-match robot placement and post-match field shots are separate
    shorter runs (broadcast cuts to graphics / replays between)."""
    hits = [f["t"] for f in frames if sum(1 for b in f["bumpers"] if b[4] >= BUMP_CONF_PROD) >= 4]
    if not hits:
        return None
    runs, start, prev = [], hits[0], hits[0]
    for t in hits[1:]:
        if t - prev > max_gap:
            runs.append((start, prev))
            start = t
        prev = t
    runs.append((start, prev))
    return max(runs, key=lambda r: r[1] - r[0])


CLOCK_ROI = (75, 122, 905, 1015)   # y1, y2, x1, x2: FIRST AGE overlay match clock (both events)


def clock_patch(img: np.ndarray):
    """Grey clock patch, or None when the overlay's white clock panel is absent
    (replay / graphics / transition)."""
    y1, y2, x1, x2 = CLOCK_ROI
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if np.percentile(g, 60) < 170 or (g < 90).mean() < 0.04:
        return None
    return g


def clock_window(samples: list, min_change: float = 6.0):
    """(t_first, t_last) of a ticking clock. samples: [(t, patch|None)] at
    SAMPLE_HZ. The clock image changes once a second while counting and holds
    still at 0:20 pre-match and 0:00 post-match."""
    ticks = []
    by_t = {round(t, 2): p for t, p in samples}
    for t, p in samples:
        q = by_t.get(round(t - 1.0, 2))
        if p is None or q is None:
            continue
        if float(np.abs(p.astype(np.int16) - q.astype(np.int16)).mean()) >= min_change:
            ticks.append(t)
    if len(ticks) < 20:
        return None
    # the longest run of ticks with gaps <= 8 s (auto->teleop hold is ~3 s)
    runs, start, prev = [], ticks[0], ticks[0]
    for t in ticks[1:]:
        if t - prev > 8.0:
            runs.append((start, prev)); start = t
        prev = t
    runs.append((start, prev))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return a - 1.0, b


def static_detections(frames: list, period, frac: float = 0.25) -> set:
    """(frame idx, det index) of bumper detections that recur at the same place
    (IoU > 0.6) in >= frac of the in-period frames: field equipment false
    positives (hub ramps, alliance walls). Left unlabelled (= hard negatives)."""
    dets, keys = [], []
    fr = [f for f in frames if period[0] <= f["t"] <= period[1]]
    for f in fr:
        for j, b in enumerate(f["bumpers"]):
            dets.append(b[:4]); keys.append((f["idx"], j))
    if not dets:
        return set()
    a = np.array(dets, np.float32)
    x1 = np.maximum(a[:, None, 0], a[None, :, 0]); y1 = np.maximum(a[:, None, 1], a[None, :, 1])
    x2 = np.minimum(a[:, None, 2], a[None, :, 2]); y2 = np.minimum(a[:, None, 3], a[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    iou_m = inter / (area[:, None] + area[None, :] - inter + 1e-6)
    counts = (iou_m > 0.6).sum(1)
    thr = max(5, frac * len(fr))
    return {keys[i] for i in np.nonzero(counts >= thr)[0]}


def static_teal_map(video: str, frames: list, period, n: int = 40) -> np.ndarray:
    """Pixels that are teal in > 40% of n in-period frames (field elements,
    polycarbonate edges) -- excluded from the unlabelled-teal-blob veto."""
    fr = [f for f in frames if period[0] <= f["t"] <= period[1]]
    pick = fr[:: max(1, len(fr) // n)][:n]
    imgs = _read_frame(video, [f["idx"] for f in pick])
    acc = None
    for im in imgs.values():
        m = hsv_masks(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))["teal"].astype(np.float32)
        acc = m if acc is None else acc + m
    return (acc / max(1, len(imgs))) > 0.4 if acc is not None else None


def field_roi(frames: list, period, W=1920, H=MAIN_ROWS) -> tuple:
    """Per-match field ROI from robust percentiles of confident robot boxes.
    The camera differs per match (cabl) so this is learned from the match."""
    xs1, xs2, ys1, ys2 = [], [], [], []
    for f in frames:
        if not (period[0] <= f["t"] <= period[1]):
            continue
        for r in f["robots"]:
            if r[4] >= 0.6:
                xs1.append(r[0]); xs2.append(r[2]); ys1.append(r[1]); ys2.append(r[3])
    if len(ys2) < 20:
        return (0, 0, W, H)
    # bumpers sit at the robot box bottom: ROI vertical span = robot bottoms
    top = max(0, int(np.percentile(ys1, 1) - 20))
    bot = min(H, int(np.percentile(ys2, 99.5) + 25))
    left = max(0, int(np.percentile(xs1, 0.5) - 40))
    right = min(W, int(np.percentile(xs2, 99.5) + 40))
    return (left, top, right, bot)


# --------------------------------------------------------------------------- labelling
def get_alliances(match_key: str) -> dict:
    sys.path.insert(0, str(ROOT / "web"))
    import tba
    m = asyncio.run(tba.get_match(match_key))
    return {al: list(m["alliances"][al]["team_keys"]) for al in ("blue", "red")}


def offpalette_classes(match_key: str, alliances: dict) -> dict:
    """family -> class id for off-palette labels in this match. Only teams that
    play in this match appear; None = the team is on RED (its off-palette
    bumper would be a red_bumper -> frames with that family are skipped)."""
    ev = match_key.split("_")[0]
    out = {}
    for fam, team in OFFPALETTE_TEAMS.get(ev, {}).items():
        if team in alliances["blue"]:
            out[fam] = CLS["blue"]
        elif team in alliances["red"]:
            out[fam] = None
    return out


def _in_lower(b, r) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    rh = r[3] - r[1]
    return r[0] <= cx <= r[2] and r[1] + 0.4 * rh <= cy <= r[3] + 0.05 * rh


def _inside(b, r) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


def label_frame(img: np.ndarray, fr: dict, roi, fam_cls: dict, med_rw: float,
                static: set = frozenset(), static_teal=None, med_rh: float = 80.0):
    """Return (labels, reason). labels: list of dicts with box/cls/src/conf/family.

    Robots >= ROBOT_CONF must each get exactly one bumper label; weaker robot
    boxes (0.20-0.35) are used only if a bumper detection sits on them."""
    W = img.shape[1]
    in_roi = lambda b: roi[0] <= (b[0] + b[2]) / 2 <= roi[2] and roi[1] <= b[3] <= roi[3] + 10
    robots = [r for r in fr["robots"] if r[4] >= 0.20 and in_roi(r)]
    bumps = [b + [(fr["idx"], j) in static] for j, b in enumerate(fr["bumpers"]) if in_roi(b)]
    if not any(r[4] >= ROBOT_CONF for r in robots):
        return None, "no_robots"

    assign = {i: [] for i in range(len(robots))}
    orphans = []
    for b in bumps:
        cands = [i for i, r in enumerate(robots) if _in_lower(b, r)]
        if cands:
            # prefer a confident robot, then the one whose bottom matches the bumper's
            i = min(cands, key=lambda i: (robots[i][4] < ROBOT_CONF, abs(robots[i][3] - b[3])))
            assign[i].append(b)
        else:
            orphans.append(b)

    labels = []
    for b in orphans:
        if b[6]:
            continue           # recurring equipment false positive: left as background
        if b[4] < 0.20:
            continue           # very weak, unexplained: left as background
        if b[4] < NOROBOT_IGNORE_CONF:
            # 0.20-0.30 robot-less bumper: often a blurred / cut-off robot that
            # both detectors half-see -> unlabelled-bumper risk
            return None, "weak_orphan_bumper"
        if any(_inside(b, r) for r in robots if r[4] >= ROBOT_CONF):
            return None, "bumper_in_robot_upper_part"
        if b[4] < NOROBOT_KEEP_CONF:
            return None, "orphan_bumper_midconf"
        if b[0] <= 2 or b[2] >= W - 2:
            return None, "orphan_bumper_at_frame_edge"
        fam_o = colour_family(box_colour(img, b[:4]))
        if fam_o in ("navy", "teal"):
            # no robot box to check completeness against: off-palette model
            # boxes are often one face only (seen on 6800)
            return None, f"{fam_o}_orphan_unverifiable"
        labels.append({"box": list(b[:4]), "cls": b[5], "src": "model_norobot", "conf": b[4]})

    for i, r in enumerate(robots):
        rbox = r[:4]
        rw, rh = r[2] - r[0], r[3] - r[1]
        bs = assign[i]
        if not bs and r[4] < ROBOT_CONF:
            # weak robot box (0.20-0.35) without a bumper: usually a motion-
            # blurred robot whose bumper would go unlabelled
            return None, "weak_robot_without_bumper"
        teal_pb = None
        if "teal" in fam_cls:
            teal_pb, _ = pixel_bumper(img, rbox, "teal")
        if bs:
            classes = {b[5] for b in bs}
            if len(classes) > 1:
                return None, "mixed_classes_on_robot"
            cls = next(iter(classes))
            bs = sorted(bs, key=lambda b: b[0])
            if len(bs) > 1:
                # the model often boxes the two visible faces of one bumper separately
                if any(bs[j + 1][0] - bs[j][2] > 0.2 * rw for j in range(len(bs) - 1)):
                    return None, "multi_bumper_on_robot"
                box = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
                conf, src = max(b[4] for b in bs), "model_merged"
            else:
                box, conf = list(bs[0][:4]), bs[0][4]
                src = "model" if conf >= BUMP_CONF_PROD else "model_low"
            if (box[2] - box[0]) > 1.3 * rw:
                return None, "bumper_wider_than_robot"
            fc = box_colour(img, box)
            # class check on the lower half (the bumper's front face): robot
            # parts above it can be either colour (7028's blue electronics)
            lo = box_colour(img, [box[0], (box[1] + box[3]) / 2, box[2], box[3]])
            if cls == CLS["blue"]:
                if not (lo["bluish"] >= 0.08 and lo["bluish"] > 1.2 * lo["red"]):
                    return None, "class_doubt_blue"
            else:
                if not (lo["red"] >= 0.08 and lo["red"] > 1.2 * lo["bluish"]):
                    return None, "class_doubt_red"
            fam = colour_family(fc)
            if fam == "teal" and fam_cls.get("teal") is None:
                return None, "teal_on_wrong_alliance"
            if fam == "navy" and fam_cls.get("navy") is None and cls == CLS["blue"] and "navy" in fam_cls:
                return None, "navy_on_wrong_alliance"
            # teal: a clean pixel component is tighter than the model box
            if teal_pb is not None and cls == CLS["blue"] and fam in ("teal", "royal", "navy", "unknown"):
                ov = iou(box, teal_pb)
                if fam == "teal" or ov >= 0.3 or frac_inside(teal_pb, box) >= 0.6:
                    if (teal_pb[2] - teal_pb[0]) >= 0.7 * (box[2] - box[0]) or fam == "teal":
                        if ov < 0.85 and teal_pb[3] <= roi[3] - 0.5 * med_rh:
                            box, src = list(teal_pb), "refined_teal"
                        fam = "teal"
            # off-palette model boxes are often one face / a slice: require the
            # box to span the robot and not swallow the robot body
            if fam in ("navy", "teal") and src != "refined_teal":
                if (box[2] - box[0]) < 0.7 * rw:
                    return None, f"{fam}_model_box_partial"
                if (box[3] - box[1]) > 0.6 * rh:
                    return None, f"{fam}_model_box_tall"
            elif (box[2] - box[0]) < 0.45 * rw and not (r[0] <= 2 or r[2] >= W - 2):
                return None, "model_box_partial"
            labels.append({"box": list(box), "cls": cls, "src": src, "conf": conf})
            continue
        # confident robot without a model bumper: teal pixel labeller only
        if teal_pb is None:
            return None, "robot_without_bumper"
        fc = box_colour(img, teal_pb)
        if fc["red"] > 0.10:
            return None, "pixel_box_has_red"
        if fam_cls.get("teal") is None:
            return None, "teal_on_wrong_alliance"
        if teal_pb[3] > roi[3] - 0.5 * med_rh:
            # robot against the near wall: bumper seen through polycarbonate,
            # pixel box is a guess
            return None, "pixel_teal_behind_near_wall"
        labels.append({"box": list(teal_pb), "cls": fam_cls["teal"], "src": "pixel_teal", "conf": 0.0})

    # veto: an unlabelled, moving, bumper-sized teal blob anywhere in the field
    if "teal" in fam_cls:
        for gb in global_blobs(img, "teal", roi, 0.4 * med_rw, 1.6 * med_rw, static_teal):
            if any(frac_inside(gb, l["box"]) >= 0.5 or iou(gb, l["box"]) >= 0.3 for l in labels):
                continue
            inside = [r for r in robots if frac_inside(gb, r[:4]) >= 0.7]
            if inside and any(frac_inside(l["box"], r[:4]) >= 0.7 for r in inside for l in labels):
                continue
            return None, "unlabelled_teal_blob"

    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            if iou(labels[a]["box"], labels[b]["box"]) > 0.5:
                return None, "duplicate_labels"
    for l in labels:
        l["family"] = colour_family(box_colour(img, l["box"]))
        if l["src"] == "refined_teal" or l["src"] == "pixel_teal":
            l["family"] = "teal"
    return labels, "ok"


def process_match(match_key: str, split: str, threads: int = 3) -> dict:
    assert match_key not in HELD_OUT, match_key
    video = None
    for d in (ROOT / "data/videos", ROOT / "data/test_footage"):
        if (d / f"{match_key}.mp4").is_file():
            video = d / f"{match_key}.mp4"
            break
    if video is None:
        return {"match": match_key, "error": "no video"}
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if (W, H) != (1920, 1080):
        return {"match": match_key, "error": f"video is {W}x{H}, not 1080p"}
    extract_frames(match_key, video)
    data = detect_match(match_key, video, threads)
    frames = data["frames"]
    period = match_period(frames)
    if period is None:
        return {"match": match_key, "error": "no match period found"}
    roi = field_roi(frames, period)
    navy_v = set_navy_threshold(match_key, frames)
    alliances = get_alliances(match_key)
    fam_cls = offpalette_classes(match_key, alliances)
    rws = [r[2] - r[0] for f in frames for r in f["robots"] if r[4] >= 0.6]
    med_rw = float(np.median(rws)) if rws else 100.0
    rhs = [r[3] - r[1] for f in frames for r in f["robots"] if r[4] >= 0.6]
    med_rh = float(np.median(rhs)) if rhs else 80.0

    static = static_detections(frames, period)
    static_teal = static_teal_map(str(video), frames, period) if "teal" in fam_cls else None
    in_period = [f for f in frames if period[0] <= f["t"] <= period[1]]
    reasons = Counter()
    cands = []
    clock_samples = []
    decided = {}
    for fr in in_period:
        idx = fr["idx"]
        img = cv2.imread(str(frame_path(match_key, idx)))
        if img is None:
            reasons["decode_fail"] += 1
            continue
        clock_samples.append((fr["t"], clock_patch(img)))
        labels, why = label_frame(img, fr, roi, fam_cls, med_rw, static, static_teal, med_rh)
        decided[fr["t"]] = why
        if labels is None:
            continue
        cands.append({"idx": idx, "t": fr["t"], "labels": labels})
    cw = clock_window(clock_samples)
    if cw is None:
        return {"match": match_key, "error": "match clock not found"}
    for t, why in decided.items():
        reasons["outside_match_clock" if not (cw[0] <= t <= cw[1]) else why] += 1
    cands = [c for c in cands if cw[0] <= c["t"] <= cw[1]]

    # selection: off-palette frames first, >= MIN_GAP_S apart, near-duplicate cap
    def has_off(c):
        return any(l["family"] in ("teal", "navy") for l in c["labels"])

    def static_dup(c, chosen):
        # a frame whose label boxes all sit within 4 px of an already chosen frame
        for o in chosen:
            if len(o["labels"]) != len(c["labels"]):
                continue
            ok = True
            for l in c["labels"]:
                if not any(max(abs(a - b) for a, b in zip(l["box"], m["box"])) <= 4 for m in o["labels"]):
                    ok = False
                    break
            if ok:
                return True
        return False

    chosen = []
    dup_skipped = 0
    for group in ([c for c in cands if has_off(c)], [c for c in cands if not has_off(c)]):
        for c in group:
            if len(chosen) >= MAX_PER_MATCH:
                break
            if any(abs(c["t"] - o["t"]) < MIN_GAP_S for o in chosen):
                continue
            if static_dup(c, chosen):
                dup_skipped += 1
                continue
            chosen.append(c)
    chosen.sort(key=lambda c: c["t"])

    # write images + labels
    img_dir = OUT / split / "images"
    lab_dir = OUT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for p in list(img_dir.glob(f"{match_key}_*")) + list(lab_dir.glob(f"{match_key}_*")):
        p.unlink()
    for c in chosen:
        idx = c["idx"]
        stem = f"{match_key}_f{idx:06d}"
        shutil.copyfile(frame_path(match_key, idx), img_dir / f"{stem}.jpg")
        lines = []
        for l in c["labels"]:
            x1, y1, x2, y2 = l["box"]
            x1, x2 = max(0.0, x1), min(1920.0, x2)
            y1, y2 = max(0.0, y1), min(float(MAIN_ROWS), y2)
            lines.append(f"{l['cls']} {(x1 + x2) / 2 / 1920:.6f} {(y1 + y2) / 2 / MAIN_ROWS:.6f} "
                         f"{(x2 - x1) / 1920:.6f} {(y2 - y1) / MAIN_ROWS:.6f}")
        (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    summary = {
        "match": match_key, "split": split, "video": str(video), "period": period, "clock_window": cw,
        "roi": roi,
        "alliances": alliances, "offpalette_classes": fam_cls, "median_robot_w": med_rw,
        "static_bumper_dets": len(static), "navy_p90v_threshold": navy_v,
        "sampled": sum(reasons.values()), "candidates": len(cands), "accepted": len(chosen),
        "dup_skipped": dup_skipped, "reasons": dict(reasons),
        "sources": dict(Counter(l["src"] for c in chosen for l in c["labels"])),
        "families": dict(Counter(l["family"] for c in chosen for l in c["labels"])),
        "frames_with_teal": sum(any(l["family"] == "teal" for l in c["labels"]) for c in chosen),
        "frames_with_navy": sum(any(l["family"] == "navy" for l in c["labels"]) for c in chosen),
        "chosen": chosen,
    }
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    (OUT / "meta" / f"{match_key}.json").write_text(json.dumps(summary, indent=1))
    make_review(summary)
    return {k: v for k, v in summary.items() if k != "chosen"}


# --------------------------------------------------------------------------- review sheets
SRC_COL = {"model": (0, 255, 0), "model_low": (0, 200, 120), "model_merged": (0, 255, 255),
           "model_norobot": (0, 140, 255), "refined_teal": (255, 255, 0), "refined_navy": (255, 255, 0),
           "pixel_teal": (255, 0, 255), "pixel_navy": (255, 0, 255)}


def _read_frame(video: str, idxs: list) -> dict:
    """Main-view frames by index: from the JPEG frame cache when present,
    else decoded from the video."""
    mk = Path(video).stem
    if (FRAMES / mk / "index.json").exists():
        return {i: cv2.imread(str(frame_path(mk, i))) for i in idxs}
    want, out = set(idxs), {}
    cap = cv2.VideoCapture(video)
    idx = -1
    while want:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx in want:
            want.discard(idx)
            out[idx] = cap.retrieve()[1][:MAIN_ROWS]
    cap.release()
    return out


def make_review(s: dict):
    rev = OUT / "review"
    rev.mkdir(parents=True, exist_ok=True)
    mk = s["match"]
    chosen = s["chosen"]
    if not chosen:
        return
    off = [c for c in chosen if any(l["family"] in ("teal", "navy") for l in c["labels"])]
    rest = [c for c in chosen if c not in off]
    pick = (off[:: max(1, len(off) // 14)][:14] + rest[:: max(1, len(rest) // 6)])[:20]
    pick.sort(key=lambda c: c["t"])
    crops_src = [(c, l) for c in chosen for l in c["labels"]
                 if l["src"].startswith(("pixel", "refined"))]
    off_src = [(c, l) for c in chosen for l in c["labels"]
               if l["family"] in ("teal", "navy") and not l["src"].startswith(("pixel", "refined"))]
    crops_src = crops_src[:: max(1, len(crops_src) // 48)][:48]
    off_src = off_src[:: max(1, len(off_src) // 48)][:48]
    need = {c["idx"] for c in pick} | {c["idx"] for c, _ in crops_src} | {c["idx"] for c, _ in off_src}
    imgs = _read_frame(s["video"], sorted(need))

    tiles = []
    for c in pick:
        im = imgs[c["idx"]].copy()
        x1, y1, x2, y2 = s["roi"]
        cv2.rectangle(im, (x1, y1), (x2 - 1, y2 - 1), (128, 128, 128), 1)
        for l in c["labels"]:
            b = [int(round(v)) for v in l["box"]]
            col = SRC_COL.get(l["src"], (255, 255, 255))
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), col, 2)
            tag = ("B" if l["cls"] == 0 else "R") + ({"teal": "t", "navy": "n"}.get(l["family"], ""))
            cv2.putText(im, tag, (b[0], b[1] - 3), 0, 0.6, col, 2)
        cv2.putText(im, f"{mk} t={c['t']:.1f}", (10, 30), 0, 1.0, (255, 255, 255), 2)
        # crop to the field ROI rows: more pixels per robot on the sheet
        cy1, cy2 = max(0, s["roi"][1] - 40), min(MAIN_ROWS, s["roi"][3] + 10)
        im = im[cy1:cy2]
        cv2.putText(im, f"{mk} t={c['t']:.1f}", (10, 30), 0, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (960, int(960 * im.shape[0] / 1920)), interpolation=cv2.INTER_AREA))
    for part in range(0, len(tiles), 10):
        ts = tiles[part:part + 10]
        while len(ts) % 2:
            ts.append(np.zeros_like(ts[0]))
        g = np.vstack([np.hstack(ts[i:i + 2]) for i in range(0, len(ts), 2)])
        cv2.imwrite(str(rev / f"{mk}_frames_{part // 10 + 1}.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def crop_sheet(items, name):
        if not items:
            return
        cs = []
        for c, l in items:
            im = imgs[c["idx"]]
            b = l["box"]
            bw, bh = b[2] - b[0], b[3] - b[1]
            pad_x, pad_y = max(20, bw * 0.4), max(30, bh * 1.5)
            x1, y1 = int(max(0, b[0] - pad_x)), int(max(0, b[1] - pad_y * 1.5))
            x2, y2 = int(min(1920, b[2] + pad_x)), int(min(MAIN_ROWS, b[3] + pad_y * 0.6))
            crop = im[y1:y2, x1:x2].copy()
            sc = 220.0 / max(1, crop.shape[1])
            if crop.shape[0] * sc > 220:
                sc = 220.0 / crop.shape[0]
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            bb = [int((b[0] - x1) * sc), int((b[1] - y1) * sc), int((b[2] - x1) * sc), int((b[3] - y1) * sc)]
            cv2.rectangle(crop, (bb[0], bb[1]), (bb[2], bb[3]), SRC_COL.get(l["src"], (255, 255, 255)), 1)
            tile = np.zeros((240, 220, 3), np.uint8)
            tile[:crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(tile, f"{l['src'][:12]} {l['family'][:4]} {c['t']:.0f}", (2, 236), 0, 0.38, (255, 255, 255), 1)
            cs.append(tile)
        while len(cs) % 8:
            cs.append(np.zeros_like(cs[0]))
        g = np.vstack([np.hstack(cs[i:i + 8]) for i in range(0, len(cs), 8)])
        cv2.imwrite(str(rev / f"{mk}_{name}.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 90])

    for old in rev.glob(f"{mk}_*crops.jpg"):
        old.unlink()
    crop_sheet(crops_src, "pixel_crops")
    crop_sheet(off_src, "model_offpalette_crops")


# --------------------------------------------------------------------------- outputs
def write_yamls():
    (OUT / "bumper_dataset.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: valid/images\n\n"
        "names:\n  0: blue_bumper\n  1: red_bumper\n\nnc: 2\n")
    (OUT / "merged.yaml").write_text(
        "# old hard-negative bumper set + off-palette harvest (v1)\n"
        "train:\n"
        f"  - {OLD_DATA}/train/images\n"
        f"  - {OUT}/train/images\n"
        "val:\n"
        f"  - {OLD_DATA}/valid/images\n"
        f"  - {OUT}/valid/images\n\n"
        "names:\n  0: blue_bumper\n  1: red_bumper\n\nnc: 2\n")


def write_stats():
    metas = [json.loads(p.read_text()) for p in sorted((OUT / "meta").glob("*.json"))]
    L = ["# Off-palette bumper harvest v1", "",
         "Generated by `scripts/harvest_offpalette_bumpers.py`. MAIN view (rows 0..722) of 1080p "
         "broadcast frames saved at 1920x723, every bumper labelled. Candidates sampled at "
         f"{SAMPLE_HZ:g} Hz, only while the broadcast match clock is ticking (clock-panel pixel change; "
         "pre-match 0:20 hold and post-match 0:00 hold excluded) and inside the longest run of samples "
         f"with >= 4 bumpers at conf >= {BUMP_CONF_PROD}; accepted frames >= {MIN_GAP_S:g} s apart, "
         f"off-palette frames preferred, max {MAX_PER_MATCH}/match.", "",
         "Label sources: `model` = bumper model conf >= 0.35 on a robot; `model_low` = 0.15-0.35 on a robot; "
         "`model_merged` = touching same-class pieces on one robot merged; `model_norobot` = conf >= 0.6 bumper "
         "with no robot box; `refined_teal` = model box replaced by the tighter teal pixel component; "
         "`pixel_teal` = no model box, teal pixel labeller (class = 4322's TBA alliance). There is NO navy "
         "pixel labeller (6800's near-black bumper is not separable by colour from its black frame / the "
         "blue-grey carpet): navy labels are model boxes spanning >= 70% of the robot width.", "",
         "Family = pixel colour of the final box: teal = bluish median H <= 101; navy = bluish p90 V below "
         "0.6 x the match's median for confident blue boxes (exposure differs per match); else royal / red.", "",
         "Validation split is by match (2026cabl_qm65, 2026cur_qm63).", "",
         "Held-out test matches (never harvested): " + ", ".join(sorted(HELD_OUT)), ""]
    tot = Counter()
    for m in metas:
        r = Counter(m["reasons"])
        ok = r.pop("ok", 0)
        L += [f"## {m['match']} ({m['split']})", "",
              f"- alliances: blue {' '.join(t[3:] for t in m['alliances']['blue'])} / red "
              f"{' '.join(t[3:] for t in m['alliances']['red'])}; pixel-label classes {m['offpalette_classes']}",
              f"- bumper period {m['period'][0]:.1f}-{m['period'][1]:.1f} s, match clock ticking "
              f"{m['clock_window'][0]:.1f}-{m['clock_window'][1]:.1f} s; field ROI {m['roi']}; "
              f"navy p90-V threshold {m.get('navy_p90v_threshold', 0):.0f}",
              f"- sampled {m['sampled']} / passed labelling {ok} / accepted {m['accepted']} "
              f"(rest dropped by the {MIN_GAP_S:g} s spacing, {m['dup_skipped']} static duplicates, cap)",
              "- rejected by reason: " + ", ".join(f"{k} {v}" for k, v in r.most_common()),
              "- labels by source: " + ", ".join(f"{k} {v}" for k, v in sorted(m["sources"].items())),
              "- labels by colour family: " + ", ".join(f"{k} {v}" for k, v in sorted(m["families"].items())),
              f"- accepted frames with teal: {m['frames_with_teal']}, with navy: {m['frames_with_navy']}", ""]
        tot[m["split"] + "_frames"] += m["accepted"]
        for k, v in m["families"].items():
            tot[m["split"] + "_" + k] += v
        for k, v in m["sources"].items():
            tot[m["split"] + "_src_" + k] += v
    L += ["## Totals", ""] + [f"- {k}: {v}" for k, v in sorted(tot.items())] + [""]
    (OUT / "STATS.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=None, help="comma list (default: all train+valid)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--finalize", action="store_true", help="only rewrite yamls + STATS.md")
    args = ap.parse_args()
    if not args.finalize:
        keys = args.matches.split(",") if args.matches else TRAIN + VALID
        bad = [k for k in keys if k in HELD_OUT]
        if bad:
            raise SystemExit(f"held-out matches may not be harvested: {bad}")
        split = {k: ("valid" if k in VALID else "train") for k in keys}
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {k: ex.submit(process_match, k, split[k], args.threads) for k in keys}
            for k, f in futs.items():
                try:
                    r = f.result()
                except Exception as e:  # keep the rest of the batch
                    r = {"match": k, "error": repr(e)}
                print(json.dumps(r), flush=True)
    write_yamls()
    write_stats()


if __name__ == "__main__":
    main()

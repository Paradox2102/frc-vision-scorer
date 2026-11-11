"""Detect a broadcast camera being re-aimed / re-zoomed DURING a match.

A static broadcast camera holds every AprilTag it sees to well under a pixel for a
whole match (2026cur corner cams: 0.1-0.5 px; a robot partly covering a tag adds
jitter up to ~5 px). A camera move shows up as a STEP: every visible tag jumps at
the same moment and stays there. A camera that moves mid-match invalidates that
camera's tracking for the match (one calibration cannot fit both framings), so
callers drop the camera for that match and calibrate on footage from after the move.

``detect_camera_move`` samples the match, tracks each tag's centre, finds each tag's
best two-segment split, and calls a move when the tags seen on both sides of a
common split time step together by >= ``move_px``. Pure apart from reading video.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class MotionConfig:
    stride_sec: float = 3.0     # one sampled frame every N seconds (~6 s per camera)
    move_px: float = 8.0        # minimum step of a tag's centre to count as a move
    min_side: int = 3           # samples needed on each side of a split
    noise_k: float = 4.0        # step must exceed noise_k x the within-segment spread
    agree_frac: float = 0.6     # share of two-sided tags that must step at the split
    single_tag_k: float = 1.5   # with only ONE two-sided tag, require move_px x this


def _best_split(times: np.ndarray, pts: np.ndarray, min_side: int):
    """(step_px, split_time, noise_px) of the best two-segment split of one tag's
    centre track, or None when the track is too short."""
    n = len(pts)
    if n < 2 * min_side:
        return None
    best = None
    for k in range(min_side, n - min_side + 1):
        a, b = pts[:k], pts[k:]
        ma, mb = np.median(a, 0), np.median(b, 0)
        step = float(np.linalg.norm(ma - mb))
        noise = max(float(np.median(np.linalg.norm(a - ma, axis=1))),
                    float(np.median(np.linalg.norm(b - mb, axis=1))))
        score = step - 3.0 * noise
        if best is None or score > best[0]:
            best = (score, step, float((times[k - 1] + times[k]) / 2.0), noise)
    return best[1:]


def analyze_tracks(tracks: dict, cfg: Optional[MotionConfig] = None) -> dict:
    """Decide from per-tag tracks ``{tag_id: [(t, u, v), ...]}`` (time-ordered) whether
    the camera moved. Returns {moved, t_s, shift_px, stepping, two_sided, per_tag}."""
    cfg = cfg or MotionConfig()
    per_tag, cands = {}, []
    for tid, rows in tracks.items():
        a = np.asarray(rows, float).reshape(-1, 3)
        a = a[np.argsort(a[:, 0])]
        r = _best_split(a[:, 0], a[:, 1:3], cfg.min_side)
        if r is None:
            continue
        step, t_split, noise = r
        per_tag[int(tid)] = {"n": len(a), "step_px": round(step, 2),
                             "t_s": round(t_split, 1), "noise_px": round(noise, 2)}
        if step >= cfg.move_px and step >= cfg.noise_k * noise + 1.0:
            cands.append(t_split)
    out = {"moved": False, "t_s": None, "shift_px": 0.0, "stepping": [],
           "two_sided": [], "per_tag": per_tag}
    if not cands:
        return out
    # Consensus split time: test each candidate time against every tag seen on both
    # sides of it; keep the time where the most tags agree.
    best = None
    for t_star in cands:
        two_sided, stepping, shifts = [], [], []
        for tid, rows in tracks.items():
            a = np.asarray(rows, float).reshape(-1, 3)
            before, after = a[a[:, 0] < t_star, 1:3], a[a[:, 0] > t_star, 1:3]
            if len(before) < cfg.min_side or len(after) < cfg.min_side:
                continue
            two_sided.append(int(tid))
            d = float(np.linalg.norm(np.median(before, 0) - np.median(after, 0)))
            if d >= cfg.move_px:
                stepping.append(int(tid))
                shifts.append(d)
        if not two_sided:
            continue
        frac = len(stepping) / len(two_sided)
        key = (frac, len(stepping))
        if best is None or key > best[0]:
            best = (key, t_star, two_sided, stepping, shifts)
    if best is None:
        return out
    (frac, n_step), t_star, two_sided, stepping, shifts = best
    if not stepping:
        return out
    shift = float(np.median(shifts))
    moved = frac >= cfg.agree_frac and (
        len(two_sided) >= 2 or shift >= cfg.move_px * cfg.single_tag_k)
    out.update(moved=bool(moved), t_s=round(float(t_star), 1), shift_px=round(shift, 1),
               stepping=stepping, two_sided=two_sided)
    return out


def detect_camera_move(video, crop, t_range, detect_fn: Callable,
                       cfg: Optional[MotionConfig] = None) -> dict:
    """Sample ``video`` every ``cfg.stride_sec`` seconds inside ``t_range`` (seconds,
    (lo, hi); hi None = end), detect tags in the camera ``crop`` and run
    ``analyze_tracks``. Adds ``samples`` (frames read) to the result."""
    cfg = cfg or MotionConfig()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = max(0, int(float(t_range[0]) * fps))
    hi = n_frames - 1 if t_range[1] is None else min(n_frames - 1, int(float(t_range[1]) * fps))
    x, y, w, h = (int(c) for c in crop)
    step = max(1, int(round(cfg.stride_sec * fps)))
    tracks: dict = {}
    samples = 0
    try:
        for f in range(lo, hi + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            samples += 1
            for tid, c in detect_fn(frame[y:y + h, x:x + w]).items():
                ctr = np.asarray(c, float).reshape(-1, 2).mean(0)
                tracks.setdefault(int(tid), []).append((f / fps, ctr[0], ctr[1]))
    finally:
        cap.release()
    res = analyze_tracks(tracks, cfg)
    res["samples"] = samples
    return res


def describe(res: dict) -> str:
    """One-line human reason for a detected move."""
    return (f"camera moved mid-match at ~{res['t_s']:.0f} s (tags {res['stepping']} "
            f"jumped ~{res['shift_px']:.0f} px)")

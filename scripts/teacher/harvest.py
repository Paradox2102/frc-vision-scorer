#!/usr/bin/env python3
"""P1 crop harvester for Claude-as-teacher distillation (design §3).

Per match: TBA roster -> video (local test_footage, prior download, or YouTube
download) -> sample at --fps -> skip frames with < 2 bumpers (pre/post-match,
crowd) -> bumper + whole-robot detection -> short IoU tracklets per detector ->
<= 6 crops per tracklet spread in time -> bumper->robot link by containment ->
deinterlaced JPEG crops + manifest.jsonl + contact sheet.

No tracker / calibration, so any season works.

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


def main_checkout(repo: Path = REPO) -> Path:
    """The main checkout (data, weights, .env live there even when this code
    runs from a git worktree)."""
    try:
        common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        p = Path(common)
        if not p.is_absolute():
            p = (repo / p).resolve()
        return p.parent
    except Exception:
        return repo


MAIN = main_checkout()
BUMPER_WEIGHTS = MAIN / "runs/train/frc_bumpers_2026cur_v3/weights/best.pt"
ROBOT_WEIGHTS = MAIN / "runs/train/hub_zone_yolo26s_2026cur_v2/weights/best.pt"
ROBOT_CLASS = 1                     # hub_zone model: 0 fuel_ball, 1 robot, 2 hub_opening
DEFAULT_OUT_ROOT = MAIN / "data/teacher/harvest"
VIDEO_ROOT = MAIN / "data/prior_years"
LOCAL_FOOTAGE = MAIN / "data/test_footage"
YTDLP_PYTHON = Path("/tmp/claude-1000/-home-nick-AI-frc-vision-scorer/"
                    "44602460-a583-4ab7-becb-4686b1d2276a/scratchpad/ytvenv/bin/python")

BUMPER_CONF = 0.40
ROBOT_CONF = 0.35
MIN_BUMPERS_PER_FRAME = 2
IOU_THR = 0.3
MAX_GAP = 1                         # samples a tracklet may skip
MAX_LEN_S = 3.0
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
                    max_len_s: float = MAX_LEN_S, same_cls: bool = True) -> List[Tracklet]:
    """Greedy frame-to-frame IoU tracklets. A tracklet extends to a detection in
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
                if d.t - tr.dets[0].t > max_len_s + 1e-9:
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
    def __init__(self, imgsz: int = 640):
        import torch
        from ultralytics import YOLO
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.imgsz = imgsz
        self.bumper = YOLO(str(BUMPER_WEIGHTS))
        self.robot = YOLO(str(ROBOT_WEIGHTS))
        self.bnames = self.bumper.names
        self.rnames = self.robot.names

    def _run(self, model, frames, conf, classes=None):
        res = model(frames, conf=conf, imgsz=self.imgsz, device=self.device,
                    classes=classes, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            out.append(list(zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy().tolist(),
                                b.cls.cpu().numpy().astype(int).tolist())))
        return out

    def bumpers(self, frames):
        return self._run(self.bumper, frames, BUMPER_CONF)

    def robots(self, frames):
        return self._run(self.robot, frames, ROBOT_CONF, classes=[ROBOT_CLASS])


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


def harvest_match(*, match_key: str, video: Path, out_root: Path, detectors: Detectors,
                  roster: Optional[Dict[str, List[str]]], fps_out: float = 2.0,
                  limit_frames: Optional[int] = None, batch: int = 8,
                  deinterlace: str = "auto") -> dict:
    import cv2
    from web.bumper_register import comb_ratio, deinterlace_crop

    event = match_key.split("_", 1)[0]
    t0 = time.time()
    vfps, nframes, W, H = video_fps(video)
    store: Dict[int, np.ndarray] = {}      # det_id -> padded raw crop
    bdets: List[Det] = []
    rdets: List[Det] = []
    n_samples = n_kept = 0

    def keep_crop(frame, box, pad) -> int:
        if min(box_wh(box)) < MIN_SIDE_PX:
            return -1
        x0, y0, x1, y1 = pad_box(box, pad, frame.shape[1], frame.shape[0])
        if x1 <= x0 or y1 <= y0:
            return -1
        store[len(store)] = frame[y0:y1, x0:x1].copy()
        return len(store) - 1

    def flush(buf):
        nonlocal n_kept
        if not buf:
            return
        bres = detectors.bumpers([f for _, _, _, f in buf])
        keep = [(s, b) for s, b in zip(buf, bres) if len(b) >= MIN_BUMPERS_PER_FRAME]
        n_kept += len(keep)
        if not keep:
            return
        rres = detectors.robots([s[3] for s, _ in keep])
        for ((si, fi, t, frame), bb), rr in zip(keep, rres):
            for box, conf, c in bb:
                d = Det(si, fi, t, tuple(box), conf, detectors.bnames[c])
                d.det_id = keep_crop(frame, d.box, PAD_BUMPER)
                bdets.append(d)
            for box, conf, c in rr:
                d = Det(si, fi, t, tuple(box), conf, detectors.rnames[c])
                d.det_id = keep_crop(frame, d.box, PAD_ROBOT)
                rdets.append(d)

    buf = []
    for s in iter_samples(video, fps_out, limit_frames):
        n_samples += 1
        buf.append(s)
        if len(buf) >= batch:
            flush(buf)
            buf = []
    flush(buf)
    t_detect = time.time() - t0

    btracks = build_tracklets(bdets, "bumper", "b")
    rtracks = build_tracklets(rdets, "robot", "r")
    b2r = link_bumpers_to_robots(btracks, rtracks)
    rlinks = robot_links(btracks, b2r)

    mdir = out_root / event / match_key
    cdir = mdir / "crops"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    sheet_b: List[Tuple[np.ndarray, str]] = []
    sheet_r: List[Tuple[np.ndarray, str]] = []
    n_deint = 0
    interlaced, idet = detect_interlaced(video, nframes / vfps if vfps else 0.0)
    # auto: woven video -> deinterlace_crop (its own comb gate decides per crop);
    # progressive video -> untouched; unknown (no ffmpeg) -> per-crop comb gate.
    do_deint = {"always": True, "never": False}.get(deinterlace, interlaced is not False)
    for tracks, detector in ((btracks, "bumper"), (rtracks, "robot")):
        for tr in tracks:
            picks = [d for d in select_crops(tr) if d.det_id >= 0]
            if detector == "bumper":
                linked, colour, extra = b2r.get(tr.tid), colour_of(tr.cls), {}
            else:
                best, colour, all_b = rlinks.get(tr.tid, (None, None, []))
                linked, extra = best, {"linked_bumper_tracklets": all_b}
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
                    tracklet_len=len(tr.dets), **extra))
                if j == len(picks) // 2:  # one tile per tracklet (its middle crop)
                    lab = f"{tr.tid} {colour or '-'} {int(min(box_wh(d.box)))}px"
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
        "width": W, "height": H, "samples": n_samples, "samples_kept": n_kept,
        "bumper_dets": len(bdets), "robot_dets": len(rdets),
        "bumper_tracklets": len(btracks), "robot_tracklets": len(rtracks),
        "bumper_crops": n_b_rows, "robot_crops": len(rows) - n_b_rows,
        "bumper_tracklets_linked": len(linked_b),
        "linked_fraction": round(len(linked_b) / len(btracks), 3) if btracks else None,
        "robot_tracklets_linked": len(rlinks),
        "deinterlaced_crops": n_deint, "video_interlaced": interlaced, "idet": idet,
        "roster": roster,
        "detect_sec": round(t_detect, 1), "total_sec": round(time.time() - t0, 1),
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
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--keep-video", action="store_true", help="keep videos downloaded by this run")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    ap.add_argument("--limit-frames", type=int, help="stop after N sampled frames (smoke runs)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--force", action="store_true", help="re-harvest matches with a manifest")
    ap.add_argument("--deinterlace", choices=("auto", "always", "never"), default="auto",
                    help="auto: deinterlace crops only when ffmpeg idet finds a woven video")
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
                              detectors=detectors, roster=roster, fps_out=args.fps,
                              limit_frames=args.limit_frames, batch=args.batch,
                              deinterlace=args.deinterlace)
        finally:
            if downloaded and not args.keep_video:
                video.unlink(missing_ok=True)
        s.update({"download_sec": round(t_dl, 1), "downloaded": downloaded,
                  "video_mb": round(video_mb, 1)})
        with open(mdir / "summary.json", "w") as f:
            json.dump(s, f, indent=1)
        results.append(s)
        print(f"[harvest] {match_key}: {s['bumper_crops']} bumper / {s['robot_crops']} robot crops, "
              f"{s['bumper_tracklets']}/{s['robot_tracklets']} tracklets, linked "
              f"{s['linked_fraction']}, kept {s['samples_kept']}/{s['samples']} samples, "
              f"{s['crop_bytes'] / 1e6:.1f} MB, dl {t_dl:.0f}s + harvest {s['total_sec']:.0f}s",
              flush=True)
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"[harvest] {len(errs)} match(es) failed: {[r['match_key'] for r in errs]}", flush=True)
    return 1 if errs and len(errs) == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())

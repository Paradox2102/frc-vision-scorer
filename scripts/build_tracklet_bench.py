"""Reconstruct short source-video tracklets around each bumper-bench crop.

For every crop in ``data/eval/bumper_crop_bench/manifest.jsonl`` whose job's
source video is still on disk, find the crop in the source frame, then follow
the same image region forward and backward for +-1.5 s so the reader bench can
show a model several moments of the same robot (``--frames multi4|video`` in
scripts/bench_bumper_readers.py).

Localisation
------------
* ``manifest.frame`` is ``self._current_frame`` at record time. The main loop
  only advances ``_current_frame`` at the END of an iteration
  (web/app.py ``self._current_frame = frame_num``), so a main-loop ``vlm_read``
  crop was cut from the NEXT processed frame: expect ``+frame_skip`` (about +7
  at 60 fps). Early-pass rows (all ``vlm_filtered`` rows, and the early
  ``vlm_read`` rows) are exact. Pre-22019bee jobs logged the early pass as
  frame 0; for those the hint is the first non-zero frame in the job's
  diagnostic file (= the first processed frame = the auto-start prompt frame).
  The source frame is searched in ``hint + [-3, +16]``; a wide search over
  ``hint + [-150, +40]`` runs when that fails, when its best frame sits on the
  window edge, or for frame-0 rows (``--no-wide`` disables it). The best match
  wins and the found offset is recorded.
* A ``vlm_read`` crop is the bumper band (or the bumper box), upscaled by an
  INTEGER factor k = max(2, ceil(384 / h0)) with INTER_LINEAR when h0 < 384,
  then JPEG q90 (web/app.py ``_bumper_crop_b64``). The candidate k values are
  the ones consistent with the stored crop's dimensions (k divides both sides
  and reproduces k). ``vlm_filtered`` crops are not upscaled (k = 1, q60).
  Each candidate template (stored crop downscaled by k, INTER_AREA) is matched
  with TM_CCOEFF_NORMED against the full frame: coarse at half resolution,
  then refined at full resolution around the coarse hit. Accept >= --loc-min.

Tracking
--------
From the located box at source frame F, step every ``round(fps / 12)`` frames
(5 at 60 fps) forward and backward up to 1.5 s. Each step searches a window of
the previous box expanded by one box size on every side, matching the previous
step's patch (template update); a direction stops when the best score drops
below --track-min (0.50), when the new patch matches the CENTRE patch below
--anchor-min (0.45; the drift guard - with template update alone, 0.45 let the
box slide onto balls, people and field ramps in about half of the moving
tracklets), when the box newly touches the frame edge, or at the video end. The
box size is fixed. ``anchor_score`` (patch vs centre patch) is recorded per
frame.

Outputs (under the bench dir)
-----------------------------
* ``tracklets/<crop_id>/<signed_offset>.jpg`` - native-resolution crops (JPEG
  q95); offsets are source frames relative to the located frame F (0 = F).
* ``tracklets.jsonl`` - one row per bench crop: {id, job_id, status, reason,
  F (manifest frame), F_src, frame_offset, search, k, loc_score, box [x,y,w,h],
  fps, step, stop_fwd, stop_bwd, frames: [{offset, file, match_score,
  anchor_score, sharpness}]}. ``status`` is ``ok`` or the failure reason
  (``no_video``, ``loc_failed``, ``read_failed``).
* ``tracklets/_sheets/<crop_id>.png`` - QA strip per tracklet (gitignored).

Usage
-----
    .venv/bin/python scripts/build_tracklet_bench.py               # all crops with video
    .venv/bin/python scripts/build_tracklet_bench.py --ids 2026cur_qm10_f01027_r005 --workers 1
    .venv/bin/python scripts/build_tracklet_bench.py --track-min 0.6   # tighter drift stop
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DB_PATH = REPO / "frc_scorer.db"
UPLOADS_DIR = REPO / "web" / "uploads"
BENCH_DIR = REPO / "data" / "eval" / "bumper_crop_bench"

UPSCALE_TARGET_H = 384     # web/app.py _bumper_crop_b64: upscale below this height
MAX_K = 16
NEAR_OFFSETS = (-3, 16)    # inclusive search range around the hint frame
WIDE_OFFSETS = (-150, 40)  # async reads record a later _current_frame (lags up to ~1 s seen)
LOC_MIN = 0.8
TRACK_MIN = 0.50           # 0.45 drifted (QA 2026-10-12); see --track-min
ANCHOR_MIN = 0.45          # drift guard vs the centre patch; see --anchor-min
WINDOW_S = 1.5
SAMPLE_FPS = 12.0
COARSE = 0.5               # coarse localisation scale
COARSE_MIN_SIDE = 12       # templates smaller than this (at coarse scale) match at full res
REFINE_PAD = 10            # px around the coarse hit searched at full resolution
TOP_KS_REFINED = 2


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def candidate_ks(w: int, h: int, stratum: str) -> list[int]:
    """Upscale factors consistent with a stored crop of w x h.

    vlm_filtered crops are never upscaled (k=1). A read crop of native height
    h0 < 384 was resized by k = max(2, ceil(384 / h0)), so both stored sides are
    multiples of k and h / k reproduces k; a native crop >= 384 tall is stored
    as-is (k=1)."""
    if stratum == "filtered":
        return [1]
    ks = [k for k in range(2, MAX_K + 1)
          if h % k == 0 and w % k == 0
          and max(2, math.ceil(UPSCALE_TARGET_H / (h // k))) == k]
    if h >= UPSCALE_TARGET_H:
        ks.append(1)
    return ks or list(range(1, MAX_K + 1))


def sample_step(fps: float, sample_fps: float = SAMPLE_FPS) -> int:
    return max(1, int(round((fps or 60.0) / sample_fps)))


def n_steps(fps: float, step: int, window_s: float = WINDOW_S) -> int:
    return max(0, int(math.floor(window_s * (fps or 60.0) / step + 1e-9)))


def search_window(box: tuple, frame_wh: tuple, expand: float = 1.0) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the previous box grown by ``expand`` box sizes per side,
    clipped to the frame."""
    x, y, w, h = box
    W, H = frame_wh
    dx, dy = int(round(w * expand)), int(round(h * expand))
    return max(0, x - dx), max(0, y - dy), min(W, x + w + dx), min(H, y + h + dy)


def touches_edge(box: tuple, frame_wh: tuple) -> bool:
    x, y, w, h = box
    W, H = frame_wh
    return x <= 0 or y <= 0 or x + w >= W or y + h >= H


def first_nonzero_frame(diag_path: Path) -> Optional[int]:
    """First non-zero ``frame`` in a diagnostic jsonl (the first processed frame
    of the main loop; frame-0 rows predate the early-pass _current_frame fix)."""
    try:
        with open(diag_path, "rb") as fh:
            for line in fh:
                i = line.find(b'"frame":')
                if i < 0:
                    continue
                j = i + len(b'"frame":')
                num = line[j:j + 16].split(b",")[0].strip()
                try:
                    f = int(num)
                except ValueError:
                    continue
                if f > 0:
                    return f
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- #
# Image ops (cv2)
# --------------------------------------------------------------------------- #

def _cv2():
    import cv2
    return cv2


def downscale(img, k: int):
    cv2 = _cv2()
    if k == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)


def best_match(region, tmpl) -> tuple[float, tuple[int, int]]:
    cv2 = _cv2()
    if region.shape[0] < tmpl.shape[0] or region.shape[1] < tmpl.shape[1]:
        return -1.0, (0, 0)
    m = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    import numpy as np
    m = np.nan_to_num(m, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, mx, _, loc = cv2.minMaxLoc(m)
    return float(mx), (int(loc[0]), int(loc[1]))


def patch_score(a, b) -> float:
    """TM_CCOEFF_NORMED of two same-size patches."""
    return best_match(a, b)[0] if a.shape == b.shape else -1.0


def sharpness(img) -> float:
    cv2 = _cv2()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def read_range(cap, start: int, end: int, keep=None) -> dict:
    """Seek to ``start`` and read sequentially through ``end`` (inclusive);
    returns {frame_idx: frame} for indices accepted by ``keep``."""
    cv2 = _cv2()
    start = max(0, start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = {}
    for idx in range(start, end + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if keep is None or keep(idx):
            out[idx] = fr
    return out


def localize(crop, ks: list[int], frames: dict) -> Optional[dict]:
    """Best (score, frame_idx, k, box) of the crop over ``frames``.

    Coarse pass at COARSE scale over every (frame, k); the TOP_KS_REFINED best
    k values are refined at full resolution around each frame's coarse hit."""
    cv2 = _cv2()
    if not frames:
        return None
    tmpls = {k: downscale(crop, k) for k in ks}
    tmpls = {k: t for k, t in tmpls.items() if t.shape[0] >= 4 and t.shape[1] >= 4}
    small = {i: cv2.resize(f, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA)
             for i, f in frames.items()}
    coarse: dict[int, dict[int, tuple[float, tuple[int, int]]]] = defaultdict(dict)
    for k, t in tmpls.items():
        th, tw = t.shape[:2]
        if min(th, tw) * COARSE >= COARSE_MIN_SIDE:
            ts = cv2.resize(t, (max(1, int(round(tw * COARSE))), max(1, int(round(th * COARSE)))),
                            interpolation=cv2.INTER_AREA)
            for i, sf in small.items():
                s, (x, y) = best_match(sf, ts)
                coarse[k][i] = (s, (int(round(x / COARSE)), int(round(y / COARSE))))
        else:
            for i, f in frames.items():
                coarse[k][i] = best_match(f, t)
    if not coarse:
        return None
    k_rank = sorted(coarse, key=lambda k: max(s for s, _ in coarse[k].values()), reverse=True)
    best = None
    for k in k_rank[:TOP_KS_REFINED]:
        t = tmpls[k]
        th, tw = t.shape[:2]
        for i, f in frames.items():
            _s, (cx, cy) = coarse[k][i]
            H, W = f.shape[:2]
            x0, y0 = max(0, cx - REFINE_PAD), max(0, cy - REFINE_PAD)
            x1, y1 = min(W, cx + tw + REFINE_PAD), min(H, cy + th + REFINE_PAD)
            s, (x, y) = best_match(f[y0:y1, x0:x1], t)
            cand = (s, i, k, (x0 + x, y0 + y, tw, th))
            if best is None or cand[0] > best[0] + 1e-6:
                best = cand
    if best is None:
        return None
    s, i, k, box = best
    return {"score": s, "frame_idx": i, "k": k, "box": box}


def track(frames: dict, center: int, box: tuple, step: int, steps: int,
          direction: int, track_min: float, anchor_min: float = -1.0) -> tuple[list[dict], str]:
    """Follow ``box`` from ``frames[center]`` in ``direction`` (+1/-1).

    Stops when the step-to-step match drops below ``track_min`` or the new
    patch's match against the centre patch drops below ``anchor_min`` (drift).
    Returns ([{idx, box, match_score, anchor_score, patch}], stop_reason)."""
    f0 = frames[center]
    H, W = f0.shape[:2]
    x, y, w, h = box
    anchor = f0[y:y + h, x:x + w]
    prev_box, prev_patch = box, anchor
    started_on_edge = touches_edge(box, (W, H))
    out: list[dict] = []
    for n in range(1, steps + 1):
        idx = center + direction * n * step
        fr = frames.get(idx)
        if fr is None:
            return out, "video_end"
        x0, y0, x1, y1 = search_window(prev_box, (W, H))
        s, (lx, ly) = best_match(fr[y0:y1, x0:x1], prev_patch)
        if not math.isfinite(s) or s < track_min:
            return out, "low_score"
        nb = (x0 + lx, y0 + ly, w, h)
        if touches_edge(nb, (W, H)) and not started_on_edge:
            return out, "edge"
        patch = fr[nb[1]:nb[1] + h, nb[0]:nb[0] + w]
        a = patch_score(patch, anchor)
        if a < anchor_min:
            return out, "anchor"
        out.append({"idx": idx, "box": nb, "match_score": s,
                    "anchor_score": a, "patch": patch})
        prev_box, prev_patch = nb, patch
    return out, "window_end"


# --------------------------------------------------------------------------- #
# QA sheet
# --------------------------------------------------------------------------- #

def render_sheet(row: dict, manifest_row: dict, bench_dir: Path, out_path: Path,
                 cols: int = 8, cell_w: int = 200) -> None:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=12)
    w, h = row["box"][2], row["box"][3]
    img_h = max(24, min(150, int(round(cell_w * h / max(1, w)))))
    cap_h = 16
    cell_h = img_h + cap_h + 6

    def cell(img_path: Path, caption: str, border) -> "Image.Image":
        c = Image.new("RGB", (cell_w, cell_h), (235, 235, 235))
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            sc = min((cell_w - 6) / im.width, img_h / im.height)
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
        c.paste(im, ((cell_w - im.width) // 2, 3 + (img_h - im.height) // 2))
        d = ImageDraw.Draw(c)
        d.text((4, img_h + 6), caption, fill=(20, 20, 20), font=font)
        d.rectangle([0, 0, cell_w - 1, cell_h - 1], outline=border, width=2 if border != (150, 150, 150) else 1)
        return c

    cells = [cell(bench_dir / manifest_row["file"], "bench crop", (0, 0, 200))]
    for fr in row["frames"]:
        ok = fr["offset"] == 0
        cap = f"{fr['offset']:+d} m{fr['match_score']:.2f} a{fr['anchor_score']:.2f}"
        cells.append(cell(bench_dir / fr["file"], cap, (0, 160, 0) if ok else (150, 150, 150)))
    nrows = (len(cells) + cols - 1) // cols
    head_h = 34
    sheet = Image.new("RGB", (cell_w * cols, head_h + cell_h * nrows), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    lab = manifest_row.get("label_status") or "unlabelled"
    lab_txt = f"{lab} {manifest_row.get('label_team') or ''} {manifest_row.get('label_alliance') or ''}".strip()
    d.text((6, 3), f"{row['id']}  label: {lab_txt}  stratum: {manifest_row.get('stratum')}",
           fill=(0, 0, 0), font=font)
    d.text((6, 18), (f"F={row['F']} F_src={row['F_src']} (offset {row['frame_offset']:+d}, "
                     f"{row['search']})  k={row['k']} loc={row['loc_score']:.3f} box={row['box']} "
                     f"step={row['step']}  stop bwd={row['stop_bwd']} fwd={row['stop_fwd']}  "
                     f"(m=match vs prev step, a=vs centre)"),
           fill=(60, 60, 60), font=font)
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * cell_w, head_h + (i // cols) * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


# --------------------------------------------------------------------------- #
# Per-video worker
# --------------------------------------------------------------------------- #

def _fail(r: dict, status: str, reason: str, **extra) -> dict:
    return {"id": r["id"], "job_id": r["job_id"], "status": status, "reason": reason,
            "F": r["frame"], "frames": [], **extra}


def process_video(video: str, items: list[dict], bench_dir: str, opts: dict) -> list[dict]:
    """Localise + track every crop of one source video. ``items`` are manifest
    rows plus ``hint`` (search centre frame)."""
    cv2 = _cv2()
    cv2.setNumThreads(opts.get("threads", 2))
    bench = Path(bench_dir)
    out_rows = []
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return [_fail(r, "read_failed", "VideoCapture could not open the source video") for r in items]
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = sample_step(fps)
    steps = n_steps(fps, step, opts["window_s"])
    for r in items:
        crop = cv2.imread(str(bench / r["file"]))
        if crop is None:
            out_rows.append(_fail(r, "read_failed", "bench crop unreadable"))
            continue
        ch, cw = crop.shape[:2]
        ks = candidate_ks(cw, ch, r["stratum"])
        hint = r["hint"]
        loc, search = None, "near"
        lo, hi = NEAR_OFFSETS
        frames = read_range(cap, hint + lo, hint + hi)
        if not frames:
            out_rows.append(_fail(r, "read_failed", f"no frames decoded at {hint + lo}"))
            continue
        loc = localize(crop, ks, frames)
        del frames
        at_edge = loc is not None and (loc["frame_idx"] - hint) in (lo, hi)
        if opts["wide"] and (loc is None or loc["score"] < opts["loc_min"] or at_edge
                             or r["frame"] == 0):
            wlo, whi = WIDE_OFFSETS
            near = loc
            wide = None
            for c0 in range(hint + wlo, hint + whi + 1, 40):  # chunked: bounded memory
                frames = read_range(cap, c0, min(c0 + 39, hint + whi))
                cand = localize(crop, ks, frames)
                del frames
                if cand is not None and (wide is None or cand["score"] > wide["score"]):
                    wide = cand
            if wide is not None and (near is None or wide["score"] > near["score"]):
                loc, search = wide, "wide"
        if loc is None or loc["score"] < opts["loc_min"]:
            out_rows.append(_fail(
                r, "loc_failed",
                f"best TM_CCOEFF_NORMED {loc['score']:.3f} < {opts['loc_min']}" if loc else "no match",
                loc_score=round(loc["score"], 4) if loc else None,
                k=loc["k"] if loc else None, ks=ks,
                F_src=loc["frame_idx"] if loc else None,
                frame_offset=(loc["frame_idx"] - r["frame"]) if loc else None))
            continue
        F_src = loc["frame_idx"]
        span = steps * step
        grid = lambda i: (i - F_src) % step == 0  # noqa: E731
        frames = read_range(cap, F_src - span, F_src + span, keep=grid)
        if F_src not in frames:
            out_rows.append(_fail(r, "read_failed", f"centre frame {F_src} not re-decoded"))
            continue
        x, y, w, h = loc["box"]
        # Re-seek determinism check: the centre patch must match the crop again.
        recheck = patch_score(frames[F_src][y:y + h, x:x + w], downscale(crop, loc["k"]))
        fwd, stop_fwd = track(frames, F_src, loc["box"], step, steps, +1,
                              opts["track_min"], opts["anchor_min"])
        bwd, stop_bwd = track(frames, F_src, loc["box"], step, steps, -1,
                              opts["track_min"], opts["anchor_min"])
        centre = {"idx": F_src, "box": loc["box"], "match_score": loc["score"],
                  "anchor_score": 1.0, "patch": frames[F_src][y:y + h, x:x + w]}
        seq = sorted(bwd + [centre] + fwd, key=lambda d: d["idx"])
        tdir = bench / "tracklets" / r["id"]
        if tdir.exists():
            shutil.rmtree(tdir)
        tdir.mkdir(parents=True)
        recs = []
        for d in seq:
            off = d["idx"] - F_src
            fname = f"{off}.jpg"
            cv2.imwrite(str(tdir / fname), d["patch"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            recs.append({"offset": off, "file": f"tracklets/{r['id']}/{fname}",
                         "match_score": round(d["match_score"], 4),
                         "anchor_score": round(d["anchor_score"], 4),
                         "sharpness": round(sharpness(d["patch"]), 2),
                         "box": list(d["box"])})
        row = {"id": r["id"], "job_id": r["job_id"], "status": "ok", "reason": None,
               "F": r["frame"], "hint": hint, "F_src": F_src, "frame_offset": F_src - r["frame"],
               "search": search, "k": loc["k"], "ks": ks, "loc_score": round(loc["score"], 4),
               "recheck_score": round(recheck, 4), "box": [x, y, w, h], "fps": round(fps, 4),
               "n_video_frames": n_total, "step": step, "steps": steps,
               "stop_fwd": stop_fwd, "stop_bwd": stop_bwd,
               "track_min": opts["track_min"], "anchor_min": opts["anchor_min"], "frames": recs}
        out_rows.append(row)
        if opts["sheets"]:
            render_sheet(row, r, bench, bench / "tracklets" / "_sheets" / f"{r['id']}.png")
    cap.release()
    return out_rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def source_videos(db_path: Path, job_ids: set) -> dict[str, Optional[str]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = {}
        for j in job_ids:
            row = con.execute("SELECT source_video_path FROM replays WHERE id = ?", (j,)).fetchone()
            out[j] = row[0] if row and row[0] else None
        return out
    finally:
        con.close()


def dir_size(path: Path, skip: str = "_sheets") -> int:
    return sum(p.stat().st_size for p in path.rglob("*")
               if p.is_file() and skip not in p.parts)


def summarize(rows: list[dict]) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [f"crops: {len(rows)}  tracklets ok: {len(ok)}"]
    lines.append("status: " + ", ".join(f"{k}={v}" for k, v in Counter(r["status"] for r in rows).items()))
    fail_reasons = Counter((r["status"], (r.get("reason") or "")[:40]) for r in rows if r["status"] != "ok")
    for (st, rs), n in fail_reasons.most_common():
        lines.append(f"  {st}: {rs} x{n}")
    if ok:
        sc = sorted(r["loc_score"] for r in ok)
        lines.append("loc_score (ok): min %.3f p10 %.3f median %.3f" % (
            sc[0], sc[len(sc) // 10], statistics.median(sc)))
        lf = sorted(r["loc_score"] for r in rows if r["status"] == "loc_failed" and r.get("loc_score") is not None)
        if lf:
            lines.append("loc_score (failed): " + " ".join(f"{s:.2f}" for s in lf))
        nf = [len(r["frames"]) for r in ok]
        lines.append(f"frames/tracklet: median {statistics.median(nf)}  min {min(nf)}  max {max(nf)}  "
                     f">=3: {sum(1 for n in nf if n >= 3)}")
        lines.append("frame_offset (F_src - manifest frame): " + ", ".join(
            f"{k:+d}:{v}" for k, v in sorted(Counter(r["frame_offset"] for r in ok).items())))
        lines.append("search: " + ", ".join(f"{k}={v}" for k, v in Counter(r["search"] for r in ok).items()))
        lines.append("stop fwd: " + ", ".join(f"{k}={v}" for k, v in Counter(r["stop_fwd"] for r in ok).items())
                     + " | bwd: " + ", ".join(f"{k}={v}" for k, v in Counter(r["stop_bwd"] for r in ok).items()))
        lines.append("k: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["k"] for r in ok).items())))
        rc = [r["recheck_score"] for r in ok]
        lines.append(f"re-seek centre recheck: min {min(rc):.3f} (should be ~1.0 = deterministic seek)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH_DIR))
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--uploads", default=str(UPLOADS_DIR))
    ap.add_argument("--ids", default=None, help="comma list of crop ids (default: all)")
    ap.add_argument("--loc-min", type=float, default=LOC_MIN)
    ap.add_argument("--track-min", type=float, default=TRACK_MIN,
                    help="stop a direction when the step-to-step match drops below this")
    ap.add_argument("--anchor-min", type=float, default=ANCHOR_MIN,
                    help="stop a direction when the patch vs the centre patch drops below "
                         "this (drift guard; -1 disables)")
    ap.add_argument("--window-s", type=float, default=WINDOW_S)
    ap.add_argument("--no-wide", action="store_true", help="skip the [-150, +40] fallback search")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    bench = Path(args.bench)
    rows = [json.loads(l) for l in (bench / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    videos = source_videos(Path(args.db), {r["job_id"] for r in rows})
    out_rows: list[dict] = []
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = videos.get(r["job_id"])
        if not v or not os.path.exists(v):
            out_rows.append(_fail(r, "no_video", "source video not on disk" if v else "no replays row/path"))
            continue
        hint = r["frame"]
        if hint == 0:
            hint = first_nonzero_frame(Path(args.uploads) / f"diagnostic_{r['job_id']}.jsonl") or 0
        by_video[v].append({**r, "hint": hint})

    opts = {"loc_min": args.loc_min, "track_min": args.track_min, "anchor_min": args.anchor_min,
            "window_s": args.window_s,
            "wide": not args.no_wide, "sheets": not args.no_sheets, "threads": 2}
    if not args.ids:
        sheets = bench / "tracklets" / "_sheets"
        if sheets.exists():
            shutil.rmtree(sheets)
    t0 = time.time()
    n_items = sum(len(v) for v in by_video.values())
    print(f"{len(rows)} crops; {n_items} with video across {len(by_video)} videos; "
          f"{len(rows) - n_items} without", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(process_video, v, items, str(bench), opts): v
                for v, items in by_video.items()}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            out_rows.extend(res)
            done += len(res)
            print(f"  {done}/{n_items} localised/tracked ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)

    order = {r["id"]: i for i, r in enumerate(rows)}
    out_rows.sort(key=lambda r: order[r["id"]])
    path = bench / "tracklets.jsonl"
    if args.ids and path.exists():  # partial rebuild: merge into the existing file
        old = {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}
        old.update({r["id"]: r for r in out_rows})
        full = [json.loads(l) for l in (bench / "manifest.jsonl").read_text().splitlines() if l.strip()]
        out_rows_all = [old[r["id"]] for r in full if r["id"] in old]
    else:
        out_rows_all = out_rows
    # Drop stale crop dirs of ids that no longer have a tracklet.
    keep = {r["id"] for r in out_rows_all if r["status"] == "ok"}
    tdir = bench / "tracklets"
    if tdir.exists() and not args.ids:
        for d in tdir.iterdir():
            if d.is_dir() and d.name != "_sheets" and d.name not in keep:
                shutil.rmtree(d)
    path.write_text("".join(json.dumps(r) + "\n" for r in out_rows_all))
    print(summarize(out_rows))
    size = dir_size(tdir) if tdir.exists() else 0
    print(f"tracklet crops on disk: {size / 1e6:.1f} MB (excl. _sheets); wrote {path}")


if __name__ == "__main__":
    main()

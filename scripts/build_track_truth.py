"""Track-derived truth for the bumper-crop bench (independent of Claude labels).

The bench's ``label_*`` fields were made by Claude looking at each crop, so they
are circular for scoring Claude-family reader arms. This script derives, per
crop, which team's robot the crop shows from evidence OTHER than this crop's
pixels: hand ground truth (2026cur_qm8 only) and reader votes on the same
continuous track at other moments.

Evidence ladder (per crop)
--------------------------
1. ``gt`` (2026cur_qm8): the crop's source-video box (``tracklets.jsonl``) is
   matched to the hand-annotated robots (data/eval/ground_truth/2026cur_qm8) at
   that frame. Main-view crops compare in composite pixels against the
   annotator's ``uv``; corner-view crops against the GT ``field_xy`` projected
   into that corner camera. Nearest robot within ``GT_GATE_W`` box widths and
   clearly nearer than the runner-up (``GT_MARGIN``) wins. GT overrides votes.
2. ``track_votes`` (``segment_kind: slot``): the crop is located in its source
   frame (``tracklets.jsonl`` F_src/box), its bumper bottom-centre projected to
   the field with the calibration in force when the job ran, and matched to the
   job's replay slot at that frame. Usually the distance is 0.00 m: the
   pipeline's slot position IS that box's projection. The slot's track is cut
   into continuous runs at gaps and jumps (the same rule and constants as the
   window-label stage, web/field/window_labels.py) and the run holding the crop,
   clipped to +-5 s (widened to +-10 s only when 5 s has too few votes; see
   ``cascade``), is the segment. Every reader read (any job run
   on a byte-identical video) whose team matched the roster is template-located
   in the source video; a read votes when it lands in the segment, more than
   ``EXCLUDE_S`` from the crop frame, on that slot (nearest slot within
   ``SLOT_GATE_M`` and ``SLOT_MARGIN_M`` nearer than the next one).
3. ``track_votes`` (``segment_kind: identity``): when the crop can't be located
   (no source video: all 2026hop matches; or tracklet localisation failed), the
   job's own reads of the same ``robot_id`` (an identity id) are used, cut at
   activity gaps > ``IDENT_GAP_S`` and clipped the same way. Weaker:
   identities re-associate every frame with no position log, so a segment can
   span two robots. The purity rule is the only guard.

Decision rule (both kinds; validated with --validate-gt on 2026cur_qm8, where
403 GT-identified reads used as pseudo-crops gave 174 decisions, 0 wrong): votes are deduplicated to one per (source frame,
team) so N jobs reading the same frame count once. The top team needs
>= ``MIN_VOTES`` distinct frames, spanning >= ``MIN_SPAN_S``, and
>= ``MIN_SHARE`` of all deduplicated votes. Otherwise ``unknown``. Votes can't
tell a robot from field equipment: a slot that sits on a ramp gets no team
reads and stays unknown, but a slot that moved onto a robot would vote for it.

Readers: every bench job predates Jev and Clef and the vlm_model knob
(8318f2d7, 2026-07-22), so every vote is from the default local Qwen3-VL-4B
reader. ``readers`` records this per vote source; a row carrying jev/clef
probability fields would be tagged accordingly.

Usage
-----
    .venv/bin/python scripts/build_track_truth.py              # writes track_truth.jsonl
    .venv/bin/python scripts/build_track_truth.py --workers 8 --validate-gt
    .venv/bin/python scripts/build_track_truth.py --ids 2026cur_qm8_f01630_r005
    .venv/bin/python scripts/build_track_truth.py --exclude-s 0.5 --out results/tt_ex05.jsonl

Localisations are cached in ``results/track_truth_loc_cache.jsonl`` (gitignored).
"""
from __future__ import annotations

import argparse
import base64
import bisect
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from web.field.window_labels import GAP_FRAME_PERIODS, JUMP_SPEED_M_S  # noqa: E402

DB_PATH = REPO / "frc_scorer.db"
UPLOADS_DIR = REPO / "web" / "uploads"
REPLAYS_DIR = REPO / "web" / "replays"
BENCH_DIR = REPO / "data" / "eval" / "bumper_crop_bench"
GT_DIR = REPO / "data" / "eval" / "ground_truth"
OUT_NAME = "track_truth.jsonl"
CACHE_NAME = "results/track_truth_loc_cache.jsonl"

DEFAULT_FPS = 60.0
EXCLUDE_S = 1.0          # reads this close to the crop frame never vote
MIN_VOTES = 3            # distinct frames for the top team
MIN_SPAN_S = 1.0         # top team's voting frames must span this much
MIN_SHARE = 0.75         # top team's share of deduplicated votes
HALF_SPANS = (5.0, 10.0)  # segment clipped to crop +- 5 s; widened to 10 s only for lack of votes
IDENT_GAP_S = 2.0        # identity fallback: cut at activity gaps longer than this
SLOT_GATE_M = 1.0        # crop / read -> slot association radius (main view)
SLOT_MARGIN_M = 0.5      # ... and must beat the next slot by this much
# Corner-view boxes land ~1-2 m from the fused slot position (main-view boxes
# land at 0.00-0.2 m: the slot position IS their projection), so corners get a
# wider gate and a proportionally wider margin.
CORNER_GATE_M = 2.0
CORNER_MARGIN_M = 1.0


def gates_for(cam: Optional[str]) -> tuple[float, float]:
    return (SLOT_GATE_M, SLOT_MARGIN_M) if cam in (None, "main") else (CORNER_GATE_M, CORNER_MARGIN_M)
LOC_MIN = 0.8            # template-match score to accept a read localisation
NEAR = (-2, 9)           # main-loop read: crop cut from the NEXT processed frame (+frame_skip)
EARLY = (-3, 100)        # frame-0 (early-pass) rows: hint = first non-zero frame
GT_GATE_W = 1.0          # GT match radius in crop-box widths (main view: annotator uv) ...
GT_GATE_W_CORNER = 3.0   # corner views: GT field_xy re-projected, so looser
GT_GATE_MIN_PX = 25.0    # ... but at least this many pixels
GT_MARGIN = 1.6          # runner-up GT robot must be this many times farther
GT_INTERP_MAX = 30       # max GT keyframe gap (frames) to interpolate across
GT_CLICK_Z = 0.1         # GT annotator click height (doc click_height_m)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def slot_tracks(frames: list) -> tuple[dict, float]:
    """Replay frames -> ({slot_key: [(frame_num, x, y), ...]}, median frame step).

    A slot label repeated within one frame gets a ``#2`` suffix (as in
    web/field/gt_scoring.replay_from_meta)."""
    tracks: dict = defaultdict(list)
    nums = []
    for f in frames:
        fn = f.get("frame_num")
        if fn is None:
            continue
        nums.append(int(fn))
        seen: Counter = Counter()
        for r in f.get("robots") or []:
            pos = r.get("field_pos")
            if not r.get("slot") or not pos or len(pos) < 2 or pos[0] is None or pos[1] is None:
                continue
            seen[r["slot"]] += 1
            key = r["slot"] if seen[r["slot"]] == 1 else f"{r['slot']}#{seen[r['slot']]}"
            tracks[key].append((int(fn), float(pos[0]), float(pos[1])))
    steps = sorted(b - a for a, b in zip(nums, nums[1:]) if b > a)
    spacing = float(steps[len(steps) // 2]) if steps else 1.0
    return dict(tracks), spacing


def cut_runs(seq: list, spacing: float, fps: float) -> list:
    """Split one slot's [(frame, x, y)] into continuous runs: a new run starts
    after a gap > GAP_FRAME_PERIODS processed frames or a jump faster than
    JUMP_SPEED_M_S (web/field/window_labels.build_windows)."""
    if not seq:
        return []
    runs = [[seq[0]]]
    for a, b in zip(seq, seq[1:]):
        dt_f = b[0] - a[0]
        dt_s = dt_f / fps
        dist = math.hypot(b[1] - a[1], b[2] - a[2])
        if dt_f > GAP_FRAME_PERIODS * spacing or (dt_s > 0 and dist / dt_s > JUMP_SPEED_M_S):
            runs.append([b])
        else:
            runs[-1].append(b)
    return runs


def run_at(runs: list, frame: float, spacing: float) -> Optional[list]:
    """The run covering ``frame`` (its sample span widened by half a spacing)."""
    for run in runs:
        if run[0][0] - spacing / 2 <= frame <= run[-1][0] + spacing / 2:
            return run
    return None


def positions_at(tracks: dict, frame: float, spacing: float) -> dict:
    """{slot: (x, y)} at ``frame``: linear interpolation when the slot has
    samples on both sides within 2.5 spacings, else the nearest sample within
    half a spacing."""
    out = {}
    for slot, seq in tracks.items():
        fr = [s[0] for s in seq]
        i = bisect.bisect_left(fr, frame)
        if i < len(seq) and abs(seq[i][0] - frame) < 1e-6:
            out[slot] = (seq[i][1], seq[i][2])
            continue
        lo = seq[i - 1] if i > 0 else None
        hi = seq[i] if i < len(seq) else None
        if lo and hi and hi[0] - lo[0] <= 2.5 * spacing:
            w = (frame - lo[0]) / (hi[0] - lo[0])
            out[slot] = (lo[1] + (hi[1] - lo[1]) * w, lo[2] + (hi[2] - lo[2]) * w)
        else:
            near = [s for s in (lo, hi) if s and abs(s[0] - frame) <= spacing / 2 + 1e-6]
            if near:
                s = min(near, key=lambda s: abs(s[0] - frame))
                out[slot] = (s[1], s[2])
    return out


def associate(xy, positions: dict, gate: float = SLOT_GATE_M,
              margin: float = SLOT_MARGIN_M) -> tuple[Optional[str], Optional[float]]:
    """Nearest slot to ``xy`` if within ``gate`` and ``margin`` nearer than the
    runner-up. Returns (slot or None, nearest distance or None)."""
    if xy is None or not positions:
        return None, None
    ds = sorted((math.hypot(p[0] - xy[0], p[1] - xy[1]), s) for s, p in positions.items())
    d0, s0 = ds[0]
    if d0 > gate:
        return None, d0
    if len(ds) > 1 and ds[1][0] - d0 < margin:
        return None, d0
    return s0, d0


def clip_segment(run: list, center: float, half_span_f: float) -> tuple[int, int]:
    return (int(max(run[0][0], center - half_span_f)), int(min(run[-1][0], center + half_span_f)))


def cascade(spans, decide_at):
    """Decide on the narrowest span first; widen only while the reason is too
    little evidence (no votes, too few, too short). An impure (share) or tied
    segment stops the cascade: widening can't make a mixed segment cleaner."""
    dec, hs = None, None
    for hs in spans:
        dec = decide_at(hs)
        if dec["team"] or dec["reason"] == "tie" or str(dec["reason"]).startswith("share"):
            break
    return dec, hs


def activity_segment(frames: list, center: int, gap_f: float, half_span_f: float) -> tuple[int, int]:
    """Identity fallback: grow from ``center`` over sorted activity frames while
    consecutive gaps are <= gap_f, staying within +-half_span_f."""
    a = sorted(set(frames) | {center})
    i = a.index(center)
    lo = hi = i
    while lo > 0 and a[lo] - a[lo - 1] <= gap_f and center - a[lo - 1] <= half_span_f:
        lo -= 1
    while hi < len(a) - 1 and a[hi + 1] - a[hi] <= gap_f and a[hi + 1] - center <= half_span_f:
        hi += 1
    return a[lo], a[hi]


def decide(votes: list, center: float, fps: float, *, exclude_s: float = EXCLUDE_S,
           min_votes: int = MIN_VOTES, min_span_s: float = MIN_SPAN_S,
           min_share: float = MIN_SHARE) -> dict:
    """``votes``: [(source_frame, team)], already restricted to the segment.

    Drops votes within ``exclude_s`` of ``center``, dedupes to one vote per
    (frame, team), then applies the truth rule. Returns
    {team, votes {team: n}, n_frames, span_s, share, reason}."""
    ex = exclude_s * fps
    kept = sorted({(int(f), str(t)) for f, t in votes if t and abs(f - center) > ex})
    counts = Counter(t for _, t in kept)
    out = {"team": None, "votes": dict(counts.most_common()), "n_frames": len({f for f, _ in kept}),
           "span_s": None, "share": None, "reason": None}
    if not counts:
        out["reason"] = "no_votes"
        return out
    team, n = counts.most_common(1)[0]
    if len(counts) > 1 and counts.most_common(2)[1][1] == n:
        out["reason"] = "tie"
        return out
    frames = [f for f, t in kept if t == team]
    span = (max(frames) - min(frames)) / fps
    share = n / sum(counts.values())
    out["span_s"], out["share"] = round(span, 2), round(share, 3)
    if n < min_votes:
        out["reason"] = f"top_votes<{min_votes}"
    elif span < min_span_s:
        out["reason"] = f"span<{min_span_s}s"
    elif share < min_share:
        out["reason"] = f"share<{min_share}"
    else:
        out["team"] = team
        out["reason"] = "ok"
    return out


def interp_keyframes(kfs: list, frame: float, key: str, max_gap: float = GT_INTERP_MAX):
    """Linear interpolation of a 2-vector keyframe field (``uv``/``field_xy``) at
    ``frame`` over positioned keyframes sorted by frame; None outside or across
    a gap > max_gap. Returns (value, state)."""
    pts = [k for k in kfs if k.get("state") in ("visible", "estimated") and k.get(key)]
    fr = [k["frame"] for k in pts]
    i = bisect.bisect_left(fr, frame)
    if i < len(pts) and abs(pts[i]["frame"] - frame) < 1e-6:
        return tuple(pts[i][key]), pts[i]["state"]
    if i == 0 or i >= len(pts):
        return None, None
    a, b = pts[i - 1], pts[i]
    if b["frame"] - a["frame"] > max_gap:
        return None, None
    w = (frame - a["frame"]) / (b["frame"] - a["frame"])
    v = (a[key][0] + (b[key][0] - a[key][0]) * w, a[key][1] + (b[key][1] - a[key][1]) * w)
    state = "visible" if a["state"] == b["state"] == "visible" else "estimated"
    return v, state


def gt_pick(dists: dict, box_w: float, gate_w: float = GT_GATE_W,
            gate_min_px: float = GT_GATE_MIN_PX, margin: float = GT_MARGIN) -> Optional[str]:
    """{team: pixel distance} -> the GT team the crop shows, or None."""
    if not dists:
        return None
    ranked = sorted(dists.items(), key=lambda kv: kv[1])
    team, d0 = ranked[0]
    if d0 > max(gate_w * box_w, gate_min_px):
        return None
    if len(ranked) > 1 and ranked[1][1] < margin * max(d0, 1.0):
        return None
    return team


VLM_MODEL_KNOB_AT = "2026-07-22T03:00"   # 8318f2d7: per-job vlm_model knob


def gt_team_at(docs: list, frame: float, box, views: list) -> tuple[Optional[str], dict]:
    """(GT team shown by ``box`` at ``frame`` or None, {team: px distance})."""
    dists = gt_dists(docs, frame, box, views)
    x, y, w, h = box
    hit = view_of(views, (x + w / 2.0, y + h / 2.0))
    gate_w = GT_GATE_W if hit is None or hit[0] == "main" else GT_GATE_W_CORNER
    return gt_pick(dists, w, gate_w=gate_w), dists


def reader_tag(row: dict, job_created_at: str = "") -> str:
    """Which reader produced a vlm_read row. Jobs don't record their reader
    config, so non-Jev/Clef rows are tagged by date: before the vlm_model knob
    every job used the default Qwen3-VL-4B; after it the local model is unknown."""
    if row.get("jev_probs"):
        return "jev"
    if row.get("clef_probs"):
        return "clef"
    if job_created_at and job_created_at < VLM_MODEL_KNOB_AT:
        return "qwen3-vl-4b"
    return "local-vlm(model unrecorded)"


LOCAL_READERS = ("qwen3-vl-4b", "local-vlm(model unrecorded)")


def box_anchor(box) -> tuple[float, float]:
    """Bumper bottom-centre in composite pixels (the point the pipeline projects)."""
    x, y, w, h = box
    return (x + w / 2.0, y + h)


# --------------------------------------------------------------------------- #
# Calibration / projection
# --------------------------------------------------------------------------- #

_CAL_CACHE: dict = {}


def job_views(event_key: str, match_key: str, created_at: str) -> list:
    """[(name, crop, calibration)] with each camera's calibration version in force
    when the job ran (versions created after the job are ignored), main first."""
    key = (event_key, match_key, created_at)
    if key in _CAL_CACHE:
        return _CAL_CACHE[key]
    from web.field import store as field_store
    from web.field.composite_layout import load_layout
    from web.field.match_order import select_epoch
    out = []
    for name, cl in load_layout().items():
        vs = [v for v in field_store.load_versions(event_key, name)
              if (v.get("created_at") or "") <= (created_at or "9999")]
        v = select_epoch(vs, match_key) if vs else None
        if v:
            out.append((name, tuple(cl.crop), v["calibration"]))
    out.sort(key=lambda t: (t[0] != "main", t[0]))
    _CAL_CACHE[key] = out
    return out


def view_of(views: list, uv) -> Optional[tuple]:
    u, v = uv
    for name, (x, y, w, h), cal in views:
        if x <= u < x + w and y <= v < y + h:
            return name, (x, y), cal
    return None


def project_box(views: list, box) -> tuple[Optional[str], Optional[tuple]]:
    hit = view_of(views, box_anchor(box))
    if hit is None:
        return None, None
    name, (ox, oy), cal = hit
    u, v = box_anchor(box)
    xy = cal.image_to_field((u - ox, v - oy), z=0.0)
    return name, (tuple(float(c) for c in xy) if xy is not None else None)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def resolve_path(stored: Optional[str]) -> Optional[Path]:
    """Stored paths may point into another worktree; fall back to this repo."""
    if not stored:
        return None
    p = Path(stored)
    if p.exists():
        return p
    s = str(p)
    for marker in ("/web/uploads/", "/web/replays/", "/data/"):
        if marker in s:
            cand = REPO / (marker.strip("/") + "/" + s.split(marker, 1)[1])
            if cand.exists():
                return cand
    return None


def replay_rows(match_keys: set) -> dict:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        out = {}
        q = ",".join("?" * len(match_keys))
        for rid, mk, created, src in con.execute(
                f"SELECT id, match_key, created_at, source_video_path FROM replays "
                f"WHERE match_key IN ({q})", sorted(match_keys)):
            out[rid] = {"id": rid, "match_key": mk, "created_at": created or "",
                        "video": resolve_path(src)}
        return out
    finally:
        con.close()


_META: dict = {}


def load_replay(job_id: str) -> Optional[dict]:
    if job_id not in _META:
        p = REPLAYS_DIR / job_id / "meta.json"
        if not p.exists():
            _META[job_id] = None
        else:
            meta = json.loads(p.read_text())
            tracks, spacing = slot_tracks(meta.get("frames") or [])
            _META[job_id] = {"tracks": tracks, "spacing": spacing}
    return _META[job_id]


def first_nonzero_frame(diag_path: Path) -> Optional[int]:
    with open(diag_path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("type") != "snapshot" and int(d.get("frame") or 0) > 0:
                return int(d["frame"])
    return None


def iter_reads(job_id: str):
    """Yield (line_no, row) for vlm_read rows of a job's diagnostic file."""
    p = UPLOADS_DIR / f"diagnostic_{job_id}.jsonl"
    if not p.exists():
        return
    with open(p) as fh:
        for i, line in enumerate(fh):
            if '"vlm_read"' not in line:
                continue
            d = json.loads(line)
            if d.get("type") == "vlm_read":
                yield i, d


def video_fps(path: Path) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 1 else DEFAULT_FPS
    except Exception:
        return DEFAULT_FPS


# --------------------------------------------------------------------------- #
# Read localisation (streaming over each video)
# --------------------------------------------------------------------------- #

def _loc_chunk(video: str, items: list) -> list:
    """Locate read crops in ``video``. items: [{key, lo, hi, b64}] with inclusive
    frame windows. Coarse half-res match per candidate k, refined at full res.
    Returns [{key, score, F_src, box}] (score -1 when nothing matched)."""
    import cv2
    import numpy as np
    from scripts.build_tracklet_bench import (COARSE, COARSE_MIN_SIDE, REFINE_PAD,
                                              best_match, candidate_ks, downscale)
    prepared, skipped = [], []
    for it in items:
        buf = np.frombuffer(base64.b64decode(it["b64"]), np.uint8)
        crop = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if crop is None:
            skipped.append(it["key"])
            continue
        h, w = crop.shape[:2]
        ks = candidate_ks(w, h, "matched")
        if len(ks) > 4:                    # inconsistent dims: brute force is too slow
            skipped.append(it["key"])
            continue
        tm = {}
        for k in ks:
            t = downscale(crop, k)
            if t.shape[0] < 4 or t.shape[1] < 4:
                continue
            th, tw = t.shape[:2]
            ts = None
            if min(th, tw) * COARSE >= min(COARSE_MIN_SIDE, 6):
                ts = cv2.resize(t, (max(1, int(round(tw * COARSE))), max(1, int(round(th * COARSE)))),
                                interpolation=cv2.INTER_AREA)
            tm[k] = (t, ts)
        if tm:
            prepared.append({**it, "tm": tm, "best": (-1.0, None, None)})
        else:
            skipped.append(it["key"])
    skip_rows = [{"key": k, "score": -1.0, "F_src": None, "box": None} for k in skipped]
    if not prepared:
        return skip_rows
    by_frame: dict = defaultdict(list)
    for p in prepared:
        for f in range(max(0, p["lo"]), p["hi"] + 1):
            by_frame[f].append(p)
    frames = sorted(by_frame)
    cap = cv2.VideoCapture(video)
    pos = None
    for f in frames:
        if pos is None or f < pos or f - pos > 90:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            pos = f
        while pos < f:
            cap.grab()
            pos += 1
        ok, fr = cap.read()
        pos += 1
        if not ok:
            continue
        small = cv2.resize(fr, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA)
        H, W = fr.shape[:2]
        for p in by_frame[f]:
            for k, (t, ts) in p["tm"].items():
                th, tw = t.shape[:2]
                if ts is not None:
                    _s, (x, y) = best_match(small, ts)
                    cx, cy = int(round(x / COARSE)), int(round(y / COARSE))
                    x0, y0 = max(0, cx - REFINE_PAD), max(0, cy - REFINE_PAD)
                    x1, y1 = min(W, cx + tw + REFINE_PAD), min(H, cy + th + REFINE_PAD)
                    s, (rx, ry) = best_match(fr[y0:y1, x0:x1], t)
                    bx, by = x0 + rx, y0 + ry
                else:
                    s, (bx, by) = best_match(fr, t)
                if s > p["best"][0]:
                    p["best"] = (s, f, (int(bx), int(by), int(tw), int(th)))
    cap.release()
    return skip_rows + [{"key": p["key"], "score": round(float(p["best"][0]), 4), "F_src": p["best"][1],
                         "box": p["best"][2]} for p in prepared]


def localize_reads(tasks: dict, cache: dict, workers: int, chunk_frames: int = 900) -> dict:
    """tasks: {video: [{key, lo, hi, b64}]}. Fills and returns ``cache``
    {key: {score, F_src, box}}."""
    jobs = []
    for video, items in tasks.items():
        todo = sorted((it for it in items if it["key"] not in cache), key=lambda it: it["lo"])
        chunk, start = [], None
        for it in todo:
            if start is None:
                start = it["lo"]
            if it["lo"] - start > chunk_frames and chunk:
                jobs.append((video, chunk))
                chunk, start = [], it["lo"]
            chunk.append(it)
        if chunk:
            jobs.append((video, chunk))
    if not jobs:
        return cache
    n = sum(len(c) for _, c in jobs)
    print(f"[loc] {n} reads in {len(jobs)} chunks over {len(tasks)} videos", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_loc_chunk, v, c) for v, c in jobs]
        for fu in as_completed(futs):
            for r in fu.result():
                cache[r["key"]] = {"score": r["score"], "F_src": r["F_src"], "box": r["box"]}
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"[loc] {done}/{len(futs)} chunks, {time.time() - t0:.0f}s", flush=True)
    return cache


def load_cache(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d.pop("key")] = d
    return out


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for k in sorted(cache):
            fh.write(json.dumps({"key": k, **cache[k]}) + "\n")


# --------------------------------------------------------------------------- #
# GT (2026cur_qm8)
# --------------------------------------------------------------------------- #

def load_gt_docs(match_key: str, fingerprint: Optional[str]) -> list:
    d = GT_DIR / match_key
    if not d.is_dir():
        return []
    docs = []
    for p in sorted(d.glob("*.json")):
        doc = json.loads(p.read_text())
        if not doc.get("team") or "keyframes" not in doc:
            continue
        if fingerprint and (doc.get("video") or {}).get("fingerprint") not in (None, fingerprint):
            continue                         # different video: frames don't line up
        doc["keyframes"] = sorted(doc["keyframes"], key=lambda k: k["frame"])
        docs.append(doc)
    return docs


def gt_dists(docs: list, frame: float, box, views: list) -> dict:
    """{team: pixel distance from the crop box centre to that GT robot} at frame.
    Main view: the annotator's uv. Corner views: GT field_xy projected into that
    corner camera at the click height."""
    x, y, w, h = box
    c = (x + w / 2.0, y + h / 2.0)
    hit = view_of(views, c)
    out = {}
    for doc in docs:
        if hit is None or hit[0] == "main":
            uv, _st = interp_keyframes(doc["keyframes"], frame, "uv")
            if uv is None:
                continue
            p = uv
        else:
            fxy, _st = interp_keyframes(doc["keyframes"], frame, "field_xy")
            if fxy is None:
                continue
            name, (ox, oy), cal = hit
            try:
                u, v = cal.field_to_image((fxy[0], fxy[1], float(doc.get("click_height_m") or GT_CLICK_Z)))
            except Exception:
                continue
            p = (u + ox, v + oy)
        out[str(doc["team"])] = math.hypot(p[0] - c[0], p[1] - c[1])
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def validate_gt_anchors(reps: dict, fp_of: dict, cache: dict, rosters: dict,
                        exclude_s: float, half_spans: tuple) -> None:
    """Score the vote rule against GT: every localised main-view read on a GT
    match whose robot GT identifies becomes a pseudo-crop (anchor); its slot
    segment is voted on exactly as a bench crop is, and the decision compared
    with the anchor's GT team. Only reads already in the localisation cache
    take part (those near bench-crop segments)."""
    for mk in sorted({r["match_key"] for r in reps.values()}):
        if not (GT_DIR / mk).is_dir():
            continue
        by_fp: dict = defaultdict(list)
        for j, r in reps.items():
            if r["match_key"] == mk and j in fp_of:
                by_fp[fp_of[j]].append(j)
        for fp, jobs in by_fp.items():
            docs = load_gt_docs(mk, fp)
            if not docs:
                continue
            team_of = {}
            for j in jobs:
                for ln, d in iter_reads(j):
                    k = f"{j}:{ln}"
                    if k in cache and d.get("matched_team") and str(d["matched_team"]) in rosters.get(mk, ()):
                        team_of[k] = str(d["matched_team"])
            loc = sorted((v["F_src"], tuple(v["box"]), k) for k, v in cache.items()
                         if k in team_of and v["score"] >= LOC_MIN and v["box"])
            fps = DEFAULT_FPS
            res = Counter()
            for F, box, k in loc:
                j = k.split(":")[0]
                rep = load_replay(j)
                views = job_views(mk.split("_")[0], mk, reps[j]["created_at"])
                g = gt_team_at(docs, F, box, views)[0]
                cam, xy = project_box(views, box)
                if g is None or rep is None or cam != "main":
                    continue
                slot, _d = associate(xy, positions_at(rep["tracks"], F, rep["spacing"]))
                if slot is None:
                    continue
                run = run_at(cut_runs(rep["tracks"][slot], rep["spacing"], fps), F, rep["spacing"])
                if run is None:
                    continue
                a, b = clip_segment(run, F, max(half_spans) * fps)
                votes = []
                for F2, box2, k2 in loc[bisect.bisect_left(loc, (a,)):]:
                    if F2 > b:
                        break
                    c2, xy2 = project_box(views, box2)
                    s2, _ = associate(xy2, positions_at(rep["tracks"], F2, rep["spacing"]), *gates_for(c2))
                    if s2 == slot:
                        votes.append((F2, team_of[k2]))
                dec, _hs = cascade(half_spans, lambda hs, F=F, votes=votes: decide(
                    [v for v in votes if abs(v[0] - F) <= hs * fps], F, fps, exclude_s=exclude_s))
                res["anchors"] += 1
                if dec["team"]:
                    res["right" if dec["team"] == g else "WRONG"] += 1
            print(f"GT anchor validation {mk} ({len(jobs)} jobs, {len(loc)} localised reads):", dict(res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", default=str(BENCH_DIR))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--ids", default="", help="comma-separated crop ids (debug; output not written)")
    ap.add_argument("--max-jobs-per-video", type=int, default=400,
                    help="cap the jobs whose reads are localised per video (crop jobs always kept)")
    ap.add_argument("--half-spans", default=",".join(f"{h:g}" for h in HALF_SPANS),
                    help="segment half-widths in seconds, tried in order (see cascade())")
    ap.add_argument("--exclude-s", type=float, default=EXCLUDE_S,
                    help="reads within this many seconds of the crop frame never vote")
    ap.add_argument("--out", default=OUT_NAME, help="output file name inside the bench dir")
    ap.add_argument("--dry-run", action="store_true", help="count reads to localise, then stop")
    ap.add_argument("--validate-gt", action="store_true",
                    help="also print GT purity of the slot votes on GT matches")
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    half_spans = tuple(float(x) for x in args.half_spans.split(","))
    half_span_s = max(half_spans)
    exclude_s = args.exclude_s

    manifest = [json.loads(l) for l in (bench / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        manifest = [m for m in manifest if m["id"] in want]
    tracklets = {}
    tp = bench / "tracklets.jsonl"
    if tp.exists():
        for l in tp.read_text().splitlines():
            if l.strip():
                t = json.loads(l)
                tracklets[t["id"]] = t

    from web.field.ground_truth import video_fingerprint
    reps = replay_rows({m["match_key"] for m in manifest})
    fps_by_video: dict = {}
    fp_of: dict = {}
    for rid, r in reps.items():
        if r["video"] is not None:
            try:
                fp_of[rid] = video_fingerprint(r["video"])
            except OSError:
                pass

    # ---- 1. crop -> slot segment -------------------------------------------
    crops = []
    for m in manifest:
        job = reps.get(m["job_id"])
        c = {"m": m, "job": job, "notes": [], "kind": None}
        crops.append(c)
        t = tracklets.get(m["id"])
        fp = fp_of.get(m["job_id"])
        if job is None or job["video"] is None or fp is None:
            c["notes"].append("no source video")
            continue
        if not t or t.get("status") != "ok":
            c["notes"].append(f"crop not located in video ({(t or {}).get('status', 'no tracklet')})")
            continue
        vid = str(job["video"])
        if vid not in fps_by_video:
            fps_by_video[vid] = float(t.get("fps") or video_fps(job["video"]))
        fps = fps_by_video[vid]
        c.update(fp=fp, fps=fps, F=int(t["F_src"]), box=tuple(t["box"]))
        views = job_views(m["match_key"].split("_")[0], m["match_key"], job["created_at"])
        c["views"] = views
        cam, xy = project_box(views, c["box"])
        c["cam"], c["xy"] = cam, xy
        rep = load_replay(m["job_id"])
        if rep is None or xy is None:
            c["notes"].append("no replay" if rep is None else "crop does not project to the field")
            continue
        pos = positions_at(rep["tracks"], c["F"], rep["spacing"])
        slot, d = associate(xy, pos, *gates_for(cam))
        c["slot_d"] = d
        if slot is None:
            c["notes"].append(f"no slot at the crop (nearest {d:.2f} m)" if d is not None
                              else "no slot at the crop")
            c["kind"] = "located"
            continue
        runs = cut_runs(rep["tracks"][slot], rep["spacing"], fps)
        run = run_at(runs, c["F"], rep["spacing"])
        if run is None:
            c["notes"].append("crop frame not inside a slot run")
            c["kind"] = "located"
            continue
        c["slot"], c["kind"] = slot, "slot"
        c["seg"] = clip_segment(run, c["F"], half_span_s * fps)   # widest span; narrowed per decision
        c["run"] = (run[0][0], run[-1][0])

    # ---- 2. candidate reads per video fingerprint ---------------------------
    by_fp_jobs: dict = defaultdict(list)
    for rid, fp in fp_of.items():
        by_fp_jobs[fp].append(rid)
    windows: dict = defaultdict(list)      # fp -> [(lo, hi)] (segments of slot crops)
    crop_jobs: dict = defaultdict(set)
    for c in crops:
        if c["kind"] == "slot":
            windows[c["fp"]].append(c["seg"])
            crop_jobs[c["fp"]].add(c["m"]["job_id"])
    rosters = {m["match_key"]: set(m["red_teams"]) | set(m["blue_teams"]) for m in manifest}

    def in_windows(fp, lo, hi):
        return any(lo <= b and hi >= a for a, b in windows[fp])

    tasks: dict = defaultdict(list)
    read_meta: dict = {}                   # key -> {job, team, reader, frame}
    for fp, wins in windows.items():
        jobs = sorted(by_fp_jobs[fp], key=lambda j: (j not in crop_jobs[fp], reps[j]["created_at"]), reverse=False)
        keep = [j for j in jobs if j in crop_jobs[fp]]
        others = [j for j in jobs if j not in crop_jobs[fp]]
        # most recent others first (newer pipelines read more), capped
        others.sort(key=lambda j: reps[j]["created_at"], reverse=True)
        keep += others[:max(0, args.max_jobs_per_video - len(keep))]
        for j in keep:
            mk = reps[j]["match_key"]
            roster = rosters.get(mk, set())
            vid = str(reps[j]["video"])
            fnz = None
            for ln, d in iter_reads(j):
                team = d.get("matched_team")
                if not team or str(team) not in roster or not d.get("crop_b64"):
                    continue
                f = int(d.get("frame") or 0)
                if f == 0:
                    if fnz is None:
                        fnz = first_nonzero_frame(UPLOADS_DIR / f"diagnostic_{j}.jsonl") or 0
                    lo, hi = fnz + EARLY[0], fnz + EARLY[1]
                else:
                    lo, hi = f + NEAR[0], f + NEAR[1]
                if not in_windows(fp, lo, hi):
                    continue
                key = f"{j}:{ln}"
                read_meta[key] = {"job": j, "team": str(team), "reader": reader_tag(d, reps[j]["created_at"]), "fp": fp}
                tasks[vid].append({"key": key, "lo": lo, "hi": hi, "b64": d["crop_b64"]})

    cache_path = bench / CACHE_NAME
    cache = load_cache(cache_path)
    if args.dry_run:
        per = Counter()
        for vid, items in tasks.items():
            per[Path(vid).name] = sum(1 for it in items if it["key"] not in cache)
        print("reads to localise (uncached):", sum(per.values()), dict(per.most_common()))
        print("readers:", dict(Counter(m["reader"] for m in read_meta.values())))
        return
    cache = localize_reads(tasks, cache, args.workers)
    save_cache(cache_path, cache)

    # localised reads per fp, sorted by frame
    loc_by_fp: dict = defaultdict(list)
    for key, meta in read_meta.items():
        L = cache.get(key)
        if not L or L["score"] is None or L["score"] < LOC_MIN or L["box"] is None:
            continue
        loc_by_fp[meta["fp"]].append((L["F_src"], tuple(L["box"]), key))
    for v in loc_by_fp.values():
        v.sort()

    # ---- 3. GT docs ----------------------------------------------------------
    gt_cache: dict = {}

    def gt_docs_for(c):
        k = (c["m"]["match_key"], c.get("fp"))
        if k not in gt_cache:
            gt_cache[k] = load_gt_docs(*k)
        return gt_cache[k]

    # ---- 4. votes + decision -------------------------------------------------
    out_rows = []
    gt_val = Counter()
    for c in crops:
        m = c["m"]
        row = {"id": m["id"], "truth_team": None, "truth_source": "unknown", "votes": {},
               "n_frames": 0, "segment": None, "readers": [], "notes": "",
               "segment_kind": None, "gt_team": None, "vote_team": None,
               "slot": c.get("slot"), "crop_frame_src": c.get("F"), "vote_jobs": 0,
               "reason": None, "half_span_s": None, "votes_by_reader": {}, "local_only_team": None}
        notes = list(c["notes"])
        fps = c.get("fps") or DEFAULT_FPS
        votes4 = None                      # [(frame, team, reader, job)] over the widest segment
        if c["kind"] == "slot":
            rep = load_replay(m["job_id"])
            a, b = c["seg"]
            fr_list = loc_by_fp.get(c["fp"], [])
            votes4, boxes = [], {}
            for F, box, key in fr_list[bisect.bisect_left(fr_list, (a,)):]:
                if F > b:
                    break
                rcam, xy = project_box(c["views"], box)
                s_, _d = associate(xy, positions_at(rep["tracks"], F, rep["spacing"]), *gates_for(rcam))
                if s_ != c["slot"]:
                    continue
                meta = read_meta[key]
                votes4.append((F, meta["team"], meta["reader"], meta["job"]))
                boxes[len(votes4) - 1] = box
            centre = c["F"]
            seg_at = lambda hs, a=a, b=b, centre=centre: (max(a, int(centre - hs * fps)),  # noqa: E731
                                                          min(b, int(centre + hs * fps)))
            kind_note = (f"slot {c['slot']} run {c['run'][0]}-{c['run'][1]}, crop->slot "
                         f"{c['slot_d']:.2f} m ({c['cam']})")
        elif c["kind"] is None and not c.get("box"):
            # identity fallback (no video / crop not located)
            votes4, act, fnz = [], [], None
            for _ln, d in iter_reads(m["job_id"]):
                if int(d.get("robot_id", -1)) != int(m["robot_id"]):
                    continue
                f = int(d.get("frame") or 0)
                if f == 0:
                    if fnz is None:
                        fnz = first_nonzero_frame(UPLOADS_DIR / f"diagnostic_{m['job_id']}.jsonl") or 0
                    f = fnz
                act.append(f)
                t = d.get("matched_team")
                if t and str(t) in rosters.get(m["match_key"], set()):
                    votes4.append((f, str(t), reader_tag(d, (c["job"] or {}).get("created_at", "")), m["job_id"]))
            centre = int(m["frame"]) or (fnz if fnz is not None else
                                        first_nonzero_frame(UPLOADS_DIR / f"diagnostic_{m['job_id']}.jsonl") or 0)
            seg_at = lambda hs, act=act, centre=centre: activity_segment(  # noqa: E731
                act, centre, IDENT_GAP_S * fps, hs * fps)
            kind_note = (f"identity {m['robot_id']} activity segment (frames are the diagnostic's "
                         f"_current_frame, not source frames)")
        if votes4 is not None:
            def decide_at(hs, local=False):
                lo, hi = seg_at(hs)
                vv = [(f, t) for f, t, r, _j in votes4 if lo <= f <= hi and (not local or r in LOCAL_READERS)]
                return {**decide(vv, centre, fps, exclude_s=exclude_s), "seg": [lo, hi], "hs": hs}
            dec, _hs = cascade(half_spans, decide_at)
            loc_dec, _ = cascade(half_spans, lambda hs: decide_at(hs, local=True))
            lo, hi = dec["seg"]
            used = [(i, v) for i, v in enumerate(votes4)
                    if lo <= v[0] <= hi and abs(v[0] - centre) > exclude_s * fps]
            byr: dict = defaultdict(Counter)
            for _i, (_f, t, r, _j) in used:
                byr[r][t] += 1
            row.update(votes=dec["votes"], n_frames=dec["n_frames"], segment=[lo, hi],
                       readers=sorted(byr), segment_kind="slot" if c["kind"] == "slot" else "identity",
                       vote_team=dec["team"], vote_jobs=len({v[3] for _i, v in used}),
                       reason=dec["reason"], half_span_s=dec["hs"],
                       votes_by_reader={r: dict(v) for r, v in sorted(byr.items())},
                       local_only_team=loc_dec["team"])
            notes.append(f"{kind_note}; +-{dec['hs']:g}s: {dec['reason']}"
                         + (f" (span {dec['span_s']}s, share {dec['share']})" if dec["span_s"] is not None else ""))
            if dec["team"]:
                row.update(truth_team=dec["team"], truth_source="track_votes")
            if args.validate_gt and c["kind"] == "slot" and gt_docs_for(c):
                docs = gt_docs_for(c)
                for i, (F, team, _r, _j) in used:
                    g = gt_team_at(docs, F, boxes[i], c["views"])[0]
                    gt_val["read_gt_none" if g is None else
                           ("read_team==gt" if g == team else "read_team!=gt")] += 1

        # GT override
        if c.get("box") is not None and c.get("fp"):
            docs = gt_docs_for(c)
            if docs:
                g, dists = gt_team_at(docs, c["F"], c["box"], c["views"])
                near = ", ".join(f"{t}:{d:.0f}px" for t, d in sorted(dists.items(), key=lambda kv: kv[1])[:3])
                row["gt_team"] = g
                if g:
                    if row["vote_team"] and row["vote_team"] != g:
                        notes.append(f"GT {g} overrides votes {row['vote_team']}")
                    row.update(truth_team=g, truth_source="gt")
                    notes.append(f"GT nearest {near}")
                else:
                    notes.append(f"GT ambiguous/none ({near or 'no GT robot positioned'})")
                    if row["truth_source"] == "track_votes":
                        notes.append("GT has no clear robot at the crop; votes kept")
        row["notes"] = "; ".join(notes)
        out_rows.append(row)

    # ---- 5. write + summary --------------------------------------------------
    if not args.ids:
        with open(bench / args.out, "w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {bench / args.out} ({len(out_rows)} rows)")
    else:
        for r in out_rows:
            print(json.dumps(r))
    src = Counter(r["truth_source"] for r in out_rows)
    kinds = Counter(r["segment_kind"] for r in out_rows)
    print("truth_source:", dict(src), " segment_kind:", dict(kinds))
    if gt_val:
        print("GT check of the reads that voted on GT-match crops:", dict(gt_val))
    if args.validate_gt:
        validate_gt_anchors(reps, fp_of, cache, rosters, exclude_s, half_spans)


if __name__ == "__main__":
    main()

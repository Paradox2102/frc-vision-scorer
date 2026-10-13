"""Bench + equivalence check: seek-per-sample vs sequential clock scan.

For N real match videos, runs both `scan_clock_samples` paths with the real
Tesseract reader (`InferenceJob.read_match_clock`, no models loaded) and
reports wall time, sample counts, differing samples, and whether
`compute_active_windows` lands on identical ranges/periods/fallback. Videos
are picked from the existing `clockscan_*.json` caches (matched to their
source video by the cache key), so each row is also compared against the
cached samples. Read-only: never writes a cache.

The OCR ROI is rebuilt the way a job builds it: the video's event (looked up
via `replays.source_video_path`) supplies `events.ocr_regions`, merged over
the defaults and scaled by `scale_ocr_regions`.

For every differing sample, both frames are re-decoded and the seek frame is
located among the exact sequentially-decoded neighbours (+-3 frames), and the
two clock ROIs are saved side by side to --diff-dir.

    .venv/bin/python scripts/bench_clock_scan.py [-n 3] [--video PATH ...]
        [--data-root /path/to/checkout] [--workers 8] [--diff-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import web.app as app_mod  # noqa: E402
from web.app import (  # noqa: E402
    InferenceJob, compute_active_windows, scan_clock_samples,
)

NEIGHBOURS = 3


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    overhead_height = int(height * app_mod.OVERHEAD_CROP_RATIO)
    sample_step = max(1, int(round(fps * app_mod.MATCH_CLOCK_SAMPLE_SECONDS)))
    return dict(width=width, height=height, fps=fps, total_frames=total,
                overhead_height=overhead_height, sample_step=sample_step)


def cache_key(path: Path, sample_step: int, overhead_height: int) -> str:
    """Same key as InferenceJob._clock_scan_cache_path."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    h.update(f"{path.stat().st_size}:{sample_step}:{overhead_height}".encode())
    return h.hexdigest()


def find_cached_videos(uploads: Path) -> list[tuple[Path, Path]]:
    """(video, cache) pairs, one per cache, deduped by video content."""
    caches = {p.name[len("clockscan_"):-len(".json")]: p
              for p in uploads.glob("clockscan_*.json")}
    seen_content, pairs, matched = set(), [], set()
    for video in sorted(uploads.glob("*.mp4")):
        with open(video, "rb") as f:
            content = (hashlib.md5(f.read(1 << 20)).hexdigest(), video.stat().st_size)
        if content in seen_content:
            continue
        seen_content.add(content)
        meta = video_meta(video)
        key = cache_key(video, meta["sample_step"], meta["overhead_height"])
        if key in caches and key not in matched:
            matched.add(key)
            pairs.append((video, caches[key]))
    return pairs


def event_regions(db: Path, video: Path) -> tuple[str | None, dict | None]:
    """The event's saved OCR regions for a video, via replays.source_video_path."""
    if not db.exists():
        return None, None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        size = video.stat().st_size
        with open(video, "rb") as f:
            head = hashlib.md5(f.read(1 << 20)).hexdigest()
        rows = con.execute(
            "SELECT DISTINCT r.event_key, e.ocr_regions FROM replays r "
            "LEFT JOIN events e ON e.event_key = r.event_key "
            "WHERE r.source_video_path IS NOT NULL"
            " AND r.source_video_path LIKE '%' || ?",
            (video.name,),
        ).fetchall()
        if not rows:
            # The job copies the upload under a fresh uuid name: match by content.
            for ev, regions, src in con.execute(
                "SELECT DISTINCT r.event_key, e.ocr_regions, r.source_video_path "
                "FROM replays r LEFT JOIN events e ON e.event_key = r.event_key "
                "WHERE r.source_video_path IS NOT NULL"
            ):
                cand = video.parent / Path(src).name
                if cand.exists() and cand.stat().st_size == size:
                    with open(cand, "rb") as f:
                        if hashlib.md5(f.read(1 << 20)).hexdigest() == head:
                            rows = [(ev, regions)]
                            break
    finally:
        con.close()
    if not rows:
        return None, None
    ev, regions = rows[0]
    return ev, (json.loads(regions) if regions else None)


def make_reader(meta: dict, regions: dict | None):
    """A real read_match_clock bound to a job stub (no models loaded)."""
    app_mod.custom_ocr_regions = regions
    job = InferenceJob.__new__(InferenceJob)
    job.ocr_regions = app_mod.scale_ocr_regions(meta["width"], meta["overhead_height"])
    return job


def warm(path: Path) -> None:
    with open(path, "rb") as f:
        while f.read(1 << 24):
            pass


def windows(samples, meta):
    ranges, periods, fell_back, reason = compute_active_windows(
        samples, meta["fps"], meta["total_frames"]
    )
    return {"ranges": ranges, "periods": periods, "fallback": fell_back, "reason": reason}


def diff_samples(a, b):
    da, db_ = dict(a), dict(b)
    idx = sorted(set(da) | set(db_))
    return [(i, da.get(i, "missing"), db_.get(i, "missing"))
            for i in idx if da.get(i, "missing") != db_.get(i, "missing")]


def inspect_diffs(video: Path, meta: dict, job, diffs, out_dir: Path, tag: str, limit: int):
    """Locate each differing seek frame among exact sequential neighbours."""
    targets = [i for i, _, _ in diffs[:limit]]
    if not targets:
        return []
    want = {i + d for i in targets for d in range(-NEIGHBOURS, NEIGHBOURS + 1)}
    oh = meta["overhead_height"]
    seq_frames = {}
    cap = cv2.VideoCapture(str(video))
    try:
        for idx in range(max(want) + 1):
            if not cap.grab():
                break
            if idx in want:
                ok, frame = cap.retrieve()
                if ok:
                    seq_frames[idx] = frame[:oh].copy()
    finally:
        cap.release()
    x, y, w, h = job.ocr_regions["match_clock"]
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = []
    cap = cv2.VideoCapture(str(video))
    try:
        for i, old_v, new_v in diffs[:limit]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or i not in seq_frames:
                findings.append({"frame": i, "note": "could not re-decode"})
                continue
            seek = frame[:oh]
            errs = {
                j - i: float(np.abs(seek.astype(np.int16) - seq_frames[j].astype(np.int16)).mean())
                for j in range(i - NEIGHBOURS, i + NEIGHBOURS + 1) if j in seq_frames
            }
            best = min(errs, key=errs.get)
            seek_roi = cv2.resize(seek[y:y + h, x:x + w], None, fx=4, fy=4,
                                  interpolation=cv2.INTER_NEAREST)
            seq_roi = cv2.resize(seq_frames[i][y:y + h, x:x + w], None, fx=4, fy=4,
                                 interpolation=cv2.INTER_NEAREST)
            pane = np.hstack([seek_roi, np.full((seek_roi.shape[0], 8, 3), 128, np.uint8), seq_roi])
            pane = cv2.copyMakeBorder(pane, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(pane, f"seek f{i}: {old_v}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(pane, f"sequential f{i}: {new_v}", (seek_roi.shape[1] + 12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            img = out_dir / f"{tag}_f{i}.png"
            cv2.imwrite(str(img), pane)
            findings.append({
                "frame": i, "seek": old_v, "sequential": new_v,
                "seek_frame_matches_offset": best,
                "seek_vs_exact_mae": round(errs.get(0, float("nan")), 2),
                "seek_vs_best_mae": round(errs[best], 2),
                "image": str(img),
            })
    finally:
        cap.release()
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=3, help="number of cached videos to bench")
    ap.add_argument("--video", type=Path, action="append", help="explicit video(s)")
    ap.add_argument("--data-root", type=Path, default=REPO,
                    help="checkout holding web/uploads and frc_scorer.db")
    ap.add_argument("--workers", type=int, default=app_mod.CLOCK_SCAN_OCR_WORKERS)
    ap.add_argument("--diff-dir", type=Path, default=Path("/tmp/clock-scan-diffs"))
    ap.add_argument("--inspect", type=int, default=4, help="diffs to inspect per video")
    ap.add_argument("--json", type=Path, help="write full results here")
    args = ap.parse_args()

    uploads = args.data_root / "web" / "uploads"
    db = args.data_root / "frc_scorer.db"
    if args.video:
        pairs = []
        for v in args.video:
            meta = video_meta(v)
            c = uploads / f"clockscan_{cache_key(v, meta['sample_step'], meta['overhead_height'])}.json"
            pairs.append((v, c if c.exists() else None))
    else:
        pairs = find_cached_videos(uploads)[: args.n]
    if not pairs:
        print("no videos found")
        return 1

    results = []
    for video, cache in pairs:
        meta = video_meta(video)
        ev, regions = event_regions(db, video)
        job = make_reader(meta, regions)
        print(f"\n== {video.name}  event={ev}  {meta['width']}x{meta['height']}@{meta['fps']:.2f}"
              f"  frames={meta['total_frames']}  step={meta['sample_step']}"
              f"  clock_roi={job.ocr_regions['match_clock']}", flush=True)
        warm(video)
        t0 = time.perf_counter()
        old = scan_clock_samples(video, meta["total_frames"], meta["sample_step"],
                                 meta["overhead_height"], job.read_match_clock,
                                 sequential=False)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = scan_clock_samples(video, meta["total_frames"], meta["sample_step"],
                                 meta["overhead_height"], job.read_match_clock,
                                 sequential=True, workers=args.workers)
        t_new = time.perf_counter() - t0
        diffs = diff_samples(old, new)
        w_old, w_new = windows(old, meta), windows(new, meta)
        row = {
            "video": video.name, "event": ev, "t_old": round(t_old, 1), "t_new": round(t_new, 1),
            "n_old": len(old), "n_new": len(new), "n_diff": len(diffs), "diffs": diffs,
            "windows_equal": w_old == w_new, "windows_old": w_old, "windows_new": w_new,
        }
        if cache is not None:
            cached = [(int(i), None if v is None else int(v)) for i, v in json.loads(cache.read_text())]
            w_cache = windows(cached, meta)
            row.update({
                "cache": cache.name,
                "n_diff_old_vs_cache": len(diff_samples(old, cached)),
                "n_diff_new_vs_cache": len(diff_samples(new, cached)),
                "windows_new_eq_cache": w_new == w_cache,
            })
        row["inspected"] = inspect_diffs(video, meta, job, diffs, args.diff_dir,
                                         video.stem[:13], args.inspect)
        results.append(row)
        print(f"  old(seek) {t_old:6.1f}s  new(seq,{args.workers}w) {t_new:6.1f}s  "
              f"speedup {t_old / max(t_new, 1e-9):.1f}x  samples {len(old)}/{len(new)}  "
              f"diff {len(diffs)}  windows_equal={row['windows_equal']}")
        print(f"  windows: {w_new['ranges']} fallback={w_new['fallback']}")
        if cache is not None:
            print(f"  vs cache: old diff {row['n_diff_old_vs_cache']}, new diff "
                  f"{row['n_diff_new_vs_cache']}, windows_new==cache {row['windows_new_eq_cache']}")
        for d in diffs[: args.inspect * 3]:
            print(f"    diff frame {d[0]}: seek={d[1]} seq={d[2]}")
        for f in row["inspected"]:
            print(f"    inspect {f}")

    print("\n| video | old s | new s | speedup | samples | diffs | windows equal | vs cache (old/new) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        cache_col = (f"{r['n_diff_old_vs_cache']}/{r['n_diff_new_vs_cache']}"
                     if "cache" in r else "n/a")
        print(f"| {r['video'][:13]} | {r['t_old']} | {r['t_new']} | "
              f"{r['t_old'] / max(r['t_new'], 1e-9):.1f}x | {r['n_new']} | {r['n_diff']} | "
              f"{r['windows_equal']} | {cache_col} |")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

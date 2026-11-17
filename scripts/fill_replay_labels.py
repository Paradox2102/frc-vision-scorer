#!/usr/bin/env python
"""Apply the window-label run fill to an already-finalised replay.

Runs web/field/window_labels.fill_runs (the window_labels_fill knob) on a
replay whose window_labels stage ran without it: no reader, no API calls, no
GPU. The replay's own labels are the input; unread windows of a track chain
take the chain's named team (unconfirmed, HUD "1234?").

Usage
-----
    .venv/bin/python scripts/fill_replay_labels.py --replay a0bbc991 --dry-run
    .venv/bin/python scripts/fill_replay_labels.py --replay a0bbc991 --render
    .venv/bin/python scripts/fill_replay_labels.py --replay a0bbc991 --force   # redo from backup

What it writes (not with --dry-run):
  * meta.json, atomically; the first run keeps the original as
    meta.prefill.json next to it (never overwritten);
  * summary.window_labels.fill = {fill_frames, fill_windows,
    fill_chains_bridged, "offline": true, point stats};
  * the replay's match_robot_identity_segment rows (database.
    replace_match_identity_segments, the helper the app uses for the
    window-label segments), rebuilt with window_segments;
  * with --render, hud.mp4 via web.hud_render.run.

Refused (exit 2): a replay with no window_labels summary (or an aborted
stage: its labels are not window labels), and a replay already filled
(summary.window_labels.fill from this script, or fill_frames from a live
run with the knob on). --force redoes an offline fill from
meta.prefill.json; a live fill has no backup and cannot be redone here.

Reconstruction (the stage's own inputs are not all stored):
  * windows: build_windows(frames, SOURCE video fps from the video file, as
    live does; summary.fps is the processed rate and would cut different
    windows);
  * labels: a slot-frame with a team is a label (RANK_DIGITS when
    confirmed, else RANK_LINEUP; p 1.0); the other frames of a window that
    has any label are "cleared" (None) and never filled; other windows are
    unread;
  * single-sample reads from summary.window_labels_digit_calls: Jev windows
    map their wide calls to sample numbers exactly through the stored
    fingerprint requests; S1 windows store one call per tile, and tiles are
    only cut for samples inside the main view, which is not stored. When a
    window has a call for every sample the mapping is exact; otherwise the
    calls are mapped to w.samples in order (an approximation, counted as
    points_approx). Windows whose stored fingerprint names a different slot
    or first frame than the rebuilt window give no points
    (points_mismatch).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.score_ground_truth import (  # noqa: E402
    REPLAYS_DIR, _probe_fps, _resolve, replay_row)
from web.field import window_labels as wl  # noqa: E402

BACKUP_NAME = "meta.prefill.json"


class Refused(Exception):
    """The replay cannot be filled; the message says why."""


def reconstruct_lab(frames: list, windows: list) -> dict:
    """assemble_labels' ``lab`` from a window-labelled replay's frames."""
    wlen = {(i, w.slot): len(w.idx) for w in windows for i in w.idx}
    lab: dict = {}
    for i, f in enumerate(frames):
        for r in f.get("robots") or []:
            slot, team = r.get("slot"), r.get("team_number")
            if slot and team not in (None, ""):
                lab[(i, slot)] = (str(team), wl.RANK_DIGITS if r.get("confirmed") else wl.RANK_LINEUP,
                                  1.0, wlen.get((i, slot), 1))
    for w in windows:
        if any((i, w.slot) in lab for i in w.idx):
            for i in w.idx:
                lab.setdefault((i, w.slot), None)
    return lab


def reconstruct_point(frames: list, windows: list, raw: Optional[dict]) -> tuple:
    """(point for fill_runs, counters) from a stored digit-call cache."""
    info = {"points_exact": 0, "points_approx": 0, "points_mismatch": 0}
    if not raw:
        return {}, info
    dc = wl.load_digit_calls(raw)
    sources = {int(k): v for k, v in (raw.get("sources") or {}).items()}
    point: dict = {}
    for w in windows:
        calls = dc["calls"].get(w.wid)
        if not calls:
            continue
        fp = dc["fingerprints"].get(w.wid)
        if fp and (fp.get("slot") != w.slot
                   or fp.get("first_frame") != int(frames[w.idx[0]]["frame_num"])):
            info["points_mismatch"] += 1
            continue
        if sources.get(w.wid) == "s1":
            if len(calls) > len(w.samples):
                info["points_mismatch"] += 1
                continue
            info["points_exact" if len(calls) == len(w.samples) else "points_approx"] += 1
            point[w.wid] = {k: c for k, c in enumerate(calls, 1)}
        elif dc["wide_p"].get(w.wid):
            info["points_exact"] += 1
            point[w.wid] = dict(dc["wide_p"][w.wid])
        elif fp and len(fp.get("requests") or []) == len(calls):
            info["points_exact"] += 1
            point[w.wid] = {int(k): c for (kind, k), c in zip(fp["requests"], calls)
                            if kind == "wide" and c}
        else:
            info["points_mismatch"] += 1
    return point, info


def check_fillable(meta: dict) -> None:
    wls = (meta.get("summary") or {}).get("window_labels")
    if not isinstance(wls, dict):
        raise Refused("the replay has no window_labels summary (the stage never ran)")
    if wls.get("aborted"):
        raise Refused(f"the window_labels stage aborted ({wls['aborted']}): no window labels")
    if "fill" in wls:
        raise Refused("already filled offline (summary.window_labels.fill); --force redoes it "
                      f"from {BACKUP_NAME}")
    if "fill_frames" in wls:
        raise Refused("already filled live (window_labels_fill was on)")


def fill_stop_slack(summary: dict) -> int:
    """fill_runs' stop slack (replay-frame indices) as the live stage would
    use it for this replay: FILL_STOP_SLACK_FRAMES (tuned at 5 fps) rescaled
    to the job's sam3_track_fps under frame_knobs_fps_relative. Both come
    from the replay's recorded config (summary.config holds the knobs that
    differed from the code default, plus sam3_track_fps /
    frame_knobs_fps_relative always since Nov 17 2026); absent = a replay
    from before that, when the default was 5 fps (ON), i.e. the 5 fps
    behaviour -- NOT today's 10 fps default. A per-job target_tracking_fps override is not
    recorded and is not recoverable here."""
    from web.job_config import FRAME_KNOB_REF_FPS, scale_frames
    cfg = summary.get("config") or {}
    if not cfg.get("frame_knobs_fps_relative", True):
        return wl.FILL_STOP_SLACK_FRAMES
    fps = cfg.get("sam3_track_fps", cfg.get("sam2_track_fps", FRAME_KNOB_REF_FPS))
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = FRAME_KNOB_REF_FPS
    if fps <= 0:
        fps = FRAME_KNOB_REF_FPS
    return scale_frames(wl.FILL_STOP_SLACK_FRAMES, fps, FRAME_KNOB_REF_FPS)


def fill_meta(meta: dict, fps: float) -> tuple:
    """Fill ``meta`` in place. Returns (summary.window_labels.fill, lab)."""
    frames = meta["frames"]
    summary = meta["summary"]
    windows = wl.build_windows(frames, fps)
    lab = reconstruct_lab(frames, windows)
    raw = summary.get("window_labels_digit_calls")
    point, info = reconstruct_point(frames, windows, raw)
    stats = wl.fill_runs(frames, windows, lab, fps, point, (raw or {}).get("roster"),
                         stop_slack_frames=fill_stop_slack(summary))
    wl.apply_labels(frames, lab)
    wl.restamp_events(frames, meta.get("events") or [])
    out = {**stats, "offline": True, "windows": len(windows), **info}
    summary["window_labels"]["fill"] = out
    return out, lab


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def segment_rows(frames: list, lab: dict) -> list:
    """match_robot_identity_segment rows, as the app stores window segments."""
    return [{"slot": sid, "slot_label": f"{'B' if sid < 3 else 'R'}{sid % 3 + 1}",
             "start_frame": sg["start_frame"], "end_frame": sg["end_frame"],
             "team_number": sg["team_number"], "confirmed": sg["confirmed"],
             "score": sg["score"], "source": sg.get("source")}
            for sid, rows in sorted(wl.window_segments(frames, lab).items()) for sg in rows]


def store_segments(replay_id: str, rows: list) -> None:
    from web import database
    asyncio.run(database.replace_match_identity_segments(replay_id, rows))


def render(replay_id: str) -> tuple:
    from web import hud_render
    return hud_render.run(replay_id)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", required=True, help="replay id or unique prefix")
    ap.add_argument("--dry-run", action="store_true", help="report the fill, write nothing")
    ap.add_argument("--render", action="store_true", help="re-render hud.mp4 afterwards")
    ap.add_argument("--force", action="store_true",
                    help=f"redo an offline fill from {BACKUP_NAME}")
    args = ap.parse_args(argv)
    try:
        row = replay_row(args.replay)
    except LookupError as e:
        print(f"REFUSED: {e}")
        return 2
    meta_p = REPLAYS_DIR / row["id"] / "meta.json"
    if not meta_p.exists():
        meta_p = _resolve(row.get("meta_path"))
    if meta_p is None or not meta_p.exists():
        print(f"REFUSED: meta.json missing for replay {row['id']}")
        return 2
    video = _resolve(row.get("source_video_path"))
    fps = _probe_fps(video) if video is not None else None
    if not fps:
        print(f"REFUSED: cannot read the source video fps ({row.get('source_video_path')}); "
              "windows must be cut at the source fps")
        return 2
    backup = meta_p.with_name(BACKUP_NAME)
    meta = json.loads(meta_p.read_text())
    try:
        try:
            check_fillable(meta)
        except Refused:
            wls = (meta.get("summary") or {}).get("window_labels") or {}
            if not (args.force and isinstance(wls, dict) and "fill" in wls):
                raise
            if not backup.exists():
                raise Refused(f"--force needs {backup}, which is missing")
            meta = json.loads(backup.read_text())
            check_fillable(meta)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    stats, lab = fill_meta(meta, float(fps))
    print(f"replay {row['id'][:8]}  {row.get('match_key')}  fps {fps:g}: " + json.dumps(stats))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    if not backup.exists():
        shutil.copy2(meta_p, backup)
    _write_json_atomic(meta_p, meta)
    print(f"wrote {meta_p} (original kept as {backup.name})")
    try:
        rows = segment_rows(meta["frames"], lab)
        store_segments(row["id"], rows)
        print(f"segments: {len(rows)} rows stored")
    except Exception as e:
        print(f"WARNING: segment rows not stored ({type(e).__name__}: {e}); meta.json is filled")
    if args.render:
        code, result = render(row["id"])
        print(json.dumps(result, default=str))
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())

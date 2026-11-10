#!/usr/bin/env python
"""Relabel a stored replay with the window-label stage and score it.

Runs web/field/window_labels.run_window_labels on a replay's meta.json and
source video: no GPU, no pipeline job, nothing written to the database or to
the replay's own folder. Tracks and positions are untouched, so any change in
the ground-truth score is the label stage alone.

This CALLS THE JEV API (about 2,500 requests, ~$0.26 per match).

Usage
-----
    .venv/bin/python scripts/relabel_replay.py --replay a0bbc991 --score
    .venv/bin/python scripts/relabel_replay.py --replay a0bbc991 --score --lineup-min-p 0.65
    .venv/bin/python scripts/relabel_replay.py --replay caa236ee \
        --calibration-version 20260630-210916-570400 --save-lineup /tmp/qm11_lineup.jpg

A replay made before a calibration refit must be reprojected with the version
its job used (--calibration-version); otherwise the crops miss the robots.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from scripts.score_ground_truth import (  # noqa: E402
    REPLAYS_DIR, _probe_fps, _resolve, _shift_meta, match_roster, replay_row)
from web import jev_reader  # noqa: E402
from web.field import store as field_store  # noqa: E402
from web.field import window_labels as wl  # noqa: E402
from web.field.composite_layout import load_layout  # noqa: E402
from web.field.ground_truth import GT_DIR, list_gt, video_fingerprint  # noqa: E402
from web.field.gt_scoring import replay_from_meta, score_match  # noqa: E402
from web.field.match_order import select_epoch  # noqa: E402

SCORE_SHIFT_M = 0.475


def load_inputs(replay_id: str, calibration_version: str | None):
    row = replay_row(replay_id)
    meta_p = REPLAYS_DIR / row["id"] / "meta.json"
    if not meta_p.exists():
        meta_p = _resolve(row.get("meta_path"))
    if meta_p is None or not meta_p.exists():
        raise SystemExit(f"meta.json missing for replay {row['id']}")
    video = _resolve(row.get("source_video_path"))
    if video is None:
        raise SystemExit(f"source video missing for replay {row['id']}")
    match_key = row["match_key"]
    event_key = match_key.split("_")[0]
    versions = field_store.load_versions(event_key, "main")
    if calibration_version:
        chosen = next((v for v in versions if v["version_id"] == calibration_version), None)
        if chosen is None:
            raise SystemExit(f"no main calibration version {calibration_version!r} for {event_key}")
    else:
        chosen = select_epoch(versions, match_key)
    if not chosen:
        raise SystemExit(f"no main calibration for {match_key}")
    roster = match_roster(match_key)
    return row, json.loads(meta_p.read_text()), video, chosen["calibration"], roster


def job_roster(match_key: str) -> tuple[list[str], list[str]]:
    """(red, blue) team numbers in the order a job gets them: match_robots by
    alliance then driver-station position, as database.get_match_by_key /
    app._load_db_roster read them. Roster order sets the option order in every
    request and the lineup-letter shuffle, so the tool must match the job.
    Read-only."""
    import sqlite3
    con = sqlite3.connect(f"file:{REPO / 'frc_scorer.db'}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT r.alliance, r.team_number FROM match_robots r "
            "JOIN matches m ON m.id = r.match_id "
            "WHERE m.tba_match_key = ? ORDER BY r.alliance, r.position",
            (match_key,)).fetchall()
    finally:
        con.close()
    red = [str(t) for a, t in rows if a == "red" and t is not None]
    blue = [str(t) for a, t in rows if a == "blue" and t is not None]
    return red, blue


def score_lines(meta: dict, row: dict, video: Path) -> list[str]:
    """Identity metrics vs hand ground truth, full match and auto."""
    match_key = row["match_key"]
    docs = [d for d in list_gt(match_key, GT_DIR)
            if isinstance(d, dict) and d.get("team") and "keyframes" in d]
    if not docs:
        return [f"no ground truth for {match_key}"]
    shifted = _shift_meta(meta, match_key.split("_")[0], match_key, SCORE_SHIFT_M)
    rt = replay_from_meta(shifted, src_fps=_probe_fps(video), fingerprint=video_fingerprint(video))
    def pct(v):                      # a rate is None when its period has no samples
        return "-" if v is None else f"{v:.0%}"

    out = []
    for period in ("full", "auto"):
        res = score_match(docs, rt, period=period, roster=match_roster(match_key))
        h, i = res["headline"], res["identity"]
        out.append(f"{period}: right {pct(h['label_correct'])}  wrong {pct(h['label_wrong'])}  "
                   f"unlabelled {pct(h['unlabeled'])}  misID {i['misid_s']:.0f}s "
                   f"(cross {i['misid_cross_s']:.0f} / same {i['misid_same_s']:.0f})  "
                   f"cov {pct(h['coverage'])}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", required=True, help="replay id or unique prefix")
    ap.add_argument("--lineup-min-p", type=float, default=0.8)
    ap.add_argument("--calibration-version", default=None,
                    help="main calibration version id the job used (default: select by match)")
    ap.add_argument("--ground-center-offset", type=float, default=0.0,
                    help="the job's ground_center_offset_m (0 unless the job set it)")
    ap.add_argument("--score", action="store_true", help="print GT identity metrics before/after")
    ap.add_argument("--out", default=None, help="write the relabelled meta.json here")
    ap.add_argument("--save-lineup", default=None, help="write the lineup card JPEG here")
    args = ap.parse_args()

    row, meta, video, cal, _roster = load_inputs(args.replay, args.calibration_version)
    fps = _probe_fps(video)
    if not fps:
        raise SystemExit(f"cannot read fps from {video}")
    red, blue = job_roster(row["match_key"])
    if not red and not blue:
        raise SystemExit(f"no roster rows in match_robots for {row['match_key']}")
    session = jev_reader.JevSession(jev_reader.DEFAULT_MODEL, timeout_s=30.0, retries=1)
    if not session.available:
        raise SystemExit(f"Jev unavailable: {session.disabled_reason}")
    params = wl.Params(
        fps=fps,
        red=red,
        blue=blue,
        view=tuple(load_layout()["main"].crop),
        ground_center_offset_m=args.ground_center_offset,
        lineup_min_p=args.lineup_min_p,
        deadline=time.monotonic() + 2 * wl.STAGE_TIMEOUT_S,
        seed=zlib.crc32(row["id"].encode()))
    print(f"replay {row['id'][:8]}  {row['match_key']}  video {video.name}")
    t0 = time.time()
    result = wl.run_window_labels(meta["frames"], str(video), cal, session, params)
    print(f"stage: {json.dumps(result['stats'])}  ({time.time() - t0:.0f}s)")
    if result["aborted"]:
        raise SystemExit(f"stage aborted: {result['aborted']}")
    relabelled = copy.deepcopy(meta)
    wl.apply_labels(relabelled["frames"], result["labels"])
    wl.restamp_events(relabelled["frames"], relabelled.get("events") or [])
    relabelled.setdefault("summary", {})["window_labels"] = result["stats"]
    if args.save_lineup and result["lineup_jpeg"]:
        Path(args.save_lineup).write_bytes(result["lineup_jpeg"])
        print(f"lineup card -> {args.save_lineup}")
    if args.score:
        print("pipeline labels:")
        for line in score_lines(meta, row, video):
            print("  " + line)
        print(f"window labels (lineup >= {args.lineup_min_p}):")
        for line in score_lines(relabelled, row, video):
            print("  " + line)
    if args.out:
        Path(args.out).write_text(json.dumps(relabelled))
        print(f"relabelled meta -> {args.out}")


if __name__ == "__main__":
    main()

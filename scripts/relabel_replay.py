#!/usr/bin/env python
"""Relabel a stored replay with the window-label stage and score it.

Runs web/field/window_labels.run_window_labels on a replay's meta.json and
source video: no GPU, no pipeline job, nothing written to the database or to
the replay's own folder (with --store-write, only this replay's anchors in
the event lineup store). Tracks and positions are untouched, so any change in
the ground-truth score is the label stage alone.

This CALLS THE JEV API (about 2,500 requests, ~$0.26 per match; with a digit
cache only the lineup pass).

Usage
-----
    .venv/bin/python scripts/relabel_replay.py --replay a0bbc991 --score
    .venv/bin/python scripts/relabel_replay.py --replay a0bbc991 --score --lineup-min-p 0.65
    .venv/bin/python scripts/relabel_replay.py --replay caa236ee \
        --calibration-version 20260630-210916-570400 --save-lineup /tmp/qm11_lineup.jpg
    .venv/bin/python scripts/relabel_replay.py --replay a0bbc991 --score --event-lineup \
        --namespace eval-full --digit-cache data/eval/event_lineup/digit_cache/a0bbc991.json
    .venv/bin/python scripts/relabel_replay.py --replay a0bbc991 --score --deinterlace \
        --reader student      # S1 student first, Jev only where S1 is undecided (GPU)

Per-replay inputs come from the replay (event-lineup spec section 5b):
ground_center_offset_m from summary.config (0.0 when the replay records no
config), the seed from crc32(job id), the roster in DB driver-station order,
and the main calibration version from summary.calibration_version (else
select_epoch, with a warning). --calibration-version and
--ground-center-offset override.

Event lineup (docs/superpowers/specs/2026-11-10-event-lineup-design.md):
--event-lineup borrows lineup rows from the anchor store (namespace
--namespace, default live); --before-match N borrows only from matches
numbered below N; --store-write writes this replay's anchors to that
namespace (off by default: read-only). --digit-cache FILE reuses one digit
pass (created there on the first run). --no-elim and --alliance-rows are
ablations.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from scripts.score_ground_truth import (  # noqa: E402
    REPLAYS_DIR, _probe_fps, _resolve, _shift_meta, match_roster, replay_row)
from web import jev_reader  # noqa: E402
from web.field import event_anchors as ea  # noqa: E402
from web.field import store as field_store  # noqa: E402
from web.field import window_labels as wl  # noqa: E402
from web.field.composite_layout import load_layout  # noqa: E402
from web.field.ground_truth import GT_DIR, list_gt, video_fingerprint  # noqa: E402
from web.field.gt_scoring import replay_from_meta, score_match  # noqa: E402
from web.field.match_order import select_epoch  # noqa: E402

SCORE_SHIFT_M = 0.475
FULL_MATCH_MIN_S = 150.0
DB_PATH = REPO / "frc_scorer.db"


def _knob_defaults() -> dict:
    from web.job_config import KNOBS
    return {k.name: k.default for k in KNOBS}


def pct(v) -> str:
    """A rate as a percentage; "-" when its period has no samples (None)."""
    return "-" if v is None else f"{v:.0%}"


def provenance(summary) -> dict:
    """What a replay records about its job (spec section 5b), with the
    fallbacks for replays made before the app recorded it: window_labels on
    = summary.window_labels present; ground_center_offset_m 0.0; stopped
    False (accepted); calibration version None (the caller then uses
    select_epoch). A full match is not auto_period_only (when a config is
    recorded) and lasts at least FULL_MATCH_MIN_S. ``fallbacks`` names the
    fields that were missing."""
    s = summary or {}
    cfg = s.get("config")
    fallbacks = []
    if isinstance(cfg, dict):
        # summary.config holds only the non-default knobs: a missing key is
        # the knob's code default, not False / 0.0
        d = _knob_defaults()
        # window_labels' default changed (OFF -> ON, Nov 11), so a missing key
        # is decided by evidence: the stage leaves summary.window_labels.
        wl_on = bool(cfg["window_labels"]) if "window_labels" in cfg else "window_labels" in s
        auto_only = bool(cfg.get("auto_period_only", d.get("auto_period_only", False)))
        gco = float(cfg.get("ground_center_offset_m", d.get("ground_center_offset_m", 0.0)) or 0.0)
    else:
        fallbacks.append("config")
        wl_on, auto_only, gco = "window_labels" in s, False, 0.0
    duration = s.get("duration_sec")
    full = (not auto_only) and duration is not None and float(duration) >= FULL_MATCH_MIN_S
    if "stopped" in s:
        stopped = bool(s["stopped"])
    else:
        fallbacks.append("stopped")
        stopped = False
    version = s.get("calibration_version") or None
    if version is None:
        fallbacks.append("calibration_version")
    return {"window_labels_on": wl_on, "full_match": full, "ground_center_offset_m": gco,
            "stopped": stopped, "calibration_version": version, "fallbacks": fallbacks}


def choose_calibration(versions: list, match_key: str, override: Optional[str] = None,
                       recorded: Optional[str] = None) -> tuple:
    """(version dict or None, warning or None): --calibration-version first,
    then the version the job recorded, then select_epoch (with a warning)."""
    if override:
        v = next((v for v in versions if v["version_id"] == override), None)
        if v is None:
            raise SystemExit(f"no main calibration version {override!r} for {match_key}")
        return v, None
    if recorded:
        v = next((v for v in versions if v["version_id"] == recorded), None)
        if v is not None:
            return v, None
        why = f"recorded calibration version {recorded!r} not found"
    else:
        why = "the replay records no calibration version"
    v = select_epoch(versions, match_key)
    return v, f"{why}; select_epoch -> {v['version_id'] if v else None}"


def calibration_for(match_key: str, meta: dict, override: Optional[str] = None) -> tuple:
    """(main calibration version dict, warning or None) for a stored replay."""
    versions = field_store.load_versions(match_key.split("_")[0], "main")
    chosen, warning = choose_calibration(versions, match_key, override,
                                         provenance(meta.get("summary"))["calibration_version"])
    if not chosen:
        raise SystemExit(f"no main calibration for {match_key}")
    return chosen, warning


def load_inputs(replay_id: str, calibration_version: Optional[str] = None):
    """(replay row, meta, source video, main calibration) for a stored replay."""
    row = replay_row(replay_id)
    meta_p = REPLAYS_DIR / row["id"] / "meta.json"
    if not meta_p.exists():
        meta_p = _resolve(row.get("meta_path"))
    if meta_p is None or not meta_p.exists():
        raise SystemExit(f"meta.json missing for replay {row['id']}")
    video = _resolve(row.get("source_video_path"))
    if video is None:
        raise SystemExit(f"source video missing for replay {row['id']}")
    meta = json.loads(meta_p.read_text())
    chosen, warning = calibration_for(row["match_key"], meta, calibration_version)
    if warning:
        print(f"WARNING: {warning}")
    return row, meta, video, chosen["calibration"]


def job_roster(match_key: str, db_path=None) -> tuple[list[str], list[str]]:
    """(red, blue) team numbers in the order a job gets them: match_robots by
    alliance then driver-station position, as database.get_match_by_key /
    app._load_db_roster read them. Roster order sets the option order in every
    request and the lineup-letter shuffle, so the tool must match the job.
    Read-only."""
    import sqlite3
    con = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True)
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


def digit_cache_for(meta: dict, path: Optional[str]) -> tuple:
    """(digit-call cache or None, origin): an existing --digit-cache file
    first, then the replay's summary.window_labels_digit_calls."""
    if path and Path(path).exists():
        return json.loads(Path(path).read_text()), "file"
    cached = (meta.get("summary") or {}).get("window_labels_digit_calls")
    return (cached, "summary") if cached else (None, "none")


@dataclass
class Options:
    lineup_min_p: float = 0.8
    ground_center_offset: Optional[float] = None    # None: from the replay
    event_lineup: bool = False
    namespace: str = ea.LIVE
    before_match: Optional[int] = None              # borrow from matches numbered below this
    store_write: bool = False
    digit_cache: Optional[str] = None               # file: read, or created on first run
    use_elimination: bool = True
    alliance_rows: bool = False
    deinterlace: bool = False                       # window_labels_deinterlace
    reader: str = "student_only"                    # window_labels_reader (live default)


def relabel_one(row: dict, meta: dict, video, cal, red: list, blue: list, opts: Options,
                session, *, fps: float, view, frame_source=None, db_path=None,
                anchor_root=None, student=None) -> dict:
    """Run the stage on one stored replay; ``meta`` is never changed. With
    opts.event_lineup the borrowed anchors come from opts.namespace; with
    opts.store_write the replay's anchors then replace its match's rows
    there (after the labels are applied; a failed write raises). Returns
    {"result": stage result, "relabelled": meta copy with the new labels
    (None on abort), "stats": the stage stats plus the store counters,
    "cache_origin", "gco", "borrow_fetch_error"}."""
    match_key = row["match_key"]
    event_key = match_key.split("_")[0]
    prov = provenance(meta.get("summary"))
    gco = (prov["ground_center_offset_m"] if opts.ground_center_offset is None
           else float(opts.ground_center_offset))
    cache, origin = digit_cache_for(meta, opts.digit_cache)
    borrowed, fetch_error = {}, None
    if opts.event_lineup:
        cutoff = None if opts.before_match is None else int(opts.before_match) - 1
        try:
            borrowed = ea.fetch(opts.namespace, event_key, red + blue, match_key, cutoff,
                                db_path=db_path, root=anchor_root)
        except Exception as e:
            fetch_error = f"{type(e).__name__}: {e}"
            print(f"WARNING: borrow fetch failed, no borrowed rows: {fetch_error}")
    params = wl.Params(
        fps=fps, red=red, blue=blue, view=tuple(view), ground_center_offset_m=gco,
        lineup_min_p=opts.lineup_min_p,
        deadline=time.monotonic() + 2 * wl.STAGE_TIMEOUT_S,
        seed=zlib.crc32(row["id"].encode()),
        event_lineup=opts.event_lineup, borrowed=borrowed,
        this_colors={**{t: "red" for t in red}, **{t: "blue" for t in blue}},
        use_elimination=opts.use_elimination, rows_alliance_only=opts.alliance_rows,
        digit_calls=cache, deinterlace=opts.deinterlace, reader=opts.reader)
    result = wl.run_window_labels(meta["frames"], str(video), cal, session, params,
                                  frame_source=frame_source, student=student)
    out = {"result": result, "relabelled": None, "stats": dict(result["stats"]),
           "cache_origin": origin, "gco": gco, "borrow_fetch_error": fetch_error}
    if result["aborted"]:
        return out
    if opts.digit_cache and not Path(opts.digit_cache).exists():
        p = Path(opts.digit_cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(result["digit_calls"]))
        os.replace(tmp, p)
    relabelled = copy.deepcopy(meta)
    wl.apply_labels(relabelled["frames"], result["labels"])
    wl.restamp_events(relabelled["frames"], relabelled.get("events") or [])
    stats = out["stats"]
    if opts.event_lineup:
        written = {ea.KIND_DIGITS: 0, ea.KIND_ELIM: 0}
        if opts.store_write:
            anchors = list(result["anchors_out"])
            ea.replace(opts.namespace, event_key, match_key, anchors, replay_id=row["id"],
                       db_path=db_path, root=anchor_root)
            for a in anchors:
                written[a.kind] += 1
        stats.update(anchors_written=written[ea.KIND_DIGITS],
                     elim_anchors_written=written[ea.KIND_ELIM], borrow_fetch_error=fetch_error)
    summary = relabelled.setdefault("summary", {})
    summary["window_labels"] = stats
    summary["window_labels_digit_calls"] = {          # wide_p is rebuilt from calls on load
        k: v for k, v in (result["digit_calls"] or {}).items() if k != "wide_p"} or None
    out["relabelled"] = relabelled
    return out


def score_numbers(meta: dict, row: dict, video) -> Optional[dict]:
    """{period: {right, wrong, unlabelled, misid_s, misid_cross_s,
    misid_same_s, coverage}} vs hand ground truth, positions shifted
    SCORE_SHIFT_M (= score_ground_truth.py --score-shift 0.475). None when
    the match has no GT."""
    match_key = row["match_key"]
    docs = [d for d in list_gt(match_key, GT_DIR)
            if isinstance(d, dict) and d.get("team") and "keyframes" in d]
    if not docs:
        return None
    shifted = _shift_meta(meta, match_key.split("_")[0], match_key, SCORE_SHIFT_M)
    rt = replay_from_meta(shifted, src_fps=_probe_fps(video), fingerprint=video_fingerprint(video))
    out = {}
    for period in ("full", "auto"):
        res = score_match(docs, rt, period=period, roster=match_roster(match_key))
        h, i = res["headline"], res["identity"]
        out[period] = {"right": h["label_correct"], "wrong": h["label_wrong"],
                       "unlabelled": h["unlabeled"], "misid_s": i["misid_s"],
                       "misid_cross_s": i["misid_cross_s"], "misid_same_s": i["misid_same_s"],
                       "coverage": h["coverage"]}
    return out


def score_lines(meta: dict, row: dict, video) -> list[str]:
    """Identity metrics vs hand ground truth, full match and auto."""
    nums = score_numbers(meta, row, video)
    if nums is None:
        return [f"no ground truth for {row['match_key']}"]

    return [f"{period}: right {pct(n['right'])}  wrong {pct(n['wrong'])}  "
            f"unlabelled {pct(n['unlabelled'])}  misID {n['misid_s']:.0f}s "
            f"(cross {n['misid_cross_s']:.0f} / same {n['misid_same_s']:.0f})  "
            f"cov {pct(n['coverage'])}" for period, n in nums.items()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", required=True, help="replay id or unique prefix")
    ap.add_argument("--lineup-min-p", type=float, default=0.8)
    ap.add_argument("--calibration-version", default=None,
                    help="main calibration version id (default: the one the replay recorded, "
                         "else select by match)")
    ap.add_argument("--ground-center-offset", type=float, default=None,
                    help="override the job's ground_center_offset_m (default: from the replay)")
    ap.add_argument("--score", action="store_true", help="print GT identity metrics before/after")
    ap.add_argument("--out", default=None, help="write the relabelled meta.json here")
    ap.add_argument("--save-lineup", default=None,
                    help="write the lineup card JPEG here (blue card: <stem>_blue<suffix>)")
    ap.add_argument("--event-lineup", action="store_true",
                    help="borrow lineup rows from the event anchor store")
    ap.add_argument("--namespace", default=ea.LIVE, help="anchor store namespace (default live)")
    ap.add_argument("--before-match", type=int, default=None, metavar="N",
                    help="borrow only from matches numbered below N")
    ap.add_argument("--store-write", action="store_true",
                    help="write this replay's anchors to --namespace (default: read-only)")
    ap.add_argument("--digit-cache", default=None, metavar="FILE",
                    help="reuse the digit pass from FILE, or create FILE on the first run")
    ap.add_argument("--no-elim", action="store_true", help="ablation: no elimination anchors")
    ap.add_argument("--alliance-rows", action="store_true",
                    help="ablation: one lineup card per alliance")
    ap.add_argument("--deinterlace", action="store_true",
                    help="deinterlace combed tiles (the window_labels_deinterlace knob); "
                         "digit calls cached without it are not reused")
    ap.add_argument("--reader", choices=wl.READERS, default="student_only",
                    help="digit reader (the window_labels_reader knob): jev, student (S1 first, "
                         "Jev only for windows S1 leaves undecided) or student_only (no Jev "
                         "digit calls; the live default); the lineup pass is Jev in every mode")
    args = ap.parse_args(argv)
    if args.store_write and not args.event_lineup:
        ap.error("--store-write needs --event-lineup")
    try:
        ea.check_name(args.namespace, "namespace")
    except ValueError as e:
        ap.error(str(e))

    row, meta, video, cal = load_inputs(args.replay, args.calibration_version)
    fps = _probe_fps(video)
    if not fps:
        raise SystemExit(f"cannot read fps from {video}")
    red, blue = job_roster(row["match_key"])
    if not red and not blue:
        raise SystemExit(f"no roster rows in match_robots for {row['match_key']}")
    session = jev_reader.JevSession(jev_reader.DEFAULT_MODEL, timeout_s=30.0, retries=1)
    if not session.available:
        raise SystemExit(f"Jev unavailable: {session.disabled_reason}")
    if args.store_write and args.namespace == ea.LIVE:
        print("WARNING: writing the live namespace")
    opts = Options(lineup_min_p=args.lineup_min_p, ground_center_offset=args.ground_center_offset,
                   event_lineup=args.event_lineup, namespace=args.namespace,
                   before_match=args.before_match, store_write=args.store_write,
                   digit_cache=args.digit_cache, use_elimination=not args.no_elim,
                   alliance_rows=args.alliance_rows, deinterlace=args.deinterlace,
                   reader=args.reader)
    print(f"replay {row['id'][:8]}  {row['match_key']}  video {video.name}")
    t0 = time.time()
    out = relabel_one(row, meta, video, cal, red, blue, opts, session, fps=fps,
                      view=load_layout(event_key=row["match_key"].split("_")[0])["main"].crop)
    result = out["result"]
    print(f"stage: {json.dumps(out['stats'])}  ({time.time() - t0:.0f}s, digit cache "
          f"{out['cache_origin']}, ground_center_offset_m {out['gco']})")
    if result["aborted"]:
        raise SystemExit(f"stage aborted: {result['aborted']}")
    relabelled = out["relabelled"]
    if args.save_lineup:
        for jpg, path in ((result["lineup_jpeg"], Path(args.save_lineup)),
                          (result.get("lineup_jpeg_blue"),
                           Path(args.save_lineup).with_name(Path(args.save_lineup).stem + "_blue"
                                                           + Path(args.save_lineup).suffix))):
            if jpg:
                path.write_bytes(jpg)
                print(f"lineup card -> {path}")
    if args.score:
        print("pipeline labels:")
        for line in score_lines(meta, row, video):
            print("  " + line)
        print(f"window labels (lineup >= {args.lineup_min_p}"
              + (f", event lineup {args.namespace}" if args.event_lineup else "") + "):")
        for line in score_lines(relabelled, row, video):
            print("  " + line)
    if args.out:
        Path(args.out).write_text(json.dumps(relabelled))
        print(f"relabelled meta -> {args.out}")


if __name__ == "__main__":
    main()

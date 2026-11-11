#!/usr/bin/env python
"""Relabel an event's matches against the event lineup store, in schedule order.

Spec: docs/superpowers/specs/2026-11-10-event-lineup-design.md section 6.
For every selected match it runs scripts/relabel_replay.relabel_one on the
match's latest qualifying replay (status complete, full match, not
user-stopped, window_labels on; read from summary.config / summary.stopped
with the section 5b fallbacks) and writes into --out-dir: each relabelled
meta.json (<match_key>_<replay8>.meta.json), results.jsonl (one row per
match: stats, row sources, unlabelled share, GT score) and run.json
(arguments, selection, skips). Replays and the database are never written,
except the --namespace anchor rows with --store-write.

There is NO --write (write-back into the replays): that is phase 2, built
only after the evaluation passes (spec section 6).

CALLS THE JEV API: the lineup pass always; the digit pass only for replays
without a digit cache (summary.window_labels_digit_calls, or
--digit-cache-dir/<replay_id>.json, which the first run creates).

Modes: incremental (default) relabels one match at a time, borrowing only
from matches numbered below it, and with --store-write writes its anchors
before the next match (stopping at the first failed match); full borrows
from every other match in the store (the end-of-day pass) and runs
--workers matches at once.

    .venv/bin/python scripts/event_relabel.py --event 2026cur --list
    .venv/bin/python scripts/event_relabel.py --event 2026cur --matches qm8,qm11 \
        --mode incremental --namespace eval-inc --store-write --out-dir /tmp/inc
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.relabel_replay as rr  # noqa: E402  (loads .env)
from scripts.score_ground_truth import REPLAYS_DIR, _probe_fps, _resolve  # noqa: E402
from web import jev_reader  # noqa: E402
from web.field import event_anchors as ea  # noqa: E402
from web.field.composite_layout import load_layout  # noqa: E402
from web.field.match_order import match_sort_key  # noqa: E402

EVAL_ROOT = REPO / "data" / "eval" / "event_lineup"
DEFAULT_CACHE_DIR = EVAL_ROOT / "digit_cache"
_COLS = "id, created_at, match_key, source_video_path, meta_path, status"


def match_keys(event: str, matches: Optional[str], con) -> list:
    """The event's match keys in schedule order: ``matches`` ("qm1,qm8" or
    full keys), else every match that has a replay."""
    if matches:
        keys = {m if m.startswith(event + "_") else f"{event}_{m}"
                for m in (x.strip() for x in matches.split(",")) if m}
    else:
        keys = {r[0] for r in con.execute(
            "SELECT DISTINCT match_key FROM replays WHERE match_key IS NOT NULL")
            if str(r[0]).startswith(event + "_")}
    return sorted(keys, key=match_sort_key)


def _meta_path(row: dict, replays_dir) -> Optional[Path]:
    p = Path(replays_dir) / row["id"] / "meta.json"
    return p if p.exists() else _resolve(row.get("meta_path"))


def replay_reason(row: dict, summary: dict) -> Optional[str]:
    """None when the replay qualifies (spec section 6), else why not."""
    if row.get("status") != "complete":
        return f"status {row.get('status')}"
    prov = rr.provenance(summary)
    if not prov["full_match"]:
        return "not a full match"
    if prov["stopped"]:
        return "user-stopped"
    if not prov["window_labels_on"]:
        return "window_labels off"
    return None


def select_replays(event: str, matches: Optional[str], *, db_path, replays_dir) -> tuple:
    """([{"match_key", "row", "meta_path", "video"}] in schedule order,
    [(match_key, why it was skipped)]). Per match: the latest qualifying
    replay. Read-only."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    chosen, skipped = [], []
    try:
        for mk in match_keys(event, matches, con):
            rows = [dict(r) for r in con.execute(
                f"SELECT {_COLS} FROM replays WHERE match_key = ? "
                "ORDER BY created_at DESC, id DESC", (mk,))]
            why, pick = [], None
            for row in rows:
                mp, video = _meta_path(row, replays_dir), _resolve(row.get("source_video_path"))
                if mp is None or not mp.exists():
                    why.append(f"{row['id'][:8]}: no meta.json")
                    continue
                if video is None:
                    why.append(f"{row['id'][:8]}: no source video")
                    continue
                reason = replay_reason(row, json.loads(mp.read_text()).get("summary") or {})
                if reason:
                    why.append(f"{row['id'][:8]}: {reason}")
                    continue
                pick = {"match_key": mk, "row": row, "meta_path": mp, "video": video}
                break
            if pick:
                chosen.append(pick)
            else:
                skipped.append((mk, "; ".join(why) or "no replays"))
    finally:
        con.close()
    return chosen, skipped


def before_match_for(mode: str, match_key: str) -> Optional[int]:
    """incremental: borrow only from matches numbered below this one."""
    if mode != "incremental":
        return None
    rank = ea.schedule_rank(match_key)
    if rank is None:                # no cutoff would mean borrow-everything
        raise ValueError(f"cannot order {match_key!r} in the schedule (incremental needs it)")
    return rank


def run_all(items: list, fn, mode: str, workers: int, stop_on_error: bool = False) -> list:
    """fn(item) for every item, results in item order. incremental: one at a
    time in order (each match must see what the previous one wrote), and
    with ``stop_on_error`` nothing after the first result carrying "error";
    full: up to ``workers`` at once."""
    if mode == "incremental" or workers <= 1:
        out = []
        for it in items:
            out.append(fn(it))
            if stop_on_error and out[-1].get("error"):
                break
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def unlabelled_share(frames: list) -> Optional[float]:
    """Share of slot rows (slot-time) without a team label."""
    rows = [r for f in frames for r in f.get("robots") or [] if r.get("slot")]
    return None if not rows else sum(r.get("team_number") is None for r in rows) / len(rows)


def result_row(item: dict, out: dict, out_dir, score: bool, seconds: float) -> dict:
    """One results.jsonl row; writes the relabelled meta.json next to it."""
    row, mk = item["row"], item["match_key"]
    res = out["result"]
    rec = {"match_key": mk, "replay_id": row["id"], "seconds": round(seconds, 1),
           "aborted": res.get("aborted"), "stats": out.get("stats") or res.get("stats"),
           "cache_origin": out.get("cache_origin"), "gco": out.get("gco"),
           "row_sources": res.get("row_sources") or {}, "unlabelled_share": None,
           "meta_out": None, "score": None}
    if out.get("relabelled") is not None:
        p = Path(out_dir) / f"{mk}_{row['id'][:8]}.meta.json"
        p.write_text(json.dumps(out["relabelled"]))
        rec["meta_out"] = str(p)
        rec["unlabelled_share"] = unlabelled_share(out["relabelled"].get("frames") or [])
        if score:
            try:
                rec["score"] = rr.score_numbers(out["relabelled"], row, item["video"])
            except Exception as e:      # a scoring failure is not a stage failure
                rec["score_error"] = f"{type(e).__name__}: {e}"
    return rec


def relabel_match(item: dict, args, out_dir) -> dict:
    """Relabel one selected match; never raises (errors come back as "error")."""
    row, mk = item["row"], item["match_key"]
    t0 = time.time()
    try:
        meta = json.loads(Path(item["meta_path"]).read_text())
        chosen, warning = rr.calibration_for(mk, meta)
        if warning:
            print(f"[{mk}] WARNING: {warning}", flush=True)
        red, blue = rr.job_roster(mk)
        if not red and not blue:
            raise RuntimeError("no roster rows in match_robots")
        fps = _probe_fps(item["video"])
        if not fps:
            raise RuntimeError(f"cannot read fps from {item['video']}")
        session = jev_reader.JevSession(jev_reader.DEFAULT_MODEL, timeout_s=30.0, retries=1)
        if not session.available:
            raise RuntimeError(f"Jev unavailable: {session.disabled_reason}")
        opts = rr.Options(lineup_min_p=args.lineup_min_p,
                          event_lineup=not args.no_event_lineup, namespace=args.namespace,
                          before_match=before_match_for(args.mode, mk),
                          store_write=args.store_write,
                          digit_cache=str(Path(args.digit_cache_dir) / f"{row['id']}.json"),
                          use_elimination=not args.no_elim, alliance_rows=args.alliance_rows,
                          reader=args.reader)
        out = rr.relabel_one(row, meta, item["video"], chosen["calibration"], red, blue, opts,
                             session, fps=fps,
                             view=load_layout(event_key=mk.split("_")[0])["main"].crop)
        rec = result_row(item, out, out_dir, args.score, time.time() - t0)
        if out.get("borrow_fetch_error"):   # would silently turn this arm into arm 1
            rec["error"] = f"borrow fetch failed: {out['borrow_fetch_error']}"
    except (Exception, SystemExit) as e:
        rec = {"match_key": mk, "replay_id": row["id"], "error": f"{type(e).__name__}: {e}"}
    if rec.get("aborted"):
        rec["error"] = f"stage aborted: {rec['aborted']}"
    st = rec.get("stats") or {}
    print(f"[{mk}] {row['id'][:8]} {rec.get('seconds', 0):.0f}s cache {rec.get('cache_origin')} req {st.get('requests')} "
          f"digits {st.get('digit_named')} lineup {st.get('lineup_named')} rows "
          f"{st.get('lineup_rows')} (own {st.get('rows_own')} / borrowed {st.get('rows_borrowed')}"
          f" / elim {st.get('rows_elim')}) unlabelled {rec.get('unlabelled_share')}"
          + (f"  SCORE ERROR {rec['score_error']}" if rec.get("score_error") else "")
          + (f"  ERROR {rec['error']}" if rec.get("error") else ""), flush=True)
    return rec


def arm_tag(args) -> str:
    """own (no event lineup) / event, plus noelim / alliance ablations."""
    tag = ["own" if args.no_event_lineup else "event"]
    if args.no_elim:
        tag.append("noelim")
    if args.alliance_rows:
        tag.append("alliance")
    return "-".join(tag)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event", required=True, help="event key, e.g. 2026cur")
    ap.add_argument("--matches", default=None,
                    help="comma list (qm1,qm8,...); default: every match with a replay")
    ap.add_argument("--namespace", default=ea.LIVE, help="anchor store namespace (default live)")
    ap.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    ap.add_argument("--store-write", action="store_true",
                    help="write each match's anchors to --namespace (default: read-only)")
    ap.add_argument("--score", action="store_true",
                    help="score matches that have hand GT (positions shifted 0.475 m)")
    ap.add_argument("--no-elim", action="store_true", help="ablation: no elimination anchors")
    ap.add_argument("--alliance-rows", action="store_true",
                    help="ablation: one lineup card per alliance")
    ap.add_argument("--no-event-lineup", action="store_true",
                    help="own anchors only: today's stage (evaluation arm 1)")
    ap.add_argument("--reader", choices=("jev", "student", "student_only"), default="student_only",
                    help="digit reader (the window_labels_reader knob; default = the live default)")
    ap.add_argument("--lineup-min-p", type=float, default=0.8,
                    help="lineup acceptance floor; above 1 disables the lineup pass")
    ap.add_argument("--workers", type=int, default=4, help="parallel matches in full mode")
    ap.add_argument("--digit-cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help="digit caches, one <replay_id>.json each (created on first use)")
    ap.add_argument("--out-dir", default=None,
                    help="default data/eval/event_lineup/runs/<namespace>_<mode>_<time>")
    ap.add_argument("--list", action="store_true", help="print the selection and stop (no API)")
    ap.add_argument("--reset", action="store_true",
                    help="first delete --namespace's anchors for --event (never live)")
    args = ap.parse_args(argv)
    if args.store_write and args.no_event_lineup:
        ap.error("--store-write needs the event lineup (drop --no-event-lineup)")
    if args.reset and args.namespace == ea.LIVE:
        ap.error("--reset never clears the live namespace")
    try:
        ea.check_name(args.namespace, "namespace")
    except ValueError as e:
        ap.error(str(e))

    chosen, skipped = select_replays(args.event, args.matches, db_path=rr.DB_PATH,
                                     replays_dir=REPLAYS_DIR)
    print(f"{len(chosen)} matches selected, {len(skipped)} skipped")
    for it in chosen:
        print(f"  {it['match_key']:<16} {it['row']['id'][:8]}  {it['row']['created_at']}")
    for mk, why in skipped:
        print(f"  SKIP {mk}: {why}")
    if args.list or not chosen:
        return
    if args.store_write and args.namespace == ea.LIVE:
        print("WARNING: writing the live namespace")
    out_dir = Path(args.out_dir or EVAL_ROOT / "runs" /
                   f"{args.namespace}_{args.mode}_{arm_tag(args)}_{time.strftime('%Y%m%d-%H%M%S')}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} is not empty")
    if args.reset:
        n = ea.clear(args.namespace, args.event)
        print(f"reset: removed {n} anchors from {args.namespace}/{args.event}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "full" and args.store_write and args.workers > 1:
        print("NOTE: full mode with --store-write runs one match at a time (--workers 1)")
        args.workers = 1
    (out_dir / "run.json").write_text(json.dumps({
        "args": vars(args), "skipped": skipped,
        "selected": [{"match_key": it["match_key"], "replay_id": it["row"]["id"],
                      "created_at": it["row"]["created_at"]} for it in chosen]}, indent=1))
    t0 = time.time()
    rows = run_all(chosen, lambda it: relabel_match(it, args, out_dir), args.mode, args.workers,
                   stop_on_error=args.mode == "incremental" and args.store_write)
    with open(out_dir / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    failed = [r for r in rows if r.get("error")]
    print(f"done: {len(rows) - len(failed)} ok, {len(failed)} failed, "
          f"{time.time() - t0:.0f}s -> {out_dir}")
    if len(rows) < len(chosen):
        print(f"STOPPED after {rows[-1]['match_key']} (incremental --store-write); "
              f"{len(chosen) - len(rows)} matches not run")
    for r in rows:
        if r.get("score"):
            f = r["score"]["full"]
            print(f"  {r['match_key']} vs GT (full): right {rr.pct(f['right'])} wrong "
                  f"{rr.pct(f['wrong'])} unlabelled {rr.pct(f['unlabelled'])} "
                  f"misID {f['misid_s']:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Upload an event's scouting results to the Google Drive folder (what the
gdrive_upload knob does after each match, on demand: backfills, re-sends,
results pulled + imported from RunPod / vast.ai runs).

  .venv/bin/python scripts/upload_gdrive.py --check
  .venv/bin/python scripts/upload_gdrive.py 2026cur --dry-run
  .venv/bin/python scripts/upload_gdrive.py 2026cur                  # every match + event export
  .venv/bin/python scripts/upload_gdrive.py 2026cur --match 2026cur_qm8 --match 2026cur_qm9
  .venv/bin/python scripts/upload_gdrive.py 2026cur --match 2026cur_qm8 --no-event-export
  .venv/bin/python scripts/upload_gdrive.py 2026cur --video          # + each match's HUD video

Uses GDRIVE_FOLDER_ID / GDRIVE_RCLONE_REMOTE from .env (web/gdrive_upload.py has
the Drive layout). The match files, event export and videos come from the same
replay per match the scouting export uses (the newest best replay). Exits 1 if
any upload failed.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from web import gdrive_upload as gd  # noqa: E402
from web.scouting import api, export  # noqa: E402


def _video_for(md) -> tuple:
    """(local hud.mp4 path or None, reason) for the replay the export used."""
    row = None
    try:
        con = sqlite3.connect(str(api._db()))
        try:
            con.row_factory = sqlite3.Row
            r = con.execute("SELECT * FROM replays WHERE id = ?", (md.replay_id,)).fetchone()
            row = dict(r) if r else None
        finally:
            con.close()
    except sqlite3.Error as e:
        return None, f"replay row unreadable: {e}"
    rs = (row or {}).get("render_status")
    if rs not in (None, "rendered"):
        return None, f"HUD render {rs}"
    cands = [(row or {}).get("video_path"),
             str(Path(md.replay_dir) / "hud.mp4") if getattr(md, "replay_dir", None) else None]
    for c in cands:
        if c and Path(c).is_file():
            return c, ""
    return None, f"no hud.mp4 for replay {md.replay_id}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("event_key", nargs="?")
    ap.add_argument("--check", action="store_true",
                    help="only check the config and that the Drive folder is reachable")
    ap.add_argument("--match", action="append", default=[], metavar="MATCH_KEY",
                    help="only this match (repeatable)")
    ap.add_argument("--no-event-export", action="store_true",
                    help="don't refresh <event>_scouting_export.json")
    ap.add_argument("--video", action="store_true", help="also upload each match's HUD video")
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded")
    args = ap.parse_args(argv)
    if not args.check and not args.event_key:
        ap.error("event_key is required unless --check")

    cfg = gd.config()
    if args.check or not args.dry_run:
        if not cfg["configured"]:
            print(f"error: Google Drive upload not configured: {gd._why_not(cfg)}",
                  file=sys.stderr)
            return 1
    if args.check:
        try:
            entries = gd.list_folder()
        except gd.GDriveError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"{cfg['remote']} (folder {'set' if cfg['folder_id_set'] else 'unset'}, "
              f"rclone {cfg['rclone']}): reachable, {len(entries)} top-level entries")
        return 0

    ev = args.event_key
    matches, skipped = api._load(ev)
    matches = sorted(matches, key=export.match_order)
    known = {m.match_key for m in matches}
    if args.match:
        unknown = [mk for mk in args.match if mk not in known]
        if unknown:
            why = {s.get("match_key"): s.get("reason") for s in skipped}
            for mk in unknown:
                print(f"error: no analysed match {mk!r} in {ev}"
                      + (f" ({why[mk]})" if why.get(mk) else ""), file=sys.stderr)
            return 1
        wanted = set(args.match)
        matches = [m for m in matches if m.match_key in wanted]

    plan = []          # (label, remote path, thunk)
    for md in matches:
        mk = md.match_key
        plan.append(("match", gd.match_path(ev, mk),
                     lambda mk=mk: gd.upload_match(ev, mk)))
        if args.video:
            local, why = _video_for(md)
            if local:
                plan.append(("video", gd.video_path(ev, mk),
                             lambda mk=mk, local=local: gd.upload_video(ev, mk, local)))
            else:
                print(f"  skip video {mk}: {why}")
    if not args.no_event_export:
        plan.append(("event export", gd.event_export_path(ev),
                     lambda: gd.upload_event_export(ev)))

    if args.dry_run:
        print(f"{ev}: would upload {len(plan)} file(s) to {cfg['remote']} "
              f"(Drive folder from GDRIVE_FOLDER_ID):")
        for label, rel, _ in plan:
            print(f"  {label:<12} {rel}")
        return 0

    failed = 0
    for label, rel, thunk in plan:
        try:
            s = thunk()
            print(f"  ok    {label:<12} {rel}  ({s['bytes'] / 1e6:.2f} MB, {s['seconds']} s)")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {label:<12} {rel}: {type(e).__name__}: {e}")
    print(f"{ev}: {len(plan) - failed} uploaded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

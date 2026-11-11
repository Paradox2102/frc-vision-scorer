#!/usr/bin/env python
"""Render replay HUD videos with the final labels, one at a time
(docs/superpowers/specs/2026-11-10-hud-post-render-design.md section 7).

    .venv/bin/python scripts/render_replays.py --replays ID [ID ...]
    .venv/bin/python scripts/render_replays.py --since 2026-11-10 [--list]

A replay with a draw list (hud_post_render knob) is rendered exactly; one
without (recorded before the knob) is back-filled with approximate boxes. Runs
web.hud_render.run in this process, sequentially: it takes each replay's
render.lock (a replay the server is rendering is skipped, and the server skips
a replay this script holds) and records render_status / render_error itself.
The first render of a live-encoded replay keeps the original as hud.live.mp4.
CPU-heavy: do not run it while a GPU job is analysing.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from web import database, hud_render  # noqa: E402


def select(db_path, ids=None, since=None) -> list:
    """Replay rows (id, created_at, label, match_key, render_status,
    frame_count), oldest first: the given ``ids`` (in that order), or every
    replay created at or after ``since`` (an ISO date or datetime)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        con.row_factory = sqlite3.Row
        cols = "id, created_at, label, match_key, render_status, frame_count"
        if ids:
            rows = []
            for rid in ids:
                r = con.execute(f"SELECT {cols} FROM replays WHERE id = ?", (rid,)).fetchone()
                rows.append(dict(r) if r else {"id": rid, "missing": True})
            return rows
        return [dict(r) for r in con.execute(
            f"SELECT {cols} FROM replays WHERE created_at >= ? ORDER BY created_at, id",
            (since,)).fetchall()]
    finally:
        con.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--replays", nargs="+", metavar="ID", help="replay ids")
    which.add_argument("--since", metavar="ISO", help="every replay created at or after this")
    ap.add_argument("--db", default=None, help="SQLite DB (default: web.database.DB_PATH)")
    ap.add_argument("--list", action="store_true", help="print the selection, render nothing")
    args = ap.parse_args(argv)
    db_path = args.db or str(database.DB_PATH)

    rows = select(db_path, ids=args.replays, since=args.since)
    missing = [r["id"] for r in rows if r.get("missing")]
    rows = [r for r in rows if not r.get("missing")]
    for rid in missing:
        print(f"  {rid}: not found")
    print(f"{len(rows)} replay(s) selected")
    if args.list:
        for r in rows:
            print(f"  {r['id']}  {r['created_at']}  {r['match_key'] or r['label']}  "
                  f"render_status={r['render_status']}")
        return 0
    done = skipped = failed = 0
    for r in rows:
        # The same deadline the server gives a render (0.5 s/frame, 600-3600 s).
        code, out = hud_render.run(r["id"], db_path, deadline_s=hud_render.render_timeout_s(r))
        name = r["match_key"] or r["label"]
        if code == 0:
            done += 1
            notes = [f"{n['frames_relabelled']} frames relabelled on {lab}"
                     for lab, n in out["corrections"].items()]
            print(f"  {r['id'][:8]} {name}: rendered {out['frames']} frames in "
                  f"{out['seconds']}s ({out['source']}; thumbnail mismatches "
                  f"{out['decode_mismatches']}/{out['decode_checked']}"
                  + (f"; {'; '.join(notes)}" if notes else "") + ")")
            for w in out.get("warnings") or []:
                print(f"    warning: {w}")
        elif code == 3:
            skipped += 1
            print(f"  {r['id'][:8]} {name}: skipped ({out['error']})")
        else:
            failed += 1
            print(f"  {r['id'][:8]} {name}: FAILED {out['error']}")
    print(f"done: {done} rendered, {skipped} skipped, {failed} failed"
          + (f", {len(missing)} not found" if missing else ""))
    return 1 if failed or missing else 0


if __name__ == "__main__":
    sys.exit(main())

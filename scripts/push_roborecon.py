#!/usr/bin/env python3
"""Push an event's scouting metrics to RoboRecon (what auto-push does after each
match, on demand: backfills, re-sends, testing a new RoboRecon deployment).

  .venv/bin/python scripts/push_roborecon.py --check
  .venv/bin/python scripts/push_roborecon.py 2026cur
  .venv/bin/python scripts/push_roborecon.py 2026cur --metrics auto_paths
  .venv/bin/python scripts/push_roborecon.py 2026cur --team 2102
  .venv/bin/python scripts/push_roborecon.py 2026cur --url https://staging.roborecon.net

Uses ROBORECON_URL / ROBORECON_TOKEN from .env (``--url`` overrides the URL).
The token is never printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from web.scouting import api, export  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("event_key", nargs="?")
    ap.add_argument("--check", action="store_true", help="only test the connection + token")
    ap.add_argument("--metrics", help="comma list (default: every metric for the season)")
    ap.add_argument("--team", type=int, help="push just this team")
    ap.add_argument("--url", help="RoboRecon base URL (default ROBORECON_URL)")
    args = ap.parse_args()
    if not args.check and not args.event_key:
        ap.error("event_key is required unless --check")

    sink = export.RoboReconSink(base_url=args.url) if args.url else export.RoboReconSink()
    try:
        info = sink.check()
        print(f"{info['url']}: reachable, token accepted, "
              f"schema versions {info['supported_schema_versions']}"
              + ("" if info["ok"] else f" (does not support ours: {export.SCHEMA_VERSION})"))
        if args.check:
            return 0 if info["ok"] else 1
        metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None
        out = api.push_event(args.event_key, metrics, sink=sink, team=args.team)
    except (export.RoboReconError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"{out['event_key']}: {out['stored']} stored, {out['rejected']} rejected "
          f"in {out['requests']} request(s)")
    for r in out["results"]:
        line = f"  {r.get('metric', '?'):<16} {r.get('status')}"
        if r.get("status") == "stored":
            line += f"  teams {r.get('teamsStored')}"
            if r.get("teamsSkippedStale"):
                line += f" (+{r['teamsSkippedStale']} skipped: RoboRecon has newer)"
            if r.get("eventKnown") is False:
                line += "  [event not in RoboRecon yet: stored, shows once imported]"
        else:
            line += "  " + "; ".join(r.get("errors") or [])
        print(line)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

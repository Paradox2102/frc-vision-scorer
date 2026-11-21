#!/usr/bin/env python3
"""Write an event's whole scouting export (what GET /api/scouting/{event}/export
returns) to a JSON file, without the web server.

  .venv/bin/python scripts/export_scouting.py 2026cur
  .venv/bin/python scripts/export_scouting.py 2026cur --out /tmp/cur.json
  .venv/bin/python scripts/export_scouting.py 2026cur --no-paths   # no robot paths

Default output: data/exports/<event>_scouting_export.json. The format is
documented for outside readers in docs/scouting-export/ (README.md + JSON Schema).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from web.scouting import api  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("event_key")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-paths", action="store_true",
                    help="leave out every robot's whole-match path (much smaller file)")
    ap.add_argument("--indent", type=int, default=None, help="pretty-print (bigger file)")
    args = ap.parse_args()

    data = api._export_sync(args.event_key, paths=not args.no_paths)
    out = args.out or REPO / "data" / "exports" / f"{args.event_key}_scouting_export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=args.indent,
                              separators=None if args.indent else (",", ":")))
    tmp.replace(out)
    n_tl = sum(1 for m in data["matches"] if m.get("timeline"))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB): {len(data['matches'])} matches "
          f"({n_tl} with video timeline), {len(data['skipped'])} skipped, "
          f"{len(data['envelopes'])} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

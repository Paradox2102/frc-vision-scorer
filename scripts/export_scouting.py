#!/usr/bin/env python3
"""Write an event's whole scouting export (what GET /api/scouting/{event}/export
returns) to a JSON file, without the web server.

  .venv/bin/python scripts/export_scouting.py 2026cur
  .venv/bin/python scripts/export_scouting.py 2026cur --out /tmp/cur.json
  .venv/bin/python scripts/export_scouting.py 2026cur --no-paths   # no robot paths / unidentified tracks
  .venv/bin/python scripts/export_scouting.py 2026cur --per-match  # + one file per match

Default output: data/exports/<event>_scouting_export.json. ``--per-match`` also
writes data/exports/<event>_matches/<match_key>.json, each exactly what
GET /api/scouting/{event}/match/{match_key} returns, pretty-printed with every
number list on one line. The format is documented for outside readers in
docs/scouting-export/ (README.md + JSON Schema).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from web.scouting import api  # noqa: E402


def _write(out: Path, text: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(text)
    tmp.replace(out)


_NUM_LIST = re.compile(r"\[\s+([-\d.eE+,\s]+?)\s+\]")
_PAIR_LIST = re.compile(r"\[\s+(\[-?\d+, -?\d+\](?:,\s+\[-?\d+, -?\d+\])*)\s+\]")


def pretty(obj) -> str:
    """Indented JSON with lists of numbers (and of index pairs) on one line."""
    txt = json.dumps(obj, indent=2)
    txt = _NUM_LIST.sub(lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", txt)
    txt = _PAIR_LIST.sub(lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", txt)
    return txt + "\n"


def write_per_match(event_key: str, match_keys: list, out_dir: Path) -> int:
    for mk in match_keys:
        _write(out_dir / f"{mk}.json", pretty(api._match_sync(event_key, mk, None)))
    return len(match_keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("event_key")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-paths", action="store_true",
                    help="leave out robot paths + unidentified tracks (much smaller file)")
    ap.add_argument("--indent", type=int, default=None, help="pretty-print (bigger file)")
    ap.add_argument("--per-match", nargs="?", const=True, default=None, metavar="DIR",
                    help="also write one single-match file per match "
                         "(default dir data/exports/<event>_matches/)")
    args = ap.parse_args()

    data = api._export_sync(args.event_key, paths=not args.no_paths)
    out = args.out or REPO / "data" / "exports" / f"{args.event_key}_scouting_export.json"
    _write(out, json.dumps(data, indent=args.indent,
                           separators=None if args.indent else (",", ":")))
    n_tl = sum(1 for m in data["matches"] if m.get("timeline"))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB): {len(data['matches'])} matches "
          f"({n_tl} with video timeline), {len(data['skipped'])} skipped, "
          f"{len(data['envelopes'])} metrics")
    if args.per_match:
        d = (REPO / "data" / "exports" / f"{args.event_key}_matches" if args.per_match is True
             else Path(args.per_match))
        n = write_per_match(args.event_key, [m["match_key"] for m in data["matches"]], d)
        size = sum(f.stat().st_size for f in d.glob("*.json")) / 1e6
        print(f"{d}/  {n} match files ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

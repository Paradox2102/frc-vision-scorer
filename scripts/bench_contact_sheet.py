"""Render bumper-bench crops into grid PNGs for visual adjudication / labelling.

Each cell shows one crop from ``data/eval/bumper_crop_bench/manifest.jsonl``
scaled to a fixed height (``--interp lanczos`` or ``nearest`` - nearest keeps
the raw pixels honest for tiny filtered crops), captioned with the crop id, the
match roster (R/B alliance teams), the incumbent reader's output, and the label
if one has been entered. ``--blind`` captions with the crop id only on a neutral
background (no roster, no baseline read, no stratum colour) so labels are read
cold instead of anchored on the incumbent or the roster. Sheets go to
``.../sheets/`` (gitignored).

Usage
-----
    .venv/bin/python scripts/bench_contact_sheet.py
    .venv/bin/python scripts/bench_contact_sheet.py --filter unlabelled --interp nearest
    .venv/bin/python scripts/bench_contact_sheet.py --stratum filtered --height 128
    .venv/bin/python scripts/bench_contact_sheet.py --blind --filter unlabelled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "data" / "eval" / "bumper_crop_bench"

STRATUM_BG = {"matched": (232, 244, 232), "null": (246, 240, 226), "filtered": (236, 236, 244)}


def caption_lines(row: dict, blind: bool = False) -> list[str]:
    if blind:
        return [row["id"]]
    red = " ".join(map(str, row.get("red_teams") or []))
    blue = " ".join(map(str, row.get("blue_teams") or []))
    if row.get("stratum") == "filtered":
        base = f"base: filtered ({row.get('baseline_filter_reason') or '?'})"
    else:
        team = row.get("baseline_matched_team") or "-"
        raw = row.get("baseline_raw_read") or ""
        conf = row.get("baseline_confidence")
        base = (f"base: {team}"
                + (f" raw={raw}" if raw and raw != team else "")
                + f" {row.get('baseline_alliance_read') or '?'}"
                + (f" c={conf:.2f}" if isinstance(conf, (int, float)) else ""))
    status = row.get("label_status") or "unlabelled"
    if status == "unlabelled":
        label = "label: -"
    else:
        label = f"label: {status} {row.get('label_team') or ''} {row.get('label_alliance') or ''}".rstrip()
    return [row["id"], f"R {red} | B {blue}", base, label]


def render_cell(row: dict, bench_dir: Path, cell_w: int, img_h: int, interp,
                font, line_h: int, blind: bool = False) -> Image.Image:
    lines = caption_lines(row, blind)
    cell_h = img_h + 6 + line_h * len(lines) + 4
    bg = (240, 240, 240) if blind else STRATUM_BG.get(row.get("stratum"), (240, 240, 240))
    cell = Image.new("RGB", (cell_w, cell_h), bg)
    try:
        with Image.open(bench_dir / row["file"]) as im:
            im = im.convert("RGB")
            w = max(1, round(im.width * img_h / im.height))
            h = img_h
            if w > cell_w - 6:  # very wide crop: fit the width instead
                w, h = cell_w - 6, max(1, round(im.height * (cell_w - 6) / im.width))
            thumb = im.resize((w, h), interp)
        cell.paste(thumb, ((cell_w - w) // 2, 3 + (img_h - h) // 2))
    except OSError as e:
        ImageDraw.Draw(cell).text((4, 4), f"unreadable: {e}", fill=(200, 0, 0), font=font)
    d = ImageDraw.Draw(cell)
    y = img_h + 6
    for i, text in enumerate(lines):
        colour = (0, 110, 0) if i == 3 and text != "label: -" else (20, 20, 20)
        d.text((4, y), text, fill=colour, font=font)
        y += line_h
    d.rectangle([0, 0, cell_w - 1, cell_h - 1], outline=(150, 150, 150))
    return cell


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH_DIR))
    ap.add_argument("--out", default=None, help="default <bench>/sheets")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--height", type=int, default=110, help="crop display height (px)")
    ap.add_argument("--cell-width", type=int, default=340)
    ap.add_argument("--interp", choices=["lanczos", "nearest"], default="lanczos")
    ap.add_argument("--filter", choices=["all", "labelled", "unlabelled"], default="all")
    ap.add_argument("--stratum", choices=["matched", "null", "filtered"], default=None)
    ap.add_argument("--match", default=None, help="only this match_key")
    ap.add_argument("--font-size", type=int, default=12)
    ap.add_argument("--blind", action="store_true",
                    help="caption with the crop id only (no roster/baseline/stratum colour)")
    args = ap.parse_args()

    bench_dir = Path(args.bench)
    out = Path(args.out) if args.out else bench_dir / "sheets"
    manifest = bench_dir / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"no manifest at {manifest}; run scripts/build_bumper_crop_bench.py first")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.filter != "all":
        want = args.filter == "labelled"
        rows = [r for r in rows if ((r.get("label_status") or "unlabelled") != "unlabelled") == want]
    if args.stratum:
        rows = [r for r in rows if r.get("stratum") == args.stratum]
    if args.match:
        rows = [r for r in rows if r.get("match_key") == args.match]
    if not rows:
        sys.exit("no crops match the filters")

    interp = Image.LANCZOS if args.interp == "lanczos" else Image.NEAREST
    font = ImageFont.load_default(size=args.font_size)
    line_h = args.font_size + 3
    per = args.cols * args.rows
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.png"):
        old.unlink()
    n_sheets = 0
    for s in range(0, len(rows), per):
        cells = [render_cell(r, bench_dir, args.cell_width, args.height, interp, font, line_h,
                             args.blind)
                 for r in rows[s:s + per]]
        cw, ch = args.cell_width, max(c.height for c in cells)
        nrows = (len(cells) + args.cols - 1) // args.cols
        sheet = Image.new("RGB", (cw * args.cols, ch * nrows), (255, 255, 255))
        for i, c in enumerate(cells):
            sheet.paste(c, ((i % args.cols) * cw, (i // args.cols) * ch))
        path = out / f"sheet_{n_sheets:03d}.png"
        sheet.save(path)
        n_sheets += 1
    print(f"wrote {n_sheets} sheet(s) of up to {per} crops ({len(rows)} crops) to {out}")


if __name__ == "__main__":
    main()

"""Register + fuse every bumper-bench tracklet into one image (web/bumper_register.py).

Writes, under the bench dir:
* ``registered/<crop_id>.jpg`` - the fused image, prepped exactly as the
  reader receives it under ``bench_bumper_readers.py --frames registered``
  (the app's >= 384 px integer upscale, JPEG q90). Gitignored.
* ``registered/diagnostics.jsonl`` - per tracklet: frames used / rejected
  (with reasons), ECC scores, timing.

and, with ``--sheet DIR``, contact-sheet PNGs (bench crop | anchor control |
fused) for eyeballing sharpness and ghosting.

Usage
-----
    .venv/bin/python scripts/build_registered_bench.py
    .venv/bin/python scripts/build_registered_bench.py --sheet /tmp/reg_sheets --fuse mean
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_bumper_readers import (  # noqa: E402
    BENCH_DIR,
    REG_DEFAULTS,
    app_bumper_jpeg,
    fuse_tracklet,
    load_manifest,
    load_tracklets,
)

CELL_H = 96
ROWS_PER_SHEET = 14


def _label_map(bench_dir: Path) -> dict:
    p = bench_dir / "tracklet_labels.jsonl"
    if not p.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, p.read_text().splitlines()) if r}


def _fit(img: np.ndarray, h: int = CELL_H, max_w: int = 420) -> np.ndarray:
    s = h / img.shape[0]
    w = int(round(img.shape[1] * s))
    if w > max_w:
        s = max_w / img.shape[1]
        w, h = max_w, int(round(img.shape[0] * s))
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((CELL_H, max_w, 3), 255, np.uint8)
    canvas[:h, :w] = out
    return canvas


def render_sheets(entries: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for p0 in range(0, len(entries), ROWS_PER_SHEET):
        rows = []
        for e in entries[p0:p0 + ROWS_PER_SHEET]:
            cells = [_fit(e["crop"]), _fit(e["anchor"]), _fit(e["fused"])]
            row = np.concatenate([np.full((CELL_H, 4, 3), 255, np.uint8)] +
                                 sum(([c, np.full((CELL_H, 6, 3), 255, np.uint8)] for c in cells), []),
                                 axis=1)
            band = np.full((22, row.shape[1], 3), 255, np.uint8)
            cv2.putText(band, e["caption"], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)
            rows += [band, row]
        head = np.full((26, rows[0].shape[1], 3), 230, np.uint8)
        for x, t in ((4, "bench crop (single)"), (436, "anchor: cubic x4 + unsharp"),
                     (868, "registered + fused")):
            cv2.putText(head, t, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                        cv2.LINE_AA)
        sheet = np.concatenate([head] + rows, axis=0)
        p = out_dir / f"registered_{p0 // ROWS_PER_SHEET:02d}.png"
        cv2.imwrite(str(p), sheet)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(BENCH_DIR))
    ap.add_argument("--sheet", default=None, help="write contact-sheet PNGs to this dir")
    ap.add_argument("--fuse", default=None, help="override the fuse mode (median/mean/weighted/sas)")
    ap.add_argument("--ids", default=None, help="comma list of crop ids")
    ap.add_argument("--no-write", action="store_true", help="don't write registered/*.jpg")
    args = ap.parse_args()
    bench = Path(args.bench)
    trks = load_tracklets(bench)
    manifest = {r["id"]: r for r in load_manifest(bench)}
    labels = _label_map(bench)
    ids = sorted(trks) if not args.ids else [i for i in args.ids.split(",") if i in trks]
    kw = {"fuse": args.fuse} if args.fuse else {}
    out_dir = bench / "registered"
    if not args.no_write:
        out_dir.mkdir(exist_ok=True)
    diags, entries = [], []
    for cid in ids:
        trk = trks[cid]
        fused, d = fuse_tracklet(trk, bench, **kw)
        anchor, _ = fuse_tracklet(trk, bench, anchor_only=True)
        jpg = app_bumper_jpeg(fused)
        if not args.no_write:
            (out_dir / f"{cid}.jpg").write_bytes(jpg)
        rec = {"id": cid, "tracklet_frames": len(trk["frames"]), "box": trk["box"],
               "fused": d["fused"], "n_used": d["n_used"], "offsets_used": d["offsets_used"],
               "n_samples": d["n_samples"], "used_samples": d["used_samples"],
               "rejected": d["rejected"], "skipped": d["skipped"], "ecc": d["ecc"],
               "ncc": d["ncc"], "merge_w": d["merge_w"], "split_fields": d["split_fields"],
               "comb": d["comb"], "ms": d["ms"], "fuse": d["fuse"], "motion": d["motion"],
               "upscale": d["upscale"], "reg_scale": d["reg_scale"]}
        diags.append(rec)
        lab = labels.get(cid) or {}
        crop = cv2.imread(str(bench / manifest[cid]["file"]))
        cap = (f"{cid}  {trk['box'][2]}x{trk['box'][3]}  frames {d['n_used']}/{len(trk['frames'])}"
               f" samples {d['n_samples']}{' fields' if d['split_fields'] else ''}"
               f"  rej {len(d['rejected'])}  label {lab.get('label_status')}"
               f" {lab.get('label_team') or ''}")
        entries.append({"crop": crop, "anchor": anchor, "fused": fused, "caption": cap})
    if not args.no_write:
        (out_dir / "diagnostics.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in diags))
    ms = [r["ms"] for r in diags]
    nu = [r["n_used"] for r in diags]
    rej = sum(len(r["rejected"]) for r in diags)
    print(f"{len(diags)} tracklets; split into fields {sum(r['split_fields'] for r in diags)}; fused (>=2 aligned) {sum(r['fused'] for r in diags)}; "
          f">=3 used {sum(n >= 3 for n in nu)}; median used {statistics.median(nu) if nu else 0}; "
          f"rejected frames {rej}; ms median {statistics.median(ms) if ms else 0:.1f} "
          f"p90 {sorted(ms)[int(0.9 * (len(ms) - 1))] if ms else 0:.1f} max {max(ms) if ms else 0:.1f}"
          f"  (defaults {REG_DEFAULTS}{', ' + str(kw) if kw else ''})")
    reasons: dict = {}
    for r in diags:
        for _i, why in r["rejected"]:
            k = why.split(" ")[0] + (" " + why.split(" ")[1] if why.startswith("warp") else "")
            reasons[k] = reasons.get(k, 0) + 1
    print("rejection reasons:", reasons)
    if args.sheet:
        for p in render_sheets(entries, Path(args.sheet)):
            print(p)


if __name__ == "__main__":
    main()

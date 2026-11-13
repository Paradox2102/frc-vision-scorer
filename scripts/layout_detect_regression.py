"""Regression check for broadcast-layout auto-detection on real footage.

Runs web.field.layout_detect the same way auto-calibration does
(detect_layout_video: 40 frames over (20 s, end), topped up when cutaways fill
the window) on every reference
video below and prints per-camera deltas against the known-good layouts.

  .venv/bin/python scripts/layout_detect_regression.py [--only cabl,cc] [--json out.json]

Footage is looked up in data/test_footage/ of this checkout, then in
FRC_TEST_FOOTAGE (or the main checkout). Missing videos are skipped.
Exit code 1 if any reference camera is off by more than --tol px (default 6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.field import layout_detect as ld  # noqa: E402

# expected [x, y, w, h] per camera; None = no reference (print only)
REFERENCE = {
    "2025hop_qm1": {"main": [0, 0, 1920, 723], "corner_left": [0, 733, 965, 347],
                    "corner_right": [1004, 733, 916, 347]},
    "2026casgv_qm36": {"main": [0, 0, 1920, 703], "corner_left": [0, 743, 939, 337],
                       "corner_right": [983, 743, 937, 337]},
    "2026casnd_qm46": {"main": [0, 0, 1920, 703], "corner_left": [0, 743, 933, 337],
                       "corner_right": [979, 743, 941, 337]},
    "2026cass_qm1": {"main": [0, 0, 1920, 647], "corner_left": [0, 647, 961, 433],
                     "corner_right": [983, 647, 937, 433]},
    "2026hiho_qm41": {"main": [0, 0, 1920, 459], "corner_left": [0, 513, 933, 312],
                      "corner_right": [979, 513, 941, 312]},
    "2026utwv_qm1": {"main": [0, 0, 1920, 704], "corner_left": [0, 743, 939, 337],
                     "corner_right": [977, 743, 943, 337]},
    # no confirmed record (calibration conflicts kept the default): print only
    "2026cur_qm1": None,
    "2026hop_qm1": None,
    # offseason graphics packages (boundaries read off frames by hand)
    "2026cabl_qm1": {"main": [0, 0, 1920, 720], "corner_left": [0, 720, 640, 360],
                     "corner_right": [1280, 720, 640, 360]},
    "2026cabl_qm2": "2026cabl_qm1",
    "2026cabl_qm3": "2026cabl_qm1",
    "2026cabl_qm4": "2026cabl_qm1",
    "2026cc_qm1": {"main": [0, 0, 1920, 645], "corner_left": [0, 645, 955, 435],
                   "corner_right": [955, 645, 965, 435]},
    "2026cc_qm2": "2026cc_qm1",
    "2026cc_qm3": "2026cc_qm1",
    "2026cc_qm4": "2026cc_qm1",
}


def footage_dirs() -> list[Path]:
    dirs = [ROOT / "data" / "test_footage"]
    env = os.environ.get("FRC_TEST_FOOTAGE")
    if env:
        dirs.append(Path(env))
    dirs.append(Path("/home/nick/AI/frc-vision-scorer/data/test_footage"))
    return dirs


def find_video(name: str) -> Path | None:
    for d in footage_dirs():
        p = d / f"{name}.mp4"
        if p.exists():
            return p
    return None


def expected_for(name: str):
    exp = REFERENCE.get(name)
    while isinstance(exp, str):
        exp = REFERENCE[exp]
    return exp


def delta(got, exp) -> int:
    """Max abs difference of the crop edges (x0, y0, x1, y1)."""
    g = [got[0], got[1], got[0] + got[2], got[1] + got[3]]
    e = [exp[0], exp[1], exp[0] + exp[2], exp[1] + exp[3]]
    return max(abs(a - b) for a, b in zip(g, e))


def run_one(path: Path) -> dict:
    """Same entry point as auto-calibration's detect_with_preview."""
    t = time.time()
    out, _used = ld.detect_layout_video(str(path), (20, None), 40)
    out["diag"]["t_total_s"] = round(time.time() - t, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of substrings to select videos")
    ap.add_argument("--tol", type=int, default=6)
    ap.add_argument("--json", default="", help="write full results here")
    ap.add_argument("--diag", action="store_true", help="print the cut log")
    a = ap.parse_args(argv)
    sel = [s for s in a.only.split(",") if s]
    results, bad = {}, 0
    for name in REFERENCE:
        if sel and not any(s in name for s in sel):
            continue
        path = find_video(name)
        if path is None:
            print(f"{name:16s} SKIP (no footage)")
            continue
        out = run_one(path)
        exp = expected_for(name)
        cams = out["cameras"]
        d = out["diag"]
        rows = []
        for cam in ("main", "corner_left", "corner_right"):
            got = cams.get(cam, {}).get("crop")
            e = (exp or {}).get(cam)
            if got is None and e is None:
                continue
            if got is None:
                rows.append(f"{cam}=MISSING")
                bad += exp is not None
                continue
            nm = len(cams[cam].get("masks", []))
            if e is None:
                rows.append(f"{cam}={got} m{nm}")
            else:
                dd = delta(got, e)
                flag = "" if dd <= a.tol else " !!"
                bad += dd > a.tol
                rows.append(f"{cam}={got} d{dd}{flag} m{nm}")
        extra = sorted(set(cams) - {"main", "corner_left", "corner_right"})
        print(f"{name:16s} used {d.get('n_used')}/{d.get('n_frames')} "
              f"t={d['t_total_s']:.1f}s" + (f" +{d['topup']['n_extra']}fr" if 'topup' in d else "") + "  " + "  ".join(rows)
              + (f"  extra={extra}" if extra else ""))
        if a.diag:
            for c in d.get("cuts", []):
                print("    ", json.dumps(c))
            for k in ("selection", "topup"):
                if k in d:
                    print("    ", k, json.dumps(d[k]))
        results[name] = {"cameras": cams, "confidence": out["confidence"],
                         "diag": {k: v for k, v in d.items() if k != "cuts"}}
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))
    print(f"{bad} camera(s) off by more than {a.tol} px")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

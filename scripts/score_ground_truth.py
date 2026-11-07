"""Score pipeline replays against hand-annotated ground-truth robot tracks.

GT docs: ``data/eval/ground_truth/<match_key>/<team>.json`` (web/field/
ground_truth.py). Pipeline: the replay's ``meta.json`` ``frames``. Metric
definitions live in web/field/gt_scoring.py.

Usage
-----
    .venv/bin/python scripts/score_ground_truth.py --match-key 2026cur_qm8
    .venv/bin/python scripts/score_ground_truth.py --replay 392e78f2 --period auto
    .venv/bin/python scripts/score_ground_truth.py --match-key 2026cur_qm8 --all-replays 6

Importable: ``score_replay(replay_id, period=..., ...) -> dict`` (``{"error": ...}``
on failure, like scripts/ab_qm8.score) for A/B harnesses.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web.field.ground_truth import GT_DIR, list_gt, video_fingerprint  # noqa: E402
from web.field.gt_scoring import (  # noqa: E402
    AlignmentError, replay_from_meta, score_match,
)

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "frc_scorer.db"
REPLAYS_DIR = REPO / "web" / "replays"
_COLS = "id, created_at, match_key, source_video_path, meta_path, status"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _resolve(stored: Optional[str]) -> Optional[Path]:
    """Stored paths may point into another worktree (.worktrees/main-server/...);
    those dirs are symlinks to the same files, so fall back to this repo."""
    if not stored:
        return None
    p = Path(stored)
    if p.exists():
        return p
    s = str(p)
    for marker in ("/web/uploads/", "/web/replays/", "/data/"):
        if marker in s:
            cand = REPO / (marker.strip("/") + "/" + s.split(marker, 1)[1])
            if cand.exists():
                return cand
    return None


def _db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def replay_row(replay_id: str) -> dict:
    """Replay row by id or unique id prefix."""
    con = _db()
    rows = con.execute(f"SELECT {_COLS} FROM replays WHERE id LIKE ? ORDER BY created_at DESC",
                       (replay_id + "%",)).fetchall()
    con.close()
    if not rows:
        raise LookupError(f"no replay {replay_id!r}")
    if len(rows) > 1 and rows[0]["id"] != replay_id:
        raise LookupError(f"replay prefix {replay_id!r} is ambiguous ({len(rows)} rows)")
    return dict(rows[0])


def latest_replays(match_key: str, n: int = 1) -> list[dict]:
    con = _db()
    rows = con.execute(f"SELECT {_COLS} FROM replays WHERE match_key = ? AND status = 'complete' "
                       "ORDER BY created_at DESC LIMIT ?", (match_key, n)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _probe_fps(path: Path) -> Optional[float]:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 1 else None
    except Exception:
        return None


def load_replay(row: dict):
    """-> ReplayTrack for a replay row (fingerprint + fps from its source video)."""
    meta_p = REPLAYS_DIR / row["id"] / "meta.json"
    if not meta_p.exists():
        meta_p = _resolve(row.get("meta_path"))
    if meta_p is None or not meta_p.exists():
        raise FileNotFoundError(f"meta.json missing for replay {row['id']}")
    meta = json.loads(meta_p.read_text())
    vid = _resolve(row.get("source_video_path"))
    fp = video_fingerprint(vid) if vid else None
    fps = _probe_fps(vid) if vid else None
    return replay_from_meta(meta, src_fps=fps, fingerprint=fp)


# --------------------------------------------------------------------------- #
# Scoring API
# --------------------------------------------------------------------------- #
def score_replay(replay_id: str, *, period: str = "full", match_radius_m: float = 1.0,
                 gt_dir: Path = GT_DIR, force: bool = False, eval_step_s: float = 0.25,
                 lag_ms: float = 0.0, details: bool = False,
                 gt_docs: Optional[list] = None) -> dict:
    """Score one replay against its match's GT. Never raises for expected
    failures (no GT, alignment, missing files): returns ``{"error": ...}``."""
    try:
        row = replay_row(replay_id)
        docs = gt_docs if gt_docs is not None else [
            d for d in list_gt(row["match_key"], Path(gt_dir))
            if isinstance(d, dict) and d.get("team") and "keyframes" in d]
        base = {"replay_id": row["id"], "match_key": row["match_key"],
                "created_at": row["created_at"]}
        if not docs:
            return {**base, "error": f"no ground truth for {row['match_key']} in {gt_dir}"}
        rt = load_replay(row)
        res = score_match(docs, rt, match_radius_m=match_radius_m, period=period,
                          eval_step_s=eval_step_s, lag_ms=lag_ms, force=force,
                          details=details)
        if rt.fingerprint is None:
            res["warnings"].append("replay source video not found; no fingerprint check")
        return {**base, **res}
    except (AlignmentError, LookupError, FileNotFoundError, ValueError) as e:
        return {"replay_id": replay_id, "error": str(e)}


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def _f(v, nd=2, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


def print_robots(res: dict) -> None:
    print(f"\nreplay {res['replay_id'][:8]}  {res['match_key']}  {res['created_at']}  "
          f"period={res['params']['period']} window={res['window']} "
          f"radius={res['params']['match_radius_m']} step={res['params']['eval_step_s']}s "
          f"lag={res['params']['lag_ms']}ms")
    for w in res["warnings"]:
        print(f"  WARNING: {w}")
    hdr = (f"{'team':>6} {'pos':>4} {'align':>11} {'n':>5} {'proc':>5} {'cov':>5} "
           f"{'errM':>5} {'errP90':>6} {'lblOK':>6} {'wrong':>6} {'unlbl':>6} "
           f"{'lblPres':>7} {'lblFar':>6} {'sw':>3} {'slot':>8}  est(n/cov/err)  unk")
    print(hdr)
    for r in res["robots"]:
        v, e = r["visible"], r["estimated"]
        pos = f"{(r['alliance'] or '?')[0].upper()}{r['station'] or '?'}"
        slot = f"{v['main_slot']}:{_f(v['main_slot_share'], pct=True)}" if v["main_slot"] else "-"
        est = f"{e['n']}/{_f(e['coverage'], pct=True)}/{_f(e['err']['mean'])}" if e["n"] else "-"
        print(f"{r['team']:>6} {pos:>4} {r['alignment']:>11} {v['n']:>5} {v['n_processed']:>5} "
              f"{_f(v['coverage'], pct=True):>5} {_f(v['err']['mean']):>5} "
              f"{_f(v['err']['p90']):>6} {_f(v['label_correct'], pct=True):>6} "
              f"{_f(v['label_wrong'], pct=True):>6} {_f(v['unlabeled'], pct=True):>6} "
              f"{_f(v['label_present'], pct=True):>7} {_f(v['label_on_wrong_robot'], pct=True):>6} "
              f"{v['slot_switches']:>3} {slot:>8}  {est:<14} {r['n_unknown']}")
    print("  " + agg_line(res))


def agg_line(res: dict) -> str:
    a = res["headline"]
    return (f"AGG visible: robots={a['n_robots']} n={a['n_processed']} cov={_f(a['coverage'], pct=True)} "
            f"err mean/med/p90={_f(a['err']['mean'])}/{_f(a['err']['median'])}/{_f(a['err']['p90'])} "
            f"lblOK={_f(a['label_correct'], pct=True)} wrong={_f(a['label_wrong'], pct=True)} "
            f"unlbl={_f(a['unlabeled'], pct=True)} lblFar={_f(a['label_on_wrong_robot'], pct=True)} "
            f"switches={a['slot_switches']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", help="replay id (or unique prefix)")
    g.add_argument("--match-key", help="score the latest complete replay of this match")
    ap.add_argument("--all-replays", type=int, default=0, metavar="N",
                    help="with --match-key: score the N most recent replays, one row each")
    ap.add_argument("--period", choices=("full", "auto"), default="full")
    ap.add_argument("--match-radius", type=float, default=1.0)
    ap.add_argument("--eval-step", type=float, default=0.25,
                    help="evaluate GT on this time grid in seconds (0 = every keyframe)")
    ap.add_argument("--lag-ms", type=float, default=0.0,
                    help="shift follow-sourced GT keyframes earlier by lag_ms * playback rate")
    ap.add_argument("--gt-dir", default=str(GT_DIR))
    ap.add_argument("--json", help="write full results JSON here")
    ap.add_argument("--details", action="store_true", help="include per-sample records in --json")
    ap.add_argument("--force", action="store_true",
                    help="assume identity frame alignment when videos differ (LOUD warning)")
    a = ap.parse_args()

    kw = dict(period=a.period, match_radius_m=a.match_radius, gt_dir=Path(a.gt_dir),
              force=a.force, eval_step_s=a.eval_step, lag_ms=a.lag_ms, details=a.details)
    if a.replay:
        ids = [a.replay]
    else:
        rows = latest_replays(a.match_key, max(1, a.all_replays))
        if not rows:
            raise SystemExit(f"no complete replays for {a.match_key}")
        ids = [r["id"] for r in rows]
    results = [score_replay(i, **kw) for i in ids]

    if len(results) == 1:
        r = results[0]
        if "error" in r:
            raise SystemExit(f"ERROR: {r['error']}")
        print_robots(r)
    else:
        print(f"{'replay':>8} {'created':>19}  aggregate (visible)")
        for r in results:
            if "error" in r:
                print(f"{r['replay_id'][:8]:>8} {'':>19}  ERROR: {r['error']}")
                continue
            per = " ".join(f"{x['team']}:{_f(x['visible']['coverage'], pct=True)}/"
                           f"{_f(x['visible']['label_correct'], pct=True)}"
                           for x in r["robots"])
            warn = "  [WARN]" if r["warnings"] else ""
            print(f"{r['replay_id'][:8]:>8} {r['created_at']:>19}  {agg_line(r)}{warn}")
            print(f"{'':>29}cov/lblOK  {per}")
    if a.json:
        Path(a.json).write_text(json.dumps(results if len(results) > 1 else results[0], indent=1))
        print(f"\njson -> {a.json}")


if __name__ == "__main__":
    main()

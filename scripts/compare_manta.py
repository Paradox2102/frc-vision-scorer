"""Compare our vision auto-paths against Manta human-scouted ground truth.

Manta (team 3128) scouts the *autonomous* robot trajectory by hand. We extract
our own per-robot autonomous trajectory during analysis (the replay summary's
``autonomous_paths``, populated only over the clock-located auto window — run the
job with the ``auto_period_only`` toggle to spend all the tracking budget there).
This tool joins the two on ``(event, match, alliance, position)`` and scores how
close our path is to the human path, so we can quantify and improve accuracy.

Both sides are in WPILib field meters:
  * Manta  -> ``scripts/manta_client.MantaClient`` converts its grid to meters.
  * Ours   -> ``web/replays/<id>/meta.json`` ``summary.autonomous_paths[slot]``.

The Manta Y-axis orientation was never validated (``GridCalibration.flip_y``),
so by default we score with flip_y both ways and report the better one per robot;
``--flip-y {auto,off,on}`` pins it once you've settled the convention.

Usage
-----
    .venv/bin/python scripts/compare_manta.py 2026cur
    .venv/bin/python scripts/compare_manta.py 2026cur --match 2 --overlay out/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.manta_client import GridCalibration, MantaClient, parse_auto_field  # noqa: E402
from web.field.manta import match_number_from_key  # noqa: E402  (single source of truth)
from web.field.path_quality import coverage, gap_segments, tracked_length_m  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "frc_scorer.db"
REPLAYS_DIR = REPO / "web" / "replays"

# slot label (B1..R3) <-> (alliance, position 1-based)
_SLOT_RE = re.compile(r"^([BR])([123])$")


def slot_to_alliance_pos(slot: str) -> Optional[tuple[str, int]]:
    m = _SLOT_RE.match(slot or "")
    if not m:
        return None
    return ("blue" if m.group(1) == "B" else "red"), int(m.group(2))


# --------------------------------------------------------------------------- #
# Path similarity metrics (pure, unit-tested)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PathMetrics:
    n_ours: int
    n_truth: int
    mean_nn_m: float        # symmetric mean nearest-neighbor distance (primary)
    hausdorff_m: float      # symmetric (worst-case) Hausdorff distance
    start_err_m: float      # distance between first points
    end_err_m: float        # distance between last points
    len_ours_m: float
    len_truth_m: float

    def as_row(self) -> dict:
        return dataclasses.asdict(self)


def _polyline_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def summarize_our_path(samples, max_gap_s: float = 1.0, window_s=None) -> dict:
    """Gap-aware quality summary for one slot's raw auto samples.

    ``samples`` is the stored list of ``{"timestamp", "field_pos"}`` dicts. A
    dropout (occlusion / lost track) shows up as a large timestamp gap, so we
    report both the naive polyline length (which bridges gaps with fabricated
    straight lines) and the honest tracked length, plus coverage — turning a
    "the lengths disagree" mystery into "we only tracked 40 % of the window".
    """
    pts = np.array([s["field_pos"] for s in samples], float).reshape(-1, 2)
    ts = np.array([s["timestamp"] for s in samples], float)
    return {
        "naive_len_m": round(_polyline_length(pts), 3),
        "tracked_len_m": round(tracked_length_m(pts, ts, max_gap_s), 3),
        **coverage(ts, max_gap_s=max_gap_s, window_s=window_s),
    }


def _directed_mean_nn(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over points of ``a`` of the nearest distance to point set ``b``."""
    # (na, nb) pairwise distances; small auto paths so the dense matrix is fine.
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def compare_paths(ours: np.ndarray, truth: np.ndarray) -> PathMetrics:
    """Symmetric similarity between two polylines (Nx2 arrays, meters)."""
    ours = np.asarray(ours, float).reshape(-1, 2)
    truth = np.asarray(truth, float).reshape(-1, 2)
    d = np.linalg.norm(ours[:, None, :] - truth[None, :, :], axis=2)
    mean_nn = 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())
    hausdorff = max(d.min(axis=1).max(), d.min(axis=0).max())
    return PathMetrics(
        n_ours=len(ours),
        n_truth=len(truth),
        mean_nn_m=round(mean_nn, 3),
        hausdorff_m=round(float(hausdorff), 3),
        start_err_m=round(float(np.linalg.norm(ours[0] - truth[0])), 3),
        end_err_m=round(float(np.linalg.norm(ours[-1] - truth[-1])), 3),
        len_ours_m=round(_polyline_length(ours), 3),
        len_truth_m=round(_polyline_length(truth), 3),
    )


# --------------------------------------------------------------------------- #
# Loading our side
# --------------------------------------------------------------------------- #
def _local_meta(replay_id: str, stored_path: str) -> Optional[Path]:
    """Prefer the in-repo meta.json; stored absolute paths can be stale."""
    local = REPLAYS_DIR / replay_id / "meta.json"
    if local.exists():
        return local
    if stored_path and Path(stored_path).exists():
        return Path(stored_path)
    return None


def load_our_auto_paths(event_key: str) -> dict[tuple[str, int, int], dict]:
    """Map ``(match_number, alliance, position)`` -> our auto path + metadata.

    One entry per slot of the most recent replay for each match (later replays
    win, matching how the UI shows the freshest analysis).
    """
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, match_key, meta_path, created_at, blue_teams, red_teams "
        "FROM replays WHERE event_key = ? ORDER BY created_at ASC",
        (event_key,),
    ).fetchall()
    con.close()

    out: dict[tuple[str, int, int], dict] = {}
    for r in rows:
        parsed = match_number_from_key(r["match_key"])
        if parsed is None or parsed[0] != "qual":
            continue  # Manta cache here is qual-only; extend when needed
        match_number = parsed[1]
        meta = _local_meta(r["id"], r["meta_path"])
        if meta is None:
            continue
        summary = json.loads(meta.read_text()).get("summary", {})
        auto = summary.get("autonomous_paths") or {}
        for slot, samples in auto.items():
            ap = slot_to_alliance_pos(slot)
            if ap is None or len(samples) < 2:
                continue
            alliance, pos = ap
            pts = np.array([s["field_pos"] for s in samples], float)
            out[(match_number, alliance, pos)] = {
                "replay_id": r["id"],
                "match_key": r["match_key"],
                "slot": slot,
                "path_m": pts,
                "samples": samples,  # raw {timestamp, field_pos} for gap/coverage
            }
    return out


# --------------------------------------------------------------------------- #
# Loading Manta side
# --------------------------------------------------------------------------- #
def load_manta_auto_paths(
    event_key: str, flip_y: bool
) -> dict[tuple[str, int, int], dict]:
    """Map ``(match_number, alliance, position)`` -> Manta path (best scout)."""
    cal = GridCalibration(flip_y=flip_y)
    client = MantaClient()
    out: dict[tuple[str, int, int], dict] = {}
    n = 1
    misses = 0
    while misses <= 5:
        try:
            m = client.match(event_key, n, "qual")  # cache-only friendly
        except Exception:
            misses += 1
            n += 1
            continue
        misses = 0
        for robot in m.get("robots", []):
            alliance, pos = robot.get("alliance"), robot.get("position")
            if alliance not in ("red", "blue") or pos not in (1, 2, 3):
                continue
            best: Optional[np.ndarray] = None
            for sc in robot.get("scouts", []):
                raw = sc.get("data", {}).get("auto_field")
                if not isinstance(raw, list) or len(raw) < 2:
                    continue
                path = parse_auto_field(raw, cal)
                if best is None or len(path) > len(best):
                    best = np.array(path, float)
            if best is not None and len(best) >= 2:
                out[(n, alliance, pos)] = {
                    "team_number": robot.get("team_number"),
                    "path_m": best,
                }
        n += 1
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def assign_alliance(
    ours: dict, truth_by_flip: dict, match_number: int, alliance: str, flip_mode: str
):
    """Optimal (Hungarian) pairing of our paths to Manta paths within one alliance.

    Joining by position label assumes our slot numbering equals Manta's
    driver-station position, which it may not. This instead finds the pairing
    that minimizes total mean nearest-neighbor error, so a low cost means good
    tracking even when the labels disagree — and the chosen pairing reveals the
    actual our-slot -> manta-position mapping.
    """
    from scipy.optimize import linear_sum_assignment

    our_pos = [p for p in (1, 2, 3) if (match_number, alliance, p) in ours]
    if not our_pos:
        return []
    modes = {"off": [False], "on": [True], "auto": [False, True]}[flip_mode]

    best = None  # (total_cost, flip_y, pairs)
    for fy in modes:
        truth = truth_by_flip[fy]
        truth_pos = [p for p in (1, 2, 3) if (match_number, alliance, p) in truth]
        if not truth_pos:
            continue
        cost = np.zeros((len(our_pos), len(truth_pos)))
        for i, op in enumerate(our_pos):
            for j, tp in enumerate(truth_pos):
                m = compare_paths(ours[(match_number, alliance, op)]["path_m"],
                                  truth[(match_number, alliance, tp)]["path_m"])
                cost[i, j] = m.mean_nn_m
        ri, ci = linear_sum_assignment(cost)
        total = float(cost[ri, ci].sum())
        if best is None or total < best[0]:
            pairs = []
            for i, j in zip(ri, ci):
                op, tp = our_pos[i], truth_pos[j]
                metrics = compare_paths(
                    ours[(match_number, alliance, op)]["path_m"],
                    truth[(match_number, alliance, tp)]["path_m"])
                cov = summarize_our_path(ours[(match_number, alliance, op)]["samples"])
                pairs.append({
                    "our_slot": ours[(match_number, alliance, op)]["slot"],
                    "manta_pos": tp,
                    "manta_team": truth[(match_number, alliance, tp)]["team_number"],
                    "flip_y": fy, **metrics.as_row(), **cov,
                })
            best = (total, fy, pairs)
    return best[2] if best else []


def _score(ours: np.ndarray, truth_loader, key, flip_mode: str):
    """Score against the requested flip_y mode; 'auto' picks the better fit."""
    modes = {"off": [False], "on": [True], "auto": [False, True]}[flip_mode]
    best = None
    for fy in modes:
        truth = truth_loader[fy].get(key)
        if truth is None:
            continue
        metrics = compare_paths(ours, truth["path_m"])
        if best is None or metrics.mean_nn_m < best[1].mean_nn_m:
            best = (fy, metrics, truth)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("event_key", help="e.g. 2026cur")
    ap.add_argument("--match", type=int, default=None, help="only this qual match number")
    ap.add_argument("--flip-y", choices=["auto", "off", "on"], default="auto")
    ap.add_argument("--assign", action="store_true",
                    help="optimal per-alliance pairing instead of joining by position "
                         "label (use when our slot numbering may not match Manta's)")
    ap.add_argument("--overlay", metavar="DIR", default=None,
                    help="write a path-overlay PNG per matched robot into DIR")
    ap.add_argument("--json", metavar="FILE", default=None, help="dump per-robot metrics as JSON")
    args = ap.parse_args()

    ours = load_our_auto_paths(args.event_key)
    if not ours:
        raise SystemExit(
            f"No processed auto paths for {args.event_key}. Process matches with "
            "the 'auto period only' toggle first.")

    truth_by_flip = {
        False: load_manta_auto_paths(args.event_key, flip_y=False),
        True: load_manta_auto_paths(args.event_key, flip_y=True),
    }
    if not (truth_by_flip[False] or truth_by_flip[True]):
        raise SystemExit(
            f"No cached Manta paths for {args.event_key}. Run "
            f"scripts/manta_client.py {args.event_key} first.")

    if args.assign:
        _run_assignment(args, ours, truth_by_flip)
        return

    results = []
    for key in sorted(ours):
        match_number, alliance, pos = key
        if args.match is not None and match_number != args.match:
            continue
        scored = _score(ours[key]["path_m"], truth_by_flip, key, args.flip_y)
        if scored is None:
            results.append({"key": key, "matched": False, **ours[key], "path_m": None})
            continue
        flip_y, metrics, truth = scored
        results.append({
            "match_number": match_number, "alliance": alliance, "position": pos,
            "slot": ours[key]["slot"], "match_key": ours[key]["match_key"],
            "manta_team": truth["team_number"], "flip_y": flip_y,
            "matched": True, **metrics.as_row(),
            **summarize_our_path(ours[key]["samples"]),
        })

    matched = [r for r in results if r.get("matched")]
    print(f"\n{args.event_key}: {len(matched)} robot paths joined "
          f"(of {len(ours)} ours, flip_y={args.flip_y})\n")
    if matched:
        hdr = f"{'match':>6} {'slot':>4} {'team':>6} {'flipY':>5} " \
              f"{'meanNN':>7} {'hausd':>7} {'cov%':>5} {'maxGap':>6} {'trkO/lenT':>11}"
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(matched, key=lambda x: (x["match_number"], x["slot"])):
            drop = " DROPOUT" if (r["coverage_frac"] < 0.6 or r["max_gap_s"] > 2.0) else ""
            print(f"{r['match_number']:>6} {r['slot']:>4} {str(r['manta_team']):>6} "
                  f"{str(r['flip_y']):>5} {r['mean_nn_m']:>7.3f} {r['hausdorff_m']:>7.3f} "
                  f"{100*r['coverage_frac']:>5.0f} {r['max_gap_s']:>6.1f} "
                  f"{r['tracked_len_m']:>5.1f}/{r['len_truth_m']:<5.1f}{drop}")
        mean_nn = np.mean([r["mean_nn_m"] for r in matched])
        print(f"\n  aggregate mean nearest-neighbor error: {mean_nn:.3f} m "
              f"over {len(matched)} robots")
        if args.flip_y == "auto":
            on = sum(1 for r in matched if r["flip_y"])
            print(f"  flip_y vote: {on} on / {len(matched) - on} off "
                  "(consistent vote => set --flip-y to pin GridCalibration)")
    else:
        print("  No (match, alliance, position) keys overlap between the two sources.")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(
            [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in r.items() if k not in ("path_m", "samples")}
             for r in results], indent=2))
        print(f"\n  wrote metrics -> {args.json}")

    if args.overlay and matched:
        _write_overlays(args.overlay, matched, ours, truth_by_flip)


def _run_assignment(args, ours, truth_by_flip):
    """Print the optimal our->Manta pairing per alliance (label-independent)."""
    matches = sorted({k[0] for k in ours} if args.match is None else {args.match})
    all_pairs = []
    print(f"\n{args.event_key}: optimal per-alliance assignment "
          f"(flip_y={args.flip_y}) — reveals our-slot -> Manta-position mapping\n")
    hdr = (f"{'match':>6} {'alliance':>8} {'ourSlot':>7} -> {'manta#':>6} "
           f"{'team':>6} {'flipY':>5} {'meanNN':>7} {'hausd':>7} "
           f"{'cov%':>5} {'maxGap':>6} {'trkO/lenT':>11}")
    print(hdr)
    print("-" * len(hdr))
    for match_number in matches:
        for alliance in ("blue", "red"):
            pairs = assign_alliance(
                ours, truth_by_flip, match_number, alliance, args.flip_y)
            for p in pairs:
                flags = "" if p["our_slot"][1:] == str(p["manta_pos"]) else "  MISLABEL"
                if p["coverage_frac"] < 0.6 or p["max_gap_s"] > 2.0:
                    flags += "  DROPOUT"
                print(f"{match_number:>6} {alliance:>8} {p['our_slot']:>7} -> "
                      f"{p['manta_pos']:>6} {str(p['manta_team']):>6} "
                      f"{str(p['flip_y']):>5} {p['mean_nn_m']:>7.3f} "
                      f"{p['hausdorff_m']:>7.3f} {100*p['coverage_frac']:>5.0f} "
                      f"{p['max_gap_s']:>6.1f} {p['tracked_len_m']:>5.1f}/"
                      f"{p['len_truth_m']:<5.1f}{flags}")
            all_pairs.extend(pairs)
    if all_pairs:
        mean_nn = np.mean([p["mean_nn_m"] for p in all_pairs])
        mism = sum(1 for p in all_pairs if p["our_slot"][1:] != str(p["manta_pos"]))
        drop = sum(1 for p in all_pairs if p["coverage_frac"] < 0.6 or p["max_gap_s"] > 2.0)
        print(f"\n  best-case aggregate mean nearest-neighbor: {mean_nn:.3f} m "
              f"over {len(all_pairs)} robots")
        print(f"  position-label mismatches: {mism}/{len(all_pairs)} "
              "(our slot numbering vs Manta driver-station position)")
        print(f"  dropout-flagged robots: {drop}/{len(all_pairs)} "
              "(coverage <60% or a >2 s tracking gap — trkO = gap-aware length)")


def _write_overlays(out_dir, matched, ours, truth_by_flip):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping overlays)")
        return
    os.makedirs(out_dir, exist_ok=True)
    for r in matched:
        key = (r["match_number"], r["alliance"], r["position"])
        o = ours[key]["path_m"]
        ts = [s["timestamp"] for s in ours[key]["samples"]]
        t = truth_by_flip[r["flip_y"]][key]["path_m"]
        fig, ax = plt.subplots(figsize=(8, 4))
        # Draw ours as separate segments broken at tracking dropouts, and show
        # the dropout as a thin dotted connector so it reads as "lost track"
        # rather than fabricated straight-line motion.
        segs = gap_segments(ts, max_gap_s=1.0)
        for si, (a, b) in enumerate(segs):
            ax.plot(o[a:b, 0], o[a:b, 1], "-o", ms=3, color="#2a7",
                    label="ours (vision)" if si == 0 else None)
            if si > 0:  # dotted bridge across the gap from the previous segment
                pa, pb = segs[si - 1][1] - 1, a
                ax.plot([o[pa, 0], o[pb, 0]], [o[pa, 1], o[pb, 1]],
                        ":", color="#999", lw=1,
                        label="dropout (no tracking)" if si == 1 else None)
        ax.plot(t[:, 0], t[:, 1], "-s", ms=3, label="Manta (human)", color="#c44")
        ax.set_title(f"{r['match_key']} {r['slot']} team {r['manta_team']} "
                     f"meanNN={r['mean_nn_m']:.2f}m cov={100*r['coverage_frac']:.0f}% "
                     f"flipY={r['flip_y']}")
        ax.set_xlabel("field x (m)"); ax.set_ylabel("field y (m)")
        ax.set_xlim(0, 16.541); ax.set_ylim(0, 8.069); ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
        path = Path(out_dir) / f"{r['match_key']}_{r['slot']}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"\n  wrote {len(matched)} overlays -> {out_dir}")


if __name__ == "__main__":
    main()

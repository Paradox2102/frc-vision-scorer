#!/usr/bin/env python3
"""Export hand-annotated ground-truth matches as scouting single-match files,
so other apps (RoboRecon, ...) can load real, human-tracked robot paths with
the same reader they use for the pipeline's exports.

  .venv/bin/python scripts/export_gt_matches.py                 # every hand GT match
  .venv/bin/python scripts/export_gt_matches.py 2026cur_qm8     # just these
  .venv/bin/python scripts/export_gt_matches.py --include-sim   # + Manta-sim truth
  .venv/bin/python scripts/export_gt_matches.py --upload        # + copy to Drive ground_truth/

Output: data/exports/ground_truth/<match_key>.json, each in the
``frc-vision-scorer.scouting-match`` layout (docs/scouting-export/, "The
single-match file") and valid against its JSON Schema. The match header
(lineup, video timeline, cameras) comes from the pipeline's replay of the same
match, so ``t`` is the exact match clock the pipeline exports use. What
differs from a pipeline file:

* ``match.robots[]`` holds the hand-tracked robots only (one per GT file). A
  lineup robot nobody annotated is absent (2026cur_qm8 has no 9066).
* Each path point is one annotation keyframe: ``state`` (extra column) is
  ``visible`` (clicked on a clear view) or ``estimated`` (occluded, best
  guess). ``cams`` is the bit of the camera the click was made in for visible
  points and ``0`` for estimated ones; ``unknown`` keyframes (robot on the
  field but not locatable) are gaps.
* ``identity`` is one ``source: "ground_truth"`` row (slot ``"GT"``); the
  team is known, not inferred.
* ``unidentified`` is empty and ``unidentified_s`` zero: every annotated robot
  is named.
* A top-level ``ground_truth`` block says who annotated what, and when.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from web.field.ground_truth import GT_DIR, POSITIONED_STATES, list_gt, list_gt_matches, video_fingerprint  # noqa: E402
from web.field.gt_scoring import AlignmentError, align_gt  # noqa: E402
from web.scouting import api, export, timeline  # noqa: E402
from web.scouting.clock import MatchClock  # noqa: E402

OUT_DIR = REPO / "data" / "exports" / "ground_truth"
DRIVE_DIR = "ground_truth"
GT_SOURCE = "ground_truth"
SIM_ANNOTATOR = "manta-sim-truth"


def _robot(doc: dict, md, clock: MatchClock, legend: list, rf_of, auto_s: float) -> dict:
    """One GT robot as a ``robots[]`` entry (pipeline layout + ``state``)."""
    pts = []          # (t, x, y, replay_frame, state, view)
    seen_frames = set()
    for kf in sorted(doc.get("keyframes") or [], key=lambda k: k["frame"]):
        fxy = kf.get("field_xy")
        if kf.get("state") not in POSITIONED_STATES or not fxy or None in fxy:
            continue
        rf = int(round(rf_of(kf["frame"])))
        at = clock.at(rf)
        if at is None or rf in seen_frames:      # outside the match clock / duplicate
            continue
        seen_frames.add(rf)
        pts.append((float(at[1]), float(fxy[0]), float(fxy[1]), rf, kf["state"], kf.get("view")))
    pts.sort(key=lambda p: (p[0], p[3]))

    # Segments break on time gaps only. Clicks come ~30/s, so a few cm of hand
    # jitter between two clicks already reads as > PATH_MAX_SPEED_MPS: the
    # pipeline's jump rule would shred a GT path into hundreds of pieces, and
    # a GT label never hops to another robot anyway.
    ts = [p[0] for p in pts]
    segments, start = [], 0
    for i in range(1, len(pts) + 1):
        if i == len(pts) or ts[i] - ts[i - 1] > export.PATH_GAP_S:
            segments.append([start, i - 1])
            start = i
    jumps = 0
    period = statistics.median([b - a for a, b in zip(ts, ts[1:])]) if len(ts) > 1 else 0.1
    # Each click stands for the time to the next one (the GT scorer
    # interpolates across gaps up to PATH_GAP_S); a segment's last click for
    # one median click spacing.
    weights = [(b - a) if b - a <= export.PATH_GAP_S else period
               for a, b in zip(ts, ts[1:])] + ([period] if ts else [])

    def mask(state, view):
        if state != "visible":
            return 0
        return 1 << legend.index(view) if view in legend else None

    cams = [mask(p[4], p[5]) for p in pts]
    cam_s = {n: 0.0 for n in legend}
    cam_s["none"] = 0.0
    seen = {"auto": 0.0, "teleop": 0.0}
    for p, mk, w in zip(pts, cams, weights):
        seen["auto" if p[0] < auto_s else "teleop"] += w
        if mk == 0:
            cam_s["none"] += w
        for i, n in enumerate(legend):
            if mk and mk >> i & 1:
                cam_s[n] += w
    n_vis = sum(1 for p in pts if p[4] == "visible")
    located = sum(seen.values())

    def t_at(rf):
        at = clock.at(rf)
        return round(at[1], 2) if at else None

    team = int(doc["team"])
    ident = []
    if pts:
        ident = [{"slot": "GT", "start_frame": pts[0][3], "end_frame": pts[-1][3],
                  "start_t": t_at(pts[0][3]), "end_t": t_at(pts[-1][3]),
                  "source": GT_SOURCE, "read_camera": None, "score": 1.0,
                  "confirmed": True, "annotator": doc.get("annotator")}]
    station = doc.get("station")
    alliance = doc.get("alliance")
    for a in ("red", "blue"):                       # trust the TBA lineup if it lists the team
        lineup = md.alliances.get(a) or []
        if team in lineup:
            alliance, station = a, lineup.index(team) + 1
    return {
        "team_number": team, "alliance": alliance, "alliance_position": station,
        "path": {
            "n_points": len(pts),
            "seen_s": {k: round(v, 1) for k, v in seen.items()},
            "first_t": round(pts[0][0], 2) if pts else None,
            "last_t": round(pts[-1][0], 2) if pts else None,
            "auto_source": "frames" if any(p[0] < auto_s for p in pts) else None,
            "auto_label_share": None,
            "confirmed_share": 1.0 if pts else None,
            "visible_share": round(n_vis / len(pts), 3) if pts else None,
            "segments": segments, "n_jumps": jumps,
            "camera_s": {k: round(v, 1) for k, v in cam_s.items()},
            "t": [round(p[0], 2) for p in pts], "x": [round(p[1], 3) for p in pts],
            "y": [round(p[2], 3) for p in pts], "frame": [p[3] for p in pts],
            "cams": cams, "state": [p[4] for p in pts],
        },
        "identity": ident,
        "identity_s": {GT_SOURCE: round(located, 1)} if pts else {},
    }


def build(match_key: str) -> dict:
    docs = list_gt(match_key)
    if not docs:
        raise SystemExit(f"{match_key}: no GT files in {GT_DIR / match_key}")
    event_key = docs[0].get("event_key") or match_key.split("_")[0]
    matches, skipped = api._load(event_key)
    md = next((m for m in matches if m.match_key == match_key), None)
    if md is None:
        why = next((s.get("reason") for s in skipped if s.get("match_key") == match_key), None)
        raise SystemExit(f"{match_key}: no usable pipeline replay to take the match clock "
                         f"from ({why or 'not analysed'})")
    tl = timeline.for_match(md)
    clock = MatchClock(md.periods)
    legend = export.camera_legend([md])
    vid = md.video_path if md.video_path and Path(md.video_path).exists() else None
    replay_fp = video_fingerprint(vid) if vid else None
    replay_fps = float(((tl or {}).get("video") or {}).get("fps") or clock.frames_per_s() or 60.0)
    auto = clock.periods.get("auto")
    auto_s = export._auto_s(md)

    robots, provenance = [], []
    for doc in sorted(docs, key=lambda d: (d.get("alliance") != "red", d.get("station") or 9)):
        try:
            al = align_gt(doc, replay_fingerprint=replay_fp,
                          replay_auto_start=auto.start_frame if auto else None,
                          replay_src_fps=replay_fps)
        except AlignmentError as e:
            raise SystemExit(f"{match_key}: {e}")
        robots.append(_robot(doc, md, clock, legend,
                             lambda g, a=al: a["scale"] * g + a["offset"], auto_s))
        provenance.append({"team_number": int(doc["team"]), "annotator": doc.get("annotator"),
                           "created_at": doc.get("created_at"),
                           "updated_at": doc.get("updated_at"),
                           "n_keyframes": len(doc.get("keyframes") or []),
                           "click_height_m": doc.get("click_height_m"),
                           "alignment": al["mode"], "notes": doc.get("notes") or ""})
    order = {(a, i + 1): k for k, (a, i) in enumerate(
        (a, i) for a in ("red", "blue") for i in range(3))}
    robots.sort(key=lambda r: (order.get((r["alliance"], r["alliance_position"]), 99),
                               r["team_number"]))

    row = export.match_row(md, tl, paths=False)
    row["unidentified_s"] = {"auto": 0.0, "teleop": 0.0}
    row["robots_version"] = export.ROBOTS_VERSION
    row["auto_s"] = auto_s
    row["robots"] = robots
    row["unidentified"] = []
    annotators = sorted({p["annotator"] or "?" for p in provenance})
    lineup = [t for a in ("red", "blue") for t in (md.alliances.get(a) or []) if t]
    return {
        "schema": export.MATCH_SCHEMA, "schema_version": export.EXPORT_SCHEMA_VERSION,
        "event_key": event_key, "year": md.year,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path_frame": export.path_frame(md.year), "camera_legend": legend,
        "cameras": export.cameras_block(event_key, [md]),
        "ground_truth": {
            "kind": "simulated" if annotators == [SIM_ANNOTATOR] else "hand_annotated",
            "annotators": annotators,
            "note": ("Robot paths are ground truth, not pipeline output: a person followed "
                     "each robot through the match video and clicked its bumper footprint "
                     "on the carpet; clicks were projected to field metres with the "
                     "event's camera calibration. path.state per point: visible = clear "
                     "view, estimated = occluded best guess (cams 0). Only annotated "
                     "robots are listed. The match header (lineup, timeline, cameras, "
                     "replay_id) comes from the pipeline replay whose match clock gives t."),
            "teams_annotated": [p["team_number"] for p in provenance],
            "teams_missing": [t for t in lineup
                              if t not in {p["team_number"] for p in provenance}],
            "robots": provenance,
        },
        "match": row,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("match_keys", nargs="*", help="default: every hand-annotated GT match")
    ap.add_argument("--include-sim", action="store_true",
                    help=f"also export simulated truth (annotator {SIM_ANNOTATOR!r})")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--upload", action="store_true",
                    help=f"copy each file to the Drive folder's {DRIVE_DIR}/ (rclone)")
    args = ap.parse_args()

    keys = args.match_keys or list_gt_matches()
    if not args.match_keys and not args.include_sim:
        keys = [k for k in keys
                if any(d.get("annotator") != SIM_ANNOTATOR for d in list_gt(k))]
    written = []
    for mk in keys:
        data = build(mk)
        out = args.out / f"{mk}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(export.pretty(data))
        tmp.replace(out)
        written.append(out)
        rs = data["match"]["robots"]
        print(f"{out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB): "
              f"{len(rs)} robots, {sum(r['path']['n_points'] for r in rs)} points"
              + (f", missing {data['ground_truth']['teams_missing']}"
                 if data["ground_truth"]["teams_missing"] else ""))
    if args.upload:
        from web import gdrive_upload
        cfg = gdrive_upload.config()
        if not cfg["configured"]:
            raise SystemExit(f"Drive not configured: {gdrive_upload._why_not(cfg)}")
        for out in written:
            gdrive_upload._copyto(out, f"{DRIVE_DIR}/{out.name}", gdrive_upload.JSON_TIMEOUT_S)
            print(f"uploaded {DRIVE_DIR}/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

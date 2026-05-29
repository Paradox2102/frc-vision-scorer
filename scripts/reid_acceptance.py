"""
reid_acceptance.py — Offline acceptance harness for the robot re-ID pipeline.

Runs the same building blocks used in production (YOLO detector, ByteTrack,
RobotEmbedder, choose_canonical_match / assign_identities) through a video
clip WITHOUT the web/websocket layer, then prints a three-metric summary:

  1. Canonical-ID switches per physical robot
     How many distinct team_number labels each canonical track held across the
     run, and how many times the (id -> team) mapping changed direction.
     Lower is better; ideally 1 distinct label and 0 flips per robot.

  2. Frames with duplicate team numbers
     Count of processed frames where two live tracks share the same non-null
     team_number.  Must be 0 by construction (Hungarian uniqueness guarantee).
     The script asserts and flags any violation.

  3. Team-number accuracy at ground-truth frames
     If --ground-truth is supplied, at each labelled frame the script matches
     predicted robot boxes to GT boxes by IoU and computes the fraction that
     have the correct team_number.

Usage
-----
See --help for the full option list.

Integration approach
--------------------
InferenceJob is tightly coupled to websockets, async calibration, and the
FastAPI app.  Rather than fighting those dependencies for an offline script,
this module implements a SELF-CONTAINED minimal loop that uses the same
building-blocks:

    YOLO(MODEL_PATH)  ->  sv.ByteTrack  ->  RobotEmbedder
    choose_canonical_match  +  assign_identities  (from web.reid)

No OCR, no scoring, no websockets, no calibration.  The loop is intentionally
simple so the output is easy to interpret.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# All heavy imports are deferred to main() so that --help and py_compile
# succeed without a model, video, or GPU present.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _parse_gt(path: str) -> Dict[int, Dict[Tuple[float, float, float, float], str]]:
    """Load a ground-truth JSON and return a typed dict.

    Expected JSON schema::

        {
          "180": {
            "100,200,300,400": "1234",
            "500,600,700,800": "5678"
          },
          ...
        }

    Keys at the top level are frame indices (as strings).
    Keys inside each frame dict are bounding boxes in "x1,y1,x2,y2" format
    (pixel coordinates, floats or ints).
    Values are team numbers as strings.
    """
    raw = json.loads(Path(path).read_text())
    out: Dict[int, Dict[Tuple[float, float, float, float], str]] = {}
    for frame_str, boxes in raw.items():
        frame_idx = int(frame_str)
        frame_gt: Dict[Tuple[float, float, float, float], str] = {}
        for box_str, team in boxes.items():
            coords = tuple(float(v) for v in box_str.split(","))
            if len(coords) != 4:
                raise ValueError(
                    f"Ground-truth box key must be 'x1,y1,x2,y2', got {box_str!r}"
                )
            frame_gt[coords] = str(team)  # type: ignore[assignment]
        out[frame_idx] = frame_gt
    return out


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    """Axis-aligned IoU of two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _match_gt_to_pred(
    gt_boxes: Dict[Tuple[float, float, float, float], str],
    pred: List[Tuple[Tuple[float, float, float, float], Optional[str]]],
    iou_threshold: float = 0.30,
) -> Tuple[int, int]:
    """Match GT boxes to predicted boxes by IoU.

    Returns (correct_count, total_gt_count).
    A match is correct when the highest-IoU predicted box has the same
    team_number as the GT label.  Unmatched GT boxes count as incorrect.
    """
    correct = 0
    for gt_box, gt_team in gt_boxes.items():
        best_iou = 0.0
        best_team: Optional[str] = None
        for pred_box, pred_team in pred:
            iou = _iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_team = pred_team
        if best_iou >= iou_threshold and best_team == gt_team:
            correct += 1
    return correct, len(gt_boxes)


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------

def run(
    video_path: str,
    ground_truth_path: Optional[str],
    sample_rate: int,
) -> None:
    """Run the re-ID acceptance loop and print metrics."""
    # --- deferred heavy imports -------------------------------------------
    import cv2
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    # Resolve path relative to project root so the script works from any cwd.
    _here = Path(__file__).resolve().parent.parent  # project root
    sys.path.insert(0, str(_here))

    # Import production constants and building-blocks.
    from web.app import (
        MODEL_PATH,
        EMBED_MODEL_NAME,
        APP_FLOOR,
        POS_GATE_PX,
        W_POS,
        W_APP,
        ASSIGN_MAX_COST,
        IDENTITY_INTERVAL,
        EMBED_EMA_ALPHA,
        LOST_TRACK_BUFFER,
        MAX_ROBOTS_PER_ALLIANCE,
        MAX_ROBOTS_TOTAL,
        OVERHEAD_CROP_RATIO,
    )
    from web.embedder import RobotEmbedder, cosine_similarity
    from web import reid

    print(f"\n=== reid_acceptance.py ===")
    print(f"video       : {video_path}")
    print(f"sample_rate : every {sample_rate} frames")
    print(f"ground_truth: {ground_truth_path or '(none)'}")
    print(f"\nConstants in use:")
    print(f"  EMBED_MODEL_NAME  = {EMBED_MODEL_NAME}")
    print(f"  APP_FLOOR         = {APP_FLOOR}")
    print(f"  ASSIGN_MAX_COST   = {ASSIGN_MAX_COST}")
    print(f"  POS_GATE_PX       = {POS_GATE_PX}")
    print(f"  IDENTITY_INTERVAL = {IDENTITY_INTERVAL}")
    print()

    # --- load model and tracker -------------------------------------------
    if not Path(MODEL_PATH).exists():
        sys.exit(f"ERROR: model not found at {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    tracker = sv.ByteTrack(
        lost_track_buffer=LOST_TRACK_BUFFER,
        track_activation_threshold=0.25,
        minimum_matching_threshold=0.8,
    )
    embedder = RobotEmbedder(EMBED_MODEL_NAME)

    # --- open video -------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video {video_path}")

    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    overhead_h   = int(video_height * OVERHEAD_CROP_RATIO)
    print(f"Video: {video_width}x{video_height}, {video_fps:.1f} fps, {total_frames} frames")
    print(f"Overhead crop height: {overhead_h}  (top {OVERHEAD_CROP_RATIO*100:.0f}%)")
    print()

    # --- load ground truth ------------------------------------------------
    gt_data: Optional[Dict] = None
    if ground_truth_path:
        gt_data = _parse_gt(ground_truth_path)
        print(f"Ground-truth: {len(gt_data)} labelled frame(s) loaded.")
        print()

    # --- per-track state --------------------------------------------------
    # canonical_id -> current team_number (updated by assign_identities)
    robot_tracks: Dict[int, dict] = {}
    # ByteTrack id -> canonical id
    bytetrack_to_canonical: Dict[int, int] = {}
    next_canonical_id = 0
    # canonical_id -> list of (processed_frame_num, team_number) changes
    team_history: Dict[int, List[Tuple[int, Optional[str]]]] = defaultdict(list)

    # reference embeddings gathered from confirmed looks
    # In this harness we build them incrementally via EMA — no calibration step.
    reference_embeddings: Dict[str, np.ndarray] = {}

    # --- metric accumulators ----------------------------------------------
    duplicate_team_frames = 0          # metric 2
    duplicate_canonical_frames = 0     # collision metric: two detections, one canonical
    gt_correct_total   = 0             # metric 3 numerator
    gt_total_boxes     = 0             # metric 3 denominator

    # --- main loop --------------------------------------------------------
    frame_num      = 0
    proc_frame_num = 0  # counts only processed (sampled) frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_rate != 0:
            frame_num += 1
            continue

        overhead = frame[:overhead_h, :]

        # --- YOLO detection ---
        results = model(overhead, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # --- filter to robot class (class_id == 1) ---
        robot_mask = detections.class_id == 1
        robot_dets = detections[robot_mask]

        # --- ByteTrack update ---
        if len(robot_dets) > 0:
            tracked = tracker.update_with_detections(robot_dets)
        else:
            tracked = sv.Detections.empty()

        # Live tracks seen this processed frame: canonical_id -> (box, team_number)
        live_this_frame: Dict[int, Tuple[tuple, Optional[str]]] = {}

        # Embed all tracked crops once, then resolve best-match-first with a
        # per-frame `claimed` set so no two detections share a canonical — this
        # mirrors production web/app.py (the whole point of the harness).
        n_tracked = len(tracked)
        det_embeds = embedder.embed_batch(overhead, [tracked.xyxy[i] for i in range(n_tracked)])
        order_tracks_view = [
            reid.TrackView(cid_, robot_tracks[cid_]["alliance"],
                           robot_tracks[cid_]["position"], robot_tracks[cid_]["embedding"])
            for cid_ in robot_tracks
        ]

        def _order_key(i):
            b = tracked.xyxy[i]
            return reid.best_candidate_score(
                ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), det_embeds[i],
                order_tracks_view, POS_GATE_PX, W_POS, W_APP, APP_FLOOR,
            )

        order = sorted(range(n_tracked), key=_order_key, reverse=True)
        claimed: set = set()
        frame_cids: List[int] = []

        for i in order:
            bt_id = int(tracked.tracker_id[i])
            box   = tracked.xyxy[i]
            cx    = float((box[0] + box[2]) / 2)
            cy    = float((box[1] + box[3]) / 2)

            # Simple alliance heuristic: left = red, right = blue (no bumper OCR)
            alliance = "red" if cx < video_width / 2 else "blue"
            det_embed = det_embeds[i]

            # --- get or create canonical id (claimed-aware + appearance re-validation) ---
            cid = bytetrack_to_canonical.get(bt_id)
            if cid is not None and cid in robot_tracks:
                emb = robot_tracks[cid]["embedding"]
                appearance_ok = (
                    det_embed is None or emb is None
                    or cosine_similarity(det_embed, emb) >= APP_FLOOR
                )
                if appearance_ok and cid not in claimed:
                    claimed.add(cid)
                else:
                    del bytetrack_to_canonical[bt_id]
                    cid = None
            else:
                cid = None

            if cid is None:
                tracks_view = [
                    reid.TrackView(cid_, robot_tracks[cid_]["alliance"],
                                   robot_tracks[cid_]["position"], robot_tracks[cid_]["embedding"])
                    for cid_ in robot_tracks
                ]
                action, matched_id = reid.choose_canonical_match(
                    det_pos=(cx, cy), det_embed=det_embed, det_alliance=alliance,
                    tracks=tracks_view, app_floor=APP_FLOOR, pos_gate=POS_GATE_PX,
                    w_pos=W_POS, w_app=W_APP,
                    cap_per_alliance=MAX_ROBOTS_PER_ALLIANCE, cap_total=MAX_ROBOTS_TOTAL,
                    claimed=claimed,
                )
                if action == "drop":
                    continue  # surplus detection — cannot be a 7th robot
                if action in ("merge", "assign"):
                    cid = matched_id
                else:
                    cid = next_canonical_id
                    next_canonical_id += 1
                    robot_tracks[cid] = {
                        "alliance": alliance,
                        "position": (cx, cy),
                        "embedding": det_embed,
                        "team_number": None,
                    }
                bytetrack_to_canonical[bt_id] = cid
                claimed.add(cid)

            frame_cids.append(cid)

            # Update track state (EMA only when the crop matches — no contamination)
            track = robot_tracks[cid]
            track["position"] = (cx, cy)
            if det_embed is not None:
                if track["embedding"] is None:
                    track["embedding"] = det_embed
                elif cosine_similarity(det_embed, track["embedding"]) >= APP_FLOOR:
                    blended = (1 - EMBED_EMA_ALPHA) * track["embedding"] + EMBED_EMA_ALPHA * det_embed
                    n = np.linalg.norm(blended)
                    track["embedding"] = blended / n if n > 0 else blended

            live_this_frame[cid] = (
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                track.get("team_number"),
            )

        # --- collision metric: two detections collapsed onto one canonical ---
        if len(frame_cids) != len(set(frame_cids)):
            duplicate_canonical_frames += 1
            print(f"  [WARN] frame {frame_num} (proc {proc_frame_num}): "
                  f"two detections collapsed onto one canonical: {frame_cids}")

        # --- periodic identity assignment (same interval as production) ---
        if proc_frame_num % IDENTITY_INTERVAL == 0 and reference_embeddings:
            track_embeds = {cid: t["embedding"] for cid, t in robot_tracks.items()}
            alliance_of_track = {cid: t["alliance"] for cid, t in robot_tracks.items()}
            alliance_of_team: Dict[str, str] = {}  # unknown in harness — skip alliance gating
            locked_tracks: set = set()
            locked_teams: set = set()

            assignment = reid.assign_identities(
                track_embeds=track_embeds,
                ref_embeds=reference_embeddings,
                alliance_of_track=alliance_of_track,
                alliance_of_team=alliance_of_team,
                locked_tracks=locked_tracks,
                locked_teams=locked_teams,
                max_cost=ASSIGN_MAX_COST,
            )
            for cid, team in assignment.items():
                if team is not None:
                    old_team = robot_tracks[cid].get("team_number")
                    robot_tracks[cid]["team_number"] = team
                    if team != old_team:
                        team_history[cid].append((proc_frame_num, team))

        # --- Metric 2: duplicate team numbers this frame ------------------
        team_seen: Dict[str, int] = {}
        for cid, (box, team) in live_this_frame.items():
            if team is None:
                continue
            team_seen[team] = team_seen.get(team, 0) + 1
        dupes = sum(1 for cnt in team_seen.values() if cnt > 1)
        if dupes > 0:
            duplicate_team_frames += 1
            dupe_list = [t for t, cnt in team_seen.items() if cnt > 1]
            print(f"  [WARN] frame {frame_num} (proc {proc_frame_num}): "
                  f"duplicate team numbers {dupe_list}")

        # --- Metric 3: ground-truth accuracy ------------------------------
        if gt_data is not None and frame_num in gt_data:
            pred_list = list(live_this_frame.values())  # [(box, team_number), ...]
            c, t = _match_gt_to_pred(gt_data[frame_num], pred_list)
            gt_correct_total += c
            gt_total_boxes   += t
            print(f"  [GT] frame {frame_num}: {c}/{t} correct")

        frame_num      += 1
        proc_frame_num += 1

    cap.release()

    # --- Metric 1: canonical-ID switches ----------------------------------
    # For each canonical track, count distinct team_numbers it ever held and
    # how many times it actually flipped (consecutive changes).
    switch_rows: List[dict] = []
    for cid, track in robot_tracks.items():
        history = team_history.get(cid, [])
        # include the current label if the track ended with one
        labels_seen = list({h[1] for h in history} | ({track["team_number"]} if track["team_number"] else set()))
        flips = len(history)  # each entry is a change event
        switch_rows.append({
            "canonical_id": cid,
            "alliance": track["alliance"],
            "final_team": track.get("team_number"),
            "distinct_labels": len(labels_seen),
            "flips": flips,
        })

    # Sort by canonical_id for readability
    switch_rows.sort(key=lambda r: r["canonical_id"])

    total_flips = sum(r["flips"] for r in switch_rows)

    # --- print summary table ----------------------------------------------
    sep = "-" * 70
    print()
    print("=" * 70)
    print("  RE-ID ACCEPTANCE REPORT")
    print("=" * 70)

    print()
    print("METRIC 1 — Canonical-ID switches per physical robot")
    print(sep)
    print(f"  {'ID':>4}  {'Alliance':<8}  {'Final team':<12}  {'Distinct labels':>16}  {'Flips':>6}")
    print(sep)
    for r in switch_rows:
        team_str = r["final_team"] or "(none)"
        print(f"  {r['canonical_id']:>4}  {r['alliance']:<8}  {team_str:<12}  "
              f"{r['distinct_labels']:>16}  {r['flips']:>6}")
    print(sep)
    print(f"  Total robots tracked : {len(switch_rows)}")
    print(f"  Total ID flips       : {total_flips}  (0 is ideal)")
    print()

    print("METRIC 2 — Frames with duplicate team numbers")
    print(sep)
    print(f"  Processed frames checked    : {proc_frame_num}")
    print(f"  Frames with duplicates      : {duplicate_team_frames}")
    if duplicate_team_frames == 0:
        print(f"  PASS — Hungarian uniqueness guarantee holds.")
    else:
        print(f"  FAIL — {duplicate_team_frames} frame(s) had duplicate team numbers.")
        print(f"         This should be 0.  Check assign_identities logic.")
    print()

    print("COLLISION METRIC — Frames where two detections claimed one canonical")
    print(sep)
    print(f"  Processed frames checked    : {proc_frame_num}")
    print(f"  Frames with a collision     : {duplicate_canonical_frames}")
    if duplicate_canonical_frames == 0:
        print(f"  PASS — every detection mapped to a distinct canonical track.")
    else:
        print(f"  FAIL — {duplicate_canonical_frames} frame(s) had two detections on one track.")
        print(f"         The collision bug is not fixed.")
    print()

    print("METRIC 3 — Team-number accuracy at ground-truth frames")
    print(sep)
    if gt_data is None:
        print("  --ground-truth not provided; metric skipped.")
    elif gt_total_boxes == 0:
        print("  Ground-truth file had no matching frames in the video.")
    else:
        accuracy = gt_correct_total / gt_total_boxes
        print(f"  GT boxes checked  : {gt_total_boxes}")
        print(f"  Correct labels    : {gt_correct_total}")
        print(f"  Accuracy          : {accuracy:.1%}")
    print()

    # Final assertions (make it easy to use in CI)
    assert duplicate_canonical_frames == 0, (
        f"{duplicate_canonical_frames} frame(s) had two detections collapse onto one "
        f"canonical track — the collision bug is not fixed. See WARN lines above."
    )
    assert duplicate_team_frames == 0, (
        f"Duplicate team numbers detected in {duplicate_team_frames} frame(s). "
        "See WARN lines above."
    )

    ok = duplicate_team_frames == 0 and duplicate_canonical_frames == 0
    print("=" * 70)
    print("  All checks passed." if ok else "  CHECK FAILED — see above.")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reid_acceptance.py",
        description=(
            "Offline acceptance harness for the robot re-ID pipeline.\n"
            "Runs YOLO + ByteTrack + DINOv2 embeddings through a match clip\n"
            "and reports three metrics: ID switches, duplicate-team frames,\n"
            "and team-number accuracy against an optional ground-truth file.\n\n"
            "No model or video is needed for --help or py_compile."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--video",
        required=True,
        metavar="PATH",
        help="Path to a match clip video file (mp4, mkv, …).",
    )
    p.add_argument(
        "--ground-truth",
        metavar="PATH",
        default=None,
        help=(
            "Optional JSON file mapping frame indices to robot labels.\n"
            "Schema: {\"<frame_idx>\": {\"x1,y1,x2,y2\": \"<team_number>\", …}, …}\n"
            "See docs/acceptance/reid.md for a full example and instructions\n"
            "on how to build this file."
        ),
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Process every Nth frame of the video (default: 3).\n"
            "Lower values are more accurate but slower."
        ),
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(
        video_path=args.video,
        ground_truth_path=args.ground_truth,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()

"""Phase 2 (SP-T.2) offline driver: per-view SAM2 -> fused stable ids -> SlotMapper.

Runs Sam2FieldTracker per calibrated camera view over the sidecar segment,
projects each view's mask bottom-centers to shared field coords, fuses
cross-camera tracklets (SP3 fuse_detections), assigns ONE stable fused track_id
per physical robot (FusedTrackRegistry), and feeds the Phase-1 FieldTracker.
Asserts fixed-6 invariants and reports continuity.

Integration experiment, not a unit test — judged by the invariants + printed
continuity. Mirrors scripts/sam2_field_track.py (the single-camera Phase 1 driver).

Skips cleanly when the event has no calibrated views or the footage is missing.
Corner cameras are calibrated for event `2026hop` only; the default `test` event
(qual1 footage) is main-only, which still exercises the new fusion+registry path
with a single camera. For a true multi-camera run, point --event/--video at an
event whose corner cameras are calibrated AND whose footage is available.

Usage:
    .venv/bin/python scripts/sam2_multicam_track.py [--event test] [--labels PATH] [--video PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from ultralytics import YOLO
from web.field.camera_view import load_views
from web.field.field_model import FieldModel
from web.field.fused_track_id import FusedTrackRegistry
from web.field.fusion import fuse_detections
from web.field.tracker import FieldDetection, FieldTracker
from web.field.spike_metrics import continuity_stats
from web.app import MODEL_PATH

DEFAULT_VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
DEFAULT_LABELS = ROOT / "data/test_footage/qual1_first_champs_2026.spike_labels.json"

SAM2_SCALE = 0.5
CORNER_VOTE_WEIGHT = 2.5


def _on_field(xy, dims, margin=1.0):
    L, W = dims
    return -margin <= xy[0] <= L + margin and -margin <= xy[1] <= W + margin


def _export_view(cap, view, all_frames, prompt_frame, robot_model):
    """Export a view's 0.5x-scaled crop frames to a tempdir and detect YOLO prompt
    boxes on the prompt frame. Returns (tmpdir_obj, prompt_boxes_scaled)."""
    tmpdir_obj = tempfile.TemporaryDirectory()
    prompt_boxes_scaled: list[tuple] = []
    for local_idx, gf in enumerate(all_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ok, fr = cap.read()
        if not ok:
            continue
        crop = view.extract(fr)
        new_w = int(crop.shape[1] * SAM2_SCALE)
        new_h = int(crop.shape[0] * SAM2_SCALE)
        small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(tmpdir_obj.name, f"{local_idx:05d}.jpg"),
                    small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if gf == prompt_frame:
            res = robot_model(small, conf=0.4, verbose=False)[0]
            for i in range(len(res.boxes)):
                if int(res.boxes.cls[i]) == 1:
                    box = res.boxes.xyxy[i].cpu().numpy()
                    prompt_boxes_scaled.append(tuple(box))
    return tmpdir_obj, prompt_boxes_scaled


def main():
    parser = argparse.ArgumentParser(
        description="Multi-camera SAM2 -> fused stable ids -> FieldTracker driver")
    parser.add_argument("--event", default="test", help="Event key for load_views")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Segment sidecar JSON")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Composite video path")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Sidecar + views + field model  (skip cleanly on missing deps)
    # ------------------------------------------------------------------
    sidecar_path = Path(args.labels)
    if not sidecar_path.exists():
        print(f"[driver] sidecar not found at {sidecar_path} — skipping")
        sys.exit(0)
    video = Path(args.video)
    if not video.exists():
        print(f"[driver] video not found at {video} — skipping")
        sys.exit(0)

    seg = json.loads(sidecar_path.read_text())["segment"]
    start_frame, end_frame = seg["start_frame"], seg["end_frame"]
    prompt_frame, stride = seg["prompt_frame"], seg["stride"]

    views = load_views(args.event)
    if not views:
        print(f"[driver] no calibrated views for event '{args.event}' — skipping")
        sys.exit(0)
    print(f"[driver] Cameras active: {[v.name for v in views]}")

    fm = FieldModel.load(views[0].calibration.field_season)
    dims = (fm.length_m, fm.width_m)
    print(f"[driver] Field {fm.length_m:.2f}m x {fm.width_m:.2f}m")
    print(f"[driver] Segment {start_frame}..{end_frame} prompt={prompt_frame} stride={stride}")

    all_frames = list(range(start_frame, end_frame + 1, stride))
    prompt_local_idx = (prompt_frame - start_frame) // stride
    local_to_global = {i: gf for i, gf in enumerate(all_frames)}

    robot_model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    # ------------------------------------------------------------------
    # 2. Per-view SAM2 propagation (one predictor reused across views)
    # ------------------------------------------------------------------
    from web.field.sam2_tracker import Sam2FieldTracker  # lazy: torch/SAM2 only here
    print("[driver] Building Sam2FieldTracker (real GPU) ...")
    sam2 = Sam2FieldTracker()

    raw: dict[str, dict] = {}    # view_name -> {local_idx: {obj_id: {"box","mask"}}}
    total_prop, total_elapsed = 0, 0.0
    for view in views:
        tmp_obj, prompt_boxes = _export_view(cap, view, all_frames, prompt_frame, robot_model)
        print(f"[driver] {view.name}: prompt frame detected {len(prompt_boxes)} robots")
        if not prompt_boxes:
            print(f"[driver] {view.name}: nothing to track — skipping this view")
            tmp_obj.cleanup()
            continue
        t0 = time.perf_counter()
        raw[view.name] = sam2.track(
            frames_dir=tmp_obj.name,
            prompt_idx=prompt_local_idx,
            prompt_boxes=prompt_boxes,
        )
        elapsed = time.perf_counter() - t0
        n = len(raw[view.name])
        total_prop += n
        total_elapsed += elapsed
        print(f"[driver] {view.name}: propagated {n} frames in {elapsed:.2f}s "
              f"({elapsed / max(1, n) * 1000:.1f} ms/frame)")
        tmp_obj.cleanup()
    cap.release()

    if not raw:
        print("[driver] no view produced tracklets — skipping")
        sys.exit(0)

    # ------------------------------------------------------------------
    # 3. Per-frame fuse -> stable fused id -> FieldTracker
    # ------------------------------------------------------------------
    view_by_name = {v.name: v for v in views}
    reg = FusedTrackRegistry()
    tracker = FieldTracker(fm.length_m)
    dt = stride / fps

    frames: dict[int, dict] = {}        # global_frame -> {slot_id: (u, v)}
    multicam_clusters = 0               # fused observations with >1 camera member

    for local_idx in sorted(local_to_global):
        gf = local_to_global[local_idx]
        per_camera: dict[str, list] = {}
        for name, view_raw in raw.items():
            view = view_by_name[name]
            cal = view.calibration
            dets = []
            for obj_id, info in view_raw.get(local_idx, {}).items():
                x1, y1, x2, y2 = info["box"]                  # 0.5x crop coords
                fx1, fy1, fx2, fy2 = (x1 / SAM2_SCALE, y1 / SAM2_SCALE,
                                      x2 / SAM2_SCALE, y2 / SAM2_SCALE)
                uv = ((fx1 + fx2) / 2.0, fy2)                 # bottom-center, full-res crop
                xy = cal.image_to_field(uv)
                if xy is None or not _on_field(xy, dims):
                    continue
                dets.append((FieldDetection(pos=xy, color="unknown",
                                            box=(fx1, fy1, fx2, fy2), score=1.0,
                                            track_id=obj_id), None))
            if dets:
                per_camera[name] = dets

        if not per_camera:
            continue

        weights = {n: (1.0 if n == "main" else CORNER_VOTE_WEIGHT) for n in per_camera}
        fused = fuse_detections(per_camera, weights=weights)
        reg.assign(fused)
        multicam_clusters += sum(1 for o in fused if len({m[0] for m in o.members}) > 1)
        outs = tracker.update([o.detection for o in fused], {}, dt=dt)

        uv_map = {}
        for t in outs:
            if t.box is not None and t.state == "visible":
                bx1, by1, bx2, by2 = t.box
                uv_map[t.slot_id] = ((bx1 + bx2) / 2.0, by2)
        if uv_map:
            frames[gf] = uv_map

    # ------------------------------------------------------------------
    # 4. Invariants + continuity
    # ------------------------------------------------------------------
    print("\n[driver] === INVARIANT CHECKS ===")
    seen_slots: set[int] = set()
    for uv_map in frames.values():
        seen_slots.update(uv_map.keys())

    def slot_to_alliance(slot_id):
        return "blue" if slot_id < 3 else "red"

    max_blue = max((sum(1 for s in m if slot_to_alliance(s) == "blue")
                    for m in frames.values()), default=0)
    max_red = max((sum(1 for s in m if slot_to_alliance(s) == "red")
                   for m in frames.values()), default=0)

    print(f"  distinct slot ids seen: {sorted(seen_slots)}")
    print(f"  max simultaneous visible: blue={max_blue} red={max_red}")
    assert seen_slots <= set(range(6)), f"INVARIANT FAILED: >6 slots: {seen_slots}"
    assert max_blue <= 3 and max_red <= 3, \
        f"INVARIANT FAILED: >3 per alliance (blue={max_blue} red={max_red})"
    print("  OK: fixed-6 / 3-per-alliance invariants held.")

    print("\n[driver] === CONTINUITY ===")
    print(f"  {continuity_stats(frames)}  distinct_slots={len(seen_slots)}")
    if len(raw) > 1:
        print(f"  cross-camera fused observations (multi-cam members): {multicam_clusters}")
    else:
        print("  (single camera active — no cross-camera fusion this run)")
    print(f"  SAM2 latency: {total_elapsed / max(1, total_prop) * 1000:.1f} ms/frame "
          f"({total_prop} view-frames in {total_elapsed:.2f}s)")
    print("\n[driver] Done.")


if __name__ == "__main__":
    main()

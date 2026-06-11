"""Task 1.4 offline driver: Sam2FieldTracker -> FieldTracker(track_id) end-to-end.

Runs real SAM2 GPU tracking over the sidecar segment, feeds the stable track_id
path through FieldTracker, asserts fixed-6 invariants, and compares continuity
stats against the anonymous (distance-only) path.

Usage:
    .venv/bin/python scripts/sam2_field_track.py [--labels PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ultralytics import YOLO
from web.field import store as field_store
from web.field.field_model import FieldModel
from web.field.tracker import FieldDetection, FieldTracker
from web.field.sam2_tracker import Sam2FieldTracker
from web.field.spike_metrics import continuity_stats
from web.app import MODEL_PATH, OVERHEAD_CROP_RATIO

VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
DEFAULT_LABELS = ROOT / "data/test_footage/qual1_first_champs_2026.spike_labels.json"

SAM2_SCALE = 0.5


def _main_crop(frame: np.ndarray, ratio: float = OVERHEAD_CROP_RATIO) -> np.ndarray:
    h = frame.shape[0]
    return frame[: int(h * ratio), :]


def _on_field(xy, dims, margin=1.0):
    L, W = dims
    return -margin <= xy[0] <= L + margin and -margin <= xy[1] <= W + margin


def main():
    parser = argparse.ArgumentParser(description="Sam2FieldTracker -> FieldTracker end-to-end driver")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Spike labels JSON sidecar")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load sidecar segment + calibration + FieldModel
    # ------------------------------------------------------------------
    sidecar_path = Path(args.labels)
    if not sidecar_path.exists():
        print(f"[driver] ERROR: sidecar not found at {sidecar_path}")
        sys.exit(1)

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    seg = sidecar["segment"]
    start_frame = seg["start_frame"]
    end_frame = seg["end_frame"]
    prompt_frame = seg["prompt_frame"]
    stride = seg["stride"]

    print(f"[driver] Sidecar loaded: segment={start_frame}..{end_frame} "
          f"prompt={prompt_frame} stride={stride}")

    if not VIDEO.exists():
        print(f"[driver] ERROR: video not found at {VIDEO}")
        sys.exit(1)

    cal = field_store.load("test", "main")
    if cal is None or cal.camera_center is None:
        print("[driver] No pose-bearing calibration for test/main — skipping")
        sys.exit(0)

    fm = FieldModel.load(cal.field_season)
    dims = (fm.length_m, fm.width_m)
    print(f"[driver] Field {fm.length_m:.2f}m x {fm.width_m:.2f}m")

    robot_model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    print(f"[driver] Video fps={fps:.1f}")

    all_frames = list(range(start_frame, end_frame + 1, stride))
    prompt_local_idx = (prompt_frame - start_frame) // stride
    print(f"[driver] Segment has {len(all_frames)} frames; "
          f"prompt local_idx={prompt_local_idx}")

    # ------------------------------------------------------------------
    # 2. Export main-crop frames at 0.5x scale to temp dir
    #    Also detect YOLO robot boxes on the prompt frame's 0.5x main crop
    # ------------------------------------------------------------------
    print(f"[driver] Exporting {len(all_frames)} frames (scale={SAM2_SCALE}) ...")

    prompt_boxes_scaled: list[tuple] = []  # SAM2-scale boxes for prompting
    prompt_boxes_fullres: list[tuple] = []  # full-res boxes for YOLO baseline

    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_obj.name

    local_to_global: dict[int, int] = {}

    for local_idx, gf in enumerate(all_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ok, fr = cap.read()
        if not ok:
            print(f"[driver] WARNING: could not read frame {gf}")
            continue
        main = _main_crop(fr)
        new_w = int(main.shape[1] * SAM2_SCALE)
        new_h = int(main.shape[0] * SAM2_SCALE)
        main_small = cv2.resize(main, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fname = f"{local_idx:05d}.jpg"
        cv2.imwrite(os.path.join(tmpdir, fname), main_small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        local_to_global[local_idx] = gf

        # Detect YOLO robot boxes on the prompt frame at 0.5x scale
        if gf == prompt_frame:
            res = robot_model(main_small, conf=0.4, verbose=False)[0]
            for i in range(len(res.boxes)):
                if int(res.boxes.cls[i]) == 1:
                    box = res.boxes.xyxy[i].cpu().numpy()
                    prompt_boxes_scaled.append(tuple(box))
                    # Rescale to full-res for reference
                    x1, y1, x2, y2 = box
                    prompt_boxes_fullres.append((
                        x1 / SAM2_SCALE, y1 / SAM2_SCALE,
                        x2 / SAM2_SCALE, y2 / SAM2_SCALE,
                    ))

    print(f"[driver] Exported. Prompt frame detected {len(prompt_boxes_scaled)} robots.")

    if not prompt_boxes_scaled:
        print("[driver] WARNING: no robot boxes detected on prompt frame — "
              "SAM2 has nothing to track.")

    # ------------------------------------------------------------------
    # 3. Build Sam2FieldTracker and run .track()
    # ------------------------------------------------------------------
    print("[driver] Building Sam2FieldTracker (real GPU) ...")
    sam2_tracker = Sam2FieldTracker()  # will auto-select CUDA if available

    print("[driver] Running SAM2 propagation ...")
    import time
    t0 = time.perf_counter()
    raw_results = sam2_tracker.track(
        frames_dir=tmpdir,
        prompt_idx=prompt_local_idx,
        prompt_boxes=prompt_boxes_scaled,
    )
    elapsed = time.perf_counter() - t0
    n_prop = len(raw_results)
    ms_per_frame = (elapsed / n_prop * 1000.0) if n_prop > 0 else 0.0
    print(f"[driver] SAM2 propagated {n_prop} frames in {elapsed:.2f}s "
          f"({ms_per_frame:.1f} ms/frame)")

    # Done with temp dir — clean up now that propagation is finished
    tmpdir_obj.cleanup()

    # ------------------------------------------------------------------
    # 4. Feed FieldTracker via the NEW track_id path
    # ------------------------------------------------------------------
    print("[driver] Running FieldTracker with track_id path ...")
    ft_trackid = FieldTracker(fm.length_m)
    dt = stride / fps

    # per-frame {slot_id: bottom_center_uv} for continuity
    trackid_frames: dict[int, dict] = {}  # global_frame -> {slot_id: (u, v)}

    for local_idx in sorted(raw_results.keys()):
        gf = local_to_global.get(local_idx)
        if gf is None:
            continue
        frame_data = raw_results[local_idx]  # {obj_id: {"box": ..., "mask": ...}}

        dets = []
        for obj_id, info in frame_data.items():
            box_scaled = info["box"]  # (x1, y1, x2, y2) in 0.5x coords
            # Rescale box to full-res
            x1, y1, x2, y2 = box_scaled
            fx1, fy1, fx2, fy2 = (
                x1 / SAM2_SCALE, y1 / SAM2_SCALE,
                x2 / SAM2_SCALE, y2 / SAM2_SCALE,
            )
            fullres_box = (fx1, fy1, fx2, fy2)

            # Bottom-center in full-res image space
            uv = ((fx1 + fx2) / 2.0, fy2)

            # Project to field coords
            xy = cal.image_to_field(uv)
            if xy is None or not _on_field(xy, dims):
                continue

            dets.append(FieldDetection(
                pos=xy,
                color="unknown",
                box=fullres_box,
                score=1.0,
                track_id=obj_id,
            ))

        outs = ft_trackid.update(dets, {}, dt=dt)

        uv_map = {}
        for t in outs:
            if t.box is not None and t.state == "visible":
                bx1, by1, bx2, by2 = t.box
                uv_map[t.slot_id] = ((bx1 + bx2) / 2.0, by2)
        if uv_map:
            trackid_frames[gf] = uv_map

    print(f"[driver] track_id path: {len(trackid_frames)} frames with data")

    # ------------------------------------------------------------------
    # 5. Anonymous path for comparison (same field detections, track_id=None)
    # ------------------------------------------------------------------
    print("[driver] Running FieldTracker with anonymous path (track_id=None) ...")
    ft_anon = FieldTracker(fm.length_m)
    anon_frames: dict[int, dict] = {}

    for local_idx in sorted(raw_results.keys()):
        gf = local_to_global.get(local_idx)
        if gf is None:
            continue
        frame_data = raw_results[local_idx]

        dets = []
        for obj_id, info in frame_data.items():
            box_scaled = info["box"]
            x1, y1, x2, y2 = box_scaled
            fx1, fy1, fx2, fy2 = (
                x1 / SAM2_SCALE, y1 / SAM2_SCALE,
                x2 / SAM2_SCALE, y2 / SAM2_SCALE,
            )
            fullres_box = (fx1, fy1, fx2, fy2)
            uv = ((fx1 + fx2) / 2.0, fy2)
            xy = cal.image_to_field(uv)
            if xy is None or not _on_field(xy, dims):
                continue

            dets.append(FieldDetection(
                pos=xy,
                color="unknown",
                box=fullres_box,
                score=1.0,
                track_id=None,  # anonymous path
            ))

        outs = ft_anon.update(dets, {}, dt=dt)

        uv_map = {}
        for t in outs:
            if t.box is not None and t.state == "visible":
                bx1, by1, bx2, by2 = t.box
                uv_map[t.slot_id] = ((bx1 + bx2) / 2.0, by2)
        if uv_map:
            anon_frames[gf] = uv_map

    print(f"[driver] anonymous path: {len(anon_frames)} frames with data")
    cap.release()

    # ------------------------------------------------------------------
    # 6. Assert fixed-6 invariants on the track_id path
    # ------------------------------------------------------------------
    print("\n[driver] === INVARIANT CHECKS (track_id path) ===")

    # Collect all slot_ids ever seen from FieldTracker outputs
    seen_slots: set[int] = set()
    for uv_map in trackid_frames.values():
        seen_slots.update(uv_map.keys())

    # Re-run a second pass to check per-frame per-alliance counts
    # We need to reconstruct alliance info per slot from FieldTracker internals.
    # FieldTracker slots 0-2 = blue, 3-5 = red (by construction in __init__).
    def slot_to_alliance(slot_id: int) -> str:
        return "blue" if slot_id < 3 else "red"

    max_blue_per_frame = 0
    max_red_per_frame = 0
    for uv_map in trackid_frames.values():
        blue_count = sum(1 for s in uv_map if slot_to_alliance(s) == "blue")
        red_count = sum(1 for s in uv_map if slot_to_alliance(s) == "red")
        max_blue_per_frame = max(max_blue_per_frame, blue_count)
        max_red_per_frame = max(max_red_per_frame, red_count)

    print(f"  distinct slot ids seen: {sorted(seen_slots)}")
    print(f"  max simultaneous visible: blue={max_blue_per_frame} red={max_red_per_frame}")

    assert seen_slots <= set(range(6)), \
        f"INVARIANT FAILED: more than 6 slots seen! {seen_slots}"
    assert max_blue_per_frame <= 3 and max_red_per_frame <= 3, \
        (f"INVARIANT FAILED: exceeded 3 per alliance "
         f"(blue={max_blue_per_frame} red={max_red_per_frame})")

    print("  OK: fixed-6 / 3-per-alliance invariants held.")

    # ------------------------------------------------------------------
    # 7. Print continuity stats for both paths
    # ------------------------------------------------------------------
    trackid_stats = continuity_stats(trackid_frames)
    anon_stats = continuity_stats(anon_frames)

    trackid_distinct = len(seen_slots)
    anon_distinct = len(set().union(*[set(m.keys()) for m in anon_frames.values()])
                        if anon_frames else set())

    print("\n[driver] === CONTINUITY COMPARISON ===")
    print(f"  track_id path  : {trackid_stats}  distinct_slots={trackid_distinct}")
    print(f"  anonymous path : {anon_stats}  distinct_slots={anon_distinct}")
    print()
    print(f"  SAM2 latency: {ms_per_frame:.1f} ms/frame  "
          f"({n_prop} frames in {elapsed:.2f}s)")
    print()

    # Interpretation
    if trackid_stats["births"] <= anon_stats["births"]:
        print("  track_id path has FEWER or EQUAL births (better or same continuity).")
    else:
        print("  track_id path has MORE births than anonymous — "
              "check if SAM2 IDs are stable.")

    print("\n[driver] Done.")


if __name__ == "__main__":
    main()

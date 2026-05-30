"""SAM2-vs-FieldTracker comparison harness (Task 3 of the SAM2 tracking spike).

Runs SAM2 mask-propagation tracking AND the existing Kalman FieldTracker over
the same video segment, computes image-space identity-swap metrics, renders a
side-by-side comparison video, and prints latency.

Usage:
    .venv/bin/python scripts/sam2_spike.py [--labels PATH] [--out PATH]

Defaults:
    --labels  data/test_footage/qual1_first_champs_2026.spike_labels.json
    --out     data/test_footage/qual1_spike_compare.mp4
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
import numpy as np
import torch

from ultralytics import YOLO
from web.field import store as field_store
from web.field.field_model import FieldModel
from web.field.tracker import FieldTracker
from web.field.detect_bridge import to_field_detections
from web.app import MODEL_PATH, OVERHEAD_CROP_RATIO, _detect_bumper_color_static
from web.field.spike_metrics import count_id_switches, continuity_stats

VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
DEFAULT_LABELS = ROOT / "data/test_footage/qual1_first_champs_2026.spike_labels.json"

PROVISIONAL_SEGMENT = {
    "start_frame": 580,
    "end_frame": 820,
    "prompt_frame": 580,
    "stride": 4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _main_crop(frame: np.ndarray, ratio: float = OVERHEAD_CROP_RATIO) -> np.ndarray:
    h = frame.shape[0]
    return frame[: int(h * ratio), :]


def _box_bottom_center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2.0), float(y2)


def _mask_to_box(mask_2d: np.ndarray):
    """Return (x1, y1, x2, y2) bounding box of nonzero pixels in mask_2d, or None."""
    ys, xs = np.nonzero(mask_2d)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _draw_boxes(frame: np.ndarray, boxes_ids: list[tuple], color_bgr, thickness=2) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2), tid in boxes_ids:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, thickness)
        cv2.putText(out, str(tid), (int(x1), max(int(y1) - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
    return out


def _draw_masks(frame: np.ndarray, masks_ids: list[tuple]) -> np.ndarray:
    """Overlay semi-transparent masks; each (mask_2d, obj_id)."""
    out = frame.copy()
    palette = [
        (255, 50, 50), (50, 255, 50), (50, 50, 255),
        (255, 200, 50), (200, 50, 255), (50, 200, 200),
    ]
    for i, (mask_2d, oid) in enumerate(masks_ids):
        color = palette[i % len(palette)]
        colored = np.zeros_like(frame)
        colored[mask_2d] = color
        out = cv2.addWeighted(out, 1.0, colored, 0.45, 0)
        # draw bounding box + id
        box = _mask_to_box(mask_2d)
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"s{oid}", (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM2-vs-FieldTracker comparison harness")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Spike labels JSON sidecar")
    parser.add_argument("--out", default=str(ROOT / "data/test_footage/qual1_spike_compare.mp4"),
                        help="Output comparison video path")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load sidecar or provisional fallback
    # ------------------------------------------------------------------
    sidecar_path = Path(args.labels)
    provisional = False

    if sidecar_path.exists():
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        seg = sidecar["segment"]
        anchors_raw = sidecar.get("anchors", [])
        print(f"[spike] Loaded sidecar: {sidecar_path}")
        print(f"[spike] Segment: {seg}")
    else:
        provisional = True
        print("\n" + "=" * 70)
        print("WARNING: Spike-labels sidecar not found at:")
        print(f"  {sidecar_path}")
        print("Falling back to PROVISIONAL segment. Swap numbers are NOT ground-")
        print("truth anchored. They are provisional / for smoke-test purposes only.")
        print("=" * 70 + "\n")
        seg = PROVISIONAL_SEGMENT
        anchors_raw = []  # will be populated with dummy anchors after prompt detection

    start_frame = seg["start_frame"]
    end_frame = seg["end_frame"]
    prompt_frame = seg["prompt_frame"]
    stride = seg["stride"]

    if not VIDEO.exists():
        print(f"[spike] ERROR: video not found at {VIDEO}"); sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    print(f"[spike] Video: {VIDEO.name}  fps={fps:.1f}  segment={start_frame}..{end_frame} stride={stride}")

    # ------------------------------------------------------------------
    # 2. Load calibration and field model
    # ------------------------------------------------------------------
    cal = field_store.load("test", "main")
    if cal is None or cal.camera_center is None:
        print("[spike] No pose-bearing calibration for test/main — skipping"); sys.exit(0)

    fm = FieldModel.load(cal.field_season)
    dims = (fm.length_m, fm.width_m)
    robot_model = YOLO(str(MODEL_PATH))
    print(f"[spike] Calibration loaded. Field {fm.length_m:.2f}m x {fm.width_m:.2f}m")

    # ------------------------------------------------------------------
    # 3. Baseline pass: FieldTracker over the segment
    # ------------------------------------------------------------------
    print("[spike] === BASELINE PASS (FieldTracker) ===")
    trk = FieldTracker(fm.length_m)

    # per-frame accumulation for rendering (global_frame -> list of (box, slot_id))
    baseline_render: dict[int, list] = {}
    baseline_frames: dict[int, dict] = {}   # global_frame -> {slot_id: (u, v)}

    # We also need the main crop at prompt frame for SAM2 prompting later
    prompt_frame_crops: dict = {}   # will hold "yolo_boxes", "main_shape"

    all_frames = list(range(start_frame, end_frame + 1, stride))

    for gf in all_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ok, fr = cap.read()
        if not ok:
            print(f"[spike] WARNING: could not read frame {gf}")
            continue
        main = _main_crop(fr)
        res = robot_model(main, conf=0.4, verbose=False)[0]
        rboxes = [res.boxes.xyxy[i].cpu().numpy() for i in range(len(res.boxes))
                  if int(res.boxes.cls[i]) == 1]

        dets = to_field_detections(rboxes, cal, main, dims,
                                   color_fn=_detect_bumper_color_static)
        outs = trk.update(dets, {}, dt=stride / fps)

        bline_uv = {}
        render_boxes = []
        for t in outs:
            if t.box is not None and t.state == "visible":
                bc = _box_bottom_center(t.box)
                bline_uv[t.slot_id] = bc
                render_boxes.append((t.box, t.slot_id))

        baseline_frames[gf] = bline_uv
        baseline_render[gf] = render_boxes

        if gf == prompt_frame:
            prompt_frame_crops["rboxes"] = rboxes
            prompt_frame_crops["main"] = main.copy()

    print(f"[spike] Baseline: processed {len(all_frames)} frames, "
          f"{len(baseline_frames)} with data")

    # ------------------------------------------------------------------
    # Build provisional dummy anchors if no sidecar
    # ------------------------------------------------------------------
    if provisional:
        # Use the YOLO boxes from the prompt frame as dummy anchor positions;
        # assign them label "robot_A", "robot_B", etc. so count_id_switches
        # has something to work with across the segment.
        pf_rboxes = prompt_frame_crops.get("rboxes", [])
        if pf_rboxes:
            anchors_raw = []
            anchor_frames = [f for f in all_frames[::max(1, len(all_frames) // 6)]]
            for ridx, box in enumerate(pf_rboxes[:3]):
                u, v = _box_bottom_center(box)
                for af in anchor_frames:
                    anchors_raw.append({
                        "frame": af,
                        "uv": [float(u), float(v)],
                        "label": f"robot_{chr(65+ridx)}",
                    })
            print(f"[spike] Built {len(anchors_raw)} provisional dummy anchors "
                  f"from {len(pf_rboxes[:3])} prompt-frame robots.")
        else:
            print("[spike] WARNING: No YOLO detections at prompt frame — "
                  "anchors list will be empty; swap count = 0.")

    # Convert anchors to (frame, (u, v), label) tuples
    anchors = [(int(a["frame"]), tuple(a["uv"]), a["label"]) for a in anchors_raw]

    # ------------------------------------------------------------------
    # 4. SAM2 pass: export frames to temp dir, propagate masks
    # ------------------------------------------------------------------
    print("[spike] === SAM2 PASS ===")

    prompt_local_idx = (prompt_frame - start_frame) // stride

    sam2_frames: dict[int, dict] = {}    # global_frame -> {obj_id: (u, v)}
    sam2_render: dict[int, list] = {}    # global_frame -> list of (mask_2d, obj_id)
    sam2_frame_shape = None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export main-crop frames as zero-padded JPEGs.
        # Resize to half width to reduce SAM2 memory usage (it internally
        # resizes anyway; smaller input = faster + less RAM).
        SAM2_SCALE = 0.5   # scale factor applied to exported frames
        print(f"[spike] Exporting {len(all_frames)} frames to temp dir "
              f"(scale={SAM2_SCALE}) ...")
        local_to_global: dict[int, int] = {}
        export_scale = SAM2_SCALE  # keep reference for box rescaling later
        for local_idx, gf in enumerate(all_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
            ok, fr = cap.read()
            if not ok:
                continue
            main = _main_crop(fr)
            if SAM2_SCALE != 1.0:
                new_w = int(main.shape[1] * SAM2_SCALE)
                new_h = int(main.shape[0] * SAM2_SCALE)
                main_small = cv2.resize(main, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                main_small = main
            fname = f"{local_idx:05d}.jpg"
            cv2.imwrite(os.path.join(tmpdir, fname), main_small, [cv2.IMWRITE_JPEG_QUALITY, 95])
            local_to_global[local_idx] = gf
            if sam2_frame_shape is None:
                sam2_frame_shape = main.shape[:2]  # full-res (H, W) for rendering

        print(f"[spike] Exported. Frame shape: {sam2_frame_shape}")

        # Build SAM2 predictor.
        # Prefer CUDA; fall back to CPU if insufficient GPU memory (e.g. another
        # process is saturating the GPU — common during development).
        from sam2.build_sam import build_sam2_video_predictor
        _free_mib = 0
        try:
            _free_mib = (torch.cuda.mem_get_info()[0] // 1024 // 1024
                         if torch.cuda.is_available() else 0)
        except Exception:
            pass
        _SAM2_MIN_GPU_MIB = 2500   # SAM2 large model needs ~2GB+ headroom
        if _free_mib >= _SAM2_MIN_GPU_MIB:
            sam2_device = "cuda"
            print(f"[spike] CUDA free: {_free_mib} MiB — using GPU")
        else:
            sam2_device = "cpu"
            print(f"[spike] CUDA free: {_free_mib} MiB (< {_SAM2_MIN_GPU_MIB} MiB threshold) "
                  f"— falling back to CPU for SAM2")

        print(f"[spike] Loading SAM2 model on {sam2_device} ...")
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(ROOT / "checkpoints/sam2.1_hiera_large.pt"),
            device=sam2_device,
        )

        # autocast only makes sense on CUDA
        _autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if sam2_device == "cuda"
            else torch.autocast("cpu", dtype=torch.bfloat16)
        )
        with torch.inference_mode(), _autocast_ctx:
            print("[spike] Initialising SAM2 video state ...")
            _offload = sam2_device == "cuda"  # on CPU everything is already on CPU
            state = predictor.init_state(
                video_path=tmpdir,
                offload_video_to_cpu=_offload,
                offload_state_to_cpu=_offload,
            )

            # Prompt: add each YOLO robot box from the prompt frame.
            # Boxes are in full-res coords; scale to match the exported SAM2 frames.
            pf_rboxes = prompt_frame_crops.get("rboxes", [])
            if not pf_rboxes:
                print("[spike] WARNING: no robot boxes at prompt frame — "
                      "SAM2 has nothing to track; masks will be empty.")

            for oid, box in enumerate(pf_rboxes):
                x1, y1, x2, y2 = box
                xyxy = np.array(
                    [x1 * export_scale, y1 * export_scale,
                     x2 * export_scale, y2 * export_scale],
                    dtype=np.float32,
                )
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=prompt_local_idx,
                    obj_id=oid,
                    box=xyxy,
                )
            print(f"[spike] Prompted SAM2 with {len(pf_rboxes)} robots on "
                  f"local frame {prompt_local_idx} (global {prompt_frame}).")

            # Propagate
            print("[spike] Propagating ...")
            t0 = time.perf_counter()
            n_prop_frames = 0
            for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                gf = local_to_global.get(f_idx)
                if gf is None:
                    continue
                # mask_logits shape: (num_obj, 1, H, W) typically
                masks = (mask_logits > 0.0).cpu().numpy()

                uv_map = {}
                render_masks = []
                for i, oid in enumerate(obj_ids):
                    # masks[i] might be (1, H, W) or (H, W)
                    if masks.ndim == 4:
                        m2d = masks[i, 0]
                    else:
                        m2d = masks[i]
                    m2d = m2d.astype(bool)

                    box = _mask_to_box(m2d)
                    if box is not None:
                        # Scale box coords back to full-res image space
                        sx1, sy1, sx2, sy2 = box
                        fx1 = sx1 / export_scale
                        fy1 = sy1 / export_scale
                        fx2 = sx2 / export_scale
                        fy2 = sy2 / export_scale
                        uv_map[int(oid)] = _box_bottom_center((fx1, fy1, fx2, fy2))
                        # For rendering, scale the mask up to full-res
                        if sam2_frame_shape is not None:
                            fh, fw = sam2_frame_shape
                            m2d_full = cv2.resize(
                                m2d.astype(np.uint8), (fw, fh),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        else:
                            m2d_full = m2d
                        render_masks.append((m2d_full, int(oid)))

                sam2_frames[gf] = uv_map
                sam2_render[gf] = render_masks
                n_prop_frames += 1

            elapsed = time.perf_counter() - t0

    # Compute latency
    sam2_ms_per_frame = (elapsed / n_prop_frames * 1000.0) if n_prop_frames > 0 else 0.0
    print(f"[spike] SAM2 propagated {n_prop_frames} frames in {elapsed:.2f}s "
          f"({sam2_ms_per_frame:.1f} ms/frame)")

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    print("\n[spike] === METRICS ===")
    bl_swaps = count_id_switches(baseline_frames, anchors)
    bl_stats = continuity_stats(baseline_frames)
    s2_swaps = count_id_switches(sam2_frames, anchors)
    s2_stats = continuity_stats(sam2_frames)

    print(f"FieldTracker swaps: {bl_swaps}  {bl_stats}")
    print(f"SAM2 swaps:         {s2_swaps}  {s2_stats}")
    print(f"SAM2 latency ms/frame: {sam2_ms_per_frame:.1f}")

    if provisional:
        print("\n*** PROVISIONAL RUN — swap numbers have NO ground-truth anchors. ***")
        print("*** Author data/test_footage/qual1_first_champs_2026.spike_labels.json ***")
        print("*** (Task 2) and re-run for the real gate verdict.                  ***")

    # ------------------------------------------------------------------
    # 6. Render side-by-side comparison video
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[spike] Rendering comparison video -> {out_path} ...")

    # Determine output frame size from the first main crop
    if sam2_frame_shape is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, fr = cap.read()
        main_h, main_w = _main_crop(fr).shape[:2] if ok else (720, 1920)
    else:
        main_h, main_w = sam2_frame_shape

    out_w = main_w * 2  # side by side
    out_h = main_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    render_fps = fps / stride
    vw = cv2.VideoWriter(str(out_path), fourcc, render_fps, (out_w, out_h))

    for gf in all_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ok, fr = cap.read()
        if not ok:
            continue
        main = _main_crop(fr)

        # Left panel: FieldTracker boxes
        left = _draw_boxes(main, baseline_render.get(gf, []), color_bgr=(0, 200, 0))
        label_left = "FieldTracker"

        # Right panel: SAM2 masks
        right = _draw_masks(main, sam2_render.get(gf, []))
        label_right = "SAM2"

        # Add frame number + label overlay
        for panel, label in [(left, label_left), (right, label_right)]:
            cv2.putText(panel, f"{label}  f={gf}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, f"{label}  f={gf}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        combined = np.concatenate([left, right], axis=1)
        vw.write(combined)

    vw.release()
    cap.release()

    print(f"[spike] Video written: {out_path}")
    print(f"[spike] Done.")


if __name__ == "__main__":
    main()

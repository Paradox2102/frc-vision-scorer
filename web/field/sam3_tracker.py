"""SAM 3 adapter for the existing offline field-tracker pipeline.

Targets Meta's full SAM3 video model (detector + tracker,
``Sam3VideoInferenceWithInstanceInteractivity``) through its session/request
API, while keeping the lower-level SAM2 video-predictor protocol the
application is built around, so re-prompting, culling, fusion, and identity
logic do not need separate SAM3 implementations.

Protocol mapping (validated against sam3 @ pinned commit, June 2026):

- Box prompts ride the *instance* path, not the semantic path. The semantic
  ``add_prompt(boxes_xywh=...)`` resets all tracking state on every call,
  which is incompatible with one-call-per-robot prompting and mid-stream
  re-prompts. Instead, each box is encoded as two points with labels 2
  (top-left) and 3 (bottom-right) — the same encoding SAM 2 was trained on —
  and sent with a stable caller ``obj_id``. That routes to
  ``add_tracker_new_points`` -> the SAM3 tracker's ``add_new_points_or_box``,
  which adds new objects or refines existing ones incrementally.
- Point coordinates are sent normalized (``rel_coordinates=True``); the model
  scales them by its square input size internally.

Chunked sessions (VRAM)
-----------------------

The SAM3 video model copies the whole session's frames to the GPU at 1008²
(``offload_video_to_cpu`` is ignored by its input-batch construction), so a
full match cannot be one session on a 24GB card. The adapter therefore opens
one session per window of ``SAM3_CHUNK_FRAMES`` frames (measured: ~14 MB
VRAM/frame + ~4 GB model; 360-frame chunk peaks at ~8.8 GB) and carries
objects across boundaries by re-seeding each live object's last observed
mask bbox under its SAME obj_id at the start of the next chunk.

What a boundary loses: the tracker's appearance memory, so an object whose
mask is empty at the boundary (occluded) is not re-seeded and will not be
re-found by SAM3 itself. The pipeline's existing detector re-prompts and
identity machinery (VLM votes + gallery) already handle exactly that case
for SAM2 re-seeds.

Windowed propagation (action history)
-------------------------------------

Unlike SAM2, a SAM3 session interprets each ``propagate_in_video`` request
through its action history (``parse_action_history_for_propagation``): with
no new prompts since the previous propagate it returns cached outputs only
(``propagation_fetch``), and after new prompts it tracks ONLY the newly
prompted obj_ids (``propagation_partial``). The pipeline propagates in
``reprompt_every`` windows, so without intervention every object emits masks
for exactly one window and then goes dark. The adapter therefore re-sends
each live object's latest observed box (fresh within SAM3_RESEED_MAX_AGE)
before every windowed propagate, keeping the full roster on the partial
path; per-object tracker states persist within the session, so this is a
re-anchor, not a cold re-seed.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

import numpy as np

from web.field.sam2_tracker import Sam2FieldTracker

# Frames per SAM3 session window. ~14 MB GPU/frame: 200 frames ≈ 2.8 GB video
# residency, leaving headroom for the model and a co-resident vLLM reader.
# Lowered from 400 after a 24GB-card whole-card OOM with co-resident state.
SAM3_CHUNK_FRAMES = int(os.environ.get("SAM3_CHUNK_FRAMES", "200"))
# Only re-seed an object across a boundary if its mask was non-empty within
# this many recent frames (stale boxes hijack background clutter).
SAM3_RESEED_MAX_AGE = int(os.environ.get("SAM3_RESEED_MAX_AGE", "24"))


def _list_frame_paths(frames_dir: str) -> list[str]:
    names = [n for n in sorted(os.listdir(frames_dir))
             if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not names:
        raise RuntimeError(f"no frames in {frames_dir}")
    return [os.path.join(frames_dir, n) for n in names]


def _probe_image_size(path: str) -> tuple[int, int]:
    import cv2

    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"unreadable frame {path}")
    h, w = frame.shape[:2]
    return w, h


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """xyxy bbox of a binary mask, or None when empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


class Sam3PredictorAdapter:
    """Expose a SAM 3 session predictor through the SAM2 predictor protocol,
    windowing the video into VRAM-sized chunk sessions internally."""

    def __init__(self, predictor: Any, chunk_frames: int = SAM3_CHUNK_FRAMES) -> None:
        self._predictor = predictor
        self._chunk_frames = max(int(chunk_frames), 16)

    # ── SAM2-protocol surface ────────────────────────────────────────

    def init_state(
        self,
        *,
        video_path: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ) -> dict:
        if not os.path.isdir(video_path):
            raise RuntimeError(
                "Sam3PredictorAdapter requires a frames directory (the "
                "pipeline's export dir); got a file: " + video_path
            )
        frame_paths = _list_frame_paths(video_path)
        width, height = _probe_image_size(frame_paths[0])
        return {
            "frame_paths": frame_paths,
            "video_size": (width, height),
            "offload": (offload_video_to_cpu, offload_state_to_cpu),
            # live chunk session: {"session_id", "lo", "hi", "dir"} or None
            "chunk": None,
            # obj_id -> (global_frame_idx, xyxy box) most recent evidence;
            # prompts and observed masks both update it
            "last_box": {},
            # prompts not yet sent to any session: [(global_idx, obj_id, box)]
            "pending": [],
            "closed": False,
        }

    def add_new_points_or_box(
        self,
        state: dict,
        *,
        frame_idx: int,
        obj_id: int,
        box,
    ) -> None:
        box = tuple(float(v) for v in box)
        state["last_box"][int(obj_id)] = (int(frame_idx), box)
        chunk = state["chunk"]
        if chunk is not None and chunk["lo"] <= frame_idx < chunk["hi"]:
            self._send_prompt(state, frame_idx, obj_id, box)
        else:
            state["pending"].append((int(frame_idx), int(obj_id), box))

    def propagate_in_video(
        self,
        state: dict,
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        num_frames = len(state["frame_paths"])
        start = int(start_frame_idx) if start_frame_idx is not None else (
            num_frames - 1 if reverse else 0
        )
        span = int(max_frame_num_to_track) if max_frame_num_to_track is not None \
            else num_frames
        if reverse:
            lo, hi = max(0, start - span), start + 1
        else:
            lo, hi = start, min(num_frames, start + span + 1)

        frames = range(hi - 1, lo - 1, -1) if reverse else range(lo, hi)
        cursor = None
        for global_idx in frames:
            chunk = state["chunk"]
            if chunk is None or not (chunk["lo"] <= global_idx < chunk["hi"]):
                self._open_chunk_for(state, global_idx, reverse=reverse)
                cursor = None
            if cursor is None or cursor != global_idx:
                # (re)start streaming within the current chunk from global_idx
                yield from self._stream_chunk(
                    state, global_idx, lo, hi, reverse=reverse
                )
                return  # _stream_chunk recurses across remaining chunks
        # zero-length range: nothing to do

    def reset_state(self, state: dict) -> None:
        if state.get("closed"):
            return
        try:
            self._close_chunk(state)
        finally:
            state["closed"] = True

    # ── chunk-session internals ──────────────────────────────────────

    def _send_prompt(self, state: dict, frame_idx: int, obj_id: int, box) -> None:
        chunk = state["chunk"]
        width, height = state["video_size"]
        x1, y1, x2, y2 = box
        # Box -> two corner points with SAM2's box labels (2: TL, 3: BR), in
        # normalized coordinates: the instance path keeps stable obj_ids and
        # adds/refines incrementally instead of resetting state.
        self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": chunk["session_id"],
                "frame_index": int(frame_idx) - chunk["lo"],
                "obj_id": int(obj_id),
                "points": [[x1 / width, y1 / height], [x2 / width, y2 / height]],
                "point_labels": [2, 3],
                "rel_coordinates": True,
            }
        )
        chunk["prompted"].add(int(obj_id))

    def _open_chunk_for(self, state: dict, global_idx: int, *, reverse: bool) -> None:
        self._close_chunk(state)
        num_frames = len(state["frame_paths"])
        if reverse:
            hi = min(num_frames, global_idx + 1)
            lo = max(0, hi - self._chunk_frames)
        else:
            lo = global_idx
            hi = min(num_frames, lo + self._chunk_frames)

        # SAM3 sessions open a resource path; hand it a directory of symlinks
        # covering just this window.
        tmp_dir = tempfile.mkdtemp(prefix="sam3_chunk_")
        for i, src in enumerate(state["frame_paths"][lo:hi]):
            os.symlink(os.path.abspath(src), os.path.join(tmp_dir, f"{i:08d}.jpg"))

        offload_video, offload_state = state["offload"]
        response = self._predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": tmp_dir,
                "offload_video_to_cpu": offload_video,
                "offload_state_to_cpu": offload_state,
            }
        )
        state["chunk"] = {
            "session_id": response["session_id"],
            "lo": lo,
            "hi": hi,
            "dir": tmp_dir,
            # obj_ids prompted since the last propagate request in this
            # session; consumed (cleared) by each _stream_chunk call.
            "prompted": set(),
        }

        # Seed: pending prompts that fall in this window, then carried
        # objects (fresh mask evidence only) that have no pending prompt here.
        seed_frame = (hi - 1) if reverse else lo
        pending_here = [(f, o, b) for (f, o, b) in state["pending"]
                        if lo <= f < hi]
        state["pending"] = [(f, o, b) for (f, o, b) in state["pending"]
                            if not (lo <= f < hi)]
        prompted_ids = set()
        for f, o, b in pending_here:
            self._send_prompt(state, f, o, b)
            prompted_ids.add(o)
        for obj_id, (seen_idx, box) in sorted(state["last_box"].items()):
            if obj_id in prompted_ids:
                continue
            if abs(seed_frame - seen_idx) > SAM3_RESEED_MAX_AGE:
                continue  # stale: let detector re-prompts recover it
            self._send_prompt(state, seed_frame, obj_id, box)

    def _close_chunk(self, state: dict) -> None:
        chunk = state.get("chunk")
        if not chunk:
            return
        try:
            self._predictor.handle_request(
                {"type": "close_session", "session_id": chunk["session_id"]}
            )
        finally:
            shutil.rmtree(chunk["dir"], ignore_errors=True)
            state["chunk"] = None
            # VRAM forensics for the recurring whole-card OOM: if allocated
            # memory climbs across chunk closes, session state is not being
            # released (close_session leak / lingering references); if it is
            # flat, the growth is intra-chunk (prompt accumulation).
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 2**30
                    reserved = torch.cuda.memory_reserved() / 2**30
                    print(f"  [SAM3] chunk closed [{chunk['lo']},{chunk['hi']}) "
                          f"VRAM allocated={alloc:.2f}GiB reserved={reserved:.2f}GiB")
            except Exception:
                pass

    def _stream_chunk(self, state: dict, start_global: int, lo: int, hi: int,
                      *, reverse: bool):
        """Stream one chunk from start_global, then recurse into the next
        chunk until the caller's [lo, hi) range is exhausted."""
        import torch

        chunk = state["chunk"]

        # SAM3 sessions route each propagate request through their action
        # history: objects WITHOUT an add/refine since the previous propagate
        # are not tracked at all (fetch/partial-only semantics, see
        # parse_action_history_for_propagation). Refresh every live object's
        # latest observed box before each window so the whole roster rides
        # ``propagation_partial``; stale objects stay with the detector
        # re-prompt recovery path, same rule as the chunk-boundary re-seed.
        for obj_id, (seen_idx, box) in sorted(state["last_box"].items()):
            if obj_id in chunk["prompted"]:
                continue
            if not (chunk["lo"] <= seen_idx < chunk["hi"]):
                continue
            if abs(start_global - seen_idx) > SAM3_RESEED_MAX_AGE:
                continue
            self._send_prompt(state, seen_idx, obj_id, box)
        chunk["prompted"] = set()

        local_start = start_global - chunk["lo"]
        # frames of this chunk that fall inside the caller's range
        chunk_lo = max(lo, chunk["lo"])
        chunk_hi = min(hi, chunk["hi"])
        span = (start_global - chunk_lo) if reverse else (chunk_hi - 1 - start_global)

        request = {
            "type": "propagate_in_video",
            "session_id": chunk["session_id"],
            "propagation_direction": "backward" if reverse else "forward",
            "start_frame_index": int(local_start),
            "max_frame_num_to_track": int(max(span, 0)),
        }
        last_global = start_global
        for response in self._predictor.handle_stream_request(request):
            outputs = response.get("outputs") or {}
            raw_obj_ids = outputs.get("out_obj_ids", [])
            raw_masks = outputs.get("out_binary_masks", [])
            raw_probs = outputs.get("out_tracker_probs")
            if hasattr(raw_obj_ids, "detach"):
                raw_obj_ids = raw_obj_ids.detach().cpu().numpy()
            if hasattr(raw_masks, "detach"):
                raw_masks = raw_masks.detach().cpu().numpy()
            obj_ids = np.asarray(raw_obj_ids, dtype=np.int64)
            masks = np.asarray(raw_masks)
            if masks.ndim == 2:
                masks = masks[None, ...]

            # Per-object tracker confidence, aligned with obj_ids. Absent on
            # older models / stubs; flatten (N,1)->(N,); drop it defensively
            # if it doesn't line up rather than mis-attaching scores.
            probs = None
            if raw_probs is not None:
                if hasattr(raw_probs, "detach"):
                    raw_probs = raw_probs.detach().cpu().numpy()
                probs = np.asarray(raw_probs, dtype=np.float64).reshape(-1)
                if probs.shape[0] != obj_ids.shape[0]:
                    probs = None
            global_idx = int(response["frame_index"]) + chunk["lo"]
            last_global = global_idx

            # record per-object boxes so the next chunk can re-seed
            for i, oid in enumerate(obj_ids.tolist()):
                bbox = _mask_bbox(masks[i, 0] if masks.ndim == 4 else masks[i])
                if bbox is not None:
                    state["last_box"][int(oid)] = (global_idx, bbox)

            yield (
                global_idx,
                obj_ids,
                torch.as_tensor(masks, dtype=torch.float32),
                probs,
            )

        # continue into the next chunk if the caller's range extends past it
        if reverse and last_global > lo:
            next_global = last_global - 1
            self._open_chunk_for(state, next_global, reverse=True)
            yield from self._stream_chunk(state, next_global, lo, hi, reverse=True)
        elif not reverse and last_global < hi - 1:
            next_global = last_global + 1
            self._open_chunk_for(state, next_global, reverse=False)
            yield from self._stream_chunk(state, next_global, lo, hi, reverse=False)


class Sam3FieldTracker(Sam2FieldTracker):
    """Run the field tracker with Meta's SAM3 video model (instance prompts,
    chunked sessions).

    ``track(..., should_stop=...)`` (cooperative user-STOP abort) is inherited
    from ``Sam2FieldTracker`` and works here too: the shared loop polls the
    predicate once per propagated frame of the adapter's stream and raises
    ``TrackStopped``; the ``finally`` cleanup calls this adapter's
    ``reset_state``, which closes the live chunk session and removes its
    temp dir.
    """

    LOG_TAG = "SAM3"
    DIAG_PREFIX = "sam3"

    def __init__(
        self,
        predictor: Any | None = None,
        *,
        checkpoint: str | None = None,
        device: str = "cuda",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        compile_model: bool = False,
        max_objects: int = 20,
        use_fa3: bool = True,  # accepted for call-site compat; multiplex-only
        chunk_frames: int | None = None,
    ) -> None:
        if device != "cuda":
            raise RuntimeError("SAM 3 video model currently requires CUDA")

        if predictor is None:
            from sam3.model_builder import build_sam3_video_predictor

            checkpoint = checkpoint or os.environ.get("SAM3_CHECKPOINT") or None
            predictor = build_sam3_video_predictor(
                checkpoint_path=checkpoint,
                compile=compile_model,
            )
            # Instance-tracking capacity is a model attribute, not a builder
            # arg; raise it so re-prompting can exceed the demo default.
            model = getattr(predictor, "model", None)
            if model is not None and hasattr(model, "max_num_objects"):
                model.max_num_objects = max(
                    int(getattr(model, "max_num_objects", 0) or 0), int(max_objects)
                )

        super().__init__(
            predictor=Sam3PredictorAdapter(
                predictor,
                chunk_frames=chunk_frames if chunk_frames else SAM3_CHUNK_FRAMES,
            ),
            device=device,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        # The adapter returns real torch tensors and benefits from inference mode.
        self._real_torch = True

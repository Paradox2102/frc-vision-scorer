"""SAM2 offline video-predictor tracker helpers and wrapper class.

The module is split into two tiers:

1. Pure, GPU-free helpers (``mask_to_box``, ``bottom_center``, ``_ios``,
   ``match_new_detections``) — importable with no torch/sam2 on the path.

2. ``Sam2FieldTracker`` — a stateful wrapper around SAM2's offline video
   predictor.  torch/sam2 are imported *lazily* inside the class so that
   the pure helpers remain torch-free.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight bounding box of nonzero pixels as (x1, y1, x2, y2).

    x2 and y2 are exclusive maxima (one past the last pixel), consistent with
    Python slice notation.  Returns None if the mask contains no True pixels.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bottom_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the bottom-center point of a box as (cx, y_bottom).

    y2 is already the exclusive-max bottom from mask_to_box, so this is
    consistent with that convention.
    """
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2, float(y2))


def _ios(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Smaller-area (IoS) between two boolean masks.

    Returns 0.0 if either mask is empty.
    """
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    smaller = min(area_a, area_b)
    if smaller == 0:
        return 0.0
    intersection = int((mask_a & mask_b).sum())
    return intersection / smaller


def match_new_detections(
    existing_masks: dict[int, np.ndarray],
    new_boxes: list[tuple[int, int, int, int]],
    wh: tuple[int, int],
    ios_thresh: float = 0.5,
) -> list[int]:
    """Return indices of new_boxes that are genuinely new robots (not already tracked).

    For each box in new_boxes the box is rasterised to a boolean mask of shape
    (height, width) and the max IoS against every existing mask is computed.
    Boxes whose best IoS is below ios_thresh are considered new and their
    indices (preserving input order) are returned.

    Args:
        existing_masks: mapping of object-id -> bool ndarray (h, w).
        new_boxes: list of (x1, y1, x2, y2) detections to test.
        wh: (width, height) of the frame.
        ios_thresh: minimum IoS to consider a detection already tracked.

    Returns:
        List of integer indices into new_boxes for unmatched (new) robots.
    """
    w, h = wh
    unmatched: list[int] = []
    for idx, (x1, y1, x2, y2) in enumerate(new_boxes):
        box_mask = np.zeros((h, w), bool)
        box_mask[y1:y2, x1:x2] = True
        best_ios = 0.0
        for existing in existing_masks.values():
            score = _ios(existing, box_mask)
            if score > best_ios:
                best_ios = score
        if best_ios < ios_thresh:
            unmatched.append(idx)
    return unmatched


# ---------------------------------------------------------------------------
# Stateful SAM2 wrapper
# ---------------------------------------------------------------------------

#: Default SAM2 config (relative to the project root, used by SAM2 itself).
_DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
#: Default SAM2 checkpoint path.
_DEFAULT_SAM2_CKPT = "checkpoints/sam2.1_hiera_large.pt"


@contextmanager
def _maybe_autocast(device: str):
    """Enter a torch.autocast context only when torch is available.

    Falls back to a null context-manager when torch is absent (stub path).
    """
    try:
        import torch
        ctx = torch.autocast(device if device.startswith("cuda") else "cpu",
                             dtype=torch.bfloat16)
        with ctx:
            yield
    except ImportError:
        # torch not installed — stub / pure-Python path; no-op
        yield


class Sam2FieldTracker:
    """Offline SAM2 mask-propagation tracker.

    Wraps SAM2's ``build_sam2_video_predictor`` flow:

    1. ``init_state`` — point the predictor at a directory of numbered JPEG
       frames.
    2. ``add_new_points_or_box`` — provide a bounding-box prompt per robot on
       the prompt frame.
    3. ``propagate_in_video`` — obtain per-frame masks for every prompted
       object.

    The class returns a nested dict keyed by ``{frame_idx: {obj_id: {"box":
    (x1,y1,x2,y2), "mask": ndarray}}}``.

    Dependency injection
    --------------------
    Pass a *predictor* instance directly to skip the SAM2 build step.  This is
    used for unit tests with a stub predictor and avoids requiring a GPU.

    Mid-stream reprompting
    ----------------------
    The ``reprompt_every`` and ``detect_fn`` parameters are accepted as a hook
    for future occlusion-recovery reprompting but are **not yet implemented**.
    Single-prompt offline propagation is the only supported mode.
    """

    def __init__(
        self,
        predictor: Any | None = None,
        *,
        config: str = _DEFAULT_SAM2_CONFIG,
        checkpoint: str = _DEFAULT_SAM2_CKPT,
        device: str = "cuda",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        vos_optimized: bool = False,
    ) -> None:
        """Initialise the tracker.

        Args:
            predictor: An already-built SAM2 predictor (or stub).  When
                ``None``, ``build_sam2_video_predictor`` is called lazily.
            config: SAM2 model config file (passed to
                ``build_sam2_video_predictor``).
            checkpoint: Path to the SAM2 checkpoint file.
            device: torch device string (``"cuda"`` or ``"cpu"``).
            offload_video_to_cpu: Offload video frames to CPU to save GPU VRAM.
            offload_state_to_cpu: Offload tracker state to CPU to save GPU VRAM.
            vos_optimized: Build the ``SAM2VideoPredictorVOS`` variant with a
                ``torch.compile``d image encoder.  The image encoder is the
                dominant per-frame cost, so compiling it gives a large speedup
                (~2x) at the price of a one-time warmup on the first chunk.
                Only meaningful on CUDA.
        """
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu

        if predictor is not None:
            self._predictor = predictor
            self._device = device
            self._real_torch = False  # stub path — skip autocast
        else:
            # Lazy import: only touch torch/sam2 when actually building a real
            # predictor.  This keeps `from web.field.sam2_tracker import
            # mask_to_box` fast and torch-free.
            import torch  # noqa: F401 — imported for side-effects (device check)
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]

            # SAM2's connected-components CUDA kernel (`sam2._C`) is an optional
            # native extension our pip install doesn't compile.  Without it,
            # SAM2 skips mask hole-filling (fill_hole_area=8) and warns once per
            # frame.  We track robots by bounding box (mask_to_box uses the
            # mask's outer extent), so filling ≤8px interior holes can't change
            # any result — silence just this message to keep analysis logs clean.
            import warnings

            warnings.filterwarnings(
                "ignore",
                message=r"(?s).*Skipping the post-processing step due to the error above.*",
                category=UserWarning,
            )

            self._predictor = build_sam2_video_predictor(
                config, checkpoint, device=device, vos_optimized=vos_optimized
            )
            self._device = device
            self._real_torch = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(
        self,
        frames_dir: str,
        prompt_idx: int,
        prompt_boxes: list[tuple],
        *,
        reprompt_every: int | None = None,
        detect_fn=None,
        object_metadata=None,
        max_objects: int | None = None,
        new_obj_ios_thresh: float = 0.5,
        progress_cb=None,
    ) -> dict[int, dict[int, dict]]:
        """Run offline SAM2 propagation over a directory of frames.

        Args:
            frames_dir: Path to a directory of zero-padded JPEG frames (e.g.
                ``00000.jpg``, ``00001.jpg`` …).  Passed verbatim to
                ``predictor.init_state``.
            prompt_idx: Frame index (local, 0-based) at which box prompts are
                given.
            prompt_boxes: Sequence of ``(x1, y1, x2, y2)`` boxes (float or
                int) defining one robot per entry.  Object IDs are assigned as
                ``1, 2, 3, …`` in list order.
            reprompt_every: When set together with ``detect_fn``, propagation is
                run in chunks of this many frames; at each chunk boundary
                ``detect_fn`` is run and any detection that does not already
                overlap a tracked object is added as a *new* object and tracked
                forward.  ``None`` (the default) runs a single forward pass —
                the original single-prompt behavior.
            detect_fn: Callable returning boxes or ``(box, metadata)`` pairs.
            object_metadata: Optional mutable ``{obj_id: metadata}`` map. Initial
                prompt metadata may be pre-populated; re-prompted objects are added.
            max_objects: Hard cap on the total number of tracked objects.  Once
                reached, re-prompting stops adding new objects (prevents false
                detections from spawning phantom tracks).  ``None`` = no cap.
            new_obj_ios_thresh: A detection whose best intersection-over-smaller
                against any tracked mask is below this threshold is treated as a
                new robot.  Passed to :func:`match_new_detections`.
            progress_cb: Optional callback taking (processed_count, total_frames).

        Returns:
            Nested dict ``{frame_idx: {obj_id: {"box": (x1,y1,x2,y2),
            "mask": ndarray_2d}}}``.  Objects with empty masks on a given
            frame are omitted from that frame's dict.
        """
        import os
        if os.path.isdir(frames_dir):
            frame_files = sorted(
                f for f in os.listdir(frames_dir)
                if f.endswith(".jpg") or f.endswith(".jpeg")
            )
            frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
        else:
            frame_paths = []
        total_frames = max(1, len(frame_paths))

        predictor = self._predictor

        state = predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
        )

        # Prompt each robot box on the prompt frame.  Object IDs are 1-based.
        next_obj_id = 1
        for box in prompt_boxes:
            predictor.add_new_points_or_box(
                state,
                frame_idx=prompt_idx,
                obj_id=next_obj_id,
                box=np.asarray(box, dtype=np.float32),
            )
            next_obj_id += 1

        results: dict[int, dict[int, dict]] = {}
        self._processed_count = 0

        reprompting = bool(reprompt_every) and detect_fn is not None

        def _propagate(start_frame_idx=None, max_frame_num_to_track=None):
            # Pass chunk kwargs only when re-prompting, so single-pass callers
            # (and the simple test stub) keep the original zero-arg signature.
            if reprompting:
                kwargs = {
                    "start_frame_idx": start_frame_idx,
                    "max_frame_num_to_track": max_frame_num_to_track,
                }
            else:
                kwargs = {}
            if self._real_torch:
                import torch
                with torch.inference_mode(), _maybe_autocast(self._device):
                    yield from predictor.propagate_in_video(state, **kwargs)
            else:
                yield from predictor.propagate_in_video(state, **kwargs)

        def _record_chunk(start_frame_idx, max_frame_num_to_track):
            """Propagate one chunk; record masks; return the last frame index seen."""
            last_idx = None
            for f_idx, obj_ids, mask_logits in _propagate(
                start_frame_idx, max_frame_num_to_track
            ):
                # mask_logits may be a real torch tensor (N,1,H,W) or a stub.
                bool_arr = (mask_logits > 0).cpu().numpy()
                frame_entry: dict[int, dict] = {}
                for i, obj_id in enumerate(obj_ids):
                    if bool_arr.ndim == 4:
                        mask2d = bool_arr[i, 0].astype(bool)
                    elif bool_arr.ndim == 3:
                        mask2d = bool_arr[i].astype(bool)
                    else:
                        mask2d = bool_arr.astype(bool)
                    box = mask_to_box(mask2d)
                    if box is None:
                        continue  # object not visible this frame
                    frame_entry[int(obj_id)] = {"box": box, "mask": mask2d}
                results[int(f_idx)] = frame_entry
                last_idx = int(f_idx)
                self._processed_count += 1
                if progress_cb is not None:
                    try:
                        progress_cb(self._processed_count, total_frames)
                    except Exception:
                        pass
            return last_idx

        try:
            if not reprompting:
                _record_chunk(None, None)
            else:
                cursor = prompt_idx
                last_end = prompt_idx - 1
                while True:
                    last_idx = _record_chunk(cursor, reprompt_every)
                    if last_idx is None or last_idx <= last_end:
                        break  # no forward progress -> end of video
                    last_end = last_idx

                    # Re-prompt at the chunk boundary: register robots that have
                    # newly come into view (not overlapping any tracked mask).
                    capped = max_objects is not None and (next_obj_id - 1) >= max_objects
                    if not capped:
                        frame_path = (
                            frame_paths[last_idx] if last_idx < len(frame_paths) else None
                        )
                        try:
                            detected = detect_fn(last_idx, frame_path) or []
                        except Exception:
                            detected = []
                        new_boxes = []
                        new_metadata = []
                        for item in detected:
                            if (
                                isinstance(item, tuple)
                                and len(item) == 2
                                and hasattr(item[0], "__len__")
                                and len(item[0]) == 4
                            ):
                                new_boxes.append(item[0])
                                new_metadata.append(item[1])
                            else:
                                new_boxes.append(item)
                                new_metadata.append(None)
                        existing = {
                            oid: entry["mask"]
                            for oid, entry in results.get(last_idx, {}).items()
                        }
                        if new_boxes:
                            # Use any tracked mask's shape for the frame size; fall
                            # back to a detection-derived size if nothing tracked.
                            if existing:
                                h, w = next(iter(existing.values())).shape
                            else:
                                w = max(int(b[2]) for b in new_boxes) + 1
                                h = max(int(b[3]) for b in new_boxes) + 1
                            int_boxes = [
                                (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                                for b in new_boxes
                            ]
                            for j in match_new_detections(
                                existing, int_boxes, (w, h), ios_thresh=new_obj_ios_thresh
                            ):
                                if max_objects is not None and (next_obj_id - 1) >= max_objects:
                                    break
                                predictor.add_new_points_or_box(
                                    state,
                                    frame_idx=last_idx,
                                    obj_id=next_obj_id,
                                    box=np.asarray(new_boxes[j], dtype=np.float32),
                                )
                                if object_metadata is not None:
                                    object_metadata[next_obj_id] = new_metadata[j]
                                next_obj_id += 1

                    cursor = last_idx + 1
        finally:
            if hasattr(predictor, "reset_state"):
                predictor.reset_state(state)

        return results

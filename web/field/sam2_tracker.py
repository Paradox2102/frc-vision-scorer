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
    ) -> None:
        """Initialise the tracker.

        Args:
            predictor: An already-built SAM2 predictor (or stub).  When
                ``None``, ``build_sam2_video_predictor`` is called lazily.
            config: SAM2 model config file (passed to
                ``build_sam2_video_predictor``).
            checkpoint: Path to the SAM2 checkpoint file.
            device: torch device string (``"cuda"`` or ``"cpu"``).
        """
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

            self._predictor = build_sam2_video_predictor(
                config, checkpoint, device=device
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
            reprompt_every: Reserved for future mid-stream reprompting.
                Currently ignored; accepted so callers can pass it without a
                TypeError.
            detect_fn: Reserved for future mid-stream reprompting.  Ignored.

        Returns:
            Nested dict ``{frame_idx: {obj_id: {"box": (x1,y1,x2,y2),
            "mask": ndarray_2d}}}``.  Objects with empty masks on a given
            frame are omitted from that frame's dict.
        """
        predictor = self._predictor

        state = predictor.init_state(video_path=frames_dir)

        # Prompt each robot box on the prompt frame.
        for i, box in enumerate(prompt_boxes):
            obj_id = i + 1
            box_arr = np.asarray(box, dtype=np.float32)
            predictor.add_new_points_or_box(
                state,
                frame_idx=prompt_idx,
                obj_id=obj_id,
                box=box_arr,
            )

        results: dict[int, dict[int, dict]] = {}

        # Propagate — use torch contexts only on the real-torch path.
        def _propagate():
            if self._real_torch:
                import torch
                with torch.inference_mode(), _maybe_autocast(self._device):
                    yield from predictor.propagate_in_video(state)
            else:
                yield from predictor.propagate_in_video(state)

        for f_idx, obj_ids, mask_logits in _propagate():
            # mask_logits may be a real torch tensor (shape N,1,H,W) or our
            # FakeTensor stub.  Either way: > 0, .cpu(), .numpy() works.
            bool_arr = (mask_logits > 0).cpu().numpy()  # numpy bool array

            frame_entry: dict[int, dict] = {}
            for i, obj_id in enumerate(obj_ids):
                # Squeeze per-object slice to 2D regardless of leading dims.
                if bool_arr.ndim == 4:
                    mask2d = bool_arr[i, 0].astype(bool)
                elif bool_arr.ndim == 3:
                    mask2d = bool_arr[i].astype(bool)
                else:
                    mask2d = bool_arr.astype(bool)

                box = mask_to_box(mask2d)
                if box is None:
                    # Object has no visible pixels this frame — skip.
                    continue

                frame_entry[int(obj_id)] = {"box": box, "mask": mask2d}

            results[int(f_idx)] = frame_entry

        return results

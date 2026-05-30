"""Pure, GPU-free helper functions for the offline SAM2 mask-propagation tracker.

These utilities handle mask/box conversions and new-detection matching using
only NumPy, so they can be unit-tested without a GPU or SAM2 installed.

The stateful Sam2FieldTracker class (which imports torch and sam2) will be
added to this same file in a later task.
"""
from __future__ import annotations

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

"""Bumper number box-to-robot mask matching.

Uses Intersection-over-Smaller (IoS) to associate detected number bounding boxes
with robot masks.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


def match_numbers_to_robots(
    frame: np.ndarray,
    number_boxes: np.ndarray | list,
    robot_masks: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Match bumper number bounding boxes to robot masks using IoS.

    Args:
        frame: The camera image frame (numpy array).
        number_boxes: A 2D numpy array or list of shape (N, 4) containing number bounding boxes
            in [x1, y1, x2, y2] format.
        robot_masks: A dict mapping robot track ID to its 2D boolean mask.

    Returns:
        A dict mapping robot track ID to the crop of the matched number box.
    """
    if len(number_boxes) == 0 or len(robot_masks) == 0:
        return {}

    boxes_arr = np.asarray(number_boxes, dtype=np.float32)
    
    # Get mask resolution from one of the robot masks
    height, width = next(iter(robot_masks.values())).shape
    resolution_wh = (width, height)

    # Rasterize number boxes to masks: shape (N, height, width)
    number_masks = sv.xyxy_to_mask(boxes_arr, resolution_wh)

    # Stack robot masks: shape (M, height, width)
    track_ids = list(robot_masks.keys())
    stacked_robot_masks = np.stack([robot_masks[tid] for tid in track_ids], axis=0)

    # Compute Intersection over Smaller (IoS) between number masks and robot masks
    # Shape of ios_matrix is (N, M)
    ios_matrix = sv.mask_iou_batch(number_masks, stacked_robot_masks, overlap_metric=sv.OverlapMetric.IOS)

    result = {}
    best_ios = {}
    h, w = frame.shape[:2]

    for i in range(len(boxes_arr)):
        # Find which robot masks this number box overlaps with by >= 0.9
        matching_robot_indices = [j for j in range(len(track_ids)) if ios_matrix[i, j] >= 0.9]

        if len(matching_robot_indices) == 1:
            # Single unambiguous match
            robot_idx = matching_robot_indices[0]
            track_id = track_ids[robot_idx]
            ios_val = ios_matrix[i, robot_idx]

            # If this is the best match (or only match) for this robot track ID
            if track_id not in best_ios or ios_val > best_ios[track_id]:
                x1, y1, x2, y2 = map(int, boxes_arr[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        result[track_id] = crop
                        best_ios[track_id] = ios_val

    return result

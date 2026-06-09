"""SAM 3.1 adapter for the existing offline field-tracker pipeline.

Meta's SAM 3.1 predictor uses a session/request API, while the application is
built around the lower-level SAM2 video-predictor protocol.  This module keeps
that protocol boundary stable so re-prompting, culling, fusion, and identity
logic do not need separate SAM3 implementations.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from web.field.sam2_tracker import Sam2FieldTracker


class Sam3PredictorAdapter:
    """Expose a SAM 3.1 session predictor through the SAM2 predictor protocol."""

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor

    def init_state(
        self,
        *,
        video_path: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ) -> dict:
        response = self._predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": video_path,
                "offload_video_to_cpu": offload_video_to_cpu,
                "offload_state_to_cpu": offload_state_to_cpu,
            }
        )
        return {"session_id": response["session_id"], "closed": False}

    def add_new_points_or_box(
        self,
        state: dict,
        *,
        frame_idx: int,
        obj_id: int,
        box,
    ) -> None:
        x1, y1, x2, y2 = (float(v) for v in box)
        self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": state["session_id"],
                "frame_index": int(frame_idx),
                "obj_id": int(obj_id),
                "bounding_boxes": [[x1, y1, x2 - x1, y2 - y1]],
                "bounding_box_labels": [1],
                "clear_old_boxes": False,
                "rel_coordinates": False,
            }
        )

    def propagate_in_video(
        self,
        state: dict,
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        import torch

        request = {
            "type": "propagate_in_video",
            "session_id": state["session_id"],
            "propagation_direction": "backward" if reverse else "forward",
        }
        if start_frame_idx is not None:
            request["start_frame_index"] = int(start_frame_idx)
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = int(max_frame_num_to_track)

        for response in self._predictor.handle_stream_request(request):
            outputs = response.get("outputs") or {}
            raw_obj_ids = outputs.get("out_obj_ids", [])
            raw_masks = outputs.get("out_binary_masks", [])
            if hasattr(raw_obj_ids, "detach"):
                raw_obj_ids = raw_obj_ids.detach().cpu().numpy()
            if hasattr(raw_masks, "detach"):
                raw_masks = raw_masks.detach().cpu().numpy()
            obj_ids = np.asarray(raw_obj_ids, dtype=np.int64)
            masks = np.asarray(raw_masks)
            if masks.ndim == 2:
                masks = masks[None, ...]
            yield (
                int(response["frame_index"]),
                obj_ids,
                torch.as_tensor(masks, dtype=torch.float32),
            )

    def reset_state(self, state: dict) -> None:
        if state.get("closed"):
            return
        try:
            self._predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": state["session_id"],
                }
            )
        finally:
            state["closed"] = True


class Sam3FieldTracker(Sam2FieldTracker):
    """Run the field tracker with Meta's SAM 3.1 Object Multiplex backend."""

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
        use_fa3: bool = True,
    ) -> None:
        if device != "cuda":
            raise RuntimeError("SAM 3.1 Object Multiplex currently requires CUDA")

        if predictor is None:
            from sam3.model_builder import build_sam3_predictor

            checkpoint = checkpoint or os.environ.get("SAM3_CHECKPOINT") or None
            predictor = build_sam3_predictor(
                checkpoint_path=checkpoint,
                version="sam3.1",
                compile=compile_model,
                max_num_objects=max_objects,
                multiplex_count=16,
                use_fa3=use_fa3,
                use_rope_real=True,
            )

        super().__init__(
            predictor=Sam3PredictorAdapter(predictor),
            device=device,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        # The adapter returns real torch tensors and benefits from inference mode.
        self._real_torch = True

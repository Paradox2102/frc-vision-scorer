import os

import cv2
import numpy as np
import pytest

from web.field.sam3_tracker import Sam3FieldTracker, Sam3PredictorAdapter


class StubSam3Predictor:
    def __init__(self):
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": f"session-{len(self.requests)}"}
        if request["type"] == "add_prompt":
            return {"frame_index": request["frame_index"], "outputs": {}}
        if request["type"] == "close_session":
            return {"is_success": True}
        raise AssertionError(f"unexpected request: {request}")

    def handle_stream_request(self, request):
        self.requests.append(request)
        mask = np.zeros((1, 8, 10), dtype=bool)
        mask[0, 2:6, 3:8] = True
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": np.array([7], dtype=np.int64),
                "out_binary_masks": mask,
            },
        }


def _write_frames(dir_path, n, width=10, height=8):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(n):
        cv2.imwrite(os.path.join(str(dir_path), f"{i:05d}.jpg"), img)


def test_sam3_adapter_translates_box_and_stream_contract(tmp_path):
    _write_frames(tmp_path, 5)
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw)

    state = adapter.init_state(
        video_path=str(tmp_path),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )
    assert state["video_size"] == (10, 8)

    adapter.add_new_points_or_box(
        state,
        frame_idx=4,
        obj_id=7,
        box=np.array([1, 2, 4, 7], dtype=np.float32),
    )
    # No session yet: the prompt is deferred until propagation opens a chunk.
    assert not any(r["type"] == "add_prompt" for r in raw.requests)

    frames = list(
        adapter.propagate_in_video(
            state,
            start_frame_idx=4,
            max_frame_num_to_track=12,
            reverse=True,
        )
    )
    adapter.reset_state(state)
    adapter.reset_state(state)

    session = next(r for r in raw.requests if r["type"] == "start_session")
    # The chunk session opens on a symlinked window covering the range.
    assert os.path.basename(session["resource_path"]).startswith("sam3_chunk_")

    prompt = next(r for r in raw.requests if r["type"] == "add_prompt")
    # Box rides the instance path: two corner points with SAM2's box labels,
    # normalized to the probed frame size, under the stable caller obj_id.
    assert prompt["points"] == [[1 / 10, 2 / 8], [4 / 10, 7 / 8]]
    assert prompt["point_labels"] == [2, 3]
    assert prompt["obj_id"] == 7
    assert prompt["rel_coordinates"] is True
    assert prompt["frame_index"] == 4  # chunk covers frames 0..4 -> local 4

    propagation = next(
        r for r in raw.requests if r["type"] == "propagate_in_video"
    )
    assert propagation["propagation_direction"] == "backward"
    assert propagation["start_frame_index"] == 4

    frame_idx, obj_ids, masks = frames[0]
    assert frame_idx == 0
    assert obj_ids.tolist() == [7]
    assert tuple(masks.shape) == (1, 8, 10)
    assert masks[0, 2:6, 3:8].all()
    # Observed mask refreshed the carry-over box for chunk-boundary re-seeds.
    seen_idx, bbox = state["last_box"][7]
    assert seen_idx == 0 and bbox == (3.0, 2.0, 8.0, 6.0)
    assert sum(r["type"] == "close_session" for r in raw.requests) == 1


def test_sam3_adapter_reseeds_across_chunk_boundary(tmp_path):
    _write_frames(tmp_path, 6)
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw, chunk_frames=16)
    # chunk_frames floors at 16; use a 6-frame dir with a 16-frame window so
    # everything fits one chunk, then force a second window manually.
    state = adapter.init_state(video_path=str(tmp_path))
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=3, box=np.array([2, 2, 6, 6], np.float32)
    )
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=5))
    # Simulate the next propagation range needing a new window: close the
    # live chunk and propagate again — live objects must be re-seeded from
    # their last observed boxes under the SAME obj_ids (3 from its prompt,
    # 7 from the stub's streamed mask), never under fresh ids.
    adapter._close_chunk(state)
    raw.requests.clear()
    list(adapter.propagate_in_video(state, start_frame_idx=2,
                                    max_frame_num_to_track=3))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert prompts, "expected boundary re-seed prompts"
    assert {p["obj_id"] for p in prompts} <= {3, 7}
    assert all(p["point_labels"] == [2, 3] for p in prompts)
    adapter.reset_state(state)


def test_sam3_field_tracker_preserves_existing_result_contract(tmp_path):
    _write_frames(tmp_path, 1)
    raw = StubSam3Predictor()
    tracker = Sam3FieldTracker(predictor=raw)

    results = tracker.track(
        str(tmp_path),
        prompt_idx=0,
        prompt_boxes=[(3, 2, 8, 6)],
    )

    assert results[0][7]["box"] == (3, 2, 8, 6)
    assert results[0][7]["mask"].dtype == bool
    assert results[0][7]["evidence"]["type"] == "initial"

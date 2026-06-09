import numpy as np

from web.field.sam3_tracker import Sam3FieldTracker, Sam3PredictorAdapter


class StubSam3Predictor:
    def __init__(self):
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session-1"}
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


def test_sam3_adapter_translates_box_and_stream_contract():
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw)

    state = adapter.init_state(
        video_path="/tmp/frames",
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )
    adapter.add_new_points_or_box(
        state,
        frame_idx=4,
        obj_id=7,
        box=np.array([10, 20, 40, 70], dtype=np.float32),
    )
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

    prompt = next(r for r in raw.requests if r["type"] == "add_prompt")
    assert prompt["bounding_boxes"] == [[10.0, 20.0, 30.0, 50.0]]
    assert prompt["bounding_box_labels"] == [1]
    assert prompt["obj_id"] == 7
    assert prompt["rel_coordinates"] is False
    propagation = next(
        r for r in raw.requests if r["type"] == "propagate_in_video"
    )
    assert propagation["propagation_direction"] == "backward"

    frame_idx, obj_ids, masks = frames[0]
    assert frame_idx == 0
    assert obj_ids.tolist() == [7]
    assert tuple(masks.shape) == (1, 8, 10)
    assert masks[0, 2:6, 3:8].all()
    assert sum(r["type"] == "close_session" for r in raw.requests) == 1


def test_sam3_field_tracker_preserves_existing_result_contract(tmp_path):
    (tmp_path / "00000.jpg").write_bytes(b"stub")
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

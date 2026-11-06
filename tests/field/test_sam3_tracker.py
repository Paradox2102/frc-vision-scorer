import os

import cv2
import numpy as np
import pytest

from web.field.sam3_tracker import (
    SAM3_MULTIPLEX_MAX_OBJECTS,
    Sam3FieldTracker,
    Sam3PredictorAdapter,
    _install_multiplex_shims,
)


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


class ActionHistoryStubPredictor:
    """Mimics the real SAM3 instance-session propagate contract: a propagate
    stream emits masks ONLY for obj_ids prompted since the previous propagate
    request in the same session (``propagation_partial``); with no new prompts
    it emits nothing (``propagation_fetch``)."""

    def __init__(self):
        self.requests = []
        self._sessions = {}
        self._next = 0

    def handle_request(self, request):
        self.requests.append(request)
        t = request["type"]
        if t == "start_session":
            self._next += 1
            sid = f"s{self._next}"
            self._sessions[sid] = {"prompted_since": set()}
            return {"session_id": sid}
        if t == "add_prompt":
            sess = self._sessions[request["session_id"]]
            sess["prompted_since"].add(request["obj_id"])
            return {"frame_index": request["frame_index"], "outputs": {}}
        if t == "close_session":
            self._sessions.pop(request["session_id"], None)
            return {"is_success": True}
        raise AssertionError(f"unexpected request: {request}")

    def handle_stream_request(self, request):
        self.requests.append(request)
        sess = self._sessions[request["session_id"]]
        live = sorted(sess["prompted_since"])
        sess["prompted_since"] = set()  # action history consumed
        start = request["start_frame_index"]
        span = request["max_frame_num_to_track"]
        for f in range(start, start + span + 1):
            masks = np.zeros((len(live), 8, 10), dtype=bool)
            masks[:, 2:6, 3:8] = True
            yield {
                "frame_index": f,
                "outputs": {
                    "out_obj_ids": np.array(live, dtype=np.int64),
                    "out_binary_masks": masks,
                },
            }


class ProbsStubPredictor:
    """Emits a single propagated frame whose outputs optionally carry
    ``out_tracker_probs``. ``probs`` is passed straight into the outputs dict
    so tests can exercise present / absent / (N,1) / length-mismatch shapes."""

    def __init__(self, obj_ids, probs):
        self.requests = []
        self._obj_ids = np.asarray(obj_ids, dtype=np.int64)
        self._probs = probs

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
        n = len(self._obj_ids)
        masks = np.zeros((n, 8, 10), dtype=bool)
        masks[:, 2:6, 3:8] = True
        outputs = {
            "out_obj_ids": self._obj_ids,
            "out_binary_masks": masks,
        }
        if self._probs is not None:
            outputs["out_tracker_probs"] = self._probs
        yield {"frame_index": 0, "outputs": outputs}


def _run_probs_adapter(tmp_path, obj_ids, probs):
    _write_frames(tmp_path, 5)
    raw = ProbsStubPredictor(obj_ids, probs)
    adapter = Sam3PredictorAdapter(raw)
    state = adapter.init_state(video_path=str(tmp_path))
    for oid in obj_ids:
        adapter.add_new_points_or_box(
            state, frame_idx=0, obj_id=int(oid),
            box=np.array([1, 2, 4, 7], np.float32),
        )
    frames = list(
        adapter.propagate_in_video(state, start_frame_idx=0,
                                   max_frame_num_to_track=3)
    )
    adapter.reset_state(state)
    return frames


def test_sam3_adapter_yields_tracker_probs_as_flat_float(tmp_path):
    frames = _run_probs_adapter(
        tmp_path, [7, 8], np.array([0.9, 0.5], dtype=np.float32)
    )
    _f, obj_ids, _m, probs = frames[0]
    assert obj_ids.tolist() == [7, 8]
    assert probs is not None
    assert probs.ndim == 1
    assert probs.dtype.kind == "f"
    assert probs.tolist() == pytest.approx([0.9, 0.5], abs=1e-6)


def test_sam3_adapter_yields_none_probs_when_absent(tmp_path):
    frames = _run_probs_adapter(tmp_path, [7], None)
    _f, _ids, _m, probs = frames[0]
    assert probs is None


def test_sam3_adapter_flattens_column_vector_probs(tmp_path):
    frames = _run_probs_adapter(
        tmp_path, [7, 8], np.array([[0.3], [0.7]], dtype=np.float32)
    )
    _f, _ids, _m, probs = frames[0]
    assert probs is not None
    assert probs.shape == (2,)
    assert probs.tolist() == pytest.approx([0.3, 0.7], abs=1e-6)


def test_sam3_adapter_drops_probs_on_length_mismatch(tmp_path):
    frames = _run_probs_adapter(
        tmp_path, [7, 8], np.array([0.3, 0.7, 0.1], dtype=np.float32)
    )
    _f, _ids, _m, probs = frames[0]
    assert probs is None


def test_sam3_tracker_records_tracker_prob_evidence(tmp_path):
    """Tracker probs from the SAM3 adapter surface as per-object per-frame
    evidence['tracker_prob']."""
    _write_frames(tmp_path, 1)
    raw = ProbsStubPredictor([7], np.array([0.82], dtype=np.float32))
    tracker = Sam3FieldTracker(predictor=raw)
    results = tracker.track(str(tmp_path), prompt_idx=0, prompt_boxes=[(3, 2, 8, 6)])
    assert results[0][7]["evidence"]["tracker_prob"] == pytest.approx(0.82)


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

    frame_idx, obj_ids, masks, _probs = frames[0]
    assert frame_idx == 0
    assert obj_ids.tolist() == [7]
    assert tuple(masks.shape) == (1, 8, 10)
    assert masks[0, 2:6, 3:8].all()
    # Observed mask refreshed the carry-over box for chunk-boundary re-seeds.
    seen_idx, bbox = state["last_box"][7]
    assert seen_idx == 0 and bbox == (3.0, 2.0, 8.0, 6.0)
    assert sum(r["type"] == "close_session" for r in raw.requests) == 1


def test_add_prompt_forwards_negative_points_as_label_zero(tmp_path):
    _write_frames(tmp_path, 5)
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw)
    state = adapter.init_state(video_path=str(tmp_path))

    adapter.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        box=(10, 10, 20, 20),
        points=[(30.0, 15.0)],
        labels=[0],
    )
    # No chunk open yet: prompt stays pending.
    assert not any(r["type"] == "add_prompt" for r in raw.requests)

    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=0))
    adapter.reset_state(state)

    req = next(r for r in raw.requests if r["type"] == "add_prompt")
    width, height = state["video_size"]
    # box corners (labels 2,3) followed by the negative point (label 0), all
    # normalized by video_size.
    assert req["point_labels"] == [2, 3, 0]
    assert req["points"] == [
        [10 / width, 10 / height],
        [20 / width, 20 / height],
        [30.0 / width, 15.0 / height],
    ]


def test_add_prompt_accepts_ndarray_points_and_labels(tmp_path):
    """_prompt_with_negatives (sam2_tracker) passes points/labels as ndarrays,
    which the real SAM2 predictor takes natively. The adapter's old
    ``points or []`` raised "truth value of an array ... is ambiguous" on any
    multi-element ndarray — with neg_point_prompts default-ON that killed a
    whole view's tracking (qm11 corner_right)."""
    _write_frames(tmp_path, 5)
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw)
    state = adapter.init_state(video_path=str(tmp_path))

    # Exactly what _prompt_with_negatives sends: float32 (N,2) points and an
    # int32 zeros label vector. Must not raise.
    adapter.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        box=(10, 10, 20, 20),
        points=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        labels=np.zeros(2, dtype=np.int32),
    )
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=0))
    adapter.reset_state(state)

    req = next(r for r in raw.requests if r["type"] == "add_prompt")
    width, height = state["video_size"]
    # Both negatives ride along after the box corners (labels 2,3 then 0,0).
    assert req["point_labels"] == [2, 3, 0, 0]
    assert req["points"] == [
        [10 / width, 10 / height],
        [20 / width, 20 / height],
        [10.0 / width, 20.0 / height],
        [30.0 / width, 40.0 / height],
    ]


def test_add_prompt_ignores_positive_label_points(tmp_path):
    _write_frames(tmp_path, 5)
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw)
    state = adapter.init_state(video_path=str(tmp_path))

    adapter.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        box=(10, 10, 20, 20),
        points=[(30.0, 15.0), (5.0, 5.0)],
        labels=[0, 1],
    )
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=0))
    adapter.reset_state(state)

    req = next(r for r in raw.requests if r["type"] == "add_prompt")
    # Only the label-0 point rides along; the label-1 (positive) point is
    # dropped since it would fight the box corners.
    assert req["point_labels"] == [2, 3, 0]
    assert len(req["points"]) == 3


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


def test_sam3_adapter_keeps_roster_alive_across_windowed_propagates(tmp_path):
    """Repeated windowed propagate calls (the pipeline's reprompt_every loop)
    must keep emitting masks for ALL live objects, not just freshly prompted
    ones — SAM3 sessions otherwise degenerate to fetch/partial and every
    object dies one window after its prompt."""
    _write_frames(tmp_path, 24)
    raw = ActionHistoryStubPredictor()
    adapter = Sam3PredictorAdapter(raw, chunk_frames=64)
    state = adapter.init_state(video_path=str(tmp_path))
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=1, box=np.array([1, 1, 5, 5], np.float32)
    )
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=2, box=np.array([4, 2, 9, 6], np.float32)
    )

    w1 = list(adapter.propagate_in_video(state, start_frame_idx=0,
                                         max_frame_num_to_track=7))
    assert [f for f, *_ in w1] == list(range(0, 8))
    assert all(set(ids.tolist()) == {1, 2} for _, ids, *_ in w1)

    w2 = list(adapter.propagate_in_video(state, start_frame_idx=8,
                                         max_frame_num_to_track=7))
    assert [f for f, *_ in w2] == list(range(8, 16))
    assert all(set(ids.tolist()) == {1, 2} for _, ids, *_ in w2), (
        "objects without fresh prompts stopped emitting in the second window"
    )
    adapter.reset_state(state)


def test_sam3_adapter_window_refresh_skips_stale_objects(tmp_path):
    """The per-window roster refresh must not re-prompt objects whose last
    evidence is older than SAM3_RESEED_MAX_AGE — stale boxes hijack
    background; detector re-prompts own that recovery path."""
    _write_frames(tmp_path, 40)
    raw = ActionHistoryStubPredictor()
    adapter = Sam3PredictorAdapter(raw, chunk_frames=64)
    state = adapter.init_state(video_path=str(tmp_path))
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=1, box=np.array([1, 1, 5, 5], np.float32)
    )
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=7))
    # Object 9 last seen at frame 0: more than SAM3_RESEED_MAX_AGE (24)
    # frames before the next window's start at 30.
    state["last_box"][9] = (0, (1.0, 1.0, 4.0, 4.0))

    raw.requests.clear()
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=30,
                                         max_frame_num_to_track=5))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert {p["obj_id"] for p in prompts} == {1}, (
        "expected a refresh prompt for the live object only"
    )
    assert all(set(ids.tolist()) == {1} for _, ids, *_ in w2)
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


def test_sam3_tracker_uses_sam3_tags_for_logs_and_diagnostics(tmp_path):
    """SAM3 runs must label themselves sam3 in diagnostics events (and the
    log tag), not inherit the literal 'sam2.*' names from the base class."""
    from web.field.sam2_tracker import Sam2FieldTracker

    assert Sam3FieldTracker.DIAG_PREFIX == "sam3"
    assert Sam3FieldTracker.LOG_TAG == "SAM3"
    assert Sam2FieldTracker.DIAG_PREFIX == "sam2"
    assert Sam2FieldTracker.LOG_TAG == "SAM2"

    class _Diag:
        def __init__(self):
            self.events = []

        def record(self, event, **kw):
            self.events.append(event)

        def increment(self, *_a, **_k):
            pass

    _write_frames(tmp_path, 4)
    raw = StubSam3Predictor()
    tracker = Sam3FieldTracker(predictor=raw)
    diag = _Diag()
    tracker.track(
        str(tmp_path),
        prompt_idx=0,
        prompt_boxes=[(3, 2, 8, 6)],
        reprompt_every=2,
        detect_fn=lambda i, p: [(30, 2, 34, 6)],
        diagnostics=diag,
    )
    assert diag.events, "expected diagnostics events"
    assert all(e.startswith("sam3.") for e in diag.events), diag.events


def test_sam3_tracker_should_stop_aborts_and_closes_chunk_session(tmp_path):
    """Task 1 (reliable STOP): ``should_stop`` reaches the SAM3 tracker through
    the inherited ``track()`` and aborts propagation mid-stream. The chunk
    session must still be torn down (close_session + temp dir removed) by the
    ``finally: predictor.reset_state(state)`` cleanup."""
    from web.field.sam2_tracker import TrackStopped

    _write_frames(tmp_path, 8)
    raw = ActionHistoryStubPredictor()
    tracker = Sam3FieldTracker(predictor=raw)

    polls = {"n": 0}

    def should_stop():
        # Let the first propagated frame land, then request the stop.
        polls["n"] += 1
        return polls["n"] >= 2

    with pytest.raises(TrackStopped):
        tracker.track(
            str(tmp_path),
            prompt_idx=0,
            prompt_boxes=[(3, 2, 8, 6)],
            should_stop=should_stop,
        )

    # It bailed after ~2 frames, not the full 8-frame stream.
    assert polls["n"] == 2
    # Cleanup ran: exactly one chunk session was opened and it was closed.
    opened = sum(r["type"] == "start_session" for r in raw.requests)
    closed = sum(r["type"] == "close_session" for r in raw.requests)
    assert opened == 1 and closed == 1


def test_sam3_tracker_should_stop_none_is_default_full_run(tmp_path):
    """Without should_stop the SAM3 tracker's behavior is unchanged."""
    _write_frames(tmp_path, 4)
    raw = ActionHistoryStubPredictor()
    tracker = Sam3FieldTracker(predictor=raw)
    results = tracker.track(
        str(tmp_path),
        prompt_idx=0,
        prompt_boxes=[(3, 2, 8, 6)],
    )
    assert set(results.keys()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Late-seed backward backfill scoping (state["solo_ids"])
# ---------------------------------------------------------------------------

def test_solo_ids_scopes_stream_chunk_window_refresh(tmp_path):
    """Late-seed backfill scoping: with state['solo_ids'] set, _stream_chunk's
    per-window roster refresh re-prompts ONLY the target ids — otherwise a
    reverse backfill pass would drag every other live object along too
    (wasted compute, and a regression of their last_box to the backfill's own
    reverse timeline)."""
    _write_frames(tmp_path, 24)
    raw = ActionHistoryStubPredictor()
    adapter = Sam3PredictorAdapter(raw, chunk_frames=64)
    state = adapter.init_state(video_path=str(tmp_path))
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=1, box=np.array([1, 1, 5, 5], np.float32)
    )
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=2, box=np.array([4, 2, 9, 6], np.float32)
    )
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=7))

    # Without solo_ids the next window would refresh BOTH live objects (as
    # test_sam3_adapter_keeps_roster_alive_across_windowed_propagates shows).
    # Scope this window to obj 1 only.
    state["solo_ids"] = {1}
    raw.requests.clear()
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=8,
                                         max_frame_num_to_track=7))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert {p["obj_id"] for p in prompts} == {1}, (
        "solo_ids must confine the window refresh to the backfill target")
    assert all(set(ids.tolist()) == {1} for _, ids, *_ in w2)
    adapter.reset_state(state)


# ---------------------------------------------------------------------------
# sam3_multiplex: builder selection (SAM 3.1 Object Multiplex opt-in)
# ---------------------------------------------------------------------------

def _patch_sam3_builders(monkeypatch):
    """Stub out both model_builder entry points so __init__'s predictor-build
    branch runs with no real model load / no download, and record the kwargs
    each was called with."""
    import sam3.model_builder as model_builder

    calls = {"video": None, "multiplex": None}

    def fake_build_sam3_video_predictor(**kwargs):
        calls["video"] = kwargs
        return object()

    def fake_build_sam3_predictor(**kwargs):
        calls["multiplex"] = kwargs
        return object()

    monkeypatch.setattr(
        model_builder, "build_sam3_video_predictor", fake_build_sam3_video_predictor
    )
    monkeypatch.setattr(
        model_builder, "build_sam3_predictor", fake_build_sam3_predictor
    )
    return calls


def test_sam3_multiplex_off_uses_base_builder_unchanged(monkeypatch):
    """Default (multiplex=False): today's exact call to
    build_sam3_video_predictor, untouched by the new knob."""
    calls = _patch_sam3_builders(monkeypatch)

    Sam3FieldTracker(checkpoint="ckpt.pt", compile_model=True, multiplex=False)

    assert calls["multiplex"] is None
    assert calls["video"] == {"checkpoint_path": "ckpt.pt", "compile": True}


def test_sam3_multiplex_on_uses_multiplex_builder_with_fa3_forced_off(monkeypatch):
    """multiplex=True selects the SAM 3.1 Object Multiplex builder
    (build_sam3_predictor(version="sam3.1", ...)) instead of the base video
    predictor, with use_fa3 forced False -- FlashAttention-3 is not installed
    on this host, and the builder's own default is use_fa3=True."""
    calls = _patch_sam3_builders(monkeypatch)

    Sam3FieldTracker(checkpoint="ckpt.pt", compile_model=True, multiplex=True)

    assert calls["video"] is None
    assert calls["multiplex"] == {
        "version": "sam3.1",
        "checkpoint_path": "ckpt.pt",
        "compile": True,
        "use_fa3": False,
        "max_num_objects": SAM3_MULTIPLEX_MAX_OBJECTS,
    }


def test_sam3_multiplex_shims_init_state_without_state_offload(monkeypatch):
    """Sam3BasePredictor.start_session unconditionally forwards
    offload_state_to_cpu into model.init_state(**kwargs), but the SAM 3.1
    multiplex model's init_state doesn't accept it — every live session died
    with TypeError and all three views fell to the no-SAM3 fallback (qm11
    2026-07-20 A/B). The multiplex branch must shim the built model's
    init_state to accept-and-drop the kwarg."""
    import sam3.model_builder as model_builder

    seen = {}

    class MuxModel:
        def init_state(self, resource_path, offload_video_to_cpu=False,
                       async_loading_frames=False):
            seen["kwargs"] = dict(resource_path=resource_path,
                                  offload_video_to_cpu=offload_video_to_cpu)
            return {"ok": True}

    class MuxPredictor:
        def __init__(self):
            self.model = MuxModel()

    built = MuxPredictor()
    monkeypatch.setattr(model_builder, "build_sam3_predictor",
                        lambda **kw: built)

    Sam3FieldTracker(checkpoint="ckpt.pt", multiplex=True)

    # Exactly what base start_session sends; must not raise, kwarg dropped.
    state = built.model.init_state(resource_path="dir",
                                   offload_video_to_cpu=True,
                                   offload_state_to_cpu=True)
    assert state == {"ok": True}
    assert seen["kwargs"] == {"resource_path": "dir",
                              "offload_video_to_cpu": True}


def test_sam3_multiplex_leaves_native_init_state_unshimmed(monkeypatch):
    """When the model's init_state already accepts offload_state_to_cpu (the
    base SAM3 predictor today, or a fixed upstream 3.1), no shim is applied."""
    import sam3.model_builder as model_builder

    class Model:
        def init_state(self, resource_path, offload_video_to_cpu=False,
                       offload_state_to_cpu=False):
            return "native"

    class Predictor:
        def __init__(self):
            self.model = Model()

    built = Predictor()
    monkeypatch.setattr(model_builder, "build_sam3_predictor",
                        lambda **kw: built)
    Sam3FieldTracker(checkpoint="ckpt.pt", multiplex=True)
    assert built.model.init_state.__func__ is Model.init_state


def test_sam3_multiplex_raises_object_capacity_above_builder_default(monkeypatch):
    """The 3.1 builder defaults max_num_objects=16, but a chunk session's
    lifetime obj_id count (every re-seed mints a new id, nothing is removed)
    blows through 16 within ~16 frames; every later add was refused and the
    following fetch-propagate crashed with KeyError. The multiplex build must
    pass an explicit, larger capacity (and never below max_objects)."""
    calls = _patch_sam3_builders(monkeypatch)
    Sam3FieldTracker(checkpoint="ckpt.pt", multiplex=True)
    assert calls["multiplex"]["max_num_objects"] >= 64
    assert SAM3_MULTIPLEX_MAX_OBJECTS >= 64

    calls = _patch_sam3_builders(monkeypatch)
    Sam3FieldTracker(checkpoint="ckpt.pt", multiplex=True, max_objects=200)
    assert calls["multiplex"]["max_num_objects"] == 200


class _FetchModel:
    """Mimics the multiplex model's unguarded propagation_fetch lookup."""

    def __init__(self):
        self.seen = []

    def fetch_and_process_single_frame_results(self, inference_state, frame_idx):
        obj_id_to_mask = inference_state["cached_frame_outputs"][frame_idx]
        self.seen.append(obj_id_to_mask)
        return frame_idx, {"n": len(obj_id_to_mask)}


def test_sam3_multiplex_fetch_shim_tolerates_uncached_frame():
    """propagation_fetch on a never-propagated frame raised a bare
    KeyError(frame_idx) upstream; the shim must yield an empty frame (base
    SAM3 semantics) without leaving a placeholder in cached_frame_outputs,
    and pass real cached outputs through untouched."""
    model = _FetchModel()
    _install_multiplex_shims(model)

    real = {7: "mask"}
    state = {"cached_frame_outputs": {3: real}}
    assert model.fetch_and_process_single_frame_results(state, 50) == (50, {"n": 0})
    assert model.seen[-1] == {}
    assert state["cached_frame_outputs"] == {3: real}, "placeholder leaked"

    assert model.fetch_and_process_single_frame_results(state, 3) == (3, {"n": 1})
    assert model.seen[-1] is real
    assert state["cached_frame_outputs"] == {3: real}


def test_sam3_multiplex_fetch_shim_cleans_up_on_error_and_is_idempotent():
    class Boom(_FetchModel):
        def fetch_and_process_single_frame_results(self, inference_state,
                                                   frame_idx):
            inference_state["cached_frame_outputs"][frame_idx]
            raise RuntimeError("boom")

    model = Boom()
    _install_multiplex_shims(model)
    first = model.fetch_and_process_single_frame_results
    _install_multiplex_shims(model)
    assert model.fetch_and_process_single_frame_results is first, "double-wrapped"

    state = {"cached_frame_outputs": {}}
    with pytest.raises(RuntimeError):
        model.fetch_and_process_single_frame_results(state, 9)
    assert state["cached_frame_outputs"] == {}


def test_sam3_multiplex_shims_tolerate_model_without_methods(monkeypatch):
    """No crash when the built model lacks init_state / fetch (or is absent)."""
    _install_multiplex_shims(None)
    _install_multiplex_shims(object())

    class Bare:
        pass

    bare = Bare()
    _install_multiplex_shims(bare)
    assert not hasattr(bare, "fetch_and_process_single_frame_results")

    calls = _patch_sam3_builders(monkeypatch)  # builder returns object()
    Sam3FieldTracker(checkpoint="ckpt.pt", multiplex=True)
    assert calls["multiplex"] is not None


class CapacityStubPredictor(ActionHistoryStubPredictor):
    """Refuses NEW obj_ids beyond ``capacity`` per session the way the
    multiplex model does: ``{"frame_index": f, "outputs": None}``, and the
    refused add never enters the action history."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def handle_request(self, request):
        if request["type"] == "add_prompt":
            self.requests.append(request)
            sess = self._sessions[request["session_id"]]
            known = sess.setdefault("known", set())
            oid = request["obj_id"]
            if oid not in known and len(known) >= self.capacity:
                return {"frame_index": request["frame_index"], "outputs": None}
            known.add(oid)
            sess["prompted_since"].add(oid)
            return {"frame_index": request["frame_index"], "outputs": {}}
        return super().handle_request(request)


def test_sam3_adapter_counts_refused_prompts_and_warns_once(tmp_path, capsys):
    _write_frames(tmp_path, 8)
    raw = CapacityStubPredictor(capacity=2)
    adapter = Sam3PredictorAdapter(raw, chunk_frames=16)
    state = adapter.init_state(video_path=str(tmp_path))
    for oid in (1, 2, 3, 4):
        adapter.add_new_points_or_box(
            state, frame_idx=0, obj_id=oid,
            box=np.array([1, 1, 5, 5], np.float32),
        )
    # Prompts are queued until the first propagate opens the chunk.
    gen = adapter.propagate_in_video(state, start_frame_idx=0,
                                     max_frame_num_to_track=3)
    first = next(gen)
    chunk = state["chunk"]
    # 3 and 4 refused at the chunk-open seed, then retried (and refused
    # again) by the pre-propagate roster refresh since they never entered
    # `prompted`.
    assert chunk["refused"] == 4
    assert chunk["prompted"] == set()  # consumed by the propagate
    assert set(first[1].tolist()) == {1, 2}
    rest = list(gen)
    assert all(set(ids.tolist()) == {1, 2} for _, ids, *_ in rest)

    # Mid-chunk adds after the propagate: accepted ones land in `prompted`,
    # refused ones don't.
    adapter.add_new_points_or_box(
        state, frame_idx=2, obj_id=1, box=np.array([1, 1, 5, 5], np.float32)
    )
    adapter.add_new_points_or_box(
        state, frame_idx=2, obj_id=5, box=np.array([1, 1, 5, 5], np.float32)
    )
    assert state["chunk"]["prompted"] == {1}
    assert state["chunk"]["refused"] == 5

    out = capsys.readouterr().out
    warn = [l for l in out.splitlines() if "add_prompt refused" in l]
    assert len(warn) == 1
    assert "obj 3" in warn[0] and "[0,8)" in warn[0]
    adapter.reset_state(state)


def test_sam3_adapter_treats_missing_outputs_as_accepted(tmp_path):
    """Only an explicit ``"outputs": None`` means refused; a None response or
    a response without the key (base predictor, stubs) is accepted."""
    _write_frames(tmp_path, 4)

    class NoneResponse(StubSam3Predictor):
        def handle_request(self, request):
            if request["type"] == "add_prompt":
                self.requests.append(request)
                return None if request["obj_id"] == 1 else {"frame_index": 0}
            return super().handle_request(request)

    adapter = Sam3PredictorAdapter(NoneResponse(), chunk_frames=16)
    state = adapter.init_state(video_path=str(tmp_path))
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=1))
    for oid in (1, 2):
        adapter.add_new_points_or_box(
            state, frame_idx=1, obj_id=oid,
            box=np.array([1, 1, 5, 5], np.float32),
        )
    assert state["chunk"]["prompted"] == {1, 2}
    assert state["chunk"]["refused"] == 0
    adapter.reset_state(state)


def test_solo_ids_scopes_open_chunk_for_boundary_reseed(tmp_path):
    """Same scoping contract at the chunk-boundary re-seed (_open_chunk_for):
    solo_ids must exclude a carried object even when its last_box is fresh
    enough to normally qualify for a boundary re-seed."""
    _write_frames(tmp_path, 6)
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw, chunk_frames=16)
    state = adapter.init_state(video_path=str(tmp_path))
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=3, box=np.array([2, 2, 6, 6], np.float32)
    )
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=5))
    # state["last_box"] now holds obj 3 (its prompt) and obj 7 (the stub's
    # streamed mask) — same setup as
    # test_sam3_adapter_reseeds_across_chunk_boundary, which (without
    # solo_ids) re-seeds both under their SAME ids at the next boundary.
    adapter._close_chunk(state)
    raw.requests.clear()
    state["solo_ids"] = {3}
    list(adapter.propagate_in_video(state, start_frame_idx=2,
                                    max_frame_num_to_track=3))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert prompts, "expected a boundary re-seed prompt"
    assert {p["obj_id"] for p in prompts} == {3}, (
        "solo_ids must exclude obj 7 from the boundary re-seed")
    adapter.reset_state(state)

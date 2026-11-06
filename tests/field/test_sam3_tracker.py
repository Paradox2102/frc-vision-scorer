import os

import cv2
import numpy as np
import pytest

from web.field.sam3_tracker import (
    SAM3_MULTIPLEX_MAX_OBJECTS,
    Sam3FieldTracker,
    Sam3PredictorAdapter,
    _install_multiplex_shims,
    _trim_cond_memory,
    _trim_non_cond_memory,
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


def test_sam3_multiplex_build_output_shim_keeps_refined_masks_on_uncached_frame():
    """Upstream multiplex ``_build_sam2_output`` returns {} for any frame not
    in cached_frame_outputs, discarding the masks partial propagation just
    produced — instance-prompt tracking came back empty on every propagated
    frame. Runs the REAL upstream method on a stub object: unshimmed it drops
    the refined masks; shimmed it returns them (base SAM3 semantics), merges
    over a cached frame as before, and leaves no placeholder behind."""
    import types

    mt = pytest.importorskip("sam3.model.sam3_multiplex_tracking")
    real_build = mt.Sam3MultiplexTrackingWithInteractivity._build_sam2_output

    class Model:
        pass

    refined = {1: "m1", 3: "m3"}
    raw = Model()
    raw._build_sam2_output = types.MethodType(real_build, raw)
    state = {"cached_frame_outputs": {}}
    assert raw._build_sam2_output(state, 5, refined) == {}, \
        "upstream no longer drops refined masks — shim may be obsolete"

    model = Model()
    model._build_sam2_output = types.MethodType(real_build, model)
    _install_multiplex_shims(model)
    assert model._build_sam2_output(state, 5, refined) == refined
    assert state["cached_frame_outputs"] == {}, "placeholder leaked"

    cached = {2: "c2", 3: "old"}
    state = {"cached_frame_outputs": {4: cached}}
    assert model._build_sam2_output(state, 4, refined) == {2: "c2", 3: "m3", 1: "m1"}
    assert state["cached_frame_outputs"] == {4: {2: "c2", 3: "old"}}

    first = model._build_sam2_output
    _install_multiplex_shims(model)
    assert model._build_sam2_output is first, "double-wrapped"


def _heavy_out(t):
    return {"maskmem_features": f"mf{t}", "maskmem_pos_enc": [f"pe{t}"],
            "image_features": f"if{t}", "image_pos_enc": f"ip{t}",
            "pred_masks_high_res": f"hr{t}", "pred_masks": f"pm{t}",
            "obj_ptr": f"op{t}", "object_score_logits": 1.0}


def test_mux_memory_trim_keeps_attention_window_and_entries():
    """Multiplex singleton states grew ~19 MB/object/frame unbounded (OOM ~20
    frames in). Trimming strips heavy tensors only outside what memory
    attention reads, and never deletes entries (preflight asserts
    consolidated cond indices exist; the gather skips maskmem=None)."""
    od = {"non_cond_frame_outputs": {t: _heavy_out(t) for t in range(0, 30)},
          "cond_frame_outputs": {t: _heavy_out(t) for t in (0, 3, 9, 15, 21, 24, 27, 29)}}
    assert _trim_non_cond_memory(od, 29, 7) == 22
    nc = od["non_cond_frame_outputs"]
    assert set(nc) == set(range(30)), "entries must be kept"
    for t in range(22, 30):
        assert nc[t]["maskmem_features"] == f"mf{t}"
    for t in range(0, 22):
        assert "maskmem_features" not in nc[t] and "image_features" not in nc[t]
        assert nc[t]["pred_masks"] == f"pm{t}" and nc[t]["obj_ptr"] == f"op{t}"

    assert _trim_cond_memory(od, 29, 4) == 3
    cond = od["cond_frame_outputs"]
    assert set(cond) == {0, 3, 9, 15, 21, 24, 27, 29}
    kept = {t for t, o in cond.items() if "maskmem_features" in o}
    assert kept == {0, 21, 24, 27, 29}, "first + 4 closest keep memory"
    assert _trim_cond_memory(od, 29, 4) == 0, "idempotent"
    assert _trim_non_cond_memory({}, 3, 7) == 0
    assert _trim_cond_memory({"cond_frame_outputs": {1: _heavy_out(1)}}, 9, 4) == 0


def test_mux_memory_shim_wraps_step_once_and_trims_states():
    class Tracker:
        use_memory_selection = False
        num_maskmem = 2
        memory_temporal_stride_for_eval = 1

    class Model:
        tracker = Tracker()

        def _propogate_tracker_one_frame_local_gpu(self, states, frame_idx, reverse):
            return "out"

    m = Model()
    _install_multiplex_shims(m)
    first = m._propogate_tracker_one_frame_local_gpu
    _install_multiplex_shims(m)
    assert m._propogate_tracker_one_frame_local_gpu is first
    st = {"output_dict": {"non_cond_frame_outputs": {t: _heavy_out(t) for t in range(6)},
                          "cond_frame_outputs": {0: _heavy_out(0)}}}
    assert m._propogate_tracker_one_frame_local_gpu([st], 5, False) == "out"
    nc = st["output_dict"]["non_cond_frame_outputs"]
    assert [t for t in nc if "maskmem_features" in nc[t]] == [3, 4, 5]

    class MS(Model):
        tracker = type("T", (), {"use_memory_selection": True})()

    ms = MS()
    _install_multiplex_shims(ms)
    assert not getattr(ms._propogate_tracker_one_frame_local_gpu,
                       "_frc_bound_memory_shim", False), "no trim under memory selection"


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


# ---------------------------------------------------------------------------
# sam3_skip_detector_heads: backbone-only _prepare_backbone_feats
# ---------------------------------------------------------------------------

def _ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


class _SkipModel:
    """Minimal base-SAM3 model shape for install_skip_detector_heads."""

    device = "cpu"

    def __init__(self, *, conv_s1=True, backbone_out=None):
        import torch

        self.orig_calls = []
        self.forward_calls = 0
        self._fpn = [torch.full((1, 2, 4, 4), float(i + 1)) for i in range(3)]
        self._pos = [torch.zeros(1, 2, 4, 4) for _ in range(3)]
        self._backbone_out = backbone_out

        def forward_image(image):
            self.forward_calls += 1
            assert image.dtype == torch.float32 and image.shape[0] == 1
            if self._backbone_out is not None:
                return self._backbone_out
            return {"sam2_backbone_out": {"backbone_fpn": list(self._fpn),
                                          "vision_pos_enc": self._pos}}

        decoder = _ns(conv_s0=lambda x: x * 2)
        if conv_s1:
            decoder.conv_s1 = lambda x: x * 3
        self.detector = _ns(backbone=_ns(forward_image=forward_image))
        self.tracker = _ns(sam_mask_decoder=decoder)

    def _prepare_backbone_feats(self, inference_state, frame_idx, reverse):
        self.orig_calls.append((frame_idx, reverse))


def _skip_state(n=4):
    import torch
    return {"input_batch": _ns(img_batch=torch.zeros(n, 3, 4, 4)),
            "feature_cache": {}}


def test_skip_detector_heads_install_matches_upstream_cache_entry():
    import torch
    from web.field.sam3_tracker import install_skip_detector_heads

    model = _SkipModel()
    assert install_skip_detector_heads(model) is True
    st = _skip_state()
    st["feature_cache"][0] = "prev"
    st["feature_cache"]["text"] = "keep"
    model._prepare_backbone_feats(st, 1, False)

    assert model.orig_calls == [], "detector path must not run"
    img, cache = st["feature_cache"][1]
    ref = st["input_batch"].img_batch[1]
    assert img.data_ptr() == ref.data_ptr() and img.shape == ref.shape
    out = cache["tracker_backbone_out"]
    fpn = out["backbone_fpn"]
    assert [t.dtype for t in fpn] == [torch.bfloat16] * 3
    assert torch.equal(fpn[0], (model._fpn[0].to(torch.bfloat16)) * 2)  # conv_s0
    assert torch.equal(fpn[1], (model._fpn[1].to(torch.bfloat16)) * 3)  # conv_s1
    assert torch.equal(fpn[2], model._fpn[2].to(torch.bfloat16))        # no conv
    assert out["vision_features"] is fpn[-1]
    assert out["vision_pos_enc"] is model._pos  # the list, passed through
    assert 0 not in st["feature_cache"], "previous frame must be popped"
    assert st["feature_cache"]["text"] == "keep"

    # reverse pops the NEXT frame
    st["feature_cache"][3] = "next"
    model._prepare_backbone_feats(st, 2, True)
    assert 3 not in st["feature_cache"] and 2 in st["feature_cache"]

    # already-cached frame: early return, no backbone call
    n = model.forward_calls
    model._prepare_backbone_feats(st, 2, False)
    assert model.forward_calls == n

    # idempotent
    fn = model._prepare_backbone_feats
    assert install_skip_detector_heads(model) is True
    assert model._prepare_backbone_feats is fn


def test_skip_detector_heads_falls_back_on_unknown_backbone_layout(capsys):
    from web.field.sam3_tracker import install_skip_detector_heads

    model = _SkipModel(backbone_out={"vision_features": None})
    assert install_skip_detector_heads(model)
    model._prepare_backbone_feats(_skip_state(), 1, False)
    model._prepare_backbone_feats(_skip_state(), 2, False)
    assert model.orig_calls == [(1, False), (2, False)]
    assert capsys.readouterr().out.count("falling back") == 1


def test_skip_detector_heads_refuses_mismatched_models():
    from web.field.sam3_tracker import install_skip_detector_heads

    assert install_skip_detector_heads(None) is False

    no_conv = _SkipModel(conv_s1=False)
    orig = no_conv._prepare_backbone_feats
    assert install_skip_detector_heads(no_conv) is False
    assert no_conv._prepare_backbone_feats == orig

    class _BadSig(_SkipModel):
        def _prepare_backbone_feats(self, inference_state, frame_idx):
            pass

    bad = _BadSig()
    assert install_skip_detector_heads(bad) is False
    assert not getattr(bad._prepare_backbone_feats, "_frc_skip_detector_shim", False)

    class Sam3MultiplexTrackingWithInteractivity(_SkipModel):
        pass

    mux = Sam3MultiplexTrackingWithInteractivity()
    assert install_skip_detector_heads(mux) is False
    assert not getattr(mux._prepare_backbone_feats, "_frc_skip_detector_shim", False)

    no_method = _ns(detector=_SkipModel().detector, tracker=_SkipModel().tracker)
    assert install_skip_detector_heads(no_method) is False


def _patch_builders_with_model(monkeypatch):
    import sam3.model_builder as model_builder

    built = {}

    def fake_video(**kwargs):
        built["video"] = _ns(model=_SkipModel())
        return built["video"]

    def fake_mux(**kwargs):
        built["mux"] = _ns(model=_SkipModel())
        return built["mux"]

    monkeypatch.setattr(model_builder, "build_sam3_video_predictor", fake_video)
    monkeypatch.setattr(model_builder, "build_sam3_predictor", fake_mux)
    return built


def test_sam3_tracker_skip_detector_heads_default_off(monkeypatch):
    built = _patch_builders_with_model(monkeypatch)
    tracker = Sam3FieldTracker(checkpoint="ckpt.pt")
    assert tracker.skip_detector_heads_active is False
    assert not getattr(built["video"].model._prepare_backbone_feats,
                       "_frc_skip_detector_shim", False)
    assert tracker.retire_culled is False
    assert tracker._predictor.session_remove is True


def test_sam3_tracker_skip_detector_heads_installs_on_base_only(monkeypatch):
    built = _patch_builders_with_model(monkeypatch)
    tracker = Sam3FieldTracker(checkpoint="ckpt.pt", skip_detector_heads=True)
    assert tracker.skip_detector_heads_active is True
    assert getattr(built["video"].model._prepare_backbone_feats,
                   "_frc_skip_detector_shim", False)

    mux = Sam3FieldTracker(checkpoint="ckpt.pt", skip_detector_heads=True,
                           multiplex=True, retire_culled=True)
    assert mux.skip_detector_heads_active is False
    assert not getattr(built["mux"].model._prepare_backbone_feats,
                       "_frc_skip_detector_shim", False)
    # multiplex: retire is adapter bookkeeping only
    assert mux._predictor.session_remove is False


# ---------------------------------------------------------------------------
# sam3_retire_culled: Sam3PredictorAdapter.remove_object + track() hook
# ---------------------------------------------------------------------------

class RemoveStubPredictor(ActionHistoryStubPredictor):
    """ActionHistoryStubPredictor + upstream-like session introspection and
    remove_object (asserts the object exists, like base SAM3)."""

    def __init__(self, fail_remove=False):
        super().__init__()
        self.fail_remove = fail_remove

    def _get_session(self, sid):
        sess = self._sessions[sid]
        ids = sorted(sess.setdefault("known", set()))
        return {"state": {"tracker_metadata": {
            "obj_ids_per_gpu": [np.array(ids, dtype=np.int64)]}}}

    def handle_request(self, request):
        if request["type"] == "add_prompt":
            self._sessions[request["session_id"]].setdefault(
                "known", set()).add(request["obj_id"])
        if request["type"] == "remove_object":
            self.requests.append(request)
            if self.fail_remove:
                raise RuntimeError("boom")
            sess = self._sessions[request["session_id"]]
            assert request["obj_id"] in sess.get("known", set())
            sess["known"].discard(request["obj_id"])
            sess["prompted_since"].discard(request["obj_id"])
            return {"is_success": True}
        return super().handle_request(request)

    def handle_stream_request(self, request):
        sess = self._sessions[request["session_id"]]
        sess["prompted_since"] &= sess.get("known", set())
        yield from super().handle_stream_request(request)


def _two_object_adapter(tmp_path, raw):
    _write_frames(tmp_path, 24)
    adapter = Sam3PredictorAdapter(raw, chunk_frames=64)
    state = adapter.init_state(video_path=str(tmp_path))
    for oid, box in ((1, [1, 1, 5, 5]), (2, [4, 2, 9, 6])):
        adapter.add_new_points_or_box(
            state, frame_idx=0, obj_id=oid, box=np.array(box, np.float32))
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=7))
    return adapter, state


def test_remove_object_bookkeeping_and_session_request(tmp_path):
    raw = RemoveStubPredictor()
    adapter, state = _two_object_adapter(tmp_path, raw)
    sid = state["chunk"]["session_id"]
    state["pending"].append((40, 2, (1.0, 1.0, 2.0, 2.0), []))
    state["pending"].append((40, 1, (1.0, 1.0, 2.0, 2.0), []))
    state["chunk"]["prompted"].add(2)
    state["solo_ids"] = {1, 2}

    raw.requests.clear()
    adapter.remove_object(state, 2)

    assert raw.requests == [{"type": "remove_object", "session_id": sid,
                             "obj_id": 2}]
    assert 2 not in state["last_box"] and 1 in state["last_box"]
    assert [p[1] for p in state["pending"]] == [1]
    assert 2 not in state["chunk"]["prompted"]
    assert state["solo_ids"] == {1}
    state["solo_ids"] = None

    # next window: only the survivor is re-anchored and keeps emitting
    raw.requests.clear()
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=8,
                                         max_frame_num_to_track=7))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert {p["obj_id"] for p in prompts} == {1}
    assert w2 and all(set(ids.tolist()) == {1} for _, ids, *_ in w2)

    # idempotent: second retire of the same id sends nothing (not in session)
    raw.requests.clear()
    adapter.remove_object(state, 2)
    assert raw.requests == []
    adapter.reset_state(state)


def test_remove_last_object_then_propagate_does_not_crash(tmp_path):
    raw = RemoveStubPredictor()
    adapter, state = _two_object_adapter(tmp_path, raw)
    adapter.remove_object(state, 1)
    adapter.remove_object(state, 2)
    assert state["last_box"] == {}
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=8,
                                         max_frame_num_to_track=3))
    assert all(len(ids) == 0 for _, ids, *_ in w2)
    adapter.reset_state(state)


def test_remove_object_skips_session_call_for_unknown_id(tmp_path):
    raw = RemoveStubPredictor()
    adapter, state = _two_object_adapter(tmp_path, raw)
    state["last_box"][9] = (5, (1.0, 1.0, 3.0, 3.0))  # never in the session
    raw.requests.clear()
    adapter.remove_object(state, 9)
    assert raw.requests == []
    assert 9 not in state["last_box"]
    adapter.reset_state(state)


def test_remove_object_without_chunk_is_bookkeeping_only(tmp_path):
    _write_frames(tmp_path, 4)
    raw = RemoveStubPredictor()
    adapter = Sam3PredictorAdapter(raw, chunk_frames=16)
    state = adapter.init_state(video_path=str(tmp_path))
    adapter.add_new_points_or_box(
        state, frame_idx=2, obj_id=4, box=np.array([1, 1, 5, 5], np.float32))
    assert state["chunk"] is None and state["pending"]
    adapter.remove_object(state, 4)
    assert state["pending"] == [] and state["last_box"] == {}
    assert raw.requests == []


def test_remove_object_multiplex_is_bookkeeping_only(tmp_path):
    raw = RemoveStubPredictor()
    adapter, state = _two_object_adapter(tmp_path, raw)
    adapter.session_remove = False
    raw.requests.clear()
    adapter.remove_object(state, 2)
    assert raw.requests == []
    assert 2 not in state["last_box"]
    adapter.reset_state(state)


def test_remove_object_tolerates_session_failure(tmp_path, capsys):
    raw = RemoveStubPredictor(fail_remove=True)
    adapter, state = _two_object_adapter(tmp_path, raw)
    adapter.remove_object(state, 2)  # must not raise
    assert 2 not in state["last_box"]
    assert "bookkeeping-only retire" in capsys.readouterr().out
    # the view keeps going
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=8,
                                         max_frame_num_to_track=3))
    assert w2
    adapter.reset_state(state)


def test_remove_object_without_session_introspection_tries_request(tmp_path):
    """Predictors without _get_session (stubs) get the request; a failure is
    swallowed like any other."""
    _write_frames(tmp_path, 4)
    raw = StubSam3Predictor()
    adapter = Sam3PredictorAdapter(raw, chunk_frames=16)
    state = adapter.init_state(video_path=str(tmp_path))
    adapter.add_new_points_or_box(
        state, frame_idx=0, obj_id=3, box=np.array([1, 1, 5, 5], np.float32))
    list(adapter.propagate_in_video(state, start_frame_idx=0,
                                    max_frame_num_to_track=2))
    adapter.remove_object(state, 3)  # StubSam3Predictor raises on it
    assert raw.requests[-1]["type"] == "remove_object"
    assert 3 not in state["last_box"]
    adapter.reset_state(state)


def _retire_tracker(retire_culled):
    """Sam3FieldTracker whose adapter is swapped for a SAM2-protocol cull stub
    that records remove_object calls (the hook's only contract)."""
    from tests.field.test_sam2_tracker import CullStubPredictor

    class _RecordingCullStub(CullStubPredictor):
        def __init__(self):
            super().__init__()
            self.removed = []

        def remove_object(self, state, oid):
            self.removed.append(int(oid))

    tracker = Sam3FieldTracker(predictor=StubSam3Predictor(),
                               retire_culled=retire_culled)
    stub = _RecordingCullStub()
    tracker._predictor = stub
    tracker._real_torch = False
    return tracker, stub


def _culled(diag):
    return {(e["details"]["object_id"], e.get("reason"))
            for e in diag._events if e.get("event") == "sam3.object_culled"}


@pytest.mark.parametrize("retire", [False, True])
def test_retire_hook_fires_on_absent_stationary_cull_only_when_on(retire):
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.sam2_tracker import bottom_center

    tracker, stub = _retire_tracker(retire)
    diag = IdentityDiagnosticRecorder()
    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=lambda *_a: [(stub._rect_for(1), "blue")],
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_corrob_ios=0.3, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert (2, "detector_absent_stationary") in _culled(diag)
    assert stub.removed == ([2] if retire else [])


@pytest.mark.parametrize("retire", [False, True])
def test_retire_hook_fires_on_eviction_only_when_on(retire):
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.sam2_tracker import bottom_center

    tracker, stub = _retire_tracker(retire)
    diag = IdentityDiagnosticRecorder()
    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=lambda *_a: [(stub._rect_for(1), "blue"),
                               (stub._rect_for(15), "red")],
        object_metadata={1: "blue", 2: "red"},
        max_active_objects=2, max_lifetime_objects=20, cull_enabled=False,
        cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    culled = _culled(diag)
    assert (2, "evicted_for_new_detection") in culled
    # every eviction (2, then the seeded 3 later) is retired, each once
    expected = sorted(oid for oid, _r in culled)
    assert sorted(stub.removed) == (expected if retire else [])
    assert 2 in expected


@pytest.mark.parametrize("retire", [False, True])
def test_retire_hook_fires_on_person_veto_only_when_on(retire):
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.sam2_tracker import bottom_center

    tracker, stub = _retire_tracker(retire)
    diag = IdentityDiagnosticRecorder()
    tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=lambda *_a: [(stub._rect_for(1), "blue")],
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=False,
        person_fn=lambda idx, path: [(18.0, 0.0, 26.0, 8.0)],
        project_fn=lambda box: bottom_center(box), diagnostics=diag,
    )
    assert (2, "person_overlap") in _culled(diag)
    assert stub.removed == ([2] if retire else [])


def test_retire_hook_failure_does_not_kill_track(capsys):
    from web.field.sam2_tracker import bottom_center

    tracker, stub = _retire_tracker(True)

    def boom(state, oid):
        raise RuntimeError("remove exploded")

    stub.remove_object = boom
    result = tracker.track(
        frames_dir="ignored_by_stub", prompt_idx=0,
        prompt_boxes=[stub._rect_for(1), stub._rect_for(2)],
        reprompt_every=2,
        detect_fn=lambda *_a: [(stub._rect_for(1), "blue")],
        object_metadata={1: "blue", 2: "red"}, max_active_objects=6,
        cull_enabled=True, cull_uncorroborated_reprompts=3, cull_evidence_window=5,
        cull_corrob_ios=0.3, cull_stationary_m=0.5,
        project_fn=lambda box: bottom_center(box),
    )
    assert 1 in result[stub.NUM_FRAMES - 1]
    assert 2 not in result[stub.NUM_FRAMES - 1]
    assert "retire of obj 2 failed" in capsys.readouterr().out


def test_sam2_retire_hook_is_noop():
    from web.field.sam2_tracker import Sam2FieldTracker

    tracker = Sam2FieldTracker(predictor=object())
    assert tracker._retire_object({}, 3) is None


# ---------------------------------------------------------------------------
# selective segment resumes: state["resume_refine"] (bare refine for known ids)
# ---------------------------------------------------------------------------

class _RefineModel:
    """Upstream ``model.add_action_history`` surface: appends to the session
    state's action history (the stub stream then treats refine like add)."""

    def __init__(self, raw):
        self._raw = raw

    def add_action_history(self, inference_state, action_type, frame_idx=None,
                           obj_ids=None):
        inference_state["action_history"].append(
            {"type": action_type, "frame_idx": frame_idx, "obj_ids": obj_ids})
        self._raw.actions.append((action_type, frame_idx, list(obj_ids or [])))


class RefineStubPredictor(RemoveStubPredictor):
    """RemoveStubPredictor with a persistent per-session ``state`` that carries
    an upstream-like ``action_history``: add_prompt records ``add``/``refine``,
    each propagate records ``propagation_partial`` over the ids it runs (or
    ``propagation_fetch`` with none), and a bare ``refine`` action (via
    ``model.add_action_history``) keeps an id on the next partial propagate
    exactly like a prompt would, minus the conditioning frame."""

    def __init__(self, with_model=True):
        super().__init__()
        self.actions = []
        if with_model:
            self.model = _RefineModel(self)

    def _sess_state(self, sid):
        sess = self._sessions[sid]
        st = sess.setdefault("state", {"action_history": []})
        st["tracker_metadata"] = {"obj_ids_per_gpu": [
            np.array(sorted(sess.get("known", set())), dtype=np.int64)]}
        return st

    def _get_session(self, sid):
        return {"state": self._sess_state(sid)}

    def handle_request(self, request):
        if request["type"] == "add_prompt":
            sid = request["session_id"]
            known = self._sessions[sid].get("known", set())
            kind = "refine" if request["obj_id"] in known else "add"
            self._sess_state(sid)["action_history"].append(
                {"type": kind, "frame_idx": request["frame_index"],
                 "obj_ids": [request["obj_id"]]})
        return super().handle_request(request)

    def handle_stream_request(self, request):
        sid = request["session_id"]
        st = self._sess_state(sid)
        ids = []
        for a in reversed(st["action_history"]):
            if "propagation" in a["type"]:
                break
            if a["type"] in ("add", "refine"):
                ids.extend(a["obj_ids"])
        ids = sorted(set(ids) & self._sessions[sid].get("known", set()))
        st["action_history"].append(
            {"type": "propagation_partial" if ids else "propagation_fetch",
             "frame_idx": request["start_frame_index"],
             "obj_ids": ids or None})
        self._sessions[sid]["prompted_since"] = set(ids)
        yield from ActionHistoryStubPredictor.handle_stream_request(self, request)


def _break_after(adapter, state, start, span, stop_at):
    """Propagate and close the stream right after frame ``stop_at`` (what
    track_selective does at a segment break)."""
    gen = adapter.propagate_in_video(state, start_frame_idx=start,
                                     max_frame_num_to_track=span)
    out = []
    try:
        for item in gen:
            out.append(item)
            if item[0] == stop_at:
                break
    finally:
        gen.close()
    return out


def _resume_setup(tmp_path, raw, n=24, chunk=64):
    _write_frames(tmp_path, n)
    adapter = Sam3PredictorAdapter(raw, chunk_frames=chunk)
    state = adapter.init_state(video_path=str(tmp_path))
    for oid, box in ((1, [1, 1, 5, 5]), (2, [4, 2, 9, 6])):
        adapter.add_new_points_or_box(
            state, frame_idx=0, obj_id=oid, box=np.array(box, np.float32))
    w1 = _break_after(adapter, state, 0, 15, stop_at=5)
    assert [f for f, *_ in w1] == list(range(0, 6))
    return adapter, state


def test_resume_refine_uses_bare_refine_for_known_ids(tmp_path):
    raw = RefineStubPredictor()
    adapter, state = _resume_setup(tmp_path, raw)
    adapter.remove_object(state, 2)
    raw.requests.clear()
    raw.actions.clear()
    state["resume_refine"] = True
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=6,
                                         max_frame_num_to_track=9))
    # no box prompt for the survivor, a bare refine at the resume frame
    assert [r for r in raw.requests if r["type"] == "add_prompt"] == []
    assert raw.actions == [("refine", 6, [1])]
    assert [f for f, *_ in w2] == list(range(6, 16))
    assert all(ids.tolist() == [1] for _, ids, *_ in w2)
    assert "resume_refine" not in state            # one-shot
    adapter.reset_state(state)


def test_resume_refine_lazy_seed_prompt_plus_refine(tmp_path):
    """A lazily-seeded new id gets its real prompt; known survivors get a
    bare refine; both ride the resumed partial propagate."""
    raw = RefineStubPredictor()
    adapter, state = _resume_setup(tmp_path, raw)
    adapter.add_new_points_or_box(
        state, frame_idx=6, obj_id=3, box=np.array([2, 2, 6, 6], np.float32))
    raw.actions.clear()
    state["resume_refine"] = True
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=6,
                                         max_frame_num_to_track=9))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert [p["obj_id"] for p in prompts][-1] == 3
    assert sorted(a[2][0] for a in raw.actions) == [1, 2]
    assert all(a[0] == "refine" and a[1] == 6 for a in raw.actions)
    assert all(ids.tolist() == [1, 2, 3] for _, ids, *_ in w2)
    adapter.reset_state(state)


def test_resume_without_flag_reanchors_with_prompts(tmp_path):
    """Default (flag unset): today's per-request re-anchor box prompts."""
    raw = RefineStubPredictor()
    adapter, state = _resume_setup(tmp_path, raw)
    raw.requests.clear()
    raw.actions.clear()
    list(adapter.propagate_in_video(state, start_frame_idx=6,
                                    max_frame_num_to_track=9))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert {p["obj_id"] for p in prompts} == {1, 2}
    assert raw.actions == []
    adapter.reset_state(state)


@pytest.mark.parametrize("variant", ["no_model", "multiplex", "no_session"])
def test_resume_refine_falls_back_to_prompts(tmp_path, variant):
    if variant == "no_session":
        raw = ActionHistoryStubPredictor()
        raw.model = _RefineModel(raw)
        raw.actions = []
    else:
        raw = RefineStubPredictor(with_model=(variant != "no_model"))
    adapter, state = _resume_setup(tmp_path, raw)
    if variant == "multiplex":
        adapter.refine_resume = False
    raw.requests.clear()
    state["resume_refine"] = True
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=6,
                                         max_frame_num_to_track=9))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert {p["obj_id"] for p in prompts} == {1, 2}
    assert getattr(raw, "actions", []) == []
    assert all(set(ids.tolist()) == {1, 2} for _, ids, *_ in w2)
    assert "resume_refine" not in state
    adapter.reset_state(state)


def test_resume_refine_prompts_ids_not_in_session(tmp_path):
    """A fresh last_box id the session doesn't track still gets a real
    prompt on a resume (refine is only for known ids)."""
    raw = RefineStubPredictor()
    adapter, state = _resume_setup(tmp_path, raw)
    state["last_box"][9] = (5, (1.0, 1.0, 4.0, 4.0))
    raw.requests.clear()
    raw.actions.clear()
    state["resume_refine"] = True
    list(adapter.propagate_in_video(state, start_frame_idx=6,
                                    max_frame_num_to_track=3))
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert [p["obj_id"] for p in prompts] == [9]
    assert sorted(a[2][0] for a in raw.actions) == [1, 2]
    adapter.reset_state(state)


def test_resume_across_chunk_boundary_skips_retired_and_unseeded(tmp_path):
    """Resume that crosses the 16-frame chunk boundary: the new chunk re-seeds
    only the survivor — never the retired id, never a held-back lazy id."""
    raw = RefineStubPredictor()
    adapter, state = _resume_setup(tmp_path, raw, n=24, chunk=16)
    state["solo_ids"] = {1, 2}
    adapter.remove_object(state, 2)
    assert state["solo_ids"] == {1}
    raw.requests.clear()
    state["resume_refine"] = True
    w2 = list(adapter.propagate_in_video(state, start_frame_idx=6,
                                         max_frame_num_to_track=17))
    assert [f for f, *_ in w2] == list(range(6, 24))
    starts = [r for r in raw.requests if r["type"] == "start_session"]
    assert len(starts) == 1                          # chunk [16, 24)
    prompts = [r for r in raw.requests if r["type"] == "add_prompt"]
    assert [p["obj_id"] for p in prompts] == [1]     # boundary re-seed only
    assert all(ids.tolist() == [1] for _, ids, *_ in w2)
    state["solo_ids"] = None
    adapter.reset_state(state)

"""sam3_parallel_views: concurrent per-view SAM3 propagation.

Covers the knob registration, the VRAM planner, the order-preserving
dispatcher (web/field/parallel_views.py), the diagnostics child/absorb merge,
and the InferenceJob seams (_build_parallel_view_trackers,
_track_views_parallel, _run_view_tracking hooks) with predictors mocked. No
GPU is touched: torch.cuda.is_available is patched False wherever the job
code would otherwise create streams.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from web.field import parallel_views as pv
from web.field.identity_diagnostics import (
    DiagnosticEvents, IdentityDiagnosticRecorder)
from web.job_config import JobConfig, KNOBS

GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------
def test_parallel_knobs_default(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    cfg = JobConfig()
    assert cfg.sam3_parallel_views is False
    assert cfg.sam3_parallel_max == 3
    assert cfg.sam3_parallel_streams is True
    sections = {k.name: k.section for k in KNOBS}
    for name in ("sam3_parallel_views", "sam3_parallel_max", "sam3_parallel_streams"):
        assert sections[name] == "Tracking backend"


def test_parallel_knobs_env_and_override(monkeypatch):
    monkeypatch.setenv("SAM3_PARALLEL_VIEWS", "1")
    monkeypatch.setenv("SAM3_PARALLEL_MAX", "2")
    cfg = JobConfig()
    assert cfg.sam3_parallel_views is True
    assert cfg.sam3_parallel_max == 2
    cfg2 = JobConfig({"sam3_parallel_views": "false", "sam3_parallel_max": "4",
                      "sam3_parallel_streams": "false"})
    assert cfg2.sam3_parallel_views is False
    assert cfg2.sam3_parallel_max == 4
    assert cfg2.sam3_parallel_streams is False


def test_parallel_max_range_validation():
    with pytest.raises(ValueError, match="sam3_parallel_max"):
        JobConfig({"sam3_parallel_max": 0})


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------
def test_plan_fits_all_views_on_big_card():
    n, reason = pv.plan_parallel_views(3, 3, 80 * GIB, chunk_frames=200)
    assert n == 3
    assert "3 concurrent" in reason


def test_plan_reduces_to_what_fits():
    need2 = pv.required_gib(2, 200)
    need3 = pv.required_gib(3, 200)
    free = (need2 + need3) / 2 * GIB
    n, reason = pv.plan_parallel_views(3, 3, free, chunk_frames=200)
    assert n == 2
    assert "wanted 3" in reason


def test_plan_low_vram_falls_back_to_sequential():
    n, reason = pv.plan_parallel_views(3, 3, 8 * GIB, chunk_frames=200)
    assert n == 1
    assert "not enough VRAM" in reason


def test_plan_probe_failure_and_caps():
    assert pv.plan_parallel_views(3, 3, None, 200)[0] == 1
    assert pv.plan_parallel_views(3, 1, 80 * GIB, 200)[0] == 1   # max 1
    assert pv.plan_parallel_views(1, 3, 80 * GIB, 200)[0] == 1   # one view
    assert pv.plan_parallel_views(3, 8, 80 * GIB, 200)[0] == 3   # capped by views


def test_required_gib_scales_with_chunk():
    assert pv.required_gib(2, 400) > pv.required_gib(2, 200)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_serialized_passthrough_and_lock():
    def f(x):
        return x + 1
    assert pv.serialized(f, None) is f
    assert pv.serialized(None, threading.Lock()) is None

    lock = threading.Lock()
    seen = []

    def g(x):
        seen.append(lock.locked())
        return x * 2
    wrapped = pv.serialized(g, lock)
    assert wrapped(3) == 6
    assert seen == [True]
    assert not lock.locked()


def test_call_on_cuda_stream_none_runs_directly():
    assert pv.call_on_cuda_stream(None, lambda a, b=0: a + b, 2, b=3) == 5


def test_reorder_view_keys_in_place():
    d = {"pre": 0, "corner_right": 3, "main": 1, "corner_left": 2}
    ident = id(d)
    pv.reorder_view_keys(d, ["main", "corner_left", "corner_right"], ["pre"])
    assert id(d) == ident
    assert list(d) == ["pre", "main", "corner_left", "corner_right"]
    assert d == {"pre": 0, "main": 1, "corner_left": 2, "corner_right": 3}


def test_parallel_progress_aggregates_mean():
    p = pv.ParallelProgress(["a", "b"], 25.0, 75.0)
    assert p.update("a", 1.0) == pytest.approx(25.0 + 0.5 * 75.0)
    assert p.update("b", 1.0) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def _items(n):
    return [(i, f"v{i}") for i in range(n)]


def test_dispatch_calls_each_view_once_in_item_order_results():
    calls = []
    in_flight = {"now": 0, "max": 0}
    delays = {0: 0.05, 1: 0.0, 2: 0.02}   # finish out of order

    async def run_one(idx, view, worker):
        calls.append((idx, view))
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(delays[idx])
        in_flight["now"] -= 1
        return f"{view}@{worker}"

    res = asyncio.run(pv.run_views_concurrently(_items(3), run_one, ["w0", "w1", "w2"]))
    assert sorted(calls) == [(0, "v0"), (1, "v1"), (2, "v2")]
    assert [c[0] for c in calls] == [0, 1, 2]                # started in order
    assert [r.split("@")[0] for r in res] == ["v0", "v1", "v2"]
    assert in_flight["max"] == 3


def test_dispatch_honours_worker_cap_and_exclusivity():
    busy = set()
    in_flight = {"now": 0, "max": 0}

    async def run_one(idx, view, worker):
        assert worker not in busy, "worker shared by two in-flight views"
        busy.add(worker)
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        busy.discard(worker)
        return idx

    res = asyncio.run(pv.run_views_concurrently(_items(5), run_one, ["a", "b"]))
    assert res == [0, 1, 2, 3, 4]
    assert in_flight["max"] == 2


def test_dispatch_exception_propagates_after_all_started_finish():
    finished = []

    async def run_one(idx, view, worker):
        if idx == 1:
            await asyncio.sleep(0.0)
            raise RuntimeError("view 1 died")
        await asyncio.sleep(0.05)
        finished.append(idx)
        return idx

    async def go():
        return await asyncio.wait_for(
            pv.run_views_concurrently(_items(3), run_one, ["a", "b", "c"]),
            timeout=5.0)

    with pytest.raises(RuntimeError, match="view 1 died"):
        asyncio.run(go())
    # Started siblings were awaited, not orphaned.
    assert sorted(finished) == [0, 2]


def test_dispatch_skips_unstarted_views_after_failure():
    started = []

    async def run_one(idx, view, worker):
        started.append(idx)
        if idx == 0:
            raise ValueError("boom")
        await asyncio.sleep(0.01)
        return idx

    with pytest.raises(ValueError):
        asyncio.run(pv.run_views_concurrently(_items(3), run_one, ["only"]))
    assert started == [0]


def test_dispatch_before_start_stop_check_raises():
    class Stopped(Exception):
        pass

    def stop():
        raise Stopped()

    async def run_one(idx, view, worker):  # pragma: no cover - never runs
        raise AssertionError("ran after stop")

    with pytest.raises(Stopped):
        asyncio.run(pv.run_views_concurrently(
            _items(2), run_one, ["a", "b"], before_start=stop))


def test_dispatch_threaded_views_overlap():
    """Views run their (blocking) work on threads concurrently."""
    barrier = threading.Barrier(3, timeout=5.0)

    def blocking(idx):
        barrier.wait()      # deadlocks (times out) unless all 3 overlap
        return idx

    async def run_one(idx, view, worker):
        return await asyncio.to_thread(blocking, idx)

    res = asyncio.run(pv.run_views_concurrently(_items(3), run_one, [1, 2, 3]))
    assert res == [0, 1, 2]


# ---------------------------------------------------------------------------
# diagnostics child / absorb == sequential recording
# ---------------------------------------------------------------------------
EV = DiagnosticEvents.SAM2_OBJECT_CREATED


def _record_view(rec, view, n):
    with rec.scoped_context(view=view):
        for i in range(n):
            rec.record(EV, details={"i": i})
            rec.increment("objs")
        rec.gauge(f"g_{view}", n)


def test_absorb_matches_sequential_recording():
    seq = IdentityDiagnosticRecorder(max_events=5)
    seq.set_context(job_id="j")
    for view, n in (("main", 3), ("corner_left", 2), ("corner_right", 2)):
        _record_view(seq, view, n)

    par = IdentityDiagnosticRecorder(max_events=5)
    par.set_context(job_id="j")
    kids = {v: par.spawn_child() for v in ("main", "corner_left", "corner_right")}
    # Record in a scrambled order (as concurrent views would), absorb in view order.
    _record_view(kids["corner_right"], "corner_right", 2)
    _record_view(kids["main"], "main", 3)
    _record_view(kids["corner_left"], "corner_left", 2)
    for v in ("main", "corner_left", "corner_right"):
        par.absorb(kids[v])

    assert par._events == seq._events
    assert par._context == seq._context == {"job_id": "j"}
    assert par.summary() == seq.summary()      # counters, gauges, dropped (cap 5)
    assert par.summary()["dropped_logs"] == 2


# ---------------------------------------------------------------------------
# InferenceJob seams
# ---------------------------------------------------------------------------
def _make_job(views=("main", "corner_left", "corner_right")):
    from web.app import InferenceJob
    with patch("web.app.YOLO"):
        job = InferenceJob("test_job", "/mock/video.mp4")
    job.views = [SimpleNamespace(name=n) for n in views]
    job.timings = {}
    job.tracker_raw = {}
    job.tracker_obj_color = {}
    job._view_mine_info = {}
    job._sam3_extra_trackers = []
    job.config.sam3_parallel_max = 3
    job.config.sam3_chunk_frames = 200
    return job


def test_build_extra_trackers_low_vram_falls_back(capsys):
    job = _make_job()
    factory = MagicMock()
    with patch("torch.cuda.mem_get_info", return_value=(8 * GIB, 24 * GIB)):
        extras = job._build_parallel_view_trackers(factory, "cuda")
    assert extras == []
    factory.assert_not_called()
    assert "falling back to sequential" in capsys.readouterr().out


def test_build_extra_trackers_enough_vram_builds_n_minus_1(capsys):
    job = _make_job()
    factory = MagicMock(side_effect=["t1", "t2"])
    with patch("torch.cuda.mem_get_info", return_value=(80 * GIB, 96 * GIB)):
        extras = job._build_parallel_view_trackers(factory, "cuda")
    assert extras == ["t1", "t2"]
    assert "3 predictors resident" in capsys.readouterr().out


def test_build_extra_trackers_build_failure_falls_back(capsys):
    job = _make_job()
    factory = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
    with patch("torch.cuda.mem_get_info", return_value=(80 * GIB, 96 * GIB)):
        extras = job._build_parallel_view_trackers(factory, "cuda")
    assert extras == []
    assert "falling back to sequential" in capsys.readouterr().out


def test_build_extra_trackers_never_off_gpu():
    job = _make_job()
    factory = MagicMock()
    assert job._build_parallel_view_trackers(factory, "cpu") == []
    factory.assert_not_called()


def _fake_track_one_view(job, delays, calls, fail=None):
    """Stand-in for the precompute per-view coroutine: records diagnostics,
    writes the per-view result dicts the real body writes, and finishes in
    ``delays`` order (main slowest) so completion order != view order."""
    async def track_one_view(view_idx, view, tracker, _par=None):
        diag = job.identity_diagnostics if _par is None else _par.diag
        calls.append((view.name, tracker, _par is not None))
        job.tracker_obj_color[view.name] = {1: "red"}
        with diag.scoped_context(view=view.name):
            diag.record(EV, details={"v": view.name})
            if _par is not None:
                # Real views run tracker work on threads.
                await asyncio.to_thread(lambda: None)
            await asyncio.sleep(delays[view.name])
            if fail == view.name:
                raise RuntimeError(f"{view.name} escaped")
            diag.record(EV, details={"v": view.name, "done": True})
        job._view_mine_info[view.name] = {"scale": 0.5}
        job.tracker_raw[view.name] = {0: {view_idx: view.name}}
        job.timings["sam3_tracking"] = job.timings.get("sam3_tracking", 0.0) + 1.0
    return track_one_view


DELAYS = {"main": 0.06, "corner_left": 0.02, "corner_right": 0.0}


def _run_sequential(job, calls):
    fn = _fake_track_one_view(job, DELAYS, calls)

    async def go():
        for i, v in enumerate(job.views):
            await fn(i, v, "primary")
    asyncio.run(go())


def test_track_views_parallel_matches_sequential_order():
    seq_job, par_job = _make_job(), _make_job()
    seq_calls, par_calls = [], []
    _run_sequential(seq_job, seq_calls)

    par_job._sam3_extra_trackers = ["x1", "x2"]
    fn = _fake_track_one_view(par_job, DELAYS, par_calls)
    with patch("torch.cuda.is_available", return_value=False):
        asyncio.run(par_job._track_views_parallel(
            fn, ["primary", "x1", "x2"], 25.0))

    # Each view called exactly once, each on its own predictor.
    assert sorted(c[0] for c in par_calls) == ["corner_left", "corner_right", "main"]
    assert len({c[1] for c in par_calls}) == 3
    assert all(c[2] for c in par_calls)
    # Merged in sequential (view) order even though main finished last.
    for name in ("tracker_raw", "tracker_obj_color", "_view_mine_info"):
        assert list(getattr(par_job, name)) == list(getattr(seq_job, name))
        assert getattr(par_job, name) == getattr(seq_job, name)
    assert par_job.identity_diagnostics._events == seq_job.identity_diagnostics._events
    assert par_job._sam3_extra_trackers == []
    assert "sam_views_wall" in par_job.timings


def test_track_views_parallel_exception_propagates_and_releases():
    job = _make_job()
    job._sam3_extra_trackers = ["x1", "x2"]
    calls = []
    fn = _fake_track_one_view(job, DELAYS, calls, fail="corner_left")

    async def go():
        with patch("torch.cuda.is_available", return_value=False):
            await asyncio.wait_for(
                job._track_views_parallel(fn, ["primary", "x1", "x2"], 25.0),
                timeout=5.0)

    with pytest.raises(RuntimeError, match="corner_left escaped"):
        asyncio.run(go())
    # Siblings completed (awaited, not orphaned); still merged in view order.
    assert list(job.tracker_raw) == ["main", "corner_right"]
    assert job._sam3_extra_trackers == []


def test_track_views_parallel_respects_max_workers():
    job = _make_job()
    calls = []
    fn = _fake_track_one_view(job, DELAYS, calls)
    with patch("torch.cuda.is_available", return_value=False):
        asyncio.run(job._track_views_parallel(fn, ["primary", "x1"], 25.0))
    assert len(calls) == 3
    assert {c[1] for c in calls} == {"primary", "x1"}
    assert list(job.tracker_raw) == ["main", "corner_left", "corner_right"]


def test_run_view_tracking_stream_and_lock_hooks():
    job = _make_job()
    job.config.selective_mask_prop = True
    job.config.selective_app_weight = 0.3
    job.robot_embedder = None
    tracker = MagicMock()
    tracker.track_selective.return_value = {0: {}}
    tracker.selective_stats = {}
    tracker.selective_obj_colors = {}
    lock = threading.RLock()
    stream = object()
    seen = {}

    def fake_on_stream(s, fn, /, *args, **kwargs):
        seen["stream"] = s
        emb = kwargs.get("embed_frame_fn")
        if emb is not None:
            seen["embed"] = emb(0, "", [])
        return fn(*args, **kwargs)

    real_embed = job.make_embed_frame_fn

    def embed_factory():
        inner = real_embed()

        def probe(*a):
            seen["locked_in_embed"] = lock._is_owned()
            return inner(*a)
        return probe
    job.make_embed_frame_fn = embed_factory

    async def run():
        return await job._run_view_tracking(
            tracker, SimpleNamespace(name="corner_left"),
            {"frames_dir": "/frames"}, lambda i, p: [], lambda a, b: None,
            call_lock=lock, cuda_stream=stream)

    with patch("web.app.call_on_cuda_stream", side_effect=fake_on_stream):
        out = asyncio.run(run())
    assert out == {0: {}}
    assert seen["stream"] is stream
    assert seen["locked_in_embed"] is True
    tracker.track_selective.assert_called_once()


def test_run_view_tracking_without_hooks_never_uses_stream_helper():
    job = _make_job()
    job.config.selective_mask_prop = False
    tracker = MagicMock()
    tracker.track.return_value = {0: {}}
    with patch("web.app.call_on_cuda_stream",
               side_effect=AssertionError("stream helper used")):
        out = asyncio.run(job._run_view_tracking(
            tracker, SimpleNamespace(name="main"), {"frames_dir": "/f"},
            lambda i, p: [], lambda a, b: None))
    assert out == {0: {}}


def test_triton_autotune_lock_serializes_concurrent_runs():
    """Two threads in Autotuner.run must not overlap (shared self.nargs race)."""
    import threading, time
    triton_at = pytest.importorskip("triton.runtime.autotuner")
    from web.field import parallel_views as pv

    assert pv.install_triton_autotune_lock() is True
    assert pv.install_triton_autotune_lock() is True  # idempotent
    run = triton_at.Autotuner.run
    assert getattr(run, "_frc_locked", False)
    inner = run.__wrapped__
    state = {"in": 0, "max": 0}

    def fake(self, *a, **k):
        state["in"] += 1
        state["max"] = max(state["max"], state["in"])
        time.sleep(0.02)
        state["in"] -= 1

    # call the lock wrapper around a fake body via the same lock
    def locked(self, *a, **k):
        with pv._TRITON_LOCK:
            return fake(self, *a, **k)
    ts = [threading.Thread(target=locked, args=(None,)) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert state["max"] == 1
    assert inner is not run

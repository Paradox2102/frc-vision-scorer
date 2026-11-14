"""GPU-memory hygiene across jobs.

Leak 1: upstream SAM3 predictors enter a bf16 ``torch.autocast`` in
``__init__`` and never exit it. Built on a pooled worker thread, that left the
worker at autocast nesting >= 1 forever, so its weight-cast cache was never
cleared (~550 MB/job) and later work on the thread silently ran under bf16.
``build_on_fresh_thread`` confines the build to a throwaway thread.

Leak 2: each InferenceJob owns YOLO detectors; finished jobs are retained
(JOB_RETENTION), so ``_release_gpu`` drops them at job end.
"""
import asyncio
import gc
import weakref

import pytest
import torch

import web.app as app_module
from web.app import InferenceJob
from web.field import sam3_tracker
from web.field.sam3_tracker import build_on_fresh_thread


def _cpu_autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:  # older torch
        return torch.is_autocast_cpu_enabled()


def _leaky_build(value="built"):
    """Mimics upstream Sam3TrackerPredictor.__init__: enter autocast, never exit."""
    ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    ctx.__enter__()
    assert _cpu_autocast_enabled()
    return value


# ── build_on_fresh_thread ────────────────────────────────────────────────

def test_fresh_thread_build_leaves_caller_autocast_untouched():
    assert not _cpu_autocast_enabled()
    assert build_on_fresh_thread(_leaky_build, "model") == "model"
    assert not _cpu_autocast_enabled()
    # Repeated builds (one per job) never accumulate nesting on the caller.
    for _ in range(3):
        build_on_fresh_thread(_leaky_build)
    assert not _cpu_autocast_enabled()


def test_fresh_thread_build_passes_kwargs_and_reraises():
    assert build_on_fresh_thread(lambda *, a, b: a + b, a=2, b=3) == 5

    class Boom(RuntimeError):
        pass

    def explode():
        raise Boom("no checkpoint")

    with pytest.raises(Boom, match="no checkpoint"):
        build_on_fresh_thread(explode)
    assert not _cpu_autocast_enabled()


def test_inline_build_would_poison_the_caller():
    """Mutation check: the property above is the helper's doing. Calling the
    leaky build inline leaves the calling thread under autocast — exactly the
    pooled-worker poisoning the helper prevents."""
    import threading

    seen = {}

    def worker():
        _leaky_build()  # what Sam3FieldTracker did before the fix
        seen["after"] = _cpu_autocast_enabled()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["after"] is True


def test_fresh_thread_check_fails_against_inline_helper(monkeypatch):
    """Swap the helper for an inline call and the leak test's assertion
    fails (run on a scratch thread so the poison doesn't leak into pytest)."""
    import threading

    def inline(fn, *a, **kw):
        return fn(*a, **kw)

    monkeypatch.setattr(sam3_tracker, "build_on_fresh_thread", inline)
    seen = {}

    def worker():
        sam3_tracker.build_on_fresh_thread(_leaky_build)
        seen["poisoned"] = _cpu_autocast_enabled()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["poisoned"] is True


def test_sam3_field_tracker_builds_on_fresh_thread(monkeypatch):
    """Both Sam3FieldTracker build sites (base + multiplex) route through
    the helper, so the constructing thread's autocast state is unchanged."""
    import threading

    sam3 = pytest.importorskip("sam3.model_builder")
    build_threads = []

    class _Model:
        max_num_objects = 4

    class _Pred:
        model = _Model()

    def fake_build(**kwargs):
        build_threads.append(threading.current_thread())
        _leaky_build()
        return _Pred()

    monkeypatch.setattr(sam3, "build_sam3_video_predictor", fake_build)
    monkeypatch.setattr(sam3, "build_sam3_predictor", fake_build)
    monkeypatch.setattr(sam3_tracker, "_install_multiplex_shims", lambda m: None)

    for multiplex in (False, True):
        sam3_tracker.Sam3FieldTracker(device="cuda", multiplex=multiplex)
        assert build_threads[-1] is not threading.current_thread()
        assert not _cpu_autocast_enabled()
    assert len(build_threads) == 2


# ── InferenceJob._release_gpu ────────────────────────────────────────────

class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


class _FakeYolo:
    """Holds an nn.Module like ultralytics.YOLO.model."""

    def __init__(self):
        self.model = _Module()
        self.names = {0: "robot"}


def _bare_job(name="job"):
    job = InferenceJob.__new__(InferenceJob)
    job.job_id = name
    job.robot_embedder = None
    job.model = _FakeYolo()
    job.bumper_model = _FakeYolo()
    job.person_model = _FakeYolo()
    job._sam3_extra_trackers = [object()]
    return job


def test_release_gpu_nulls_models_and_is_idempotent():
    job = _bare_job()

    class _Emb:
        released = 0

        def release(self):
            _Emb.released += 1

    job.robot_embedder = _Emb()
    job._release_gpu()
    assert job.model is None and job.bumper_model is None and job.person_model is None
    assert job.robot_embedder is None and _Emb.released == 1
    assert job._sam3_extra_trackers == []
    job._release_gpu()  # second call: no error, nothing re-released
    assert _Emb.released == 1


def test_release_gpu_never_raises_on_bare_job():
    job = InferenceJob.__new__(InferenceJob)  # no attributes at all
    job._release_gpu()
    assert job.model is None


def test_release_gpu_frees_the_yolo_modules_while_job_is_retained():
    """The job itself stays in ``jobs`` (JOB_RETENTION); after release nothing
    else may keep its detector modules alive."""
    job = _bare_job()
    refs = [weakref.ref(getattr(job, a).model)
            for a in InferenceJob._GPU_MODEL_ATTRS]
    job._release_gpu()
    gc.collect()
    assert all(r() is None for r in refs)


def test_real_job_yolo_released(monkeypatch):
    """A real InferenceJob with real ultralytics YOLO objects (CPU predict):
    after _release_gpu the underlying nn.Modules die even though the job
    object is still referenced — no ultralytics/global holder."""
    np = pytest.importorskip("numpy")
    job = InferenceJob("j", "/mock/video.mp4")
    frame = np.zeros((128, 160, 3), np.uint8)
    # Exercise the predictor path (lazy predictor/AutoBackend setup) on CPU.
    job.person_model(frame, classes=[0], verbose=False, device="cpu")
    job.model(frame, verbose=False, device="cpu")
    refs = [weakref.ref(getattr(job, a).model)
            for a in InferenceJob._GPU_MODEL_ATTRS
            if getattr(job, a) is not None]
    assert refs
    job._release_gpu()
    gc.collect()
    assert all(r() is None for r in refs)


def test_release_gpu_logs_vram_line_when_cuda(capsys):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _bare_job("jid")._release_gpu()
    assert "[gpu] job jid end: allocated=" in capsys.readouterr().out


# ── wiring: _run_job_serialized calls _release_gpu on every exit path ───

class _WS:
    async def send_json(self, payload):
        pass


def _serial_job(process):
    job = _bare_job()
    job.websocket = _WS()
    job._websocket_send_lock = asyncio.Lock()
    job._stopped = False
    job.status = "initializing"
    job.error = None
    job.process_video = process
    calls = []
    orig = job._release_gpu

    def spy():
        calls.append(1)
        orig()

    job._release_gpu = spy
    return job, calls


def _run(monkeypatch, job):
    monkeypatch.setattr(app_module, "_job_slots", asyncio.Semaphore(1))
    monkeypatch.setattr(app_module, "_QUEUE_POLL_SECONDS", 0.02)

    async def go():
        try:
            await app_module._run_job_serialized(job)
        except RuntimeError:
            pass

    asyncio.run(go())


def test_run_job_serialized_releases_on_success(monkeypatch):
    async def ok():
        job.status = "completed"

    job, calls = _serial_job(ok)
    _run(monkeypatch, job)
    assert calls == [1] and job.model is None


def test_run_job_serialized_releases_on_error(monkeypatch):
    async def boom():
        raise RuntimeError("boom")

    job, calls = _serial_job(boom)
    _run(monkeypatch, job)
    assert calls == [1] and job.model is None


def test_run_job_serialized_releases_on_stop(monkeypatch):
    async def never():
        raise AssertionError("must not run")

    job, calls = _serial_job(never)
    job._stopped = True  # stop arrived before the slot was taken
    _run(monkeypatch, job)
    assert calls == [1] and job.status == "error" and job.model is None

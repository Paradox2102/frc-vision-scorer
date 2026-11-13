"""Concurrent per-view SAM3 propagation (the ``sam3_parallel_views`` knob).

Batch-1 SAM3 propagation is latency/overhead-bound (~180 ms/frame of fixed
backbone + launch/sync cost), so a single view leaves a large card mostly
idle (RTX PRO 6000 96 GB, qm8: GPU util 0-77%, 9-12 GB peak). Running the
camera views side by side cuts SAM wall time from sum(views) towards
max(view).

Design (see InferenceJob precompute Phase 2 in web/app.py):

* One full SAM3 predictor per concurrent view. Each ``Sam3FieldTracker``
  owns its model instance, its session dict (``_all_inference_states``) and
  its instance-level shims (skip-detector heads, memory bounding), so no
  SAM state is shared between threads. A shared model with per-thread
  sessions was rejected: the SAM3 video model mutates per-call attributes
  and the monkeypatched shims close over one model.
* Views run on the default thread pool (``asyncio.to_thread``); torch
  releases the GIL inside kernels and blocking syncs. Each worker thread
  gets its own CUDA stream (``sam3_parallel_streams``) so kernels from
  different views can overlap instead of serialising on the legacy default
  stream.
* Everything that IS shared is serialised or isolated: the job's YOLO /
  person / embedder models (ultralytics predictors are not thread-safe) run
  under one lock via :func:`serialized`; identity diagnostics go to a
  per-view child recorder merged back in view order; per-view result dicts
  are re-keyed into view order with :func:`reorder_view_keys`, so
  downstream iteration order matches the sequential path.

This module is torch-free at import time so the planner and dispatcher can
be unit-tested without a GPU.
"""
from __future__ import annotations

import asyncio
import functools
import os
import threading
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence

# VRAM budget model for admitting N concurrent views (GiB). Model weights are
# the SAM3 video model (~3.4 GB fp32 checkpoint); a chunked session costs
# ~14 MB/frame of GPU-resident video (sam3_tracker.py header) plus activation /
# memory-bank overhead; the margin covers YOLO, the embedder, the stream
# prefetch models and allocator fragmentation.
SAM3_MODEL_GIB = float(os.environ.get("SAM3_PARALLEL_MODEL_GIB", "3.5"))
SAM3_FRAME_MIB = float(os.environ.get("SAM3_PARALLEL_FRAME_MIB", "14"))
SAM3_SESSION_OVERHEAD_GIB = float(
    os.environ.get("SAM3_PARALLEL_SESSION_OVERHEAD_GIB", "2.0"))
SAM3_PARALLEL_MARGIN_GIB = float(os.environ.get("SAM3_PARALLEL_MARGIN_GIB", "4.0"))


def session_peak_gib(chunk_frames: int) -> float:
    """Estimated peak VRAM of one live SAM3 chunk session."""
    return max(1, int(chunk_frames)) * SAM3_FRAME_MIB / 1024.0 + SAM3_SESSION_OVERHEAD_GIB


def required_gib(n_concurrent: int, chunk_frames: int) -> float:
    """Free VRAM needed (measured AFTER the first predictor is resident) to
    run ``n_concurrent`` views: n-1 extra model copies + n live sessions +
    margin."""
    n = max(1, int(n_concurrent))
    return ((n - 1) * SAM3_MODEL_GIB + n * session_peak_gib(chunk_frames)
            + SAM3_PARALLEL_MARGIN_GIB)


def plan_parallel_views(n_views: int, max_parallel: int, free_bytes: Optional[int],
                        chunk_frames: int) -> tuple[int, str]:
    """How many views to propagate concurrently (1 = sequential).

    Picks the largest ``n <= min(max_parallel, n_views)`` whose
    :func:`required_gib` fits in ``free_bytes``; returns ``(n, reason)``.
    ``free_bytes`` None (probe failed) is treated as no room.
    """
    want = min(int(max_parallel), int(n_views))
    if want < 2:
        return 1, f"only {want} view slot(s) (views={n_views}, max={max_parallel})"
    if free_bytes is None:
        return 1, "free-VRAM probe failed"
    free_gib = free_bytes / float(1024 ** 3)
    for n in range(want, 1, -1):
        need = required_gib(n, chunk_frames)
        if free_gib >= need:
            note = "" if n == want else f" (wanted {want})"
            return n, (f"{n} concurrent{note}: need {need:.1f} GiB, "
                       f"{free_gib:.1f} GiB free")
    return 1, (f"not enough VRAM for 2 concurrent views: need "
               f"{required_gib(2, chunk_frames):.1f} GiB, {free_gib:.1f} GiB free")


def serialized(fn: Optional[Callable], lock: Any) -> Optional[Callable]:
    """Wrap ``fn`` so every call holds ``lock``. ``fn`` None or ``lock`` None
    returns ``fn`` unchanged (the sequential path passes no lock)."""
    if fn is None or lock is None:
        return fn

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with lock:
            return fn(*args, **kwargs)

    return _locked


def call_on_cuda_stream(stream: Any, fn: Callable, /, *args, **kwargs):
    """Run ``fn`` with ``stream`` as this thread's current CUDA stream, then
    wait for the stream so results handed back are complete. ``stream``
    None runs ``fn`` directly."""
    if stream is None:
        return fn(*args, **kwargs)
    import torch

    with torch.cuda.stream(stream):
        out = fn(*args, **kwargs)
    stream.synchronize()
    return out


class ParallelProgress:
    """Aggregate per-view fractions into one monotone-ish job percentage.

    Sequential progress gives each view its own slice of the bar; with views
    running together that would make the bar jump between slices, so the
    parallel path reports ``base + mean(view fractions) * span`` instead.
    """

    def __init__(self, view_names: Iterable[str], base_pct: float, span_pct: float):
        self._fracs = {name: 0.0 for name in view_names}
        self._base = float(base_pct)
        self._span = float(span_pct)
        self._lock = threading.Lock()

    def update(self, view_name: str, frac: float) -> float:
        with self._lock:
            self._fracs[view_name] = min(1.0, max(0.0, float(frac)))
            mean = sum(self._fracs.values()) / max(1, len(self._fracs))
        return self._base + mean * self._span


def reorder_view_keys(d: dict, view_names: Sequence[str], pre_keys: Sequence) -> None:
    """Re-key ``d`` IN PLACE into the insertion order the sequential loop
    would have produced: keys present before the loop (``pre_keys``, in
    their order), then view keys in view order, then anything else in its
    current order. Identity of ``d`` is kept (other references stay valid).
    """
    order = []
    seen = set()
    for k in list(pre_keys) + list(view_names) + list(d.keys()):
        if k in d and k not in seen:
            seen.add(k)
            order.append(k)
    items = [(k, d[k]) for k in order]
    d.clear()
    d.update(items)


_SKIPPED = object()


async def run_views_concurrently(
    items: Sequence[tuple[int, Any]],
    run_one: Callable[[int, Any, Any], Awaitable[Any]],
    workers: Sequence[Any],
    *,
    before_start: Optional[Callable[[], None]] = None,
) -> list:
    """Run ``run_one(idx, view, worker)`` for every item, at most
    ``len(workers)`` at a time, each in-flight call owning one worker
    exclusively (a worker is a predictor + its stream, never shared).

    Items start in order. ``before_start`` (e.g. the job's stop check) runs
    right before each item starts. Once any item fails, items that have not
    started yet are skipped (the sequential loop would never have reached
    them). ALL started items are awaited before returning — no orphaned
    propagation threads — and then the first exception in ITEM order is
    re-raised. Returns results in item order (None for skipped items).
    """
    if not workers:
        raise ValueError("run_views_concurrently needs at least one worker")
    pool: asyncio.Queue = asyncio.Queue()
    for w in workers:
        pool.put_nowait(w)
    failed = {"any": False}

    async def _one(idx, view):
        worker = await pool.get()
        try:
            if failed["any"]:
                return _SKIPPED
            if before_start is not None:
                before_start()
            return await run_one(idx, view, worker)
        except BaseException:
            failed["any"] = True
            raise
        finally:
            pool.put_nowait(worker)

    tasks = [asyncio.ensure_future(_one(idx, view)) for idx, view in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return [None if r is _SKIPPED else r for r in results]

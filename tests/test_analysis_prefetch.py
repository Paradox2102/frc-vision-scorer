"""stream_analysis: prefetch of the analysis loop's SAM-independent work.

Covers the pure pieces (OCR schedule, plan/fallback resolution, the
prefetcher's cache contract) and the app seam: with the knob OFF the job never
builds a prefetcher, so every loop site keeps its original inline call.
"""
import asyncio
import threading
import types

import numpy as np
import pytest

from web.analysis_prefetch import (
    MISS, AnalysisPrefetcher, ocr_frames_for, stream_analysis_plan)
from web.job_config import JobConfig


# ---------------------------------------------------------------------------
# OCR schedule mirrors the loop's processed_frame_num rule
# ---------------------------------------------------------------------------
def test_ocr_frames_match_loop_rule():
    frames = list(range(100, 100 + 3 * 60, 3))  # frame_skip 3
    interval = 5
    # Simulate the loop: processed_frame_num counts analysis frames only.
    expected = set()
    for processed_frame_num, f in enumerate(frames):
        if processed_frame_num >= 10 and processed_frame_num % interval == 0:
            expected.add(f)
    assert ocr_frames_for(frames, interval) == expected
    # Order-insensitive input (the app passes a sorted list anyway).
    assert ocr_frames_for(reversed(frames), interval) == expected


# ---------------------------------------------------------------------------
# Plan / fallback conditions
# ---------------------------------------------------------------------------
def _plan(**kw):
    base = dict(enabled=True, use_mask_tracker=True, device="cuda",
                needs_vllm=False, vram_fit=False)
    base.update(kw)
    return stream_analysis_plan(**base)


def test_plan_off_is_inert():
    p = _plan(enabled=False)
    assert not p.active and not p.ocr and not p.gpu and p.notes == []


def test_plan_full_on_cuda():
    p = _plan()
    assert p.ocr and p.gpu and p.notes == []


def test_plan_no_sam_falls_back_to_sequential():
    p = _plan(use_mask_tracker=False)
    assert not p.active
    assert any("sequential" in n for n in p.notes)


@pytest.mark.parametrize("kw", [
    {"device": "cpu"}, {"needs_vllm": True}, {"vram_fit": True}])
def test_plan_gpu_fallbacks_keep_ocr(kw):
    p = _plan(**kw)
    assert p.ocr and not p.gpu
    assert p.notes


# ---------------------------------------------------------------------------
# Prefetcher cache contract
# ---------------------------------------------------------------------------
def _frames(nums):
    def read():
        for n in nums:
            yield n, np.full((8, 4, 3), n % 251, dtype=np.uint8)
    return read


def _gpu_setup(calls):
    def setup():
        def main_det(overhead):
            calls.append(("main", int(overhead[0, 0, 0]), overhead.shape[0]))
            return {"main": int(overhead[0, 0, 0])}

        def person(view_frame):
            calls.append(("person", int(view_frame[0, 0, 0])))
            return [(0, 0, 1, int(view_frame[0, 0, 0]))]
        return main_det, person
    return setup


def _views():
    return [("main", lambda fr: fr[:, :2]), ("corner_left", lambda fr: fr[:, 2:])]


def test_prefetch_fills_and_take_pops():
    calls = []
    nums = [10, 13, 16, 19]
    pf = AnalysisPrefetcher(
        _frames(nums), overhead_height=5, views=_views(),
        ocr_frames={13, 19}, ocr_fn=lambda ov, a: (a, int(ov[0, 0, 0]), ov.shape[0]),
        setup_gpu=_gpu_setup(calls), log=lambda *_: None)
    pf.run()
    assert pf.done and pf.gpu_active and pf.frames_done == 4
    # main YOLO sees the overhead crop only.
    assert ("main", 13, 5) in calls
    assert pf.take_main(13) == {"main": 13}
    assert pf.take_main(13) is MISS  # popped
    assert pf.take_person(16, "corner_left") == [(0, 0, 1, 16)]
    assert pf.take_person(16, "nope") is MISS
    assert pf.take_ocr(19, "red") == ("red", 19, 5)
    assert pf.take_ocr(19, "blue") == ("blue", 19, 5)
    assert pf.take_ocr(16, "red") is MISS  # not an OCR frame -> loop inline
    s = pf.stats()
    assert s["stream.hit.main"] == 1 and s["stream.miss.main"] == 1
    assert s["stream.hit.ocr"] == 2 and s["stream.miss.ocr"] == 1
    assert s["stream.prefetch_frames"] == 4


def test_ocr_none_result_is_a_hit_not_a_miss():
    pf = AnalysisPrefetcher(
        _frames([1, 2]), overhead_height=4, views=[], ocr_frames={2},
        ocr_fn=lambda ov, a: None, log=lambda *_: None)
    pf.run()
    assert pf.take_ocr(2, "blue") is None


def test_prefetch_skips_frames_the_loop_reached():
    calls = []
    pf = AnalysisPrefetcher(
        _frames([1, 2, 3, 4]), overhead_height=4, views=_views(),
        ocr_frames=set(), ocr_fn=None, setup_gpu=_gpu_setup(calls),
        log=lambda *_: None)
    pf.note_loop_frame(2)
    pf.run()
    assert pf.take_main(1) is MISS and pf.take_main(2) is MISS
    assert pf.take_main(3) == {"main": 3}


def test_prefetch_stops_on_job_stop_and_cancel():
    seen = []

    def read():
        for n in range(100):
            seen.append(n)
            yield n, np.zeros((4, 4, 3), dtype=np.uint8)

    flag = {"stop": False}
    pf = AnalysisPrefetcher(
        read, overhead_height=2, views=[], ocr_frames=set(range(100)),
        ocr_fn=lambda ov, a: (flag.__setitem__("stop", len(seen) >= 3) or 0),
        should_stop=lambda: flag["stop"], log=lambda *_: None)
    pf.run()
    assert pf.frames_done == 3 and len(seen) == 4

    pf2 = AnalysisPrefetcher(
        _frames(range(50)), overhead_height=2, views=[],
        ocr_frames=set(range(50)), ocr_fn=lambda ov, a: 0, log=lambda *_: None)
    pf2.cancel()
    pf2.run()
    assert pf2.frames_done == 0 and pf2.done


def test_gpu_setup_failure_degrades_to_ocr_only():
    def bad_setup():
        raise RuntimeError("no cuda")
    pf = AnalysisPrefetcher(
        _frames([5, 6]), overhead_height=4, views=_views(), ocr_frames={6},
        ocr_fn=lambda ov, a: 7, setup_gpu=bad_setup, log=lambda *_: None)
    pf.run()
    assert not pf.gpu_active
    assert pf.take_main(6) is MISS
    assert pf.take_ocr(6, "red") == 7


def test_worker_error_leaves_rest_to_loop():
    def boom(ov, a):
        raise ValueError("tesseract died")
    pf = AnalysisPrefetcher(
        _frames([1, 2, 3]), overhead_height=4, views=[], ocr_frames={2},
        ocr_fn=boom, log=lambda *_: None)
    pf.run()  # never raises
    assert pf.done and pf.error and pf.frames_done == 1
    assert pf.take_ocr(3, "red") is MISS


def test_prefetch_runs_on_worker_thread():
    """The cache is filled from a worker thread and read from another —
    the way the app drives it (asyncio.to_thread + loop-side take)."""
    pf = AnalysisPrefetcher(
        _frames(range(20)), overhead_height=4, views=[],
        ocr_frames=set(range(20)), ocr_fn=lambda ov, a: int(ov[0, 0, 0]),
        log=lambda *_: None)
    t = threading.Thread(target=pf.run)
    t.start()
    t.join(timeout=10)
    assert pf.done
    assert [pf.take_ocr(i, "blue") for i in range(20)] == list(range(20))


# ---------------------------------------------------------------------------
# Knob registry + A/B speed-profile composition
# ---------------------------------------------------------------------------
def test_knob_default_on():
    assert JobConfig().stream_analysis is True
    assert "stream_analysis" not in JobConfig().non_defaults()


def test_speed_profile_arms_validate():
    from scripts.ab_qm8 import PROFILES
    arms = dict(PROFILES["speed"])
    assert set(arms) == {"B1-fps5-cs3", "B2-+selective", "B3-+stream"}
    for name, cfg in arms.items():
        c = JobConfig(cfg)  # raises on an invalid / unknown knob
        assert c.team_reader == "jev" and c.read_census and c.read_keep_evicted
        assert c.sam3_track_fps == 5 and c.corner_track_stride == 3
    assert JobConfig(arms["B3-+stream"]).stream_analysis is True
    assert JobConfig(arms["B2-+selective"]).selective_mask_prop is True
    assert JobConfig(arms["B2-+selective"]).selective_app_weight == 0.3


# ---------------------------------------------------------------------------
# App seam: knob OFF never builds a prefetcher
# ---------------------------------------------------------------------------
def _stub_job(stream: bool, use_mask_tracker: bool = True):
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)
    job.config = types.SimpleNamespace(
        stream_analysis=stream, vllm_phase_exclusive=False)
    job.team_reader = "jev"
    job.use_mask_tracker = use_mask_tracker
    job.sam3_max_gpu_frames = 0
    job._stopped = False
    job._analysis_prefetch = None
    job._analysis_prefetch_task = None
    job.video_path = "/nonexistent.mp4"
    job.overhead_height = 4
    job.views = []
    job.ocr_regions = {}
    job.timings = {}
    return job


def test_app_knob_off_starts_nothing():
    job = _stub_job(stream=False)

    async def go():
        pf = job._start_analysis_prefetch([1, 2, 3], "cuda")
        await job._finish_analysis_prefetch()  # no-op
        return pf
    assert asyncio.run(go()) is None
    assert job._analysis_prefetch is None and job._analysis_prefetch_task is None
    assert job.timings == {}


def test_app_knob_on_without_sam_starts_nothing():
    job = _stub_job(stream=True, use_mask_tracker=False)

    async def go():
        return job._start_analysis_prefetch([1, 2, 3], "cuda")
    assert asyncio.run(go()) is None


def test_app_knob_on_cpu_runs_ocr_prefetch_and_records_stats(monkeypatch):
    """CPU device -> OCR-only plan; an unreadable video just yields no
    frames, the worker finishes, and stats land in timings."""
    job = _stub_job(stream=True)

    async def go():
        pf = job._start_analysis_prefetch([1, 2, 3], "cpu")
        assert pf is not None and job._analysis_prefetch_task is not None
        await job._finish_analysis_prefetch()
        return pf
    pf = asyncio.run(go())
    assert pf.done and not pf.gpu_active
    assert job.timings["stream.prefetch_frames"] == 0
    assert job._analysis_prefetch is None

"""The fused-bumper VLM read must not block the event loop.

A Qwen generate takes 100-500ms; if it runs synchronously inside the async
frame loop, websocket keepalives and pause/stop handling freeze for the
duration. The read itself must be pushed to a worker thread while all
identity-state mutation stays on the loop.
"""
import asyncio
import time
import types

import web.app as app_module
from web.app import InferenceJob


def _make_job():
    job = InferenceJob.__new__(InferenceJob)
    job.match_teams = ["4174"]
    job.match_blue_teams = ["4174"]
    job.match_red_teams = []
    job.vlm_status = {}
    job._vlm_task = None
    job._claude_vision_calls = 0
    job._current_frame = 100
    job.video_fps = 30.0
    job.field_tracker = types.SimpleNamespace(last_det_to_slot={0: 1})
    ident = types.SimpleNamespace(
        active_slot=1, alliance="blue",
        last_vlm_time=0.0, vlm_fail_count=0, backoff_until=-999.0,
    )
    job.identity_registry = types.SimpleNamespace(identities={7: ident})
    job._occ = {}
    job.confidence_config = types.SimpleNamespace(backoff_base_sec=2.0)
    job.team_reader = "local"
    # Post-read bookkeeping that needs live tracker/gallery state is stubbed:
    # this test only cares about event-loop responsiveness during the read.
    job._get_or_create_slot_track = lambda slot_id: None
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    return job


def _fused_obs():
    return [types.SimpleNamespace(evidence=[("main", "fakecropb64", 1.0)])]


def test_vlm_read_runs_off_event_loop(monkeypatch):
    def slow_read(crops, teams, blue, red, expected_alliances=None, provider="local"):
        time.sleep(0.3)  # stand-in for a Qwen generate
        return [{"team_number": "4174", "confidence": 0.9,
                 "alliance": "blue", "raw_read": "4174"}]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", slow_read)

    async def run():
        job = _make_job()
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(ticker())
        try:
            await job.read_fused_bumper_numbers(_fused_obs(), {7: 1.0})
        finally:
            ticker_task.cancel()
        return ticks

    ticks = asyncio.run(run())
    # A blocked loop yields ~1 tick over the 0.3s read; an unblocked one ~30.
    assert ticks >= 10, f"event loop only ticked {ticks} times during VLM read"


def test_vlm_read_results_still_applied(monkeypatch):
    """Threading the read must not change the result flow."""
    monkeypatch.setattr(
        app_module, "_vision_read_bumpers",
        lambda crops, teams, blue, red, expected_alliances=None, provider="local":
            [{"team_number": "4174", "confidence": 0.9,
              "alliance": "blue", "raw_read": "4174"}],
    )

    async def run():
        job = _make_job()
        await job.read_fused_bumper_numbers(_fused_obs(), {7: 1.0})
        return job

    job = asyncio.run(run())
    assert job.vlm_status["last_sent"] == 1
    assert job.vlm_status["last_error"] is None
    assert job._ocr_crops_dirty is True


def test_single_flight_guard_skips_while_pass_in_flight(monkeypatch):
    """A second scheduling decision while a pass is still running must NOT
    launch a second read — the guard keeps at most one pass in flight."""
    started = 0

    def slow_read(crops, teams, blue, red, expected_alliances=None, provider="local"):
        nonlocal started
        started += 1
        time.sleep(0.3)  # still running when the second schedule decision fires
        return [{"team_number": "4174", "confidence": 0.9,
                 "alliance": "blue", "raw_read": "4174"}]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", slow_read)

    async def run():
        job = _make_job()
        job._claude_vision_calls = 0
        # First decision: no pass in flight -> schedules one.
        job._schedule_vlm_pass(_fused_obs(), {7: 1.0})
        assert job._vlm_task is not None
        # Yield so the task body actually starts and enters the slow read.
        await asyncio.sleep(0.05)
        assert not job._vlm_task.done(), "pass should still be running"
        # Second decision while in flight: must be a no-op (single-flight).
        job._schedule_vlm_pass(_fused_obs(), {7: 1.0})
        await job._vlm_task  # let the one real pass finish
        return job

    job = asyncio.run(run())
    assert started == 1, f"expected exactly one read pass, got {started}"
    assert job._claude_vision_calls == 1


def test_drain_awaits_pending_pass_and_lands_votes(monkeypatch):
    """_drain_vlm_task must await an in-flight pass so its mutations land
    before finalization, and clear the handle afterward."""
    monkeypatch.setattr(
        app_module, "_vision_read_bumpers",
        lambda crops, teams, blue, red, expected_alliances=None, provider="local":
            [{"team_number": "4174", "confidence": 0.9,
              "alliance": "blue", "raw_read": "4174"}],
    )

    async def run():
        job = _make_job()
        job._schedule_vlm_pass(_fused_obs(), {7: 1.0})
        pending = job._vlm_task
        assert pending is not None and not pending.done()
        await job._drain_vlm_task()
        # Drain awaited the pass: it completed and the handle was cleared.
        assert pending.done()
        assert job._vlm_task is None
        return job

    job = asyncio.run(run())
    # The awaited pass applied its results before we returned.
    assert job.vlm_status["last_sent"] == 1


def test_drain_surfaces_failed_pass_without_raising(capsys):
    """A read pass whose task raises must be logged by the drain, never
    re-raised — a dead pass cannot be allowed to abort finalization."""
    async def run():
        job = _make_job()

        async def boom():
            raise RuntimeError("vLLM exploded")

        job._vlm_task = asyncio.create_task(boom())
        # Must not raise even though the task raised.
        await job._drain_vlm_task()
        assert job._vlm_task is None

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "VLM pass failed (drain)" in out
    assert "vLLM exploded" in out

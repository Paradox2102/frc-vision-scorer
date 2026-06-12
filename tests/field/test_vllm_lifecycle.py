from __future__ import annotations

import asyncio

from web.vllm_lifecycle import ensure_started, ensure_stopped


def _noop_sleep():
    async def sleep(_seconds):
        pass
    return sleep


def test_ensure_started_returns_true_without_launch_when_already_up():
    launches = []

    async def run_script(_name):
        launches.append(_name)

    result = asyncio.run(ensure_started(
        check_fn=lambda: True,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=10,
    ))
    assert result is True
    assert launches == []


def test_ensure_started_launches_and_polls_until_ready():
    launches = []
    polls = {"n": 0}

    async def run_script(name):
        launches.append(name)

    def check():
        polls["n"] += 1
        return polls["n"] >= 3  # ready on the third poll

    result = asyncio.run(ensure_started(
        check_fn=check,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
    ))
    assert result is True
    assert launches == ["start"]


def test_ensure_started_times_out_and_returns_false():
    async def run_script(_name):
        pass

    clock = {"t": 0.0}

    async def sleep(seconds):
        clock["t"] += seconds

    result = asyncio.run(ensure_started(
        check_fn=lambda: False,
        run_script=run_script,
        sleep=sleep,
        ready_timeout=10,
        now=lambda: clock["t"],
    ))
    assert result is False


def test_ensure_stopped_runs_stop_script_and_waits_for_down():
    stops = []
    polls = {"n": 0}

    async def run_script(name):
        stops.append(name)

    def check():
        polls["n"] += 1
        return polls["n"] < 3  # up for two polls after the stop, then down

    result = asyncio.run(ensure_stopped(
        check_fn=check,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
    ))
    assert result is True
    assert stops == ["stop"]


def test_ensure_stopped_noop_when_already_down():
    stops = []

    async def run_script(name):
        stops.append(name)

    result = asyncio.run(ensure_stopped(
        check_fn=lambda: False,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=10,
    ))
    assert result is True
    assert stops == []

from __future__ import annotations

import asyncio
import json

from web.vllm_lifecycle import ensure_started, ensure_stopped


def _noop_sleep():
    async def sleep(_seconds):
        pass
    return sleep


class _FakeHttp:
    """Records (method, path) calls; serves scripted (status, body) responses.

    `is_sleeping_states` is popped left-to-right per GET /is_sleeping so a test
    can model the state flipping after a sleep/wake POST. When exhausted it
    holds the last value. `raise_on` maps a (method, path) to an exception to
    raise instead of responding.
    """

    def __init__(self, *, is_sleeping_states=None, raise_on=None,
                 post_status=200):
        self.calls = []
        self._states = list(is_sleeping_states or [])
        self._raise_on = dict(raise_on or {})
        self._post_status = post_status

    def __call__(self, method, path):
        self.calls.append((method, path))
        exc = self._raise_on.get((method, path))
        if exc is not None:
            raise exc
        if method == "GET" and path == "/is_sleeping":
            if len(self._states) > 1:
                state = self._states.pop(0)
            else:
                state = self._states[0] if self._states else False
            return 200, json.dumps({"is_sleeping": state})
        return self._post_status, ""


def test_ensure_started_returns_true_without_launch_when_already_up():
    launches = []

    async def run_script(_name):
        launches.append(_name)

    result = asyncio.run(ensure_started(
        check_fn=lambda: True,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=10,
        http_fn=_FakeHttp(is_sleeping_states=[False]),  # up and awake
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


# --- sleep/wake fast path ---------------------------------------------------


def test_ensure_stopped_sleeps_running_server_without_killing_process():
    stops = []

    async def run_script(name):
        stops.append(name)

    # Awake at the sleep POST, reports sleeping on the following poll.
    http = _FakeHttp(is_sleeping_states=[False, True])

    result = asyncio.run(ensure_stopped(
        check_fn=lambda: True,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
        http_fn=http,
    ))
    assert result is True
    assert ("POST", "/sleep?level=1") in http.calls
    assert ("GET", "/is_sleeping") in http.calls
    assert stops == []  # process was never killed


def test_ensure_stopped_falls_back_to_stop_script_when_sleep_errors():
    stops = []
    polls = {"n": 0}

    async def run_script(name):
        stops.append(name)

    def check():
        polls["n"] += 1
        return polls["n"] < 3  # up initially, then down after the stop

    # The /sleep POST raises — sleep path must bail to the stop script.
    http = _FakeHttp(raise_on={("POST", "/sleep?level=1"): OSError("boom")})

    result = asyncio.run(ensure_stopped(
        check_fn=check,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
        http_fn=http,
    ))
    assert result is True
    assert stops == ["stop"]


def test_ensure_started_wakes_sleeping_server_without_start_script():
    launches = []

    async def run_script(name):
        launches.append(name)

    # Sleeping at first probe, awake after the wake_up POST.
    http = _FakeHttp(is_sleeping_states=[True, False])

    result = asyncio.run(ensure_started(
        check_fn=lambda: True,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
        http_fn=http,
    ))
    assert result is True
    assert ("POST", "/wake_up") in http.calls
    assert launches == []  # no cold start


def test_ensure_started_cold_starts_when_nothing_running():
    launches = []
    polls = {"n": 0}

    async def run_script(name):
        launches.append(name)

    def check():
        polls["n"] += 1
        return polls["n"] >= 3  # nothing up until the start script boots it

    result = asyncio.run(ensure_started(
        check_fn=check,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
        http_fn=_FakeHttp(is_sleeping_states=[False]),
    ))
    assert result is True
    assert launches == ["start"]  # never touched the sleep/wake routes


def test_ensure_started_falls_back_to_cold_start_when_wake_times_out():
    launches = []
    clock = {"t": 0.0}

    async def sleep(seconds):
        clock["t"] += seconds

    async def run_script(name):
        launches.append(name)

    # Server answers readiness and is sleeping, but never wakes: is_sleeping
    # stays True forever, so _try_wake times out.
    up_and_stuck = {"first": True}

    def check():
        # True on the initial gate; after the wedged sleeper is killed and the
        # cold start runs, it comes up on the next readiness poll.
        if up_and_stuck["first"]:
            up_and_stuck["first"] = False
            return True
        return True

    http = _FakeHttp(is_sleeping_states=[True])  # never flips to False

    result = asyncio.run(ensure_started(
        check_fn=check,
        run_script=run_script,
        sleep=sleep,
        ready_timeout=10,
        now=lambda: clock["t"],
        http_fn=http,
    ))
    assert result is True
    # Wedged sleeper killed, then cold-started.
    assert launches == ["stop", "start"]

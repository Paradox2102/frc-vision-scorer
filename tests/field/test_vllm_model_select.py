"""Model-aware vLLM lifecycle: verify the RIGHT model is serving, restart on
mismatch, preserve the sleep/wake fast path on a match."""
from __future__ import annotations

import asyncio
import json

import pytest

from web import vllm_lifecycle
from web.vllm_lifecycle import ensure_started, ensure_stopped


def _noop_sleep():
    async def sleep(_seconds):
        pass
    return sleep


class _FakeHttp:
    """Serves scripted /is_sleeping responses; records POST calls."""

    def __init__(self, *, is_sleeping_states=None):
        self.calls = []
        self._states = list(is_sleeping_states or [])

    def __call__(self, method, path):
        self.calls.append((method, path))
        if method == "GET" and path == "/is_sleeping":
            state = self._states.pop(0) if len(self._states) > 1 else (
                self._states[0] if self._states else False)
            return 200, json.dumps({"is_sleeping": state})
        return 200, ""


@pytest.fixture(autouse=True)
def _restore_active_model():
    """Every test starts and ends on the incumbent so state can't leak."""
    vllm_lifecycle.set_active_model(vllm_lifecycle.vlm_models.DEFAULT_KEY)
    yield
    vllm_lifecycle.set_active_model(vllm_lifecycle.vlm_models.DEFAULT_KEY)


def test_mismatch_triggers_stop_then_start():
    """A different model serving on the shared port is stopped and the requested
    model cold-started."""
    launches = []

    async def run_script(name):
        launches.append(name)

    # Ask for the 8B model; the server reports it is serving the 4B model.
    vllm_lifecycle.set_active_model("qwen3-vl-8b")
    served = "Qwen/Qwen3-VL-4B-Instruct"

    result = asyncio.run(ensure_started(
        check_fn=lambda: True,  # a server is up at the gate and re-ready after start
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
        http_fn=_FakeHttp(is_sleeping_states=[False]),
        served_model_fn=lambda: served,
    ))
    assert result is True
    assert launches == ["stop", "start"]


def test_match_preserves_sleep_wake_fast_path():
    """When the requested model is already serving, a sleeper is woken with no
    process launch."""
    launches = []

    async def run_script(name):
        launches.append(name)

    vllm_lifecycle.set_active_model("qwen3-vl-4b")
    http = _FakeHttp(is_sleeping_states=[True, False])  # sleeping, then awake

    result = asyncio.run(ensure_started(
        check_fn=lambda: True,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
        http_fn=http,
        served_model_fn=lambda: "Qwen/Qwen3-VL-4B-Instruct",
    ))
    assert result is True
    assert ("POST", "/wake_up") in http.calls
    assert launches == []  # no cold start — right model was already resident


def test_served_none_leaves_server_running():
    """An undeterminable served model is NOT a confirmed mismatch: an up+awake
    server is used as-is."""
    launches = []

    async def run_script(name):
        launches.append(name)

    vllm_lifecycle.set_active_model("qwen3-vl-8b")

    result = asyncio.run(ensure_started(
        check_fn=lambda: True,
        run_script=run_script,
        sleep=_noop_sleep(),
        ready_timeout=60,
        http_fn=_FakeHttp(is_sleeping_states=[False]),  # up and awake
        served_model_fn=lambda: None,  # can't tell what's serving
    ))
    assert result is True
    assert launches == []


def test_default_run_script_start_exports_model_env(monkeypatch):
    """The default 'start' runner exports the active model's VLM_MODEL_NAME and
    VLM_SERVER_GPU_UTIL into the child env."""
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env")

        class _P:
            async def wait(self_inner):
                return 0
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    vllm_lifecycle.set_active_model("qwen3-vl-8b")

    asyncio.run(vllm_lifecycle._default_run_script("start"))

    assert captured["args"][0].endswith("start_vllm.sh")
    env = captured["env"]
    assert env["VLM_MODEL_NAME"] == vllm_lifecycle.vlm_models.get("qwen3-vl-8b").hf_repo
    assert env["VLM_SERVER_GPU_UTIL"] == str(vllm_lifecycle.vlm_models.gpu_util("qwen3-vl-8b"))


def test_default_run_script_start_appends_extra_serve_args(monkeypatch):
    """GLM's --reasoning-parser flag is forwarded to start_vllm.sh."""
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args

        class _P:
            async def wait(self_inner):
                return 0
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    vllm_lifecycle.set_active_model("glm-4.6v-flash")

    asyncio.run(vllm_lifecycle._default_run_script("start"))

    args = captured["args"]
    assert args[0].endswith("start_vllm.sh")
    assert list(args[1:]) == ["--reasoning-parser", "glm45"]


def test_default_run_script_stop_has_no_model_env(monkeypatch):
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env")

        class _P:
            async def wait(self_inner):
                return 0
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(vllm_lifecycle._default_run_script("stop"))

    assert captured["args"][0].endswith("stop_vllm.sh")
    assert captured["env"] is None  # inherit parent env, no model override

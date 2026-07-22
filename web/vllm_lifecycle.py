"""Phase-exclusive vLLM lifecycle for inference jobs.

The 3090 cannot hold vLLM's ~11GB pool and SAM3 propagation's ~11GB peak at
once (three OOM'd matches on 2026-06-11, across chunk sizes 350/250/150 —
the peak is model + per-object state, not frame residency). But the two
never need the card simultaneously: vLLM is completely idle during the
tracking precompute. So jobs free the GPU before tracking and reclaim it
before the analysis loop's bumper reads.

Fast path — sleep/wake (vLLM 0.22.1 `--enable-sleep-mode`, dev routes):
  ``ensure_stopped`` puts a running server to sleep (``POST /sleep?level=1``)
  instead of killing the process; ``ensure_started`` wakes it back up
  (``POST /wake_up``). Level 1 keeps the fp16 weights resident in CPU RAM and
  only discards the KV cache, so waking needs no disk/weight reload — this
  eliminates the ~60s cold start (process spawn + import ~30s, weight load
  ~12s, engine init ~10s, poll slop) that dominated per-job latency.

  Tradeoff: a *sleeping* vLLM process still holds its CUDA context (~1-2GB) —
  it is not fully off the card. Tracking previously saw ~21GB free with vLLM
  fully stopped; with a sleeper resident it sees ~19GB, which still clears the
  ~11GB SAM3 peak, so coexistence headroom is unaffected.

Fallback — process stop/start (web/stop_vllm.sh / web/start_vllm.sh):
  Used when there is no server to sleep/wake (fresh boot), when the sleep or
  wake request errors or times out, or when a sleeper is wedged. A wedged
  sleeper must never strand a job — if wake-up fails, ``ensure_started`` tears
  the process down and cold-starts so reads actually fire instead of silently
  never waking. A fresh boot always takes this cold-start path.

The dev sleep routes (``/sleep``, ``/wake_up``, ``/is_sleeping``) are mounted
at the *server root*, not under ``/v1`` — the OpenAI-compatible readiness
endpoint is ``{VLM_SERVER_URL}/models`` where VLM_SERVER_URL ends in ``/v1``,
so we derive the root URL by stripping that trailing ``/v1``. They only exist
because web/start_vllm.sh launches with ``--enable-sleep-mode`` and
``VLLM_SERVER_DEV_MODE=1``.

Model selection (web/vlm_models.py): the served bumper reader is per-job, but
every model is served by this same vLLM server in HF format. A job calls
``set_active_model(key)`` at start; ``ensure_started`` then verifies the RIGHT
model is serving (GET ``{VLM_SERVER_URL}/models`` reports vLLM's launched HF id)
and, on a confirmed mismatch, stops the running server and cold-starts it with
the requested model's ``VLM_MODEL_NAME`` / ``VLM_SERVER_GPU_UTIL``. When the
right model is already serving, the sleep/wake fast path applies unchanged. The
default active model is Qwen3-VL-4B at gpu-util 0.45, so with no
``set_active_model`` call the behavior is byte-for-byte the incumbent.

All effectful inputs (readiness check, served-model probe, HTTP transport,
script runner, sleep, clock) are injectable for tests; the defaults shell out to
the scripts and poll over HTTP.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from pathlib import Path
from typing import Awaitable, Callable

from web import vlm_models

_PROJECT_ROOT = Path(__file__).parent.parent

#: The bumper-reader model a job wants served. Set per-job by set_active_model;
#: defaults to the incumbent so untouched callers/tests keep vLLM behavior.
_active_model_key: str = vlm_models.DEFAULT_KEY


def set_active_model(key: str) -> None:
    """Record the model the next ensure_started should serve/verify.

    Validates against the registry (raises ValueError on an unknown key) so a
    bad key surfaces at job start, not mid-lifecycle.
    """
    global _active_model_key
    vlm_models.get(key)  # validate
    _active_model_key = key


def active_model() -> str:
    return _active_model_key

#: How long to wait for the vLLM server to serve /v1/models after launch.
#: Model load is ~90s on the 3090; double it for cold page cache.
VLLM_START_TIMEOUT = float(os.environ.get("VLLM_START_TIMEOUT", "240"))
#: How long to wait for the server to release the GPU (sleep or process stop).
VLLM_STOP_TIMEOUT = float(os.environ.get("VLLM_STOP_TIMEOUT", "60"))
_POLL_SECONDS = 3.0

#: HTTP transport seam: (method, path) -> (status_code, body_text). Path is
#: relative to the server root (e.g. "/sleep?level=1"). Injectable for tests.
HttpFn = Callable[[str, str], "tuple[int, str]"]


def _root_url() -> str:
    """Server root URL — VLM_SERVER_URL with its trailing ``/v1`` stripped.

    The dev sleep routes live at the root, not under ``/v1``.
    """
    from web.app import VLM_SERVER_URL  # late import: avoid cycle at module load
    url = VLM_SERVER_URL
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


def _default_check() -> bool:
    """True when the vLLM server answers /v1/models."""
    from web.app import VLM_SERVER_URL  # late import: avoid cycle at module load
    try:
        req = urllib.request.Request(f"{VLM_SERVER_URL}/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _default_http(method: str, path: str) -> "tuple[int, str]":
    """Fire one HTTP request at the server root; return (status, body)."""
    req = urllib.request.Request(f"{_root_url()}{path}", method=method)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode()


#: Served-model probe seam: () -> served model id (or None if undeterminable).
ServedModelFn = Callable[[], "str | None"]


def _default_served_model() -> "str | None":
    """The id of the model currently served at ``{VLM_SERVER_URL}/models``.

    vLLM reports the HF id it was launched with (e.g.
    ``Qwen/Qwen3-VL-4B-Instruct``). Returns None on any error (no server,
    malformed body) — callers treat None as "cannot confirm a mismatch", so an
    unidentifiable-but-up server is left running rather than needlessly cycled.
    """
    from web.app import VLM_SERVER_URL  # late import: avoid cycle at module load
    try:
        req = urllib.request.Request(f"{VLM_SERVER_URL}/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read().decode())
        data = body.get("data") or []
        return str(data[0].get("id")) if data else None
    except Exception:
        return None


def _served_matches(served_model_fn: ServedModelFn) -> "bool | None":
    """Tri-state: True if the served model is the active model, False on a
    confirmed mismatch, None if it can't be determined (probe returned None)."""
    served = served_model_fn()
    if served is None:
        return None
    return vlm_models.served_matches(_active_model_key, served)


def _is_sleeping(http_fn: HttpFn) -> bool | None:
    """Tri-state sleep probe: True/False from /is_sleeping, None if unknown.

    None means we could not determine the state (route missing, server not
    answering, malformed body) — callers treat None as "not usefully asleep".
    """
    try:
        status, body = http_fn("GET", "/is_sleeping")
    except Exception:
        return None
    if status != 200:
        return None
    try:
        return bool(json.loads(body).get("is_sleeping"))
    except Exception:
        return None


async def _try_sleep(
    http_fn: HttpFn,
    sleep: Callable[[float], Awaitable[None]],
    timeout: float,
    now: Callable[[], float] | None,
) -> bool:
    """POST /sleep?level=1 and poll /is_sleeping until true. False on any error."""
    try:
        status, _ = http_fn("POST", "/sleep?level=1")
    except Exception:
        return False
    if status != 200:
        return False
    clock = now or asyncio.get_event_loop().time
    deadline = clock() + timeout
    while clock() < deadline:
        if _is_sleeping(http_fn) is True:
            return True
        await sleep(_POLL_SECONDS)
    return False


async def _try_wake(
    http_fn: HttpFn,
    check_fn: Callable[[], bool],
    sleep: Callable[[float], Awaitable[None]],
    timeout: float,
    now: Callable[[], float] | None,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """POST /wake_up and poll until is_sleeping is False AND readiness passes.

    Returns False early (giving up the wake) if ``should_stop`` fires, so a
    user STOP does not have to wait out the full readiness timeout.
    """
    try:
        status, _ = http_fn("POST", "/wake_up")
    except Exception:
        return False
    if status != 200:
        return False
    clock = now or asyncio.get_event_loop().time
    deadline = clock() + timeout
    while clock() < deadline:
        if should_stop is not None and should_stop():
            return False
        if _is_sleeping(http_fn) is False and check_fn():
            return True
        await sleep(_POLL_SECONDS)
    return False


async def _default_run_script(name: str) -> None:
    """Start/stop the vLLM server for the active model.

    "start" launches start_vllm.sh with the active model's ``VLM_MODEL_NAME`` and
    ``VLM_SERVER_GPU_UTIL`` exported into the child env (the script reads both,
    defaulting to the incumbent) plus any per-model extra ``vllm serve`` args
    (e.g. GLM's ``--reasoning-parser glm45``) as trailing script arguments the
    script forwards to ``vllm serve``; "stop" runs stop_vllm.sh (pid-based).
    """
    web_dir = _PROJECT_ROOT / "web"
    if name == "start":
        model = vlm_models.get(_active_model_key)
        env = {
            **os.environ,
            "VLM_MODEL_NAME": model.hf_repo,
            "VLM_SERVER_GPU_UTIL": str(model.gpu_util),
        }
        args = [str(web_dir / "start_vllm.sh"), *model.extra_serve_args]
    else:
        env = None
        args = [str(web_dir / "stop_vllm.sh")]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    await proc.wait()


async def ensure_started(
    *,
    check_fn: Callable[[], bool] = _default_check,
    run_script: Callable[[str], Awaitable[None]] = _default_run_script,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ready_timeout: float = VLLM_START_TIMEOUT,
    now: Callable[[], float] | None = None,
    http_fn: HttpFn = _default_http,
    should_stop: Callable[[], bool] | None = None,
    served_model_fn: ServedModelFn = _default_served_model,
) -> bool:
    """Bring the requested reader model up; True when serving, False on timeout.

    Model-aware: if a server is up but serving the WRONG model (confirmed via
    the served-model probe), stop it and cold-start with the requested model's
    env. When the right model is up, the sleep/wake fast path applies (a sleeper
    is woken, no process launch). If nothing is running, cold-start via the
    launch script. A wedged sleeper that refuses to wake is torn down and
    cold-started.

    ``should_stop`` (optional) is polled inside the wait loops so a user STOP
    aborts the ~90s startup wait within a poll interval instead of blocking
    until the readiness timeout; it returns False (server not up) in that case.
    """
    if check_fn():
        # A server answers /v1/models. If it is definitively serving the WRONG
        # model, take it down and cold-start the requested model below. None
        # (undeterminable) is NOT a confirmed mismatch — leave it running.
        if _served_matches(served_model_fn) is False:
            await run_script("stop")
        else:
            # Right model (or undeterminable) — apply the sleep/wake fast path.
            # It might be a sleeper (readiness alone can't tell); only wake if
            # /is_sleeping explicitly says True.
            if _is_sleeping(http_fn) is not True:
                return True
            if await _try_wake(http_fn, check_fn, sleep, ready_timeout, now,
                               should_stop):
                return True
            if should_stop is not None and should_stop():
                return False
            # Wedged sleeper: kill it, then fall through to a clean cold start.
            await run_script("stop")

    await run_script("start")
    clock = now or asyncio.get_event_loop().time
    deadline = clock() + ready_timeout
    while clock() < deadline:
        if should_stop is not None and should_stop():
            return False
        if check_fn():
            return True
        await sleep(_POLL_SECONDS)
    return False


async def ensure_stopped(
    *,
    check_fn: Callable[[], bool] = _default_check,
    run_script: Callable[[str], Awaitable[None]] = _default_run_script,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ready_timeout: float = VLLM_STOP_TIMEOUT,
    now: Callable[[], float] | None = None,
    http_fn: HttpFn = _default_http,
) -> bool:
    """Free the GPU for tracking; True once vLLM releases its pool.

    Fast path: if the server is up, put it to sleep (offloads weights to CPU
    RAM, discards the KV cache — frees the ~11GB pool) and confirm via
    /is_sleeping. On any sleep error/timeout, fall back to the stop script and
    poll until the process is gone. If the server was never up, it is already
    stopped.
    """
    if not check_fn():
        return True
    if await _try_sleep(http_fn, sleep, ready_timeout, now):
        return True

    # Sleep unavailable or wedged — kill the process instead.
    await run_script("stop")
    clock = now or asyncio.get_event_loop().time
    deadline = clock() + ready_timeout
    while clock() < deadline:
        if not check_fn():
            return True
        await sleep(_POLL_SECONDS)
    return False

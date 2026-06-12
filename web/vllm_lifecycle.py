"""Phase-exclusive vLLM lifecycle for inference jobs.

The 3090 cannot hold vLLM's ~11GB pool and SAM3 propagation's ~11GB peak at
once (three OOM'd matches on 2026-06-11, across chunk sizes 350/250/150 —
the peak is model + per-object state, not frame residency). But the two
never need the card simultaneously: vLLM is completely idle during the
tracking precompute. So jobs stop the vLLM server before tracking and start
it again before the analysis loop's bumper reads (~90s model load).

All effectful inputs (readiness check, script runner, sleep, clock) are
injectable for tests; the defaults shell out to web/start_vllm.sh /
web/stop_vllm.sh and poll the OpenAI-compatible /v1/models endpoint.
"""
from __future__ import annotations

import asyncio
import os
import urllib.request
from pathlib import Path
from typing import Awaitable, Callable

_PROJECT_ROOT = Path(__file__).parent.parent

#: How long to wait for the vLLM server to serve /v1/models after launch.
#: Model load is ~90s on the 3090; double it for cold page cache.
VLLM_START_TIMEOUT = float(os.environ.get("VLLM_START_TIMEOUT", "240"))
#: How long to wait for the server process to release the GPU after stop.
VLLM_STOP_TIMEOUT = float(os.environ.get("VLLM_STOP_TIMEOUT", "60"))
_POLL_SECONDS = 3.0


def _default_check() -> bool:
    """True when the vLLM server answers /v1/models."""
    from web.app import VLM_SERVER_URL  # late import: avoid cycle at module load
    try:
        req = urllib.request.Request(f"{VLM_SERVER_URL}/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _default_run_script(name: str) -> None:
    script = _PROJECT_ROOT / "web" / f"{'start_vllm.sh' if name == 'start' else 'stop_vllm.sh'}"
    proc = await asyncio.create_subprocess_exec(
        str(script),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


async def ensure_started(
    *,
    check_fn: Callable[[], bool] = _default_check,
    run_script: Callable[[str], Awaitable[None]] = _default_run_script,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ready_timeout: float = VLLM_START_TIMEOUT,
    now: Callable[[], float] | None = None,
) -> bool:
    """Bring the vLLM server up; True when it is serving, False on timeout."""
    if check_fn():
        return True
    await run_script("start")
    loop = asyncio.get_event_loop()
    clock = now or loop.time
    deadline = clock() + ready_timeout
    while clock() < deadline:
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
) -> bool:
    """Take the vLLM server down; True when it stopped serving, False on timeout."""
    if not check_fn():
        return True
    await run_script("stop")
    loop = asyncio.get_event_loop()
    clock = now or loop.time
    deadline = clock() + ready_timeout
    while clock() < deadline:
        if not check_fn():
            return True
        await sleep(_POLL_SECONDS)
    return False

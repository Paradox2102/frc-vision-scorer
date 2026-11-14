"""glibc malloc tuning for the long-lived app server (host RAM, not VRAM).

Why: one server process runs job after job, and every job churns many
multi-MB host buffers (decoded video frames, SAM3 frame tensors, crops,
numpy/torch temporaries) from dozens of threads (SAM3 frame-load pool, torch
intra-op pool, asyncio executor, analysis prefetch). glibc's defaults turn
that into permanent RSS:

* the mmap threshold is dynamic: it starts at 128 KiB, but every free() of an
  mmapped chunk raises it to that chunk's size (up to 32 MiB). After the first
  few frames, 1-32 MiB buffers stop being mmapped (returned to the kernel on
  free) and are carved out of malloc arenas instead;
* each thread gets its own arena (up to 8 x cores of them), each a chain of
  64 MiB heaps, and a heap only shrinks from its top, so freed space behind a
  live chunk stays resident until the process exits.

Measured Nov 14 2026 on the local 3090 box (2026cur_qm8, auto_period_only,
main 5c22e6c6): idle RSS 0.57 GiB -> 10.0 GiB after job 1 -> 16.2 GiB
(RSS+swap) after job 2, while mallinfo2 showed only 2.2 GiB in use and
13.7 GiB free inside 354 thread-arena heaps; Python-reachable numpy/torch
CPU data was ~0.1 GiB. malloc_trim(0) on that heap ran >15 CPU-minutes
without returning anything, so trimming after a job is not a fix.

Fix: pin the mmap threshold at process start (``mallopt(M_MMAP_THRESHOLD)``
also switches the dynamic adjustment off), so every buffer >= the threshold
is mmapped and handed back to the kernel the moment it is freed, and cap
the arena count so small-object fragmentation cannot multiply by the thread
count. An explicit ``MALLOC_MMAP_THRESHOLD_`` / ``MALLOC_ARENA_MAX`` in the
environment (read by glibc itself at startup) always wins.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

# glibc <malloc.h> mallopt parameters
M_MMAP_THRESHOLD = -3
M_ARENA_MAX = -8

DEFAULT_MMAP_THRESHOLD = 1 << 20     # 1 MiB
DEFAULT_ARENA_MAX = 4


def _env_int(env, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _load_libc():
    try:
        name = ctypes.util.find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(name)
        libc.mallopt  # glibc only (musl / macOS have no mallopt)
        return libc
    except (OSError, AttributeError):
        return None


def configure_malloc(env=None, libc=None) -> dict:
    """Apply the server's malloc settings; returns what was applied.

    ``FRC_MALLOC_MMAP_THRESHOLD`` / ``FRC_MALLOC_ARENA_MAX`` override the
    defaults (0 = leave glibc's default alone). glibc's own
    ``MALLOC_MMAP_THRESHOLD_`` / ``MALLOC_ARENA_MAX`` env vars, when set,
    win and this does nothing for that parameter. Never raises."""
    env = os.environ if env is None else env
    applied: dict = {}
    libc = _load_libc() if libc is None else libc
    if libc is None:
        applied["skipped"] = "no glibc mallopt"
        return applied
    plan = []
    if "MALLOC_MMAP_THRESHOLD_" not in env:
        plan.append(("mmap_threshold", M_MMAP_THRESHOLD,
                     _env_int(env, "FRC_MALLOC_MMAP_THRESHOLD", DEFAULT_MMAP_THRESHOLD)))
    if "MALLOC_ARENA_MAX" not in env:
        plan.append(("arena_max", M_ARENA_MAX,
                     _env_int(env, "FRC_MALLOC_ARENA_MAX", DEFAULT_ARENA_MAX)))
    for key, param, value in plan:
        if value <= 0:
            continue
        try:
            if libc.mallopt(param, int(value)) == 1:   # 1 = success
                applied[key] = int(value)
        except Exception:
            pass
    return applied

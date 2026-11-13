"""CPU budget for this process: how many CPUs we can really use.

On a container host (RunPod) ``os.cpu_count()`` reports the whole machine's
vCPUs while a cgroup quota caps what this process may actually run. Torch's
default intra-op pool sizes itself from the physical core count, so it can
oversubscribe the quota and get CFS-throttled. ``effective_cpu_count`` folds
the sched affinity mask and the cgroup quota (v2, then v1) into one number;
``configure_torch_threads`` sets torch's pool to min(physical cores,
that count); ``seed_omp_env`` stops ultralytics pinning OpenMP to 1 thread.
"""

from __future__ import annotations

import functools
import math
import os

_CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _affinity_count() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, int(n))


def _cgroup_quota_cpus() -> int | None:
    """CPUs allowed by the cgroup CFS quota (ceil), or None when unlimited
    or unreadable. cgroup v2 ``cpu.max`` = "<quota|max> <period>"; v1 =
    ``cpu.cfs_quota_us`` (-1 = unlimited) / ``cpu.cfs_period_us``."""
    v2 = _read(_CGROUP_V2_CPU_MAX)
    if v2:
        parts = v2.split()
        if parts and parts[0] == "max":
            return None
        try:
            quota = int(parts[0])
            period = int(parts[1]) if len(parts) > 1 else 100000
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
        except (ValueError, IndexError):
            pass
        return None
    q1 = _read(_CGROUP_V1_QUOTA)
    p1 = _read(_CGROUP_V1_PERIOD)
    if q1 is None or p1 is None:
        return None
    try:
        quota, period = int(q1), int(p1)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:  # -1 = unlimited
        return None
    return max(1, math.ceil(quota / period))


@functools.lru_cache(maxsize=1)
def _budget() -> tuple[int, int | None, int]:
    aff = _affinity_count()
    quota = _cgroup_quota_cpus()
    eff = aff if quota is None else min(aff, quota)
    return aff, quota, max(1, eff)


def effective_cpu_count() -> int:
    """CPUs this process can really use: min(sched affinity, cgroup quota),
    never < 1. Cached for the process lifetime."""
    return _budget()[2]


def _physical_cores() -> int | None:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        return int(n) if n else None
    except Exception:
        return None


def torch_thread_target() -> int:
    """Torch intra-op threads we want: ``FRC_TORCH_THREADS`` if set, else
    min(physical cores, effective CPUs) -- torch's own default (physical
    cores), capped by the cgroup quota/affinity. Never < 1."""
    override = os.environ.get("FRC_TORCH_THREADS", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    eff = effective_cpu_count()
    phys = _physical_cores() or max(1, math.ceil(_budget()[0] / 2))
    return max(1, min(eff, phys))


def seed_omp_env() -> bool:
    """Set ``OMP_NUM_THREADS`` to ``torch_thread_target()`` if unset.

    Must run before ultralytics (or torch) is first imported:
    ``ultralytics/__init__.py`` sets OMP_NUM_THREADS=1 when it is unset ("for
    training"), which pins torch's CPU ops -- and every other OpenMP user in
    the process -- to a single thread for the server's lifetime. An explicit
    shell value still wins. Returns True when it set the variable."""
    if os.environ.get("OMP_NUM_THREADS"):
        return False
    os.environ["OMP_NUM_THREADS"] = str(torch_thread_target())
    return True


def configure_torch_threads() -> str:
    """Set torch's intra-op pool to ``torch_thread_target()`` (lowers it under
    a cgroup quota -- oversubscription = CFS throttling -- and restores it if an
    import pinned it to 1). Returns a one-line description for logging."""
    aff, quota, eff = _budget()
    quota_s = "none" if quota is None else str(quota)
    head = f"cpus: affinity={aff} quota={quota_s} effective={eff}"
    try:
        import torch
    except Exception:
        return f"{head} torch=unavailable"
    target = torch_thread_target()
    before = int(torch.get_num_threads())
    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    if before != target:
        torch.set_num_threads(target)
        return f"{head} omp={omp} torch_threads={before}->{target} (set)"
    return f"{head} omp={omp} torch_threads={before} (unchanged)"

"""CPU budget for this process: how many CPUs we can really use.

On a container host (RunPod) ``os.cpu_count()`` reports the whole machine's
vCPUs while a cgroup quota caps what this process may actually run. Torch's
default intra-op pool sizes itself from the physical core count, so it can
oversubscribe the quota and get CFS-throttled. ``effective_cpu_count`` folds
the sched affinity mask and the cgroup quota (v2, then v1) into one number;
``clamp_torch_threads`` lowers torch's pool to it (never raises).
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


def clamp_torch_threads() -> str:
    """Lower torch's intra-op thread count to ``effective_cpu_count()`` if it
    exceeds it (oversubscribing a cgroup quota = throttling). Never raises the
    count, so hosts whose quota covers every core are untouched. Returns a
    one-line description for logging."""
    aff, quota, eff = _budget()
    quota_s = "none" if quota is None else str(quota)
    try:
        import torch
    except Exception:
        return f"cpus: affinity={aff} quota={quota_s} effective={eff} torch=unavailable"
    before = int(torch.get_num_threads())
    if before > eff:
        torch.set_num_threads(eff)
        return (f"cpus: affinity={aff} quota={quota_s} effective={eff} "
                f"torch_threads={before}->{eff} (clamped)")
    return (f"cpus: affinity={aff} quota={quota_s} effective={eff} "
            f"torch_threads={before} (unchanged)")

"""web/malloc_tune.py: pinned mmap threshold + arena cap for the app server."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from web import malloc_tune as mt


class FakeLibc:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def mallopt(self, param, value):
        self.calls.append((param, value))
        return 1 if self.ok else 0


def test_defaults_applied():
    libc = FakeLibc()
    applied = mt.configure_malloc(env={}, libc=libc)
    assert libc.calls == [(mt.M_MMAP_THRESHOLD, mt.DEFAULT_MMAP_THRESHOLD),
                          (mt.M_ARENA_MAX, mt.DEFAULT_ARENA_MAX)]
    assert applied == {"mmap_threshold": mt.DEFAULT_MMAP_THRESHOLD,
                       "arena_max": mt.DEFAULT_ARENA_MAX}


def test_glibc_env_vars_win():
    libc = FakeLibc()
    applied = mt.configure_malloc(
        env={"MALLOC_MMAP_THRESHOLD_": "131072", "MALLOC_ARENA_MAX": "8"}, libc=libc)
    assert libc.calls == [] and applied == {}


def test_frc_overrides_and_zero_disables():
    libc = FakeLibc()
    applied = mt.configure_malloc(
        env={"FRC_MALLOC_MMAP_THRESHOLD": "4194304", "FRC_MALLOC_ARENA_MAX": "0"}, libc=libc)
    assert libc.calls == [(mt.M_MMAP_THRESHOLD, 4194304)]
    assert applied == {"mmap_threshold": 4194304}


def test_bad_values_fall_back_and_failures_not_reported():
    libc = FakeLibc(ok=False)
    applied = mt.configure_malloc(env={"FRC_MALLOC_MMAP_THRESHOLD": "lots"}, libc=libc)
    assert libc.calls[0] == (mt.M_MMAP_THRESHOLD, mt.DEFAULT_MMAP_THRESHOLD)
    assert applied == {}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc only")
def test_real_glibc_returns_freed_thread_buffers_to_kernel():
    """The server's RSS ratchet in miniature, in fresh interpreters: once
    glibc's dynamic mmap threshold has been raised (free of a 16 MiB mmapped
    chunk), 6 MiB buffers malloc'd in worker threads come from per-thread
    arena heaps and stay resident after free() when a small live chunk sits
    behind them. With configure_malloc() they are mmapped and RSS drops back."""
    if mt._load_libc() is None:
        pytest.skip("no glibc mallopt")
    root = str(Path(mt.__file__).resolve().parents[1])
    script = textwrap.dedent("""
        import ctypes, sys, threading
        sys.path.insert(0, %r)
        from web import malloc_tune as mt
        if sys.argv[1] == "tuned":
            assert mt.configure_malloc(env={})["mmap_threshold"] == 1 << 20
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc.restype = ctypes.c_void_p
        libc.free.argtypes = [ctypes.c_void_p]
        def rss():
            for l in open("/proc/self/status"):
                if l.startswith("VmRSS"):
                    return int(l.split()[1]) >> 10
        big = libc.malloc(16 << 20); libc.free(big)   # raises the dynamic threshold
        base = rss()
        pins = []
        def work():
            bufs = []
            for _ in range(8):
                b = libc.malloc(6 << 20)
                ctypes.memset(b, 1, 6 << 20)
                bufs.append(b)
                pins.append(libc.malloc(64))          # small live chunk behind it
            for b in bufs:
                libc.free(b)
        ts = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        print(rss() - base)                            # MiB still resident
    """) % (root,)
    out = {}
    for arm in ("default", "tuned"):
        r = subprocess.run([sys.executable, "-c", script, arm], capture_output=True,
                           text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
        assert r.returncode == 0, r.stderr
        out[arm] = int(r.stdout.strip().splitlines()[-1])
    assert out["default"] > 150, out   # 8 threads x 8 x 6 MiB: ~385 MiB stays resident
    assert out["tuned"] < 32, out      # pinned threshold hands it back

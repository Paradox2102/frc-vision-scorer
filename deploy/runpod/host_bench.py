"""Host fingerprint + single-thread CPU score (runs ON the pod, app venv).

    .venv/bin/python deploy/runpod/host_bench.py [--seconds 8] [--no-gpu]

Prints ONE JSON line. `runpod_cli.py tune` runs it once per pod (cached in
the tune results under "host"; `tune --redo` re-runs it) so every results
file says what CPU the timings came from. Single-thread clock matters more
than core count here (the Nov 12 EPYC pod was ~1.7x slower per thread than a
desktop Ryzen 7600X, and the SAM3 frame loader is one PIL thread).

Benchmarks (all CPU, deterministic inputs, nothing touches the GPU):
  py_mops             pure-Python integer loop, million loop iterations/s
  pil_ms_per_frame    decode a 960x540 JPEG + RGB + bilinear resize to
                      1008x1008, one thread (Pillow only)
  load_ms_per_frame   the full SAM3 frame-loader op, one thread: the above +
                      TF.to_tensor (sam3 io_utils._load_img_as_tensor;
                      load_op says torchvision, or numpy/pil stand-ins)
  thread_speedup      the same op on load_threads = min(16, affinity)
                      threads: throughput / single-thread throughput
Stdlib + Pillow (+ torchvision / numpy when installed).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SAM3_SIZE = 1008
SRC_W, SRC_H = 960, 540


# ------------------------------------------------------------------ parsers (pure)
def parse_cpuinfo(text: str) -> dict:
    model = None
    mhz = []
    for line in text.splitlines():
        k, _, v = line.partition(":")
        k = k.strip()
        if k == "model name" and model is None:
            model = v.strip()
        elif k == "cpu MHz":
            try:
                mhz.append(float(v))
            except ValueError:
                pass
    out = {"model": model}
    if mhz:
        out["mhz_now_mean"] = round(sum(mhz) / len(mhz))
    return out


def parse_lscpu(text: str) -> dict:
    vals = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        vals[k.strip()] = v.strip()
    out = {}
    if vals.get("Model name"):
        out["model"] = vals["Model name"]
    for key, name in (("CPU max MHz", "mhz_max"), ("CPU min MHz", "mhz_min")):
        try:
            out[name] = float(vals[key])
        except (KeyError, ValueError):
            pass
    for key, name in (("Socket(s)", "sockets"), ("Core(s) per socket", "cores_per_socket"),
                      ("Thread(s) per core", "threads_per_core")):
        try:
            out[name] = int(vals[key])
        except (KeyError, ValueError):
            pass
    return out


def parse_cgroup_v2_cpu_max(text: str) -> float | None:
    """'max 100000' -> None (unlimited); '400000 100000' -> 4.0 CPUs."""
    f = text.split()
    if len(f) < 1 or f[0] == "max":
        return None
    try:
        period = float(f[1]) if len(f) > 1 else 100000.0
        return round(float(f[0]) / period, 2) if period > 0 else None
    except ValueError:
        return None


def parse_cgroup_v1_quota(quota_text: str, period_text: str) -> float | None:
    try:
        q, p = int(quota_text.strip()), int(period_text.strip())
    except ValueError:
        return None
    return round(q / p, 2) if q > 0 and p > 0 else None


def parse_meminfo_gb(text: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            try:
                return round(int(line.split()[1]) / 1024 / 1024, 1)
            except (IndexError, ValueError):
                return None
    return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


# ------------------------------------------------------------------ facts
def cgroup_cpu_quota() -> tuple[float | None, str]:
    t = _read("/sys/fs/cgroup/cpu.max")
    if t is not None:
        return parse_cgroup_v2_cpu_max(t), "v2"
    for d in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        q, p = _read(f"{d}/cpu.cfs_quota_us"), _read(f"{d}/cpu.cfs_period_us")
        if q is not None and p is not None:
            return parse_cgroup_v1_quota(q, p), "v1"
    return None, "none"


def cgroup_mem_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        t = _read(p)
        if t is None:
            continue
        t = t.strip()
        if t == "max":
            return None
        try:
            v = int(t)
        except ValueError:
            return None
        return None if v >= 1 << 60 else round(v / 1024 ** 3, 1)
    return None


def cpu_facts() -> dict:
    out = parse_cpuinfo(_read("/proc/cpuinfo") or "")
    try:
        ls = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10,
                            env={**os.environ, "LC_ALL": "C"}).stdout
        lsd = parse_lscpu(ls)
        out = {**lsd, **{k: v for k, v in out.items() if v is not None}}
    except (OSError, subprocess.SubprocessError):
        pass
    if not out.get("model"):
        out["model"] = platform.processor() or platform.machine()
    out["logical"] = os.cpu_count()
    try:
        out["affinity"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        out["affinity"] = os.cpu_count()
    out["cgroup_cpu_quota"], out["cgroup"] = cgroup_cpu_quota()
    return out


def gpu_facts() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        line = (r.stdout.strip().splitlines() or [""])[0]
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 3:
            return {"name": f[0], "driver": f[1], "vram_mib": int(float(f[2])),
                    "count": len(r.stdout.strip().splitlines())}
        return {"error": (r.stderr or r.stdout).strip()[:200] or "no output"}
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


# ------------------------------------------------------------------ benchmarks
def py_loop(seconds: float) -> float:
    """Million loop iterations per second of a pure-Python integer loop."""
    n_iter = 200_000
    done = 0
    t0 = time.perf_counter()
    while True:
        s = 0
        for i in range(n_iter):
            s += i * i % 7
        done += n_iter
        el = time.perf_counter() - t0
        if el >= seconds:
            return round(done / el / 1e6, 2)


def make_jpeg(quality: int = 90) -> bytes:
    """Deterministic 960x540 'broadcast-like' JPEG: smooth colour blobs from
    seeded noise (bicubic upscale) over a gradient, so it compresses like a
    real frame rather than like pure noise or a flat image."""
    from PIL import Image
    rnd = random.Random(1234)
    small = Image.frombytes("RGB", (96, 54), rnd.randbytes(96 * 54 * 3))
    img = small.resize((SRC_W, SRC_H), Image.BICUBIC)
    grad = Image.linear_gradient("L").resize((SRC_W, SRC_H)).convert("RGB")
    img = Image.blend(img, grad, 0.35)
    fine = Image.frombytes("RGB", (480, 270), rnd.randbytes(480 * 270 * 3)).resize((SRC_W, SRC_H), Image.NEAREST)
    img = Image.blend(img, fine, 0.12)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def pil_op(jpeg: bytes):
    """Decode + RGB + resize only (pure Pillow)."""
    from PIL import Image
    return Image.open(io.BytesIO(jpeg)).convert("RGB").resize((SAM3_SIZE, SAM3_SIZE), Image.BILINEAR)


def make_load_op():
    """The SAM3 frame-loader op, exactly as sam3.model.io_utils._load_img_as_tensor
    does it (PIL open -> RGB -> TF.resize(1008, 1008) -> TF.to_tensor) when
    torchvision is importable (the app venv), else Pillow + numpy as a close
    stand-in. torch intra-op threads are pinned to 1 so this is a
    single-thread number. Returns (fn, name)."""
    try:
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        torch.set_num_threads(1)

        def load_tv(jpeg: bytes):
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            img = TF.resize(img, size=(SAM3_SIZE, SAM3_SIZE))
            return TF.to_tensor(img)
        return load_tv, "torchvision"
    except ImportError:
        pass
    try:
        import numpy as np

        def load_np(jpeg: bytes):
            a = np.asarray(pil_op(jpeg), dtype=np.uint8).transpose(2, 0, 1).astype(np.float32)
            a /= 255.0
            return a
        return load_np, "numpy"
    except ImportError:
        return pil_op, "pil"


def time_single(fn, jpeg: bytes, seconds: float) -> tuple[float, int]:
    """ms per call on this thread (at least 5 calls)."""
    fn(jpeg)                              # warm-up (imports, allocator)
    n = 0
    t0 = time.perf_counter()
    while True:
        fn(jpeg)
        n += 1
        el = time.perf_counter() - t0
        if el >= seconds and n >= 5:
            return el / n * 1000.0, n


def time_threads(fn, jpeg: bytes, threads: int, seconds: float) -> tuple[float, int]:
    """Wall ms per frame with `threads` workers (throughput)."""
    per_batch = max(threads * 4, 16)
    n = 0
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(fn, [jpeg] * threads))                  # warm the pool
        t0 = time.perf_counter()
        while True:
            list(ex.map(fn, [jpeg] * per_batch))
            n += per_batch
            el = time.perf_counter() - t0
            if el >= seconds:
                return el / n * 1000.0, n


def run(seconds: float = 8.0, gpu: bool = True, threads: int | None = None) -> dict:
    t_start = time.time()
    cpu = cpu_facts()
    out: dict = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "kernel": platform.release(),
        "cpu": cpu,
        "ram_gb": parse_meminfo_gb(_read("/proc/meminfo") or ""),
        "cgroup_mem_gb": cgroup_mem_gb(),
    }
    out["gpu"] = gpu_facts() if gpu else {"skipped": True}
    try:
        import PIL
        out["pil"] = PIL.__version__
    except ImportError:
        out["pil"] = None
    bench: dict = {"seconds": seconds}
    bench["py_mops"] = py_loop(seconds * 0.25)
    if out["pil"]:
        jpeg = make_jpeg()
        bench["jpeg_bytes"] = len(jpeg)
        ms, _ = time_single(pil_op, jpeg, seconds * 0.15)
        bench["pil_ms_per_frame"] = round(ms, 2)                # decode + resize only
        load, bench["load_op"] = make_load_op()
        ms1, n1 = time_single(load, jpeg, seconds * 0.3)
        bench["load_ms_per_frame"] = round(ms1, 2)              # full SAM3 frame-loader op
        bench["load_frames_1t"] = n1
        nt = threads or max(1, min(16, cpu.get("affinity") or 1))
        msn, nn = time_threads(load, jpeg, nt, seconds * 0.3)
        bench["load_threads"] = nt
        bench["load_ms_per_frame_nt"] = round(msn, 2)
        bench["load_frames_nt"] = nn
        bench["thread_speedup"] = round(ms1 / msn, 2) if msn > 0 else None
    out["bench"] = bench
    out["elapsed_s"] = round(time.time() - t_start, 1)
    out["measured_at"] = round(t_start)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8.0, help="total benchmark budget (default 8)")
    ap.add_argument("--threads", type=int, default=None, help="thread count for the parallel PIL run")
    ap.add_argument("--no-gpu", action="store_true", help="skip the nvidia-smi query")
    a = ap.parse_args(argv)
    try:
        res = run(a.seconds, gpu=not a.no_gpu, threads=a.threads)
    except Exception as e:  # always one JSON line for the CLI
        res = {"error": f"{type(e).__name__}: {e}"[:400]}
    print(json.dumps(res, separators=(",", ":")))
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main())

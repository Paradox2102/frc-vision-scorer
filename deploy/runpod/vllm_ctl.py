"""Pod-side vLLM reader control for `tune` (runs ON the pod, app venv).

    python deploy/runpod/vllm_ctl.py start qwen3.5-4b [--timeout 1200]
    python deploy/runpod/vllm_ctl.py stop

`start` goes through the app's own lifecycle (web.vllm_lifecycle.ensure_started
with the model active), so the served model, gpu-memory-utilization (absolute
GiB budget on big cards via vlm_models.serve_util) and extra serve args are
exactly what a job's in-pipeline ensure_started would use -- the job then finds
the right model up and returns at once, keeping the ~106 s cold start (and any
first-time weight download) out of the arm's measured wall time. It also runs
vLLM's memory profiling while the GPU is otherwise idle (on Nov 12 a start
overlapping another process's allocations crashed vLLM's profiler).

Prints one JSON line: {"ok": bool, "seconds": float, "model": str|null, ...}.
Run with the server's environment (pod_env.sh + profile.env), as runpod_cli
does over ssh.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _start(model_key: str, timeout: float) -> dict:
    from web import vlm_models, vllm_lifecycle as vl

    vlm_models.get(model_key)  # KeyError on an unknown key, before any launch
    vl.set_active_model(model_key)
    t0 = time.time()
    ok = asyncio.run(vl.ensure_started(ready_timeout=timeout))
    return {"ok": bool(ok), "seconds": round(time.time() - t0, 1),
            "model": vl._default_served_model(), "requested": vlm_models.get(model_key).hf_repo}


def _stop() -> dict:
    from web import vllm_lifecycle as vl

    t0 = time.time()
    asyncio.run(vl._default_run_script("stop"))
    return {"ok": True, "seconds": round(time.time() - t0, 1), "model": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("model_key")
    s.add_argument("--timeout", type=float, default=1200.0,
                   help="readiness timeout incl. a first-time weight download (s)")
    sub.add_parser("stop")
    a = ap.parse_args(argv)
    try:
        out = _start(a.model_key, a.timeout) if a.cmd == "start" else _stop()
    except Exception as e:  # report, don't traceback over ssh
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out), flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

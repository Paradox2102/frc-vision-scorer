"""Run ONE timed inference job on the pod (runs ON the pod, with the app venv).

Launched detached by ``runpod_cli.py tune`` so a dropped ssh session or a
closed laptop never kills the job driver (the job runs inside the websocket
handler of the connection that claimed it — scripts/ab_qm8.py drive()).

    .venv/bin/python deploy/runpod/pod_job.py --video /workspace/tune/2026cur_qm8.mp4 \
        --match-key 2026cur_qm8 --event 2026cur --config-json '{"sam3_track_fps": 8}' \
        --out /workspace/tune/jobs/R2.json

Writes ``--out`` atomically and repeatedly (status / progress / timings), so the
local CLI can poll it. Wall times:
  wall_job_s    submit -> websocket "complete" (tracking + analysis + finalize core)
  wall_final_s  submit -> replay row + meta.json carry the window_labels summary
                (the finalize relabel stage, default ON, runs after "complete";
                this is when the match's labelled result actually exists)
If the websocket drops before "complete" (seen Nov 13 with 2 concurrent jobs)
the job keeps running on the server, so the driver keeps waiting for the
final replay (up to --final-timeout) instead of failing; wall_job_s is then
None and ws_status "closed".
Peak VRAM is the max of ``nvidia-smi memory.used`` sampled every 2 s from
submit to wall_final (whole card: includes the idle server's resident models).

Profiling (default ON, ``--no-profile`` to skip): from just before submit to
wall_final (or to the error), ``py-spy record --gil --rate 20`` is attached to
the app server (whoever LISTENs on the app port) and ``nvidia-smi`` logs GPU
util/memory at 1 Hz; raw files land next to ``--out`` as
``<label>.pyspy.txt`` / ``<label>.gpu.csv`` and the summary (profiling.py) is
written into the result as ``profile`` BEFORE the final status, so the CLI
sees it the moment it sees "done". A profiler failure is recorded, never fatal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import profiling  # noqa: E402  (deploy/runpod/profiling.py, stdlib only)

SERVER_PIDFILE = os.environ.get("FRC_LOGS", "/workspace/logs") + "/server.pid"   # serve.sh


class Recorder:
    def __init__(self, out: Path, base: dict):
        self.out = out
        self.data = dict(base)
        self.lock = threading.Lock()

    def update(self, **kw) -> None:
        with self.lock:
            self.data.update(kw)
            self.data["updated_at"] = time.time()
            tmp = self.out.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            os.replace(tmp, self.out)


class VramSampler(threading.Thread):
    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.n = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
                if out:
                    self.peak = max(self.peak, int(float(out[0])))
                    self.n += 1
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def http_json(url: str, body: dict | None = None, timeout: float = 300) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def replay_row(job_id: str) -> dict | None:
    db = APP / "frc_scorer.db"
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute("SELECT * FROM replays WHERE id = ?", (job_id,)).fetchone()
        finally:
            con.close()
        return dict(row) if row else None
    except sqlite3.Error:
        return None


def final_ready(job_id: str, want_window_labels: bool) -> bool:
    if replay_row(job_id) is None:
        return False
    meta = APP / "web" / "replays" / job_id / "meta.json"
    if not meta.exists():
        return False
    if not want_window_labels:
        return True
    try:
        summary = json.loads(meta.read_text()).get("summary") or {}
    except (OSError, ValueError):
        return False
    return "window_labels" in summary


def wait_final(job_id: str, want_window_labels: bool, timeout: float, poll_s: float = 2.0) -> bool:
    """Poll final_ready() until it holds or ``timeout`` seconds pass."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if final_ready(job_id, want_window_labels):
            return True
        time.sleep(poll_s)
    return final_ready(job_id, want_window_labels)


async def drive(host: str, job_id: str, rec: Recorder) -> str:
    import websockets  # app venv
    saw_complete = False
    last = 0.0
    async with websockets.connect(f"ws://{host}/ws/infer/{job_id}", max_size=None,
                                  ping_interval=20, ping_timeout=1200) as ws:
        while True:
            try:
                msg = json.loads(await ws.recv())
            except websockets.ConnectionClosed:
                if saw_complete or replay_row(job_id) is not None:
                    return "completed"
                return "closed"
            except (TypeError, ValueError):
                continue
            t = msg.get("type")
            if t == "status":
                now = time.time()
                if now - last > 5:
                    last = now
                    rec.update(progress=msg.get("progress"), message=str(msg.get("message"))[:200])
                if msg.get("status") in ("completed", "error"):
                    if msg.get("status") == "error":
                        rec.update(error=str(msg.get("error") or msg.get("message"))[:500])
                    return msg.get("status")
            elif t == "complete":
                saw_complete = True
            elif t == "error":
                rec.update(error=str(msg.get("error"))[:500])
                return "error"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--match-key", required=True)
    ap.add_argument("--event", required=True)
    ap.add_argument("--config-json", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--host", default="127.0.0.1:8004")
    ap.add_argument("--final-timeout", type=float, default=900.0)
    ap.add_argument("--no-profile", action="store_true", help="skip py-spy + nvidia-smi profiling")
    ap.add_argument("--pyspy-rate", type=int, default=20)
    ap.add_argument("--allow-busy", action="store_true",
                    help="submit even if another job is active (concurrency test; "
                         "needs MAX_CONCURRENT_JOBS > 1 on the server)")
    a = ap.parse_args()

    cfg = json.loads(a.config_json)
    cfg.setdefault("auto_period_only", False)   # full match
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out, {"label": a.label, "config": cfg, "match_key": a.match_key,
                         "pid": os.getpid(), "status": "starting", "started_at": time.time()})
    base = f"http://{a.host}"
    prof = None
    # setsid'd + nohup'd: SIGHUP never arrives, but a `kill` must still stop
    # py-spy / nvidia-smi (finally below) instead of orphaning them.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    def final(**kw) -> None:
        """Terminal result: profile summary first, then the status the CLI waits on."""
        if prof is not None:
            kw["profile"] = prof.summary()
        rec.update(**kw)

    try:
        active = http_json(f"{base}/api/job/active", timeout=30)
        if active.get("job_id") and not a.allow_busy:
            rec.update(status="error", error=f"server busy with job {active['job_id']}")
            return 2
        rec.update(status="loading_event")
        ev = http_json(f"{base}/api/event/load", {"event_key": a.event}, timeout=300)
        if "event_key" not in ev:
            rec.update(status="error", error=f"event load failed: {str(ev)[:300]}")
            return 2

        sampler = VramSampler()
        sampler.start()
        if not a.no_profile:
            label = a.label or out.stem
            prof = profiling.JobProfiler(out.parent, label, int(a.host.rsplit(":", 1)[1]),
                                         rate=a.pyspy_rate, pidfile=SERVER_PIDFILE)
            rec.update(profiler=prof.start())
        t0 = time.time()
        res = subprocess.run(
            ["curl", "-s", "-F", f"video_file=@{a.video}", "-F", f"match_key={a.match_key}",
             "-F", f"config_json={json.dumps(cfg)}", f"{base}/api/infer"],
            capture_output=True, text=True, timeout=600)
        try:
            job_id = json.loads(res.stdout)["job_id"]
        except (ValueError, KeyError):
            sampler.stop()
            final(status="error", error=f"submit failed: {res.stdout[:300]} {res.stderr[:200]}")
            return 2
        rec.update(status="running", job_id=job_id, t_submit=t0)
        status = asyncio.run(drive(a.host, job_id, rec))
        wall_job = time.time() - t0
        # "closed": the websocket dropped before "complete" but the job keeps
        # running on the server (seen Nov 13 with 2 concurrent jobs). Keep
        # waiting for its replay below (up to --final-timeout) instead of
        # failing a job that is still working; wall_job_s is then unknown.
        ok = status in ("completed", "closed")
        rec.update(status="finalizing" if ok else "error",
                   ws_status=status, wall_job_s=round(wall_job, 1) if status == "completed" else None,
                   peak_vram_mib=sampler.peak,
                   **({"message": "websocket closed; job still running, waiting for its replay"}
                      if status == "closed" else {}))
        if not ok:
            sampler.stop()
            final(status="error", error=rec.data.get("error") or f"job ended: {status}")
            return 1
        want_wl = bool(cfg.get("window_labels", True))
        if not wait_final(job_id, want_wl, a.final_timeout):
            if status == "closed":
                rec.update(error=f"websocket closed and no final replay after {a.final_timeout:.0f}s")
        sampler.stop()
        ready = final_ready(job_id, want_wl)
        row = replay_row(job_id) or {}
        timings = {}
        try:
            timings = json.loads(row.get("timings") or "{}")
        except (TypeError, ValueError):
            pass
        summary_wl = None
        try:
            meta = json.loads((APP / "web" / "replays" / job_id / "meta.json").read_text())
            summary_wl = (meta.get("summary") or {}).get("window_labels")
        except (OSError, ValueError):
            pass
        final(
            status="done" if ready else "error",
            error=None if ready else (rec.data.get("error")
                                      or f"replay not final after {a.final_timeout:.0f}s"),
            wall_final_s=round(time.time() - t0, 1) if ready else None,
            peak_vram_mib=sampler.peak, vram_samples=sampler.n,
            timings=timings,
            window_labels_aborted=(summary_wl or {}).get("aborted") if isinstance(summary_wl, dict) else None,
            replay={k: row.get(k) for k in ("id", "match_key", "event_key", "created_at",
                                             "blue_teams", "red_teams", "status")},
        )
        return 0 if ready else 1
    except BaseException as e:  # recorded for the CLI, then re-raised for the log
        try:
            final(status="error", error=f"{type(e).__name__}: {e}"[:500])
        except Exception:
            rec.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
        raise
    finally:
        if prof is not None:
            prof.stop()      # idempotent; normally already stopped by final()


if __name__ == "__main__":
    raise SystemExit(main())

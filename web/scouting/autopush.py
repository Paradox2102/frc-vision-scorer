"""Push scouting metrics to RoboRecon automatically after each match analysis.

``schedule(event_key)`` is called from the job finaliser right after a replay is
saved as ``complete``. It never blocks or raises: the push (metric compute +
HTTP) runs in a daemon thread and failures are only logged.

Pushes are coalesced per event: if one is already running when another match
finishes, the event is marked dirty and ONE more push runs afterwards. Every
push recomputes the whole event from the newest replay per match, so the final
push covers every match analysed in the meantime, and a re-analysed match
replaces its old result in RoboRecon rather than adding to it.

Env (repo .env):
  ROBORECON_AUTO_PUSH=1              turn it on (off by default)
  ROBORECON_AUTO_PUSH_METRICS=...    comma list, default "auto_paths"
  ROBORECON_URL / ROBORECON_TOKEN    where to push (see export.RoboReconSink)
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from web.scouting import export

DEFAULT_METRICS = ("auto_paths",)

_lock = threading.Lock()
_state: dict = {}     # event_key -> "running" | "pending"
_last: dict = {}      # event_key -> summary of the last finished auto-push


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _under_pytest() -> bool:
    """Never push for real from a test run that finalises a job (tests patch this)."""
    return "PYTEST_CURRENT_TEST" in os.environ


def config() -> dict:
    export._load_dotenv()
    metrics = [m.strip() for m in os.environ.get("ROBORECON_AUTO_PUSH_METRICS", "").split(",")
               if m.strip()] or list(DEFAULT_METRICS)
    configured = bool(os.environ.get("ROBORECON_URL") and os.environ.get("ROBORECON_TOKEN"))
    enabled = (_truthy(os.environ.get("ROBORECON_AUTO_PUSH")) and configured
               and not _under_pytest())
    return {"enabled": enabled, "configured": configured, "metrics": metrics}


def schedule(event_key: Optional[str]) -> bool:
    """Queue an auto-push for ``event_key``. Returns whether one was queued."""
    if not event_key:
        return False
    cfg = config()
    if not cfg["enabled"]:
        return False
    with _lock:
        if event_key in _state:
            _state[event_key] = "pending"           # coalesce into one re-run
            return True
        _state[event_key] = "running"
    threading.Thread(target=_worker, args=(event_key, cfg["metrics"]), daemon=True,
                     name=f"roborecon-push-{event_key}").start()
    return True


def _push_once(event_key: str, metrics: list) -> dict:
    from web.scouting.api import push_event          # late: api imports this module's peers
    t0 = time.perf_counter()
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        out = push_event(event_key, metrics)
        summary = {"at": at, "ok": out["ok"], "stored": out["stored"],
                   "rejected": out["rejected"], "metrics": out["metrics"],
                   "seconds": round(time.perf_counter() - t0, 1)}
        if out["rejected"]:
            summary["errors"] = [e for r in out["results"] if r.get("status") != "stored"
                                 for e in (r.get("errors") or [])][:5]
        status = "ok" if out["ok"] else f"{out['rejected']} rejected {summary.get('errors')}"
        print(f"[roborecon] auto-push {event_key} {','.join(metrics)}: "
              f"{out['stored']} stored, {status} ({summary['seconds']} s)")
    except Exception as e:                           # never let a push hurt analysis
        summary = {"at": at, "ok": False, "metrics": list(metrics),
                   "error": f"{type(e).__name__}: {e}"}
        print(f"[roborecon] auto-push {event_key} failed: {summary['error']}")
    return summary


def _worker(event_key: str, metrics: list) -> None:
    while True:
        summary = _push_once(event_key, metrics)
        with _lock:
            _last[event_key] = summary
            if _state.get(event_key) == "pending":
                _state[event_key] = "running"
                continue
            _state.pop(event_key, None)
            return


def status() -> dict:
    """Config + the last auto-push per event (for /api/scouting/roborecon/status)."""
    with _lock:
        return {**config(), "in_flight": dict(_state), "last": dict(_last)}

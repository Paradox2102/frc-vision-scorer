"""Upload match results to a Google Drive folder via rclone after each analysis.

Turned on per job by the ``gdrive_upload`` knob (web/job_config.py, section
"Uploads"); an event batch sets it for every match with
``swarm run --name N --local --config-json '{"gdrive_upload": true}'``. The job
finaliser calls ``schedule_match`` once the replay and its identity rows are
saved. Nothing here blocks the job or raises into it: uploads run in daemon
threads and failures are only logged (``[gdrive] ...``) and kept for
``status()`` (GET /api/gdrive/status).

Env (repo .env, loaded like the RoboRecon settings):
  GDRIVE_FOLDER_ID        the Drive folder to upload into (required): the last
                          part of the folder's URL
  GDRIVE_RCLONE_REMOTE    the rclone remote, default ``gdrive:`` (a configured
                          remote of type drive; ``rclone config`` to make one)

Every rclone call passes ``--drive-root-folder-id <GDRIVE_FOLDER_ID>``, so the
paths below are relative to that folder:

  <event>/<event>_scouting_export.json      the event export (compact JSON, as
                                            scripts/export_scouting.py writes it)
  <event>/<event>_matches/<match_key>.json  the single-match file (pretty JSON,
                                            as export_scouting.py --per-match)
  <event>/videos/<match_key>.mp4            the HUD replay video (gdrive_upload_video)

``rclone copyto`` makes missing folders and replaces a same-name file in place,
so re-analysing a match updates its files and their Drive links stay the same.

Uploads are coalesced per event: one worker thread per event uploads every
pending match file, then the event export ONCE for the batch, and loops while
more matches arrive. Videos (~100 MB) go through one global serial worker so
they never compete with each other for bandwidth. With hud_post_render on, the
video only exists after the render subprocess: the finaliser registers it with
``expect_video_after_render`` and the render worker hands it over through
``on_render_done``. That hand-off is in memory only, so a server restart drops
videos still waiting for their render; re-send them with
``scripts/upload_gdrive.py <event> --video``, which is also how results pulled
from RunPod / vast.ai runs (imported locally) are uploaded.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_REMOTE = "gdrive:"
JSON_TIMEOUT_S = 300
VIDEO_TIMEOUT_S = 1800

_lock = threading.Lock()
_pending: dict = {}        # event_key -> {match_key: None} waiting for the event worker
_running: set = set()      # events with a live worker thread
_in_flight: dict = {}      # event_key -> match keys of the batch being uploaded
_last: dict = {}           # event_key -> summary of the last finished batch
_last_video: dict = {}     # event_key -> summary of the last video upload
_video_jobs: deque = deque()     # (event_key, match_key, video_path)
_video_state: dict = {"running": False, "current": None}
_expected: dict = {}       # replay_id -> (event_key, match_key): video after its render


class GDriveError(RuntimeError):
    """An rclone call failed (non-zero exit, timeout, not runnable)."""


def _under_pytest() -> bool:
    """Never run rclone for real from a test run (tests patch this)."""
    return "PYTEST_CURRENT_TEST" in os.environ


# ----------------------------------------------------------------- config
def _folder_id() -> str:
    return (os.environ.get("GDRIVE_FOLDER_ID") or "").strip()


def _remote() -> str:
    """The rclone remote with its separator: ``gdrive`` -> ``gdrive:``,
    ``gdrive:sub`` -> ``gdrive:sub/``."""
    r = (os.environ.get("GDRIVE_RCLONE_REMOTE") or "").strip() or DEFAULT_REMOTE
    if ":" not in r:
        return r + ":"
    return r if r.endswith((":", "/")) else r + "/"


def config() -> dict:
    from web.scouting import export
    export._load_dotenv()
    folder_id = _folder_id()
    rclone = shutil.which("rclone")
    configured = bool(folder_id) and rclone is not None
    return {"configured": configured, "enabled": configured and not _under_pytest(),
            "folder_id_set": bool(folder_id), "remote": _remote(), "rclone": rclone}


def _why_not(cfg: dict) -> str:
    why = []
    if not cfg["folder_id_set"]:
        why.append("GDRIVE_FOLDER_ID not set")
    if not cfg["rclone"]:
        why.append("rclone not found")
    if cfg["configured"] and not cfg["enabled"]:
        why.append("disabled under pytest")
    return " / ".join(why) or "not configured"


# ----------------------------------------------------------------- remote paths
def event_export_path(event_key: str) -> str:
    return f"{event_key}/{event_key}_scouting_export.json"


def match_path(event_key: str, match_key: str) -> str:
    return f"{event_key}/{event_key}_matches/{match_key}.json"


def video_path(event_key: str, match_key: str) -> str:
    return f"{event_key}/videos/{match_key}.mp4"


def remote_path(rel: str) -> str:
    """``rel`` (relative to the Drive folder) on the rclone remote."""
    return _remote() + rel


# ----------------------------------------------------------------- rclone
def _run(args: list, timeout: float) -> str:
    """Run ``rclone <args> --drive-root-folder-id <folder>``; its stdout.
    Raises GDriveError (with a short stderr tail) on failure. Tests replace it."""
    if _under_pytest():
        raise GDriveError("refusing to run rclone under pytest")
    folder_id = _folder_id()
    if not folder_id:
        raise GDriveError("GDRIVE_FOLDER_ID not set")
    cmd = ["rclone", *args, "--drive-root-folder-id", folder_id]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise GDriveError(f"rclone {args[0]} timed out after {timeout:g} s") from None
    except OSError as e:
        raise GDriveError(f"rclone not runnable: {e}") from None
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])[-400:]
        raise GDriveError(f"rclone {args[0]} exited {proc.returncode}: {tail}")
    return proc.stdout or ""


def _copyto(local: Path, rel: str, timeout: float) -> None:
    _run(["copyto", str(local), remote_path(rel)], timeout)


def list_folder() -> list:
    """Top-level entries of the Drive folder (reachability check)."""
    out = _run(["lsf", _remote(), "--max-depth", "1"], 60)
    return [ln for ln in out.splitlines() if ln.strip()]


# ----------------------------------------------------------------- synchronous uploads
def _upload_text(text: str, name: str, rel: str) -> dict:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gdrive-") as d:
        local = Path(d) / name
        local.write_text(text)
        _copyto(local, rel, JSON_TIMEOUT_S)
    return {"remote": rel, "bytes": len(text.encode()),
            "seconds": round(time.perf_counter() - t0, 1)}


def match_json(event_key: str, match_key: str) -> str:
    """The single-match file's text (GET /api/scouting/{event}/match/{mk})."""
    from web.scouting import api, export
    try:
        return export.pretty(api._match_sync(event_key, match_key, None))
    except Exception as e:                  # HTTPException carries the reason in .detail
        detail = getattr(e, "detail", None)
        if detail:
            raise GDriveError(str(detail)) from None
        raise


def event_export_json(event_key: str) -> str:
    """The event export's text (compact, like scripts/export_scouting.py)."""
    from web.scouting import api
    return json.dumps(api._export_sync(event_key, paths=True), separators=(",", ":"))


def upload_match(event_key: str, match_key: str) -> dict:
    return _upload_text(match_json(event_key, match_key), f"{match_key}.json",
                        match_path(event_key, match_key))


def upload_event_export(event_key: str) -> dict:
    return _upload_text(event_export_json(event_key), f"{event_key}_scouting_export.json",
                        event_export_path(event_key))


def upload_video(event_key: str, match_key: str, local_video) -> dict:
    p = Path(str(local_video or ""))
    if not local_video or not p.is_file():
        raise GDriveError(f"video not found: {local_video}")
    t0 = time.perf_counter()
    rel = video_path(event_key, match_key)
    size = p.stat().st_size
    _copyto(p, rel, VIDEO_TIMEOUT_S)
    return {"remote": rel, "bytes": size, "seconds": round(time.perf_counter() - t0, 1)}


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------- background: match files
def schedule_match(event_key: Optional[str], match_key: Optional[str]) -> bool:
    """Queue the match file + an event-export refresh. Returns whether queued."""
    try:
        if not event_key or not match_key:
            return False
        cfg = config()
        if not cfg["enabled"]:
            print(f"[gdrive] upload skipped: {_why_not(cfg)}")
            return False
        with _lock:
            _pending.setdefault(event_key, {})[match_key] = None
            if event_key in _running:
                return True                 # the event's worker picks it up
            _running.add(event_key)
        try:
            threading.Thread(target=_event_worker, args=(event_key,), daemon=True,
                             name=f"gdrive-{event_key}").start()
        except Exception:
            with _lock:
                _running.discard(event_key)
            raise
        return True
    except Exception as e:
        print(f"[gdrive] {event_key}/{match_key} not queued: {_err(e)}")
        return False


def _upload_batch(event_key: str, batch: list) -> dict:
    t0, at = time.perf_counter(), _now()
    done, errors = [], {}
    for mk in batch:
        try:
            s = upload_match(event_key, mk)
            done.append(mk)
            print(f"[gdrive] {event_key}/{mk}: uploaded {s['remote']} "
                  f"({s['bytes'] / 1e3:.0f} KB, {s['seconds']} s)")
        except Exception as e:              # one bad match never blocks the rest
            errors[mk] = _err(e)
            print(f"[gdrive] {event_key}/{mk}: match file failed: {errors[mk]}")
    export_summary = None
    try:
        export_summary = upload_event_export(event_key)
        print(f"[gdrive] {event_key}: event export uploaded {export_summary['remote']} "
              f"({export_summary['bytes'] / 1e6:.1f} MB, {export_summary['seconds']} s)")
    except Exception as e:
        errors["event_export"] = _err(e)
        print(f"[gdrive] {event_key}: event export failed: {errors['event_export']}")
    return {"at": at, "ok": not errors, "matches": done, "event_export": export_summary,
            "errors": errors, "seconds": round(time.perf_counter() - t0, 1)}


def _event_worker(event_key: str) -> None:
    try:
        while True:
            with _lock:
                batch = list(_pending.pop(event_key, {}))
                if not batch:
                    _in_flight.pop(event_key, None)
                    _running.discard(event_key)
                    return
                _in_flight[event_key] = batch
            summary = _upload_batch(event_key, batch)
            with _lock:
                _last[event_key] = summary
    except Exception as e:                  # a bug here must not wedge the event
        print(f"[gdrive] {event_key}: worker error: {_err(e)}")
        with _lock:
            _in_flight.pop(event_key, None)
            _running.discard(event_key)


# ----------------------------------------------------------------- background: videos
def schedule_video(event_key: Optional[str], match_key: Optional[str], local_video) -> bool:
    """Queue the HUD video on the global serial video worker."""
    try:
        if not event_key or not match_key:
            return False
        cfg = config()
        if not cfg["enabled"]:
            print(f"[gdrive] video upload skipped: {_why_not(cfg)}")
            return False
        if not local_video or not Path(str(local_video)).is_file():
            print(f"[gdrive] {event_key}/{match_key}: video skipped, no file at {local_video}")
            return False
        with _lock:
            _video_jobs.append((event_key, match_key, str(local_video)))
            if _video_state["running"]:
                return True
            _video_state["running"] = True
        try:
            threading.Thread(target=_video_worker, daemon=True, name="gdrive-video").start()
        except Exception:
            with _lock:
                _video_state["running"] = False
            raise
        return True
    except Exception as e:
        print(f"[gdrive] {event_key}/{match_key} video not queued: {_err(e)}")
        return False


def _video_worker() -> None:
    while True:
        with _lock:
            if not _video_jobs:
                _video_state.update(running=False, current=None)
                return
            item = _video_jobs.popleft()
            _video_state["current"] = item
        event_key, match_key, local = item
        at = _now()
        try:
            s = upload_video(event_key, match_key, local)
            summary = {"at": at, "ok": True, "match_key": match_key, **s}
            print(f"[gdrive] {event_key}/{match_key}: video uploaded {s['remote']} "
                  f"({s['bytes'] / 1e6:.0f} MB, {s['seconds']} s)")
        except Exception as e:
            summary = {"at": at, "ok": False, "match_key": match_key, "error": _err(e)}
            print(f"[gdrive] {event_key}/{match_key}: video failed: {summary['error']}")
        with _lock:
            _last_video[event_key] = summary


def expect_video_after_render(replay_id: str, event_key: str, match_key: str) -> None:
    """The video will exist once the HUD post-render of ``replay_id`` finishes."""
    with _lock:
        _expected[str(replay_id)] = (event_key, match_key)
    print(f"[gdrive] {event_key}/{match_key}: video upload waits for the HUD render")


def on_render_done(replay_id: str, status: Optional[str], local_video) -> bool:
    """Render worker hand-off: queue the expected video if the render worked."""
    try:
        with _lock:
            exp = _expected.pop(str(replay_id), None)
        if exp is None:
            return False
        event_key, match_key = exp
        if status != "rendered":
            print(f"[gdrive] {event_key}/{match_key}: video not uploaded (render {status})")
            return False
        return schedule_video(event_key, match_key, local_video)
    except Exception as e:
        print(f"[gdrive] {replay_id}: video not queued: {_err(e)}")
        return False


def status() -> dict:
    """Config + uploads in flight + the last result per event (GET /api/gdrive/status)."""
    cfg = config()
    with _lock:
        in_flight = {ev: {"uploading": list(_in_flight.get(ev, [])),
                          "pending": list(_pending.get(ev, {}))}
                     for ev in sorted(set(_in_flight) | set(_pending))}
        videos_pending = len(_video_jobs) + (1 if _video_state["current"] else 0)
        return {**cfg, "in_flight": in_flight, "last": dict(_last),
                "last_video": dict(_last_video), "videos_pending": videos_pending,
                "videos_awaiting_render": len(_expected)}

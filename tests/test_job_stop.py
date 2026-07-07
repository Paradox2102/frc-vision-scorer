"""Task 1: STOP must kill the job in every phase.

``POST /api/job/{id}/stop`` sets ``job._stopped``. Only the queue-wait and the
analysis loop polled it before; the long pre-loop phases (clock scan,
calibration, model load, frame export, tracker propagation, vLLM startup) now
poll it too via ``_raise_if_stopped`` / the trackers' ``should_stop`` hook.

These tests cover the two app.py seams that don't need a GPU:
  * ``_raise_if_stopped`` raises ``JobStopped`` iff the flag is set;
  * a job flagged ``_stopped`` at a pre-loop phase boundary drives
    ``process_video`` to terminal ``error`` state ("Stopped by user"), sends the
    standard error websocket message, and still runs the ``finally`` cleanup.
"""
import asyncio

import pytest

from web.app import InferenceJob, JobStopped


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in: reports metadata, reads nothing."""

    def __init__(self, *_a, **_k):
        pass

    def get(self, _prop):
        return 0.0

    def set(self, *_a):
        return True

    def read(self):
        return False, None

    def release(self):
        pass


# ---------------------------------------------------------------------------
# _raise_if_stopped unit
# ---------------------------------------------------------------------------

def test_raise_if_stopped_raises_when_flagged():
    job = InferenceJob.__new__(InferenceJob)
    job._stopped = True
    with pytest.raises(JobStopped):
        job._raise_if_stopped()


def test_raise_if_stopped_noop_when_not_flagged():
    job = InferenceJob.__new__(InferenceJob)
    job._stopped = False
    job._raise_if_stopped()  # must not raise


# ---------------------------------------------------------------------------
# Phase-boundary: stop during a pre-loop phase -> terminal error + cleanup
# ---------------------------------------------------------------------------

def _make_stoppable_job(ws, monkeypatch):
    """A job whose heavy setup is faked so process_video reaches the
    clock-scan phase boundary; the faked clock scan flips ``_stopped`` so the
    boundary poll raises JobStopped. Records whether the log file was closed."""
    job = InferenceJob.__new__(InferenceJob)
    job.job_id = "test-stop"
    job.video_path = "/nonexistent/video.mp4"
    job.websocket = ws
    job._websocket_send_lock = asyncio.Lock()
    job._stopped = False
    job._paused = False
    job.status = "processing"
    job.error = None
    job.config = type("C", (), {"gallery_enabled": False,
                                "auto_period_only": False,
                                "vllm_phase_exclusive": False,
                                "non_defaults": lambda self: {}})()
    job.event_key = None
    job.match_key = None
    job.timings = {}
    job._replay = None
    job._adjudicate_task = None
    job._vlm_task = None
    job.diagnostic = type("D", (), {"close": lambda self: None})()

    cleanup = {"log_closed": False, "cap_released": False}

    # Fake the heavy setup so we reach the "after clock scan" poll quickly.
    job._init_field_tracking = lambda: True
    job._open_log_file = lambda: None

    def _close_log():
        cleanup["log_closed"] = True
    job._close_log_file = _close_log

    def _scan():
        # A user STOP arrives mid-scan; the phase boundary right after the
        # scan is the first poll that should see it.
        job._stopped = True
        return []
    job._scan_match_clock = _scan

    async def _drain():
        return None
    job._drain_vlm_task = _drain

    async def _stop_senders(drain=True):
        return None
    job._stop_ws_senders = _stop_senders

    import web.app as app_mod
    monkeypatch.setattr(app_mod.cv2, "VideoCapture", _FakeCap)

    return job, cleanup


def test_stop_before_analysis_loop_is_terminal_error(monkeypatch):
    ws = RecordingWebSocket()

    async def run():
        job, cleanup = _make_stoppable_job(ws, monkeypatch)
        await job.process_video()
        return job, cleanup

    job, cleanup = asyncio.run(run())

    # Terminal error with the standard "Stopped by user" message.
    assert job.status == "error"
    assert job.error == "Stopped by user"
    # Standard error websocket message on the same path as other job errors.
    errors = [m for m in ws.messages if m.get("type") == "error"]
    assert errors and errors[-1]["error"] == "Stopped by user"
    # The finally cleanup still ran even though the phase raised.
    assert cleanup["log_closed"] is True
    # JobStopped never escaped as an unhandled exception (asyncio.run returned).
    assert job.status == "error"  # never left non-terminal

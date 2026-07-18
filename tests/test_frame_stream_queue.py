"""Frame streaming must be decoupled from inference.

Frames go through a latest-only queue drained by a background sender task:
a slow websocket client can never back-pressure the frame loop, display sends
are rate-capped, and a dropped frame can't swallow a fresh VLM crops payload.
"""
import asyncio
import time

from web.app import InferenceJob
from web.job_config import JobConfig


class RecordingWebSocket:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.messages = []

    async def send_json(self, payload):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.messages.append(payload)


def _make_job(websocket):
    job = InferenceJob.__new__(InferenceJob)
    job.websocket = websocket
    job._websocket_send_lock = asyncio.Lock()
    job.config = JobConfig()
    return job


def test_frame_stream_rate_cap():
    job = _make_job(None)
    assert job._frame_stream_due(100.0) is True
    assert job._frame_stream_due(100.05) is False  # within 0.1s window
    assert job._frame_stream_due(100.11) is True
    assert job._frame_stream_due(100.15) is False


def test_enqueue_keeps_only_latest_frame():
    async def run():
        ws = RecordingWebSocket(delay=10.0)  # sender stuck on first send
        job = _make_job(ws)
        job._enqueue_frame_payload({"type": "frame", "frame_num": 0})
        await asyncio.sleep(0.05)  # sender picks up frame 0, blocks in send
        for n in range(1, 6):
            job._enqueue_frame_payload({"type": "frame", "frame_num": n})
        queued = job._frame_send_queue.get_nowait()
        await job._stop_frame_sender(drain=False)
        return queued

    queued = asyncio.run(run())
    assert queued["frame_num"] == 5  # frames 1-4 dropped, newest kept


def test_slow_client_does_not_block_enqueue():
    async def run():
        ws = RecordingWebSocket(delay=0.2)
        job = _make_job(ws)
        t0 = time.perf_counter()
        for n in range(5):
            job._enqueue_frame_payload({"type": "frame", "frame_num": n})
            await asyncio.sleep(0)
        elapsed = time.perf_counter() - t0
        await job._stop_frame_sender(drain=False)
        return elapsed

    elapsed = asyncio.run(run())
    # Five sends at 0.2s each would take 1s synchronously.
    assert elapsed < 0.1, f"enqueue blocked for {elapsed:.3f}s"


def test_dropped_frame_carries_ocr_crops_forward():
    async def run():
        ws = RecordingWebSocket(delay=10.0)
        job = _make_job(ws)
        job._enqueue_frame_payload({"type": "frame", "frame_num": 0})
        await asyncio.sleep(0.05)  # sender blocked on frame 0
        crops = [{"slot": "B1", "crop": "b64"}]
        job._enqueue_frame_payload(
            {"type": "frame", "frame_num": 1, "ocr_crops": crops})
        job._enqueue_frame_payload(
            {"type": "frame", "frame_num": 2, "ocr_crops": None})
        queued = job._frame_send_queue.get_nowait()
        await job._stop_frame_sender(drain=False)
        return crops, queued

    crops, queued = asyncio.run(run())
    assert queued["frame_num"] == 2
    assert queued["ocr_crops"] == crops  # not lost with the dropped frame


def test_status_queue_keeps_only_latest():
    """Progress status is latest-wins; a slow client sees fewer, fresher updates."""
    async def run():
        ws = RecordingWebSocket(delay=10.0)
        job = _make_job(ws)
        job._enqueue_ws("status", {"type": "status", "progress": 0})
        await asyncio.sleep(0.05)  # sender blocked on the first status
        for n in range(1, 6):
            job._enqueue_ws("status", {"type": "status", "progress": n})
        queued = job._ws_queues["status"].get_nowait()
        await job._stop_ws_senders(drain=False)
        return queued

    queued = asyncio.run(run())
    assert queued["progress"] == 5


def test_diag_queue_bounded_drops_oldest():
    """Diagnostics shed oldest under backpressure; JSONL stays canonical."""
    async def run():
        ws = RecordingWebSocket(delay=10.0)
        job = _make_job(ws)
        job._enqueue_ws("diag", {"type": "diagnostic_event", "n": 0})
        await asyncio.sleep(0.05)  # sender blocked on record 0
        for n in range(1, 80):
            job._enqueue_ws("diag", {"type": "diagnostic_event", "n": n})
        oldest_kept = job._ws_queues["diag"].get_nowait()
        depth = job._ws_queues["diag"].qsize() + 1
        drops = job._ws_drop_counts.get("diag", 0)
        await job._stop_ws_senders(drain=False)
        return oldest_kept, depth, drops

    oldest_kept, depth, drops = asyncio.run(run())
    assert depth == 64               # bounded backlog
    assert drops == 15               # 79 enqueued into 64 slots
    assert oldest_kept["n"] == 16    # records 1-15 were shed, oldest first


def test_stop_ws_senders_drains_all_kinds():
    async def run():
        ws = RecordingWebSocket()
        job = _make_job(ws)
        job._enqueue_frame_payload({"type": "frame", "frame_num": 1})
        job._enqueue_ws("status", {"type": "status", "progress": 50})
        job._enqueue_ws("diag", {"type": "diagnostic_event", "n": 7})
        await job._stop_ws_senders(drain=True)
        return ws, job

    ws, job = asyncio.run(run())
    types = sorted(m["type"] for m in ws.messages)
    assert types == ["diagnostic_event", "frame", "status"]
    assert job._frame_sender_task is None


def test_sender_delivers_then_stops_cleanly():
    async def run():
        ws = RecordingWebSocket()
        job = _make_job(ws)
        job._enqueue_frame_payload({"type": "frame", "frame_num": 7})
        await job._stop_frame_sender()  # default: drain pending sends first
        return ws, job

    ws, job = asyncio.run(run())
    assert [m["frame_num"] for m in ws.messages] == [7]
    assert job._frame_sender_task is None


class WedgedWebSocket:
    """Every send awaits forever — a stalled transport. uvicorn's ws send can
    await indefinitely on one, even after the client disconnects."""

    async def send_json(self, payload):
        await asyncio.Event().wait()  # never resolves


def test_stop_ws_senders_bounded_when_sender_wedged(monkeypatch):
    """Completion must never hang on a sender wedged inside a stalled send.
    Observed live: a finished job blocked 60+ minutes in the drain's put()
    on a full maxsize-1 queue — status already "completed", `complete` never
    sent, replay row never saved. The bounded send times out (detaching the
    socket so the backlog short-circuits), the sentinel lands, drain returns."""
    import web.app as app_module
    monkeypatch.setattr(app_module, "WS_SEND_TIMEOUT", 0.05)
    monkeypatch.setattr(app_module, "WS_STOP_GRACE", 0.2)

    async def run():
        job = _make_job(WedgedWebSocket())
        job._enqueue_frame_payload({"type": "frame", "frame_num": 0})
        await asyncio.sleep(0.02)  # sender picks up frame 0, wedges in send
        # A second payload keeps the maxsize-1 queue FULL at stop time — the
        # exact shape of the production hang (unbounded put blocked forever).
        job._enqueue_frame_payload({"type": "frame", "frame_num": 1})
        t0 = time.perf_counter()
        await job._stop_ws_senders(drain=True)
        return time.perf_counter() - t0, job

    elapsed, job = asyncio.run(run())
    assert elapsed < 2.0, f"completion drain hung for {elapsed:.2f}s"
    assert job.websocket is None  # the timed-out send detached the socket
    assert job._frame_sender_task is None


def test_stop_ws_senders_healthy_drain_keeps_ordering(monkeypatch):
    """Bounding the drain must not turn it into a data race: a healthy (just
    slow-ish) transport still receives BOTH the in-flight and the queued
    payload, in order, before the drain returns — the sentinel lands behind
    the pending payload."""
    import web.app as app_module
    monkeypatch.setattr(app_module, "WS_SEND_TIMEOUT", 5.0)
    monkeypatch.setattr(app_module, "WS_STOP_GRACE", 5.0)

    async def run():
        ws = RecordingWebSocket(delay=0.05)
        job = _make_job(ws)
        job._enqueue_frame_payload({"type": "frame", "frame_num": 0})
        await asyncio.sleep(0.01)  # sender in flight on frame 0
        job._enqueue_frame_payload({"type": "frame", "frame_num": 1})  # queue full
        await job._stop_ws_senders(drain=True)
        return ws

    ws = asyncio.run(run())
    assert [m["frame_num"] for m in ws.messages] == [0, 1]

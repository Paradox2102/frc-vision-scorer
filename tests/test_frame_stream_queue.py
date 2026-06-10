"""Frame streaming must be decoupled from inference.

Frames go through a latest-only queue drained by a background sender task:
a slow websocket client can never back-pressure the frame loop, display sends
are rate-capped, and a dropped frame can't swallow a fresh VLM crops payload.
"""
import asyncio
import time

from web.app import InferenceJob


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

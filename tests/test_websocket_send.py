import asyncio
import time

from web.app import InferenceJob


class DetectConcurrentWebSocket:
    def __init__(self):
        self.active_sends = 0
        self.max_active_sends = 0
        self.messages = []

    async def send_json(self, payload):
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        try:
            await asyncio.sleep(0)
            self.messages.append(payload)
        finally:
            self.active_sends -= 1


class BrokenWebSocket:
    async def send_json(self, payload):
        raise AssertionError("broken drain state")


def _job_with_websocket(websocket):
    job = InferenceJob.__new__(InferenceJob)
    job.websocket = websocket
    job._websocket_send_lock = asyncio.Lock()
    return job


def test_websocket_sends_are_serialized():
    async def run():
        websocket = DetectConcurrentWebSocket()
        job = _job_with_websocket(websocket)

        results = await asyncio.gather(
            *(job._send_ws({"index": index}) for index in range(10))
        )

        assert all(results)
        assert websocket.max_active_sends == 1
        assert len(websocket.messages) == 10

    asyncio.run(run())


def test_websocket_failure_is_detached_and_not_raised():
    async def run():
        job = _job_with_websocket(BrokenWebSocket())

        assert await job._send_ws({"type": "frame"}) is False
        assert job.websocket is None
        assert await job._send_ws({"type": "complete"}) is False

    asyncio.run(run())


class HangingWebSocket:
    """send_json awaits forever — a stalled transport. uvicorn's ws send can
    await indefinitely on one, even after the client disconnects."""

    async def send_json(self, payload):
        await asyncio.Event().wait()  # never resolves


def test_websocket_send_timeout_detaches_socket(monkeypatch):
    """A send exceeding WS_SEND_TIMEOUT gets the transport-failure treatment
    (return False, socket detached) instead of wedging the sender forever —
    the root of the 60+ minute job-completion hang."""
    import web.app as app_module
    monkeypatch.setattr(app_module, "WS_SEND_TIMEOUT", 0.05)

    async def run():
        job = _job_with_websocket(HangingWebSocket())
        t0 = time.perf_counter()
        ok = await job._send_ws({"type": "complete"})
        elapsed = time.perf_counter() - t0
        return ok, elapsed, job

    ok, elapsed, job = asyncio.run(run())
    assert ok is False
    assert job.websocket is None       # detached like a transport failure
    assert elapsed < 1.0, f"send blocked for {elapsed:.2f}s past the bound"
    # wait_for cancelled the underlying send, so the send lock is free again
    # for other senders (a still-held lock would wedge every later send).
    assert not job._websocket_send_lock.locked()

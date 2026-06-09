import asyncio

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

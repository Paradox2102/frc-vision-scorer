"""Quality-degrading failures (SAM2 init, gallery load, frame evidence) must
be surfaced to the client, not just printed — a silently degraded run
invalidates A/B comparisons. One websocket message per component; repeats
increment a count instead of spamming."""
import asyncio

from web.app import InferenceJob


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


def _make_job(websocket):
    job = InferenceJob.__new__(InferenceJob)
    job.websocket = websocket
    job._websocket_send_lock = asyncio.Lock()
    return job


def test_warn_degraded_records_and_sends():
    async def run():
        ws = RecordingWebSocket()
        job = _make_job(ws)
        await job._warn_degraded("sam2", "SAM2 init failed; fell back to Kalman tracking")
        return job, ws

    job, ws = asyncio.run(run())
    assert len(job.degradation_warnings) == 1
    assert job.degradation_warnings[0]["component"] == "sam2"
    degraded = [m for m in ws.messages if m.get("type") == "degradation_warning"]
    assert len(degraded) == 1
    assert degraded[0]["component"] == "sam2"
    assert "Kalman" in degraded[0]["message"]


def test_warn_degraded_dedupes_per_component():
    async def run():
        ws = RecordingWebSocket()
        job = _make_job(ws)
        for _ in range(50):  # e.g. frame-evidence failure every frame
            await job._warn_degraded("frame_evidence", "record failed: disk full")
        await job._warn_degraded("gallery", "gallery load failed")
        return job, ws

    job, ws = asyncio.run(run())
    degraded = [m for m in ws.messages if m.get("type") == "degradation_warning"]
    assert len(degraded) == 2  # one per component, not 51
    assert len(job.degradation_warnings) == 2
    frame_ev = next(w for w in job.degradation_warnings
                    if w["component"] == "frame_evidence")
    assert frame_ev["count"] == 50


def test_warn_degraded_without_websocket_does_not_raise():
    async def run():
        job = _make_job(None)
        await job._warn_degraded("sam2", "no client attached")
        return job

    job = asyncio.run(run())
    assert job.degradation_warnings[0]["component"] == "sam2"

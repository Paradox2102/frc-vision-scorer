"""Job-slot serialization and jobs-dict hygiene.

One job runs at a time by default (SAM2+VLM saturate the GPU; concurrent
jobs offload state to CPU and confound A/B timing). Extra submissions queue:
the client gets "queued" heartbeats, stop/disconnect while queued abandons
the run, and finished jobs are pruned so they stop holding YOLO models.
"""
import asyncio
import time

import web.app as app_module
from web.app import InferenceJob


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


class DeadWebSocket:
    async def send_json(self, payload):
        raise ConnectionError("client gone")


def _make_job(websocket, run_log=None, name="job", duration=0.0):
    job = InferenceJob.__new__(InferenceJob)
    job.job_id = name
    job.websocket = websocket
    job._websocket_send_lock = asyncio.Lock()
    job._stopped = False
    job.status = "initializing"
    job.error = None

    async def fake_process_video():
        if run_log is not None:
            run_log.append((name, "start", time.monotonic()))
        await asyncio.sleep(duration)
        job.status = "completed"
        if run_log is not None:
            run_log.append((name, "end", time.monotonic()))

    job.process_video = fake_process_video
    return job


def _fresh_slots(monkeypatch, n=1, poll=0.02):
    monkeypatch.setattr(app_module, "_job_slots", asyncio.Semaphore(n))
    monkeypatch.setattr(app_module, "_QUEUE_POLL_SECONDS", poll)


def test_second_job_waits_for_first(monkeypatch):
    async def run():
        _fresh_slots(monkeypatch)
        log = []
        a = _make_job(RecordingWebSocket(), log, "a", duration=0.15)
        b_ws = RecordingWebSocket()
        b = _make_job(b_ws, log, "b")
        await asyncio.gather(
            app_module._run_job_serialized(a),
            app_module._run_job_serialized(b),
        )
        return log, b_ws

    log, b_ws = asyncio.run(run())
    order = [(name, ev) for name, ev, _t in log]
    assert order == [("a", "start"), ("a", "end"), ("b", "start"), ("b", "end")]
    # The waiting client was told it was queued.
    assert any(m.get("status") == "queued" for m in b_ws.messages)


def test_stop_while_queued_never_runs(monkeypatch):
    async def run():
        _fresh_slots(monkeypatch)
        log = []
        a = _make_job(RecordingWebSocket(), log, "a", duration=0.2)
        b_ws = RecordingWebSocket()
        b = _make_job(b_ws, log, "b")
        task_a = asyncio.create_task(app_module._run_job_serialized(a))
        await asyncio.sleep(0.05)  # a holds the slot
        task_b = asyncio.create_task(app_module._run_job_serialized(b))
        await asyncio.sleep(0.05)  # b is waiting
        b._stopped = True
        await asyncio.gather(task_a, task_b)
        return log, b, b_ws

    log, b, b_ws = asyncio.run(run())
    assert ("b", "start") not in [(n, e) for n, e, _t in log]
    assert b.status == "error"
    assert any(m.get("type") == "error" for m in b_ws.messages)


def test_disconnect_while_queued_abandons_run(monkeypatch):
    async def run():
        _fresh_slots(monkeypatch)
        log = []
        a = _make_job(RecordingWebSocket(), log, "a", duration=0.2)
        b = _make_job(DeadWebSocket(), log, "b")
        await asyncio.gather(
            app_module._run_job_serialized(a),
            app_module._run_job_serialized(b),
        )
        return log, b

    log, b = asyncio.run(run())
    assert ("b", "start") not in [(n, e) for n, e, _t in log]
    assert b.status == "error"


def test_slot_released_after_failure(monkeypatch):
    async def run():
        _fresh_slots(monkeypatch)
        boom = _make_job(RecordingWebSocket())

        async def explode():
            raise RuntimeError("boom")

        boom.process_video = explode
        try:
            await app_module._run_job_serialized(boom)
        except RuntimeError:
            pass
        # Slot must be free again for the next job.
        ok = _make_job(RecordingWebSocket())
        await asyncio.wait_for(app_module._run_job_serialized(ok), timeout=1.0)
        return ok

    ok = asyncio.run(run())
    assert ok.status == "completed"


def test_prune_finished_jobs(monkeypatch):
    monkeypatch.setattr(app_module, "JOB_RETENTION", 2)
    monkeypatch.setattr(app_module, "jobs", {})

    def stub(status):
        job = InferenceJob.__new__(InferenceJob)
        job.status = status
        return job

    app_module.jobs.update({
        "old1": stub("completed"), "old2": stub("error"),
        "running": stub("processing"), "old3": stub("completed"),
        "old4": stub("completed"),
    })
    app_module._prune_finished_jobs()
    # Running jobs are never pruned; the 2 newest finished survive.
    assert set(app_module.jobs) == {"running", "old3", "old4"}

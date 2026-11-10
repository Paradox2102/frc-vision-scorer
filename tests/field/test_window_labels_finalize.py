"""window_labels knob through InferenceJob._apply_window_labels: when it runs,
what it writes, and that every failure leaves the replay untouched. The stage
itself (web/field/window_labels.run_window_labels) is stubbed."""
import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest

import web.app as app_mod
from web.app import InferenceJob
from web.field import window_labels as wl

FRAMES = [{"frame_num": 420 + 12 * i, "scores": {},
           "robots": [{"slot": "B1", "alliance": "blue", "team_number": "6773",
                       "confirmed": True, "field_pos": [3.0, 2.0]},
                      {"slot": "R1", "alliance": "red", "team_number": "2724",
                       "confirmed": False, "field_pos": [12.0, 3.0]}]} for i in range(10)]


def _job(on=True, status="completed"):
    job = InferenceJob.__new__(InferenceJob)
    job.status = status
    job.job_id = "wl_test"
    job.video_fps = 60.0
    job.video_path = "/nonexistent/video.mp4"
    job.match_red_teams = ["2637", "2724", "6497"]
    job.match_blue_teams = ["3128", "6773", "9066"]
    job.config = SimpleNamespace(window_labels=on, window_labels_lineup_min_p=0.65,
                                 ground_center_offset_m=0.2, jev_model=None)
    job.calib = object()
    job.views = [SimpleNamespace(name="main", crop=(0, 0, 1920, 723), calibration=job.calib)]
    job._slot_segments = {0: [{"start_frame": 420, "end_frame": 528, "team_number": "6773",
                               "confirmed": True, "score": 1.0, "source": "read"}]}
    job._window_label_session = lambda: "SESSION"
    rec = SimpleNamespace(frames=copy.deepcopy(FRAMES), events=[
        {"frame_num": 2, "slot": "B1", "team_number": "6773", "robot_label": "6773"},
        {"frame_num": 3, "slot": "R1", "team_number": "2724", "robot_label": "2724"}])
    return job, rec


def _ok_result():
    lab = {(i, "B1"): ("3128", wl.RANK_DIGITS, 0.9, 10) for i in range(10)}
    return {"aborted": None, "labels": lab,
            "segments": {0: [{"start_frame": 420, "end_frame": 528, "team_number": "3128",
                              "confirmed": True, "score": 0.9, "source": "window_digits"}]},
            "stats": {"windows": 2, "digit_named": 1, "lineup_named": 0,
                      "cleared_by_uniqueness": 0, "unlabelled": 1, "lineup_rows": 0,
                      "requests": 18, "input_tokens": 30000, "cost_cents": 0.13,
                      "request_errors": 0, "aborted": None},
            "lineup_jpeg": None}


def _stub(monkeypatch, result=None, exc=None):
    seen = {}

    def fake(frames, video_path, cal, session, params, **kw):
        seen.update(frames=frames, video_path=video_path, cal=cal, session=session, params=params)
        if exc:
            raise exc
        return result

    monkeypatch.setattr(wl, "run_window_labels", fake)
    return seen


def _apply(job, rec):
    summary, timings = {}, {"total": 100.0}
    asyncio.run(job._apply_window_labels(rec, summary, timings))
    return summary, timings


def test_knob_off_does_nothing(monkeypatch):
    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job(on=False)
    summary, timings = _apply(job, rec)
    assert not seen and rec.frames == FRAMES
    assert "window_labels" not in summary and timings == {"total": 100.0}
    assert getattr(job, "_window_segments", None) is None


@pytest.mark.parametrize("status", ["error", "stopped", "running"])
def test_incomplete_jobs_make_no_reader_calls(monkeypatch, status):
    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job(status=status)
    summary, _ = _apply(job, rec)
    assert not seen and rec.frames == FRAMES and "window_labels" not in summary


def test_a_user_stop_makes_no_reader_calls(monkeypatch):
    # a STOP ends the analysis loop early but the status still becomes "completed"
    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job()
    job._stopped = True
    summary, timings = _apply(job, rec)
    assert not seen and rec.frames == FRAMES
    assert "window_labels" not in summary and timings == {"total": 100.0}
    assert job._window_segments is None


def test_success_rewrites_frames_events_segments_and_reports(monkeypatch):
    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job()
    summary, timings = _apply(job, rec)
    assert seen["session"] == "SESSION" and seen["cal"] is job.calib
    p = seen["params"]
    assert p.fps == 60.0 and p.lineup_min_p == 0.65 and p.ground_center_offset_m == 0.2
    assert p.red == job.match_red_teams and p.blue == job.match_blue_teams
    assert p.view == (0, 0, 1920, 723) and p.deadline is not None
    rows = {r["slot"]: r for r in rec.frames[4]["robots"]}
    assert rows["B1"]["team_number"] == "3128" and rows["B1"]["confirmed"] is True
    assert rows["R1"]["team_number"] is None
    assert rec.events[0]["team_number"] == "3128" and rec.events[0]["robot_label"] == "3128"
    assert rec.events[1]["team_number"] is None and rec.events[1]["robot_label"] == "R1"
    assert job._window_segments[0][0]["source"] == "window_digits"
    st = summary["window_labels"]
    assert st["digit_named"] == 1 and st["aborted"] is None and st["seconds"] >= 0
    assert timings["window_labels.requests"] == 18
    assert timings["window_labels.cost_cents"] == 0.13
    assert "window_labels.seconds" in timings and "window_labels.aborted" not in timings


def test_abort_leaves_the_replay_untouched(monkeypatch):
    res = {"aborted": "request errors: 300 of 2000 in the digit pass", "labels": None,
           "segments": None, "lineup_jpeg": None,
           "stats": {**_ok_result()["stats"], "aborted": "request errors: 300 of 2000 in the digit pass"}}
    _stub(monkeypatch, res)
    job, rec = _job()
    summary, timings = _apply(job, rec)
    assert rec.frames == FRAMES and rec.events[0]["team_number"] == "6773"
    assert job._window_segments is None
    assert summary["window_labels"]["aborted"].startswith("request errors")
    assert timings["window_labels.aborted"] == 1


def test_exception_and_missing_inputs_leave_the_replay_untouched(monkeypatch):
    _stub(monkeypatch, exc=RuntimeError("boom"))
    job, rec = _job()
    summary, _ = _apply(job, rec)
    assert rec.frames == FRAMES and "boom" in summary["window_labels"]["aborted"]

    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job()
    job.views = []
    summary, _ = _apply(job, rec)
    assert not seen and rec.frames == FRAMES
    assert "calibration" in summary["window_labels"]["aborted"]


def test_cancelling_the_job_zeroes_the_deadline_the_worker_thread_holds(monkeypatch):
    started, release = threading.Event(), threading.Event()
    seen = {}

    def blocking(frames, video_path, cal, session, params, **kw):
        seen["params"] = params
        started.set()
        release.wait(5.0)
        return _ok_result()

    monkeypatch.setattr(wl, "run_window_labels", blocking)
    job, rec = _job()

    async def main():
        task = asyncio.create_task(job._apply_window_labels(rec, {}, {"total": 100.0}))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deadline = seen["params"].deadline
        release.set()                       # let the worker exit before the loop shuts down
        return deadline

    try:
        assert asyncio.run(main()) == 0.0
        assert rec.frames == FRAMES
    finally:
        release.set()


def test_window_label_session_is_a_dedicated_jev_session(monkeypatch):
    monkeypatch.setenv("JEV_TYPESAFE_API_TOKEN", "x")
    job, _rec = _job()
    del job._window_label_session
    s = InferenceJob._window_label_session(job)
    assert isinstance(s, app_mod.jev_reader.JevSession) and s.timeout_s == 30.0
    assert s.retries == 1

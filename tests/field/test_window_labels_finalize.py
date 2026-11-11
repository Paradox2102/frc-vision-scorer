"""window_labels knob through InferenceJob._apply_window_labels: when it runs,
what it writes, and that every failure leaves the replay untouched. The stage
itself (web/field/window_labels.run_window_labels) is stubbed."""
import asyncio
import copy
import json
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


def test_the_deinterlace_knob_reaches_the_stage_params(monkeypatch):
    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job()
    _apply(job, rec)
    assert seen["params"].deinterlace is False                # knob absent: off
    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job()
    job.config.window_labels_deinterlace = True
    _apply(job, rec)
    assert seen["params"].deinterlace is True


def test_the_reader_knob_reaches_the_stage_params_and_its_stats_reach_timings(monkeypatch):
    seen = _stub(monkeypatch, _ok_result())
    job, rec = _job()
    _apply(job, rec)
    assert seen["params"].reader == "jev"                     # knob absent: jev
    res = _ok_result()
    res["stats"] = {**res["stats"], "student": {"reader": "student", "windows_s1": 7,
                                                "windows_jev": 2, "jev_digit_calls": 18,
                                                "bumper_found": 30, "samples": 40,
                                                "fallback": None}}
    seen = _stub(monkeypatch, res)
    job, rec = _job()
    job.config.window_labels_reader = "student"
    summary, timings = _apply(job, rec)
    assert seen["params"].reader == "student"
    assert summary["window_labels"]["student"]["windows_s1"] == 7
    assert timings["window_labels.student.windows_s1"] == 7
    assert timings["window_labels.student.jev_digit_calls"] == 18


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


from web.field import event_anchors as ea


def _anchor(kind="digits", team="3128"):
    return ea.Anchor(team=team, kind=kind, p=0.9, frame_num=420, alliance_color="blue",
                     tile=b"\xff\xd8x")


def _event_job(on=True, event=True):
    job, rec = _job()
    job.config.event_lineup = on
    job.event_key, job.match_key = ("2026cur", "2026cur_qm8") if event else (None, None)
    return job, rec


def _store(monkeypatch, order, fetched=None, fetch_exc=None, write_exc=None):
    seen = {}

    async def fetch(namespace, event_key, teams, exclude_match_key, max_match_number=None, **kw):
        order.append("fetch")
        seen["fetch"] = (namespace, event_key, list(teams), exclude_match_key, max_match_number)
        if fetch_exc:
            raise fetch_exc
        return fetched or {}

    async def replace(namespace, event_key, match_key, anchors, *, replay_id=None, **kw):
        order.append("replace")
        seen["replace"] = (namespace, event_key, match_key, list(anchors), replay_id)
        if write_exc:
            raise write_exc
        return len(anchors)

    monkeypatch.setattr(app_mod.database, "fetch_event_anchors", fetch)
    monkeypatch.setattr(app_mod.database, "replace_event_anchors", replace)
    return seen


def _stage(monkeypatch, order, result):
    seen = {}

    def fake(frames, video_path, cal, session, params, **kw):
        order.append("stage")
        seen["params"] = params
        return result

    monkeypatch.setattr(wl, "run_window_labels", fake)
    return seen


def _ok_event_result():
    res = _ok_result()
    res.update(anchors_out=[_anchor(), _anchor("elimination", "9066")],
               digit_calls={"calls": {0: [{"3128": 0.9}]}, "wide_p": {}, "fingerprints": {}},
               row_sources={}, lineup_jpeg_blue=None)
    res["stats"] = {**res["stats"], "rows_own": 1, "rows_borrowed": 1, "rows_elim": 0}
    return res


def test_digit_calls_go_into_the_summary_with_the_knob_off(monkeypatch):
    order = []
    _store(monkeypatch, order)
    res = _ok_event_result()
    _stage(monkeypatch, order, res)
    job, rec = _event_job(on=False)
    summary, _ = _apply(job, rec)
    saved = summary["window_labels_digit_calls"]
    assert "wide_p" not in saved                             # rebuilt from calls on load
    assert saved == {k: v for k, v in res["digit_calls"].items() if k != "wide_p"}
    assert order == ["stage"]                                # no store access
    assert set(summary["window_labels"]) == set(res["stats"]) | {"seconds"}


def test_event_lineup_fetches_before_the_stage_and_writes_after_the_labels(monkeypatch):
    order = []
    borrowed = {"9066": [_anchor(team="9066")]}
    seen = _store(monkeypatch, order, fetched=borrowed)
    st = _stage(monkeypatch, order, _ok_event_result())
    job, rec = _event_job()
    fake_replace = app_mod.database.replace_event_anchors

    async def check_then_replace(*a, **kw):
        # the labels are already applied when the anchors are written
        assert {r["slot"]: r for r in rec.frames[4]["robots"]}["B1"]["team_number"] == "3128"
        return await fake_replace(*a, **kw)

    monkeypatch.setattr(app_mod.database, "replace_event_anchors", check_then_replace)
    summary, _ = _apply(job, rec)
    assert order == ["fetch", "stage", "replace"]
    assert seen["fetch"] == ("live", "2026cur", job.match_red_teams + job.match_blue_teams,
                             "2026cur_qm8", None)
    p = st["params"]
    assert p.event_lineup is True and p.borrowed == borrowed
    assert p.this_colors["2637"] == "red" and p.this_colors["9066"] == "blue"
    ns, ev, mk, anchors, rid = seen["replace"]
    assert (ns, ev, mk, rid) == ("live", "2026cur", "2026cur_qm8", "wl_test")
    assert len(anchors) == 2
    s = summary["window_labels"]
    assert (s["anchors_written"], s["elim_anchors_written"], s["borrow_fetch_error"]) == (1, 1, None)
    assert s["rows_borrowed"] == 1 and "window_labels_digit_calls" in summary


def test_a_failed_fetch_runs_the_stage_without_borrowed_rows(monkeypatch):
    order = []
    _store(monkeypatch, order, fetch_exc=RuntimeError("db locked"))
    st = _stage(monkeypatch, order, _ok_event_result())
    job, rec = _event_job()
    summary, _ = _apply(job, rec)
    assert st["params"].borrowed == {} and order == ["fetch", "stage", "replace"]
    assert "db locked" in summary["window_labels"]["borrow_fetch_error"]
    assert summary["window_labels"]["aborted"] is None


def test_a_failed_write_keeps_the_labels(monkeypatch):
    order = []
    _store(monkeypatch, order, write_exc=OSError("disk full"))
    _stage(monkeypatch, order, _ok_event_result())
    job, rec = _event_job()
    summary, _ = _apply(job, rec)
    assert {r["slot"]: r for r in rec.frames[4]["robots"]}["B1"]["team_number"] == "3128"
    assert job._window_segments is not None and summary["window_labels"]["aborted"] is None
    assert summary["window_labels"]["anchors_written"] == 0


def test_an_abort_writes_no_anchors_and_no_digit_calls(monkeypatch):
    order = []
    _store(monkeypatch, order)
    res = {**_ok_event_result(), "aborted": "deadline passed in the lineup pass",
           "labels": None, "segments": None, "digit_calls": None, "anchors_out": []}
    res["stats"] = {**res["stats"], "aborted": res["aborted"]}
    _stage(monkeypatch, order, res)
    job, rec = _event_job()
    summary, _ = _apply(job, rec)
    assert order == ["fetch", "stage"] and "window_labels_digit_calls" not in summary
    assert summary["window_labels"]["anchors_written"] == 0 and rec.frames == FRAMES


def test_no_match_key_means_no_store_access(monkeypatch):
    order = []
    _store(monkeypatch, order)
    _stage(monkeypatch, order, _ok_event_result())
    job, rec = _event_job(event=False)
    summary, _ = _apply(job, rec)
    assert order == ["stage"]
    assert summary["window_labels"]["borrow_fetch_error"] == "no event_key or match_key"


def test_event_lineup_without_window_labels_does_nothing_and_says_so(monkeypatch, capsys):
    order = []
    _store(monkeypatch, order)
    _stage(monkeypatch, order, _ok_event_result())
    job, rec = _event_job()
    job.config.window_labels = False
    summary, _ = _apply(job, rec)
    assert order == [] and "window_labels" not in summary and rec.frames == FRAMES
    assert "[event-lineup]" in capsys.readouterr().out


def test_replay_provenance_records_config_stop_and_calibration():
    from web.job_config import JobConfig
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig({"window_labels": True, "ground_center_offset_m": 0.2})
    job._stopped = True
    job._calib_version_id = "20260630-210916-570400"
    prov = InferenceJob._replay_provenance(job)
    assert prov == {"config": job.config.non_defaults(), "stopped": True,
                    "calibration_version": "20260630-210916-570400"}
    assert prov["config"]["window_labels"] is True
    assert prov["config"]["ground_center_offset_m"] == 0.2
    json.dumps(prov)                                        # goes into meta.json
    bare = InferenceJob.__new__(InferenceJob)
    assert InferenceJob._replay_provenance(bare) == {"config": {}, "stopped": False,
                                                     "calibration_version": None}


def _realistic_event_result():
    res = _ok_event_result()
    res["digit_calls"] = {
        "roster": ["3128", "2637"], "model_key": "jev-1.14",
        "calls": {0: [{"3128": 0.91, "2637": 0.04}, None], 12: [{"3128": 0.5}]},
        "wide_p": {0: {1: {"3128": 0.7}}},
        "fingerprints": {0: {"slot": "B1", "first_frame": 4, "samples": [1, 2], "requests": 2}},
    }
    return res


@pytest.mark.parametrize("on", [True, False])
def test_finalize_summary_is_json_serialisable(monkeypatch, on):
    order = []
    _store(monkeypatch, order)
    _stage(monkeypatch, order, _realistic_event_result())
    job, rec = _event_job(on=on)
    summary, _ = _apply(job, rec)
    assert summary["window_labels_digit_calls"]["calls"][0][0]["3128"] == 0.91
    json.dumps(summary)


def test_a_stopped_event_job_touches_neither_store_nor_stage(monkeypatch):
    order = []
    _store(monkeypatch, order)
    _stage(monkeypatch, order, _ok_event_result())
    job, rec = _event_job()
    job._stopped = True
    summary, _ = _apply(job, rec)
    assert order == [] and "window_labels" not in summary and rec.frames == FRAMES

"""Read-trigger telemetry (always on) and the per-slot read census
(``read_census`` knob, default OFF).

qm11 full match: 107-170 bumper reads, blue slots attempted every 7-10 s
(median) — 328 of 355 immediate escalations were identity re-attachments, so
read supply followed identity churn (identity_continuity cut births 575->326
and reads 170->145; binding got worse). Census makes every bound slot
read-due once its last read attempt is ``read_census_interval_s`` old,
weakest-evidence slot first; every vlm_read row now says why it was read.
"""
import asyncio
import json
import random
import threading
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import web.app as app_module
from web import clef_reader
from web.app import InferenceJob
from web.diagnostic import DiagnosticRecorder
from web.field.identity import IdentityRegistry, RobotIdentity
from web.field.identity_confidence import (
    CENSUS_BASE_PRIORITY,
    ConfidenceConfig,
    census_read_decision,
    decide_vlm_read,
    read_trigger,
    slot_evidence_margin,
)
from web.job_config import JobConfig

FPS = 59.94
STRIDE = 8                      # frame_skip of the qm11 full-match job
TEAMS = ["111", "222", "333", "444", "555", "666"]
SLOTS = list(range(6))
IDENT_OF_SLOT = {s: 10 + s for s in SLOTS}


# ---------------------------------------------------------------- harness

def _job(tmp_path=None, *, census=False, interval=2.0, honor=False, reader="local",
         slots=SLOTS, multiframe=False):
    """Bare InferenceJob with a REAL IdentityRegistry / ConfidenceConfig and
    the REAL scheduler + read pass (only the reader is faked)."""
    job = InferenceJob.__new__(InferenceJob)
    # Pin the legacy (pre-Nov-6) OFF behaviour explicitly: defaults now ship ON.
    cfg = {"team_reader": reader, "read_census": False, "read_keep_evicted": False}
    if census:
        cfg.update(read_census=True, read_census_interval_s=interval,
                   read_census_honor_backoff=honor)
    if multiframe:
        cfg["clef_multiframe"] = True
    job.config = JobConfig(cfg)
    job.team_reader = reader
    job._camera_centers_xy = {}
    job.match_teams = list(TEAMS)
    job.match_blue_teams = TEAMS[:3]
    job.match_red_teams = TEAMS[3:]
    job.vlm_status = {}
    job._vlm_task = None
    job._claude_vision_calls = 0
    job._current_frame = 0
    job._evidence_frame = 0
    job.video_fps = FPS
    job.field_tracker = types.SimpleNamespace(
        last_det_to_slot={i: s for i, s in enumerate(slots)}, _id_to_slot={},
        binding_events=[])
    reg = IdentityRegistry()
    for s in slots:
        iid = IDENT_OF_SLOT[s]
        reg.identities[iid] = RobotIdentity(
            identity_id=iid, alliance="blue" if s < 3 else "red", active_slot=s)
    reg._next_id = 100
    job.identity_registry = reg
    job.robot_tracks = {}
    job._last_bindings = {}
    job._slot_source = {}
    job._slot_crops = {}
    job.scoring_events = []
    job.user_seed_zones_raw = []
    job._occ = {}
    job._occ_by_cam = {}
    job.confidence_config = ConfidenceConfig()
    job.gallery = None
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    job.identity_diagnostics = MagicMock()
    job.timings = {}
    if tmp_path is not None:
        job.diagnostic = DiagnosticRecorder(tmp_path / "diag.jsonl", ws_send=None)
    else:
        job.diagnostic = MagicMock()
    return job


def _fused(slots=SLOTS, cams=("main",)):
    return [types.SimpleNamespace(
        evidence=[(cam, f"crop{s}", 1.0, (float(s), 1.0)) for cam in cams],
        detection=types.SimpleNamespace(pos=(float(s), 1.0)),
    ) for s in slots]


class _Reader:
    """Fake _vision_read_bumpers: legible slots read their team, failing
    slots return no team. Logs (crop_frame, slot) per read."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.crop_frame = None
        self.log = []
        self.calls = 0

    def __call__(self, crops, teams, blue, red, expected_alliances=None,
                 provider="local", **_kw):
        self.calls += 1
        out = []
        for c in crops:
            s = int(c.replace("crop", ""))
            self.log.append((self.crop_frame, s))
            if s in self.failing:
                out.append({"team_number": None, "confidence": 0.0,
                            "raw_read": "", "alliance": "unknown"})
            else:
                out.append({"team_number": TEAMS[s], "confidence": 0.95,
                            "raw_read": TEAMS[s],
                            "alliance": "blue" if s < 3 else "red"})
        return out

    def read_times(self, slot):
        return [f / FPS for f, s in self.log if s == slot]


def _rows(job, tmp_path):
    job.diagnostic.flush()
    return [json.loads(l) for l in (tmp_path / "diag.jsonl").read_text().splitlines()
            if json.loads(l)["type"] == "vlm_read"]


def _schedule_and_run(job, fused, crop_frame, clock=None):
    """One main-loop VLM stage at `crop_frame`: collect -> schedule -> await."""
    async def run():
        t = crop_frame / FPS if clock is None else clock
        due = job._collect_due_vlm_reads(t)
        if due:
            job._schedule_vlm_pass(fused, due, crop_frame=crop_frame)
            await job._vlm_task
        return due
    return asyncio.run(run())


async def _simulate(job, reader, *, seconds, slots=SLOTS, start_frame=0):
    """Frame-by-frame: real clock + scheduler + read pass (latency 0)."""
    job._current_frame = start_frame
    for i in range(int(seconds * FPS / STRIDE)):
        frame_num = start_frame + i * STRIDE
        clock = job._read_clock_sec(frame_num, i)
        due = job._collect_due_vlm_reads(clock)
        if due:
            reader.crop_frame = frame_num
            await job._read_fused_bumper_numbers_impl(
                _fused(slots), due, det_to_slot={k: s for k, s in enumerate(slots)},
                crop_frame=frame_num, read_triggers=dict(job._due_triggers))
        job._current_frame = frame_num


def _sim(monkeypatch, job, *, seconds, failing=(), slots=SLOTS):
    reader = _Reader(failing)
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    asyncio.run(_simulate(job, reader, seconds=seconds, slots=slots))
    return reader


# ======================================================= trigger telemetry

def test_recorder_writes_trigger_only_when_passed(tmp_path):
    rec = DiagnosticRecorder(tmp_path / "d.jsonl")
    common = dict(frame=1, robot_id=2, crop_b64="x", legibility=0.0, raw_read="4786",
                  matched_team="4786", alliance_read="blue", confidence=0.9,
                  vote_weight=1.0)
    rec.record_vlm_read(**common)
    rec.record_vlm_read(**common, trigger="census")
    rec.close()
    rows = [json.loads(l) for l in (tmp_path / "d.jsonl").read_text().splitlines()]
    assert "trigger" not in rows[0]
    assert rows[1]["trigger"] == "census"


def test_read_trigger_names():
    assert read_trigger("immediate_read_requested", "re_attachment") == "re_attachment"
    assert read_trigger("immediate_read_requested", None) == "immediate"
    for r in ("unconfirmed_retry", "audit_due", "census"):
        assert read_trigger(r, "re_attachment") == r


def test_rows_carry_the_scheduler_trigger(monkeypatch, tmp_path):
    """Immediate (with its escalation), unconfirmed retry and audit reads are
    each recorded with why they were read — through the real collect ->
    _schedule_vlm_pass -> background pass path."""
    reader = _Reader()
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _job(tmp_path, slots=[0, 1, 2])
    reg = job.identity_registry.identities
    job._request_immediate_read(reg[10], "re_attachment", renew=True)
    reg[12].is_confirmed = True
    reg[12].last_success_time = reg[12].last_vlm_time = 0.0
    crop = int(50 * FPS)
    _schedule_and_run(job, _fused([0, 1, 2]), crop)
    rows = {r["robot_id"]: r for r in _rows(job, tmp_path)}
    assert rows[10]["trigger"] == "re_attachment"
    assert rows[11]["trigger"] == "unconfirmed_retry"
    assert rows[12]["trigger"] == "audit_due"
    assert job._read_trigger_counts == {"re_attachment": 1, "unconfirmed_retry": 1,
                                        "audit_due": 1}


def test_direct_read_pass_callers_record_unspecified(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "_vision_read_bumpers", _Reader())
    job = _job(tmp_path, slots=[0])
    asyncio.run(job.read_fused_bumper_numbers(
        _fused([0]), {10: 1.0}, det_to_slot={0: 0}, crop_frame=70))
    assert _rows(job, tmp_path)[0]["trigger"] == "unspecified"


def test_immediate_reason_first_wins_until_renewed():
    job = _job()
    ident = job.identity_registry.identities[10]
    job._request_immediate_read(ident, "co_occlusion_ambiguous")
    job._request_immediate_read(ident, "roster_clear")        # still pending
    assert job._immediate_reasons[10] == "co_occlusion_ambiguous"
    job._request_immediate_read(ident, "low_margin", renew=True)  # new queue stamp
    assert job._immediate_reasons[10] == "low_margin"
    ident.needs_immediate_read = False                         # consumed
    job._request_immediate_read(ident, "contradiction")
    assert job._immediate_reasons[10] == "contradiction"
    assert ident.needs_immediate_read is True


def _sync_job(current_frame):
    from web.field.identity_ambiguity import CoOcclusionTracker
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    job = InferenceJob.__new__(InferenceJob)
    job.identity_registry = IdentityRegistry()
    job.confidence_config = ConfidenceConfig()
    job.co_occlusion_tracker = CoOcclusionTracker(re_emergence_window=30)
    job.identity_diagnostics = IdentityDiagnosticRecorder()
    job._current_frame = current_frame
    job.video_fps = FPS
    job._last_bindings = {}
    job._mock_db = True
    job.job_id = "test_job"
    job.scoring_events = []
    job._identity_log = []
    job.field_tracker = MagicMock()
    job.field_tracker.binding_events = []
    job.field_tracker.assignment_evidence = {}
    job.field_tracker._emit.return_value = []

    async def _bind(slot_id, ident_id):
        pass
    job._bind_detection_identity = _bind
    return job


def test_identity_sync_records_the_re_attachment_escalation(monkeypatch):
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    job = _sync_job(current_frame=93)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    asyncio.run(job._sync_identity_registry_frame(
        [MagicMock(track_id=1, detection=MagicMock(pos=(1.0, 1.0), embedding=emb))],
        [], [MagicMock(slot_id=0, pos=(1.0, 1.0), alliance="blue")],
        log_frame_num=100))
    (ident,) = job.identity_registry.identities.values()
    assert ident.needs_immediate_read is True
    assert job._immediate_reasons[ident.identity_id] == "re_attachment"
    job.vlm_status = {}
    job._occ = {}
    job.robot_tracks = {}
    job.config = JobConfig()
    job._collect_due_vlm_reads(100 / FPS)
    assert job._due_triggers == {ident.identity_id: "re_attachment"}


def test_roster_clear_reason(monkeypatch):
    job = _job()
    a = job._get_or_create_identity_track(10, "blue")
    b = job._get_or_create_identity_track(11, "blue")
    b.team_number = "111"
    job.identity_registry.identities[11].team_number = "111"
    job._clear_team_from_other_robots(a.robot_id, "111")
    assert job.identity_registry.identities[11].needs_immediate_read is True
    assert job._immediate_reasons[11] == "roster_clear"


def test_early_ocr_rows_carry_trigger(monkeypatch, tmp_path):
    monkeypatch.setattr(
        app_module, "_vision_read_bumpers",
        lambda crops, teams, blue, red, expected_alliances=None, provider="local", **_k:
            [{"team_number": "111", "confidence": 0.9, "alliance": "blue",
              "raw_read": "111"} for _ in crops])
    job = _job(tmp_path, slots=[0])
    job._bumper_crop_b64 = MagicMock(return_value=("crop-b64", 0.5))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    job._vision_identify_robots_impl(frame, [(0, (20, 20, 180, 180))], crop_frame=420)
    (row,) = _rows(job, tmp_path)
    assert row["trigger"] == "early_ocr"
    assert job._read_trigger_counts == {"early_ocr": 1}


def test_finalize_folds_trigger_counts_for_non_clef_reader(capsys):
    job = _job()
    job._count_read_trigger("re_attachment", 10.0)
    job._count_read_trigger("census", 12.0)
    job._count_read_trigger("census", 20.0)
    job._count_read_trigger("re_attachment", dropped=True)
    job._finalize_clef_usage()
    job._finalize_clef_usage()                               # idempotent
    t = job.timings
    assert t["reads.trigger.re_attachment"] == 1 and t["reads.trigger.census"] == 2
    assert t["reads.total"] == 3 and t["reads.dropped_evicted"] == 1
    assert t["reads.census_per_s"] == pytest.approx(2 / 10.0)
    assert all(isinstance(v, (int, float)) for v in t.values())
    lines = [l for l in capsys.readouterr().out.splitlines() if "[reads]" in l]
    assert len(lines) == 1
    assert "triggers: census=2, re_attachment=1" in lines[0]
    assert "1 dropped (identity evicted in flight)" in lines[0]


def test_finalize_appends_trigger_counts_to_the_clef_line(capsys):
    job = _job(reader="clef")
    job._clef_session = clef_reader.ClefSession(clef_reader.DEFAULT_MODEL,
                                                account="acct", token="tok")
    job._count_read_trigger("unconfirmed_retry", 1.0)
    job._finalize_clef_usage()
    out = capsys.readouterr().out.splitlines()
    clef = [l for l in out if "[clef]" in l]
    assert len(clef) == 1 and "; triggers: unconfirmed_retry=1" in clef[0]
    assert not any("[reads]" in l for l in out)
    assert job.timings["reads.trigger.unconfirmed_retry"] == 1
    assert job.timings["clef.requests"] == 0


def test_read_for_identity_evicted_in_flight_is_counted_as_dropped(monkeypatch, tmp_path):
    job = _job(tmp_path, slots=[0, 1])

    def evicting_reader(crops, *a, **k):
        job.identity_registry.identities.pop(11)   # cap eviction mid-flight
        return [{"team_number": None, "confidence": 0.0, "raw_read": "",
                 "alliance": "unknown"} for _ in crops]
    monkeypatch.setattr(app_module, "_vision_read_bumpers", evicting_reader)
    asyncio.run(job._read_fused_bumper_numbers_impl(
        _fused([0, 1]), {10: 5.0, 11: 5.0}, det_to_slot={0: 0, 1: 1}, crop_frame=70,
        read_triggers={10: "census", 11: "re_attachment"}))
    assert [r["robot_id"] for r in _rows(job, tmp_path)] == [10]
    assert job._read_dropped_counts == {"re_attachment": 1}


# ===================================================== census: pure decision

def test_slot_evidence_margin():
    assert slot_evidence_margin(None) == 0.0
    assert slot_evidence_margin({}) == 0.0
    assert slot_evidence_margin({"4256": 3.5}) == 3.5
    assert slot_evidence_margin({"a": 2.0, "b": 1.5, "c": 0.5}) == pytest.approx(0.5)


def test_census_decision_due_after_interval():
    kw = dict(last_attempt_time=10.0, backoff_until=-999.0, interval_sec=2.0,
              evidence_margin=0.0)
    assert not census_read_decision(current_match_time=11.99, **kw).due
    d = census_read_decision(current_match_time=12.0, **kw)
    assert d.due and d.reason == "census" and not d.is_audit


def test_census_decision_backoff_only_when_honoured():
    kw = dict(current_match_time=13.0, last_attempt_time=10.0, backoff_until=14.0,
              interval_sec=2.0, evidence_margin=0.0)
    assert census_read_decision(**kw, honor_backoff=False).due
    d = census_read_decision(**kw, honor_backoff=True)
    assert not d.due and d.reason == "backed_off"


def test_census_priority_weakest_first_and_below_immediate():
    def prio(margin, now=12.0):
        return census_read_decision(current_match_time=now, last_attempt_time=10.0,
                                    backoff_until=-999.0, interval_sec=2.0,
                                    evidence_margin=margin).priority
    assert prio(0.0) > prio(0.5) > prio(3.0) > prio(20.0) >= CENSUS_BASE_PRIORITY
    # overdue term is capped: never reaches the priority-10 escalations
    assert prio(0.0, now=1e6) < 10.0
    assert prio(-5.0) == prio(0.0)          # negative margins clamp


# ================================================== census: the scheduler

def test_census_reads_confirmed_slot_every_interval(monkeypatch):
    """A confirmed identity (audits only every 30 s) is read every census
    interval: the gap is interval + at most one processed-frame stride."""
    job = _job(census=True, slots=[0])
    ident = job.identity_registry.identities[10]
    ident.is_confirmed = True
    ident.last_success_time = ident.last_vlm_time = 0.0
    reader = _sim(monkeypatch, job, seconds=20.0, slots=[0])
    t = reader.read_times(0)
    assert len(t) >= 8
    for gap in (b - a for a, b in zip(t, t[1:])):
        assert 2.0 <= gap < 2.0 + STRIDE / FPS + 1e-9
    assert set(job._read_trigger_counts) == {"census"}


def test_census_off_confirmed_slot_waits_for_audit(monkeypatch):
    job = _job(slots=[0])
    ident = job.identity_registry.identities[10]
    ident.is_confirmed = True
    ident.last_success_time = ident.last_vlm_time = 0.0
    reader = _sim(monkeypatch, job, seconds=20.0, slots=[0])
    assert reader.read_times(0) == []


def test_census_failing_slot_not_held_by_backoff(monkeypatch):
    """An illegible robot: census reads it every interval (the failure
    backoff, 3 s then 4 s, would cap it at ~1 per 4.1 s)."""
    job = _job(census=True, slots=[0])
    reader = _sim(monkeypatch, job, seconds=20.0, slots=[0], failing={0})
    gaps = [b - a for a, b in zip(reader.read_times(0), reader.read_times(0)[1:])]
    assert gaps and max(gaps) < 2.0 + STRIDE / FPS + 1e-9


def test_census_honor_backoff_knob_waits_out_backoff(monkeypatch):
    job = _job(census=True, honor=True, slots=[0])
    reader = _sim(monkeypatch, job, seconds=30.0, slots=[0], failing={0})
    gaps = [b - a for a, b in zip(reader.read_times(0), reader.read_times(0)[1:])]
    assert gaps[0] >= 3.0                      # backoff after 1 failure
    assert all(g >= 3.0 for g in gaps) and max(gaps) < 4.0 + STRIDE / FPS + 1e-9


def test_census_slot_clock_survives_identity_birth():
    """A new identity on a just-read slot is not census-due: births do not
    reset the slot's census clock (read supply decoupled from churn)."""
    job = _job(census=True, slots=[0])
    job._slot_last_read_attempt = {0: 10.0}
    reg = job.identity_registry.identities
    reg[10].active_slot = None
    reg[50] = RobotIdentity(identity_id=50, alliance="blue", active_slot=0)
    reg[50].is_confirmed = True               # nothing else would schedule it
    reg[50].last_success_time = 10.0
    assert job._collect_due_vlm_reads(11.0) == {}
    due = job._collect_due_vlm_reads(12.0)
    assert set(due) == {50} and job._due_triggers == {50: "census"}


def test_census_ignores_novelty_gate():
    """Failed read of an unmoved robot: the novelty gate holds the
    unconfirmed retry (no_new_view, up to 8 s); census does not."""
    for census, want in ((False, {}), (True, {10: "census"})):
        job = _job(census=census, slots=[0])
        ident = job.identity_registry.identities[10]
        ident.last_vlm_time = 10.0                # failed (no success since)
        ident.last_vlm_pos = (0.0, 1.0)
        ident.backoff_until = 12.0
        job._get_or_create_identity_track(10, "blue").field_position = (0.0, 1.0)
        job._collect_due_vlm_reads(13.0)
        assert job._due_triggers == want


def test_census_respects_occlusion_gate():
    job = _job(census=True, slots=[0])
    job._occ = {0: {"bumper_occluded": True}}
    assert job._collect_due_vlm_reads(50.0) == {}


def test_census_keeps_existing_trigger_and_takes_higher_priority():
    """An identity due anyway keeps its own trigger; census can only raise
    its priority (never lower an escalation's 10)."""
    job = _job(census=True, slots=[0, 1])
    reg = job.identity_registry.identities
    job._request_immediate_read(reg[10], "re_attachment", renew=True)
    due = job._collect_due_vlm_reads(50.0)
    assert job._due_triggers == {10: "re_attachment", 11: "unconfirmed_retry"}
    assert due[10] == 10.0
    assert due[11] > CENSUS_BASE_PRIORITY           # census priority beat 3-5


def test_weakest_slot_reads_first_when_pass_budget_is_limited(monkeypatch):
    reader = _Reader()
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    monkeypatch.setattr(app_module, "VLM_TOP_K", 2)
    job = _job(census=True, slots=[0, 1, 2, 3])
    for iid in (10, 11, 12, 13):
        i = job.identity_registry.identities[iid]
        i.is_confirmed = True
        i.last_success_time = i.last_vlm_time = 40.0
    job._census_slot_votes = {0: {"111": 5.0}, 1: {}, 2: {"333": 2.0, "111": 1.5},
                              3: {"444": 3.0}}
    reader.crop_frame = int(50 * FPS)
    _schedule_and_run(job, _fused([0, 1, 2, 3]), reader.crop_frame, clock=50.0)
    assert sorted(s for _f, s in reader.log) == [1, 2]   # margins 0 and 0.5


def test_census_tally_accumulates_named_reads_per_crop_slot(monkeypatch):
    reader = _Reader(failing={1})
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _job(census=True, slots=[0, 1])
    asyncio.run(job._read_fused_bumper_numbers_impl(
        _fused([0, 1]), {10: 6.0, 11: 6.0}, det_to_slot={0: 0, 1: 1}, crop_frame=600))
    asyncio.run(job._read_fused_bumper_numbers_impl(
        _fused([0, 1]), {10: 6.0, 11: 6.0}, det_to_slot={0: 0, 1: 1}, crop_frame=800))
    assert job._census_slot_votes == {0: {"111": 2.0}}
    assert job._slot_last_read_attempt == {0: pytest.approx(800 / FPS),
                                           1: pytest.approx(800 / FPS)}


def test_census_respects_single_flight(monkeypatch):
    """While a pass is in flight, census-due identities stay due and no second
    pass launches; each identity is read once per pass even with two crops."""
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocking(crops, *a, **k):
        calls.append(list(crops))
        entered.set()
        release.wait(timeout=5.0)
        return [{"team_number": None, "confidence": 0.0, "raw_read": "",
                 "alliance": "unknown"} for _ in crops]
    monkeypatch.setattr(app_module, "_vision_read_bumpers", blocking)
    job = _job(census=True, slots=[0, 1])
    fused = _fused([0, 1], cams=("main", "corner_left"))

    async def run():
        due = job._collect_due_vlm_reads(50.0)
        job._schedule_vlm_pass(fused, due, crop_frame=int(50 * FPS))
        first = job._vlm_task
        assert await asyncio.to_thread(entered.wait, 5.0)
        due2 = job._collect_due_vlm_reads(50.5)
        assert set(due2) == {10, 11}                 # still due ...
        job._schedule_vlm_pass(fused, due2, crop_frame=int(50.5 * FPS))
        assert job._vlm_task is first                # ... but single-flight
        release.set()
        await first
    asyncio.run(run())
    assert len(calls) == 1 and len(calls[0]) == 2    # one crop per identity


# ================================================== default OFF unchanged

def test_default_off_never_consults_census(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("census consulted with read_census off")
    monkeypatch.setattr(InferenceJob, "_census_due", boom)
    job = _job()
    assert job.config.read_census is False
    _sim(monkeypatch, job, seconds=60.0, failing={1, 4})
    assert not getattr(job, "_slot_last_read_attempt", None)
    assert not getattr(job, "_census_slot_votes", None)
    assert "census" not in job._read_trigger_counts
    assert sum(job._read_trigger_counts.values()) > 0


def test_default_off_collect_equals_decide_vlm_read():
    """Knob off: the due set and priorities are exactly decide_vlm_read's,
    over randomized identity states."""
    rng = random.Random(7)
    for _ in range(200):
        job = _job()
        now = rng.uniform(0, 160)
        for ident in job.identity_registry.identities.values():
            ident.is_confirmed = rng.random() < 0.3
            ident.confidence = rng.uniform(0, 1)
            ident.last_vlm_time = rng.choice([-999.0, now - rng.uniform(0, 40)])
            ident.last_success_time = rng.choice([-999.0, ident.last_vlm_time])
            ident.backoff_until = rng.choice([-999.0, now + rng.uniform(-3, 3)])
            ident.needs_immediate_read = rng.random() < 0.2
        unconfirmed = sum(1 for i in job.identity_registry.identities.values()
                          if not i.is_confirmed)
        want = {}
        for ident in job.identity_registry.identities.values():
            d = decide_vlm_read(
                is_confirmed=ident.is_confirmed, confidence=ident.confidence,
                last_vlm_time=ident.last_vlm_time,
                last_success_time=ident.last_success_time,
                backoff_until=ident.backoff_until,
                needs_immediate_read=ident.needs_immediate_read,
                current_match_time=now, unconfirmed_count=unconfirmed,
                config=job.confidence_config,
                last_read_failed=ident.last_vlm_time > ident.last_success_time,
                moved_since_read=True)
            if d.due:
                want[ident.identity_id] = d.priority
        assert job._collect_due_vlm_reads(now) == want


def test_knobs_default_on_and_bounds():
    cfg = JobConfig()
    assert cfg.read_census is True
    assert cfg.read_census_interval_s == 2.0
    assert cfg.read_census_honor_backoff is False
    for k in ("read_census", "read_census_interval_s", "read_census_honor_backoff"):
        assert k not in cfg.non_defaults()
    assert JobConfig({"read_census": False}).read_census is False
    on = JobConfig({"read_census": True, "read_census_interval_s": 3})
    assert on.read_census is True and on.read_census_interval_s == 3.0
    for bad in ({"read_census_interval_s": 0.1}, {"read_census_interval_s": 60.0}):
        with pytest.raises(ValueError):
            JobConfig(bad)


def test_knob_env_override(monkeypatch):
    monkeypatch.setenv("READ_CENSUS", "1")
    monkeypatch.setenv("READ_CENSUS_INTERVAL_S", "2.5")
    cfg = JobConfig()
    assert cfg.read_census is True and cfg.read_census_interval_s == 2.5


@pytest.mark.parametrize("reader, needle", [("vllm", "local GPU VLM time"),
                                            ("local", "local GPU VLM time"),
                                            ("clef", "on Clef")])
def test_census_prints_one_line_per_job(capsys, reader, needle):
    job = _job(census=True, reader=reader)
    for _ in range(3):
        assert job._read_census_enabled() is True
    lines = [l for l in capsys.readouterr().out.splitlines() if "[census]" in l]
    assert len(lines) == 1 and needle in lines[0]
    assert ("WARNING" in lines[0]) == (reader != "clef")


# ============================================ multi-frame (clef_multiframe)

_BOX = (200, 100, 340, 220)


def _view(f):
    rng = np.random.default_rng(f)
    img = rng.integers(0, 60, (400, 600, 3), dtype=np.uint8)
    img[160:220, 200:340] = (30, 30, 200)
    img[175:205, 220 + (f % 7) * 4:240 + (f % 7) * 4] = 255
    return img


def test_census_read_uses_the_identity_crop_buffer(monkeypatch, tmp_path):
    """With clef_multiframe on, a census read goes through the same pass and
    takes its earlier frames from the identity's crop buffer (it neither
    bypasses nor flushes it)."""
    seen = {}

    def fake(items, teams, blue, red, expected_alliances=None, provider="local",
             clef_item_meta=None, **_k):
        seen["items"], seen["meta"] = items, clef_item_meta
        return [{"team_number": None, "confidence": 0.0, "raw_read": "none",
                 "alliance": "unknown",
                 **({"clef_frames": m["frames"]} if m else {})}
                for m in (clef_item_meta or [None] * len(items))]
    monkeypatch.setattr(app_module, "_vision_read_bumpers", fake)
    job = _job(tmp_path, census=True, reader="clef", multiframe=True, slots=[1])
    ident = job.identity_registry.identities[11]
    crop_frame = 4000 + 8 * 11
    ident.is_confirmed = True                       # audits only: census reads it
    ident.last_success_time = ident.last_vlm_time = crop_frame / FPS - 5.0
    job._clef_session = types.SimpleNamespace(available=True)
    fused = None
    for f in range(0, 8 * 12, 8):
        frame = _view(f)

        def _enc(_f=frame):
            return job._bumper_crop_b64(_f, _BOX, enforce_floor=False)[0]
        lazy = app_module._LazyCrop(_enc, src=(frame, _BOX, False))
        fused = [types.SimpleNamespace(
            evidence=[("main", lazy, 1.0, (4.0, 3.0))], track_id=5,
            detection=types.SimpleNamespace(pos=(4.0, 3.0)),
            relink_evidence=types.SimpleNamespace(type="continued"))]
        job.field_tracker.last_det_to_slot = {0: 1}
        job._clef_mf_push(fused, 4000 + f)
    due = _schedule_and_run(job, fused, crop_frame)
    assert job._due_triggers == {11: "census"} and set(due) == {11}
    assert isinstance(seen["items"][0], list) and len(seen["items"][0]) >= 2
    assert seen["meta"][0]["frames"][0]["frame"] == crop_frame
    (row,) = _rows(job, tmp_path)
    assert row["trigger"] == "census" and len(row["clef_frames"]) >= 2
    assert job._crop_buffer.stats()["flushes"] == {}


# ====================================================== offline simulation

def test_offline_sim_qm11_census_read_supply(capsys):
    """Replays the qm11 full-match identity timeline (326 identities, 445
    bindings) through the real scheduler + read pass. Census ON must give
    every slot a read every ~2-3 s (p90 gap) and 400-600 reads/match."""
    import scripts.sim_read_census as sim
    fx = sim.load_fixture()
    arms = sim.run_arms(fx)
    with capsys.disabled():
        print("\n" + sim.format_report(fx, arms, 2.0))
    off, on, honor = (a for _n, a in arms)
    assert 400 <= on["total"] <= 600
    for lab in sim.LABELS:
        assert on["per_slot"][lab]["p90_gap_s"] <= 3.5, lab
        assert on["per_slot"][lab]["reads"] > off["per_slot"][lab]["reads"], lab
    assert "census" not in off["triggers"] and on["triggers"]["census"] > 200
    assert off["total"] < honor["total"] < on["total"]
    assert on["named"] > off["named"]

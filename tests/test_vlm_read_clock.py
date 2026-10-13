"""Bumper-read scheduling must run on ONE real-time clock.

The main loop scheduled reads with ``processed_frame_num / fps`` (processed
frames, i.e. a clock running ``frame_skip``x slow and starting at 0 at the
first processing frame) while the read pass stamped ``last_vlm_time`` /
``last_success_time`` / ``backoff_until`` with ``_current_frame / fps``
(source-video seconds). Elapsed time was negative or tiny, so interval-based
reads (unconfirmed retries, audits) never came due: the last 40 replays logged
89,357 ``immediate_read_requested`` vs 14 ``unconfirmed_retry`` and 0
``audit_due``.

Second, ``needs_immediate_read`` was cleared only by a SUCCESSFUL read, and a
priority-10 immediate read bypasses backoff — so an identity whose bumper is
illegible was due for a read on every processed frame, throttled only by the
single-flight pass.

Fix (``VLM_READ_CLOCK_FIX``, default ON):
  * scheduler clock = source frame being processed / fps (``frame_num``);
  * read stamps = the CROP frame / fps (the frame the read pass was scheduled
    for), not the live ``_current_frame`` at task start;
  * a completed attempt without a usable team consumes the immediate request
    (requests raised after the crop frame survive), so the identity falls
    back to backoff + unconfirmed cadence until a NEW trigger fires.
"""
import ast
import asyncio
import inspect
import types
from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pytest

import web.app as app_module
from web.app import InferenceJob
from web.field.identity import IdentityRegistry, RobotIdentity
from web.field.identity_confidence import ConfidenceConfig, read_clock_sec
from web.job_config import JobConfig

FPS = 59.94
STRIDE = 7  # frame_skip at 59.94 fps -> ~8.5 processed fps

TEAMS = ["111", "222", "333", "444", "555", "666"]
SLOTS = list(range(6))          # B1..B3 = 0..2, R1..R3 = 3..5
IDENT_OF_SLOT = {s: 10 + s for s in SLOTS}


# ---------------------------------------------------------------- harness

def _make_job(slots=SLOTS):
    """Bare InferenceJob with a REAL IdentityRegistry / ConfidenceConfig and
    the REAL scheduler + read-pass methods (only the reader is faked)."""
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig()
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
        last_det_to_slot={i: s for i, s in enumerate(slots)})
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
    job.confidence_config = ConfidenceConfig()
    job.team_reader = "local"
    job.gallery = None
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    job.identity_diagnostics = MagicMock()
    job.diagnostic = MagicMock()
    return job


def _fused(slots=SLOTS):
    return [types.SimpleNamespace(
        evidence=[("main", f"crop{s}", 1.0, (float(s), 1.0))],
        detection=types.SimpleNamespace(pos=(float(s), 1.0)),
    ) for s in slots]


class _Reader:
    """Fake _vision_read_bumpers: legible slots read their team, failing
    slots return no team. Records (crop_frame, slot) per read."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.crop_frame = None
        self.log = []

    def __call__(self, crops, teams, blue, red, expected_alliances=None,
                 provider="local", **_kw):
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

    def reads_per_slot(self):
        return Counter(s for _f, s in self.log)

    def read_times(self, slot):
        return [f / FPS for f, s in self.log if s == slot]


async def _run_pass(job, reader, due, crop_frame, slots=SLOTS):
    reader.crop_frame = crop_frame
    await job._read_fused_bumper_numbers_impl(
        _fused(slots), due,
        det_to_slot={i: s for i, s in enumerate(slots)},
        crop_frame=crop_frame)


async def _simulate(job, reader, *, seconds=150.0, start_frame=0, latency=0,
                    triggers=None, slots=SLOTS):
    """Drive the real scheduler (_read_clock_sec + _collect_due_vlm_reads)
    and the real read pass frame by frame, mirroring the main loop: the
    clock is computed for the frame being processed, at most one pass is in
    flight (single-flight), a pass completes `latency` processed frames after
    it was scheduled, and _current_frame is assigned at the END of the frame.

    triggers: {processed_index: [identity_id, ...]} — a new immediate-read
    trigger (re-attachment etc.) raised in that frame's identity sync."""
    triggers = triggers or {}
    # The early-OCR pass leaves _current_frame at the processing start.
    job._current_frame = start_frame
    pending = None  # (complete_at_index, crop_frame, due)
    n = int(seconds * FPS / STRIDE)
    for i in range(n):
        frame_num = start_frame + i * STRIDE
        for iid in triggers.get(i, ()):
            ident = job.identity_registry.identities[iid]
            ident.needs_immediate_read = True
            ident.last_immediate_queue_time = frame_num / FPS
        if pending is not None and i >= pending[0]:
            await _run_pass(job, reader, pending[2], pending[1], slots)
            pending = None
        clock = job._read_clock_sec(frame_num, i)
        due = job._collect_due_vlm_reads(clock)
        if due and pending is None:
            if latency == 0:
                await _run_pass(job, reader, due, frame_num, slots)
            else:
                pending = (i + latency, frame_num, due)
        job._current_frame = frame_num
    return reader


def _sim(monkeypatch, *, fixed, failing=(), slots=SLOTS, **kw):
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", fixed)
    reader = _Reader(failing)
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job(slots)
    kw.setdefault("triggers", {0: [IDENT_OF_SLOT[s] for s in slots]})
    asyncio.run(_simulate(job, reader, slots=slots, **kw))
    return job, reader


# ---------------------------------------------------------------- clock

def test_read_clock_fixed_is_source_video_seconds():
    assert read_clock_sec(700, 100, FPS, fixed=True) == pytest.approx(700 / FPS)


def test_read_clock_legacy_is_processed_index_seconds():
    assert read_clock_sec(700, 100, FPS, fixed=False) == pytest.approx(100 / FPS)


def test_read_clock_falls_back_to_10fps_when_fps_unknown():
    assert read_clock_sec(50, 5, 0, fixed=True) == pytest.approx(5.0)
    assert read_clock_sec(50, 5, None, fixed=False) == pytest.approx(0.5)


def test_job_read_clock_follows_kill_switch(monkeypatch):
    job = _make_job()
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    assert job._read_clock_sec(700, 100) == pytest.approx(700 / FPS)
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", False)
    assert job._read_clock_sec(700, 100) == pytest.approx(100 / FPS)


def test_kill_switch_defaults_on():
    assert app_module.VLM_READ_CLOCK_FIX is True


def _app_tree():
    with open(inspect.getsourcefile(app_module), "r", encoding="utf-8") as f:
        return ast.parse(f.read())


def test_main_loop_clock_is_frame_being_processed():
    """The loop's read clock must be computed from frame_num (the source frame
    being processed), not processed_frame_num or the lagging _current_frame."""
    calls = [n for n in ast.walk(_app_tree())
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_read_clock_sec"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"]
    assert calls, "main loop must compute its clock via self._read_clock_sec"
    for c in calls:
        assert isinstance(c.args[0], ast.Name) and c.args[0].id == "frame_num"
        assert isinstance(c.args[1], ast.Name) and c.args[1].id == "processed_frame_num"


def test_no_processed_frame_seconds_left_in_app():
    """No call site may divide processed_frame_num by fps outside the legacy
    branch of read_clock_sec."""
    src = inspect.getsource(app_module)
    assert "processed_frame_num / (self.video_fps" not in src


# ---------------------------------------------------------------- stamps

def test_read_pass_stamps_crop_frame_seconds(monkeypatch):
    """The pass body starts at a later loop yield (_current_frame has moved
    on); stamps must name the crop frame the scheduler compared against."""
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    reader = _Reader(failing={1})
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job()
    job._current_frame = 7 * 40  # live frame several strides past the crop
    asyncio.run(_run_pass(job, reader, {10: 10.0, 11: 10.0}, crop_frame=7 * 30))
    good, bad = (job.identity_registry.identities[i] for i in (10, 11))
    t_crop = 7 * 30 / FPS
    assert good.last_vlm_time == pytest.approx(t_crop)
    assert good.last_success_time == pytest.approx(t_crop)
    assert bad.last_vlm_time == pytest.approx(t_crop)
    assert bad.backoff_until == pytest.approx(t_crop + 2.0 * 1.5)


def test_legacy_read_pass_stamps_live_current_frame(monkeypatch):
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", False)
    reader = _Reader()
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job()
    job._current_frame = 7 * 40
    asyncio.run(_run_pass(job, reader, {10: 10.0}, crop_frame=7 * 30))
    assert job.identity_registry.identities[10].last_vlm_time == pytest.approx(
        7 * 40 / FPS)


# ------------------------------------------------- scheduler, stride 7 sim

def test_unconfirmed_retry_elapses_in_real_seconds(monkeypatch):
    """Legible single robot: immediate read at t=0, then unconfirmed retries
    every unconfirmed_med_interval_sec (1.25 s, confidence 0.5) of REAL time
    until the 3-read consecutive lock."""
    _job, reader = _sim(monkeypatch, fixed=True, slots=[0], seconds=10.0)
    t = reader.read_times(0)
    assert t[0] == pytest.approx(0.0)
    assert len(t) >= 3
    for gap in (t[1] - t[0], t[2] - t[1]):
        assert 1.25 <= gap < 1.25 + STRIDE / FPS + 1e-9


def test_audit_due_after_audit_floor(monkeypatch):
    """Once confirmed, the identity is re-read (audit) every
    audit_floor_seconds of real time — never in legacy timing."""
    job, reader = _sim(monkeypatch, fixed=True, slots=[0], seconds=150.0)
    assert job.identity_registry.identities[10].is_confirmed
    t = reader.read_times(0)
    lock_t = t[2]
    audits = t[3:]
    assert len(audits) >= 3, t
    gaps = [b - a for a, b in zip([lock_t] + audits, audits)]
    for gap in gaps:
        assert 30.0 <= gap < 30.0 + STRIDE / FPS + 1e-9, gaps
    reasons = [c.args[0] for c in job.identity_diagnostics.increment.call_args_list]
    assert "vlm_scheduled.audit_due" in reasons
    assert "vlm_scheduled.unconfirmed_retry" in reasons


def test_failing_immediate_read_backs_off(monkeypatch):
    """An illegible robot flagged immediate: the first failed attempt
    consumes the request; later reads follow backoff_base_sec * 1.5**n,
    clamped to backoff_max_sec (4 s) so it is still retried every few s."""
    job, reader = _sim(monkeypatch, fixed=True, slots=[0], failing={0},
                       seconds=40.0)
    ident = job.identity_registry.identities[10]
    assert ident.needs_immediate_read is False
    t = reader.read_times(0)
    gaps = [b - a for a, b in zip(t, t[1:])]
    # backoff after n consecutive failures = min(2.0 * 1.5**n, 4.0)
    for n, gap in enumerate(gaps, start=1):
        want = min(2.0 * 1.5 ** n, 4.0)
        assert want <= gap < want + STRIDE / FPS + 1e-9, gaps
    assert len(t) <= 11, t  # ~40 s / ~4.1 s, never every frame


def test_new_trigger_rerequests_immediately(monkeypatch):
    """A failing identity deep in backoff is read on the very frame a NEW
    trigger (re-attachment, low margin, ...) fires — then backs off again."""
    trig = int(20.0 * FPS / STRIDE)  # t ~= 20 s, mid-backoff
    job, reader = _sim(monkeypatch, fixed=True, slots=[0], failing={0},
                       seconds=30.0, triggers={0: [10], trig: [10]})
    frames = [f for f, _s in reader.log]
    assert trig * STRIDE in frames, "new trigger did not read on its own frame"
    before_trig = [f for f in frames if f < trig * STRIDE]
    # 0, 3.0, then every ~4.1 s (backoff capped at 4 s): 7.1, 11.2, 15.3, 19.4
    assert len(before_trig) == 6, before_trig
    after = [f for f in frames if f > trig * STRIDE]
    # then back to the capped backoff (~4 s), not every frame: ~2 more by 30 s
    assert 1 <= len(after) <= 3, after
    assert job.identity_registry.identities[10].needs_immediate_read is False


def test_trigger_raised_after_crop_survives_failed_read(monkeypatch):
    """A trigger that fires while the pass is in flight (queued AFTER the
    crop frame) is a new request: the failed read must not consume it."""
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    reader = _Reader(failing={0, 1})
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job()
    crop = 7 * 30
    late, early = (job.identity_registry.identities[i] for i in (10, 11))
    late.needs_immediate_read = True
    late.last_immediate_queue_time = (crop + 2 * STRIDE) / FPS
    early.needs_immediate_read = True
    early.last_immediate_queue_time = crop / FPS
    asyncio.run(_run_pass(job, reader, {10: 10.0, 11: 10.0}, crop_frame=crop))
    assert late.needs_immediate_read is True
    assert early.needs_immediate_read is False


def test_no_crop_attempt_consumes_request(monkeypatch):
    """An identity scheduled but absent from the crop frame's fused set is a
    completed (no-crop) attempt: backoff applies, request consumed."""
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    reader = _Reader()
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job()
    ident = job.identity_registry.identities[15]
    ident.needs_immediate_read = True
    reader.crop_frame = 70
    asyncio.run(job._read_fused_bumper_numbers_impl(
        _fused([0]), {10: 10.0, 15: 10.0}, det_to_slot={0: 0}, crop_frame=70))
    assert ident.needs_immediate_read is False
    assert ident.backoff_until == pytest.approx(70 / FPS + 3.0)


def test_successful_read_still_clears_request(monkeypatch):
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    reader = _Reader()
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job()
    ident = job.identity_registry.identities[10]
    ident.needs_immediate_read = True
    ident.last_immediate_queue_time = 70 / FPS
    asyncio.run(_run_pass(job, reader, {10: 10.0}, crop_frame=70))
    assert ident.needs_immediate_read is False


def test_ambiguous_success_keeps_request(monkeypatch):
    """Unchanged: a co-occlusion-ambiguous identity stays immediate after a
    successful read until its group resolves."""
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    reader = _Reader()
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job()
    ident = job.identity_registry.identities[10]
    ident.needs_immediate_read = True
    ident.is_ambiguous = True
    asyncio.run(_run_pass(job, reader, {10: 10.0}, crop_frame=70))
    assert ident.needs_immediate_read is True


def test_lock_on_one_robot_reflags_another_in_same_batch(monkeypatch):
    """_clear_team_from_other_robots re-requests a read on any identity that
    held the just-locked team. That new request must survive even when the
    other identity's own (failed) result is processed later in the batch."""
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", True)
    reader = _Reader(failing={1})
    monkeypatch.setattr(app_module, "_vision_read_bumpers", reader)
    job = _make_job()
    a, b = (job.identity_registry.identities[i] for i in (10, 11))
    b.team_number = TEAMS[0]  # B wrongly holds A's team
    b.needs_immediate_read = True         # B's own request predates the crop
    b.last_immediate_queue_time = 70 / FPS
    job._get_or_create_slot_track(1)      # B's track exists (it was read before)
    track_a = job._get_or_create_slot_track(0)
    track_a.consecutive_tracker.update(TEAMS[0])
    track_a.consecutive_tracker.update(TEAMS[0])  # A's 3rd read locks
    asyncio.run(_run_pass(job, reader, {10: 10.0, 11: 10.0}, crop_frame=70))
    assert a.is_confirmed and a.team_number == TEAMS[0]
    assert b.team_number is None
    assert b.needs_immediate_read is True


# ------------------------------------------- identity-sync escalation stamp

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


def _sync_once(job, log_frame_num):
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    asyncio.run(job._sync_identity_registry_frame(
        [MagicMock(track_id=1, detection=MagicMock(pos=(1.0, 1.0), embedding=emb))],
        [], [MagicMock(slot_id=0, pos=(1.0, 1.0), alliance="blue")],
        log_frame_num=log_frame_num))
    (ident,) = job.identity_registry.identities.values()
    return ident


@pytest.mark.parametrize("fixed, want_frame", [(True, 100), (False, 93)])
def test_sync_escalation_stamp_uses_frame_being_processed(
        monkeypatch, fixed, want_frame):
    """_current_frame lags one stride inside the loop (assigned at loop END);
    the escalation cooldown clock must be the frame being processed."""
    monkeypatch.setattr(app_module, "VLM_READ_CLOCK_FIX", fixed)
    ident = _sync_once(_sync_job(current_frame=93), log_frame_num=100)
    assert ident.needs_immediate_read is True  # re-attachment
    assert ident.last_immediate_queue_time == pytest.approx(want_frame / FPS)


# ------------------------------------------------------------ kill-switch

def test_kill_switch_restores_legacy_timing(monkeypatch):
    """VLM_READ_CLOCK_FIX=0: slow processed clock vs real-time stamps (no
    retries / audits after the first couple of reads) and failing immediate
    reads re-fire on every pass."""
    job, reader = _sim(monkeypatch, fixed=False, slots=[0, 1], failing={1},
                       seconds=150.0)
    per = reader.reads_per_slot()
    n_frames = int(150.0 * FPS / STRIDE)
    assert per[1] == n_frames            # every processed frame
    assert per[0] <= 3                    # retries/audits never come due
    reasons = [c.args[0] for c in job.identity_diagnostics.increment.call_args_list]
    assert "vlm_scheduled.audit_due" not in reasons
    assert job.identity_registry.identities[11].needs_immediate_read is True


# ------------------------------------------------- offline cost estimate

def _estimate(monkeypatch, fixed, latency):
    _job, reader = _sim(monkeypatch, fixed=fixed, failing={1, 4},
                        seconds=150.0, start_frame=int(5.0 * FPS),
                        latency=latency)
    per = reader.reads_per_slot()
    return [per[s] for s in SLOTS]


@pytest.mark.parametrize("latency", [0, 3])
def test_offline_reads_per_match_estimate(monkeypatch, latency, capsys):
    """150 s match, stride 7 @ 59.94 fps, 6 identities (B2 + R2 always
    illegible), all flagged immediate at t=0. `latency` = processed frames a
    read pass stays in flight (0 = upper bound; 3 ~= a 0.35 s video-time
    pass). Prints reads per identity before/after."""
    before = _estimate(monkeypatch, False, latency)
    after = _estimate(monkeypatch, True, latency)
    with capsys.disabled():
        print(f"\n[latency={latency}] reads/identity  "
              + "  ".join(app_module.InferenceJob._slot_label(s) for s in SLOTS))
        print(f"  before (legacy): {before}  total={sum(before)}")
        print(f"  after  (fixed):  {after}  total={sum(after)}")
    for s in (1, 4):                      # illegible: cost collapses
        assert after[s] < before[s] / 10
        assert after[s] <= 40             # ~150 s / ~4.1 s capped backoff
    for s in (0, 2, 3, 5):                # legible: lock + periodic audits
        assert after[s] >= 3 + 3
        assert after[s] > before[s]


# --- backoff ceiling (review follow-up) ------------------------------------

def test_read_backoff_capped_under_clock_fix():
    from web.field.identity_confidence import ConfidenceConfig, read_backoff_sec
    cfg = ConfidenceConfig()
    assert read_backoff_sec(cfg, 0) == cfg.backoff_base_sec
    # 2 * 1.5^7 ~= 34s uncapped; the fix clamps a long-unreadable robot to
    # the ceiling so it is still retried every few seconds.
    assert read_backoff_sec(cfg, 7) == cfg.backoff_max_sec == 4.0
    assert read_backoff_sec(cfg, 7, capped=False) > 30.0


def test_read_backoff_ceiling_env_override(monkeypatch):
    from web.field.identity_confidence import ConfidenceConfig, read_backoff_sec
    monkeypatch.setenv("VLM_READ_BACKOFF_MAX_S", "6")
    assert read_backoff_sec(ConfidenceConfig(), 10) == 6.0

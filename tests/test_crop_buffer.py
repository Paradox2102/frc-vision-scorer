"""Multi-frame Clef reads: the per-identity rolling crop buffer and the app's
per-frame push / flush hooks (web/field/crop_buffer.py, InferenceJob._clef_mf_push).

All offline: no HTTP, no GPU, no database."""
from __future__ import annotations

import socket
import time
import types
import urllib.request

import numpy as np
import pytest

from web.field.crop_buffer import CropBuffer, CropEntry

FPS = 60.0


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []

    def _tripwire(*a, **k):
        attempts.append(a[:1])
        raise AssertionError("network call attempted in an offline test")

    monkeypatch.setattr(urllib.request, "urlopen", _tripwire)
    monkeypatch.setattr(socket, "create_connection", _tripwire)
    yield
    assert not attempts, f"offline test reached the network: {attempts}"


def _crop(h=30, w=80, val=128):
    return np.full((h, w, 3), val, np.uint8)


def _buf(**kw):
    kw.setdefault("fps", FPS)
    kw.setdefault("window_s", 3.0)
    return CropBuffer(**kw)


# --------------------------------------------------------------------------- #
# Window bounds, cap, copies
# --------------------------------------------------------------------------- #

def test_window_bounds_entries_to_window_of_source_time():
    b = _buf()
    for f in range(0, 601, 7):
        b.push(1, f, [("main", _crop(), 0.1)], pos=(1.0, 1.0), slot=0)
    frames = [e.frame for e in b.entries(1)]
    newest = max(frames)
    assert newest == 595
    assert min(frames) >= newest - 3.0 * FPS
    assert all(newest - f <= 180 for f in frames)


def test_hard_cap_keeps_newest_entries():
    b = _buf(max_entries=10)
    for f in range(0, 100, 7):
        b.push(1, f, [(c, _crop(), 0.1) for c in ("main", "corner_left", "corner_right")],
               pos=(1.0, 1.0), slot=0)
    ents = b.entries(1)
    assert len(ents) == 10
    assert max(e.frame for e in ents) == 98
    # Oldest dropped first.
    assert min(e.frame for e in ents) >= 98 - 7 * 3


def test_buffered_crop_is_an_independent_copy():
    b = _buf()
    frame = np.zeros((100, 200, 3), np.uint8)
    view = frame[10:40, 20:100]          # a view into the frame
    b.push(1, 0, [("main", view, 0.2)], pos=(0.0, 0.0), slot=0)
    frame[:] = 255
    e = b.entries(1)[0]
    assert not np.shares_memory(e.crop, frame)
    assert int(e.crop.max()) == 0
    assert e.area == 30 * 80


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def _fill(b, frames, leg_of=lambda f: 0.1, cam="main", slot=0, area=(30, 80)):
    for f in frames:
        b.push(1, f, [(cam, _crop(*area), leg_of(f))], pos=(1.0, 1.0), slot=slot)


def test_selection_is_newest_plus_sharpest_of_each_earlier_third():
    b = _buf()
    frames = list(range(0, 181, 6))      # window [0, 180]; newest 180
    peaks = {12: 0.9, 84: 0.8, 150: 0.7}  # one sharp frame per third
    _fill(b, frames, leg_of=lambda f: peaks.get(f, 0.1))
    sel = b.select(1, 180)
    assert [e.frame for e in sel] == [180, 150, 84, 12]     # newest first
    assert all(isinstance(e, CropEntry) for e in sel)


def test_selection_respects_n_frames():
    b = _buf()
    _fill(b, range(0, 181, 6))
    assert len(b.select(1, 180, n=2)) == 2
    assert len(b.select(1, 180, n=3)) == 3


def test_sharpness_ties_break_on_area_then_recency():
    b = _buf()
    # Every frame in the first third has the same (clamped) legibility.
    for f in (6, 12, 18):
        b.push(1, f, [("main", _crop(30, 80 if f != 12 else 120), 0.5)], pos=(1, 1), slot=0)
    for f in (24, 30, 36):   # equal legibility AND area -> most recent wins
        b.push(1, f, [("main", _crop(30, 80), 0.5)], pos=(1, 1), slot=0)
    b.push(1, 180, [("main", _crop(), 0.5)], pos=(1, 1), slot=0)
    sel = b.select(1, 180, n=2)     # one earlier pick from the single segment
    assert sel[1].frame == 12       # biggest crop
    b2 = _buf()
    for f in (24, 30, 36):
        b2.push(1, f, [("main", _crop(30, 80), 0.5)], pos=(1, 1), slot=0)
    b2.push(1, 180, [("main", _crop(), 0.5)], pos=(1, 1), slot=0)
    assert b2.select(1, 180, n=2)[1].frame == 36


def test_legibility_compares_within_camera_only():
    """A corner camera's raw legibility scale differs from main's; each
    camera's entries are ranked against that camera's own window, so a corner
    crop that is its camera's sharpest competes on equal terms (then area)."""
    b = _buf()
    for f in range(6, 60, 6):          # first third: main 0.30..., corner tiny scale
        b.push(1, f, [("main", _crop(30, 80), 0.30 if f == 30 else 0.2),
                      ("corner_left", _crop(60, 160), 0.02 if f == 42 else 0.01)],
               pos=(1, 1), slot=0)
    b.push(1, 180, [("main", _crop(), 0.2)], pos=(1, 1), slot=0)
    sel = b.select(1, 180, n=2)
    # Both camera bests have rank 1.0; the corner crop is bigger -> it wins.
    assert (sel[1].frame, sel[1].cam) == (42, "corner_left")


def test_zero_legibility_never_beats_a_scored_crop():
    b = _buf()
    b.push(1, 6, [("corner_left", _crop(200, 400), 0.0)], pos=(1, 1), slot=0)
    b.push(1, 12, [("main", _crop(30, 80), 0.05)], pos=(1, 1), slot=0)
    b.push(1, 180, [("main", _crop(), 0.2)], pos=(1, 1), slot=0)
    assert b.select(1, 180, n=2)[1].frame == 12
    # Zero-score earlier crops are never sent at all.
    b2 = _buf()
    b2.push(1, 6, [("main", _crop(), 0.0)], pos=(1, 1), slot=0)
    b2.push(1, 180, [("main", _crop(), 0.0)], pos=(1, 1), slot=0)
    assert [e.frame for e in b2.select(1, 180)] == [180]


def test_sent_frames_are_excluded_but_newest_is_kept():
    b = _buf()
    _fill(b, range(0, 181, 6), leg_of=lambda f: f / 1000.0)
    first = b.select(1, 180)
    assert len(first) == 4
    b.mark_sent(1, [e.frame for e in first])
    b.push(1, 186, [("main", _crop(), 0.1)], pos=(1, 1), slot=0)
    second = b.select(1, 186)
    assert second[0].frame == 186
    assert not ({e.frame for e in second[1:]} & {e.frame for e in first})
    assert set(b.sent_frames(1)) >= {e.frame for e in first}


def test_backfill_when_thirds_are_empty():
    """Right after a flush only the newest part of the window has entries: the
    empty thirds are backfilled with the next-best fresh frames."""
    b = _buf()
    _fill(b, [160, 166, 172, 178, 180], leg_of=lambda f: 0.1 + f / 1e4)
    sel = b.select(1, 180)
    assert [e.frame for e in sel] == [180, 178, 172, 166]
    assert len({e.frame for e in sel}) == len(sel)          # one entry per frame


def test_select_needs_an_entry_at_the_newest_frame_and_slot():
    b = _buf()
    _fill(b, range(0, 121, 6), slot=2)
    assert b.select(1, 126) == []                 # nothing at the newest frame
    assert b.select(1, 120, slot=3) == []         # newest frame owned by another slot
    assert b.select(1, 120, slot=2)[0].frame == 120
    assert b.select(99, 120) == []                # unknown identity


def test_select_prefers_the_requested_newest_camera():
    b = _buf()
    b.push(1, 6, [("main", _crop(), 0.1)], pos=(1, 1), slot=0)
    b.push(1, 12, [("main", _crop(), 0.9), ("corner_right", _crop(), 0.1)],
           pos=(1, 1), slot=0)
    assert b.select(1, 12, newest_cam="corner_right")[0].cam == "corner_right"
    assert b.select(1, 12, newest_cam="main")[0].cam == "main"


def test_single_entry_buffer_selects_only_the_newest():
    b = _buf()
    b.push(1, 60, [("main", _crop(), 0.3)], pos=(1, 1), slot=0)
    assert [e.frame for e in b.select(1, 60)] == [60]


def test_entries_newer_than_the_read_frame_are_ignored():
    """A background pass that starts a frame late must not pick crops cut after
    its scheduling frame."""
    b = _buf()
    _fill(b, range(0, 199, 6))
    sel = b.select(1, 120)
    assert sel[0].frame == 120 and all(e.frame <= 120 for e in sel)


# --------------------------------------------------------------------------- #
# Flushes inside the buffer
# --------------------------------------------------------------------------- #

def test_position_jump_flushes_before_push():
    b = _buf(max_jump_m=2.0)
    b.push(1, 0, [("main", _crop(), 0.1)], pos=(1.0, 1.0), slot=0)
    b.push(1, 7, [("main", _crop(), 0.1)], pos=(1.5, 1.0), slot=0)
    assert len(b.entries(1)) == 2
    b.push(1, 14, [("main", _crop(), 0.1)], pos=(4.0, 1.0), slot=0)   # 2.5 m jump
    assert [e.frame for e in b.entries(1)] == [14]
    assert b.stats()["flushes"]["jump"] == 1


def test_flush_clears_entries_and_sent_frames():
    b = _buf()
    _fill(b, range(0, 61, 6))
    b.mark_sent(1, [0, 6])
    assert b.flush(1, "rebind") is True
    assert b.entries(1) == [] and not b.sent_frames(1)
    assert b.flush(1, "rebind") is False          # nothing left
    assert b.stats()["flushes"] == {"rebind": 1}


def test_retain_drops_evicted_identities():
    b = _buf()
    for iid in (1, 2, 3):
        b.push(iid, 0, [("main", _crop(), 0.1)], pos=(1, 1), slot=iid)
    assert b.retain({1, 3}) == 1
    assert b.entries(2) == [] and b.entries(1) and b.entries(3)
    assert b.stats()["flushes"]["evicted"] == 1


# --------------------------------------------------------------------------- #
# App hooks: push + flush triggers (InferenceJob._clef_mf_push)
# --------------------------------------------------------------------------- #

import web.app as app_module                   # noqa: E402
from web.job_config import JobConfig           # noqa: E402

VIEW = np.random.default_rng(0).integers(0, 255, (723, 1920, 3), dtype=np.uint8)
CORNER = np.random.default_rng(1).integers(0, 255, (377, 960, 3), dtype=np.uint8)


def _ident(iid, slot, **kw):
    base = dict(identity_id=iid, active_slot=slot, needs_immediate_read=False,
                last_immediate_queue_time=-999.0, alliance="blue")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _mf_job(idents, multiframe=True, reader="clef"):
    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    cfg = {"team_reader": reader}
    if multiframe:
        cfg["clef_multiframe"] = True
    job.config = JobConfig(cfg)
    job.team_reader = reader
    job.video_fps = FPS
    job.identity_registry = types.SimpleNamespace(identities={i.identity_id: i for i in idents})
    job.field_tracker = types.SimpleNamespace(binding_events=[], last_det_to_slot={})
    job._occ = {}
    job._occ_by_cam = {}
    job._camera_centers_xy = {}
    return job


def _lazy(frame, box):
    return app_module._LazyCrop(lambda: "unused", src=(frame, box, False))


def _obs(tid, pos=(4.0, 3.0), ev_type="continued", cams=("main",), box=(400, 300, 520, 400)):
    evidence = []
    for cam in cams:
        frame = VIEW if cam == "main" else CORNER
        evidence.append((cam, _lazy(frame, box), 1.0 if cam == "main" else 2.5, pos))
    return types.SimpleNamespace(
        evidence=evidence, track_id=tid,
        detection=types.SimpleNamespace(pos=pos),
        relink_evidence=types.SimpleNamespace(type=ev_type))


def _frame(job, frame_num, obs_by_slot, events=()):
    """Run one push: obs_by_slot = {slot: obs}; det index order = dict order."""
    fused = list(obs_by_slot.values())
    job.field_tracker.last_det_to_slot = {i: s for i, s in enumerate(obs_by_slot)}
    job.field_tracker.binding_events = list(events)
    job._clef_mf_push(fused, frame_num)
    return fused


def _frames_of(job, iid):
    return [e.frame for e in job._crop_buffer.entries(iid)]


def test_push_is_a_noop_when_knob_off_or_reader_not_clef():
    for job in (_mf_job([_ident(7, 1)], multiframe=False),
                _mf_job([_ident(7, 1)], reader="vllm")):
        _frame(job, 0, {1: _obs(5)})
        assert getattr(job, "_crop_buffer", None) is None


def test_push_buffers_band_crops_per_identity_and_releases_frame_refs():
    job = _mf_job([_ident(7, 1), _ident(8, 2)])
    fused = _frame(job, 0, {1: _obs(5, cams=("main", "corner_left")), 2: _obs(6)})
    _frame(job, 7, {1: _obs(5, cams=("main", "corner_left")), 2: _obs(6)})
    ents = job._crop_buffer.entries(7)
    assert sorted({(e.frame, e.cam) for e in ents}) == [
        (0, "corner_left"), (0, "main"), (7, "corner_left"), (7, "main")]
    assert all(e.slot == 1 and e.crop.ndim == 3 for e in ents)
    assert not any(np.shares_memory(e.crop, VIEW) or np.shares_memory(e.crop, CORNER)
                   for e in ents)
    assert _frames_of(job, 8) == [0, 7]
    # The lazy crops no longer pin the view frame once pushed.
    assert all(c.src is None for o in fused for (_cam, c, _w, _p) in o.evidence)


def test_push_skips_main_crop_when_main_view_bumper_is_occluded():
    job = _mf_job([_ident(7, 1)])
    job._occ_by_cam = {"main": {1: {"bumper_occluded": True}}}
    job._occ = {1: {"bumper_occluded": False}}
    _frame(job, 0, {1: _obs(5, cams=("main", "corner_left"))})
    assert {e.cam for e in job._crop_buffer.entries(7)} == {"corner_left"}


def test_push_skips_identity_occluded_everywhere_without_corner_crop():
    job = _mf_job([_ident(7, 1)])
    job._occ = {1: {"bumper_occluded": True}}
    _frame(job, 0, {1: _obs(5)})
    assert job._crop_buffer.entries(7) == []


def _two_frames_then(job, third_obs, events=(), mutate=None):
    _frame(job, 0, {1: _obs(5)})
    _frame(job, 7, {1: _obs(5)})
    assert _frames_of(job, 7) == [0, 7]
    if mutate:
        mutate(job)
    _frame(job, 14, third_obs, events=events)
    return _frames_of(job, 7)


@pytest.mark.parametrize("ev_type", ["merged", "appearance_relinked", "newly_created"])
def test_flush_on_fused_id_event(ev_type):
    job = _mf_job([_ident(7, 1)])
    assert _two_frames_then(job, {1: _obs(5, ev_type=ev_type)}) == [14]


@pytest.mark.parametrize("event", ["slot_seeded", "slot_expired",
                                   "kinematic_reject", "occlusion_rebind_held"])
def test_flush_on_tracker_binding_event(event):
    job = _mf_job([_ident(7, 1)])
    frames = _two_frames_then(job, {1: _obs(5)},
                              events=[{"event": event, "slot_id": 1, "track_id": 5}])
    assert frames == [14]
    assert job._crop_buffer.stats()["flushes"][event] == 1


def test_slot_rebound_of_the_same_id_is_continuation_not_a_flush():
    """FieldTracker emits slot_rebound on EVERY frame a known fused id
    sticky-rebinds to its own slot; only a change of id is a new object."""
    job = _mf_job([_ident(7, 1)])
    ev = [{"event": "slot_rebound", "slot_id": 1, "track_id": 5}]
    _frame(job, 0, {1: _obs(5)}, events=ev)
    _frame(job, 7, {1: _obs(5)}, events=ev)
    _frame(job, 14, {1: _obs(5)}, events=ev)
    assert _frames_of(job, 7) == [0, 7, 14]
    ev2 = [{"event": "slot_rebound", "slot_id": 1, "track_id": 6}]
    _frame(job, 21, {1: _obs(6)}, events=ev2)
    assert _frames_of(job, 7) == [21]
    assert job._crop_buffer.stats()["flushes"] == {"slot_track_change": 1}


def test_binding_event_on_another_slot_does_not_flush():
    job = _mf_job([_ident(7, 1)])
    frames = _two_frames_then(job, {1: _obs(5)},
                              events=[{"event": "slot_rebound", "slot_id": 4, "track_id": 9}])
    assert frames == [0, 7, 14]


def test_flush_on_silent_slot_track_id_rewire():
    """tracker.py's distance match can rewire a slot to a different fused
    track id without any binding event (_id_to_slot overwrite)."""
    job = _mf_job([_ident(7, 1)])
    assert _two_frames_then(job, {1: _obs(99)}) == [14]
    assert job._crop_buffer.stats()["flushes"]["slot_track_change"] == 1


def test_flush_on_identity_slot_rebinding():
    job = _mf_job([_ident(7, 1)])

    def _move(j):
        j.identity_registry.identities[7].active_slot = 2
    frames = _two_frames_then(job, {2: _obs(5)}, mutate=_move)
    assert frames == [14]
    assert job._crop_buffer.stats()["flushes"]["rebind"] == 1


def test_flush_on_reattachment_after_unbound_frame():
    job = _mf_job([_ident(7, 1)])
    _frame(job, 0, {1: _obs(5)})
    job.identity_registry.identities[7].active_slot = None
    _frame(job, 7, {})
    job.identity_registry.identities[7].active_slot = 1
    _frame(job, 14, {1: _obs(5)})
    assert _frames_of(job, 7) == [14]


def test_flush_on_immediate_read_trigger_rising_edge():
    job = _mf_job([_ident(7, 1)])

    def _trigger(j):
        j.identity_registry.identities[7].needs_immediate_read = True
    assert _two_frames_then(job, {1: _obs(5)}, mutate=_trigger) == [14]
    # The flag staying set (request still pending) is not a new trigger.
    _frame(job, 21, {1: _obs(5)})
    assert _frames_of(job, 7) == [14, 21]


def test_flush_on_immediate_read_queue_stamp_while_flag_already_set():
    job = _mf_job([_ident(7, 1, needs_immediate_read=True)])

    def _requeue(j):
        j.identity_registry.identities[7].last_immediate_queue_time = 0.2
    assert _two_frames_then(job, {1: _obs(5)}, mutate=_requeue) == [14]


def test_flush_on_identity_eviction():
    job = _mf_job([_ident(7, 1), _ident(8, 2)])
    _frame(job, 0, {1: _obs(5), 2: _obs(6)})
    del job.identity_registry.identities[8]
    _frame(job, 7, {1: _obs(5)})
    assert job._crop_buffer.entries(8) == []
    assert job._crop_buffer.stats()["flushes"]["evicted"] == 1


def test_flush_on_field_position_jump():
    job = _mf_job([_ident(7, 1)])
    stale = job.config.vote_stale_max_m
    assert _two_frames_then(job, {1: _obs(5, pos=(4.0 + stale + 0.5, 3.0))}) == [14]
    assert job._crop_buffer.stats()["flushes"]["jump"] == 1


def test_explicit_flush_helper_is_a_noop_without_buffer():
    job = _mf_job([_ident(7, 1)], multiframe=False)
    job._mf_flush(7, "contradiction")          # must not raise
    assert getattr(job, "_crop_buffer", None) is None


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #

def test_per_frame_push_cost_18_crops_under_10ms():
    """6 robots x 3 views per processed frame: band crop + legibility + copy +
    push + flush bookkeeping must stay a few ms (budget ~4 ms; assert < 10)."""
    idents = [_ident(i, i) for i in range(6)]
    job = _mf_job(idents)
    boxes = [(100 + 250 * i, 300, 100 + 250 * i + 140, 420) for i in range(6)]

    def _fused():
        out = {}
        for i in range(6):
            ev = [("main", _lazy(VIEW, boxes[i]), 1.0, (4.0, 3.0)),
                  ("corner_left", _lazy(CORNER, (50 + 120 * i, 100, 50 + 120 * i + 110, 260)),
                   2.5, (4.0, 3.0)),
                  ("corner_right", _lazy(CORNER, (60 + 120 * i, 90, 60 + 120 * i + 110, 250)),
                   2.5, (4.0, 3.0))]
            out[i] = types.SimpleNamespace(
                evidence=ev, track_id=100 + i,
                detection=types.SimpleNamespace(pos=(1.0 + i, 3.0)),
                relink_evidence=types.SimpleNamespace(type="continued"))
        return out

    for f in range(0, 7 * 30, 7):          # warm up + fill the window
        _frame(job, f, _fused())
    times = []
    for k in range(15):
        obs = _fused()
        job.field_tracker.last_det_to_slot = {i: s for i, s in enumerate(obs)}
        job.field_tracker.binding_events = []
        t0 = time.perf_counter()
        job._clef_mf_push(list(obs.values()), 7 * (30 + k))
        times.append(time.perf_counter() - t0)
    med_ms = sorted(times)[len(times) // 2] * 1000
    print(f"\n  per-frame crop-buffer push (18 crops): median {med_ms:.2f} ms")
    assert sum(len(job._crop_buffer.entries(i)) for i in range(6)) > 0
    assert med_ms < 10.0


# --------------------------------------------------------------------------- #
# Integration: real fusion -> fused-id registry -> tracker -> identity sync
# --------------------------------------------------------------------------- #

def test_push_with_real_fusion_tracker_and_identity_sync():
    """Drive the REAL per-frame structures the hook reads (fuse_detections,
    FusedTrackRegistry relink evidence, FieldTracker.last_det_to_slot /
    binding_events, the identity sync) and check the buffer follows each
    identity, then flushes only the robot whose SAM object was re-seeded."""
    import asyncio
    from web.field.fusion import fuse_detections, merge_same_id_observations
    from web.field.fused_track_id import FusedTrackRegistry
    from web.field.identity import IdentityRegistry
    from web.field.identity_ambiguity import CoOcclusionTracker
    from web.field.identity_confidence import ConfidenceConfig
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    from web.field.tracker import FieldDetection, FieldTracker

    job = app_module.InferenceJob.__new__(app_module.InferenceJob)
    job.config = JobConfig({"team_reader": "clef", "clef_multiframe": True})
    job.team_reader = "clef"
    job.video_fps = FPS
    job.frame_skip = 7
    job._mock_db = True
    job.identity_registry = IdentityRegistry()
    job.field_tracker = FieldTracker(16.5)
    job.fused_registry = FusedTrackRegistry()
    job.co_occlusion_tracker = CoOcclusionTracker()
    job.identity_diagnostics = IdentityDiagnosticRecorder()
    job.confidence_config = ConfidenceConfig()
    job._last_bindings = {}
    job._identity_log = []
    job.field_model = None
    job.field_length = 16.5
    job._current_frame = 0
    job._occ = {}
    job._occ_by_cam = {}
    job._camera_centers_xy = {}

    def step(f, a_obj=1):
        dets = []
        for obj, color, x, box in ((a_obj, "blue", 3.0 + 0.02 * f, (300, 300, 440, 420)),
                                   (2, "red", 12.0 - 0.02 * f, (1200, 300, 1340, 420))):
            d = FieldDetection(pos=(x, 4.0), color=color, box=box, track_id=obj)
            dets.append((d, _lazy(VIEW, box)))
        fused = fuse_detections({"main": dets}, weights={"main": 1.0})
        ev = job.fused_registry.assign(fused)
        fused, ev = merge_same_id_observations(fused, ev)
        outs = job.field_tracker.update([o.detection for o in fused], {}, 7 / FPS,
                                        frame_num=f)
        asyncio.run(job._sync_identity_registry_frame(fused, ev, outs, log_frame_num=f))
        job._clef_mf_push(fused, f)
        job._current_frame = f

    for f in range(0, 70, 7):
        step(f)
    ids = {i.alliance: i.identity_id for i in job.identity_registry.identities.values()}
    assert set(ids) == {"blue", "red"}
    buf = job._crop_buffer
    assert len(buf.entries(ids["blue"])) == 10 and len(buf.entries(ids["red"])) == 10
    assert {e.slot for e in buf.entries(ids["blue"])} == {
        job.identity_registry.identities[ids["blue"]].active_slot}

    # The blue robot's SAM object is re-seeded under a new id at the same spot:
    # a newly_created fused id the tracker silently re-binds to the same slot.
    # (The red robot's per-frame "slot_rebound" sticky continuation above must
    # not have flushed anything.)
    assert buf.stats()["flushes"] == {}
    step(70, a_obj=9)
    assert [e.frame for e in buf.entries(ids["blue"])] == [70]
    assert len(buf.entries(ids["red"])) == 11            # untouched
    assert buf.stats()["flushes"] == {"fused_newly_created": 1}

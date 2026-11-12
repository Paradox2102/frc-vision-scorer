"""Loop wiring of the identity churn fixes in _sync_identity_registry_frame:
tracker binding disruptions -> IdentityObservation.slot_disrupted, evicted
identities' tracks retired, rejected matches written as identity_reject
diagnostic rows."""
import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from web.app import InferenceJob, RobotTrack
from web.diagnostic import DiagnosticRecorder
from web.field.identity import IdentityRegistry
from web.field.identity_ambiguity import CoOcclusionTracker
from web.field.identity_confidence import ConfidenceConfig
from web.field.identity_diagnostics import IdentityDiagnosticRecorder


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


A = _e(1, 0, 0)
B = _e(0, 1, 0)


class _Capture:
    def __init__(self):
        self.rejects = []

    def record_identity_reject(self, **kw):
        self.rejects.append(kw)

    def __getattr__(self, name):
        if name.startswith("record_"):
            return lambda *a, **k: None
        raise AttributeError(name)


def _job(registry=None):
    job = InferenceJob.__new__(InferenceJob)
    job.identity_registry = registry or IdentityRegistry()
    job.confidence_config = ConfidenceConfig()
    job.co_occlusion_tracker = CoOcclusionTracker(re_emergence_window=30)
    job.identity_diagnostics = IdentityDiagnosticRecorder()
    job._current_frame = 0
    job.video_fps = 60.0
    job._last_bindings = {}
    job._mock_db = True
    job.job_id = "test_sync_telemetry"
    job.scoring_events = []
    job.robot_tracks = {}
    job.diagnostic = _Capture()
    job.field_tracker = MagicMock()
    job.field_tracker.binding_events = []
    job.field_tracker.assignment_evidence = {}
    job.field_tracker._emit.return_value = []

    async def _bind(slot_id, ident_id):
        pass
    job._bind_detection_identity = _bind
    return job


def _frame(job, frame, slot, pos, emb, track_id, events=()):
    job._current_frame = frame
    job.field_tracker.binding_events = list(events)
    job.field_tracker.assignment_evidence = {slot: {"detection_idx": 0}}
    fused = [SimpleNamespace(track_id=track_id,
                             detection=SimpleNamespace(pos=pos, embedding=emb, box=None))]
    outs = [SimpleNamespace(slot_id=slot, pos=pos, alliance="blue")]
    asyncio.run(job._sync_identity_registry_frame(fused, [], outs, log_frame_num=frame))


def test_record_identity_reject_row(tmp_path):
    path = tmp_path / "d.jsonl"
    rec = DiagnosticRecorder(path)
    rec.record_identity_reject(
        frame=4064, slot_id=3, candidate_id=337, max_sim=0.41234567,
        distance_m=0.2345678, cost=1.05, reason="appearance_veto",
        continuity_eligible=True, slot_disrupted=False, outcome="born")
    rec.close()
    row = json.loads(path.read_text().splitlines()[0])
    assert row["type"] == "identity_reject" and row["frame"] == 4064
    assert row["slot_id"] == 3 and row["candidate_id"] == 337
    assert row["reason"] == "appearance_veto" and row["outcome"] == "born"
    assert abs(row["max_sim"] - 0.4123) < 1e-4 and abs(row["distance_m"] - 0.235) < 1e-3
    assert row["continuity_eligible"] is True and row["slot_disrupted"] is False


def test_undisturbed_slot_veto_is_reported_as_continuity_eligible():
    job = _job()
    _frame(job, 8, 0, (2.0, 2.0), A, track_id=7)
    _frame(job, 16, 0, (2.2, 2.0), B, track_id=7)
    [row] = job.diagnostic.rejects
    assert row["frame"] == 16 and row["reason"] == "appearance_veto"
    assert row["slot_disrupted"] is False and row["continuity_eligible"] is True


def test_binding_event_marks_slot_disrupted():
    job = _job()
    _frame(job, 8, 0, (2.0, 2.0), A, track_id=7)
    _frame(job, 16, 0, (2.2, 2.0), B, track_id=7,
           events=[{"event": "kinematic_reject", "slot_id": 0, "track_id": 7}])
    [row] = job.diagnostic.rejects
    assert row["slot_disrupted"] is True and row["continuity_eligible"] is False


def test_feeding_track_id_change_marks_slot_disrupted():
    """A slot re-bound to a different track id (rebound/distance match) is a
    potential swap even without a tracker event."""
    job = _job()
    _frame(job, 8, 0, (2.0, 2.0), A, track_id=7)
    _frame(job, 16, 0, (2.2, 2.0), B, track_id=8,
           events=[{"event": "slot_rebound", "slot_id": 0, "track_id": 8}])
    [row] = job.diagnostic.rejects
    assert row["slot_disrupted"] is True


def test_steady_sticky_rebound_is_not_a_disruption():
    """slot_rebound fires EVERY frame a known track id re-binds to its slot —
    the steady state, not a swap."""
    job = _job(IdentityRegistry(continuity=True))
    _frame(job, 8, 0, (2.0, 2.0), A, track_id=7,
           events=[{"event": "slot_seeded", "slot_id": 0, "track_id": 7}])
    first = job.identity_registry.get_active_identities()[0].identity_id
    _frame(job, 16, 0, (2.2, 2.0), B, track_id=7,
           events=[{"event": "slot_rebound", "slot_id": 0, "track_id": 7}])
    assert job.identity_registry.get_active_identities()[0].identity_id == first
    assert job.diagnostic.rejects == []


def test_evicted_identity_track_is_retired():
    job = _job()
    reg = job.identity_registry
    embs = [_e(1, 0, 0), _e(0, 1, 0), _e(0, 0, 1)]
    for i in range(3):
        _frame(job, 8 * (2 * i + 1), i, (10.0 * i, 1.0), embs[i], track_id=i)
        job._current_frame = 8 * (2 * i + 2)
        job.field_tracker.binding_events = []
        job.field_tracker.assignment_evidence = {}
        asyncio.run(job._sync_identity_registry_frame([], [], [], log_frame_num=job._current_frame))
    oldest = min(reg.identities)
    job.robot_tracks[oldest] = RobotTrack(
        robot_id=oldest, alliance="blue", team_number="4786",
        last_position=(0, 0), positions_history=deque(maxlen=30))
    job.robot_tracks[oldest].team_number_confirmed = True
    _frame(job, 100, 2, (40.0, 1.0), _e(1, 1, 1), track_id=99)
    assert oldest not in reg.identities
    assert job.robot_tracks[oldest].team_number is None
    assert job.robot_tracks[oldest].team_number_confirmed is False


def test_identity_churn_summary_uses_final_slot_holders():
    from web.field.identity import RobotIdentity
    job = _job()
    reg = job.identity_registry
    reg.stats.update({"births": 3, "births_appearance_veto": 2, "births_no_candidate": 1,
                      "evictions": 1})
    reg.identities = {
        5: RobotIdentity(5, "blue", active_slot=0, born_frame=600, last_seen_frame=1200),
        7: RobotIdentity(7, "red", active_slot=4, born_frame=1140, last_seen_frame=1200),
    }
    job._identity_log = [{"frame_num": 0}, {"frame_num": 3600}]
    line, payload = job._identity_churn_summary()
    assert line.startswith("[identity] births 3 (appearance_veto 2")
    assert "3.0/min" in line                      # 3 births over 60 s
    assert payload["final_ages_s"]["B1"] == 10.0  # (1200-600)/60
    assert payload["final_ages_s"]["R2"] == 1.0
    assert payload["evictions"] == 1


def _cam_frame(job, frame, pos, emb, cam):
    job._current_frame = frame
    job.field_tracker.binding_events = []
    job.field_tracker.assignment_evidence = {0: {"detection_idx": 0}}
    fused = [SimpleNamespace(track_id=7, detection=SimpleNamespace(
        pos=pos, embedding=emb, box=None, cam_embeddings={cam: emb}))]
    outs = [SimpleNamespace(slot_id=0, pos=pos, alliance="blue")]
    asyncio.run(job._sync_identity_registry_frame(fused, [], outs, log_frame_num=frame))


def test_per_cam_appearance_reaches_registry_through_the_loop():
    """identity_per_cam_appearance: the fused detection's cam_embeddings feed
    the IdentityObservation, so a corner-only view of a main-built identity
    is neutral (no veto row); the legacy registry vetoes the same input."""
    for per_cam, expect_rejects in ((False, 1), (True, 0)):
        job = _job(IdentityRegistry(per_cam_appearance=per_cam))
        _cam_frame(job, 8, (2.0, 2.0), A, "main")
        _cam_frame(job, 16, (2.2, 2.0), B, "corner_left")
        assert len(job.diagnostic.rejects) == expect_rejects
        assert len(job.identity_registry.identities) == 1 + expect_rejects

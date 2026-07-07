import pytest
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np

from web.app import InferenceJob, ScoringEvent
from web.field.identity import IdentityRegistry, IdentityObservation, RobotIdentity
from web.field.identity_confidence import ConfidenceConfig
from web.field.identity_ambiguity import CoOcclusionTracker
from web.field.identity_diagnostics import IdentityDiagnosticRecorder
from web.job_config import JobConfig

def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)

@pytest.fixture
def base_job():
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig()
    job._camera_centers_xy = {}
    job.identity_registry = IdentityRegistry()
    job.confidence_config = ConfidenceConfig()
    job.co_occlusion_tracker = CoOcclusionTracker(re_emergence_window=30)
    job.identity_diagnostics = IdentityDiagnosticRecorder()
    job._current_frame = 0
    job.video_fps = 10.0
    job._last_bindings = {}
    job._mock_db = True
    job.job_id = "test_job"
    job.scoring_events = []
    job.deferred_attributions = []
    job.scored_balls = []
    job.match_teams = ["254", "971"]
    job.match_blue_teams = ["254", "971"]
    job.match_red_teams = []
    job.hubs = []
    job.vlm_status = {}
    job._occ = {}
    job.diagnostic = MagicMock()
    job.gallery = None
    
    # Mock field tracker
    job.field_tracker = MagicMock()
    job.field_tracker.binding_events = []
    job.field_tracker.assignment_evidence = {}
    job.field_tracker.last_det_to_slot = {0: 0, 1: 1}
    job.field_tracker._emit.return_value = []
    
    async def mock_bind(slot_id, ident_id):
        pass
    job._bind_detection_identity = mock_bind
    job.robot_tracks = {}
    job._slot_crops = {}
    def mock_get_or_create_track(slot_id):
        ident = job.identity_registry.identities.get(slot_id)
        identity_id = ident.identity_id if ident else slot_id
        alliance = ident.alliance if ident else "blue"
        if identity_id not in job.robot_tracks:
            track = MagicMock()
            track.robot_id = identity_id
            track.alliance = alliance
            track.team_number = ident.team_number if ident else None
            track.team_number_confirmed = ident.is_confirmed if ident else False
            track.scores = 0
            track.team_ocr_votes = {}
            track.consecutive_tracker = None
            job.robot_tracks[identity_id] = track
        return job.robot_tracks[identity_id]
    job._get_or_create_slot_track = mock_get_or_create_track
    job._slot_label = lambda slot: f"Robot {slot}"
    job._slot_source = {}
    job._clear_team_from_other_robots = lambda slot, team: None
    
    return job

def test_separable_evidence_binds_without_vlm(base_job):
    # Two blue robots 0 and 1
    # Frame 1: Robots 0 and 1 are active
    obs1 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    obs2 = IdentityObservation(slot_id=1, pos=(2.0, 2.0), alliance="blue", embedding=_e(0, 1, 0))
    base_job.identity_registry.update_frame(1, [obs1, obs2])
    
    # Frame 2: Both go occluded
    base_job._current_frame = 2
    base_job.co_occlusion_tracker.record_occlusion(2, 0, "blue")
    base_job.co_occlusion_tracker.record_occlusion(2, 1, "blue")
    base_job.identity_registry.update_frame(2, [])
    
    # Frame 3: Re-emerge with very separable appearance evidence
    # (High similarity margin)
    obs1_re = IdentityObservation(slot_id=0, pos=(1.05, 1.05), alliance="blue", embedding=_e(1, 0.1, 0), relink_margin=0.8)
    obs2_re = IdentityObservation(slot_id=1, pos=(2.05, 2.05), alliance="blue", embedding=_e(0.1, 1, 0), relink_margin=0.8)
    
    event = base_job.co_occlusion_tracker.handle_reemergence(3, 0, 0, margin=0.8)
    assert event is None # High margin -> not ambiguous!
    
    base_job.identity_registry.update_frame(3, [obs1_re, obs2_re])
    # Verify no ambiguous tags
    assert not any(i.is_ambiguous for i in base_job.identity_registry.identities.values())

def test_inseparable_evidence_triggers_vlm_and_resolves(base_job):
    # Frame 1: Robots active
    obs1 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    obs2 = IdentityObservation(slot_id=1, pos=(2.0, 2.0), alliance="blue", embedding=_e(0, 1, 0))
    base_job.identity_registry.update_frame(1, [obs1, obs2])
    
    # Set team numbers beforehand so VLM can confirm who is who
    base_job.identity_registry.identities[0].team_number = "254"
    base_job.identity_registry.identities[1].team_number = "971"
    
    # Frame 2: Occluded
    base_job._current_frame = 2
    base_job.co_occlusion_tracker.record_occlusion(2, 0, "blue")
    base_job.co_occlusion_tracker.record_occlusion(2, 1, "blue")
    
    # Frame 3: Re-emerge with inseparable evidence (low margin = 0.05)
    base_job._current_frame = 3
    # We run _sync_identity_registry_frame with mock observations
    ev_list = [MagicMock(margin=0.05, type="reappeared"), MagicMock(margin=0.05, type="reappeared")]
    
    # Mock assignments
    base_job.field_tracker.assignment_evidence = {
        0: {"detection_idx": 0, "margin": 0.05, "type": "reappeared"},
        1: {"detection_idx": 1, "margin": 0.05, "type": "reappeared"}
    }
    
    fused_obs1 = MagicMock(track_id=1, detection=MagicMock(pos=(1.0, 1.0), embedding=_e(1, 0, 0)))
    fused_obs2 = MagicMock(track_id=2, detection=MagicMock(pos=(2.0, 2.0), embedding=_e(0, 1, 0)))
    
    asyncio.run(base_job._sync_identity_registry_frame([fused_obs1, fused_obs2], ev_list, [
        MagicMock(slot_id=0, pos=(1.0, 1.0), alliance="blue"),
        MagicMock(slot_id=1, pos=(2.0, 2.0), alliance="blue")
    ]))
    
    # Verify both identities are marked ambiguous and need immediate VLM reads
    ident0 = base_job.identity_registry.identities[0]
    ident1 = base_job.identity_registry.identities[1]
    assert ident0.is_ambiguous is True
    assert ident1.is_ambiguous is True
    assert ident0.needs_immediate_read is True
    assert ident1.needs_immediate_read is True
    
    # Populate track maps
    base_job._get_or_create_slot_track(0)
    base_job._get_or_create_slot_track(1)
    
    # Defer scoring attribution during ambiguity
    base_job.attribute_score("blue", frame_num=3, timestamp=0.3)
    assert len(base_job.deferred_attributions) == 1
    assert len(base_job.scoring_events) == 0
    
    # Frame 4: VLM runs and resolves the permutation
    # Simulate VLM results
    mock_qwen_results = [
        {"team_number": "254", "confidence": 0.95},
        {"team_number": "971", "confidence": 0.95}
    ]
    
    # We mock _vision_read_bumpers
    with patch("web.app._vision_read_bumpers", return_value=mock_qwen_results):
        # We need to construct the fused obs evidence to match
        fused_obs1.evidence = [("main", "crop1_b64", 1.0, (1.0, 1.0))]
        fused_obs2.evidence = [("main", "crop2_b64", 1.0, (2.0, 2.0))]

        # Trigger read_fused_bumper_numbers with priorities
        priorities = {0: 10.0, 1: 10.0}
        asyncio.run(base_job.read_fused_bumper_numbers([fused_obs1, fused_obs2], priorities))
        
    # Verify co-occlusion event is resolved and ambiguity cleared
    assert not base_job.co_occlusion_tracker.get_active_events()
    assert ident0.is_ambiguous is False
    assert ident1.is_ambiguous is False
    
    # Verify deferred scoring event is retroactively attributed
    assert len(base_job.deferred_attributions) == 0
    assert len(base_job.scoring_events) == 1
    assert base_job.scoring_events[0].team_number == "254"

def test_unreadable_crops_remains_ambiguous(base_job):
    # Same as above but crops are unreadable
    obs1 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    obs2 = IdentityObservation(slot_id=1, pos=(2.0, 2.0), alliance="blue", embedding=_e(0, 1, 0))
    base_job.identity_registry.update_frame(1, [obs1, obs2])
    
    base_job._current_frame = 2
    base_job.co_occlusion_tracker.record_occlusion(2, 0, "blue")
    base_job.co_occlusion_tracker.record_occlusion(2, 1, "blue")
    
    base_job._current_frame = 3
    ev_list = [MagicMock(margin=0.05, type="reappeared"), MagicMock(margin=0.05, type="reappeared")]
    base_job.field_tracker.assignment_evidence = {
        0: {"detection_idx": 0, "margin": 0.05, "type": "reappeared"},
        1: {"detection_idx": 1, "margin": 0.05, "type": "reappeared"}
    }
    
    fused_obs1 = MagicMock(track_id=1, detection=MagicMock(pos=(1.0, 1.0), embedding=_e(1, 0, 0)))
    fused_obs2 = MagicMock(track_id=2, detection=MagicMock(pos=(2.0, 2.0), embedding=_e(0, 1, 0)))
    
    asyncio.run(base_job._sync_identity_registry_frame([fused_obs1, fused_obs2], ev_list, [
        MagicMock(slot_id=0, pos=(1.0, 1.0), alliance="blue"),
        MagicMock(slot_id=1, pos=(2.0, 2.0), alliance="blue")
    ]))
    
    ident0 = base_job.identity_registry.identities[0]
    ident1 = base_job.identity_registry.identities[1]
    
    # Populate track maps
    base_job._get_or_create_slot_track(0)
    base_job._get_or_create_slot_track(1)
    
    # VLM fails to read (unreadable crop returns error or no team)
    mock_qwen_results = [
        {"error": "blurry", "confidence": 0.0},
        {"error": "blurry", "confidence": 0.0}
    ]
    
    with patch("web.app._vision_read_bumpers", return_value=mock_qwen_results):
        fused_obs1.evidence = [("main", "crop1_b64", 1.0, (1.0, 1.0))]
        fused_obs2.evidence = [("main", "crop2_b64", 1.0, (2.0, 2.0))]
        priorities = {0: 10.0, 1: 10.0}
        asyncio.run(base_job.read_fused_bumper_numbers([fused_obs1, fused_obs2], priorities))
        
    # Verify still active/unresolved
    assert len(base_job.co_occlusion_tracker.get_active_events()) == 1
    # Bindings are not swapped/resolved, remain ambiguous
    assert ident0.is_ambiguous is True
    assert ident1.is_ambiguous is True


def test_confirmation_uses_identity_id_when_slot_differs(base_job):
    ident = base_job.identity_registry.identities.setdefault(
        4, RobotIdentity(identity_id=4, alliance="red", active_slot=3)
    )
    base_job.match_teams = ["2199"]
    base_job.match_blue_teams = []
    base_job.match_red_teams = ["2199"]
    base_job.field_tracker.last_det_to_slot = {0: 3}
    robot = base_job._get_or_create_slot_track(3)
    robot.robot_id = 4
    robot.consecutive_tracker = MagicMock()
    robot.consecutive_tracker.update.return_value = ("2199", True)
    fused = MagicMock()
    fused.evidence = [("corner_right", "crop", 2.5, (4.0, 3.0))]

    with patch("web.app._vision_read_bumpers", return_value=[
        {"team_number": "2199", "confidence": 0.95}
    ]):
        asyncio.run(base_job.read_fused_bumper_numbers([fused], {4: 10.0}))

    assert robot.team_ocr_votes["2199"] == 2.5
    assert robot.team_number == "2199"
    assert ident.team_number == "2199"

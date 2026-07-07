import pytest
import asyncio
import numpy as np
from dataclasses import dataclass
from collections import deque
from web.field.identity import IdentityRegistry, IdentityObservation, RobotIdentity
from web.field.tracker import FieldTracker
import web.database as database

def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)

@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database

def test_long_occlusion_and_rebind():
    # Verify slot 0 -> occluded -> expires -> slot 1 -> re-binds to original identity
    reg = IdentityRegistry(association_threshold=0.75, app_floor=0.55)
    
    # Frame 1: slot 0 (blue) appears with embedding A
    obs1 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    reg.update_frame(frame_idx=1, observations=[obs1])
    
    active = reg.get_active_identities()
    assert len(active) == 1
    ident_id = active[0].identity_id
    assert active[0].active_slot == 0
    
    # Frame 100: slot 0 expired/inactive. We mock the slot detachment.
    # Registry updates with empty observations (coasting/occluded)
    reg.update_frame(frame_idx=100, observations=[])
    inactive = reg.get_inactive_identities()
    assert len(inactive) == 1
    assert inactive[0].identity_id == ident_id
    assert inactive[0].active_slot is None
    
    # Frame 101: Robot re-appears on slot 1, with a similar embedding
    obs2 = IdentityObservation(slot_id=1, pos=(1.1, 0.9), alliance="blue", embedding=_e(1.0, 0.1, 0))
    reg.update_frame(frame_idx=101, observations=[obs2])
    
    active = reg.get_active_identities()
    assert len(active) == 1
    assert active[0].identity_id == ident_id
    assert active[0].active_slot == 1


def test_database_identity_and_binding_roundtrips(db):
    job_id = "test_job_123"
    
    # 1. Insert/upsert identity
    asyncio.run(database.upsert_match_robot_identity({
        "job_id": job_id,
        "identity_id": 5,
        "slot": 2,
        "slot_label": "B3",
        "alliance": "blue",
        "team_number": "254",
        "confirmed": 1,
        "source": "ocr",
        "confidence": 0.95
    }))
    
    identities = asyncio.run(database.get_match_identities(job_id))
    assert len(identities) == 1
    assert identities[0]["identity_id"] == 5
    assert identities[0]["slot"] == 2
    assert identities[0]["team_number"] == "254"
    
    # 2. Insert binding event
    asyncio.run(database.insert_match_robot_binding({
        "job_id": job_id,
        "identity_id": 5,
        "slot": 2,
        "start_frame": 10,
        "end_frame": 100
    }))
    
    bindings = asyncio.run(database.get_match_bindings(job_id))
    assert len(bindings) == 1
    assert bindings[0]["identity_id"] == 5
    assert bindings[0]["slot"] == 2
    assert bindings[0]["start_frame"] == 10
    assert bindings[0]["end_frame"] == 100


def test_database_migrates_legacy_slot_keyed_identity_tables(tmp_path, monkeypatch):
    legacy_path = tmp_path / "legacy.db"
    monkeypatch.setattr(database, "DB_PATH", legacy_path)

    async def create_legacy_db():
        import aiosqlite

        async with aiosqlite.connect(legacy_path) as conn:
            await conn.executescript("""
                CREATE TABLE match_robot_identity (
                    job_id TEXT NOT NULL,
                    slot INTEGER NOT NULL,
                    slot_label TEXT,
                    alliance TEXT,
                    team_number TEXT,
                    confirmed INTEGER DEFAULT 0,
                    source TEXT,
                    confidence REAL,
                    created_at TEXT,
                    PRIMARY KEY (job_id, slot)
                );
                INSERT INTO match_robot_identity
                    (job_id, slot, slot_label, alliance, team_number, confirmed)
                VALUES ('legacy-job', 2, 'B3', 'blue', '254', 1);
                CREATE TABLE match_robot_crop (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    slot INTEGER NOT NULL,
                    legibility REAL,
                    embedding BLOB,
                    image BLOB
                );
                CREATE INDEX idx_match_crop ON match_robot_crop(job_id, slot);
                INSERT INTO match_robot_crop (job_id, slot, legibility)
                VALUES ('legacy-job', 2, 0.9);
            """)
            await conn.commit()

    asyncio.run(create_legacy_db())
    asyncio.run(database.init_db())

    identities = asyncio.run(database.get_match_identities("legacy-job"))
    crops = asyncio.run(database.get_match_crops("legacy-job", 2))
    assert identities[0]["identity_id"] == 2
    assert identities[0]["slot"] == 2
    assert crops[0]["identity_id"] == 2

def test_confidence_wiring_and_decisions():
    from unittest.mock import MagicMock
    from web.app import InferenceJob
    from web.field.identity_confidence import ConfidenceConfig
    from web.field.identity import IdentityRegistry, IdentityObservation
    from web.field.identity_ambiguity import CoOcclusionTracker
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    
    job = InferenceJob.__new__(InferenceJob)
    job.identity_registry = IdentityRegistry()
    job.confidence_config = ConfidenceConfig(decay_rate=0.01, observation_rise=0.02)
    job.co_occlusion_tracker = CoOcclusionTracker(re_emergence_window=30)
    job.identity_diagnostics = IdentityDiagnosticRecorder()
    job._current_frame = 0
    job.video_fps = 10.0
    job._last_bindings = {}
    job._mock_db = True
    job.job_id = "test_job"
    job.scoring_events = []
    
    # Mock field_tracker and its binding_events / assignment_evidence
    job.field_tracker = MagicMock()
    job.field_tracker.binding_events = []
    job.field_tracker.assignment_evidence = {}
    job.field_tracker._emit.return_value = []
    
    # Mock _bind_detection_identity as an async function
    async def mock_bind(slot_id, ident_id):
        pass
    job._bind_detection_identity = mock_bind
    
    # Step 1: Create a new identity with confidence 0.5
    obs = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    asyncio.run(job._sync_identity_registry_frame([MagicMock(track_id=1, detection=MagicMock(pos=(1.0, 1.0), embedding=_e(1, 0, 0)))], [], [MagicMock(slot_id=0, pos=(1.0, 1.0), alliance="blue")]))
    
    idents = list(job.identity_registry.identities.values())
    assert len(idents) == 1
    ident = idents[0]
    assert ident.confidence == pytest.approx(0.52)  # initial 0.5 + 0.02 rise
    
    # Step 2: Next frame, robot is occluded/inactive, so confidence decays
    asyncio.run(job._sync_identity_registry_frame([], [], []))
    assert ident.confidence == pytest.approx(0.51)  # 0.52 - 0.01 decay
    
    # Step 3: Test low relink margin drops confidence and queues immediate read
    # We pass an assignment evidence with a margin of 0.05.
    # Advance the clock past the escalation cooldown: step 1's re-attachment
    # queued an immediate read at t=0.0, and a second escalation within
    # immediate_read_cooldown_sec is deliberately suppressed.
    job._current_frame = 30  # 3.0s at 10fps
    job.field_tracker.assignment_evidence = {
        0: {"detection_idx": 0, "margin": 0.05, "type": "reappeared"}
    }
    ev_list = [MagicMock(margin=0.05, type="reappeared")]
    fused_obs = MagicMock(track_id=1, detection=MagicMock(pos=(1.0, 1.0), embedding=_e(1, 0, 0)))
    
    # Reset needs_immediate_read to check if it gets set
    ident.needs_immediate_read = False
    
    asyncio.run(job._sync_identity_registry_frame([fused_obs], ev_list, [MagicMock(slot_id=0, pos=(1.0, 1.0), alliance="blue")]))
    assert ident.confidence == pytest.approx(0.46)  # 0.51 - 0.05 low_margin_drop
    assert ident.needs_immediate_read is True


def test_anchor_update_floor():
    reg = IdentityRegistry(association_threshold=0.75, app_floor=0.55, app_update_floor=0.6, ema_alpha=0.5)
    
    # Initial frame: confidence is 0.5 (below 0.6, but appearance_embedding is None, so it gets initialized)
    obs1 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    reg.update_frame(frame_idx=1, observations=[obs1])
    ident = reg.identities[0]
    assert np.allclose(ident.appearance_embedding, _e(1, 0, 0))
    
    # Second frame: confidence is set below 0.6 (e.g. 0.5), we try to update with a similar embedding
    ident.confidence = 0.5
    obs2 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0.2, 0), relink_margin=0.5)
    reg.update_frame(frame_idx=2, observations=[obs2])
    # Embedding should NOT change because confidence is below 0.6
    assert np.allclose(ident.appearance_embedding, _e(1, 0, 0))
    
    # Third frame: confidence is set above 0.6 (e.g. 0.7)
    ident.confidence = 0.7
    obs3 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0.2, 0), relink_margin=0.5)
    reg.update_frame(frame_idx=3, observations=[obs3])
    # Embedding SHOULD be updated (blended with EMA)
    # blended = 0.5 * [1, 0, 0] + 0.5 * [0.9806, 0.1961, 0] = [0.9903, 0.0980, 0] -> normalized = [0.9951, 0.0985, 0]
    assert np.allclose(ident.appearance_embedding, [0.9951, 0.0985, 0], atol=1e-3)

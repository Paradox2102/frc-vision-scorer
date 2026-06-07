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

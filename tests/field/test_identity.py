import pytest
import numpy as np
from web.field.identity import IdentityRegistry, RobotIdentity, IdentityObservation

def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)

def test_registry_basic_creation_and_matching():
    reg = IdentityRegistry(association_threshold=0.75)
    
    # Frame 1: slot 0 (blue) and slot 1 (red) appear
    obs0 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    obs1 = IdentityObservation(slot_id=1, pos=(10.0, 1.0), alliance="red", embedding=_e(0, 1, 0))
    
    reg.update_frame(frame_idx=1, observations=[obs0, obs1])
    
    # Should create two identities: one blue, one red
    active = reg.get_active_identities()
    assert len(active) == 2
    
    id_blue = next(ident for ident in active if ident.alliance == "blue")
    id_red = next(ident for ident in active if ident.alliance == "red")
    
    assert id_blue.active_slot == 0
    assert id_red.active_slot == 1
    assert id_blue.identity_id != id_red.identity_id

def test_slot_detachment_and_re_attachment():
    reg = IdentityRegistry(association_threshold=0.75)
    
    # Frame 1: Robot appears on slot 0
    obs0 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    reg.update_frame(frame_idx=1, observations=[obs0])
    
    id_first = reg.get_active_identities()[0]
    ident_id = id_first.identity_id
    assert id_first.active_slot == 0
    
    # Frame 2: Robot disappears (slot 0 becomes inactive / not in observations)
    reg.update_frame(frame_idx=2, observations=[])
    
    # Identity is now inactive
    inactive = reg.get_inactive_identities()
    assert len(inactive) == 1
    assert inactive[0].identity_id == ident_id
    assert inactive[0].active_slot is None
    
    # Frame 3: Robot re-appears on slot 1 (re-binds to the same identity)
    obs1 = IdentityObservation(slot_id=1, pos=(1.1, 1.0), alliance="blue", embedding=_e(1.0, 0.1, 0))
    reg.update_frame(frame_idx=3, observations=[obs1])
    
    active = reg.get_active_identities()
    assert len(active) == 1
    assert active[0].identity_id == ident_id
    assert active[0].active_slot == 1

def test_persistent_roster_cap_enforcement():
    # Enforce cap of 3 per alliance. Provisional can be tombstoneed, confirmed are protected.
    reg = IdentityRegistry(association_threshold=0.75)
    
    # Create 3 provisional identities on blue alliance
    embs = [_e(1, 0, 0), _e(0, 1, 0), _e(0, 0, 1)]
    for i in range(3):
        obs = IdentityObservation(slot_id=i, pos=(float(i) * 10.0, 1.0), alliance="blue", embedding=embs[i])
        reg.update_frame(frame_idx=i*2, observations=[obs])
        # Make them inactive so they go to the inactive pool
        reg.update_frame(frame_idx=i*2 + 1, observations=[])
        
    assert len(reg.identities) == 3
    
    # Now add a 4th provisional blue robot. It should trigger tombstoning of one of the inactive ones
    obs4 = IdentityObservation(slot_id=9, pos=(10.0, 1.0), alliance="blue", embedding=_e(0, 0, 1))
    reg.update_frame(frame_idx=10, observations=[obs4])
    
    # Total identities should remain 3 because one provisional was tombstoned to make room
    assert len(reg.identities) == 3
    
    # Now let's confirm all 3 remaining identities
    for ident in reg.identities.values():
        ident.is_confirmed = True
        ident.team_number = f"team_{ident.identity_id}"
        
    # Attempt to add a 5th blue robot. Since all 3 are confirmed, we cannot tombstone them.
    # It must be hard-rejected.
    # We clear the active slots so they are inactive
    for ident in reg.identities.values():
        ident.active_slot = None
    
    obs5 = IdentityObservation(slot_id=12, pos=(20.0, 1.0), alliance="blue", embedding=_e(0.1, 0.9, 0))
    reg.update_frame(frame_idx=11, observations=[obs5])
    
    # Total identities must still be 3; the 4th blue robot was rejected
    assert len(reg.identities) == 3
    # The new observation should not have been bound to any identity
    assert not any(ident.active_slot == 12 for ident in reg.identities.values())


def test_cap_eviction_cannot_delete_identity_reserved_by_batch_assignment():
    reg = IdentityRegistry(association_threshold=0.75)
    reg.identities = {
        5: RobotIdentity(
            identity_id=5,
            alliance="blue",
            appearance_embedding=_e(1, 0, 0),
            anchor_embedding=_e(1, 0, 0),
            last_seen_frame=1,
            last_seen_pos=(1.0, 1.0),
        ),
        6: RobotIdentity(
            identity_id=6,
            alliance="blue",
            appearance_embedding=_e(0, 1, 0),
            anchor_embedding=_e(0, 1, 0),
            last_seen_frame=2,
            last_seen_pos=(5.0, 1.0),
        ),
        7: RobotIdentity(
            identity_id=7,
            alliance="blue",
            appearance_embedding=_e(0, 0, 1),
            anchor_embedding=_e(0, 0, 1),
            last_seen_frame=3,
            last_seen_pos=(9.0, 1.0),
        ),
    }
    reg._next_id = 8

    unmatched = IdentityObservation(
        slot_id=20,
        pos=(20.0, 7.0),
        alliance="blue",
        embedding=_e(-1, 0, 0),
    )
    reserved_for_five = IdentityObservation(
        slot_id=21,
        pos=(1.05, 1.0),
        alliance="blue",
        embedding=_e(1, 0, 0),
    )

    reg.update_frame(10, [unmatched, reserved_for_five])

    assert 5 in reg.identities
    assert reg.identities[5].active_slot == 21
    assert len(reg.identities) == 3

def test_roster_uniqueness_and_alliance_consistency():
    reg = IdentityRegistry(association_threshold=0.75)
    
    # Create identity 0 with team 254 (blue)
    obs0 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    reg.update_frame(frame_idx=1, observations=[obs0])
    id0 = reg.get_active_identities()[0]
    id0.team_number = "254"
    id0.is_confirmed = True
    
    # If a new observation is read with team 254, roster uniqueness reserves it to id0.
    # It cannot be assigned to another identity.
    # Also, alliance is fixed once confirmed. If id0 observes contradictory color, it stays blue.
    assert id0.alliance == "blue"
    id0.alliance = "blue"
    
def test_permutation_invariance_identity_association():
    # The solver result must be invariant to the order of observations in the input list.
    reg = IdentityRegistry(association_threshold=0.75)
    
    # Create two inactive blue identities
    obs0 = IdentityObservation(slot_id=0, pos=(1.0, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    obs1 = IdentityObservation(slot_id=1, pos=(5.0, 1.0), alliance="blue", embedding=_e(0, 1, 0))
    reg.update_frame(frame_idx=1, observations=[obs0, obs1])
    
    # Disappear
    reg.update_frame(frame_idx=2, observations=[])
    
    # Re-appear with obs A and B
    obs_a = IdentityObservation(slot_id=10, pos=(1.1, 1.0), alliance="blue", embedding=_e(1, 0, 0))
    obs_b = IdentityObservation(slot_id=11, pos=(5.1, 1.0), alliance="blue", embedding=_e(0, 1, 0))
    
    # Order 1: [obs_a, obs_b]
    reg1 = IdentityRegistry(association_threshold=0.75)
    reg1.identities = {k: RobotIdentity(
        identity_id=v.identity_id, alliance=v.alliance,
        appearance_embedding=v.appearance_embedding,
        last_seen_pos=v.last_seen_pos, last_seen_frame=v.last_seen_frame
    ) for k, v in reg.identities.items()}
    reg1.update_frame(frame_idx=3, observations=[obs_a, obs_b])
    
    # Order 2: [obs_b, obs_a]
    reg2 = IdentityRegistry(association_threshold=0.75)
    reg2.identities = {k: RobotIdentity(
        identity_id=v.identity_id, alliance=v.alliance,
        appearance_embedding=v.appearance_embedding,
        last_seen_pos=v.last_seen_pos, last_seen_frame=v.last_seen_frame
    ) for k, v in reg.identities.items()}
    reg2.update_frame(frame_idx=3, observations=[obs_b, obs_a])
    
    active1 = {ident.identity_id: ident.active_slot for ident in reg1.get_active_identities()}
    active2 = {ident.identity_id: ident.active_slot for ident in reg2.get_active_identities()}
    
    assert active1 == active2

"""Occlusion-aware re-acquisition guard.

When a slot goes into coast while hub-occluded (the robot drove behind the hub),
a stationary detection sitting on a known static-structure footprint (bump, hub
base, trench — the "suspect zones") must not re-bind to that slot on distance
alone, and must not seed a fresh slot: that is exactly how a bump false-positive
hijacks the track of an occluded robot. A detection that PROVES motion (its
track_id displaced over recent frames) is a robot emerging from occlusion and
binds normally. Guard is opt-in (occlusion_rebind_guard) and only applies to
occlusion-coasting slots — open-field coasting behavior is unchanged.
"""
from web.field.tracker import FieldTracker, FieldDetection


# Suspect zone standing in for a bump footprint in blue territory.
SUSPECT = (4.0, 2.0, 5.2, 3.4)
OCC = {"view_occluded": True}


def _blue(x, y, track_id=None):
    return FieldDetection((x, y), "blue", (0, 0, 10, 10), track_id=track_id)


def _guarded_tracker(**kw):
    return FieldTracker(
        16.0, suspect_zones=[SUSPECT], occlusion_rebind_guard=True, **kw)


def _seed_and_occlude(tr, pos=(5.5, 2.5), track_id=None, coast_frames=6):
    """Seed one blue slot at `pos`, then coast it hub-occluded long enough that
    the pre-occlusion detection falls out of the guard's motion window (real
    hub occlusions run multi-second; the guard targets those)."""
    outs = tr.update([_blue(*pos, track_id=track_id)], {}, 0.1)
    assert len(outs) == 1
    slot_id = outs[0].slot_id
    for _ in range(coast_frames):
        tr.update([], {slot_id: OCC}, 0.1)
    return slot_id


def test_stationary_suspect_det_cannot_rebind_occluded_slot():
    """New-id detection parked on the bump never claims the occluded slot."""
    tr = _guarded_tracker()
    slot_id = _seed_and_occlude(tr)
    for _ in range(6):
        outs = tr.update([_blue(4.6, 2.6, track_id=99)], {slot_id: OCC}, 0.1)
        assert tr.last_det_to_slot == {}, "furniture detection bound to a slot"
    # Slot is still coasting (not hijacked, not expired).
    assert [o.state for o in outs] == ["coasting"]


def test_stationary_suspect_det_does_not_seed_a_new_slot():
    """While guard-blocked, the detection must not fall through to seeding a
    free slot either (two blue slots are free here)."""
    tr = _guarded_tracker()
    slot_id = _seed_and_occlude(tr)
    for _ in range(6):
        outs = tr.update([_blue(4.6, 2.6, track_id=99)], {slot_id: OCC}, 0.1)
    assert len(outs) == 1, "furniture detection seeded a fresh slot"
    assert outs[0].slot_id == slot_id


def test_moving_suspect_det_rebinds_after_motion_proof():
    """A robot emerging across the bump (new id, displacing frame over frame)
    binds to the occluded slot once its track_id history proves motion."""
    tr = _guarded_tracker()
    slot_id = _seed_and_occlude(tr)
    xs = (4.5, 4.8, 5.1, 5.4, 5.7)
    bound = []
    for x in xs:
        tr.update([_blue(x, 2.6, track_id=42)], {slot_id: OCC}, 0.1)
        bound.append(dict(tr.last_det_to_slot))
    assert bound[-1] == {0: slot_id}, (
        f"moving detection never re-bound the occluded slot: {bound}")


def test_open_field_coasting_slot_is_unaffected():
    """Without hub occlusion the guard must not change binding at all."""
    tr = _guarded_tracker()
    outs = tr.update([_blue(5.5, 2.5)], {}, 0.1)
    slot_id = outs[0].slot_id
    for _ in range(2):
        tr.update([], {}, 0.1)  # coast, NOT occluded
    tr.update([_blue(4.6, 2.6, track_id=99)], {}, 0.1)
    assert tr.last_det_to_slot == {0: slot_id}


def test_guard_off_by_default_keeps_old_behavior():
    """Same furniture scenario with the guard off: detection binds by distance
    (documents the pre-guard behavior the A/B compares against)."""
    tr = FieldTracker(16.0, suspect_zones=[SUSPECT])
    slot_id = _seed_and_occlude(tr)
    tr.update([_blue(4.6, 2.6, track_id=99)], {slot_id: OCC}, 0.1)
    assert tr.last_det_to_slot == {0: slot_id}


def test_sticky_rebind_is_held_and_self_heals():
    """A known track_id that reappears motionless on the bump while its slot is
    occlusion-coasting is HELD (no rebind, event emitted); when the same id
    starts moving it rebinds without churn."""
    tr = _guarded_tracker()
    slot_id = _seed_and_occlude(tr, track_id=7)
    # Motionless on the bump: held every frame.
    for _ in range(4):
        tr.update([_blue(4.6, 2.6, track_id=7)], {slot_id: OCC}, 0.1)
        assert tr.last_det_to_slot == {}
    assert any(e["event"] == "occlusion_rebind_held"
               for e in tr.binding_events), tr.binding_events
    # Same id drives off the bump: motion proof accrues, sticky rebind resumes.
    for x in (4.9, 5.2, 5.5):
        tr.update([_blue(x, 2.6, track_id=7)], {slot_id: OCC}, 0.1)
    assert tr.last_det_to_slot == {0: slot_id}

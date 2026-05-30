"""FieldTracker static-zone gating: a detection inside a static-structure zone
cannot SEED a new slot, but an already-active track is updated normally even as
it moves into the zone."""
from web.field.tracker import FieldTracker, FieldDetection


ZONE = (1.0, 1.0, 3.0, 3.0)  # blue territory (x < field_length/2)


def _det(x, y, color="blue"):
    return FieldDetection((x, y), color, (0, 0, 10, 10))


def test_detection_inside_zone_does_not_seed_a_slot():
    tr = FieldTracker(16.0, zones=[ZONE])
    outs = tr.update([_det(2.0, 2.0)], {}, 0.1)
    assert outs == []
    assert tr.last_det_to_slot == {}


def test_detection_outside_zone_seeds_normally():
    tr = FieldTracker(16.0, zones=[ZONE])
    outs = tr.update([_det(5.0, 5.0)], {}, 0.1)
    assert len(outs) == 1
    assert tr.last_det_to_slot == {0: outs[0].slot_id}


def test_active_track_moving_into_zone_is_still_updated():
    tr = FieldTracker(16.0, zones=[ZONE])
    # Seed OUTSIDE the zone.
    tr.update([_det(5.0, 2.0)], {}, 0.1)
    # Drive toward the zone (within the gate of the predicted position).
    outs = tr.update([_det(3.5, 2.0)], {}, 0.1)
    assert len(outs) == 1
    # Now fully INSIDE the zone — must keep tracking, not get dropped by the gate.
    outs = tr.update([_det(2.2, 2.0)], {}, 0.1)
    assert len(outs) == 1
    assert tr.last_det_to_slot  # the in-zone detection mapped to the existing slot


def test_zones_default_none_preserves_seeding():
    tr = FieldTracker(16.0)
    outs = tr.update([_det(2.0, 2.0)], {}, 0.1)
    assert len(outs) == 1

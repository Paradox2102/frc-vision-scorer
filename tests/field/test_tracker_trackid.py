"""Tests for optional stable track_id with sticky slot binding.

Three scenarios:
(a) Sticky bind survives a closer rival.
(b) Back-compat: track_id=None (default) behaves identically to before.
(c) Stable id re-acquisition after a coasting frame.
"""
from web.field.tracker import FieldTracker, FieldDetection


def _blue(x, y, track_id=None):
    return FieldDetection((x, y), "blue", (0, 0, 10, 10), track_id=track_id)


def _red(x, y, track_id=None):
    return FieldDetection((x, y), "red", (0, 0, 10, 10), track_id=track_id)


# ---------------------------------------------------------------------------
# (a) Sticky bind survives a closer rival
# ---------------------------------------------------------------------------

def test_sticky_bind_survives_closer_rival():
    """A detection with track_id=100 that previously bound slot S keeps S even
    when a rival detection with track_id=200 is placed closer to S's predicted pos."""
    tr = FieldTracker(field_length=16.0)

    # Frame 1: seed track_id=100 at (2, 2) — blue territory
    outs1 = tr.update([_blue(2.0, 2.0, track_id=100)], {}, 0.1)
    assert len(outs1) == 1
    slot_100 = outs1[0].slot_id

    # Frame 2:
    #   track_id=100 moved to (3.0, 2.0) — 1 m away from the F1 KF prediction
    #   track_id=200 at (2.05, 2.0) — extremely close to slot_100's predicted pos
    # Without sticky, 200 would steal slot_100 by distance. With sticky, 100 must
    # keep slot_100 and 200 must land in a different (free) slot.
    dets2 = [
        _blue(3.0, 2.0, track_id=100),
        _blue(2.05, 2.0, track_id=200),
    ]
    outs2 = tr.update(dets2, {}, 0.1)

    # track_id=100 stays in slot_100
    assert tr.last_det_to_slot[0] == slot_100, (
        f"track_id=100 was stolen from slot {slot_100}; "
        f"got {tr.last_det_to_slot[0]}"
    )
    # track_id=200 gets a DIFFERENT slot
    assert tr.last_det_to_slot[1] != slot_100, (
        f"track_id=200 was incorrectly placed into slot {slot_100}"
    )


# ---------------------------------------------------------------------------
# (b) Back-compat: track_id=None path must be purely distance-based
# ---------------------------------------------------------------------------

def test_backcompat_none_trackid_uses_distance():
    """With track_id=None (default) two robots are assigned by proximity,
    matching pre-change behavior."""
    tr = FieldTracker(field_length=16.0)

    # Seed two blue robots
    outs1 = tr.update([_blue(2.0, 2.0), _blue(5.0, 5.0)], {}, 0.1)
    assert len(outs1) == 2
    # Map position → slot_id from frame 1
    slot_near = tr.last_det_to_slot[0]   # seeded by (2, 2)
    slot_far  = tr.last_det_to_slot[1]   # seeded by (5, 5)

    # Frame 2: same positions, no track_id — should match by proximity
    outs2 = tr.update([_blue(2.1, 2.1), _blue(5.1, 5.1)], {}, 0.1)
    assert len(outs2) == 2
    # (2.1, 2.1) is closest to slot_near; (5.1, 5.1) is closest to slot_far
    assert tr.last_det_to_slot[0] == slot_near
    assert tr.last_det_to_slot[1] == slot_far


def test_backcompat_anonymous_detections_unchanged():
    """Single anonymous detection: same slot assignment as before the change."""
    tr = FieldTracker(field_length=16.0)
    outs1 = tr.update([_blue(4.0, 3.0)], {}, 0.1)
    assert len(outs1) == 1
    slot_a = tr.last_det_to_slot[0]

    outs2 = tr.update([_blue(4.1, 3.0)], {}, 0.1)
    assert len(outs2) == 1
    assert tr.last_det_to_slot[0] == slot_a


# ---------------------------------------------------------------------------
# (c) Stable id re-acquisition: returns to the SAME slot after coasting
# ---------------------------------------------------------------------------

def test_trackid_reacquires_same_slot_after_coast():
    """track_id=100 binds slot S, disappears one frame (slot coasts), then
    reappears — it must return to slot S, not a different one."""
    # Use generous coast budget so the slot doesn't expire in one miss.
    tr = FieldTracker(field_length=16.0, max_coast_open=8)

    # Frame 1: seed track_id=100 at (2, 2)
    outs1 = tr.update([_blue(2.0, 2.0, track_id=100)], {}, 0.1)
    assert len(outs1) == 1
    slot_100 = outs1[0].slot_id

    # Frame 2: track_id=100 absent — slot coasts
    tr.update([], {}, 0.1)

    # Frame 3: track_id=100 reappears at a slightly different position
    outs3 = tr.update([_blue(2.3, 2.0, track_id=100)], {}, 0.1)
    assert len(outs3) == 1
    assert tr.last_det_to_slot[0] == slot_100, (
        f"Re-acquired track_id=100 went to slot {tr.last_det_to_slot[0]} "
        f"instead of original slot {slot_100}"
    )


def test_fresh_trackid_rebinds_to_coasting_slot_not_duplicate():
    """The qm8 B1<->B3 churn: one robot fragmented across two slots. A robot that
    coasts (occluded) and reappears with a NEW SAM3 track_id (mask re-seeded) must
    re-bind to its still-coasting slot, not spawn a duplicate in a free slot."""
    tr = FieldTracker(field_length=16.0, max_coast_open=8)

    # Frame 1: seed track_id=100 at (2, 2) -> slot S
    outs1 = tr.update([_blue(2.0, 2.0, track_id=100)], {}, 0.1)
    slot_S = outs1[0].slot_id

    # Frame 2: absent -> slot S coasts (not expired; budget is 8)
    tr.update([], {}, 0.1)

    # Frame 3: SAME robot reappears near S but with a FRESH track_id=200
    outs3 = tr.update([_blue(2.3, 2.0, track_id=200)], {}, 0.1)
    assert len(outs3) == 1
    assert tr.last_det_to_slot[0] == slot_S, (
        f"fresh track_id spawned slot {tr.last_det_to_slot[0]} instead of "
        f"re-binding to coasting slot {slot_S}")
    assert tr._id_to_slot[200] == slot_S   # new id now owns the original slot


def test_fresh_trackid_on_visible_robot_stays_in_slot_no_occlusion():
    """The qm8 B1<->B3 flip observed with NO obstruction between frames: a
    continuously-visible robot whose SAM3 track_id churns mid-track must keep its
    slot, not flip into a new one. (The old code seeded fresh ids before distance
    matching, so a re-ided-but-visible robot spawned a duplicate slot.)"""
    tr = FieldTracker(field_length=16.0)
    outs1 = tr.update([_blue(2.0, 2.0, track_id=100)], {}, 0.1)
    slot_S = outs1[0].slot_id
    # Next frame: SAME robot, still visible (no miss), but SAM3 gave a fresh id.
    outs2 = tr.update([_blue(2.1, 2.0, track_id=200)], {}, 0.1)
    assert len(outs2) == 1                    # not duplicated into a 2nd slot
    assert tr.last_det_to_slot[0] == slot_S   # stayed in its slot
    assert tr._id_to_slot[200] == slot_S      # new id now owns the slot


def test_fresh_trackid_far_from_coasting_still_seeds_new():
    """Do-no-harm: a genuinely new robot (fresh track_id) far from any coasting
    slot must still seed its own slot, not get vacuumed into a distant coaster."""
    tr = FieldTracker(field_length=16.0, max_coast_open=8)
    outs1 = tr.update([_blue(2.0, 2.0, track_id=100)], {}, 0.1)
    slot_S = outs1[0].slot_id
    tr.update([], {}, 0.1)  # slot S coasts near (2, 2)
    # A new blue robot appears far away (7, 6) with a fresh id -> its own slot.
    outs3 = tr.update([_blue(7.0, 6.0, track_id=300)], {}, 0.1)
    assert tr.last_det_to_slot[0] != slot_S
    assert tr._id_to_slot[300] != slot_S


def test_expiration_clears_bindings_and_emits_events():
    tr = FieldTracker(field_length=16.0, max_coast_open=2)

    # Frame 1: Seed track_id=100
    tr.update([_blue(2.0, 2.0, track_id=100)], {}, 0.1)
    slot_id = tr.tracks[0].slot_id
    assert tr._id_to_slot[100] == slot_id
    
    # Assert binding events recorded
    assert len(tr.binding_events) == 1
    assert tr.binding_events[0]["event"] == "slot_seeded"
    assert tr.binding_events[0]["track_id"] == 100
    assert tr.binding_events[0]["slot_id"] == slot_id

    # Frame 2: Miss 1
    tr.update([], {}, 0.1)
    assert tr._id_to_slot[100] == slot_id
    assert len(tr.binding_events) == 0

    # Frame 3: Miss 2
    tr.update([], {}, 0.1)
    assert tr._id_to_slot[100] == slot_id
    assert len(tr.binding_events) == 0

    # Frame 4: Miss 3 -> expires (since budget is 2, expired at missed > 2)
    tr.update([], {}, 0.1)
    
    # Should delete the binding
    assert 100 not in tr._id_to_slot
    assert len(tr.binding_events) == 1
    assert tr.binding_events[0]["event"] == "slot_expired"
    assert tr.binding_events[0]["track_id"] == 100
    assert tr.binding_events[0]["slot_id"] == slot_id


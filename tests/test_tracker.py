import numpy as np
from web.field.tracker import FieldTracker, FieldDetection


def _det(x, y, color, box=(0, 0, 10, 10)):
    return FieldDetection(pos=(x, y), color=color, box=box, score=0.9)


def _step(trk, dets, occ=None, dt=0.1):
    return trk.update(dets, occ or {}, dt)


def test_six_tracks_three_per_alliance():
    trk = FieldTracker(field_length=16.541)
    dets = [_det(2, 2, "blue"), _det(3, 5, "blue"), _det(4, 7, "blue"),
            _det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    out = _step(trk, dets)
    assert len([t for t in out if t.state == "visible"]) == 6
    assert sum(t.alliance == "blue" for t in out) == 3
    assert sum(t.alliance == "red" for t in out) == 3


def test_never_exceeds_three_per_alliance():
    trk = FieldTracker(field_length=16.541)
    dets = [_det(2 + i, 2 + i, "blue") for i in range(5)]
    out = _step(trk, dets)
    assert sum(t.alliance == "blue" and t.state == "visible" for t in out) <= 3


def test_slot_id_stable_through_crossing():
    trk = FieldTracker(field_length=16.541)
    a, b = (2.0, 3.0), (6.0, 3.0)
    for _ in range(6):
        _step(trk, [_det(*a, "blue"), _det(*b, "blue"),
                    _det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")])
        a = (a[0] + 0.6, 3.0); b = (b[0] - 0.6, 3.0)
    out = {t.slot_id: t.pos for t in trk.update(
        [_det(*a, "blue"), _det(*b, "blue")], {}, 0.1) if t.alliance == "blue"}
    assert len(out) == 2


def test_coast_when_occluded_then_recover():
    trk = FieldTracker(field_length=16.541, max_coast_open=2, max_coast_occluded=30)
    base = [_det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    seed = [_det(2.0, 3.0, "blue"), _det(5.0, 3.0, "blue"), _det(8.0, 3.0, "blue")]
    _step(trk, seed + base)
    blue0 = [t for t in trk.tracks if t.alliance == "blue"][0]
    sid = blue0.slot_id
    for _ in range(10):
        out = trk.update([_det(5.0, 3.0, "blue"), _det(8.0, 3.0, "blue")] + base,
                         {sid: {"view_occluded": True}}, 0.1)
    coasting = [t for t in out if t.slot_id == sid][0]
    assert coasting.state == "coasting"
    out = trk.update([_det(5.6, 3.0, "blue"), _det(5.0, 3.0, "blue"),
                      _det(8.0, 3.0, "blue")] + base, {}, 0.1)
    assert any(t.slot_id == sid and t.state == "visible" for t in out)


def test_open_track_expires_and_reseeds():
    trk = FieldTracker(field_length=16.541, max_coast_open=2, max_coast_occluded=30)
    base = [_det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    _step(trk, [_det(2, 3, "blue"), _det(5, 3, "blue"), _det(8, 3, "blue")] + base)
    for _ in range(4):
        out = trk.update([_det(5, 3, "blue"), _det(8, 3, "blue")] + base, {}, 0.1)
    out = trk.update([_det(3, 6, "blue"), _det(5, 3, "blue"), _det(8, 3, "blue")] + base,
                     {}, 0.1)
    assert sum(t.alliance == "blue" and t.state == "visible" for t in out) == 3


def test_color_unknown_still_associates_by_position():
    trk = FieldTracker(field_length=16.541)
    base = [_det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    _step(trk, [_det(2, 3, "blue"), _det(5, 3, "blue"), _det(8, 3, "blue")] + base)
    out = trk.update([_det(2.1, 3.0, "unknown"), _det(5, 3, "blue"),
                      _det(8, 3, "blue")] + base, {}, 0.1)
    assert sum(t.alliance == "blue" and t.state == "visible" for t in out) == 3


def test_coasting_track_not_matched_across_field():
    """Regression: a coasting track must NOT be matched to a detection 13 m away."""
    trk = FieldTracker(field_length=16.541, max_coast_open=4, max_coast_occluded=90)
    # Seed a single blue track at (2, 3)
    _step(trk, [_det(2, 3, "blue")])
    blue_slot = [t for t in trk.tracks if t.alliance == "blue" and t.active][0]
    sid = blue_slot.slot_id

    # Let it coast for 3 frames with view_occluded=True (no matching detections)
    for _ in range(3):
        trk.update([], {sid: {"view_occluded": True}}, 0.1)

    # Confirm the slot is still active and coasting near its origin
    assert blue_slot.active
    assert blue_slot.coasting
    origin_x, origin_y = blue_slot.kf.pos

    # Present a detection very far away (across the field, ~13 m from origin)
    far_det = _det(15, 7, "blue")
    trk.update([far_det], {sid: {"view_occluded": True}}, 0.1)

    # The coasting slot must NOT have teleported to ~(15, 7)
    px, py = blue_slot.kf.pos
    dist_moved = ((px - origin_x) ** 2 + (py - origin_y) ** 2) ** 0.5
    assert dist_moved < 5.0, (
        f"Coasting track teleported: was ~({origin_x:.1f},{origin_y:.1f}), "
        f"now ({px:.1f},{py:.1f}) — dist {dist_moved:.1f} m"
    )


def test_unknown_detection_seeds_one_slot_only():
    """Regression: an unknown detection in blue territory must seed exactly one slot."""
    trk = FieldTracker(field_length=16.541)
    # x=2 is well within the blue half (< 16.541/2 ≈ 8.27)
    _step(trk, [_det(2, 4, "unknown")])

    active = [t for t in trk.tracks if t.active]
    assert len(active) == 1, (
        f"Expected 1 active slot, got {len(active)}: "
        + str([(t.slot_id, t.alliance) for t in active])
    )
    assert active[0].alliance == "blue", (
        f"Expected blue slot, got {active[0].alliance}"
    )

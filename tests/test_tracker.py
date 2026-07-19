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


def _mk_tracker(**kw):
    return FieldTracker(field_length=16.54, zones=[], **kw)


def test_kinematic_gate_rejects_teleport():
    # gate_m widened so the 5 m jump reaches the kinematic check in the cost
    # matrix instead of being excluded earlier by the plain distance gate
    # (default gate_m=2.5 would reject it before _kin_violates ever runs).
    tr = _mk_tracker(kin_gate_enabled=True, kin_max_speed_mps=6.5, gate_m=10.0)
    d0 = FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 5, 5))
    tr.update([d0], {}, dt=0.125)
    orig_slot_id = [t for t in tr.tracks if t.alliance == "blue" and t.active][0].slot_id
    # 5 m in one 0.125s frame = 40 m/s: impossible — must not update the slot
    d1 = FieldDetection(pos=(7.0, 2.0), color="blue", box=(0, 0, 5, 5))
    out = tr.update([d1], {}, dt=0.125)
    # The original slot must not have followed the jump (it may still seed a
    # *free* slot at (7, 2) instead — that's intended, it might be a
    # different robot; see brief note).
    orig = [t for t in out if t.slot_id == orig_slot_id][0]
    assert abs(orig.pos[0] - 7.0) > 2.0
    assert any(e["event"] == "kinematic_reject" for e in tr.binding_events)


def test_kinematic_gate_allows_post_coast_reacquire():
    tr = _mk_tracker(kin_gate_enabled=True, kin_max_speed_mps=6.5)
    d0 = FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 5, 5))
    tr.update([d0], {}, dt=0.125)
    for _ in range(4):                      # coast 4 frames (missed accrues)
        tr.update([], {}, dt=0.125)
    # 2.5 m over 5 frame-intervals = 4 m/s: fine — must re-acquire
    d1 = FieldDetection(pos=(4.5, 2.0), color="blue", box=(0, 0, 5, 5))
    out = tr.update([d1], {}, dt=0.125)
    vis = [t for t in out if t.state == "visible"]
    assert len(vis) == 1 and abs(vis[0].pos[0] - 4.5) < 1.0


def test_kinematic_gate_default_off_preserves_behavior():
    tr = _mk_tracker()
    d0 = FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 5, 5))
    tr.update([d0], {}, dt=0.125)
    d1 = FieldDetection(pos=(7.0, 2.0), color="blue", box=(0, 0, 5, 5))
    tr.update([d1], {}, dt=0.125)
    assert not any(e["event"] == "kinematic_reject" for e in tr.binding_events)


def _drive_into_occ_coast(tr, sid, seed_pos=(5.0, 3.0)):
    """Seed a blue slot, drive it hub-occluded, then miss frames so it enters
    occ_coasting via the public API (as existing occlusion tests do)."""
    tr.update([_det(*seed_pos, "blue")], {}, 0.1)
    # Miss frames while marked view_occluded -> slot coasts and sets occ_coasting.
    for _ in range(3):
        tr.update([], {sid: {"view_occluded": True}}, 0.1)


def test_portal_reacquire_outside_gate():
    # gate small so the far detection is well outside even gate*4.
    tr = FieldTracker(field_length=16.541, gate_m=0.5,
                      max_coast_open=90, max_coast_occluded=90)
    sid = [t for t in tr.tracks if t.alliance == "blue"][0].slot_id
    _drive_into_occ_coast(tr, sid)
    slot = [t for t in tr.tracks if t.slot_id == sid][0]
    assert slot.occ_coasting
    # Detection at (9.2, 3.1): >> gate*4 (=2.0) from the coasting slot near (5,3).
    far = _det(9.2, 3.1, "blue")
    out = tr.update([far], {sid: {"view_occluded": True}},
                    0.1, portals={sid: (9.0, 3.0, 1.0)})
    reacq = [t for t in out if t.slot_id == sid][0]
    # Portal accepted the association: slot re-acquired (visible) and the far
    # detection bound to it. (Exact KF pos lags — the jump trips the
    # max-speed r_scale downweight — but the binding is the contract.)
    assert reacq.state == "visible"
    assert tr.last_det_to_slot.get(0) == sid


def test_portal_off_no_reacquire():
    tr = FieldTracker(field_length=16.541, gate_m=0.5,
                      max_coast_open=90, max_coast_occluded=90)
    sid = [t for t in tr.tracks if t.alliance == "blue"][0].slot_id
    _drive_into_occ_coast(tr, sid)
    far = _det(9.2, 3.1, "blue")
    # No portals passed: the far detection must NOT re-acquire the coasting slot.
    out = tr.update([far], {sid: {"view_occluded": True}}, 0.1)
    reacq = [t for t in out if t.slot_id == sid][0]
    assert not (abs(reacq.pos[0] - 9.2) < 1.0 and abs(reacq.pos[1] - 3.1) < 1.0)


def test_history_off_by_default():
    trk = FieldTracker(field_length=16.541)
    _step(trk, [_det(2, 2, "blue")])
    assert trk.history == {}


def test_history_recording_segments_and_gaps():
    trk = FieldTracker(field_length=16.541, max_coast_open=2, record_history=True)
    red = [_det(13, 2, "red"), _det(12, 5, "red"), _det(14, 7, "red")]
    trk.update([_det(2.0, 2.0, "blue")] + red, {}, 0.1, frame_num=100)  # seed
    trk.update([_det(2.2, 2.0, "blue")] + red, {}, 0.1, frame_num=101)  # measure
    trk.update(red, {}, 0.1, frame_num=102)  # miss 1 -> coast record
    trk.update(red, {}, 0.1, frame_num=103)  # miss 2 -> coast record
    trk.update(red, {}, 0.1, frame_num=104)  # miss 3 > budget -> expire, no record
    trk.update([_det(5.0, 2.0, "blue")] + red, {}, 0.1, frame_num=105)  # re-seed
    # blue slots are 0..2; seeding uses the first free slot both times
    segs = trk.history[0]
    assert len(segs) == 2
    seg1, seg2 = segs
    assert [r["frame_num"] for r in seg1] == [100, 101, 102, 103]
    assert seg1[0]["seed"] is True and seg1[0]["z"] == (2.0, 2.0)
    assert seg1[1]["seed"] is False and seg1[1]["z"] == (2.2, 2.0)
    assert seg1[1]["r_scale"] == 1.0
    assert seg1[2]["z"] is None and seg1[2]["r_scale"] is None
    assert seg1[3]["z"] is None
    assert seg2[0]["seed"] is True and seg2[0]["z"] == (5.0, 2.0)
    assert seg2[0]["frame_num"] == 105


def test_history_records_corner_only_r_scale():
    trk = FieldTracker(field_length=16.541, record_history=True)
    trk.update([_det(2.0, 2.0, "blue")], {}, 0.1, frame_num=1)  # seed (has box)
    corner = FieldDetection(pos=(2.1, 2.0), color="blue", box=None, score=0.9)
    trk.update([corner], {}, 0.1, frame_num=2)
    rec = trk.history[0][0][1]
    assert rec["r_scale"] == 8.0          # corner_only_r_scale default
    assert rec["corner_only"] is True

"""Exact slot discontinuity logging: FieldTracker.slot_events (observation-only)
and web.field.slot_boundaries.frame_boundary_rows."""
from web.field.slot_boundaries import frame_boundary_rows
from web.field.tracker import FieldDetection, FieldTracker

DT = 0.1


def _red(x, y, tid=None, box=(0, 0, 10, 10)):
    return FieldDetection((x, y), "red", box, track_id=tid)


def _events(tr, kind):
    return [e for e in tr.slot_events if e["event"] == kind]


def test_seed_expire_reseed_records_the_teleport_distance():
    tr = FieldTracker(field_length=16.0, max_coast_open=2)
    tr.update([_red(12.0, 2.0, tid=1)], {}, DT)
    (seed,) = _events(tr, "seed")
    assert seed["slot_id"] == 3 and seed["prev_pos"] is None and seed["seed_dist_m"] is None
    for _ in range(3):                         # coast past the open budget
        tr.update([], {}, DT)
    (exp,) = _events(tr, "expire")
    assert exp["slot_id"] == 3 and exp["missed"] == 3
    assert abs(exp["pos"][0] - 12.0) < 0.5
    tr.update([_red(14.0, 9.0, tid=2)], {}, DT)  # free_slots.pop(0): slot 3 again
    (seed,) = _events(tr, "seed")
    assert seed["slot_id"] == 3 and abs(seed["seed_dist_m"] - 7.28) < 0.6
    assert tr.update([], {}, DT) and tr.slot_events == []          # reset per update


def test_coasting_reacquisition_records_missed_and_jumps():
    tr = FieldTracker(field_length=16.0, max_coast_open=8)
    tr.update([_red(10.0, 2.0)], {}, DT)
    tr.update([_red(10.1, 2.0)], {}, DT)
    tr.update([], {}, DT)
    tr.update([], {}, DT)                      # 2 missed updates
    tr.update([_red(12.0, 2.0)], {}, DT)       # within the widened gate (2.5 x 2)
    (ev,) = _events(tr, "reacquire")
    assert ev["slot_id"] == 3 and ev["missed"] == 2 and ev["via"] == "distance"
    assert abs(ev["jump_m"] - 1.9) < 1e-6
    assert ev["gate_m"] == 2.5 * 2.0
    assert abs(ev["speed_mps"] - 1.9 / (DT * 3)) < 1e-6
    assert ev["pred_jump_m"] < ev["jump_m"]    # the coast carried the slot forward
    tr.update([_red(12.1, 2.0)], {}, DT)       # visible -> visible: no event
    assert _events(tr, "reacquire") == []


def test_sticky_reacquisition_is_recorded():
    tr = FieldTracker(field_length=16.0)
    tr.update([_red(10.0, 2.0, tid=7)], {}, DT)
    tr.update([], {}, DT)
    tr.update([_red(10.5, 2.0, tid=7)], {}, DT)
    (ev,) = _events(tr, "reacquire")
    assert ev["via"] == "sticky" and ev["gate_m"] is None and ev["track_id"] == 7


def _run(scenario, silence=False):
    tr = FieldTracker(field_length=16.0, max_coast_open=2)
    if silence:
        tr._note_seed = lambda *a, **k: None
        tr._note_reacquire = lambda *a, **k: None
    out = []
    for dets in scenario:
        outs = tr.update(dets, {}, DT)
        out.append([(t.slot_id, round(t.pos[0], 9), round(t.pos[1], 9), t.state,
                     t.corner_only) for t in outs])
        out.append(dict(tr.last_det_to_slot))
        out.append([dict(e) for e in tr.binding_events])
    return out


def test_slot_events_never_change_tracking():
    """Same detections with the event notes silenced: identical outputs."""
    scenario = [[_red(10, 2, 1), _red(12, 6, 2)], [_red(10.2, 2, 1), _red(12, 6.1, 2)],
                [_red(10.4, 2, 1)], [], [_red(11.5, 2.2, 1), _red(13, 7, None, None)],
                [], [], [], [_red(4, 4, 9)], [_red(4.2, 4, 9), _red(9, 1, 10)]]
    assert _run(scenario) == _run(scenario, silence=True)


def test_frame_boundary_rows_merge_sources_per_slot():
    rows = frame_boundary_rows(
        800,
        [{"event": "seed", "slot_id": 4, "seed_dist_m": 12.3456, "track_id": 9},
         {"event": "expire", "slot_id": 5, "missed": 9},
         {"event": "reacquire", "slot_id": 3, "missed": 2, "jump_m": 3.21,
          "pred_jump_m": 2.6, "speed_mps": 10.7, "gate_m": 5.0, "via": "distance",
          "track_id": 4},
         {"event": "slot_rebound", "slot_id": 0}],
        {3: (11, 12)},
        [{"slot_id": 3, "distance_m": 2.4, "slot_disrupted": True, "reason": "distance",
          "outcome": "born"}],
        {3: True, 4: False}, {3: False, 4: False},
        positions={3: (1.0, 2.0)})
    by = {r["slot_id"]: r for r in rows}
    assert sorted(by) == [3, 4, 5]
    assert by[3]["kinds"] == ["reacquire", "feed_change", "birth", "corner_only_on"]
    assert by[3]["jump_m"] == 3.21 and by[3]["prev_track_id"] == 11
    assert by[3]["birth_disrupted"] is True and by[3]["birth_distance_m"] == 2.4
    assert by[3]["pos"] == [1.0, 2.0] and by[3]["frame"] == 800
    assert by[4]["kinds"] == ["seed"] and by[4]["seed_dist_m"] == 12.346
    assert by[5]["kinds"] == ["expire"] and by[5]["pos"] is None
    assert frame_boundary_rows(8, [], {}, [], {3: True}, {}) == []   # no prior state

"""Field-tracker re-acquisition knobs (all default OFF):

- assign_partial_fix: gated-out pairs no longer collapse the assignment search;
- coast_gate_phys: a coasting slot's widened gate is bounded by physical reach;
- reseed_identity: leftover detections reseed the free slot that explains them,
  and a just-expired slot is held against far teleports;
- corner_dup_suppress: unfused corner views of a main-tracked robot can't seed.

Plus the default-off guarantee: with every knob off the tracker reproduces the
pre-change tracker exactly on a randomized scenario (golden digest).
"""
import hashlib
import json
import random

from web.field.tracker import FieldDetection, FieldTracker

DT = 8 / 60.0
MAIN_BOX = (0, 0, 10, 10)


def _det(x, y, color="blue", box=MAIN_BOX, tid=None):
    return FieldDetection(pos=(float(x), float(y)), color=color, box=box,
                          track_id=tid)


def _corner(x, y, color="blue", tid=None):
    return _det(x, y, color, box=None, tid=tid)


def _slots(out, alliance="blue"):
    return {t.slot_id: t for t in out if t.alliance == alliance}


# ---------------------------------------------------------------------------
# default-off identical (golden digest of the pre-change tracker)
# ---------------------------------------------------------------------------

def _scenario(seed=7, frames=400):
    """Six robots wandering with noise, dropouts, corner-only views, unknown
    colours, duplicated/ambiguous track ids and the odd far false positive."""
    rng = random.Random(seed)
    robots = []
    for i in range(6):
        color = "blue" if i < 3 else "red"
        x0 = rng.uniform(1, 7) if color == "blue" else rng.uniform(9.5, 15.5)
        robots.append({"color": color, "p": [x0, rng.uniform(0.5, 7.5)],
                       "v": [0.0, 0.0], "tid": i})
    seq = []
    for f in range(frames):
        dets = []
        for r in robots:
            r["v"][0] = 0.8 * r["v"][0] + rng.gauss(0, 0.6)
            r["v"][1] = 0.8 * r["v"][1] + rng.gauss(0, 0.6)
            for k, hi in ((0, 16.0), (1, 8.0)):
                r["p"][k] = min(hi, max(0.0, r["p"][k] + r["v"][k] * DT))
            u = rng.random()
            if u < 0.12:
                continue                        # dropout
            pos = (r["p"][0] + rng.gauss(0, 0.15), r["p"][1] + rng.gauss(0, 0.15))
            box = None if u < 0.22 else MAIN_BOX   # corner-only view
            color = "unknown" if rng.random() < 0.05 else r["color"]
            tid = r["tid"] if rng.random() < 0.7 else (0 if rng.random() < 0.5 else None)
            dets.append(_det(pos[0], pos[1], color, box, tid))
            if box is not None and rng.random() < 0.08:   # unfused corner dup
                dets.append(_corner(pos[0] + 1.2, pos[1] + 0.6, r["color"]))
        if rng.random() < 0.05:                 # far false positive
            dets.append(_det(rng.uniform(0, 16), rng.uniform(0, 8),
                             rng.choice(["blue", "red"]), None, None))
        rng.shuffle(dets)
        occ = {s: {"view_occluded": rng.random() < 0.1} for s in range(6)}
        seq.append((dets, occ))
    return seq


def _digest(tracker, seq):
    h = hashlib.sha256()
    for dets, occ in seq:
        out = tracker.update(dets, occ, DT)
        rec = {"d2s": sorted(tracker.last_det_to_slot.items()),
               "tr": [(t.slot_id, t.state, t.corner_only, round(t.pos[0], 6),
                       round(t.pos[1], 6)) for t in out],
               "ev": tracker.binding_events}
        h.update(json.dumps(rec, sort_keys=True, default=str).encode())
    return h.hexdigest()


# Digest of the tracker BEFORE the re-acquisition knobs existed (commit
# ba31934f web/field/tracker.py) on _scenario(), sticky_max_dist_m=2.5 as the
# live job default.
GOLDEN_DEFAULT_DIGEST = (
    "58da8beb318ad538701179a34a9fc23a05615abd523f64265cd1773c313c98d1")


def test_default_off_reproduces_pre_change_tracker():
    trk = FieldTracker(16.541, sticky_max_dist_m=2.5)
    assert _digest(trk, _scenario()) == GOLDEN_DEFAULT_DIGEST


def test_knob_defaults_are_off():
    trk = FieldTracker(16.541)
    assert not trk.assign_partial_fix
    assert not trk.coast_gate_phys
    assert not trk.reseed_identity
    assert not trk.corner_dup_suppress


def test_config_dict_round_trips():
    trk = FieldTracker(16.541, zones=[(1, 1, 2, 2)], sticky_max_dist_m=2.5,
                       coast_gate_phys=True, reseed_hold_s=2.0)
    cfg = trk.config_dict()
    json.dumps(cfg)                      # trace header must serialise
    again = FieldTracker(**cfg)
    assert again.config_dict() == cfg


# ---------------------------------------------------------------------------
# assign_partial_fix
# ---------------------------------------------------------------------------

def _coasting_thief_setup(**kw):
    """Blue slot 0 coasting near (2, 2) (3 missed frames, widened gate), slot 1
    visible at (5, 2). Then slot 1's robot is detected at (5.1, 2) together
    with an extra blue det at (12, 7) that is out of gate for both slots."""
    trk = FieldTracker(16.541, **kw)
    trk.update([_det(2, 2), _det(5, 2)], {}, DT)
    for _ in range(3):
        trk.update([_det(5, 2)], {}, DT)
    out = trk.update([_det(5.1, 2), _det(12, 7)], {}, DT)
    return trk, _slots(out)


def test_partial_assignment_degenerates_without_fix():
    trk, s = _coasting_thief_setup()
    # Every full pairing contains the out-of-gate det -> all sums tie at inf
    # -> the first permutation wins: coasting slot 0 steals slot 1's robot.
    assert trk.last_det_to_slot[0] == 0
    assert s[1].state == "coasting"


def test_partial_assignment_fix_keeps_visible_slot():
    trk, s = _coasting_thief_setup(assign_partial_fix=True)
    assert trk.last_det_to_slot[0] == 1
    assert s[1].state == "visible"
    assert s[0].state == "coasting"
    assert trk.last_det_to_slot[1] == 2      # the extra det seeds the free slot


# ---------------------------------------------------------------------------
# coast_gate_phys
# ---------------------------------------------------------------------------

def _coast_then(det_pos, missed, **kw):
    trk = FieldTracker(16.541, **kw)
    trk.update([_det(2, 2)], {}, DT)
    for _ in range(missed):
        trk.update([], {}, DT)
    out = trk.update([_det(*det_pos)], {}, DT)
    return trk, _slots(out)


def test_coast_within_reach_reacquires():
    # 6 missed frames = 0.93 s since the last measurement: 4 m is reachable
    # (5 m/s x 0.93 s + 1 m slack) and inside the widened gate.
    trk, s = _coast_then((6.0, 2.0), 6, coast_gate_phys=True)
    assert trk.last_det_to_slot == {0: 0}
    assert s[0].state == "visible"


def test_coast_beyond_reach_does_not_reacquire():
    # 1 missed frame = 0.27 s: 3.5 m is inside the widened gate (3.75 m) but
    # far beyond reach (2.33 m) -> another robot; it seeds its own slot.
    trk, s = _coast_then((5.5, 2.0), 1, coast_gate_phys=True)
    assert trk.last_det_to_slot == {0: 1}
    assert s[0].state == "coasting"
    assert any(e["event"] == "coast_gate_reject" for e in trk.slot_events)


def test_coast_beyond_reach_reacquires_when_off():
    trk, s = _coast_then((5.5, 2.0), 1)
    assert trk.last_det_to_slot == {0: 0}


def test_coast_gate_never_touches_visible_slots():
    a = FieldTracker(16.541, coast_gate_phys=True)
    b = FieldTracker(16.541)
    for f in range(20):
        dets = [_det(2 + 0.4 * f, 2), _det(6, 2 + 0.3 * (f % 3))]
        oa, ob = a.update(dets, {}, DT), b.update(dets, {}, DT)
        assert a.last_det_to_slot == b.last_det_to_slot
        assert [t.pos for t in oa] == [t.pos for t in ob]


# ---------------------------------------------------------------------------
# reseed_identity
# ---------------------------------------------------------------------------

def _expire_two(**kw):
    """Blue slots 0 at (2, 2) and 1 at (2, 6) both expire (9+ missed)."""
    trk = FieldTracker(16.541, **kw)
    trk.update([_det(2, 2), _det(2, 6)], {}, DT)
    for _ in range(9):
        trk.update([], {}, DT)
    assert not [t for t in trk.tracks if t.active]
    return trk


def test_reseed_picks_the_explaining_slot():
    trk = _expire_two(reseed_identity=True)
    trk.update([_det(2.3, 6.2)], {}, DT)       # slot 1's robot reappears
    assert trk.last_det_to_slot == {0: 1}


def test_reseed_default_takes_lowest_free_slot():
    trk = _expire_two()
    trk.update([_det(2.3, 6.2)], {}, DT)
    assert trk.last_det_to_slot == {0: 0}


def test_reseed_prefers_fresh_slot_over_far_expired_one():
    # Slot 0 expired at (2, 2); slot 2 was never seeded. A robot appearing
    # 12 m away is not slot 0's robot (beyond reach) -> the fresh slot.
    trk = FieldTracker(16.541, reseed_identity=True, reseed_hold_s=0.0)
    trk.update([_det(2, 2), _det(4, 4)], {}, DT)
    for _ in range(9):
        trk.update([_det(4, 4)], {}, DT)       # slot 0 expires, slot 1 lives
    trk.update([_det(4, 4), _det(14, 7)], {}, DT)
    assert trk.last_det_to_slot == {0: 1, 1: 2}


def test_reseed_holds_recent_slot_against_teleport():
    trk = FieldTracker(16.541, reseed_identity=True, reseed_hold_s=1.5)
    trk.update([_det(2, 2), _det(4, 4), _det(6, 6)], {}, DT)
    for _ in range(9):
        trk.update([_det(4, 4), _det(6, 6)], {}, DT)   # slot 0 expires
    far = _det(14, 7)
    trk.update([_det(4, 4), _det(6, 6), far], {}, DT)
    assert 2 not in trk.last_det_to_slot                # held: not seeded
    assert any(e["event"] == "reseed_hold" for e in trk.slot_events)
    # The lost robot reappears near its last position: it gets its slot back.
    trk.update([_det(4, 4), _det(6, 6), _det(2.4, 2.2)], {}, DT)
    assert trk.last_det_to_slot[2] == 0


def test_reseed_hold_expires():
    trk = FieldTracker(16.541, reseed_identity=True, reseed_hold_s=1.0)
    trk.update([_det(2, 2), _det(4, 4), _det(6, 6)], {}, DT)
    for _ in range(9):
        trk.update([_det(4, 4), _det(6, 6)], {}, DT)
    seeded_at = None
    for f in range(20):
        trk.update([_det(4, 4), _det(6, 6), _det(14, 7)], {}, DT)
        if trk.last_det_to_slot.get(2) == 0:
            seeded_at = f
            break
    # Held while the teleport is beyond reach within the hold window, then
    # the slot is reseeded (coverage is delayed, not lost).
    assert seeded_at is not None and 1 <= seeded_at <= 9


# ---------------------------------------------------------------------------
# corner_dup_suppress
# ---------------------------------------------------------------------------

def test_corner_dup_does_not_seed_second_slot():
    for on in (False, True):
        trk = FieldTracker(16.541, corner_dup_suppress=on)
        trk.update([_det(5, 5)], {}, DT)
        out = trk.update([_det(5, 5.05), _corner(5.8, 6.4)], {}, DT)
        active = [t for t in out if t.alliance == "blue"]
        assert len(active) == (1 if on else 2)


def test_corner_only_far_from_main_still_seeds():
    trk = FieldTracker(16.541, corner_dup_suppress=True)
    trk.update([_det(5, 5)], {}, DT)
    out = trk.update([_det(5, 5.05), _corner(9, 1)], {}, DT)
    assert len([t for t in out if t.alliance == "blue"]) == 2


def test_corner_dup_cannot_feed_coasting_main_slot():
    # Slot 0 (main-backed robot A) loses its detection; robot B (main) at
    # (5, 5) has slot 1, and B's unfused corner view sits 1.5 m away — inside
    # slot 0's widened gate. It must not re-acquire slot 0.
    for on, expect in ((False, 0), (True, None)):
        trk = FieldTracker(16.541, corner_dup_suppress=on)
        trk.update([_det(5.5, 7.5), _det(5, 5)], {}, DT)
        trk.update([_det(5, 5)], {}, DT)
        trk.update([_det(5, 5), _corner(5.6, 6.4)], {}, DT)
        assert trk.last_det_to_slot.get(1) == expect


def test_corner_view_of_a_second_robot_is_not_a_dup():
    # Robot B is seen only by corner_right (main occluded by robot A) and
    # stays within the dup radius of A. A's main cluster already holds
    # corner_right's view of A, so B's corner detection is a second robot:
    # it seeds and keeps its own slot.
    trk = FieldTracker(16.541, corner_dup_suppress=True)
    a = dict(color="blue", box=MAIN_BOX)
    for _ in range(5):
        out = trk.update([
            FieldDetection(pos=(5.0, 5.6), cams=("corner_right", "main"), **a),
            FieldDetection(pos=(5.0, 7.2), color="blue", box=None,
                           cams=("corner_right",))], {}, DT)
    s = _slots(out)
    assert s[0].state == "visible" and not s[0].corner_only
    assert s[1].state == "visible" and s[1].corner_only


def test_corner_view_missing_from_main_cluster_is_a_dup():
    trk = FieldTracker(16.541, corner_dup_suppress=True)
    for _ in range(5):
        out = trk.update([
            FieldDetection(pos=(5.0, 5.6), color="blue", box=MAIN_BOX, cams=("main",)),
            FieldDetection(pos=(5.0, 7.2), color="blue", box=None,
                           cams=("corner_right",))], {}, DT)
    assert len(_slots(out)) == 1


def test_slot_riding_a_corner_dup_moves_to_the_main_view():
    # Slot 1 was seeded from robot C's corner view before main saw C. When
    # main sees C (1.24 m from the unfused corner view) the duplicate is
    # dropped: slot 1 continues on C's main detection, no second slot.
    trk = FieldTracker(16.541, corner_dup_suppress=True)
    trk.update([_det(1, 1), _corner(6.2, 3.0)], {}, DT)
    out = trk.update([_det(1, 1), _det(5.0, 3.3), _corner(6.2, 3.0)], {}, DT)
    assert trk.last_det_to_slot == {0: 0, 1: 1}
    s = _slots(out)
    assert set(s) == {0, 1} and not s[1].corner_only


def test_fusion_stamps_contributing_cameras():
    from web.field.fusion import fuse_detections, merge_same_id_observations
    main = _det(5.0, 5.0)
    near_corner = _det(5.3, 5.1)
    far_corner = _det(12.0, 2.0)
    fused = fuse_detections({"main": [(main, None)],
                             "corner_right": [(near_corner, None), (far_corner, None)]},
                            weights={"main": 1.0, "corner_right": 1.0})
    cams = sorted(o.detection.cams for o in fused)
    assert cams == [("corner_right",), ("corner_right", "main")]
    for i, o in enumerate(fused):
        o.track_id = 7
        o.detection.track_id = 7
    merged, _ = merge_same_id_observations(fused)
    assert merged[0].detection.cams == ("corner_right", "main")

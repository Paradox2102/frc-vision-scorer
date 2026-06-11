"""SP-T.2 end-to-end (pure): per-camera SAM2 tracklets -> fuse -> stable fused id
-> FieldTracker sticky slot. No GPU; tracklets are hand-built field detections."""
from web.field.fusion import fuse_detections
from web.field.fused_track_id import FusedTrackRegistry
from web.field.tracker import FieldDetection, FieldTracker

FIELD_LEN = 16.0
WEIGHTS = {"main": 1.0, "corner_left": 2.5, "corner_right": 2.5}


def _det(x, y, track_id, color="blue", box=(0, 0, 10, 10)):
    return FieldDetection(pos=(x, y), color=color, box=box, track_id=track_id)


def _step(tracker, reg, per_cam, dt=0.1):
    fused = fuse_detections(per_cam, weights=WEIGHTS, radius_m=0.6)
    reg.assign(fused)                       # stamps obs.detection.track_id
    dets = [obs.detection for obs in fused]
    return tracker.update(dets, {}, dt)


def test_two_views_one_robot_one_slot():
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    # one blue robot at ~(3,3), seen by main (id 1) and corner_left (id 1)
    per_cam = {"main": [(_det(3.0, 3.0, 1), b"m")],
               "corner_left": [(_det(3.2, 3.1, 1), b"c")]}
    outs = _step(tracker, reg, per_cam)
    assert len(outs) == 1                   # exactly one slot active
    slot = outs[0].slot_id
    # next frame, robot moved a bit; same per-camera ids -> same slot
    per_cam2 = {"main": [(_det(3.3, 3.0, 1), b"m")],
                "corner_left": [(_det(3.5, 3.1, 1), b"c")]}
    outs2 = _step(tracker, reg, per_cam2)
    assert [o.slot_id for o in outs2] == [slot]


def test_corner_only_when_main_occluded_keeps_slot():
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    # frame 1: both cameras see the robot
    outs1 = _step(tracker, reg, {
        "main": [(_det(4.0, 2.0, 1), b"m")],
        "corner_left": [(_det(4.1, 2.0, 1), b"c")],
    })
    slot = outs1[0].slot_id
    # frame 2: main has NO detection (robot is hidden from main); corner_left still
    # tracks it. The fused observation now has only the corner member key, but the
    # registry keeps the same fused id, so FieldTracker re-binds the SAME slot.
    outs2 = _step(tracker, reg, {
        "main": [],
        "corner_left": [(_det(4.2, 2.0, 1), b"c")],
    })
    assert slot in [o.slot_id for o in outs2]
    # frame 3: main re-acquires (SAM2 id persists as 1) — still the same slot
    outs3 = _step(tracker, reg, {
        "main": [(_det(4.3, 2.0, 1), b"m")],
        "corner_left": [(_det(4.4, 2.0, 1), b"c")],
    })
    assert slot in [o.slot_id for o in outs3]


def test_two_distinct_robots_two_slots():
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    outs = _step(tracker, reg, {
        "main": [(_det(2.0, 2.0, 1), b"a"), (_det(6.0, 5.0, 2), b"b")],
    })
    assert len({o.slot_id for o in outs}) == 2


def test_duplicate_sam2_masks_on_one_robot_use_one_slot():
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    outs = _step(tracker, reg, {
        "main": [
            (_det(6.70, 3.40, 1, "red", (690, 410, 890, 535)), b"a"),
            (_det(6.74, 3.43, 3, "red", (700, 418, 885, 530)), b"b"),
        ],
    })

    assert len(outs) == 1
    assert outs[0].alliance == "red"


def test_abrupt_position_swap_keeps_slot_by_track_id():
    # The real teeth: two BLUE robots (same alliance -> same matching pool) swap
    # positions ABRUPTLY in one frame — the constant-velocity model can't predict
    # it (a scrum/occlusion proxy, the exact case SAM2 stable ids fix). Plain
    # distance matching steals each slot to the now-nearest robot (a swap); stable
    # fused track_ids must pin each robot to its ORIGINAL slot.
    # last_det_to_slot[i] is the slot the i-th fused detection got; fused order
    # tracks input order, so index 0 == robot A, index 1 == robot B.
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()

    def frame(ax, bx):
        # A is SAM2 id 1, B is SAM2 id 2, both seen only by main.
        per_cam = {"main": [(_det(ax, 3.0, 1), b"a"), (_det(bx, 3.0, 2), b"b")]}
        fused = fuse_detections(per_cam, weights=WEIGHTS, radius_m=0.6)
        reg.assign(fused)
        dets = [o.detection for o in fused]
        tracker.update(dets, {}, 0.1)
        return tracker.last_det_to_slot.get(0), tracker.last_det_to_slot.get(1)

    a0, b0 = frame(2.0, 5.0)        # A left, B right (velocities seed at 0)
    assert a0 is not None and b0 is not None and a0 != b0
    # One-frame swap: A jumps to B's old spot, B to A's. Velocity (0) predicts
    # "stay put", so distance matching would hand A's old slot to B and vice versa.
    a1, b1 = frame(5.0, 2.0)
    assert a1 == a0, "robot A must keep its original slot via stable track_id"
    assert b1 == b0, "robot B must keep its original slot via stable track_id"


def test_lost_track_relink_retains_slot_by_embedding():
    import numpy as np
    tracker = FieldTracker(FIELD_LEN)
    reg = FusedTrackRegistry()
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Frame 1: Robot A seen on main:1 with embedding A.
    det1 = _det(3.0, 3.0, 1)
    det1.embedding = emb_a
    per_cam1 = {"main": [(det1, b"a")]}
    outs1 = _step(tracker, reg, per_cam1)
    assert len(outs1) == 1
    slot = outs1[0].slot_id

    # Frame 2: Robot A disappears (e.g. occluded, no detections).
    # FieldTracker coasts it.
    outs2 = _step(tracker, reg, {"main": []})
    assert len(outs2) == 1
    assert outs2[0].state == "coasting"

    # Frame 3: Robot A re-appears at a completely different position (beyond the position gate),
    # and under a new SAM2 ID (main:2), but carries the same embedding A.
    det3 = _det(8.0, 8.0, 2)
    det3.embedding = emb_a
    per_cam3 = {"main": [(det3, b"a")]}
    outs3 = _step(tracker, reg, per_cam3)
    
    # It should have re-linked to fused ID 0, and FieldTracker should have re-bound it to the SAME slot!
    assert len(outs3) == 1
    assert outs3[0].slot_id == slot
    assert outs3[0].state == "visible"

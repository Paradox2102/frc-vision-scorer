"""tracker_prob is payload-only: it rides FieldDetection -> matched slot ->
TrackOut without ever influencing assignment, gating, or fusion weights.

The companion change plumbs SAM3 per-object confidence into SAM track evidence
(``evidence["tracker_prob"]``); this carry lets the app's auto-path sample site
record the per-slot SAM confidence for the frame. Correctness criteria:
  * a matched slot's TrackOut carries the winning detection's tracker_prob;
  * a slot with no prob (legacy / corner-only) emits None;
  * adding tracker_prob must not change which detection wins a slot.
"""
from web.field.tracker import FieldTracker, FieldDetection
from web.field import fusion
from web.field.fusion import fuse_detections


def test_tracker_prob_rides_matched_slot_to_trackout():
    tr = FieldTracker(field_length=16.0)
    outs = tr.update(
        [FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1),
                        tracker_prob=0.87)],
        {}, 0.1,
    )
    assert len(outs) == 1
    assert outs[0].tracker_prob == 0.87


def test_tracker_prob_none_when_detection_lacks_it():
    tr = FieldTracker(field_length=16.0)
    outs = tr.update(
        [FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1))],
        {}, 0.1,
    )
    assert outs[0].tracker_prob is None


def test_tracker_prob_updates_on_rematch():
    tr = FieldTracker(field_length=16.0)
    tr.update([FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1),
                              tracker_prob=0.5)], {}, 0.1)
    outs = tr.update([FieldDetection(pos=(2.1, 2.0), color="blue", box=(0, 0, 1, 1),
                                     tracker_prob=0.9)], {}, 0.1)
    slot = next(o for o in outs if o.state == "visible")
    assert slot.tracker_prob == 0.9


def test_coasting_slot_keeps_last_prob():
    """When a slot goes unmatched (coasts), it is not re-measured, so its
    tracker_prob is left as last observed rather than being reset."""
    tr = FieldTracker(field_length=16.0)
    tr.update([FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1),
                              tracker_prob=0.6)], {}, 0.1)
    outs = tr.update([], {}, 0.1)  # no detections -> the slot coasts
    coasting = [o for o in outs if o.state == "coasting"]
    assert coasting and coasting[0].tracker_prob == 0.6


def test_tracker_prob_does_not_change_assignment():
    """Assignment must be identical with and without tracker_prob set. Two
    detections, two slots: the pairing is decided by distance only."""
    def run(with_prob):
        tr = FieldTracker(field_length=16.0)
        tr.update([
            FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1)),
            FieldDetection(pos=(6.0, 6.0), color="blue", box=(0, 0, 1, 1)),
        ], {}, 0.1)
        outs = tr.update([
            FieldDetection(pos=(2.1, 2.0), color="blue", box=(0, 0, 1, 1),
                           tracker_prob=0.99 if with_prob else None),
            FieldDetection(pos=(6.1, 6.0), color="blue", box=(0, 0, 1, 1),
                           tracker_prob=0.01 if with_prob else None),
        ], {}, 0.1)
        return {round(o.pos[0]): o.slot_id for o in outs}

    assert run(False) == run(True)


# --------------------------------------------------------------------------- #
# fusion carry: the merged detection takes the max-weight member's tracker_prob
# --------------------------------------------------------------------------- #
def test_fusion_carries_max_weight_member_prob():
    # Two cameras see one robot; main (weight 1.0) has prob 0.8, corner
    # (weight 0.3) has prob 0.2. The merged detection carries the dominant
    # (max-weight) member's prob = 0.8.
    per_camera = {
        "main": [(FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 1, 1),
                                 tracker_prob=0.8), None)],
        "corner_left": [(FieldDetection(pos=(3.05, 3.0), color="blue", box=None,
                                        tracker_prob=0.2), None)],
    }
    fused = fuse_detections(per_camera, weights={"main": 1.0, "corner_left": 0.3})
    assert len(fused) == 1
    assert fused[0].detection.tracker_prob == 0.8


def test_fusion_prob_none_when_no_member_has_prob():
    per_camera = {
        "main": [(FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 1, 1)), None)],
    }
    fused = fuse_detections(per_camera, weights={"main": 1.0})
    assert fused[0].detection.tracker_prob is None


# --------------------------------------------------------------------------- #
# det_conf (detector box confidence) rides exactly like tracker_prob
# --------------------------------------------------------------------------- #
def test_det_conf_rides_slot_and_persists_while_coasting():
    tr = FieldTracker(field_length=16.0)
    outs = tr.update([FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1),
                                     det_conf=0.73)], {}, 0.1)
    assert outs[0].det_conf == 0.73 and outs[0].tracker_prob is None
    outs = tr.update([FieldDetection(pos=(2.1, 2.0), color="blue", box=(0, 0, 1, 1),
                                     det_conf=0.41)], {}, 0.1)
    assert next(o for o in outs if o.state == "visible").det_conf == 0.41
    outs = tr.update([], {}, 0.1)
    coasting = [o for o in outs if o.state == "coasting"]
    assert coasting and coasting[0].det_conf == 0.41   # app nulls it for coasting


def test_det_conf_does_not_change_assignment():
    def run(with_conf):
        tr = FieldTracker(field_length=16.0)
        tr.update([
            FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1)),
            FieldDetection(pos=(6.0, 6.0), color="blue", box=(0, 0, 1, 1)),
        ], {}, 0.1)
        outs = tr.update([
            FieldDetection(pos=(2.1, 2.0), color="blue", box=(0, 0, 1, 1),
                           det_conf=0.99 if with_conf else None),
            FieldDetection(pos=(6.1, 6.0), color="blue", box=(0, 0, 1, 1),
                           det_conf=0.01 if with_conf else None),
        ], {}, 0.1)
        return {round(o.pos[0]): (o.slot_id, tuple(o.pos)) for o in outs}

    assert run(False) == run(True)


def test_fusion_carries_max_weight_member_det_conf():
    per_camera = {
        "main": [(FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 1, 1),
                                 det_conf=0.35), None)],
        "corner_left": [(FieldDetection(pos=(3.05, 3.0), color="blue", box=None,
                                        det_conf=0.95), None)],
    }
    fused = fuse_detections(per_camera, weights={"main": 1.0, "corner_left": 0.3})
    assert len(fused) == 1
    assert fused[0].detection.det_conf == 0.35
    solo = fuse_detections(
        {"main": [(FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 1, 1)), None)]},
        weights={"main": 1.0})
    assert solo[0].detection.det_conf is None

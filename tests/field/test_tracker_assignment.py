from web.field.tracker import FieldTracker, FieldDetection


def test_update_reports_detection_to_slot():
    tr = FieldTracker(field_length=16.0)
    dets = [FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1))]
    outs = tr.update(dets, {}, 0.1)
    # new API: assignment maps detection index -> slot_id
    assert hasattr(tr, "last_det_to_slot")
    assert tr.last_det_to_slot.get(0) in {t.slot_id for t in outs}

from web.field.tracker import FieldTracker, FieldDetection


def test_update_reports_detection_to_slot():
    tr = FieldTracker(field_length=16.0)
    dets = [FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1))]
    outs = tr.update(dets, {}, 0.1)
    # new API: assignment maps detection index -> slot_id
    assert hasattr(tr, "last_det_to_slot")
    assert tr.last_det_to_slot.get(0) in {t.slot_id for t in outs}


def test_tracker_assignment_evidence_margins():
    tr = FieldTracker(field_length=16.0)
    
    # Seed two tracks in active slots
    tr.update([
        FieldDetection(pos=(2.0, 2.0), color="blue", box=(0, 0, 1, 1)),
        FieldDetection(pos=(5.0, 5.0), color="blue", box=(0, 0, 1, 1))
    ], {}, 0.1)
    
    # Frame 2: Move them and see if assignment_evidence is recorded
    tr.update([
        FieldDetection(pos=(2.1, 2.0), color="blue", box=(0, 0, 1, 1)),
        FieldDetection(pos=(5.1, 5.0), color="blue", box=(0, 0, 1, 1))
    ], {}, 0.1)
    
    assert len(tr.assignment_evidence) == 2
    for slot_id, evidence in tr.assignment_evidence.items():
        assert "distance" in evidence
        assert "margin" in evidence
        assert evidence["margin"] > 3.0

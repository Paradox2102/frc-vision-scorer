from types import SimpleNamespace
from web.field.track_evidence import MatchEvidence


def _job():
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)        # bypass heavy __init__ deps
    job._match_evidence = MatchEvidence()
    job._binding_log = []
    job.field_tracker = SimpleNamespace(_id_to_slot={1: 0, 2: 3})
    return job


def _obs(track_id, pos, box, color):
    det = SimpleNamespace(pos=pos, box=box, color=color)
    return SimpleNamespace(track_id=track_id, detection=det)


def test_record_frame_evidence_populates_and_keys_by_fused_id():
    job = _job()
    person = [(95, 50, 130, 200)]  # overlaps track 2's box below
    fused = [
        _obs(1, (2.0, 4.0), (10, 10, 40, 120), "red"),
        _obs(2, (8.0, 4.0), (100, 60, 140, 190), None),
    ]
    job._record_frame_evidence(fused, frame_num=7, main_person_boxes=person)
    assert set(job._match_evidence.by_track) == {1, 2}
    assert job._match_evidence.by_track[1].ever_bumper_color is True
    assert job._match_evidence.by_track[1].field_positions == [(7, 2.0, 4.0)]
    assert job._match_evidence.by_track[2].person_overlap_frames == 1
    assert job._binding_log == [{"frame_num": 7, "bindings": {1: 0, 2: 3}}]


def test_record_frame_evidence_skips_none_track_id():
    job = _job()
    job._record_frame_evidence([_obs(None, (1.0, 1.0), (0, 0, 5, 5), None)],
                               frame_num=0, main_person_boxes=[])
    assert job._match_evidence.by_track == {}


def test_track_id_for_slot_inverts_binding():
    job = _job()
    assert job._track_id_for_slot(0) == 1
    assert job._track_id_for_slot(3) == 2
    assert job._track_id_for_slot(5) is None

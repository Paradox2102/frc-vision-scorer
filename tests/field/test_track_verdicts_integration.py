from types import SimpleNamespace
from web.field.track_evidence import MatchEvidence


def _job():
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)
    job._match_evidence = MatchEvidence()
    job._binding_log = []
    job._replay = SimpleNamespace(frames=[])
    job.identity_diagnostics = SimpleNamespace(gauge=lambda *a, **k: None)
    return job


def test_apply_track_verdicts_prunes_human_and_static_rows():
    job = _job()
    job._match_evidence.confirm(1, 0, team_read=True)
    for f in range(8):
        job._match_evidence.observe(2, f, (5.0, 5.0), overlapped_person=True)
        job._match_evidence.observe(3, f, (9.0, 9.0), overlapped_person=False)
    job._binding_log = [{"frame_num": 0, "bindings": {1: 0, 2: 3, 3: 1}}]
    job._replay.frames = [{"frame_num": 0, "robots": [
        {"slot": "B1"}, {"slot": "R1"}, {"slot": "B2"},
    ]}]
    job._apply_track_verdicts()
    assert [r["slot"] for r in job._replay.frames[0]["robots"]] == ["B1"]


def test_apply_track_verdicts_noop_without_evidence():
    job = _job()
    job._replay.frames = [{"frame_num": 0, "robots": [{"slot": "B1"}]}]
    job._match_evidence = MatchEvidence()  # empty
    job._binding_log = []
    job._apply_track_verdicts()
    assert len(job._replay.frames[0]["robots"]) == 1

"""Vote arithmetic for additive corner-weighted bumper reads (Task 16).

Calls the real InferenceJob._apply_bumper_read bound to a tiny duck-typed
fake so we exercise the production vote math without loading the heavy
vision models that InferenceJob.__init__ would pull in.
"""
from web import app as appmod


class _Robot:
    def __init__(self):
        self.team_ocr_votes = None


def test_corner_outweighs_main(monkeypatch):
    monkeypatch.setattr(
        appmod, "_vision_read_bumpers",
        lambda crops, a, b, c: [{"team_number": "2102", "confidence": 0.9}],
    )
    job = appmod.InferenceJob.__new__(appmod.InferenceJob)  # no __init__ (skip model load)
    r = _Robot()
    job.robot_tracks = {3: r}

    # Corner read: conf 0.9 -> conf-weight 1.0, x source_weight 2.5 = 2.5
    appmod.InferenceJob._apply_bumper_read(
        job, 3, "x", ["2102"], ["2102"], [], source_weight=2.5)
    assert r.team_ocr_votes["2102"] == 2.5

    # Main-equivalent read: conf 0.9 -> 1.0, x source_weight 1.0 = 1.0
    appmod.InferenceJob._apply_bumper_read(
        job, 3, "x", ["2102"], ["2102"], [], source_weight=1.0)
    assert r.team_ocr_votes["2102"] == 3.5  # 2.5 + 1.0

    # One corner read out-weighs one main read for the same team.
    assert 2.5 > 1.0


def test_low_confidence_halves_then_weights(monkeypatch):
    monkeypatch.setattr(
        appmod, "_vision_read_bumpers",
        lambda crops, a, b, c: [{"team_number": "254", "confidence": 0.5}],
    )
    job = appmod.InferenceJob.__new__(appmod.InferenceJob)
    r = _Robot()
    job.robot_tracks = {1: r}
    # conf 0.5 < 0.85 -> conf-weight 0.5, x source_weight 2.5 = 1.25
    appmod.InferenceJob._apply_bumper_read(
        job, 1, "x", ["254"], [], ["254"], source_weight=2.5)
    assert r.team_ocr_votes["254"] == 1.25


def test_no_team_read_is_noop(monkeypatch):
    monkeypatch.setattr(
        appmod, "_vision_read_bumpers",
        lambda crops, a, b, c: [{"team_number": None, "confidence": 0.0}],
    )
    job = appmod.InferenceJob.__new__(appmod.InferenceJob)
    r = _Robot()
    job.robot_tracks = {2: r}
    appmod.InferenceJob._apply_bumper_read(
        job, 2, "x", [], [], [], source_weight=2.5)
    assert r.team_ocr_votes is None  # untouched — no vote added


def test_missing_slot_is_noop(monkeypatch):
    called = {"n": 0}

    def _reader(crops, a, b, c):
        called["n"] += 1
        return [{"team_number": "9999", "confidence": 0.9}]

    monkeypatch.setattr(appmod, "_vision_read_bumpers", _reader)
    job = appmod.InferenceJob.__new__(appmod.InferenceJob)
    job.robot_tracks = {}
    appmod.InferenceJob._apply_bumper_read(
        job, 7, "x", ["9999"], [], [], source_weight=2.5)
    assert called["n"] == 0  # bailed before ever reading the crop

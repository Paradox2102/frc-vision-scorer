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

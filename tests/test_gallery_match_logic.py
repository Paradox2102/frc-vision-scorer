import numpy as np

from web.field.appearance_gallery import AppearanceGallery
from web.field.number_lock import ConsecutiveValueTracker
from web.app import _apply_gallery_vote, GALLERY_CONFIRM_SIM


class FakeTrack:
    def __init__(self, color, emb):
        self.alliance = color
        self.appearance_embedding = emb
        self.team_number = None
        self.team_number_confirmed = False
        self.team_ocr_votes = None
        self.consecutive_tracker = None


def _unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_strong_match_three_times_locks_identity():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("1787", "blue", e, 0.9)
    t = FakeTrack("blue", _unit(1, 0.01, 0))  # ~identical -> strong, no rival
    verdict = None
    for _ in range(3):
        verdict, team, sim = _apply_gallery_vote(t, g)
    assert verdict == "confirm"
    assert t.team_number_confirmed is True
    assert t.team_number == "1787"


def test_ambiguous_match_does_not_confirm():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    g.add("254", "blue", _unit(1, 0.02, 0), 0.9)  # two near-identical teams
    t = FakeTrack("blue", _unit(1, 0.01, 0))
    for _ in range(5):
        _apply_gallery_vote(t, g)
    assert t.team_number_confirmed is False  # gap guard prevents a confirm vote


def test_no_match_when_below_supplement():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    t = FakeTrack("blue", _unit(0, 1, 0))  # orthogonal -> weak
    verdict, team, sim = _apply_gallery_vote(t, g)
    assert verdict == "none"
    assert (t.team_ocr_votes or {}) == {}

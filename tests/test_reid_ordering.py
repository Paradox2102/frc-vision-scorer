import numpy as np

from web.reid import TrackView, best_candidate_score

APP_FLOOR, POS_GATE, W_POS, W_APP = 0.55, 200.0, 0.4, 0.6


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _score(det_pos, det_embed, tracks):
    return best_candidate_score(det_pos, det_embed, tracks, POS_GATE, W_POS, W_APP, APP_FLOOR)


def test_no_candidate_returns_negative():
    assert _score((0, 0), _e(1, 0, 0), []) == -1.0


def test_closer_and_more_similar_scores_higher():
    a = _e(1, 0, 0)
    tracks = [TrackView(1, "blue", (100, 100), a)]
    near = _score((100, 100), a, tracks)   # identical + on top
    far = _score((100, 290), a, tracks)    # same look, 190px away (< gate)
    assert near > far > 0.0


def test_vetoed_match_does_not_count():
    a = _e(1, 0, 0)
    tracks = [TrackView(1, "blue", (100, 100), a)]
    # Visually orthogonal detection right on the track → vetoed → no candidate.
    assert _score((100, 100), _e(0, 1, 0), tracks) == -1.0

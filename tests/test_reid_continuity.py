import numpy as np
import pytest

from web.reid import TrackView, choose_canonical_match

# Constants mirror app.py defaults; passed explicitly to keep the function pure.
APP_FLOOR = 0.55
POS_GATE = 200.0
W_POS, W_APP = 0.4, 0.6
CAP_PER_ALLIANCE = 3
CAP_TOTAL = 6


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _decide(det_pos, det_embed, det_alliance, tracks, claimed=None):
    return choose_canonical_match(
        det_pos=det_pos, det_embed=det_embed, det_alliance=det_alliance,
        tracks=tracks, app_floor=APP_FLOOR, pos_gate=POS_GATE,
        w_pos=W_POS, w_app=W_APP,
        cap_per_alliance=CAP_PER_ALLIANCE, cap_total=CAP_TOTAL,
        claimed=claimed,
    )


def test_ambiguous_position_picks_embedding_match():
    # Two blue tracks both ~near the detection; embedding matches track 2.
    a = _e(1, 0, 0)
    b = _e(0, 1, 0)
    tracks = [
        TrackView(robot_id=1, alliance="blue", position=(100, 100), embedding=a),
        TrackView(robot_id=2, alliance="blue", position=(110, 100), embedding=b),
    ]
    action, rid = _decide((105, 100), b, "blue", tracks)
    assert (action, rid) == ("merge", 2)


def test_below_floor_refuses_merge_and_creates():
    a = _e(1, 0, 0)
    far = _e(0, 0, 1)  # cosine ~0 with the only track, below floor
    tracks = [TrackView(robot_id=1, alliance="blue", position=(100, 100), embedding=a)]
    action, rid = _decide((101, 100), far, "blue", tracks)  # close in position
    assert action == "create"  # appearance veto beats proximity


def test_at_cap_assigns_best_embedding_not_nearest():
    a = _e(1, 0, 0)
    b = _e(0, 1, 0)
    c = _e(0, 0, 1)
    tracks = [
        TrackView(1, "blue", (100, 100), a),
        TrackView(2, "blue", (105, 100), b),
        TrackView(3, "blue", (110, 100), c),  # alliance full (3/3)
    ]
    # Detection is positionally nearest track 1 but visually identical to track 3.
    # The appearance veto rejects tracks 1 and 2, so it resolves to track 3 by
    # embedding rather than to the nearest track. Whether that surfaces as a
    # "merge" (passed the veto) or an "assign" (cap-full fallback) is an internal
    # detail — both map the detection to robot 3, which is what matters.
    action, rid = _decide((101, 100), c, "blue", tracks)
    assert rid == 3
    assert action in ("merge", "assign")


def test_track_without_embedding_uses_position():
    tracks = [TrackView(1, "blue", (100, 100), None)]
    action, rid = _decide((105, 100), _e(1, 0, 0), "blue", tracks)
    assert (action, rid) == ("merge", 1)  # neutral app_sim, position close enough


def test_empty_tracks_creates():
    action, rid = _decide((100, 100), _e(1, 0, 0), "red", [])
    assert action == "create"


def test_claimed_track_is_never_returned():
    # Detection is visually identical + positionally on track 1, but track 1 is
    # already claimed this frame. It must NOT be returned.
    a = _e(1, 0, 0)
    tracks = [
        TrackView(1, "blue", (100, 100), a),
        TrackView(2, "blue", (400, 400), _e(0, 1, 0)),
    ]
    action, rid = _decide((100, 100), a, "blue", tracks, claimed={1})
    assert rid != 1  # cannot merge into the claimed track


def test_claimed_best_match_falls_to_create_when_caps_allow():
    # Only one (claimed) track exists and caps allow more → split into a new track.
    a = _e(1, 0, 0)
    tracks = [TrackView(1, "blue", (100, 100), a)]
    action, rid = _decide((100, 100), a, "blue", tracks, claimed={1})
    assert action == "create"


def test_caps_full_and_all_claimed_drops():
    # 6 tracks (3/alliance) all claimed this frame → a 7th detection cannot get a
    # canonical and must be dropped.
    tracks = [TrackView(i, "blue" if i < 3 else "red", (i * 10, 0), _e(1, 0, 0))
              for i in range(6)]
    action, rid = _decide((5, 0), _e(1, 0, 0), "blue", tracks, claimed=set(range(6)))
    assert action == "drop"
    assert rid is None


def test_caps_full_assigns_to_unclaimed_only():
    # Alliance full (3 blue); two claimed. The forced assign must pick the one
    # UNCLAIMED blue track, never a claimed one.
    tracks = [
        TrackView(1, "blue", (100, 100), _e(1, 0, 0)),
        TrackView(2, "blue", (105, 100), _e(0, 1, 0)),
        TrackView(3, "blue", (110, 100), _e(0, 0, 1)),
    ]
    action, rid = _decide((101, 100), _e(1, 0, 0), "blue", tracks, claimed={1, 2})
    assert rid == 3
    assert action in ("merge", "assign")

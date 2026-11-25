"""Tests for the image-space TBD tracker with assignment margins.

Pure numpy / synthetic boxes; no GPU, no torch. The `_box` helper mirrors the
`_step`-style helpers in tests/test_tracker.py -- small synthetic axis-aligned
boxes in export coords.
"""
import math

from web.field.selective_prop import (
    SelectiveConfig,
    TbdTracker,
    expanded_iou,
)


def _box(cx, cy, w=40, h=40):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


# --------------------------------------------------------------------------
# Required cases (verbatim from the task brief).
# --------------------------------------------------------------------------
def test_two_separated_tracks_high_margin():
    trk = TbdTracker(SelectiveConfig())
    obs = trk.step(0, [(_box(100, 100), "red"), (_box(500, 100), "blue")])
    assert set(obs) == {1, 2}
    obs = trk.step(1, [(_box(105, 100), "red"), (_box(495, 100), "blue")])
    # far-apart robots: competitor cost is forbidden/absent -> margin inf
    assert obs[1].margin == float("inf")


def test_crossing_tracks_low_margin():
    trk = TbdTracker(SelectiveConfig())
    trk.step(0, [(_box(100, 100), None), (_box(160, 100), None)])
    obs = trk.step(1, [(_box(128, 100), None), (_box(132, 100), None)])
    # nearly-coincident detections: both assignments near-equal cost
    margins = [o.margin for o in obs.values()]
    assert all(m < 0.2 for m in margins)


def test_identity_continuity_and_new_oid():
    trk = TbdTracker(SelectiveConfig())
    trk.step(0, [(_box(100, 100), "red")])
    obs = trk.step(1, [(_box(110, 100), "red"), (_box(400, 300), "blue")])
    assert 1 in obs and obs[1].box[0] > 85
    assert 2 in obs  # new spawn


def test_gap_reappearance_recorded():
    cfg = SelectiveConfig(gap_trigger=3, max_missed=30)
    trk = TbdTracker(cfg)
    trk.step(0, [(_box(100, 100), None)])
    for f in range(1, 5):
        trk.step(f, [])              # 4 missed frames
    trk.step(5, [(_box(120, 100), None)])
    assert trk.gap_reappearances == [(1, 0, 5)]


def test_track_death_spawns_fresh_oid():
    cfg = SelectiveConfig(max_missed=2)
    trk = TbdTracker(cfg)
    trk.step(0, [(_box(100, 100), None)])
    for f in range(1, 4):
        trk.step(f, [])
    obs = trk.step(4, [(_box(100, 100), None)])
    assert set(obs) == {2}


def test_deterministic_replay():
    seq = [[(_box(100 + 5 * f, 100), "red"), (_box(200 - 5 * f, 100), "blue")]
           for f in range(20)]

    def run():
        trk = TbdTracker(SelectiveConfig())
        for f, dets in enumerate(seq):
            trk.step(f, dets)
        return {f: {o: (obs.box, obs.margin) for o, obs in fr.items()}
                for f, fr in trk.history.items()}

    assert run() == run()


# --------------------------------------------------------------------------
# Additional coverage (edge cases from the self-review checklist).
# --------------------------------------------------------------------------
def test_expanded_iou_bounds():
    b = _box(100, 100)
    # identical boxes -> IoU 1.0 regardless of the symmetric expansion
    assert math.isclose(expanded_iou(b, b, 0.25), 1.0)
    assert math.isclose(expanded_iou(b, b, 0.0), 1.0)
    # far-apart boxes -> even expanded they do not overlap
    assert expanded_iou(_box(100, 100), _box(500, 100), 0.25) == 0.0


def test_forbidden_pair_is_gated():
    # No overlap even after expansion -> cost is forbidden, assignment absent.
    cfg = SelectiveConfig()
    assert expanded_iou(_box(0, 0), _box(1000, 1000), cfg.iou_expand) == 0.0


def test_empty_det_list_no_crash_then_spawn():
    trk = TbdTracker(SelectiveConfig())
    assert trk.step(0, []) == {}          # empty frame, no tracks
    obs = trk.step(1, [(_box(100, 100), "red")])
    assert set(obs) == {1}
    assert trk.step(2, []) == {}          # track coasts, no obs emitted
    assert trk.history[2] == {}


def test_color_contradiction_penalty_flips_match():
    # A red track; frame 1 offers a geometrically perfect BLUE det and a
    # slightly-offset RED det. The +0.3 contradiction penalty must make the
    # tracker prefer the red det and spawn a fresh oid for the blue one.
    trk = TbdTracker(SelectiveConfig())
    trk.step(0, [(_box(100, 100), "red")])
    obs = trk.step(1, [(_box(100, 100), "blue"), (_box(105, 100), "red")])
    assert obs[1].color == "red"
    assert obs[1].box[0] > 80          # matched the offset red det, not the blue
    assert 2 in obs                    # blue det spawned a new track


def test_color_stickiness_first_known_color_sticks():
    # Track spawns colorless, then observes a known color which must stick and
    # thereafter drive the contradiction penalty.
    trk = TbdTracker(SelectiveConfig())
    trk.step(0, [(_box(100, 100), None)])          # oid1, color unknown
    trk.step(1, [(_box(102, 100), "red")])         # adopts red
    # frame 2: perfect blue det + offset colorless det. Sticky red penalizes
    # the blue, so the colorless det should win the continuation.
    obs = trk.step(2, [(_box(102, 100), "blue"), (_box(108, 100), None)])
    assert obs[1].box[0] > 82          # matched the colorless offset det
    assert 2 in obs                    # blue spawned fresh


def test_survives_exactly_max_missed_then_rematches():
    # die boundary is missed > max_missed: a track survives exactly max_missed
    # consecutive misses and re-matches under the same oid.
    cfg = SelectiveConfig(max_missed=2, gap_trigger=99)
    trk = TbdTracker(cfg)
    trk.step(0, [(_box(100, 100), None)])
    trk.step(1, [])
    trk.step(2, [])                     # missed == 2 == max_missed, still alive
    obs = trk.step(3, [(_box(100, 100), None)])
    assert set(obs) == {1}             # same track, no fresh spawn


def test_next_oid_and_history_tracking():
    trk = TbdTracker(SelectiveConfig())
    assert trk.next_oid == 1
    trk.step(0, [(_box(100, 100), "red"), (_box(400, 400), "blue")])
    assert trk.next_oid == 3           # two spawns consumed oid 1 and 2
    assert set(trk.history[0]) == {1, 2}


def test_custom_first_oid():
    trk = TbdTracker(SelectiveConfig(), first_oid=10)
    obs = trk.step(0, [(_box(100, 100), "red")])
    assert set(obs) == {10}
    assert trk.next_oid == 11


def test_spawned_track_margin_is_inf():
    trk = TbdTracker(SelectiveConfig())
    obs = trk.step(0, [(_box(100, 100), "red"), (_box(500, 500), "blue")])
    assert obs[1].margin == float("inf")
    assert obs[2].margin == float("inf")


def test_gap_not_recorded_below_trigger():
    cfg = SelectiveConfig(gap_trigger=5, max_missed=30)
    trk = TbdTracker(cfg)
    trk.step(0, [(_box(100, 100), None)])
    for f in range(1, 4):
        trk.step(f, [])                # only 3 missed frames (< trigger)
    trk.step(4, [(_box(112, 100), None)])
    assert trk.gap_reappearances == []


def test_simultaneous_death_and_spawn():
    # An old track dies (exceeds max_missed) on the very frame a far-away, never
    # matchable detection spawns a fresh track. Both must resolve independently.
    cfg = SelectiveConfig(max_missed=1, gap_trigger=99)
    trk = TbdTracker(cfg)
    trk.step(0, [(_box(100, 100), "red")])     # oid1
    trk.step(1, [])                            # oid1 missed == 1 (alive)
    # frame 2: oid1 would reach missed 2 (> max_missed -> dies); a far det spawns.
    obs = trk.step(2, [(_box(900, 900), "blue")])
    assert set(obs) == {2}                     # only the fresh spawn is emitted
    assert 1 not in trk.history[2]
    # oid1 is gone: a later detection back at its old spot spawns yet another oid.
    obs = trk.step(3, [(_box(100, 100), "red")])
    assert set(obs) == {3}


def test_constant_velocity_prediction_tracks_mover():
    # A steadily-moving robot: constant-velocity prediction should keep the
    # single track (oid 1) matched every frame even at a stride the raw
    # last-box IoU alone would miss.
    trk = TbdTracker(SelectiveConfig())
    trk.step(0, [(_box(100, 100), "red")])
    trk.step(1, [(_box(130, 100), "red")])
    trk.step(2, [(_box(160, 100), "red")])
    obs = trk.step(3, [(_box(190, 100), "red")])
    assert set(obs) == {1}             # never lost, never re-spawned
    assert trk.next_oid == 2


# --------------------------------------------------------------------------
# det_conf payload: optional 4th tuple slot, copied onto FrameObs only.
# --------------------------------------------------------------------------
def test_det_conf_rides_fourth_slot_onto_frame_obs():
    a, b = TbdTracker(SelectiveConfig()), TbdTracker(SelectiveConfig())
    for f, dx in enumerate((0, 5, 10)):
        plain = a.step(f, [(_box(100 + dx, 100), "red"), (_box(500, 100), "blue")])
        conf = b.step(f, [(_box(100 + dx, 100), "red", None, 0.87),
                          (_box(500, 100), "blue", None, None)])
        # Matching is identical with and without the confidence slot.
        assert {o: (v.box, v.margin, v.color) for o, v in plain.items()} == \
            {o: (v.box, v.margin, v.color) for o, v in conf.items()}
        assert all(v.conf is None for v in plain.values())
        assert conf[1].conf == 0.87          # spawned (f=0) and matched (f>0)
        assert conf[2].conf is None


def test_det_conf_non_finite_becomes_none():
    trk = TbdTracker(SelectiveConfig())
    obs = trk.step(0, [(_box(100, 100), "red", None, float("nan"))])
    assert obs[1].conf is None

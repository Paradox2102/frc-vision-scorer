"""Per-segment track resolution: split a slot's track at suspicious
reacquisitions, classify each segment, and drop phantom/alien segments
(do-no-harm). Pure functions, no GPU."""
from web.field.track_segments import (
    split_at_jumps,
    classify_segment,
    resolve_segments,
    slot_dropped_ranges,
    filter_paths_by_dropped,
)


# --------------------------------------------------------------------------- #
# split_at_jumps
# --------------------------------------------------------------------------- #
def test_uniform_track_is_one_segment():
    s = [(0.0, 1.0, 1.0), (0.13, 1.1, 1.0), (0.26, 1.2, 1.0)]
    assert split_at_jumps(s) == [(0, 3)]


def test_gap_and_jump_splits():
    # 0.13 s cadence, then a 3 s gap AND a 6 m jump between idx 1 and 2.
    s = [(0.0, 1.0, 1.0), (0.13, 1.1, 1.0), (3.2, 7.0, 5.0), (3.33, 7.1, 5.0)]
    assert split_at_jumps(s, max_gap_s=1.0, jump_m=2.5) == [(0, 2), (2, 4)]


def test_gap_without_jump_does_not_split():
    # 3 s gap but the robot reappears ~0.3 m away -> same robot, no split.
    s = [(0.0, 1.0, 1.0), (3.2, 1.3, 1.0), (3.33, 1.4, 1.0)]
    assert split_at_jumps(s, max_gap_s=1.0, jump_m=2.5) == [(0, 3)]


def test_teleport_without_gap_splits():
    # The identity log is DENSE (a row every frame, even while coasting), so a
    # wrong reacquisition shows as a big spatial step at tiny dt (impossible
    # speed) rather than a time gap. That must still split.
    s = [(0.0, 1.0, 1.0), (0.13, 7.0, 5.0), (0.26, 7.1, 5.0)]  # 7.2 m in 0.13 s
    assert split_at_jumps(s, max_gap_s=1.0, jump_m=2.5) == [(0, 1), (1, 3)]


def test_continuous_fast_move_does_not_split():
    # A real robot at ~4 m/s moves ~0.5 m per dense frame — under jump_m, no split.
    s = [(0.0, 1.0, 1.0), (0.13, 1.5, 1.0), (0.26, 2.0, 1.0), (0.39, 2.5, 1.0)]
    assert split_at_jumps(s, max_gap_s=1.0, jump_m=2.5) == [(0, 4)]


def test_empty_and_single():
    assert split_at_jumps([]) == []
    assert split_at_jumps([(1.0, 2.0, 3.0)]) == [(0, 1)]


# --------------------------------------------------------------------------- #
# classify_segment
# --------------------------------------------------------------------------- #
def _entry(frame, x, y, *, confirmed=False, person=False):
    return {"frame_num": frame, "pos": [x, y], "confirmed": confirmed,
            "person_overlap": person}


def test_static_unconfirmed_segment_is_static():
    entries = [_entry(f, 15.0, 2.4) for f in range(12)]
    assert classify_segment(entries) == "static"


def test_team_read_segment_is_robot_even_if_static():
    entries = [_entry(f, 15.0, 2.4, confirmed=(f == 6)) for f in range(12)]
    assert classify_segment(entries) == "robot"


def test_moving_segment_is_robot():
    entries = [_entry(f, 9.0 + 0.3 * f, 5.0) for f in range(12)]  # ~3.3 m travel
    assert classify_segment(entries) == "robot"


def test_person_overlap_segment_is_human():
    entries = [_entry(f, 8.0, 3.0, person=True) for f in range(12)]
    assert classify_segment(entries) == "human"


# --------------------------------------------------------------------------- #
# resolve_segments (do-no-harm)
# --------------------------------------------------------------------------- #
def test_drop_static_prefix_before_robot():   # R3: phantom then real robot
    assert resolve_segments(["static", "robot"]) == [False, True]


def test_drop_static_excursion_between_robots():  # B2: robot, parked excursion, robot
    assert resolve_segments(["robot", "static", "robot"]) == [True, False, True]


def test_single_segment_always_kept():
    assert resolve_segments(["static"]) == [True]
    assert resolve_segments(["robot"]) == [True]


def test_all_robot_segments_kept_intact():   # ambiguous same-robot -> do-no-harm
    assert resolve_segments(["robot", "robot"]) == [True, True]


def test_whole_slot_non_robot_kept_here():   # slot-level verdict handles this, not us
    assert resolve_segments(["static", "static"]) == [True, True]


def test_drop_human_segment_in_robot_slot():
    assert resolve_segments(["robot", "human"]) == [True, False]


# --------------------------------------------------------------------------- #
# slot_dropped_ranges
# --------------------------------------------------------------------------- #
def _log(frame, slot, x, y, *, confirmed=False, person=False):
    return {"frame_num": frame, "slot_id": slot, "pos": [x, y],
            "confirmed": confirmed, "person_overlap": person}


def test_slot_dropped_ranges_flags_phantom_prefix():
    # slot 5 (R3): 10 frames parked static at x=15 (frames 0..9, 60 fps),
    # then a >1 s gap + big jump, then 12 frames moving (frames 120..131).
    log = [_log(f, 5, 15.0, 2.4) for f in range(10)]
    log += [_log(120 + i, 5, 9.0 + 0.3 * i, 5.0) for i in range(12)]
    dropped = slot_dropped_ranges(log, video_fps=60.0)
    assert dropped == {5: [(0, 9)]}   # the static prefix range is dropped


def test_slot_dropped_ranges_dense_log_teleport_prefix():
    # Mirrors the real identity log: DENSE (consecutive ~0.13 s frames, no time
    # gap). Slot 5 is a static phantom for frames 0..72, then teleports (~6.5 m
    # in one frame -> impossible speed) onto a moving robot. Prefix must drop.
    log = [_log(f, 5, 15.0, 2.4) for f in range(0, 80, 8)]            # 10 static
    log += [_log(80 + 8 * i, 5, 9.0 + 0.3 * i, 5.0) for i in range(12)]  # moving
    dropped = slot_dropped_ranges(log, video_fps=60.0)
    assert dropped == {5: [(0, 72)]}


def test_slot_dropped_ranges_empty_for_clean_slot():
    log = [_log(f, 3, 9.0 + 0.3 * f, 5.0) for f in range(20)]  # one moving robot
    assert slot_dropped_ranges(log, video_fps=60.0) == {}


def test_slot_dropped_ranges_empty_for_wholly_static_slot():
    log = [_log(f, 2, 15.0, 2.4) for f in range(20)]  # never splits, one segment
    assert slot_dropped_ranges(log, video_fps=60.0) == {}


# --------------------------------------------------------------------------- #
# filter_paths_by_dropped
# --------------------------------------------------------------------------- #
def test_filter_paths_removes_dropped_frames():
    paths = {"R3": [{"frame_num": f, "timestamp": f / 60.0, "field_pos": [15.0, 2.4]}
                    for f in range(10)]
             + [{"frame_num": 120 + i, "timestamp": (120 + i) / 60.0,
                 "field_pos": [9.0 + 0.3 * i, 5.0]} for i in range(12)]}
    out = filter_paths_by_dropped(paths, {5: [(0, 9)]})   # slot_id 5 == "R3"
    frames = [s["frame_num"] for s in out["R3"]]
    assert min(frames) == 120 and len(frames) == 12   # static prefix gone


def test_filter_paths_untouched_when_no_drops():
    paths = {"B1": [{"frame_num": 1, "timestamp": 0.0, "field_pos": [1.0, 1.0]}]}
    assert filter_paths_by_dropped(paths, {}) == paths

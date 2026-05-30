from web.field.spike_metrics import count_id_switches, continuity_stats


def test_count_id_switches_detects_a_swap():
    # Two tracks, two labels. Anchors pin label "A" and "B" at known points.
    # tracker_frames[frame] = {track_id: (x, y)}
    tracker_frames = {
        0: {1: (0.0, 0.0), 2: (10.0, 0.0)},
        1: {1: (1.0, 0.0), 2: (9.0, 0.0)},
        2: {1: (9.0, 0.0), 2: (1.0, 0.0)},  # tracks crossed but kept ids
    }
    anchors = [
        (0, (0.0, 0.0), "A"), (0, (10.0, 0.0), "B"),
        (2, (9.0, 0.0), "A"), (2, (1.0, 0.0), "B"),
    ]
    # Track 1 holds A throughout (0,0)->(9,0); track 2 holds B. No switch.
    assert count_id_switches(tracker_frames, anchors) == 0


def test_count_id_switches_counts_a_real_swap():
    tracker_frames = {
        0: {1: (0.0, 0.0), 2: (10.0, 0.0)},
        2: {1: (1.0, 0.0), 2: (9.0, 0.0)},  # tracks stayed put; the robots crossed under them
    }
    anchors = [
        (0, (0.0, 0.0), "A"), (0, (10.0, 0.0), "B"),
        (2, (9.0, 0.0), "A"), (2, (1.0, 0.0), "B"),  # A is now on the RIGHT, B on the LEFT
    ]
    # Counting is per-label. Robot A's nearest track flips 1->2 (one switch) and
    # robot B's flips 2->1 (a second switch). A single physical crossing-swap is
    # counted once per involved robot, so the total is 2. Both trackers are scored
    # identically, so the ratio between them stays meaningful.
    assert count_id_switches(tracker_frames, anchors) == 2


def test_continuity_stats_counts_births_and_deaths():
    tracker_frames = {
        0: {1: (0, 0)},
        1: {1: (0, 0), 2: (5, 5)},  # 2 born
        2: {2: (5, 5)},             # 1 died
    }
    s = continuity_stats(tracker_frames)
    assert s["n_tracks"] == 2
    assert s["births"] == 2 and s["deaths"] == 2

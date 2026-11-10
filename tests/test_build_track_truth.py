"""Pure parts of scripts/build_track_truth.py (no video, DB or replay files)."""
from scripts.build_track_truth import (
    activity_segment,
    associate,
    box_anchor,
    cascade,
    clip_segment,
    cut_runs,
    decide,
    gt_pick,
    interp_keyframes,
    positions_at,
    reader_tag,
    run_at,
    slot_tracks,
)

FPS = 60.0


def _frames(rows):
    """rows: [(frame_num, [(slot, x, y), ...])] -> replay frames."""
    return [{"frame_num": fn, "robots": [{"slot": s, "field_pos": [x, y]} for s, x, y in rs]}
            for fn, rs in rows]


def test_slot_tracks_spacing_and_duplicate_slots():
    frames = _frames([(0, [("B1", 1, 1), ("B1", 5, 5)]), (7, [("B1", 1.1, 1)]), (14, [("R1", 9, 9)])])
    frames.append({"frame_num": 21, "robots": [{"slot": "R1", "field_pos": None}]})
    tracks, spacing = slot_tracks(frames)
    assert spacing == 7.0
    assert tracks["B1"] == [(0, 1.0, 1.0), (7, 1.1, 1.0)]
    assert tracks["B1#2"] == [(0, 5.0, 5.0)]
    assert tracks["R1"] == [(14, 9.0, 9.0)]          # None position skipped


def test_cut_runs_at_gaps_and_jumps():
    seq = [(0, 0.0, 0.0), (7, 0.1, 0.0), (14, 0.2, 0.0),     # run 1
           (42, 0.3, 0.0),                                  # gap 28 > 2.5*7 -> run 2
           (49, 3.3, 0.0),                                  # 3 m in 7/60 s -> jump -> run 3
           (56, 3.4, 0.0)]
    runs = cut_runs(seq, 7.0, FPS)
    assert [[p[0] for p in r] for r in runs] == [[0, 7, 14], [42], [49, 56]]
    assert run_at(runs, 10, 7.0)[0][0] == 0
    assert run_at(runs, 52, 7.0)[0][0] == 49
    assert run_at(runs, 30, 7.0) is None
    assert clip_segment(runs[0], 7, 5) == (2, 12)
    assert clip_segment(runs[0], 7, 600) == (0, 14)


def test_positions_at_interpolates_and_respects_gaps():
    tracks = {"B1": [(0, 0.0, 0.0), (7, 7.0, 0.0)], "R1": [(0, 5.0, 5.0)], "R2": [(0, 1.0, 1.0), (70, 2.0, 2.0)]}
    p = positions_at(tracks, 3.5, 7.0)
    assert p["B1"] == (3.5, 0.0)
    assert p["R1"] == (5.0, 5.0)                      # within half a spacing of its only sample
    assert "R2" not in p or p["R2"] == (1.0, 1.0)
    assert "R1" not in positions_at(tracks, 6.0, 7.0)  # too far from its only sample
    assert "R2" not in positions_at(tracks, 35, 7.0)   # bracket spans a gap


def test_associate_gate_and_margin():
    pos = {"B1": (0.0, 0.0), "B2": (3.0, 0.0)}
    assert associate((0.2, 0.0), pos) == ("B1", 0.2)
    slot, d = associate((1.4, 0.0), pos)                  # outside the 1 m gate
    assert slot is None and abs(d - 1.4) < 1e-9
    close = {"B1": (0.0, 0.0), "B2": (0.6, 0.0)}
    assert associate((0.2, 0.0), close)[0] is None        # runner-up within 0.5 m margin
    assert associate(None, pos) == (None, None)


def test_activity_segment_grows_until_gap_or_span():
    act = [100, 160, 220, 600, 660]
    assert activity_segment(act, 160, gap_f=120, half_span_f=600) == (100, 220)
    assert activity_segment(act, 660, gap_f=120, half_span_f=600) == (600, 660)
    assert activity_segment(act, 160, gap_f=120, half_span_f=30) == (160, 160)
    assert activity_segment([], 50, 120, 600) == (50, 50)


def test_decide_excludes_own_frame_and_dedupes_jobs():
    # 3 jobs reading the crop's own frame and 2 more frames: not enough
    votes = [(1000, "254")] * 3 + [(1100, "254"), (1100, "254"), (1200, "254")]
    d = decide(votes, 1000, FPS)
    assert d["team"] is None and d["votes"] == {"254": 2} and d["reason"] == "top_votes<3"


def test_decide_rules():
    ok = [(0, "254"), (100, "254"), (200, "254"), (300, "1678")]
    d = decide(ok, 1000, FPS)
    assert d["team"] == "254" and d["share"] == 0.75 and d["reason"] == "ok"
    assert d["n_frames"] == 4
    impure = ok + [(400, "1678")]
    assert decide(impure, 1000, FPS)["reason"] == "share<0.75"
    narrow = [(0, "254"), (20, "254"), (40, "254")]      # spans 0.67 s
    assert decide(narrow, 1000, FPS)["reason"] == "span<1.0s"
    tie = [(0, "254"), (100, "254"), (200, "1678"), (300, "1678")]
    assert decide(tie, 1000, FPS)["reason"] == "tie"
    assert decide([], 1000, FPS)["reason"] == "no_votes"
    # two teams read at the same frame both keep their vote
    same_frame = [(0, "254"), (0, "1678"), (100, "254"), (200, "254"), (300, "254")]
    assert decide(same_frame, 1000, FPS)["votes"] == {"254": 4, "1678": 1}


def test_interp_keyframes():
    kfs = [{"frame": 0, "state": "visible", "uv": [0, 0]},
           {"frame": 10, "state": "visible", "uv": [10, 20]},
           {"frame": 12, "state": "unknown"},
           {"frame": 20, "state": "estimated", "uv": [20, 20]},
           {"frame": 100, "state": "visible", "uv": [0, 0]}]
    assert interp_keyframes(kfs, 5, "uv") == ((5.0, 10.0), "visible")
    assert interp_keyframes(kfs, 15, "uv") == ((15.0, 20.0), "estimated")
    assert interp_keyframes(kfs, 10, "uv") == ((10, 20), "visible")
    assert interp_keyframes(kfs, 50, "uv") == (None, None)     # gap > 30 frames
    assert interp_keyframes(kfs, -1, "uv") == (None, None)


def test_gt_pick_gate_and_margin():
    assert gt_pick({"254": 20.0, "1678": 200.0}, box_w=60) == "254"
    assert gt_pick({"254": 90.0, "1678": 400.0}, box_w=60) is None    # beyond 1 box width
    assert gt_pick({"254": 20.0}, box_w=10) == "254"                  # 25 px floor
    assert gt_pick({"254": 40.0, "1678": 50.0}, box_w=60) is None     # runner-up too close
    assert gt_pick({}, box_w=60) is None


def test_reader_tag_and_anchor():
    assert reader_tag({"jev_probs": {"254": 0.9}}) == "jev"
    assert reader_tag({"clef_probs": {"a": 1}, "jev_probs": None}) == "clef"
    assert reader_tag({"clef_probs": None}, "2026-07-01T12:00:00") == "qwen3-vl-4b"
    assert reader_tag({}, "2026-11-01T00:00:00") == "local-vlm(model unrecorded)"
    assert box_anchor((10, 20, 30, 40)) == (25.0, 60.0)


def test_cascade_widens_only_for_missing_evidence():
    calls = []

    def at(results):
        def f(hs):
            calls.append(hs)
            return results[hs]
        return f

    few = {"team": None, "reason": "top_votes<3"}
    ok = {"team": "254", "reason": "ok"}
    impure = {"team": None, "reason": "share<0.75"}
    assert cascade((5, 10), at({5: few, 10: ok})) == (ok, 10)
    calls.clear()
    assert cascade((5, 10), at({5: impure, 10: ok})) == (impure, 5) and calls == [5]
    assert cascade((5, 10), at({5: ok, 10: impure})) == (ok, 5)
    assert cascade((5, 10), at({5: few, 10: few})) == (few, 10)

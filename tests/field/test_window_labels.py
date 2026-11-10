"""Window labels: pure logic (windows, decisions, assembly) and the
orchestration with a fake reader and a fake frame source. No network."""
import pytest

from web.field import window_labels as wl

FPS = 60.0
STEP = 12          # source frames per processed frame -> 5 processed fps


def _frames(tracks, n, step=STEP):
    """tracks: {slot: {index: (x, y)}} -> replay frames (frame_num = 420 + index * step)."""
    out = []
    for i in range(n):
        robots = [{"slot": s, "alliance": "blue" if s.startswith("B") else "red",
                   "team_number": "9999", "confirmed": False, "field_pos": list(p[i])}
                  for s, p in tracks.items() if i in p]
        out.append({"frame_num": 420 + i * step, "robots": robots})
    return out


def _still(indices, xy=(3.0, 2.0)):
    return {i: xy for i in indices}


def test_frame_period_from_median_step():
    assert wl.frame_period_s(_frames({"B1": _still(range(5))}, 5), FPS) == pytest.approx(0.2)
    assert wl.frame_period_s([], FPS) is None
    assert wl.frame_period_s(_frames({"B1": _still(range(5))}, 5), 0) is None


def test_long_run_splits_into_even_windows_of_at_most_4s():
    ws = wl.build_windows(_frames({"B1": _still(range(50))}, 50), FPS)     # 10 s
    assert [len(w.idx) for w in ws] == [17, 17, 16]
    assert ws[0].idx[0] == 0 and ws[-1].idx[-1] == 49
    assert all(len(w.samples) == 4 for w in ws)
    assert ws[0].samples[0] == ws[0].idx[0] and ws[0].samples[-1] == ws[0].idx[-1]
    assert len({w.run for w in ws}) == 1 and [w.wid for w in ws] == [0, 1, 2]


def test_exactly_4s_is_one_window_and_one_more_sample_makes_two():
    assert len(wl.build_windows(_frames({"B1": _still(range(20))}, 20), FPS)) == 1
    assert len(wl.build_windows(_frames({"B1": _still(range(21))}, 21), FPS)) == 2


def test_gap_longer_than_two_and_a_half_periods_starts_a_new_run():
    present = list(range(10)) + list(range(13, 23))          # 3 frames missing
    ws = wl.build_windows(_frames({"B1": _still(present)}, 23), FPS)
    assert len(ws) == 2 and ws[0].run != ws[1].run
    assert ws[0].idx == list(range(10)) and ws[1].idx == list(range(13, 23))


def test_one_missing_frame_does_not_split():
    present = [i for i in range(12) if i != 5]
    ws = wl.build_windows(_frames({"B1": _still(present)}, 12), FPS)
    assert len(ws) == 1 and ws[0].idx == present


def test_jump_faster_than_7_5_mps_starts_a_new_run():
    track = {i: (3.0, 2.0) for i in range(6)}
    track.update({i: (8.0, 2.0) for i in range(6, 12)})       # 5 m in 0.2 s
    ws = wl.build_windows(_frames({"R1": track}, 12), FPS)
    assert [w.idx for w in ws] == [list(range(6)), list(range(6, 12))]


def test_single_sample_runs_are_dropped_and_slots_are_independent():
    tracks = {"B1": _still([5] + list(range(20, 30))), "R2": _still(range(4))}
    ws = wl.build_windows(_frames(tracks, 30), FPS)
    assert [(w.slot, w.idx) for w in ws] == [("B1", list(range(20, 30))), ("R2", [0, 1, 2, 3])]


def test_two_sample_window_uses_both_as_samples_and_no_frames_gives_no_windows():
    ws = wl.build_windows(_frames({"B1": _still([0, 1])}, 2), FPS)
    assert ws[0].samples == [0, 1]
    assert wl.build_windows([], FPS) == []

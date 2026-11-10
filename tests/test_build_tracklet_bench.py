"""Pure parts of scripts/build_tracklet_bench.py (no video files)."""
import numpy as np

from scripts.build_tracklet_bench import (
    candidate_ks,
    first_nonzero_frame,
    localize,
    n_steps,
    sample_step,
    search_window,
    touches_edge,
    track,
)


def test_candidate_ks_from_stored_dims():
    # 384x416 read crop: native 48x52 (k=8), 96x104 (k=4), 192x208 (k=2) or as-is.
    assert candidate_ks(384, 416, "matched") == [2, 4, 8, 1]
    # 49x37 filtered crop: never upscaled.
    assert candidate_ks(49, 37, "filtered") == [1]
    # Below 384 tall can only be... nothing consistent -> brute-force fallback.
    assert candidate_ks(101, 383, "null") == list(range(1, 17))
    for w, h in [(816, 426), (600, 405), (1476, 396)]:
        for k in candidate_ks(w, h, "matched"):
            assert w % k == 0 and h % k == 0


def test_sampling_grid():
    assert sample_step(59.716) == 5 and sample_step(60.0) == 5 and sample_step(30.0) == 2
    assert n_steps(60.0, 5) == 18 and n_steps(59.716, 5) == 17


def test_search_window_and_edges():
    assert search_window((100, 50, 40, 20), (1920, 1080)) == (60, 30, 180, 90)
    assert search_window((10, 5, 40, 20), (1920, 1080)) == (0, 0, 90, 45)
    assert touches_edge((0, 10, 5, 5), (100, 100)) and not touches_edge((1, 1, 5, 5), (100, 100))


def _scene(n, step, dx):
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 60, size=(240, 320, 3), dtype=np.uint8)
    obj = rng.integers(0, 255, size=(20, 40, 3), dtype=np.uint8)
    frames = {}
    for i in range(-n, n + 1):
        f = bg.copy()
        x = 140 + i * dx
        f[100:120, x:x + 40] = obj
        frames[i * step] = f
    return frames, obj


def test_track_follows_moving_patch_both_ways():
    frames, _ = _scene(4, 5, 6)
    fwd, stop_f = track(frames, 0, (140, 100, 40, 20), 5, 4, +1, 0.5)
    bwd, stop_b = track(frames, 0, (140, 100, 40, 20), 5, 4, -1, 0.5)
    assert [d["box"][0] for d in fwd] == [146, 152, 158, 164] and stop_f == "window_end"
    assert [d["box"][0] for d in bwd] == [134, 128, 122, 116] and stop_b == "window_end"
    assert all(d["match_score"] > 0.95 for d in fwd + bwd)


def test_track_stops_on_low_score_and_anchor():
    frames, _ = _scene(2, 5, 6)
    frames[10] = np.zeros_like(frames[10])  # object vanishes -> flat frame
    fwd, stop = track(frames, 0, (140, 100, 40, 20), 5, 4, +1, 0.5)
    assert len(fwd) == 1 and stop == "low_score"
    fwd2, stop2 = track(frames, 0, (140, 100, 40, 20), 5, 1, +1, 0.5, anchor_min=1.01)
    assert fwd2 == [] and stop2 == "anchor"
    _, stop3 = track(frames, 0, (140, 100, 40, 20), 5, 4, -1, 0.5)
    assert stop3 == "video_end"


def test_localize_finds_crop_and_frame():
    frames, obj = _scene(2, 1, 6)
    import cv2
    stored = cv2.resize(obj, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)  # k=4 upscale
    loc = localize(stored, [2, 4], frames)
    assert loc["k"] == 4 and loc["score"] > 0.9
    x, y, w, h = loc["box"]
    assert (y, w, h) == (100, 40, 20) and x == 140 + loc["frame_idx"] * 6


def test_first_nonzero_frame(tmp_path):
    p = tmp_path / "diagnostic_x.jsonl"
    p.write_text('{"frame": 0, "type": "vlm_filtered"}\n{"frame": 0}\n{"frame": 390, "t": 1}\n')
    assert first_nonzero_frame(p) == 390
    assert first_nonzero_frame(tmp_path / "missing.jsonl") is None

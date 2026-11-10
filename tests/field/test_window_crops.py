"""Window-label crops: target projection, tile geometry, mosaics. No video
file is needed: the frame reader is exercised on a tiny synthetic clip."""
import cv2
import numpy as np
import pytest

from web.field import window_crops as wc


class FakeCal:
    """Linear camera: 100 px per metre in x, 60 px per metre in y (y up)."""
    camera_center = np.array([8.0, -5.0, 3.0])

    def field_to_image(self, p):
        return (100.0 * p[0], 700.0 - 60.0 * p[1])


def test_target_pushes_half_a_robot_away_from_the_camera_and_sizes_the_crop():
    u, v, side = wc.target(FakeCal(), (8.0, 2.0))
    # straight ahead of the camera (x equal): pushed +0.475 m in y only
    assert u == pytest.approx(800.0) and v == pytest.approx(700.0 - 60.0 * 2.475)
    # projected 0.95 m bumper: 95 px in x, 57 px in y -> larger span 95
    assert side == round(150 + 0.5 * 95)


def test_target_uses_only_the_remaining_offset_when_positions_are_already_pushed():
    _, v0, _ = wc.target(FakeCal(), (8.0, 2.0), ground_center_offset_m=0.475)
    assert v0 == pytest.approx(700.0 - 60.0 * 2.0)
    _, v1, _ = wc.target(FakeCal(), (8.0, 2.0), ground_center_offset_m=0.2)
    assert v1 == pytest.approx(700.0 - 60.0 * 2.275)


def test_target_side_is_clamped_and_bad_projection_returns_none():
    class Wide(FakeCal):
        def field_to_image(self, p):
            return (2000.0 * p[0], 700.0)

    assert wc.target(Wide(), (1.0, 2.0))[2] == wc.CROP_MAX_PX

    class Broken(FakeCal):
        def field_to_image(self, p):
            raise ValueError("behind camera")

    assert wc.target(Broken(), (1.0, 2.0)) is None
    assert wc.in_view(100, 700, (1920, 723)) and wc.in_view(100, 760, (1920, 723))
    assert not wc.in_view(-5, 300, (1920, 723)) and not wc.in_view(100, 800, (1920, 723))


def _view(h=723, w=1920):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = 40
    return img


def test_wide_tile_puts_the_target_at_50_62_and_draws_ticks():
    tile, (tx, ty) = wc.wide_tile(_view(), 960.0, 400.0, 200)
    assert tile.shape == (240, 240, 3)
    assert tx == 120 and ty == int(0.62 * 240)
    assert tuple(tile[1, 120]) == (255, 255, 255)             # top tick
    assert tuple(tile[ty, 2]) == (255, 255, 255)              # left tick
    assert tuple(tile[120, 60]) == (0, 40, 0)                 # interior untouched


def test_wide_tile_clamps_at_the_view_edge_and_reports_the_shifted_target():
    tile, (tx, ty) = wc.wide_tile(_view(), 20.0, 700.0, 200)
    assert tile.shape == (240, 240, 3)
    assert tx == int(20 / 200 * 240)                          # crop starts at x = 0
    assert ty == int((700 - (723 - 200)) / 200 * 240)         # crop ends at the view bottom


def _white_ticks(tile, tx, ty):
    """Which of the four ticks are drawn, probed just inside each edge at the
    (clamped) tick coordinate."""
    cx = min(max(tx, 0), 239)
    cy = min(max(ty, 0), 239)
    white = (255, 255, 255)
    return {"top": tuple(tile[1, cx]) == white, "bottom": tuple(tile[238, cx]) == white,
            "left": tuple(tile[cy, 1]) == white, "right": tuple(tile[cy, 238]) == white}


@pytest.mark.parametrize("u,v", [(960.0, 723.0 + 30.0),    # 30 px below the view bottom
                                 (1920.0, 400.0)])         # u equal to the view width
def test_all_four_ticks_are_drawn_when_the_target_is_just_outside_the_view(u, v):
    assert wc.in_view(u, v, (1920, 723))
    tile, (tx, ty) = wc.wide_tile(_view(), u, v, 200)
    assert tx >= 240 or ty >= 240                             # true target is off the tile
    assert _white_ticks(tile, tx, ty) == {"top": True, "bottom": True,
                                          "left": True, "right": True}


def test_zoom_send_grid_and_lineup_sizes():
    tile = np.full((240, 240, 3), 77, np.uint8)
    assert wc.zoom_tile(tile).shape == (504, 504, 3)
    assert wc.send_wide(tile).shape == (480, 480, 3)
    g = wc.grid_image([tile, tile, tile])
    assert g.shape == (960, 960, 3)
    assert tuple(g[700, 700]) == (128, 128, 128)              # empty fourth cell is grey
    card = wc.lineup_card([("A", [tile] * 6), ("B", [tile] * 3)])
    assert card.shape == (480, 96 + 6 * 240, 3)
    six = wc.lineup_card([(l, [tile] * 6) for l in "ABCDEF"])
    assert six.shape == (1440, 1536, 3) and max(six.shape[:2]) <= 1568
    jpg = wc.jpeg(tile)
    assert jpg[:2] == b"\xff\xd8" and cv2.imdecode(np.frombuffer(jpg, np.uint8), 1).shape == (240, 240, 3)


def test_iter_view_frames_reads_only_wanted_frames_in_order(tmp_path):
    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(12):
        vw.write(np.full((48, 64, 3), i * 20, np.uint8))
    vw.release()
    got = list(wc.iter_view_frames(path, [7, 2, 2, 40], (8, 4, 32, 24)))
    assert [n for n, _ in got] == [2, 7]
    assert got[0][1].shape == (24, 32, 3)
    assert abs(int(got[1][1][0, 0, 0]) - 140) <= 6            # frame 7 has value 140
    assert list(wc.iter_view_frames(path, [], (0, 0, 64, 48))) == []
    with pytest.raises(OSError):
        list(wc.iter_view_frames(str(tmp_path / "missing.mp4"), [1], (0, 0, 64, 48)))


def test_lineup_card_colour_bar_only_when_asked():
    tile = np.full((240, 240, 3), 77, np.uint8)
    rows = [("A", [tile] * 2), ("B", [tile])]
    plain = wc.lineup_card(rows)
    assert np.array_equal(plain, wc.lineup_card(rows, colors=None))
    bar = wc.lineup_card(rows, colors=["red", "blue"])
    y, x = (wc.BAR_ROWS[0] + wc.BAR_ROWS[1]) // 2, wc.LABEL_W // 2
    assert tuple(int(c) for c in bar[y, x]) == wc.COLOUR_BGR["red"]
    assert tuple(int(c) for c in bar[240 + y, x]) == wc.COLOUR_BGR["blue"]
    assert tuple(int(c) for c in plain[y, x]) == wc._DARK
    assert np.array_equal(bar[:, wc.LABEL_W:], plain[:, wc.LABEL_W:])   # only label cells change
    one = wc.lineup_card(rows, colors=["red", None])                     # unknown colour: no bar
    assert np.array_equal(one[240:], plain[240:])


def test_decode_tile_round_trips_resizes_and_rejects_junk():
    back = wc.decode_tile(wc.jpeg(np.full((240, 240, 3), 90, np.uint8)))
    assert back.shape == (240, 240, 3) and abs(int(back[120, 120, 0]) - 90) <= 2
    assert wc.decode_tile(wc.jpeg(np.zeros((100, 120, 3), np.uint8))).shape == (240, 240, 3)
    assert wc.decode_tile(b"not a jpeg") is None and wc.decode_tile(b"") is None

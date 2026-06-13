import numpy as np
from web.field.radar import Radar, RadarTrack

def test_render_returns_image_of_configured_size():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    img = r.render([])
    assert img.shape == (round(8.069*40), round(16.541*40), 3)

def test_point_maps_to_expected_pixel():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    # Observer HUD flips field Y so +Y is upward on screen.
    origin_px = r.field_to_pixel((0.0, 0.0))
    top_px = r.field_to_pixel((0.0, 8.069))
    assert abs(origin_px[1] - 8.069 * 40) <= 1
    assert abs(top_px[1]) <= 1

    # Field center still maps to screen center.
    px = r.field_to_pixel((8.2705, 4.0345))  # field center
    assert abs(px[0] - 16.541*40/2) <= 1
    assert abs(px[1] - 8.069*40/2) <= 1

def test_side_tint_blue_left_red_right():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    img = r.render([RadarTrack(pos=(2.0, 4.0), alliance="blue"),
                    RadarTrack(pos=(14.0, 4.0), alliance="red")])
    bu, bv = r.field_to_pixel((2.0, 4.0))
    ru, rv = r.field_to_pixel((14.0, 4.0))
    blue_px = img[int(bv), int(bu)]   # BGR; _TINT blue = (255, 120, 0)
    red_px  = img[int(rv), int(ru)]   # BGR; _TINT red  = (0,   0, 255)
    assert blue_px[0] > blue_px[2], f"Blue dot: B channel should dominate, got BGR={blue_px}"
    assert red_px[2]  > red_px[0],  f"Red dot: R channel should dominate, got BGR={red_px}"

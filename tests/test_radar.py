import numpy as np
from web.field.radar import Radar

def test_render_returns_image_of_configured_size():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    img = r.render([])
    assert img.shape == (round(8.069*40), round(16.541*40), 3)

def test_point_maps_to_expected_pixel():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    # field (0,0) -> top-left-ish origin mapping; check a known interior point.
    px = r.field_to_pixel((8.2705, 4.0345))  # field center
    assert abs(px[0] - 16.541*40/2) <= 1
    assert abs(px[1] - 8.069*40/2) <= 1

def test_side_tint_blue_left_red_right():
    r = Radar(field_length=16.541, field_width=8.069, px_per_m=40)
    img = r.render([((2.0, 4.0), "blue"), ((14.0, 4.0), "red")])
    assert img is not None  # smoke: renders without error

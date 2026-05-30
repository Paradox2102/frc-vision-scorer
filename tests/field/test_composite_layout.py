from web.field.composite_layout import load_layout, CameraLayout

def test_loads_three_cameras_with_crops():
    layout = load_layout()
    assert set(layout) >= {"main", "corner_left", "corner_right"}
    cl = layout["corner_left"]
    assert isinstance(cl, CameraLayout)
    x, y, w, h = cl.crop
    assert w > 0 and h > 0
    for m in cl.masks:
        assert len(m) == 4

def test_main_crop_starts_at_origin():
    layout = load_layout()
    assert layout["main"].crop[0] == 0 and layout["main"].crop[1] == 0

from web.field.occlusion import _merge_occlusion

def test_visible_if_any_camera_sees():
    a = {0: {"view_occluded": True,  "bumper_occluded": True,  "cause": "hub"}}
    b = {0: {"view_occluded": False, "bumper_occluded": False, "cause": None}}
    m = _merge_occlusion([a, b])
    assert m[0]["view_occluded"] is False
    assert m[0]["bumper_occluded"] is False

def test_occluded_if_all_cameras_blocked():
    a = {0: {"view_occluded": True, "bumper_occluded": True, "cause": "hub"}}
    b = {0: {"view_occluded": True, "bumper_occluded": True, "cause": "person"}}
    m = _merge_occlusion([a, b])
    assert m[0]["view_occluded"] is True
    assert m[0]["bumper_occluded"] is True

def test_empty_returns_empty():
    assert _merge_occlusion([]) == {}

def test_single_camera_parity():
    a = {0: {"view_occluded": True, "bumper_occluded": False, "cause": "hub"}}
    assert _merge_occlusion([a]) == a

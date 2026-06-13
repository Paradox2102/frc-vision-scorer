"""Static-structure zones: field-coordinate rectangles that block NEW track
seeding (a YOLO false-positive can't be 'born' inside a known field element)."""
import json

import pytest

from web.field import zones as Z


def test_point_inside_a_zone():
    assert Z.point_in_zones((2.0, 2.0), [(1.0, 1.0, 3.0, 3.0)]) is True


def test_point_outside_all_zones():
    assert Z.point_in_zones((5.0, 5.0), [(1.0, 1.0, 3.0, 3.0)]) is False


def test_no_zones_is_never_blocked():
    assert Z.point_in_zones((2.0, 2.0), []) is False
    assert Z.point_in_zones((2.0, 2.0), None) is False


def test_prism_bounding_zones_adds_margin():
    from types import SimpleNamespace
    import numpy as np

    prism = SimpleNamespace(
        footprint=np.array([[1.0, 2.0], [3.0, 2.0], [3.0, 4.0]])
    )
    assert Z.prism_bounding_zones([prism], margin=0.25) == [
        (0.75, 1.75, 3.25, 4.25)
    ]


def test_corners_may_be_given_in_any_order():
    # The UI hands back drag corners that may be (x2<x1) or (y2<y1).
    assert Z.point_in_zones((2.0, 2.0), [(3.0, 3.0, 1.0, 1.0)]) is True


def test_boundary_is_inclusive():
    assert Z.point_in_zones((1.0, 1.0), [(1.0, 1.0, 3.0, 3.0)]) is True


def test_inflate_zones_grows_each_side_by_margin():
    assert Z.inflate_zones([(1.0, 1.0, 3.0, 3.0)], 0.5) == [
        (0.5, 0.5, 3.5, 3.5)
    ]


def test_inflate_zones_normalizes_corner_order():
    # UI drag corners may arrive reversed; inflation must not invert the rect.
    assert Z.inflate_zones([(3.0, 3.0, 1.0, 1.0)], 0.5) == [
        (0.5, 0.5, 3.5, 3.5)
    ]


def test_inflate_zones_empty_and_none():
    assert Z.inflate_zones([], 0.5) == []
    assert Z.inflate_zones(None, 0.5) == []


def test_inflate_zones_zero_margin_keeps_extent():
    assert Z.inflate_zones([(1.0, 2.0, 3.0, 4.0)], 0.0) == [(1.0, 2.0, 3.0, 4.0)]


def test_qm28_outpost_phantom_blocked_by_inflated_zone():
    # Regression: 2026hop qm28 corner_left projected the red-outpost false
    # positive to (15.52, 1.72) — 17cm outside the drawn zone — and a phantom
    # robot held slot R2 for ~60s. The seed gate must use inflated zones so
    # corner-cam projection error can't slip a structure past a tight rect.
    outpost_zone = (15.69, 1.67, 16.58, 2.88)
    phantom_pos = (15.52, 1.72)
    assert Z.point_in_zones(phantom_pos, [outpost_zone]) is False  # the miss
    assert Z.point_in_zones(phantom_pos, Z.inflate_zones([outpost_zone], 0.4)) is True


def test_ray_into_zone_blocks_occluded_structure_contact():
    # Regression: 2026hop qm28 corner_left. The blue-outpost structure's base
    # is occluded (balls/wall lip), so its detected bottom edge projects to
    # (1.25, 3.63) — 1.9m short of the zone, beyond any sane point margin.
    # The ground ray from the camera (0.64, -0.3) through that contact enters
    # the inflated zone ~1.5m beyond the contact: classic occluded-base
    # signature from a low camera.
    zone = Z.inflate_zones([(0.03, 5.51, 1.34, 7.42)], 0.4)
    camera = (0.64, -0.3)
    contact = (1.25, 3.63)
    assert Z.point_in_zones(contact, zone) is False  # point test misses
    assert Z.ray_enters_zones(camera, contact, zone) is True


def test_ray_away_from_zone_keeps_real_robot():
    # Same match, corner_right's real robot 7451 at (11.58, 6.13): the ray
    # from the camera (15.82, -0.23) through it heads off-field, away from
    # the red-outpost zone — must NOT be blocked.
    zone = Z.inflate_zones([(15.36, 0.70, 16.51, 2.45)], 0.4)
    assert Z.ray_enters_zones((15.82, -0.23), (11.58, 6.13), zone) is False


def test_ray_respects_max_beyond_distance():
    # A zone far down the ray (beyond max_beyond_m) is not the structure the
    # detection clipped — don't block half the field behind a contact.
    zone = [(0.0, 9.0, 1.0, 10.0)]
    camera = (0.5, 0.0)
    contact = (0.5, 2.0)  # ray aims at the zone but it's 7m beyond
    assert Z.ray_enters_zones(camera, contact, zone, max_beyond_m=3.0) is False
    assert Z.ray_enters_zones(camera, contact, zone, max_beyond_m=8.0) is True


def test_ray_with_no_zones_or_degenerate_geometry():
    assert Z.ray_enters_zones((0.0, 0.0), (1.0, 1.0), []) is False
    assert Z.ray_enters_zones((0.0, 0.0), (1.0, 1.0), None) is False
    # Contact coincident with camera: no direction — never blocks.
    assert Z.ray_enters_zones((1.0, 1.0), (1.0, 1.0), [(0.0, 0.0, 2.0, 2.0)]) is False


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, "UPLOAD_DIR", tmp_path)
    rects = [(1.0, 1.0, 3.0, 3.0), (10.0, 5.0, 12.5, 7.0)]
    Z.save_zones("2026hop", rects)
    loaded = Z.load_zones("2026hop")
    assert loaded == rects


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, "UPLOAD_DIR", tmp_path)
    assert Z.load_zones("never-saved") == []


def test_saved_file_is_plain_json(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, "UPLOAD_DIR", tmp_path)
    Z.save_zones("2026hop", [(1.0, 1.0, 3.0, 3.0)])
    raw = json.loads((tmp_path / "zones_2026hop.json").read_text())
    assert raw == {"zones": [[1.0, 1.0, 3.0, 3.0]]}

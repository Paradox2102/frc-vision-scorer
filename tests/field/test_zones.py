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


def test_corners_may_be_given_in_any_order():
    # The UI hands back drag corners that may be (x2<x1) or (y2<y1).
    assert Z.point_in_zones((2.0, 2.0), [(3.0, 3.0, 1.0, 1.0)]) is True


def test_boundary_is_inclusive():
    assert Z.point_in_zones((1.0, 1.0), [(1.0, 1.0, 3.0, 3.0)]) is True


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

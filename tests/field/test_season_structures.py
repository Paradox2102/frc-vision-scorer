"""Season-aware structures and wireframe: 2025 reefs, no structures on fields we do
not model, and 2026 exactly as before."""
import numpy as np
import pytest

from tests._field_synth import truth_pinhole_camera
from web import hud_draw
from web.field import occlusion
from web.field.coverage import coverage_grid
from web.field.field_model import FieldModel
from web.field.occlusion import Prism, hub_prisms, occluded_by_hub, reef_prisms
from web.field.overlay_geometry import (bump_wireframe_edges, hub_wireframe_edges,
                                        structure_footprints, tower_wireframe_edges,
                                        trench_wireframe_edges)
from web.field.radar import Radar
from web.field.zones import prism_bounding_zones, point_in_zones

FM25 = FieldModel.load("2025-reefscape-welded")
FM24 = FieldModel.load("2024-crescendo")
FM23 = FieldModel.load("2023-chargedup")
FM26 = FieldModel.load("2026-rebuilt-welded")
REEF_COLOR = hud_draw.WIREFRAME_GROUP_COLORS["reef"]


# ------------------------------------------------------------------- 2025 reefs

def test_2025_reef_prisms_are_two_hexagons_at_the_reef_centres():
    prisms = hub_prisms(FM25)
    assert len(prisms) == 2 and all(p.kind == "reef" for p in prisms)
    centres = sorted(tuple(np.round(p.footprint.mean(axis=0), 2)) for p in prisms)
    assert centres == [(4.49, 4.03), (13.06, 4.03)]
    for p in prisms:
        assert p.footprint.shape == (6, 2)
        c = p.footprint.mean(axis=0)
        r = np.linalg.norm(p.footprint - c, axis=1)
        # wireframe reef base hexagon: circumradius 0.956 m
        assert np.allclose(r, 0.956, atol=0.01)
        # flat faces at +-x (the tags facing the alliance wall / midfield)
        assert np.isclose(p.footprint[:, 0].max() - c[0], 0.83, atol=0.01)
        assert 1.8 < p.z_top < 1.9


def test_2025_reef_prisms_match_the_wireframe_reef_footprint():
    from web.field.season import load_wireframe
    reef = load_wireframe("2025-reefscape-welded")["reef"].reshape(-1, 3)
    ground = reef[reef[:, 2] < 0.05, :2]
    for p in reef_prisms(FM25):
        c = p.footprint.mean(axis=0)
        mine = ground[np.linalg.norm(ground - c, axis=1) < 1.5]
        r = np.linalg.norm(mine - c, axis=1).max()
        assert abs(r - 0.956) < 0.02


def test_andymark_2025_also_gets_reefs():
    assert len(hub_prisms(FieldModel.load("2025-reefscape-andymark"))) == 2


def test_2025_no_robot_zones_cover_reefs_not_barge_or_scoring_robots():
    zones = prism_bounding_zones(hub_prisms(FM25))
    assert len(zones) == 2
    for cx in (4.489, 13.059):
        assert point_in_zones((cx, 4.026), zones)
        # a robot (centre ~0.4 m off the face) scoring on an angled face or a flat
        # face is NOT inside a no-seed zone
        for ang in (0, 60, 120, 180, 240, 300):
            a = np.radians(ang)
            pt = (cx + 1.23 * np.cos(a), 4.026 + 1.23 * np.sin(a))
            assert not point_in_zones(pt, zones), (cx, ang)
    # the barge is on the midline; robots drive under it
    for y in (1.0, 4.026, 7.0):
        assert not point_in_zones((8.774, y), zones)


def test_2025_structure_footprints_and_radar():
    fp = structure_footprints(FM25, hub_prisms(FM25))
    assert set(fp) == {"reefs"} and len(fp["reefs"]) == 2
    assert all(len(poly) == 6 for poly in fp["reefs"])
    r = Radar(FM25.length_m, FM25.width_m, field_model=FM25)
    assert r.field_image == "2025_field2d"        # the map is the 2025 drawing
    px, py = (int(v) for v in r.field_to_pixel((4.489 + 0.6, 4.026)))
    cx, cy = (int(v) for v in r.field_to_pixel((2.0, 1.0)))  # open carpet
    assert abs(int(r._base[py, px].sum()) - int(r._base[cy, cx].sum())) > 60  # reef shown


def test_reef_occludes_a_robot_behind_it_from_a_low_camera():
    blue = [p for p in hub_prisms(FM25) if p.footprint.mean(axis=0)[0] < 8][0]
    cam = (4.489, -1.0, 0.7)
    assert occluded_by_hub(cam, blue, (4.489, 6.0))
    assert not occluded_by_hub(cam, blue, (4.489, 1.5))


def test_2025_tower_trench_and_hub_edges_are_empty():
    assert tower_wireframe_edges(FM25) == []
    assert trench_wireframe_edges(FM25) == []
    prisms = hub_prisms(FM25)
    assert hub_wireframe_edges(prisms) == []
    assert bump_wireframe_edges(prisms, FM25.width_m) == []


# ------------------------------------------------------ fields without structures

@pytest.mark.parametrize("fm", [FM24, FM23], ids=["2024", "2023"])
def test_unmodelled_fields_have_no_structures_and_do_not_raise(fm):
    assert hub_prisms(fm) == []
    assert structure_footprints(fm, []) == {}
    assert prism_bounding_zones(hub_prisms(fm)) == []
    assert tower_wireframe_edges(fm) == [] and trench_wireframe_edges(fm) == []
    Radar(fm.length_m, fm.width_m, field_model=fm)
    xs, ys, grid = coverage_grid([], fm, cell_m=2.0)
    assert grid.shape == (len(ys), len(xs))


def test_hand_built_model_missing_2026_tags_does_not_raise():
    fm = FieldModel(tags={}, length_m=16.5, width_m=8.0)
    assert hub_prisms(fm) == []
    assert tower_wireframe_edges(fm) == [] and trench_wireframe_edges(fm) == []
    assert structure_footprints(fm, []) == {"hubs": [], "bumps": [], "towers": [],
                                            "trenches": []}


# ---------------------------------------------------------------- 2026 unchanged

def test_2026_structures_unchanged():
    prisms = hub_prisms(FM26)
    legacy = occlusion._hub_prisms_2026(FM26)
    assert len(prisms) == 2 and all(p.kind == "hub" for p in prisms)
    for a, b in zip(prisms, legacy):
        assert np.array_equal(a.footprint, b.footprint) and a.z_top == b.z_top
    fp = structure_footprints(FM26, prisms)
    assert {k: len(v) for k, v in fp.items()} == {"hubs": 2, "bumps": 4, "towers": 2,
                                                  "trenches": 4}
    assert len(tower_wireframe_edges(FM26)) > 0 and len(trench_wireframe_edges(FM26)) > 0
    assert len(prism_bounding_zones(prisms)) == 2


def test_hub_edge_functions_skip_reef_prisms():
    reef = Prism(footprint=np.array([[0, 0], [1, 0], [1, 1]], float), z_top=1.8,
                 kind="reef")
    hub = Prism(footprint=np.array([[4.0, 3.4], [5.2, 3.4], [5.2, 4.6], [4.0, 4.6]]),
                z_top=1.3)
    assert hub_wireframe_edges([reef, hub]) == hub_wireframe_edges([hub])
    assert bump_wireframe_edges([reef, hub], 8.0) == bump_wireframe_edges([hub], 8.0)


# ------------------------------------------------------------------- wireframe

def _cam(fm, season):
    cx = fm.length_m / 2.0
    cam = truth_pinhole_camera(eye=(cx, -3.0, 6.0), target=(cx, 4.03, 0.0))
    cam.field_season = season
    return cam


def _color_px(img, color):
    return int(np.all(img == np.array(color, np.uint8), axis=2).sum())


def test_2025_wireframe_draws_reefs_on_a_synthetic_calibration():
    cam = _cam(FM25, "2025-reefscape-welded")
    img = np.zeros((1080, 1920, 3), np.uint8)
    hud_draw.draw_field_wireframe(img, cam, FM25)
    assert _color_px(img, REEF_COLOR) > 500
    # reef pixels sit around the projected reef centres
    for cx in (4.489, 13.059):
        u, v = cam.field_to_image((cx, 4.026, 0.3))
        win = img[int(v) - 80:int(v) + 80, int(u) - 120:int(u) + 120]
        assert _color_px(win, REEF_COLOR) > 50, cx
    assert _color_px(img, hud_draw.WIREFRAME_GROUP_COLORS["barge"]) > 100
    # no 2026 structures on a 2025 field
    assert _color_px(img, (255, 0, 255)) == _color_px(
        img, hud_draw.WIREFRAME_GROUP_COLORS["barge"])          # barge only, no tower
    assert _color_px(img, (0, 140, 255)) == _color_px(
        img, hud_draw.WIREFRAME_GROUP_COLORS["coral_station"])  # no bumps


def test_wireframe_skips_floor_and_other_groups(monkeypatch):
    from web.field import season as sn
    seen = []
    real = hud_draw._project_drawable

    def spy(cal, pts, *a):
        seen.append(np.asarray(pts, float))
        return real(cal, pts, *a)
    monkeypatch.setattr(hud_draw, "_project_drawable", spy)
    groups = {"floor": np.ones((3, 6)), "other": np.ones((2, 6)),
              "reef": np.array([[4.0, 4.0, 0.0, 5.0, 4.0, 0.0]])}
    monkeypatch.setattr(sn, "wireframe_for", lambda fm: groups)
    img = np.zeros((1080, 1920, 3), np.uint8)
    hud_draw.draw_field_wireframe(img, _cam(FM25, "2025-reefscape-welded"), FM25)
    # only the reef group was projected (perimeter/grid lines sit at z = 0)
    glb = [p for p in seen if np.allclose(p[:, 1], 4.0) and np.allclose(p[:, 2], 0.0)
           and p[:, 0].min() >= 4.0 and p[:, 0].max() <= 5.0]
    assert len(glb) == 1
    assert not any(np.allclose(p, 1.0) for p in seen)        # floor / other


def test_season_without_wireframe_draws_only_perimeter_and_grid():
    cam = _cam(FM24, "2024-crescendo")
    img = np.zeros((1080, 1920, 3), np.uint8)
    hud_draw.draw_field_wireframe(img, cam, FM24)
    colors = {tuple(c) for c in img.reshape(-1, 3)[img.reshape(-1, 3).any(axis=1)]}
    assert colors <= {(0, 255, 0), (0, 180, 0)}
    assert (0, 255, 0) in colors


def test_glb_wireframe_drops_points_behind_the_camera(monkeypatch):
    from web.field import season as sn
    cam = _cam(FM25, "2025-reefscape-welded")   # at (cx, -3, 6) looking down at +y
    front, behind = (8.77, 4.0, 0.0), (8.77, -6.0, 8.0)
    assert list(cam.in_view([front, behind], margin_px=None)) == [True, False]
    monkeypatch.setattr(sn, "wireframe_for", lambda fm: {
        "reef": np.array([[8.0, -6.0, 8.0, 9.5, -6.5, 8.5]])})   # wholly behind
    img = np.zeros((1080, 1920, 3), np.uint8)
    hud_draw.draw_field_wireframe(img, cam, FM25)
    assert _color_px(img, REEF_COLOR) == 0


def test_glb_projection_matches_field_to_image():
    cam = _cam(FM25, "2025-reefscape-welded")
    pts = np.array([[4.489, 4.026, 0.3], [8.774, 1.9, 1.8], [0.0, 0.0, 0.0]])
    batch = hud_draw._project_many(cam, pts)
    for p, uv in zip(pts, batch):
        assert np.allclose(uv, cam.field_to_image(tuple(p)), atol=1e-6)

import numpy as np

from web.field.occlusion import Prism, hub_prisms
from web.field.overlay_geometry import (
    BUMP_HEIGHT_M,
    HUB_BASE_WIDTH_M,
    HUB_DECK_HEIGHT_M,
    HUB_HEIGHT_M,
    HUB_OPENING_WIDTH_M,
    HUB_TOP_WIDTH_M,
    TOWER_DEPTH_M,
    TOWER_HEIGHT_M,
    TOWER_RUNG_HEIGHTS_M,
    TRENCH_CLEARANCE_M,
    TRENCH_FIXED_INNER_CAD_M,
    TRENCH_FIXED_OUTER_CAD_M,
    TRENCH_HINGED_INNER_CAD_M,
    TRENCH_HINGED_OUTER_CAD_M,
    bump_wireframe_edges,
    hub_wireframe_edges,
    tower_wireframe_edges,
    trench_wireframe_edges,
)
from web.field.field_model import FieldModel


def _prism():
    return Prism(
        footprint=np.array([
            [4.0, 3.4], [5.2, 3.4], [5.2, 4.6], [4.0, 4.6],
        ]),
        z_top=1.3,
    )


def test_hub_wireframe_has_top_bottom_and_vertical_edges():
    edges = hub_wireframe_edges([_prism()])
    assert len(edges) == 26
    assert any(a[2] > 3.0 and b[2] > 3.0 for a, b in edges)
    assert any(a[2] == 0.0 and b[2] == HUB_DECK_HEIGHT_M for a, b in edges)


def test_bump_wireframe_builds_two_ridges_per_hub():
    edges = bump_wireframe_edges([_prism()], field_width=8.0)
    assert len(edges) == 18
    raised = [
        point for edge in edges for point in edge
        if abs(point[2] - BUMP_HEIGHT_M) < 1e-9
    ]
    assert raised
    assert all(0.0 <= point[1] <= 8.0 for edge in edges for point in edge)


def test_hub_wireframe_is_rectangular_even_for_octagonal_footprint():
    prism = Prism(
        footprint=np.array([
            [4.0, 3.5], [4.3, 3.2], [4.9, 3.2], [5.2, 3.5],
            [5.2, 4.1], [4.9, 4.4], [4.3, 4.4], [4.0, 4.1],
        ]),
        z_top=1.3,
    )
    assert len(hub_wireframe_edges([prism])) == 26


def test_hub_wireframe_uses_drawing_dimensions():
    edges = hub_wireframe_edges([_prism()])
    points = [point for edge in edges for point in edge]
    deck_points = [p for p in points if abs(p[2] - HUB_DECK_HEIGHT_M) < 1e-9]
    top_points = [p for p in points if abs(p[2] - HUB_HEIGHT_M) < 1e-9]
    assert abs((max(p[0] for p in deck_points) - min(p[0] for p in deck_points))
               - HUB_BASE_WIDTH_M) < 1e-9
    assert abs((max(p[0] for p in top_points) - min(p[0] for p in top_points))
               - HUB_TOP_WIDTH_M) < 1e-9


def test_hub_opening_is_hexagonal_and_measured():
    edges = hub_wireframe_edges([_prism()])
    opening_edges = edges[12:18]
    assert len(opening_edges) == 6
    opening_points = [point for edge in opening_edges for point in edge]
    assert all(abs(p[2] - HUB_DECK_HEIGHT_M) < 1e-9 for p in opening_points)
    assert abs((max(p[0] for p in opening_points) - min(p[0] for p in opening_points))
               - HUB_OPENING_WIDTH_M) < 1e-9


def test_towers_have_panel_uprights_rungs_and_braces():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = tower_wireframe_edges(fm)
    assert len(edges) == 28
    heights = {round(point[2], 6) for edge in edges for point in edge}
    assert round(TOWER_HEIGHT_M, 6) in heights
    assert all(round(z, 6) in heights for z in TOWER_RUNG_HEIGHTS_M)


def test_tower_frame_extends_40_inches_inward_from_panel():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = tower_wireframe_edges(fm)
    blue_edges = edges[:14]
    xs = [point[0] for edge in blue_edges for point in edge]
    assert abs((max(xs) - min(xs)) - TOWER_DEPTH_M) < 1e-9


def test_tower_is_centered_on_field_not_tag_midpoint():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = tower_wireframe_edges(fm)
    blue_edges = edges[:14]
    ys = [point[1] for edge in blue_edges for point in edge]
    assert abs((min(ys) + max(ys)) / 2.0 - fm.width_m / 2.0) < 1e-9

    tag_midpoint = float(np.mean([
        fm.tag_translation(31)[1],
        fm.tag_translation(32)[1],
    ]))
    assert abs(tag_midpoint - fm.width_m / 2.0) > 0.05


def test_trenches_have_rails_and_dimensioned_support_stations():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = trench_wireframe_edges(fm)
    assert len(edges) == 72
    raised_horizontal = [
        (a, b) for a, b in edges
        if abs(a[2] - TRENCH_CLEARANCE_M) < 1e-9
        and abs(b[2] - TRENCH_CLEARANCE_M) < 1e-9
    ]
    assert raised_horizontal


def test_trench_rail_uses_cad_fixed_arm_span():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = trench_wireframe_edges(fm)
    ys = [point[1] for edge in edges[:12] for point in edge]
    expected = TRENCH_FIXED_OUTER_CAD_M - TRENCH_FIXED_INNER_CAD_M
    assert abs((max(ys) - min(ys)) - expected) < 1e-9


def test_trench_rail_is_not_assumed_centered_on_tags():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = trench_wireframe_edges(fm)
    blue_near = edges[:18]
    ys = [point[1] for edge in blue_near for point in edge]
    tagged_y = float(np.mean([
        fm.tag_translation(17)[1],
        fm.tag_translation(28)[1],
    ]))
    assert abs((min(ys) + max(ys)) / 2.0 - tagged_y) > 0.02


def test_trench_rails_align_with_field_perimeter():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = trench_wireframe_edges(fm)
    near_ys = [point[1] for edge in edges[:18] for point in edge]
    far_ys = [point[1] for edge in edges[18:36] for point in edge]
    assert abs(min(near_ys) - (
        fm.width_m / 2.0 - TRENCH_FIXED_OUTER_CAD_M
    )) < 1e-9
    assert abs(max(far_ys) - (
        fm.width_m / 2.0 + TRENCH_HINGED_OUTER_CAD_M
    )) < 1e-9


def test_hinged_trench_uses_cad_arm_span():
    fm = FieldModel.load("2026-rebuilt-welded")
    edges = trench_wireframe_edges(fm)
    far_ys = [point[1] for edge in edges[18:30] for point in edge]
    expected = TRENCH_HINGED_OUTER_CAD_M - TRENCH_HINGED_INNER_CAD_M
    assert abs((max(far_ys) - min(far_ys)) - expected) < 1e-9


def test_trench_rails_do_not_overlap_bumps():
    fm = FieldModel.load("2026-rebuilt-welded")
    trench_edges = trench_wireframe_edges(fm)
    bump_edges = bump_wireframe_edges(hub_prisms(fm), fm.width_m)

    near_trench_max_y = max(
        point[1] for edge in trench_edges[:18] for point in edge
    )
    near_bump_min_y = min(
        point[1] for edge in bump_edges[:9] for point in edge
    )
    assert near_trench_max_y <= near_bump_min_y + 0.01

"""Wireframe geometry used to validate field-camera calibration."""
from __future__ import annotations

import numpy as np

BUMP_WIDTH_M = 1.854
BUMP_DEPTH_M = 1.128
BUMP_HEIGHT_M = 0.1654
HUB_BASE_WIDTH_M = 47.0 * 0.0254
HUB_DECK_HEIGHT_M = 72.0 * 0.0254
HUB_TOP_WIDTH_M = 58.41 * 0.0254
HUB_HEIGHT_M = 120.36 * 0.0254
HUB_OPENING_WIDTH_M = 41.932 * 0.0254
TOWER_DEPTH_M = 40.0 * 0.0254
TOWER_WIDTH_M = 47.0 * 0.0254
TOWER_UPRIGHT_SPACING_M = 32.25 * 0.0254
TOWER_HEIGHT_M = 72.125 * 0.0254
TOWER_RUNG_HEIGHTS_M = tuple(v * 0.0254 for v in (27.0, 45.0, 63.0))
TOWER_BRACE_HEIGHTS_M = tuple(v * 0.0254 for v in (35.125, 43.375))
TRENCH_SPAN_M = 49.82 * 0.0254
TRENCH_CLEARANCE_M = 565.15 / 1000.0
TRENCH_RAIL_HEIGHT_M = 101.6 / 1000.0
TRENCH_FIXED_ARM_WIDTH_M = 152.4 / 1000.0
TRENCH_FIXED_INNER_CAD_M = 2455.862 / 1000.0
TRENCH_FIXED_OUTER_CAD_M = 4254.5 / 1000.0
TRENCH_HINGED_ARM_WIDTH_M = 203.2 / 1000.0
TRENCH_HINGED_INNER_CAD_M = 2838.847 / 1000.0
TRENCH_HINGED_OUTER_CAD_M = 4114.006 / 1000.0


def _box_edges(x0, x1, y0, y1, z0, z1):
    bottom = (
        (x0, y0, z0), (x1, y0, z0),
        (x1, y1, z0), (x0, y1, z0),
    )
    top = tuple((x, y, z1) for x, y, _z in bottom)
    edges = []
    for ring in (bottom, top):
        for i in range(4):
            edges.append((ring[i], ring[(i + 1) % 4]))
    edges.extend((bottom[i], top[i]) for i in range(4))
    return edges


def structure_footprints(field_model, prisms) -> dict:
    """Ground-plane outlines of the fixed field structures, in field meters.

    Single source of truth shared by the calibrate overlay (wireframes above)
    and the 2D field map: hub cabinet bases, bumps, towers, trenches. Returns
    {"hubs": [poly], "bumps": [...], "towers": [...], "trenches": [...]}
    where each poly is a list of (x, y) vertices in drawing order."""
    out = {"hubs": [], "bumps": [], "towers": [], "trenches": []}

    for prism in prisms:
        poly = np.asarray(prism.footprint, float)
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        bh = HUB_BASE_WIDTH_M / 2.0
        out["hubs"].append(
            [(cx - bh, cy - bh), (cx + bh, cy - bh),
             (cx + bh, cy + bh), (cx - bh, cy + bh)])

        x0 = cx - BUMP_DEPTH_M / 2.0
        x1 = cx + BUMP_DEPTH_M / 2.0
        near_y1 = float(poly[:, 1].min())
        far_y0 = float(poly[:, 1].max())
        for y0, y1 in (
                (max(0.0, near_y1 - BUMP_WIDTH_M), near_y1),
                (far_y0, min(float(field_model.width_m), far_y0 + BUMP_WIDTH_M))):
            out["bumps"].append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    for tag_ids, inward_sign in (((31, 32), 1.0), ((15, 16), -1.0)):
        pts = np.array([field_model.tag_translation(i) for i in tag_ids], float)
        panel_x = float(pts[:, 0].mean())
        frame_x = panel_x + inward_sign * TOWER_DEPTH_M
        cy = float(field_model.width_m) / 2.0
        y0, y1 = cy - TOWER_WIDTH_M / 2.0, cy + TOWER_WIDTH_M / 2.0
        xa, xb = sorted((panel_x, frame_x))
        out["towers"].append([(xa, y0), (xb, y0), (xb, y1), (xa, y1)])

    for tag_ids in ((17, 28), (22, 23), (6, 7), (1, 12)):
        pts = np.array([field_model.tag_translation(i) for i in tag_ids], float)
        center_x = float(pts[:, 0].mean())
        tagged_y = float(pts[:, 1].mean())
        cy = float(field_model.width_m) / 2.0
        if tagged_y < cy:
            arm_w = TRENCH_FIXED_ARM_WIDTH_M
            y0 = cy - TRENCH_FIXED_OUTER_CAD_M
            y1 = cy - TRENCH_FIXED_INNER_CAD_M
        else:
            arm_w = TRENCH_HINGED_ARM_WIDTH_M
            y0 = cy + TRENCH_HINGED_INNER_CAD_M
            y1 = cy + TRENCH_HINGED_OUTER_CAD_M
        x0, x1 = center_x - arm_w / 2.0, center_x + arm_w / 2.0
        out["trenches"].append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    return out


def hub_wireframe_edges(prisms):
    """Dimensioned hub cabinet, opening, and tapered upper frame."""
    edges = []
    for prism in prisms:
        poly = np.asarray(prism.footprint, float)
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        base_half = HUB_BASE_WIDTH_M / 2.0
        top_half = HUB_TOP_WIDTH_M / 2.0

        # Four-sided 47-inch cabinet up to the 72-inch deck.
        edges.extend(_box_edges(
            cx - base_half, cx + base_half,
            cy - base_half, cy + base_half,
            0.0, HUB_DECK_HEIGHT_M,
        ))

        # Hexagonal opening on the deck, matching the measured outside width.
        radius = HUB_OPENING_WIDTH_M / 2.0
        opening = [
            (
                cx + radius * np.cos(np.deg2rad(60.0 * i)),
                cy + radius * np.sin(np.deg2rad(60.0 * i)),
                HUB_DECK_HEIGHT_M,
            )
            for i in range(6)
        ]
        for i in range(6):
            edges.append((opening[i], opening[(i + 1) % 6]))

        # 58.41-inch top net frame and tapered corner supports.
        deck = (
            (cx - base_half, cy - base_half, HUB_DECK_HEIGHT_M),
            (cx + base_half, cy - base_half, HUB_DECK_HEIGHT_M),
            (cx + base_half, cy + base_half, HUB_DECK_HEIGHT_M),
            (cx - base_half, cy + base_half, HUB_DECK_HEIGHT_M),
        )
        top = (
            (cx - top_half, cy - top_half, HUB_HEIGHT_M),
            (cx + top_half, cy - top_half, HUB_HEIGHT_M),
            (cx + top_half, cy + top_half, HUB_HEIGHT_M),
            (cx - top_half, cy + top_half, HUB_HEIGHT_M),
        )
        for i in range(4):
            edges.append((top[i], top[(i + 1) % 4]))
            edges.append((deck[i], top[i]))
    return edges


def tower_wireframe_edges(field_model):
    """Tower panel, uprights, rungs, and depth braces anchored to tower tags."""
    edges = []
    for tag_ids, inward_sign in (((31, 32), 1.0), ((15, 16), -1.0)):
        pts = np.array([field_model.tag_translation(i) for i in tag_ids], float)
        panel_x = float(pts[:, 0].mean())
        frame_x = panel_x + inward_sign * TOWER_DEPTH_M
        # The top-view drawing centers the tower frame on the field centerline.
        # The two tower tags are offset on the rear panel, so their midpoint is
        # not the structure center and would shift the wireframe by about 3.4 in.
        center_y = float(field_model.width_m) / 2.0
        panel_y0 = center_y - TOWER_WIDTH_M / 2.0
        panel_y1 = center_y + TOWER_WIDTH_M / 2.0
        upright_ys = (
            center_y - TOWER_UPRIGHT_SPACING_M / 2.0,
            center_y + TOWER_UPRIGHT_SPACING_M / 2.0,
        )

        # Rear alliance-wall panel.
        edges.extend((
            ((panel_x, panel_y0, 0.0), (panel_x, panel_y1, 0.0)),
            ((panel_x, panel_y1, 0.0), (panel_x, panel_y1, TOWER_HEIGHT_M)),
            ((panel_x, panel_y1, TOWER_HEIGHT_M),
             (panel_x, panel_y0, TOWER_HEIGHT_M)),
            ((panel_x, panel_y0, TOWER_HEIGHT_M), (panel_x, panel_y0, 0.0)),
        ))

        # Two forward uprights and the 47-inch top crossbar.
        for y in upright_ys:
            edges.append(((frame_x, y, 0.0), (frame_x, y, TOWER_HEIGHT_M)))
        edges.append((
            (frame_x, panel_y0, TOWER_HEIGHT_M),
            (frame_x, panel_y1, TOWER_HEIGHT_M),
        ))

        # Three climbing rungs at the drawing heights.
        for z in TOWER_RUNG_HEIGHTS_M:
            edges.append(((frame_x, panel_y0, z), (frame_x, panel_y1, z)))

        # Two horizontal members connecting the panel to each forward upright.
        for z in TOWER_BRACE_HEIGHTS_M:
            for y in upright_ys:
                edges.append(((panel_x, y, z), (frame_x, y, z)))
    return edges


def trench_wireframe_edges(field_model):
    """CAD-derived fixed and hinged trench arm envelopes.

    The STEP assembly uses a centered field coordinate frame. The fixed arm
    reaches the adjacent bump, while the hinged arm has a shorter field-side
    extent. AprilTag pairs anchor each arm's x center.
    """
    edges = []
    for tag_ids in ((17, 28), (22, 23), (6, 7), (1, 12)):
        pts = np.array([field_model.tag_translation(i) for i in tag_ids], float)
        center_x = float(pts[:, 0].mean())
        tagged_y = float(pts[:, 1].mean())
        center_y = float(field_model.width_m) / 2.0
        if tagged_y < center_y:
            arm_width = TRENCH_FIXED_ARM_WIDTH_M
            y0 = center_y - TRENCH_FIXED_OUTER_CAD_M
            y1 = center_y - TRENCH_FIXED_INNER_CAD_M
        else:
            arm_width = TRENCH_HINGED_ARM_WIDTH_M
            y0 = center_y + TRENCH_HINGED_INNER_CAD_M
            y1 = center_y + TRENCH_HINGED_OUTER_CAD_M
        x0 = center_x - arm_width / 2.0
        x1 = center_x + arm_width / 2.0
        rail_bottom = TRENCH_CLEARANCE_M
        rail_top = rail_bottom + TRENCH_RAIL_HEIGHT_M
        edges.extend(_box_edges(x0, x1, y0, y1, rail_bottom, rail_top))
        # Drop the arm endpoints to the carpet to make alignment easy to inspect.
        for y in (y0, y1):
            edges.extend((
                ((x0, y, 0.0), (x0, y, rail_bottom)),
                ((x1, y, 0.0), (x1, y, rail_bottom)),
                ((x0, y, 0.0), (x1, y, 0.0)),
            ))
    return edges


def bump_wireframe_edges(prisms, field_width):
    """Return triangular-prism wireframes for the four bumps.

    Each hub has one bump on its near-sideline side and one on its far-sideline
    side. The hub footprint anchors their exact field placement.
    """
    edges = []
    for prism in prisms:
        poly = np.asarray(prism.footprint, float)
        center_x = float(poly[:, 0].mean())
        x0 = center_x - BUMP_DEPTH_M / 2.0
        x1 = center_x + BUMP_DEPTH_M / 2.0
        ridge_x = center_x
        near_y1 = float(poly[:, 1].min())
        far_y0 = float(poly[:, 1].max())
        ranges = (
            (max(0.0, near_y1 - BUMP_WIDTH_M), near_y1),
            (far_y0, min(float(field_width), far_y0 + BUMP_WIDTH_M)),
        )
        for y0, y1 in ranges:
            corners = (
                (x0, y0, 0.0), (x1, y0, 0.0),
                (x1, y1, 0.0), (x0, y1, 0.0),
            )
            ridges = (
                (ridge_x, y0, BUMP_HEIGHT_M),
                (ridge_x, y1, BUMP_HEIGHT_M),
            )
            for i in range(4):
                edges.append((corners[i], corners[(i + 1) % 4]))
            edges.extend((
                (corners[0], ridges[0]), (corners[1], ridges[0]),
                (corners[3], ridges[1]), (corners[2], ridges[1]),
                (ridges[0], ridges[1]),
            ))
    return edges

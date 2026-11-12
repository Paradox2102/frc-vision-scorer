"""Wireframe geometry used to validate field-camera calibration.

The hand-coded structures here (hub, bump, tower, trench) are the 2026 REBUILT
field. Other seasons draw their AprilTag-derived structures (2025 reefs) and the
AdvantageScope GLB wireframe instead (web.hud_draw); every function here returns
nothing for a field that is not 2026 (web.field.season.is_2026) rather than look up
2026 tag ids that field does not have."""
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

# (tag pair, tag the tower is centred on, +x direction into the field).
# The tower is NOT centred on the field centreline: its panel and uprights are
# centred on the outer tag of each pair (31 blue, 15 red; point-symmetric).
# Measured Nov 11 2026: AdvantageScope 2026 field mesh puts the blue panel
# centre at y=3.7455 (tag 31 y=3.746) and red at 4.3235 (tag 15 y=4.324), and
# 2026utwv qm1 overhead footage shows both real ladders where the mesh puts
# them, ~29 cm off the old centreline wireframe.
TOWER_ANCHORS = (((31, 32), 31, 1.0), ((15, 16), 15, -1.0))


def hub_center(prism) -> tuple[float, float]:
    """Hub cabinet centre (x, y) from a tag-ring prism footprint.

    The footprint is the convex hull of the hub tag centres. Every hub tag
    sits on one of the four cabinet faces, so the hull's bounding box spans
    face plane to face plane and its midpoint is the cabinet centre. The
    hull's VERTEX mean is not: the tags are unevenly spaced along the faces
    (24/27 and 25 are off-centre), which pulled it ~9 cm toward the alliance
    wall (blue 4.537 vs 4.6255; the AdvantageScope mesh cabinet is 4.6225)."""
    poly = np.asarray(prism.footprint, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    return float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0)


def _has_tags(field_model, ids) -> bool:
    tags = getattr(field_model, "tags", None) or {}
    return all(i in tags for i in ids)


def _is_2026(field_model) -> bool:
    from web.field.season import is_2026
    return is_2026(field_model)


def _hub_only(prisms):
    return [p for p in prisms if getattr(p, "kind", "hub") == "hub"]


def _tower_center_y(field_model, anchor_tag: int) -> float:
    return float(field_model.tag_translation(anchor_tag)[1])


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
    where each poly is a list of (x, y) vertices in drawing order.

    2025: {"reefs": [hexagon, ...]} from the reef prisms. A field without
    modelled structures: {}."""
    from web.field.season import STRUCTURE_2025_REEF, STRUCTURE_2026_HUB, structure_kind
    kind = structure_kind(field_model)
    if kind == STRUCTURE_2025_REEF:
        return {"reefs": [[(float(x), float(y)) for x, y in np.asarray(p.footprint, float)]
                          for p in prisms if getattr(p, "kind", "hub") == "reef"]}
    if kind != STRUCTURE_2026_HUB:
        return {}
    out = {"hubs": [], "bumps": [], "towers": [], "trenches": []}

    for prism in _hub_only(prisms):
        poly = np.asarray(prism.footprint, float)
        cx, cy = hub_center(prism)
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

    for tag_ids, anchor_tag, inward_sign in TOWER_ANCHORS:
        if not _has_tags(field_model, tag_ids):
            continue
        pts = np.array([field_model.tag_translation(i) for i in tag_ids], float)
        panel_x = float(pts[:, 0].mean())
        frame_x = panel_x + inward_sign * TOWER_DEPTH_M
        cy = _tower_center_y(field_model, anchor_tag)
        y0, y1 = cy - TOWER_WIDTH_M / 2.0, cy + TOWER_WIDTH_M / 2.0
        xa, xb = sorted((panel_x, frame_x))
        out["towers"].append([(xa, y0), (xb, y0), (xb, y1), (xa, y1)])

    for tag_ids in ((17, 28), (22, 23), (6, 7), (1, 12)):
        if not _has_tags(field_model, tag_ids):
            continue
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
    """Dimensioned hub cabinet, opening, and tapered upper frame (2026 hub
    prisms only; reef prisms are skipped)."""
    edges = []
    for prism in _hub_only(prisms):
        cx, cy = hub_center(prism)
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
    """Tower panel, uprights, rungs, and depth braces anchored to tower tags.
    [] on a non-2026 field."""
    edges = []
    if not _is_2026(field_model):
        return edges
    for tag_ids, anchor_tag, inward_sign in TOWER_ANCHORS:
        if not _has_tags(field_model, tag_ids):
            continue
        pts = np.array([field_model.tag_translation(i) for i in tag_ids], float)
        panel_x = float(pts[:, 0].mean())
        frame_x = panel_x + inward_sign * TOWER_DEPTH_M
        # Centred on the outer tag, not the field centreline or the tag-pair
        # midpoint (see TOWER_ANCHORS for the mesh + footage measurement).
        center_y = _tower_center_y(field_model, anchor_tag)
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
    extent. AprilTag pairs anchor each arm's x center. [] on a non-2026 field.
    """
    edges = []
    if not _is_2026(field_model):
        return edges
    for tag_ids in ((17, 28), (22, 23), (6, 7), (1, 12)):
        if not _has_tags(field_model, tag_ids):
            continue
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
    side. The hub footprint anchors their exact field placement. (2026 hub
    prisms only; reef prisms are skipped.)
    """
    edges = []
    for prism in _hub_only(prisms):
        poly = np.asarray(prism.footprint, float)
        center_x, _ = hub_center(prism)
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

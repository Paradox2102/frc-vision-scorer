"""Field map v2: exact structure footprints + track-state rendering."""
import numpy as np

from web.field.field_model import FieldModel
from web.field.manta import MantaAutoPath
from web.field.occlusion import hub_prisms
from web.field.overlay_geometry import structure_footprints
from web.field.radar import Radar, RadarTrack, _ALLIANCE


def _fm():
    return FieldModel.load("2026-rebuilt-welded")


def test_structure_footprints_counts_and_bounds():
    fm = _fm()
    fp = structure_footprints(fm, hub_prisms(fm))
    assert len(fp["hubs"]) == 2
    assert len(fp["bumps"]) == 4
    assert len(fp["towers"]) == 2
    assert len(fp["trenches"]) == 4
    for kind, polys in fp.items():
        for poly in polys:
            arr = np.asarray(poly, float)
            assert arr.ndim == 2 and arr.shape[1] == 2, kind
            assert arr[:, 0].min() >= -0.5 and arr[:, 0].max() <= fm.length_m + 0.5
            assert arr[:, 1].min() >= -0.5 and arr[:, 1].max() <= fm.width_m + 0.5


def test_structure_footprints_hubs_mirror_about_midline():
    fm = _fm()
    fp = structure_footprints(fm, hub_prisms(fm))
    centers_x = sorted(float(np.asarray(p, float)[:, 0].mean()) for p in fp["hubs"])
    mid = fm.length_m / 2.0
    # One hub each side, symmetric within CAD/tag tolerance.
    assert centers_x[0] < mid < centers_x[1]
    assert abs((mid - centers_x[0]) - (centers_x[1] - mid)) < 0.3


def _track(**kw):
    base = dict(pos=(4.0, 4.0), alliance="blue", team="4174", slot_id=1,
                state="visible", corner_only=False, trail=())
    base.update(kw)
    return RadarTrack(**base)


def test_render_accepts_rich_tracks_and_draws_field():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    img = r.render([
        _track(),
        _track(pos=(12.0, 3.0), alliance="red", team=None, slot_id=4,
               state="coasting", corner_only=True,
               trail=[(11.5, 3.0), (11.7, 3.0), (11.9, 3.0)]),
    ])
    assert img.shape == (r.h, r.w, 3)
    # The dot region differs from the empty-field render at the same spot.
    empty = r.render([])
    u, v = r.field_to_pixel((4.0, 4.0))
    patch = img[int(v) - 10:int(v) + 10, int(u) - 10:int(u) + 10]
    patch0 = empty[int(v) - 10:int(v) + 10, int(u) - 10:int(u) + 10]
    assert (patch != patch0).any()


def test_render_draws_structures_at_exact_positions():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    img = r.render([])
    fp = structure_footprints(fm, hub_prisms(fm))
    # The hub footprint edge must be inked (different from carpet background).
    hub = np.asarray(fp["hubs"][0], float)
    edge_xy = hub.mean(axis=0)  # centroid is inside the filled/outlined hub
    u, v = r.field_to_pixel(tuple(edge_xy))
    bg = img[5, r.w // 2 + 30].tolist()  # plain carpet pixel near top
    assert img[int(v), int(u)].tolist() != bg


def test_trail_renders_fading_polyline():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    trail = [(6.0 + 0.1 * i, 2.0) for i in range(10)]
    with_trail = r.render([_track(pos=(7.0, 2.0), trail=trail)])
    without = r.render([_track(pos=(7.0, 2.0), trail=())])
    u, v = r.field_to_pixel((6.2, 2.0))  # a point mid-trail, far from the dot
    assert (with_trail[int(v) - 3:int(v) + 3, int(u) - 3:int(u) + 3]
            != without[int(v) - 3:int(v) + 3, int(u) - 3:int(u) + 3]).any()


# --- Manta scouted-path overlay (ghost routes) ---
def _ref(**kw):
    base = dict(alliance="blue", team_number=1684, position=1,
                path_m=[(3.0, 2.0), (4.0, 2.0), (5.0, 2.0), (6.0, 2.0)])
    base.update(kw)
    return MantaAutoPath(**base)


def test_reference_path_marks_start_within_bounds():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    baseline = r.render([]).copy()
    r.set_reference_paths([_ref()])
    img = r.render([])
    assert img.shape == baseline.shape
    # A start marker is inked at the path's first point.
    u, v = r.field_to_pixel((3.0, 2.0))
    patch = img[int(v) - 8:int(v) + 8, int(u) - 8:int(u) + 8]
    patch0 = baseline[int(v) - 8:int(v) + 8, int(u) - 8:int(u) + 8]
    assert (patch != patch0).any()


def test_reference_path_draws_route():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    baseline = r.render([]).copy()
    r.set_reference_paths([_ref(path_m=[(3.0, 6.0), (7.0, 6.0)])])
    img = r.render([])
    # The ghost route inks several points along its length (dashed, so not all).
    hits = 0
    for x in (3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5):
        u, v = r.field_to_pixel((x, 6.0))
        if (img[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2]
                != baseline[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2]).any():
            hits += 1
    assert hits >= 3


def _max_turning(pts):
    """Sharpest single-vertex heading change — a jaggedness proxy.

    (Total turning is conserved under corner-cutting; the *max* per-vertex turn is
    what smoothing reduces, spreading each sharp corner over gentler ones.)
    """
    p = np.asarray(pts, float)
    seg = np.diff(p, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    d = np.abs(np.diff(ang))
    d = np.minimum(d, 2 * np.pi - d)  # wrap to [0, pi]
    return float(d.max())


def _grid_zigzag(x0=3.0, y0=2.0, step=0.17, teeth=10):
    """An axis-aligned staircase (up/right/down/right ...) like a grid-drawn path,
    with every corner a sharp 90 degrees — the 'up down up down' case."""
    pts, x = [(x0, y0)], x0
    for i in range(teeth):
        y = y0 + step if i % 2 == 0 else y0
        pts.append((x, y))        # vertical move
        x += step
        pts.append((x, y))        # horizontal move
    return pts


def test_smooth_arc_dezigzags_and_preserves_endpoints():
    from web.field.radar import _smooth_arc
    zig = _grid_zigzag()
    sm = _smooth_arc(zig)
    assert sm[0] == (zig[0][0], zig[0][1])       # endpoints pinned exactly
    assert sm[-1] == (zig[-1][0], zig[-1][1])
    assert _max_turning(zig) > 1.4               # raw has ~90-degree corners
    assert _max_turning(sm) < 0.8                # smoothed into gentle arcs
    # never wanders far from the drawn envelope
    xs = [p[0] for p in sm]
    ys = [p[1] for p in sm]
    assert min(x for x, _ in zig) - 0.3 <= min(xs) and max(xs) <= max(x for x, _ in zig) + 0.3
    assert min(y for _, y in zig) - 0.3 <= min(ys) and max(ys) <= max(y for _, y in zig) + 0.3


def test_smooth_arc_leaves_short_paths_untouched():
    from web.field.radar import _smooth_arc
    assert _smooth_arc([(1.0, 2.0), (3.0, 4.0)]) == [(1.0, 2.0), (3.0, 4.0)]


def test_reference_paths_smoothed_reduce_jaggedness():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    zig = _grid_zigzag()
    r.set_reference_paths([_ref(path_m=zig)])
    stored = r._reference[0]["pts_px"]
    raw_px = [r.field_to_pixel(xy) for xy in zig]
    assert _max_turning(stored) < _max_turning(raw_px)  # rendered route is smoother


def test_reference_paths_near_edge_do_not_crash():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    # Corner-to-corner path, exercising pixel coords at the image extremes.
    r.set_reference_paths([_ref(path_m=[(0.0, 0.0), (fm.length_m, fm.width_m)])])
    img = r.render([])
    assert img.shape == (r.h, r.w, 3)


def test_reference_path_is_wide_band():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    r.set_reference_paths([_ref(path_m=[(2.0, 7.0), (8.0, 7.0)], team_number=None)])
    img = r.render([]).astype(int)
    base = r._base.astype(int)
    changed = np.any(img != base, axis=2)
    # Measure the stroke in the MIDDLE of the route (cols 180-380), away from the
    # endpoint markers, whose rings/crosses would otherwise dominate the count.
    mid = changed[:200, 180:380]
    thickness = int(mid.sum(axis=0).max())  # thickest inked column (a dash-on spot)
    assert thickness >= 6     # ~4x the old 2px live-trail stroke


def test_reference_path_is_semi_transparent():
    fm = _fm()
    r = Radar(fm.length_m, fm.width_m, field_model=fm)
    r.set_reference_paths([_ref(path_m=[(2.0, 7.0), (8.0, 7.0)],
                                alliance="red", team_number=None)])
    img = r.render([]).astype(int)
    base = r._base.astype(int)
    full = np.array(_ALLIANCE["red"], int)
    dev = np.abs(img - base).sum(axis=2)
    dev[200:, :] = 0
    peak = np.unravel_index(int(dev.argmax()), dev.shape)
    # 50% opacity => the route core is halfway from its background to full color.
    expected_half = 0.5 * base[peak] + 0.5 * full
    assert np.abs(img[peak] - expected_half).max() <= 12
    assert np.abs(img[peak] - full).max() > 40    # and not fully opaque

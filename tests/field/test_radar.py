"""Field map v2: exact structure footprints + track-state rendering."""
import numpy as np

from web.field.field_model import FieldModel
from web.field.occlusion import hub_prisms
from web.field.overlay_geometry import structure_footprints
from web.field.radar import Radar, RadarTrack


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

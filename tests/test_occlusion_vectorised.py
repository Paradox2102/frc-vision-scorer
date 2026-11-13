"""The vectorised occluded_by_hub must be DECISION-IDENTICAL to the original
scalar per-sample _point_in_poly loop (it feeds tracking, and A/Bs rely on
byte-identical outputs). The reference below is a verbatim copy of the
pre-vectorisation code; the randomized test hammers both with real 2026 hub /
2025 reef prisms plus synthetic footprints, including targets on/near vertices
and edges, segments tangent to the polygon and segments running exactly along a
vertex's y level."""
import numpy as np
import pytest

from web.field.field_model import FieldModel
from web.field.occlusion import (Prism, _HUB_MARGIN_M, _HUB_GEOM_CACHE, hub_prisms,
                                 occluded_by_hub)


# ---------------------------------------------------------------- reference
# Verbatim copy of the ORIGINAL implementation (web/field/occlusion.py before the
# vectorisation). Do not "fix" or modernise it -- it is the oracle.

def _ref_point_in_poly(pt, poly):
    x, y = pt; inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _ref_expand_polygon(poly, margin):
    """Expand a convex polygon outward from its centroid by `margin` meters."""
    centroid = poly.mean(axis=0)
    vecs = poly - centroid
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return poly + vecs / (norms + 1e-9) * margin


def ref_occluded_by_hub(cam_center, prism, track_xy, n=100, margin=_HUB_MARGIN_M):
    cam_xy = np.asarray(cam_center, float)[:2]
    tgt_xy = np.array([track_xy[0], track_xy[1]], float)

    # Hub must be closer to the camera than the target
    hub_center = prism.footprint.mean(axis=0)
    d_hub = np.linalg.norm(cam_xy - hub_center)
    d_target = np.linalg.norm(cam_xy - tgt_xy)
    if d_hub >= d_target:
        return False

    expanded = _ref_expand_polygon(prism.footprint, margin)
    for t in np.linspace(0.0, 1.0, n):
        p = cam_xy + (tgt_xy - cam_xy) * t
        if _ref_point_in_poly(p, expanded):
            return True
    return False


# ---------------------------------------------------------------- fixtures

FIELD_L, FIELD_W = 16.54, 8.07


def _regular(cx, cy, r, k, rot):
    a = rot + 2 * np.pi * np.arange(k) / k
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _all_prisms():
    out = list(hub_prisms(FieldModel.load("2026-rebuilt-welded")))
    out += list(hub_prisms(FieldModel.load("2025-reefscape-welded")))
    assert len(out) == 4
    # synthetic hub-like footprints near field centre, incl. axis-aligned edges
    # (yi == yj exactly -> the y-test's degenerate case) and CW ordering
    out.append(Prism(_regular(8.27, 4.1, 0.6, 6, 0.0), 1.8))
    out.append(Prism(_regular(8.27, 4.1, 0.6, 8, np.pi / 8), 1.8))       # flat top/bottom
    out.append(Prism(_regular(8.27, 4.1, 0.6, 4, np.pi / 4)[::-1], 1.8))  # square, CW
    out.append(Prism(np.array([[7.7, 3.5], [8.9, 3.5], [8.9, 4.7], [7.7, 4.7]]), 1.8))
    return out


def _special_targets(prism, margin, rng):
    """Targets on / within ulps of expanded vertices and edges."""
    exp = _ref_expand_polygon(prism.footprint, margin)
    pts = [v for v in exp]
    k = len(exp)
    for i in range(k):
        a, b = exp[i], exp[(i + 1) % k]
        for s in (0.0, 0.5, 1.0, rng.random()):
            pts.append(a + (b - a) * s)
    out = []
    for p in pts:
        out.append(p)
        for eps in (1e-15, 1e-12, 1e-9, 1e-6):
            out.append(p + np.array([eps, 0.0])); out.append(p - np.array([eps, 0.0]))
            out.append(p + np.array([0.0, eps])); out.append(p - np.array([0.0, eps]))
        out.append(np.nextafter(p, p + 1)); out.append(np.nextafter(p, p - 1))
    return out


def _random_cam(rng):
    """Around/above the field perimeter, z 1-5 m (a few inside the field too)."""
    side = rng.integers(5)
    if side == 0:
        x, y = rng.uniform(-3, FIELD_L + 3), rng.uniform(-4, -0.2)
    elif side == 1:
        x, y = rng.uniform(-3, FIELD_L + 3), rng.uniform(FIELD_W + 0.2, FIELD_W + 4)
    elif side == 2:
        x, y = rng.uniform(-4, -0.2), rng.uniform(-3, FIELD_W + 3)
    elif side == 3:
        x, y = rng.uniform(FIELD_L + 0.2, FIELD_L + 4), rng.uniform(-3, FIELD_W + 3)
    else:
        x, y = rng.uniform(0, FIELD_L), rng.uniform(0, FIELD_W)
    return np.array([x, y, rng.uniform(1.0, 5.0)])


def _cases(rng, prisms, margin, count):
    """Yield (cam, prism, target) tuples mixing random, near-edge, tangent and
    horizontal-through-vertex geometry."""
    for _ in range(count):
        prism = prisms[rng.integers(len(prisms))]
        cam = _random_cam(rng)
        kind = rng.random()
        exp = _ref_expand_polygon(prism.footprint, margin)
        if kind < 0.45:                                    # anywhere on the field
            tgt = (rng.uniform(0, FIELD_L), rng.uniform(0, FIELD_W))
        elif kind < 0.65:                                  # on/near a vertex or edge
            sp = _special_targets(prism, margin, rng)
            tgt = tuple(sp[rng.integers(len(sp))])
        elif kind < 0.85:                                  # tangent: ray cam->vertex, past it
            v = exp[rng.integers(len(exp))]
            d = v - cam[:2]
            s = rng.choice([1.0, 1.0 + 1e-12, 1.5, 2.0, rng.uniform(1.0, 4.0)])
            tgt = tuple(cam[:2] + d * s)
        else:                                              # horizontal through a vertex y
            v = exp[rng.integers(len(exp))]
            cam = np.array([rng.choice([-1.0, FIELD_L + 1.0]), v[1], rng.uniform(1, 5)])
            tgt = (rng.uniform(0, FIELD_L), v[1])
        yield cam, prism, tgt


# ---------------------------------------------------------------- tests

def test_vectorised_matches_scalar_reference_default_n_and_margin():
    rng = np.random.default_rng(20261113)
    prisms = _all_prisms()
    n_cases = 0; n_true = 0; mismatches = []
    for cam, prism, tgt in _cases(rng, prisms, _HUB_MARGIN_M, 24000):
        a = occluded_by_hub(cam, prism, tgt)
        b = ref_occluded_by_hub(cam, prism, tgt)
        assert type(a) is bool
        n_cases += 1; n_true += b
        if a != b:
            mismatches.append((cam, prism.footprint, tgt, a, b))
    assert not mismatches, mismatches[:3]
    assert n_cases == 24000
    assert 0.03 * n_cases < n_true < 0.6 * n_cases   # both outcomes well exercised


@pytest.mark.parametrize("n,margin", [(1, 0.35), (2, 0.35), (7, 0.0), (50, 0.1),
                                      (100, 0.0), (100, 0.6), (257, 0.35)])
def test_vectorised_matches_scalar_reference_other_n_and_margins(n, margin):
    rng = np.random.default_rng(1000 + n + int(margin * 100))
    prisms = _all_prisms()
    mism = 0; n_true = 0
    for cam, prism, tgt in _cases(rng, prisms, margin, 2500):
        a = occluded_by_hub(cam, prism, tgt, n=n, margin=margin)
        b = ref_occluded_by_hub(cam, prism, tgt, n=n, margin=margin)
        n_true += b
        mism += a != b
    assert mism == 0
    if n >= 7:
        assert n_true > 0


def test_cache_is_keyed_on_footprint_content_not_identity():
    fp = _regular(8.27, 4.1, 0.6, 6, 0.0)
    prism = Prism(fp.copy(), 1.8)
    cam, tgt = (8.27, -2.0, 3.0), (8.27, 6.5)
    assert occluded_by_hub(cam, prism, tgt) is True
    # mutate in place: moved far away -> must miss the cache and stop occluding
    prism.footprint[:] = fp + np.array([5.0, 0.0])
    assert occluded_by_hub(cam, prism, tgt) is ref_occluded_by_hub(cam, prism, tgt) is False
    # reassign, recreate: identical answers to the reference
    prism.footprint = fp.copy()
    assert occluded_by_hub(cam, prism, tgt) is True
    assert occluded_by_hub(cam, Prism(fp.copy(), 1.8), tgt) is True


def test_cache_is_bounded():
    for k in range(200):
        occluded_by_hub((0.0, 0.0, 2.0), Prism(_regular(5.0 + k * 1e-3, 4.0, 0.5, 6, 0.0), 1.8),
                        (10.0, 8.0))
    assert len(_HUB_GEOM_CACHE) <= 64


def test_nan_target_matches_reference():
    prism = _all_prisms()[0]
    for tgt in [(np.nan, 4.0), (4.0, np.nan), (np.nan, np.nan)]:
        cam = (0.0, -2.0, 3.0)
        assert occluded_by_hub(cam, prism, tgt) == ref_occluded_by_hub(cam, prism, tgt)

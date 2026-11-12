"""Offline camera-coverage / blind-spot analysis (pure computation).

Rasterizes the field into cells and scores each cell's *effective camera
coverage*: the sum, across every calibrated view, of a per-camera weight that
factors in frustum membership, mask occlusion, hub occlusion, corner-camera
falloff, and close-range detector blindness. Consumed by
``scripts/coverage_map.py``; kept dependency-free of rendering libraries so it
can be unit tested directly.
"""
from __future__ import annotations

import numpy as np

from web.field.detect_bridge import (
    DET_MAX_BOX_FRAC,
    corner_detection_weight,
    expected_robot_pixel_width,
)
from web.field.calibration import visible_point
from web.field.occlusion import occluded_by_hub


def _in_frustum(calibration, x: float, y: float) -> bool:
    """Whether field point (x, y, 0.0) projects inside the camera's image bounds.

    field_to_image can raise or return nonsensical values for points behind the
    camera in pose modes (fisheye/pinhole), and a wide lens folds points far
    outside the view back INTO the image — none of that is "in view"
    (visible_point), so treat any such failure as outside the frustum rather than
    letting it propagate."""
    try:
        uv = visible_point(calibration, (x, y, 0.0))
    except Exception:
        return False
    if uv is None:
        return False
    u, v = uv
    if not (np.isfinite(u) and np.isfinite(v)):
        return False
    w, h = calibration.image_size
    return 0.0 <= u <= float(w) and 0.0 <= v <= float(h)


def _in_any_mask(calibration, x: float, y: float) -> bool:
    """Whether the field point's projection lands inside any of the camera's
    crop-local mask rects (x, y, w, h)."""
    if not calibration.masks:
        return False
    try:
        uv = calibration.field_to_image((x, y, 0.0))
    except Exception:
        return False
    if uv is None:
        return False
    u, v = uv
    for (mx, my, mw, mh) in calibration.masks:
        if mx <= u <= mx + mw and my <= v <= my + mh:
            return True
    return False


def cell_coverage(x: float, y: float, views: list, field_model, prisms: list) -> dict:
    """Per-camera effective weight (0..1) at field cell (x, y).

    Weight is a product of five factors: frustum membership, mask exclusion,
    hub occlusion, corner-camera falloff, and close-range detector blindness
    (the frac-backstop regime). Pose-free (homography) calibrations have no
    ``camera_center`` — occlusion and close-range terms are skipped (left at
    1.0) for those views, matching the live pipeline's behavior."""
    field_dims = (field_model.length_m, field_model.width_m)
    out: dict[str, float] = {}
    for view in views:
        cal = view.calibration
        weight = 1.0

        if not _in_frustum(cal, x, y):
            out[view.name] = 0.0
            continue

        if _in_any_mask(cal, x, y):
            out[view.name] = 0.0
            continue

        cam_center = getattr(cal, "camera_center", None)
        if cam_center is not None:
            for prism in prisms:
                if occluded_by_hub(cam_center, prism, (x, y)):
                    weight = 0.0
                    break

        if weight > 0.0:
            weight *= corner_detection_weight(view.name, (x, y), field_dims)

        if weight > 0.0 and cam_center is not None:
            expected_px = expected_robot_pixel_width((x, y), cal, 1.1)
            if expected_px is not None and expected_px > DET_MAX_BOX_FRAC * cal.image_size[0]:
                weight = 0.0

        out[view.name] = weight
    return out


def coverage_grid(views: list, field_model, cell_m: float = 0.25, prisms: list | None = None):
    """Rasterize the field into a ``cell_m``-spaced grid of total coverage.

    Returns (xs, ys, grid) where xs/ys are 1-D cell-center coordinate arrays
    (meters) and grid[j, i] is the summed per-camera weight at (xs[i], ys[j])
    — the sum of ``cell_coverage``'s per-camera values.

    ``prisms`` defaults to the field model's hub prisms (via
    :func:`web.field.occlusion.hub_prisms`), computed lazily so a minimal
    field-model stand-in (only ``length_m``/``width_m``, no AprilTag data —
    e.g. in tests) can skip hub occlusion by passing ``prisms=[]`` explicitly."""
    if prisms is None:
        from web.field.occlusion import hub_prisms  # local import: avoid a hard field_model dep at module load
        prisms = hub_prisms(field_model)
    n_x = max(1, int(round(field_model.length_m / cell_m)))
    n_y = max(1, int(round(field_model.width_m / cell_m)))
    xs = (np.arange(n_x) + 0.5) * cell_m
    ys = (np.arange(n_y) + 0.5) * cell_m

    grid = np.zeros((n_y, n_x), dtype=float)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            weights = cell_coverage(float(x), float(y), views, field_model, prisms)
            grid[j, i] = sum(weights.values())
    return xs, ys, grid

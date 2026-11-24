"""scripts/manta_sim_truth.py: exact sim camera -> our pinhole calibration.

Synthetic cameras only (no sim data files needed)."""
import cv2
import numpy as np
import pytest

import scripts.manta_sim_truth as mst


def _look_at_gl(centre, target):
    """4x4 field -> OpenGL camera (+x right, +y up, looking down -z), z-up world."""
    c, t = np.asarray(centre, float), np.asarray(target, float)
    fwd = (t - c) / np.linalg.norm(t - c)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, -R @ c
    return M


def _cam(panel=(0, 750, 937, 330), hfov=78.0, k=0.12, centre=(1.0, -0.05, 0.5),
         target=(3.2, 4.6, 0.6)):
    w = panel[2]
    fx = (w / 2) / np.tan(np.radians(hfov) / 2)
    return mst.SimCamera("corner_left", "blueCorner", panel, fx, fx, k,
                         _look_at_gl(centre, target))


def test_lens_inverse_roundtrip():
    cam = _cam()
    rng = np.random.default_rng(0)
    p_out = rng.uniform(-1, 1, (500, 2)) * [cam.aspect, 1.0]
    back = cam.lens_out_from_src(cam.lens_src_from_out(p_out))
    assert np.abs(back - p_out).max() < 1e-9
    # r = 1 at the panel corner: the lens leaves the corner where it is
    corner = np.array([[cam.aspect, 1.0]])
    assert np.allclose(cam.lens_src_from_out(corner), corner)


@pytest.mark.parametrize("k", [0.0, 0.04, 0.12])
def test_converted_calibration_matches_exact_projection(k):
    cam = _cam(k=k)
    cal, fit = mst.to_calibration(cam, "2026-rebuilt-welded")
    assert fit["max_px"] < 0.1
    assert fit["fold_radius"] > fit["corner_radius"]
    # OpenGL -> OpenCV: same camera centre, paraxial focal fx (1 + k)
    assert np.allclose(cal.camera_center, cam.centre, atol=1e-9)
    assert cal.K[0, 0] == pytest.approx(cam.fx * (1 + k))
    # field points in front of the camera, inside the panel
    rng = np.random.default_rng(1)
    P = np.column_stack([rng.uniform(1.5, 8, 400), rng.uniform(0, 8, 400),
                         rng.uniform(0, 1.2, 400)])
    exact, front = cam.project(P)
    x, y, w, h = cam.panel
    inside = front & (exact[:, 0] >= x) & (exact[:, 0] < x + w) \
        & (exact[:, 1] >= y) & (exact[:, 1] < y + h)
    assert inside.sum() > 50
    ours, ok = mst.cal_project_full(cal, P)
    assert ok[inside].all()
    assert np.abs(ours[inside] - exact[inside]).max() < 0.1
    # and back: pixel -> field on the point's own height plane
    for (u, v), p in list(zip(exact[inside], P[inside]))[:20]:
        xy = cal.image_to_field((u - x, v - y), z=p[2])
        assert np.hypot(xy[0] - p[0], xy[1] - p[1]) < 0.01


def test_opengl_to_opencv_flip():
    cam = _cam(k=0.0)
    R_cv = cam.R_cv
    assert np.allclose(R_cv @ R_cv.T, np.eye(3))
    assert np.linalg.det(R_cv) == pytest.approx(1.0)
    # a point straight ahead is at +z (OpenCV) and projects to the panel centre
    ahead = cam.centre + 5 * (R_cv.T @ np.array([0, 0, 1.0]))
    px, front = cam.project(ahead[None])
    assert front[0]
    assert np.allclose(px[0], [cam.panel[0] + cam.panel[2] / 2, cam.panel[1] + cam.panel[3] / 2])
    rvec, _ = cv2.Rodrigues(R_cv)
    assert np.allclose(cv2.Rodrigues(rvec)[0], R_cv)

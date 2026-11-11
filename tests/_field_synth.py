"""Shared synthetic fisheye camera for field-calibration tests."""
import cv2
import numpy as np
from web.field.calibration import CameraCalibration

IMAGE_SIZE = (1920, 1080)
TRUTH_K = np.array([[900., 0., 960.], [0., 900., 540.], [0., 0., 1.]])
TRUTH_D = np.array([-0.05, 0.01, 0.0, 0.0])      # mild fisheye

def look_at(eye, target, world_up=(0., 0., 1.)):
    """OpenCV-convention extrinsics: camera +z = forward, +x = right, +y = down."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    # x = forward x up (camera +x = image right, +y = image down): an UPRIGHT camera.
    # (up x forward gave a camera rolled 180 deg — upside down — which the pose
    # plausibility gate in web.field.calibration rightly rejects.)
    x = np.cross(z, np.asarray(world_up, float)); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])                      # world -> camera
    rvec, _ = cv2.Rodrigues(R)
    tvec = (-R @ eye).reshape(3, 1)
    return rvec, tvec

def truth_camera(eye=(8.27, -3.0, 6.0), target=(8.27, 4.03, 0.0),
                 K=TRUTH_K, D=TRUTH_D):
    """Camera behind/above the scoring table (y<0) looking across the field."""
    rvec, tvec = look_at(eye, target)
    return CameraCalibration(K=K.copy(), D=D.copy(), rvec=rvec, tvec=tvec,
                             image_size=IMAGE_SIZE, crop_region=None,
                             field_season="2026-rebuilt-welded")

# Pinhole truth camera for single-view calibration tests. Principal point MUST be
# at image center (cx=w/2, cy=h/2) because solve_camera_from_tags FIXES it there.
TRUTH_K_PINHOLE = np.array([[1200., 0., 960.], [0., 1200., 540.], [0., 0., 1.]])
TRUTH_D_PINHOLE = np.array([-0.25, -0.08, 0.0, 0.0, 0.0])   # k1,k2,p1,p2,k3 (mild radial)


def truth_pinhole_camera(eye=(8.27, -3.0, 6.0), target=(8.27, 4.03, 0.0),
                         K=TRUTH_K_PINHOLE, D=TRUTH_D_PINHOLE,
                         image_size=(1920, 1080)):
    """Pinhole camera above/behind the scoring table looking across the field.
    cx/cy are image-centered to match the solver's fixed principal point."""
    rvec, tvec = look_at(eye, target)
    return CameraCalibration(mode="pinhole", K=K.copy(), D=D.copy(),
                             rvec=rvec, tvec=tvec, image_size=image_size,
                             crop_region=(0, 0, image_size[0], image_size[1]),
                             field_season="2026-rebuilt-welded")


def project_tag_corners(cam, fm):
    """{tag_id: (corners_2d (4,2), corners_3d (4,3))} for tags fully in-frame."""
    w, h = cam.image_size
    corr = {}
    for tid in fm.tags:
        c3d = fm.tag_corners_3d(tid)
        uv = np.array([cam.field_to_image(p) for p in c3d])
        if np.all((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)):
            corr[tid] = (uv, c3d)
    return corr

def in_frame_points(cam, candidates):
    w, h = cam.image_size
    keep = []
    for x, y in candidates:
        u, v = cam.field_to_image((x, y, 0.0))
        if 0 <= u < w and 0 <= v < h:
            keep.append((x, y))
    return keep

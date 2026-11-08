"""bumper_color_read: per-frame main-view bumper colour read
(web/field/bumper_colour.py) + its job knob / app wiring."""
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from web.field import bumper_colour as bc
from web.field.calibration import CameraCalibration, push_from_camera
from web.field.tracker import FieldDetection
from web.job_config import JobConfig

W, H = 1920, 723
GRAY = (128, 128, 128)
RED = (0, 0, 220)     # BGR
BLUE = (220, 60, 0)   # BGR, hue ~113 in OpenCV HSV


def _look_at_cal(cam=(8.0, -5.0, 3.3), target=(8.0, 4.0, 0.0), size=(W, H)):
    cam, target = np.array(cam, float), np.array(target, float)
    z = target - cam; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    rvec, _ = cv2.Rodrigues(R)
    tvec = (-R @ cam).reshape(3, 1)
    K = np.array([[1300.0, 0, size[0] / 2], [0, 1300.0, size[1] / 2], [0, 0, 1]])
    return CameraCalibration(K=K, D=np.zeros(5), rvec=rvec, tvec=tvec, image_size=size,
                             field_season="test", mode="pinhole")


CAL = _look_at_cal()
POS = (8.0, 3.0)   # near-bumper-edge ground point (pipeline pos)


def _blank():
    img = np.empty((H, W, 3), np.uint8)
    img[:] = GRAY
    return img


def _hull(pos=POS, offset=bc.DEFAULT_OFFSET_M):
    return bc.band_poly(CAL, push_from_camera(CAL, pos, offset))


def _robot_frame(colour, pos=POS):
    img = _blank()
    cv2.fillConvexPoly(img, _hull(pos), colour)
    return img


# --- the read itself -------------------------------------------------------

def test_band_projects_on_image_where_the_robot_is():
    hull = _hull().reshape(-1, 2)
    assert hull.shape[0] >= 4
    assert (hull[:, 0] > 0).all() and (hull[:, 0] < W).all()
    assert (hull[:, 1] > 0).all() and (hull[:, 1] < H).all()


@pytest.mark.parametrize("colour,name", [(RED, "red"), (BLUE, "blue")])
def test_reads_red_and_blue_band(colour, name):
    out, r, b = bc.read_bumper_colour(_robot_frame(colour), CAL, POS, None)
    assert out == name
    assert max(r, b) > 0.9 and min(r, b) < 0.01


def test_push_matters_band_found_at_robot_centre_only():
    # Robot drawn at the pushed centre; reading 2 m away sees nothing.
    img = _robot_frame(RED)
    assert bc.read_bumper_colour(img, CAL, (8.0, 6.0), None)[0] == "unknown"
    # offset 0 with a pos that IS the centre reads the same robot.
    centre = push_from_camera(CAL, POS, bc.DEFAULT_OFFSET_M)
    assert bc.read_bumper_colour(img, CAL, centre, None, offset_m=0.0)[0] == "red"


def test_undecided_gray_and_split():
    out, r, b = bc.read_bumper_colour(_blank(), CAL, POS, None)
    assert out == "unknown" and r == 0 and b == 0
    # Half red / half blue: margin < 0.04 -> unknown.
    img = _blank()
    hull = _hull()
    cv2.fillConvexPoly(img, hull, RED)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # pick the split column so both halves hold the same pixel count
    cum = np.cumsum((mask > 0).sum(axis=0))
    split = int(np.searchsorted(cum, cum[-1] / 2))
    img[:, split:][mask[:, split:] > 0] = BLUE
    out, r, b = bc.read_bumper_colour(img, CAL, POS, None)
    assert out == "unknown" and abs(r - b) < 0.04 and r > 0.3


def test_weak_colour_below_min_frac_is_unknown():
    img = _blank()
    hull = _hull()
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    ys, xs = np.nonzero(mask)
    k = int(len(ys) * 0.03)          # 3% red speckle < 6% floor
    img[ys[:k], xs[:k]] = RED
    assert bc.read_bumper_colour(img, CAL, POS, None)[0] == "unknown"


def test_background_mask_excludes_static_blue_ramp():
    hull = _hull()
    x, y, w, h = cv2.boundingRect(hull)
    ramp = (x, y, int(w * 0.65), h)   # static blue ramp under 65% of the band

    def with_ramp(img):
        cv2.rectangle(img, (ramp[0], ramp[1]),
                      (ramp[0] + ramp[2], ramp[1] + ramp[3]), BLUE, -1)
        return img

    # Background frames: the ramp always; a red "robot" passing through a few.
    frames = []
    for i in range(40):
        f = with_ramp(_blank())
        if i % 10 == 0:
            cv2.rectangle(f, (x + 20, y), (x + w, y + h), RED, -1)
        frames.append(f)
    bg = bc.build_background(frames)
    assert bg.shape == (H, W) and bg.dtype == bool
    assert bg[ramp[1] + ramp[3] // 2, ramp[0] + 5]            # ramp masked
    assert not bg[y + h // 2, x + w - 3]                      # robot not baked in

    # Live frame: red bumper, the blue ramp still visible through most of it.
    img = _robot_frame(RED)
    with_ramp(img)
    assert bc.read_bumper_colour(img, CAL, POS, None)[0] == "blue"   # fooled
    out, r, b = bc.read_bumper_colour(img, CAL, POS, bg)
    assert out == "red" and b < 0.01


def test_background_fully_masked_band_is_unknown():
    bg = np.ones((H, W), bool)
    assert bc.read_bumper_colour(_robot_frame(RED), CAL, POS, bg)[0] == "unknown"


def test_build_background_empty():
    assert bc.build_background([]) is None


def test_off_image_behind_camera_and_no_pose_are_unknown():
    img = _robot_frame(RED)
    # Far to the side: projects off-image.
    assert bc.read_bumper_colour(img, CAL, (-40.0, 3.0), None)[0] == "unknown"
    # Behind the camera (camera at y=-5 looking +y).
    assert bc.read_bumper_colour(img, CAL, (8.0, -12.0), None)[0] == "unknown"
    # Pose-free (homography) calibration.
    hom = CameraCalibration(mode="homography", H=np.eye(3), field_season="test")
    assert bc.read_bumper_colour(img, hom, POS, None)[0] == "unknown"
    # Missing inputs.
    assert bc.read_bumper_colour(None, CAL, POS, None)[0] == "unknown"
    assert bc.read_bumper_colour(img, CAL, None, None)[0] == "unknown"


def test_partly_off_image_still_reads():
    # Robot near the left image edge: hull clipped, remaining pixels read.
    for xpos in np.arange(2.0, -4.0, -0.25):
        hull = _hull((float(xpos), 3.0)).reshape(-1, 2)
        if hull[:, 0].min() < 0 < hull[:, 0].max():
            break
    else:
        pytest.skip("no straddling position found")
    pos = (float(xpos), 3.0)
    assert bc.read_bumper_colour(_robot_frame(RED, pos), CAL, pos, None)[0] == "red"


def test_read_cost_per_detection_is_small():
    img = _robot_frame(RED)
    bg = np.zeros((H, W), bool)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        bc.read_bumper_colour(img, CAL, POS, bg)
    per = (time.perf_counter() - t0) / n
    assert per < 0.01, per    # ~0.1-0.3 ms typical


# --- mode semantics --------------------------------------------------------

def _det(color, main=True, pos=POS):
    return FieldDetection(pos=pos, color=color, box=(1, 2, 3, 4) if main else None)


def test_override_mode_semantics():
    img = _robot_frame(RED)
    stale = _det("blue")                       # known but wrong -> flipped
    unk = _det("unknown")                      # unknown -> red
    corner = _det("unknown", main=False)       # corner-only: untouched
    elsewhere = _det("blue", pos=(3.0, 6.0))   # undecided read: keeps pipeline
    st = bc.BumperColourStats()
    n = bc.apply_bumper_colour("override", [stale, unk, corner, elsewhere],
                               img, CAL, None, bc.DEFAULT_OFFSET_M, stats=st)
    assert (stale.color, unk.color, corner.color, elsewhere.color) == \
        ("red", "red", "unknown", "blue")
    assert n == 2
    c = st.as_dict()
    assert c["reads"] == 3 and c["red"] == 2 and c["undecided"] == 1
    assert c["flipped"] == 1 and c["unknown_to_known"] == 1 and c["cleared"] == 0


def test_strict_mode_semantics():
    img = _robot_frame(RED)
    stale = _det("blue")
    corner = _det("blue", main=False)
    elsewhere = _det("blue", pos=(3.0, 6.0))   # undecided -> unknown in strict
    agree = _det("red")
    st = bc.BumperColourStats()
    bc.apply_bumper_colour("strict", [stale, corner, elsewhere, agree],
                           img, CAL, None, bc.DEFAULT_OFFSET_M, stats=st)
    assert (stale.color, corner.color, elsewhere.color, agree.color) == \
        ("red", "blue", "unknown", "red")
    c = st.as_dict()
    assert c["cleared"] == 1 and c["flipped"] == 1 and c["agree"] == 1


def test_off_mode_is_noop():
    img = _robot_frame(RED)
    d = _det("blue")
    assert bc.apply_bumper_colour("off", [d], img, CAL, None, 0.475) == 0
    assert d.color == "blue"


# --- knob + app wiring -----------------------------------------------------

def test_knob_default_strict_and_validated():
    assert JobConfig().bumper_color_read == "strict"           # default flipped Nov 7 2026
    assert JobConfig({"bumper_color_read": "off"}).bumper_color_read == "off"
    assert JobConfig({"bumper_color_read": "strict"}).bumper_color_read == "strict"
    with pytest.raises(ValueError):
        JobConfig({"bumper_color_read": "sometimes"})


def _job(mode, ground_offset=0.0):
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)
    job.config = SimpleNamespace(bumper_color_read=mode,
                                 ground_center_offset_m=ground_offset)
    job.views = [SimpleNamespace(name="main", crop=(0, 0, W, H), calibration=CAL)]
    job.timings = {}
    job.processing_ranges = [(0, 10)]
    from collections import defaultdict
    job._loop_stage_s = defaultdict(float)
    return job


def _composite_with(main_img):
    comp = np.zeros((H + 300, W, 3), np.uint8)   # composite: main on top
    comp[:H] = main_img
    return comp


def test_app_default_off_leaves_colours_untouched():
    import asyncio
    job = _job("off")
    asyncio.run(job._setup_bumper_colour_read())
    assert job._bc_ready is False
    d = _det("blue")
    job._apply_bumper_colour_read([d], _composite_with(_robot_frame(RED)))
    assert d.color == "blue"
    job._finalize_bumper_colour_stats()
    assert job.timings == {}


def test_app_override_wiring_and_ground_offset_remainder(monkeypatch):
    import asyncio
    job = _job("override")
    monkeypatch.setattr(job, "_build_bumper_colour_background",
                        lambda n_frames=40: (np.zeros((H, W), bool), 40, 0.5))
    asyncio.run(job._setup_bumper_colour_read())
    assert job._bc_ready is True
    comp = _composite_with(_robot_frame(RED))
    d, corner = _det("blue"), _det("blue", main=False)
    job._apply_bumper_colour_read([d, corner], comp, early=True)
    assert d.color == "red" and corner.color == "blue"
    # ground_center_offset_m already pushed pos to the centre: only the
    # remainder (0.475 - 0.475 = 0) is applied, so the centre reads red.
    job2 = _job("override", ground_offset=0.475)
    job2._bc_ready, job2._bc_bg = True, None
    job2._bc_stats = bc.BumperColourStats()
    d2 = _det("blue", pos=push_from_camera(CAL, POS, 0.475))
    job2._apply_bumper_colour_read([d2], comp)
    assert d2.color == "red"
    job._finalize_bumper_colour_stats()
    assert job.timings["bumper_colour.reads"] == 1
    assert job.timings["bumper_colour.early_reads"] == 1
    assert job.timings["bumper_colour.flipped"] == 1
    assert job.timings["bumper_colour.bg_build"] == 0.5

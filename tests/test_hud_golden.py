"""Golden image of the live HUD (docs/superpowers/specs/2026-11-10-hud-post-render-design.md
section 8). One synthetic frame with 2 robots (one with a SAM mask), a coasting
ghost, 2 hubs, a hub box, a skipped ball and robot detection, OCR regions, the
wireframe off, a radar with a trail and a scouted route, and 2 corner views (one
detection near a slot, one not). tests/data/hud_live_golden.png was captured from
the analysis loop's inline drawing code BEFORE the hud_draw refactor; the live
HUD (InferenceJob._live_hud, which the loop calls) must keep matching it byte
for byte."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv

import web.app as app_mod
from web.app import DetectedHub, InferenceJob, RobotTrack
from web.field.camera_view import CameraView
from web.field.field_model import FieldModel
from web.field.radar import Radar
from web.field.tracker import FieldDetection, TrackOut

GOLDEN = Path(__file__).parent / "data" / "hud_live_golden.png"
FRAME_NUM = 1234
DEFAULT_FLAGS = {"HUD_DRAW_ROBOT_MASK": True, "HUD_DRAW_BUMPER_BAND": True,
                 "HUD_DRAW_ROBOT_LABEL": True, "HUD_DRAW_ROBOT_BOX": False,
                 "HUD_DRAW_GROUND_PLANE": False, "HUD_DRAW_VELOCITY_ARROW": False,
                 "HUD_DRAW_BALL_BOXES": False, "HUD_DRAW_COASTING_GHOST": True}


def _composite():
    """640x360 composite: main view on top (240 rows), two corner views below."""
    yy, xx = np.mgrid[0:360, 0:640]
    img = np.stack([(xx * 0.3 + 40) % 256, (yy * 0.5 + 60) % 256,
                    ((xx + yy) * 0.2 + 90) % 256], axis=-1).astype(np.uint8)
    cv2.rectangle(img, (100, 80), (160, 150), (90, 90, 90), -1)     # robot B1
    cv2.rectangle(img, (400, 90), (470, 170), (60, 60, 70), -1)     # robot R2
    cv2.circle(img, (200, 120), 18, (200, 200, 200), -1)            # hub
    cv2.rectangle(img, (40, 262), (100, 322), (80, 80, 80), -1)     # corner_left robot
    return img


def _robot(rid, alliance, team=None, confirmed=False, scores=0, votes=None):
    return RobotTrack(robot_id=rid, alliance=alliance, team_number=team,
                      last_position=(0.0, 0.0), positions_history=deque(),
                      scores=scores, team_ocr_votes=votes,
                      team_number_confirmed=confirmed)


def scene():
    """(job, kwargs) for one HUD frame. Fresh objects every call (the HUD code
    mutates the radar trails)."""
    job = InferenceJob.__new__(InferenceJob)
    composite = _composite()
    overhead = composite[:240, :]
    main_cal = SimpleNamespace(robot_track_top_crop=0.0)
    job.views = [
        SimpleNamespace(name="main", crop=(0, 0, 640, 240), masks=(), calibration=main_cal),
        CameraView(name="corner_left", crop=(0, 232, 320, 128), masks=(),
                           calibration=None),
        CameraView(name="corner_right", crop=(320, 232, 320, 128),
                           masks=((250, 0, 70, 20),), calibration=None),
    ]
    job.calib = None
    job.field_model = None
    job.config = SimpleNamespace(field_wireframe_overlay=False)
    job.use_sam2 = True
    job.sam2_source_to_local = {FRAME_NUM: 0}
    job.field_tracker = SimpleNamespace(_id_to_slot={7: 0})
    mask2d = np.zeros((120, 320), np.uint8)
    cv2.ellipse(mask2d, (65, 57), (16, 18), 0, 0, 360, 1, -1)
    job.sam2_raw = {"main": {0: {3: {"mask": mask2d.astype(bool)}}}}
    job.hubs = [DetectedHub(x=200.0, y=120.0, alliance="blue", x1=182, y1=102, x2=218, y2=138),
                DetectedHub(x=440.0, y=60.0, alliance="red", x1=425, y1=45, x2=455, y2=75)]
    job.ocr_regions = {"blue_score": (20, 10, 60, 20), "red_score": (560, 10, 60, 20),
                       "match_clock": (290, 5, 60, 20)}
    fm = FieldModel.load("2026-rebuilt-welded")
    job.radar = Radar(fm.length_m, fm.width_m, field_model=fm)
    job.radar.set_reference_paths([SimpleNamespace(
        alliance="blue", team_number="3128", path_m=[(2.0, 2.0), (3.0, 2.5), (4.0, 3.0)])])
    job._radar_trails = {0: deque([(2.0, 2.0), (2.5, 2.0)], maxlen=16),
                         5: deque([(9.0, 1.0)], maxlen=16)}
    job.identity_registry = None
    job.robot_tracks = {
        0: _robot(0, "blue", team="3128", confirmed=True, scores=2),
        2: _robot(2, "blue", scores=0),
        4: _robot(4, "red", scores=1, votes={"2637": 5.0}),
    }
    job._get_slot_track_if_exists = lambda sid: job.robot_tracks.get(sid)
    job._get_or_create_slot_track = lambda sid: job.robot_tracks.get(sid)
    track_outs = [
        TrackOut(slot_id=0, alliance="blue", pos=(3.0, 2.0), vel=(0.5, 0.0),
                 box=(100, 80, 160, 150), state="visible"),
        TrackOut(slot_id=4, alliance="red", pos=(12.0, 5.0), vel=(0.0, 0.0),
                 box=(400, 90, 470, 170), state="visible"),
        TrackOut(slot_id=2, alliance="blue", pos=(6.0, 1.0), vel=(0.0, 0.0),
                 box=(250, 60, 300, 110), state="coasting"),
        TrackOut(slot_id=3, alliance="red", pos=(10.0, 6.0), vel=(0.0, 0.0),
                 box=None, state="visible", corner_only=True),
        TrackOut(slot_id=1, alliance="blue", pos=(0.0, 0.0), vel=(0.0, 0.0),
                 box=None, state="empty"),
    ]
    detections = sv.Detections(
        xyxy=np.array([[182, 102, 218, 138], [300, 150, 310, 160], [50, 50, 90, 90]], float),
        class_id=np.array([2, 0, 1]), confidence=np.array([0.9, 0.5, 0.8]))
    per_camera = {
        "corner_left": [(FieldDetection(pos=(3.2, 2.1), color="blue", box=(40, 30, 100, 90)),
                         None)],
        "corner_right": [(FieldDetection(pos=(15.0, 7.0), color="red", box=(200, 40, 260, 100)),
                          None)],
    }
    fused = [SimpleNamespace(track_id=7, members=[("main", 3)])]
    kwargs = dict(overhead_frame=overhead, frame=composite, detections=detections,
                  tracked_robot_boxes=[(0, (100, 80, 160, 150)), (4, (400, 90, 470, 170))],
                  track_outs=track_outs, per_camera=per_camera, fused=fused,
                  frame_num=FRAME_NUM)
    return job, kwargs


def golden() -> np.ndarray:
    img = cv2.imread(str(GOLDEN), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"missing {GOLDEN}"
    return img


def test_live_hud_matches_the_golden_image(monkeypatch):
    for k, v in DEFAULT_FLAGS.items():
        monkeypatch.setattr(app_mod, k, v)
    job, kw = scene()
    out = job._live_hud(**kw)
    want = golden()
    assert out.shape == want.shape and out.dtype == want.dtype
    assert np.array_equal(out, want)

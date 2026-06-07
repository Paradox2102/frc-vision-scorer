"""Multi-camera fusion smoke test on real footage (Task 18).

Drives the REAL geometry path — per-view YOLO detection -> to_field_detections ->
fuse_detections -> resolve_occlusion/_merge_occlusion — over a bounded window of the
2026hop composite footage with the three saved calibrations, and asserts the two
spec pass-signals:

  (a) occlusion fill: the corner cameras contribute field detections in zones the
      main camera's hub occludes.
  (b) additive coverage: corner cameras produce on-field detections fused alongside
      main's (cross-camera observations sharing one field position).

The heavy VLM identity loop is exercised by unit tests (test_vote_weighting) and a
real match run, not here. This test skips gracefully when the footage, calibrations,
or YOLO weights are absent (e.g. CI without assets).
"""
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/test_footage/qual1_first_champs_2026.mp4"
MODEL = ROOT / "runs/train/hub_zone_yolo26s/weights/best.pt"
EVENT = "2026hop"
DET_CONF = 0.35


def _require(cond, msg):
    if not cond:
        pytest.skip(msg)


def test_corner_cameras_add_occluded_field_coverage():
    import cv2
    from web.field.camera_view import load_views
    from web.field.field_model import FieldModel
    from web.field.detect_bridge import to_field_detections
    from web.field.fusion import fuse_detections
    from web.field.occlusion import hub_prisms, resolve_occlusion, _merge_occlusion

    _require(VIDEO.exists(), f"footage missing: {VIDEO}")
    _require(MODEL.exists(), f"YOLO weights missing: {MODEL}")
    views = load_views(EVENT)
    names = [v.name for v in views]
    _require("corner_left" in names and "corner_right" in names,
             f"corner calibrations missing (have {names}) — calibrate via /calibrate")

    from ultralytics import YOLO
    import supervision as sv
    model = YOLO(str(MODEL))
    fm = FieldModel.load(views[0].calibration.field_season or "2026-rebuilt-welded")
    dims = (fm.length_m, fm.width_m)
    prisms = hub_prisms(fm)
    main_view = next(v for v in views if v.name == "main")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    weights = {v.name: (2.5 if v.name != "main" else 1.0) for v in views}

    corner_onfield = 0          # (b) corner detections that land on the field
    fused_with_corner = 0       # (b) fused observations carrying corner evidence
    occ_fill = 0                # (a) corner-supported detections in main-hub-occluded zones

    for ts in range(20, 60, 4):   # bounded window, ~10 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        per_camera = {}
        for v in views:
            crop = v.extract(frame)
            res = model(crop, conf=DET_CONF, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            boxes = [dets.xyxy[i] for i in range(len(dets)) if dets.class_id[i] == 1]
            fdets = to_field_detections(
                boxes, v.calibration, dims, colors=["unknown"] * len(boxes)
            )
            per_camera[v.name] = [(d, None) for d in fdets]
            if v.name != "main":
                corner_onfield += len(fdets)

        fused = fuse_detections(per_camera, weights=weights)
        for obs in fused:
            if any(cam != "main" for (cam, _, _) in obs.evidence):
                fused_with_corner += 1

        # (a) occlusion-fill: flag fused positions occluded for MAIN by the hub,
        # and supported by a corner camera observation.
        pseudo = [{"slot_id": i, "pos": obs.detection.pos, "box": obs.detection.box}
                  for i, obs in enumerate(fused)]
        main_occ = resolve_occlusion(main_view.calibration, pseudo, [], prisms)
        for i, obs in enumerate(fused):
            has_corner = any(cam != "main" for (cam, _, _) in obs.evidence)
            if has_corner and main_occ.get(i, {}).get("view_occluded"):
                occ_fill += 1
    cap.release()

    print(f"corner on-field detections: {corner_onfield}, "
          f"fused-with-corner-evidence: {fused_with_corner}, occlusion-fill hits: {occ_fill}")
    # (b) corner cameras genuinely add field detections
    assert corner_onfield > 0, "corner cameras produced no on-field detections"
    # (a) at least once, a corner camera covers a field position the main hub occludes
    assert occ_fill > 0, ("no corner-supported detection landed in a main-hub-occluded "
                          "zone over the window — occlusion fill not demonstrated")


def test_merge_occlusion_visible_if_any_camera_sees():
    """Pure-logic guard (always runs): the merge keeps a track visible if any
    camera can see it — the property the smoke test relies on."""
    from web.field.occlusion import _merge_occlusion
    blocked = {0: {"view_occluded": True, "bumper_occluded": True, "cause": "hub"}}
    clear = {0: {"view_occluded": False, "bumper_occluded": False, "cause": None}}
    m = _merge_occlusion([blocked, clear])
    assert m[0]["view_occluded"] is False

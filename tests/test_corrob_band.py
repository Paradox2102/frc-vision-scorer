"""_sam2_robot_candidates corroborate-only band (corrob_conf_floor).

At re-prompt/seed boundaries the bumper detector normally runs at ``conf``
(or ``det_conf_floor`` when motion/zone-gated). corrob_conf_floor pushes the
run confidence lower still and captures the extra sub-conf detections the
motion/zone probes did NOT already admit as a separate corroborate-only
band: presence evidence that can never seed, match, re-anchor, or evict
(see web/job_config.py). With corrob_floor=0.0 (default) the band must be
empty and the primary return value byte-identical to the pre-Task-2 shape.
"""
import numpy as np
import supervision as sv
from unittest.mock import MagicMock, patch

from web.app import InferenceJob
from web.field.detect_schema import DetectionSchema


def _make_job():
    with patch("web.app.YOLO"):
        job = InferenceJob("test_job", "/mock/video.mp4")
    # No whole-robot fills, no persons, no motion/blob sources.
    job._detect_whole_robots = MagicMock(return_value=([], []))
    job.person_model = MagicMock(return_value=[MagicMock()])
    empty_person_dets = MagicMock(spec=sv.Detections)
    empty_person_dets.xyxy = np.zeros((0, 4))
    empty_person_dets.__len__.return_value = 0
    job.det_schema = DetectionSchema.from_names(
        {0: "blue_bumper", 1: "red_bumper", 2: "whole_robot"})
    job.config.motion_gated_conf = False
    job.config.motion_seed_blobs = False
    job.config.start_zone_seed = False

    def _person_from_ultralytics(_res, _empty=empty_person_dets):
        return _empty
    return job, _person_from_ultralytics


def _bumper_detections(boxes, class_ids, confidences):
    dets = MagicMock(spec=sv.Detections)
    dets.xyxy = np.array(boxes, dtype=float)
    dets.class_id = np.array(class_ids)
    dets.confidence = np.array(confidences, dtype=float)
    dets.__len__.return_value = len(boxes)
    return dets


# A wide, bumper-shaped box (width/height >= 0.5) so it clears
# `bumper_admissible`'s shape gate without needing a whole-robot overlap.
BOX_STRONG = (100.0, 300.0, 220.0, 340.0)   # conf 0.6 -> strong tier
BOX_CORROB = (400.0, 300.0, 520.0, 340.0)   # conf 0.15 -> corrob band only


def test_corrob_floor_zero_yields_empty_band_and_unchanged_result():
    job, person_from_ultralytics = _make_job()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = _bumper_detections(
        [BOX_STRONG, BOX_CORROB], [0, 0], [0.6, 0.15])
    job.bumper_model = MagicMock(
        return_value=[MagicMock()])

    with patch("supervision.Detections.from_ultralytics",
               side_effect=[person_from_ultralytics(None), dets]):
        result = job._sam2_robot_candidates(frame, conf=0.4)
        # Legacy call shape: a plain list, no band.
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0][2] is True  # strong

    with patch("supervision.Detections.from_ultralytics",
               side_effect=[person_from_ultralytics(None), dets]):
        result2, band = job._sam2_robot_candidates(
            frame, conf=0.4, corrob_floor=0.0, with_corrob_band=True)
        assert band == []
        assert result2 == result


def test_corrob_floor_captures_sub_conf_band_not_admitted_elsewhere():
    job, person_from_ultralytics = _make_job()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = _bumper_detections(
        [BOX_STRONG, BOX_CORROB], [0, 0], [0.6, 0.15])
    job.bumper_model = MagicMock(return_value=[MagicMock()])

    with patch("supervision.Detections.from_ultralytics",
               side_effect=[person_from_ultralytics(None), dets]):
        result, band = job._sam2_robot_candidates(
            frame, conf=0.4, corrob_floor=0.1, with_corrob_band=True)

    # Strong candidate still comes back through the normal tiered list.
    assert len(result) == 1
    assert result[0][2] is True
    # The sub-conf detection rides the corrob band instead, expanded to a
    # robot box, as (box, color) pairs (no `strong` flag).
    assert len(band) == 1
    box, color = band[0]
    assert color == "blue"
    assert len(box) == 4


def test_corrob_band_excludes_detections_below_floor():
    job, person_from_ultralytics = _make_job()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Second detection (conf 0.05) is below even the corrob floor (0.1).
    dets = _bumper_detections(
        [BOX_STRONG, BOX_CORROB], [0, 0], [0.6, 0.05])
    job.bumper_model = MagicMock(return_value=[MagicMock()])

    with patch("supervision.Detections.from_ultralytics",
               side_effect=[person_from_ultralytics(None), dets]):
        result, band = job._sam2_robot_candidates(
            frame, conf=0.4, corrob_floor=0.1, with_corrob_band=True)

    assert len(result) == 1
    assert band == []

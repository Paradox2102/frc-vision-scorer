import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import supervision as sv

from web.app import InferenceJob, ROBOT_FILL_CONF, ROBOT_FILL_MAX_BOX_FRAC, ROBOT_FILL_IOS
from web.field.detect_schema import DetectionSchema, expand_bumper_to_robot, merge_robot_detections, select_gap_fills
from web.field.detect_bridge import box_exceeds_frac

@patch("web.app.YOLO")
def test_detect_whole_robots(mock_yolo):
    """Verify that _detect_whole_robots correctly filters whole-robot detections by class ID and size guard."""
    job = InferenceJob("test_job", "/mock/video.mp4")
    job.whole_robot_class_id = 1
    
    # Mock YOLO model output
    mock_results = MagicMock()
    job.model = MagicMock(return_value=[mock_results])
    
    # Set up mock detections with sv.Detections
    mock_detections = MagicMock(spec=sv.Detections)
    mock_detections.class_id = np.array([1, 1, 0])
    mock_detections.xyxy = np.array([
        [100, 100, 200, 200],  # Valid class 1, small
        [0, 0, 900, 900],      # Valid class 1, but too large (90% of a 1000x1000 frame)
        [150, 150, 250, 250]   # Class 0 (wrong class)
    ])
    mock_detections.confidence = np.array([0.8, 0.9, 0.7])
    mock_detections.__len__.return_value = 3
    
    with patch("supervision.Detections.from_ultralytics", return_value=mock_detections):
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        boxes = job._detect_whole_robots(frame)
        
        # Verify only the first box is kept
        assert len(boxes) == 1
        assert boxes[0] == (100.0, 100.0, 200.0, 200.0)

@patch("web.app.YOLO")
def test_gap_fill_hook_a_logic(mock_yolo):
    """Verify the integration logic used in Hook A (prompt seeding)."""
    job = InferenceJob("test_job", "/mock/video.mp4")
    job.whole_robot_class_id = 1
    
    # 1. Mock Bumper Model Detections
    bumper_robots = [
        ((100.0, 300.0, 200.0, 350.0), "blue"),
        ((400.0, 300.0, 500.0, 350.0), "red")
    ]
    
    # 2. Mock Whole-Robot gap-fill detections
    # One overlapping with blue bumper, one new robot on blue half, one new robot on red half
    whole_boxes = [
        (100.0, 150.0, 200.0, 350.0),  # overlaps with blue bumper
        (250.0, 150.0, 350.0, 350.0),  # new robot on blue side (assuming blue half is X < 8.0)
        (800.0, 150.0, 900.0, 350.0)   # new robot on red side
    ]
    
    # Mock _detect_whole_robots
    job._detect_whole_robots = MagicMock(return_value=whole_boxes)
    
    # Perform merge
    merged = merge_robot_detections(
        [ (expand_bumper_to_robot(b, (1000, 1600, 3), 3.0), c) for b, c in bumper_robots ],
        whole_boxes,
        ROBOT_FILL_IOS
    )
    
    # Verify the overlapping one is merged/suppressed, and the 2 new ones are added as 'unknown'
    assert len(merged) == 4
    assert merged[0][1] == "blue"
    assert merged[1][1] == "red"
    assert merged[2][1] == "unknown"
    assert merged[3][1] == "unknown"

    # Verify calibration mapping and select_gap_fills behavior
    # We mock a simple calibration where X_image -> X_field directly
    mock_cal = MagicMock()
    mock_cal.image_to_field.side_effect = lambda uv: (uv[0] / 100.0, uv[1] / 100.0)
    
    # We mock the field model
    mock_field_model = MagicMock()
    mock_field_model.length_m = 16.0
    mock_field_model.alliance_side.side_effect = lambda xy: "blue" if xy[0] < 8.0 else "red"
    job.field_model = mock_field_model
    
    top_crop_px = 0
    bumper_halves = []
    unknown_halves = []
    unknown_boxes = []
    
    for box, color in merged:
        x1, y1, x2, y2 = box
        full_box = (x1, y1 + top_crop_px, x2, y2 + top_crop_px)
        uv = ((full_box[0] + full_box[2]) / 2.0, full_box[3])
        xy = mock_cal.image_to_field(uv)
        half = mock_field_model.alliance_side(xy)
        
        if color != "unknown":
            bumper_halves.append(half)
        else:
            unknown_halves.append(half)
            unknown_boxes.append(box)
            
    # bumper_halves: [ 'blue', 'red' ]
    # unknown_halves: [ 'blue', 'red' ]
    # both blue and red halves have 1 bumper, so they have room for gap fills
    keep_indices = select_gap_fills(bumper_halves, unknown_halves, max_per_half=3)
    assert keep_indices == [0, 1]  # Keep both gap fills


def test_calibrated_top_exclusion_rejects_above_field_box():
    view = MagicMock()
    view.calibration.robot_track_top_crop = 0.3

    assert InferenceJob._below_robot_tracking_top(
        view, (100, 20, 200, 250), frame_height=1000
    ) is False
    assert InferenceJob._below_robot_tracking_top(
        view, (100, 250, 200, 350), frame_height=1000
    ) is True

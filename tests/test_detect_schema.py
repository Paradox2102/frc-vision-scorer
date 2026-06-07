import numpy as np

from web.field.detect_schema import DetectionSchema, expand_bumper_to_robot


def test_schema_bumper_model_two_class():
    schema = DetectionSchema.from_names({0: "blue_bumper", 1: "red_bumper"})
    assert schema.robot_class_color == {0: "blue", 1: "red"}
    assert schema.fuel_class is None
    assert schema.hub_class is None


def test_schema_legacy_model():
    schema = DetectionSchema.from_names(
        {0: "fuel_ball", 1: "robot", 2: "hub_opening"}
    )
    assert schema.robot_class_color == {1: None}
    assert schema.fuel_class == 0
    assert schema.hub_class == 2


class _FakeDets:
    def __init__(self, boxes, ids, confs):
        self.xyxy = np.array(boxes, float)
        self.class_id = np.array(ids, int)
        self.confidence = np.array(confs, float)

    def __len__(self):
        return len(self.class_id)


def test_split_bumper_only():
    schema = DetectionSchema.from_names({0: "blue_bumper", 1: "red_bumper"})
    detections = _FakeDets(
        [(0, 0, 10, 10), (5, 5, 15, 15)], [1, 0], [0.9, 0.8]
    )
    robots, balls, hubs = schema.split(detections)
    assert robots == [
        ((0.0, 0.0, 10.0, 10.0), "red", 0.9),
        ((5.0, 5.0, 15.0, 15.0), "blue", 0.8),
    ]
    assert balls == []
    assert hubs == []


def test_expand_grows_upward_only():
    assert expand_bumper_to_robot(
        (100, 400, 160, 420), (480, 640, 3), factor=3.0
    ) == (100.0, 360.0, 160.0, 420.0)


def test_expand_clamps_to_frame_top():
    assert expand_bumper_to_robot(
        (0, 30, 50, 60), (480, 640, 3), factor=5.0
    )[1] == 0.0

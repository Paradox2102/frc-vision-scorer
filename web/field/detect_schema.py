"""Name-driven normalization for detector outputs."""
from __future__ import annotations


class DetectionSchema:
    def __init__(self, names, robot_class_color, fuel_class=None, hub_class=None):
        self.names = dict(names)
        self.robot_class_color = dict(robot_class_color)
        self.fuel_class = fuel_class
        self.hub_class = hub_class

    @classmethod
    def from_names(cls, names):
        normalized = {int(class_id): str(name) for class_id, name in names.items()}
        robot_class_color = {}
        fuel_class = None
        hub_class = None
        for class_id, raw_name in normalized.items():
            name = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
            if name == "red_bumper":
                robot_class_color[class_id] = "red"
            elif name == "blue_bumper":
                robot_class_color[class_id] = "blue"
            elif name == "robot":
                robot_class_color[class_id] = None
            elif name in {"fuel", "fuel_ball"}:
                fuel_class = class_id
            elif name in {"hub", "hub_opening"}:
                hub_class = class_id
        return cls(normalized, robot_class_color, fuel_class, hub_class)

    def split(self, detections):
        robots = []
        balls = []
        hubs = []
        for i in range(len(detections)):
            class_id = int(detections.class_id[i])
            box = tuple(float(v) for v in detections.xyxy[i])
            confidence = float(detections.confidence[i])
            if class_id in self.robot_class_color:
                robots.append((box, self.robot_class_color[class_id], confidence))
            elif class_id == self.fuel_class:
                balls.append(box)
            elif class_id == self.hub_class:
                hubs.append(box)
        return robots, balls, hubs


def expand_bumper_to_robot(box, frame_shape, factor=3.0):
    """Grow a bumper box upward while preserving its footprint and bottom edge."""
    x1, y1, x2, y2 = (float(v) for v in box)
    height = max(0.0, y2 - y1)
    return x1, max(0.0, y2 - height * float(factor)), x2, y2

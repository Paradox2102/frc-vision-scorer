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


def box_ios(a, b):
    """Intersection-over-smaller of two xyxy boxes (0.0–1.0).

    IoS, not IoU: a small box fully inside a large one scores 1.0, which is what
    we want when asking "is this whole-robot box already covered by a bumper box?"
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def merge_robot_detections(bumper_robots, robot_boxes, ios_thresh):
    """Combine bumper-derived robot boxes with whole-robot gap-fill boxes.

    bumper_robots: list of tuples starting with (box_xyxy, color).
    robot_boxes:   list[box_xyxy], whole-robot (colorless) boxes.
    Returns list[(box_xyxy, color)]: every bumper entry, followed by each
    whole-robot box NOT already covered by any bumper box (max IoS over all
    bumper boxes <= ios_thresh), tagged color="unknown".
    """
    bumper_boxes = [item[0] for item in bumper_robots]
    merged = [(item[0], item[1]) for item in bumper_robots]
    for fb in robot_boxes:
        if not bumper_boxes or all(box_ios(fb, bb) <= ios_thresh for bb in bumper_boxes):
            merged.append((fb, "unknown"))
    return merged


def select_gap_fills(bumper_halves, unknown_halves, max_per_half=3):
    """Pick which unknown gap-fill boxes to seed, capping each field half.

    bumper_halves:  list of "blue"/"red" for the bumper boxes in this frame.
    unknown_halves: list of "blue"/"red"/None for the candidate fills (None =
                    off-field, never selected), in their original order.
    Returns the indices into unknown_halves to keep. A half accepts a fill only
    while its running count (bumpers + already-kept fills) is below max_per_half.
    """
    counts = {"blue": 0, "red": 0}
    for h in bumper_halves:
        if h in counts:
            counts[h] += 1
    keep = []
    for idx, h in enumerate(unknown_halves):
        if h in counts and counts[h] < max_per_half:
            counts[h] += 1
            keep.append(idx)
    return keep


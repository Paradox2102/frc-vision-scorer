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


def plausible_bumper_box(box, min_width_height_ratio=0.5):
    """Return whether a detection has a physically plausible bumper shape."""
    x1, y1, x2, y2 = (float(v) for v in box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return height > 0.0 and width / height >= float(min_width_height_ratio)


def bumper_admissible(box, whole_robot_boxes, min_width_height_ratio=0.5,
                      corrob_ios=0.5):
    """Whether a bumper detection may become a robot candidate.

    A plausibly-shaped bumper always qualifies. A sliver that fails the shape
    check (e.g. a ball-occluded bumper fragment) is admitted only when a
    whole-robot detection overlaps it — two-detector agreement is itself the
    plausibility evidence the shape check was standing in for.
    """
    if plausible_bumper_box(box, min_width_height_ratio):
        return True
    return any(box_ios(box, wb) >= corrob_ios for wb in whole_robot_boxes)


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


def box_coverage(box, occluder):
    """Fraction of ``box`` covered by ``occluder`` (0.0-1.0).

    Unlike IoS, the denominator is always the candidate box. This lets a person
    veto a person-shaped robot candidate without rejecting a larger real robot
    just because a person occupies a small part of it.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    ox1, oy1, ox2, oy2 = (float(v) for v in occluder)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0:
        return 0.0
    ix1, iy1 = max(x1, ox1), max(y1, oy1)
    ix2, iy2 = min(x2, ox2), min(y2, oy2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / area


def reject_person_candidates(candidates, person_boxes, coverage_thresh=0.6):
    """Drop candidate tuples whose first element is strongly covered by a person."""
    return [
        candidate
        for candidate in candidates
        if not any(
            box_coverage(candidate[0], person_box) >= coverage_thresh
            for person_box in person_boxes
        )
    ]


def admit_low_conf_movers(robots, conf, motion_probe):
    """Filter ``(box, color, confidence)`` detections at ``conf``, admitting
    the sub-threshold band only when ``motion_probe(box)`` proves pixel motion.

    Motion blur is WHY a fast robot's bumper reads low confidence, and motion
    is what proves the low read is a robot rather than static clutter — the
    failure mode self-identifies. A ``None`` probe (gating disabled, or no
    previous frame yet) reduces to the plain threshold; a raising probe
    rejects only that detection.
    """
    out = []
    for box, color, confidence in robots:
        if confidence >= conf:
            out.append((box, color, confidence))
            continue
        if motion_probe is None:
            continue
        try:
            moving = bool(motion_probe(box))
        except Exception:
            moving = False
        if moving:
            out.append((box, color, confidence))
    return out


def motion_blob_boxes(prev_gray, cur_gray, diff_thresh=25, min_side_px=48,
                      max_area_frac=0.25, max_aspect=4.0, dilate_px=6):
    """Detector-independent robot candidates from frame differencing.

    Fixed cameras make the absdiff of consecutive tracked frames a direct
    motion map; a fast robot YOLO whiffs on (motion blur, dark bumpers) is
    still a fat bright blob here — the blur widens it. Returns (x1,y1,x2,y2)
    component boxes filtered to robot-plausible sizes: ``min_side_px`` drops
    balls (even dilation-merged pairs), ``max_area_frac`` drops global light
    changes, ``max_aspect`` drops thin ball-trail streaks. Callers feed these
    into the colorless gap-fill path; field/person/zone filters and the
    tracker's own gates do the rest of the vetting.
    """
    import cv2

    diff = cv2.absdiff(prev_gray, cur_gray)
    _thr, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    if dilate_px:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    frame_area = float(cur_gray.shape[0] * cur_gray.shape[1])
    out = []
    for i in range(1, n):
        x, y, w, h = (int(stats[i][j]) for j in range(4))
        if min(w, h) < min_side_px:
            continue
        if w * h > max_area_frac * frame_area:
            continue
        if max(w, h) > max_aspect * min(w, h):
            continue
        out.append((x, y, x + w, y + h))
    return out


def merge_colorless_candidates(whole_boxes, motion_boxes, ios_thresh):
    """Union whole-robot detections with motion blobs, motion deduped last.

    A robot both YOLO and the diff see must contribute ONE colorless
    candidate; a blob is added only when it overlaps nothing already kept
    (max IoS <= ``ios_thresh``)."""
    out = list(whole_boxes)
    for mb in motion_boxes:
        if all(box_ios(mb, wb) <= ios_thresh for wb in out):
            out.append(tuple(mb))
    return out


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

import numpy as np

from web.field.detect_schema import (
    DetectionSchema,
    box_coverage,
    expand_bumper_to_robot,
    plausible_bumper_box,
    reject_person_candidates,
)


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


def test_plausible_bumper_box_rejects_vertical_field_line():
    assert not plausible_bumper_box((100, 100, 130, 280))


def test_plausible_bumper_box_keeps_square_and_wide_bumpers():
    assert plausible_bumper_box((100, 100, 160, 180))
    assert plausible_bumper_box((100, 100, 260, 160))


def test_box_ios_full_containment_is_one():
    from web.field.detect_schema import box_ios
    # smaller box fully inside larger -> IoS == 1.0
    assert box_ios((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0


def test_box_ios_disjoint_is_zero():
    from web.field.detect_schema import box_ios
    assert box_ios((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_box_ios_partial_overlap():
    from web.field.detect_schema import box_ios
    # overlap 5x10=50; smaller area = 10x10=100 -> 0.5
    assert box_ios((0, 0, 10, 10), (5, 0, 25, 10)) == 0.5


def test_box_coverage_uses_candidate_area_not_smaller_area():
    assert box_coverage((0, 0, 20, 20), (0, 0, 10, 20)) == 0.5


def test_reject_person_candidates_drops_person_shaped_box():
    candidates = [
        ((10, 20, 50, 100), "red"),
        ((100, 100, 180, 180), "blue"),
    ]
    people = [(5, 10, 55, 110)]
    assert reject_person_candidates(candidates, people) == [
        ((100, 100, 180, 180), "blue")
    ]


def test_reject_person_candidates_keeps_partially_occluded_robot():
    candidates = [((0, 0, 100, 100), "red")]
    people = [(0, 0, 30, 100)]
    assert reject_person_candidates(candidates, people) == candidates


def test_merge_no_bumpers_keeps_all_as_unknown():
    from web.field.detect_schema import merge_robot_detections
    fills = [(0, 0, 10, 10), (50, 50, 60, 60)]
    merged = merge_robot_detections([], fills, ios_thresh=0.5)
    assert merged == [
        ((0, 0, 10, 10), "unknown"),
        ((50, 50, 60, 60), "unknown"),
    ]


def test_merge_passes_bumper_entries_through_unchanged():
    from web.field.detect_schema import merge_robot_detections
    bumpers = [((0, 0, 10, 30), "red"), ((40, 0, 50, 30), "blue")]
    merged = merge_robot_detections(bumpers, [], ios_thresh=0.5)
    assert merged == bumpers


def test_merge_suppresses_fill_overlapping_a_bumper():
    from web.field.detect_schema import merge_robot_detections
    bumpers = [((0, 0, 20, 20), "red")]
    fills = [(0, 0, 10, 10),        # IoS 1.0 with the bumper -> dropped
             (100, 100, 120, 120)]  # disjoint -> kept as unknown
    merged = merge_robot_detections(bumpers, fills, ios_thresh=0.5)
    assert merged == [
        ((0, 0, 20, 20), "red"),
        ((100, 100, 120, 120), "unknown"),
    ]


def test_select_gap_fills_caps_per_half():
    from web.field.detect_schema import select_gap_fills
    # blue half already full (3 bumpers) -> no blue fill; red has room for one
    bumper_halves = ["blue", "blue", "blue", "red"]
    unknown_halves = ["blue", "red", "red"]
    # index 0 (blue) rejected; index 1 (red) accepted -> red now 2; index 2 (red) accepted -> red 3
    assert select_gap_fills(bumper_halves, unknown_halves, max_per_half=3) == [1, 2]


def test_select_gap_fills_skips_none_territory():
    from web.field.detect_schema import select_gap_fills
    # an off-field (None) unknown is never selected
    assert select_gap_fills([], [None, "blue"], max_per_half=3) == [1]


def test_select_gap_fills_stops_at_cap_within_unknowns():
    from web.field.detect_schema import select_gap_fills
    # no bumpers; 4 blue unknowns, cap 3 -> first three kept
    assert select_gap_fills([], ["blue"] * 4, max_per_half=3) == [0, 1, 2]


def test_bumper_model_path_prefers_hard_negative_checkpoint(tmp_path):
    from web.app import _resolve_bumper_model_path

    base = tmp_path / "base.pt"
    hardneg = tmp_path / "hardneg.pt"
    base.touch()
    hardneg.touch()

    assert _resolve_bumper_model_path(
        hardneg_path=hardneg, base_path=base
    ) == hardneg


def test_bumper_model_path_falls_back_and_honors_override(tmp_path):
    from web.app import _resolve_bumper_model_path

    base = tmp_path / "base.pt"
    base.touch()
    missing_hardneg = tmp_path / "missing.pt"
    override = tmp_path / "custom.pt"

    assert _resolve_bumper_model_path(
        hardneg_path=missing_hardneg, base_path=base
    ) == base
    assert _resolve_bumper_model_path(
        override=str(override),
        hardneg_path=missing_hardneg,
        base_path=base,
    ) == override


def test_bumper_admissible_accepts_plausible_shape():
    from web.field.detect_schema import bumper_admissible
    # Wide bumper (w/h >= 0.5): admissible regardless of whole-robot boxes.
    assert bumper_admissible((10, 10, 50, 20), []) is True


def test_bumper_admissible_rejects_sliver_without_corroboration():
    from web.field.detect_schema import bumper_admissible
    # Tall sliver (w/h < 0.5), no whole-robot box anywhere near it.
    assert bumper_admissible((10, 10, 14, 30), []) is False
    assert bumper_admissible((10, 10, 14, 30), [(100, 100, 160, 160)]) is False


def test_bumper_admissible_accepts_corroborated_sliver():
    """A ball-occluded bumper fragment fails the shape check but is admitted
    when a whole-robot detection overlaps it — two-detector agreement IS the
    plausibility evidence."""
    from web.field.detect_schema import bumper_admissible
    sliver = (10, 10, 14, 30)            # w/h = 0.2
    whole = (0, 0, 60, 60)               # whole-robot box covering it
    assert bumper_admissible(sliver, [whole]) is True


def test_admit_low_conf_movers_keeps_high_conf_without_probe():
    from web.field.detect_schema import admit_low_conf_movers
    robots = [((0, 0, 10, 10), "red", 0.9), ((20, 0, 30, 10), "blue", 0.25)]
    out = admit_low_conf_movers(robots, conf=0.4, motion_probe=None)
    assert out == [((0, 0, 10, 10), "red", 0.9)]


def test_admit_low_conf_movers_admits_moving_sub_threshold_detection():
    """A blurred fast robot reads low confidence BECAUSE it is moving; the
    motion probe is what proves the sub-threshold band is a robot (qm11 6800:
    red bumper at 0.26-0.34 vs the 0.4 seed conf, whole sweep untracked)."""
    from web.field.detect_schema import admit_low_conf_movers
    robots = [((0, 0, 10, 10), "red", 0.9),
              ((20, 0, 30, 10), "red", 0.26),   # moving -> admit
              ((40, 0, 50, 10), "blue", 0.30)]  # static -> reject
    moving = {(20, 0, 30, 10)}
    out = admit_low_conf_movers(robots, conf=0.4,
                                motion_probe=lambda box: tuple(box) in moving)
    assert ((20, 0, 30, 10), "red", 0.26) in out
    assert all(b != (40, 0, 50, 10) for b, _c, _cf in out)
    assert ((0, 0, 10, 10), "red", 0.9) in out


def test_admit_low_conf_movers_probe_exception_rejects_only_that_box():
    from web.field.detect_schema import admit_low_conf_movers
    def probe(box):
        raise RuntimeError("boom")
    robots = [((0, 0, 10, 10), "red", 0.9), ((20, 0, 30, 10), "red", 0.26)]
    out = admit_low_conf_movers(robots, conf=0.4, motion_probe=probe)
    assert out == [((0, 0, 10, 10), "red", 0.9)]


def test_motion_blob_boxes_finds_moving_robot_ignores_static():
    from web.field.detect_schema import motion_blob_boxes
    rng = np.random.default_rng(42)
    texture = rng.integers(60, 255, size=(100, 140), dtype=np.uint8)
    prev = np.zeros((400, 800), np.uint8)
    cur = np.zeros((400, 800), np.uint8)
    prev[100:200, 300:440] = texture   # textured robot-sized blob...
    cur[100:200, 330:470] = texture    # ...moves 30 px intact
    boxes = motion_blob_boxes(prev, cur)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    # Blob spans the union of the two positions (plus dilation slack).
    assert x1 <= 310 and x2 >= 460 and y1 <= 110 and y2 >= 190
    assert motion_blob_boxes(prev, prev) == []


def test_motion_blob_boxes_rejects_ball_sized_blobs():
    from web.field.detect_schema import motion_blob_boxes
    prev = np.zeros((400, 800), np.uint8)
    cur = np.zeros((400, 800), np.uint8)
    prev[100:128, 300:328] = 180   # ~28 px ball...
    cur[100:128, 340:368] = 180    # ...flies right
    assert motion_blob_boxes(prev, cur) == []


def test_motion_blob_boxes_rejects_full_frame_change():
    from web.field.detect_schema import motion_blob_boxes
    prev = np.zeros((400, 800), np.uint8)
    cur = np.full((400, 800), 120, np.uint8)  # global light change
    assert motion_blob_boxes(prev, cur) == []


def test_merge_colorless_candidates_dedups_blobs_against_whole_robots():
    from web.field.detect_schema import merge_colorless_candidates
    whole = [(100, 100, 220, 200)]
    blobs = [(110, 105, 215, 195),   # same robot seen by both -> dropped
             (500, 100, 620, 210)]   # blob-only mover -> kept
    out = merge_colorless_candidates(whole, blobs, ios_thresh=0.5)
    assert out == [(100, 100, 220, 200), (500, 100, 620, 210)]

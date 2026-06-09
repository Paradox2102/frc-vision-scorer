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

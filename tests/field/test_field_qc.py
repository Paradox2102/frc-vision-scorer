"""Pure helpers for VLM field quality checks (web/field/field_qc.py)."""
from web.field.field_qc import track_is_static, parse_object_class


class TestTrackIsStatic:
    def test_full_window_no_motion_is_static(self):
        hist = [(t * 0.5, 3.0, 4.0) for t in range(21)]  # 0..10s, parked
        assert track_is_static(hist, now=10.0, window_sec=6.0, max_move_m=0.5)

    def test_moving_track_is_not_static(self):
        hist = [(t * 0.5, 3.0 + t * 0.2, 4.0) for t in range(21)]  # drifts 0.4 m/s
        assert not track_is_static(hist, now=10.0, window_sec=6.0, max_move_m=0.5)

    def test_short_history_is_not_static(self):
        # Only 2s of samples cannot prove 6s of stillness.
        hist = [(8.0 + t * 0.5, 3.0, 4.0) for t in range(5)]
        assert not track_is_static(hist, now=10.0, window_sec=6.0, max_move_m=0.5)

    def test_empty_history_is_not_static(self):
        assert not track_is_static([], now=10.0, window_sec=6.0, max_move_m=0.5)

    def test_small_jitter_within_threshold_is_static(self):
        hist = [(t * 0.5, 3.0 + (0.1 if t % 2 else -0.1), 4.0) for t in range(21)]
        assert track_is_static(hist, now=10.0, window_sec=6.0, max_move_m=0.5)

    def test_old_motion_outside_window_ignored(self):
        # Drove until t=3, parked since; at now=10 the 6s window (t>=4) is still.
        hist = [(t * 0.5, 2.0 * min(t * 0.5, 3.0), 4.0) for t in range(21)]
        assert track_is_static(hist, now=10.0, window_sec=6.0, max_move_m=0.5)


class TestParseObjectClass:
    def test_clean_json(self):
        assert parse_object_class('{"object": "robot"}') == "robot"
        assert parse_object_class('{"object": "equipment"}') == "equipment"

    def test_fenced_json(self):
        assert parse_object_class('```json\n{"object": "person"}\n```') == "person"

    def test_surrounding_prose(self):
        assert parse_object_class('Sure! {"object": "equipment"} hope that helps') == "equipment"

    def test_garbage_is_unclear(self):
        assert parse_object_class("no json here") == "unclear"
        assert parse_object_class("") == "unclear"
        assert parse_object_class(None) == "unclear"

    def test_unexpected_value_is_unclear(self):
        assert parse_object_class('{"object": "banana"}') == "unclear"


from web.field.field_qc import parse_census, boxes_to_field_points, reconcile_census


class TestParseCensus:
    def test_clean(self):
        c = parse_census(
            '{"robots": [{"box": [1, 2, 3, 4], "color": "red"}],'
            ' "lookalikes": [{"box": [5, 6, 7, 8], "kind": "crate"}]}')
        assert c["robots"] == [{"box": [1.0, 2.0, 3.0, 4.0], "color": "red"}]
        assert c["lookalikes"] == [{"box": [5.0, 6.0, 7.0, 8.0], "kind": "crate"}]

    def test_missing_lookalikes_defaults_empty(self):
        c = parse_census('{"robots": []}')
        assert c == {"robots": [], "lookalikes": []}

    def test_fenced(self):
        c = parse_census('```json\n{"robots": [{"box": [0,0,10,10]}]}\n```')
        assert len(c["robots"]) == 1
        assert c["robots"][0]["color"] == "unknown"

    def test_malformed_boxes_dropped(self):
        c = parse_census('{"robots": [{"box": [1,2,3]}, {"box": "x"}, '
                         '{"box": [0,0,4,4], "color": "blue"}]}')
        assert c["robots"] == [{"box": [0.0, 0.0, 4.0, 4.0], "color": "blue"}]

    def test_garbage_is_none(self):
        assert parse_census("nope") is None
        assert parse_census(None) is None


class TestBoxesToFieldPoints:
    def test_bottom_center_descaled_then_projected(self):
        # scale 0.5: VLM saw a half-size image, so pixel coords double back.
        # identity projection makes the expectation transparent.
        pts = boxes_to_field_points(
            [[10, 10, 30, 50]], scale=0.5, image_to_field=lambda uv: uv)
        assert pts == [(40.0, 100.0)]  # bottom-center (20, 50) / 0.5

    def test_projection_failure_skips_point(self):
        def boom(uv):
            raise ValueError("outside homography")
        assert boxes_to_field_points([[0, 0, 2, 2]], 1.0, boom) == []


class TestReconcileCensus:
    def test_exact_match(self):
        r = reconcile_census([(3.0, 4.0)], {7: (3.2, 4.1)}, match_radius_m=1.5)
        assert r["matched"] == [[7, 0, round(((0.2)**2 + (0.1)**2) ** 0.5, 2)]]
        assert r["unmatched_census"] == []
        assert r["unmatched_tracks"] == []

    def test_unseeded_robot_reported(self):
        r = reconcile_census([(3.0, 4.0), (10.0, 2.0)], {7: (3.0, 4.0)}, 1.5)
        assert r["unmatched_census"] == [1]

    def test_ghost_track_reported(self):
        r = reconcile_census([(3.0, 4.0)], {7: (3.0, 4.0), 9: (12.0, 5.0)}, 1.5)
        assert r["unmatched_tracks"] == [9]

    def test_radius_excludes_far_pairs(self):
        r = reconcile_census([(0.0, 0.0)], {7: (5.0, 5.0)}, 1.5)
        assert r["matched"] == []
        assert r["unmatched_census"] == [0]
        assert r["unmatched_tracks"] == [7]

    def test_one_to_one_greedy_nearest(self):
        # Two census points near one track: only the nearer one matches.
        r = reconcile_census([(3.0, 4.0), (3.5, 4.0)], {7: (3.1, 4.0)}, 1.5)
        assert [m[0] for m in r["matched"]] == [7]
        assert r["matched"][0][1] == 0
        assert r["unmatched_census"] == [1]

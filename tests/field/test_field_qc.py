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

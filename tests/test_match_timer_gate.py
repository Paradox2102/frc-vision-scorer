import asyncio

import web.app as app_mod
from web.app import compute_active_windows, parse_match_clock_text


def _countdown(start_frame, start_seconds, fps=10):
    return [
        (start_frame + elapsed * fps, start_seconds - elapsed)
        for elapsed in range(start_seconds + 1)
    ]


def test_parse_match_clock_text():
    assert parse_match_clock_text("1:27") == 87
    assert parse_match_clock_text("0:23") == 23
    assert parse_match_clock_text("2:20") == 140
    assert parse_match_clock_text(" 44 ") == 44
    assert parse_match_clock_text("") is None
    assert parse_match_clock_text("garbage") is None
    assert parse_match_clock_text("9:99") is None
    assert parse_match_clock_text("3:01") is None


def test_compute_active_windows_finds_both_periods_and_excludes_break():
    fps = 10
    auto = _countdown(100, 20, fps)
    teleop = _countdown(500, 140, fps)
    ranges, periods, fell_back, reason = compute_active_windows(
        auto + [(350, None), (400, None)] + teleop,
        fps=fps,
        total_frames=2200,
    )

    assert not fell_back
    assert reason is None
    assert [p["label"] for p in periods] == ["auto", "teleop"]
    assert periods[0]["start_frame"] == 100
    assert periods[0]["end_frame"] == 300
    assert periods[0]["processing_start_frame"] == 95
    assert periods[0]["processing_end_frame"] == 330
    assert periods[1]["processing_start_frame"] == 495
    assert ranges == [(95, 330), (495, 1930)]


def test_compute_active_windows_tolerates_one_dropout_and_merges_padding():
    fps = 10
    auto = _countdown(100, 20, fps)
    auto[8] = (auto[8][0], None)
    teleop = _countdown(330, 140, fps)
    ranges, periods, fell_back, _ = compute_active_windows(
        auto + teleop, fps=fps, total_frames=1900
    )

    assert not fell_back
    assert len(periods) == 2
    assert ranges == [(95, 1760)]


def test_compute_active_windows_auto_only_restricts_to_auto_window():
    fps = 10
    auto = _countdown(100, 20, fps)
    teleop = _countdown(500, 140, fps)
    ranges, periods, fell_back, reason = compute_active_windows(
        auto + [(350, None), (400, None)] + teleop,
        fps=fps,
        total_frames=2200,
        auto_only=True,
    )

    assert not fell_back
    assert reason is None
    # Only the auto period survives; teleop is never even searched for.
    assert [p["label"] for p in periods] == ["auto"]
    assert ranges == [(95, 330)]


def test_compute_active_windows_auto_only_locks_clip_ending_in_auto():
    # A clip that cuts off during/just after auto has no teleop countdown.
    # Full-match mode would fall back to the whole range, but auto-only still
    # locks the auto window.
    fps = 10
    auto = _countdown(100, 20, fps)
    ranges, periods, fell_back, reason = compute_active_windows(
        auto, fps=fps, total_frames=400, auto_only=True
    )

    assert not fell_back
    assert reason is None
    assert [p["label"] for p in periods] == ["auto"]
    assert ranges == [(95, 330)]


def test_compute_active_windows_auto_only_falls_back_without_auto():
    ranges, periods, fell_back, reason = compute_active_windows(
        [(0, None), (15, None)],
        fps=30,
        total_frames=900,
        auto_only=True,
    )
    assert fell_back
    assert periods == []
    assert ranges == [(0, 899)]
    assert "autonomous" in reason


def test_compute_active_windows_falls_back_without_complete_match():
    ranges, periods, fell_back, reason = compute_active_windows(
        [(0, None), (15, None), (30, 20)],
        fps=30,
        total_frames=900,
    )
    assert fell_back
    assert periods == []
    assert ranges == [(0, 899)]
    assert "autonomous" in reason


def test_scale_ocr_regions_merges_custom_over_defaults(monkeypatch):
    monkeypatch.setattr(
        app_mod,
        "custom_ocr_regions",
        {"red_center": (1, 2, 3, 4)},
    )
    scaled = app_mod.scale_ocr_regions(1920, 723)
    assert scaled["red_center"] == (1, 2, 3, 4)
    assert scaled["blue_center"] == app_mod.OCR_REGIONS_1080P["blue_center"]
    assert scaled["match_clock"] == app_mod.OCR_REGIONS_1080P["match_clock"]


def test_ocr_region_api_accepts_match_clock(monkeypatch):
    monkeypatch.setattr(app_mod, "current_event", None)
    monkeypatch.setattr(app_mod, "custom_ocr_regions", None)

    class Request:
        async def json(self):
            return {
            "red_center": [1, 2, 3, 4],
            "blue_center": [5, 6, 7, 8],
            "match_clock": [9, 10, 11, 12],
            }

    response = asyncio.run(app_mod.set_ocr_regions(Request()))
    assert response["regions"]["match_clock"] == (9, 10, 11, 12)

    fetched = asyncio.run(app_mod.get_ocr_regions())["regions"]
    assert fetched["match_clock"] == [9, 10, 11, 12]

"""Manta auto-path overlay: coordinate reconciliation + cache-only loader.

The pure Manta pieces (grid->meter reconciliation, path parsing, match-key
parsing) live in web/field/manta.py as the single source of truth; the CLI
scripts re-import them. The loader is cache-only: it must never hit the network,
so these tests exercise it against a fixture cache directory.
"""
import json

import pytest

from web.field.manta import (
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    GridCalibration,
    MantaAutoPath,
    load_match_auto_paths,
    match_number_from_key,
    overlay_paths_for_job,
    parse_auto_field,
)


def test_match_number_from_key():
    assert match_number_from_key("2026cur_qm2") == ("qual", 2)
    assert match_number_from_key("2026cur_qm115") == ("qual", 115)
    assert match_number_from_key("2026cur_sf3m1") == ("sf", 3)
    assert match_number_from_key("garbage") is None
    assert match_number_from_key("") is None


def test_grid_calibration_flips_x_not_y():
    cal = GridCalibration()  # flip_x=True, flip_y=False
    # Manta grid origin (0,0) is RED's low-X corner -> our high-x, y=0.
    x, y = cal.to_meters(0.0, 0.0)
    assert x == pytest.approx(FIELD_LENGTH_M)
    assert y == pytest.approx(0.0)
    # Manta grid max-X (100) -> our x=0 (blue corner).
    x, y = cal.to_meters(100.0, 52.5)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(FIELD_WIDTH_M)


def test_parse_auto_field_converts_and_skips_garbage():
    pts = parse_auto_field(["50,26.25", "not-a-point", "0,0"], GridCalibration())
    assert len(pts) == 2  # the middle garbage entry is skipped
    # midfield grid (50, 26.25) -> field center-ish
    assert pts[0][0] == pytest.approx(FIELD_LENGTH_M / 2)
    assert pts[0][1] == pytest.approx(FIELD_WIDTH_M / 2)


def _write_cache(cache_dir, event="2026cur", mtype="qual", number=2):
    """Write a qm2-shaped fixture: 6 robots, red3 has a single (unusable) point."""
    robots = [
        {"alliance": "blue", "position": 1, "team_number": 1684,
         "scouts": [{"scout_team": 3128, "team_number": 1684,
                     "data": {"auto_field": ["74,7", "70,10", "66,12"]}}]},
        {"alliance": "blue", "position": 2, "team_number": 6615,
         "scouts": [{"scout_team": 3128, "team_number": 6615,
                     "data": {"auto_field": ["72,20", "71,22"]}}]},
        {"alliance": "blue", "position": 3, "team_number": 2053,
         "scouts": [{"scout_team": 3128, "team_number": 2053,
                     "data": {"auto_field": ["75,40", "76,42"]}}]},
        {"alliance": "red", "position": 1, "team_number": 7421,
         "scouts": [{"scout_team": 3128, "team_number": 7421,
                     "data": {"auto_field": ["25,7", "28,10", "30,12"]}}]},
        {"alliance": "red", "position": 2, "team_number": 8575,
         "scouts": [{"scout_team": 3128, "team_number": 8575,
                     "data": {"auto_field": ["28,20", "29,22"]}}]},
        {"alliance": "red", "position": 3, "team_number": 5166,
         "scouts": [{"scout_team": 3128, "team_number": 5166,
                     "data": {"auto_field": ["20,26"]}}]},  # single point -> filtered
    ]
    payload = {"event": event, "match": {"type": mtype, "number": number}, "robots": robots}
    (cache_dir / f"{event}_{mtype}_{number}.json").write_text(json.dumps(payload))


def test_load_missing_cache_returns_empty(tmp_path):
    assert load_match_auto_paths("2026cur_qm2", cache_dir=tmp_path) == []


def test_load_unparseable_key_returns_empty(tmp_path):
    _write_cache(tmp_path)
    assert load_match_auto_paths("garbage", cache_dir=tmp_path) == []
    assert load_match_auto_paths("", cache_dir=tmp_path) == []


def test_load_parses_fixture_and_filters_short_paths(tmp_path):
    _write_cache(tmp_path)
    paths = load_match_auto_paths("2026cur_qm2", cache_dir=tmp_path)
    # 6 robots, but red3's single-point path is dropped -> 5 usable.
    assert len(paths) == 5
    assert all(isinstance(p, MantaAutoPath) for p in paths)
    teams = {p.team_number for p in paths}
    assert 5166 not in teams  # the single-point robot
    assert {1684, 6615, 2053, 7421, 8575} == teams


def test_load_fixture_alliance_sides(tmp_path):
    _write_cache(tmp_path)
    paths = load_match_auto_paths("2026cur_qm2", cache_dir=tmp_path)
    half = FIELD_LENGTH_M / 2
    by_team = {p.team_number: p for p in paths}
    # flip_x maps Manta blue (high grid-X) to our low-x half, red to high-x half.
    blue_mean_x = sum(pt[0] for pt in by_team[1684].path_m) / len(by_team[1684].path_m)
    red_mean_x = sum(pt[0] for pt in by_team[7421].path_m) / len(by_team[7421].path_m)
    assert blue_mean_x < half
    assert red_mean_x > half
    # points stay inside the field
    for p in paths:
        for x, y in p.path_m:
            assert 0.0 <= x <= FIELD_LENGTH_M
            assert 0.0 <= y <= FIELD_WIDTH_M


def test_load_picks_longest_scout_path(tmp_path):
    robots = [{"alliance": "blue", "position": 1, "team_number": 1111, "scouts": [
        {"scout_team": 3128, "team_number": 1111, "data": {"auto_field": ["50,26", "51,26"]}},
        {"scout_team": 9999, "team_number": 1111, "data": {"auto_field": ["50,26", "51,26", "52,26", "53,26"]}},
    ]}]
    (tmp_path / "2026cur_qual_2.json").write_text(
        json.dumps({"event": "2026cur", "robots": robots}))
    paths = load_match_auto_paths("2026cur_qm2", cache_dir=tmp_path)
    assert len(paths) == 1
    assert len(paths[0].path_m) == 4  # the longer scout path wins


def test_overlay_paths_for_job_gated_on_auto_only(tmp_path):
    _write_cache(tmp_path)
    # auto_period_only off -> no overlay even though cache exists
    assert overlay_paths_for_job(False, "2026cur_qm2", cache_dir=tmp_path) == []
    # no match key -> no overlay
    assert overlay_paths_for_job(True, None, cache_dir=tmp_path) == []
    # both conditions met -> paths returned
    assert len(overlay_paths_for_job(True, "2026cur_qm2", cache_dir=tmp_path)) == 5

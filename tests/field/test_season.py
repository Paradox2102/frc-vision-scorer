"""Season registry: an event's field follows its year (web/field/season.py)."""
import pytest

from web.field import season as sn
from web.field import store as field_store
from web.field.field_model import FieldModel


@pytest.fixture(autouse=True)
def _tmp_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(field_store, "UPLOAD_DIR", tmp_path)


def test_event_year_parses_first_four_digits():
    assert sn.event_year("2025hop") == 2025
    assert sn.event_year("2026casnd") == 2026
    assert sn.event_year("test") is None
    assert sn.event_year("") is None
    assert sn.event_year(None) is None
    assert sn.event_year("20x5abc") is None


def test_2025_event_gets_reefscape_welded():
    s = sn.season_for_event("2025hop")
    assert s.year == 2025 and s.tag_layout == "2025-reefscape-welded"
    assert s.variant == "welded" and s.source == "year"
    assert s.structures == sn.STRUCTURE_2025_REEF
    assert s.wireframe == "2025-reefscape-welded"
    assert (s.auto_s, s.teleop_s, s.endgame_s) == (15.0, 135.0, 20.0)


def test_2026_event_and_no_event_keep_the_2026_default():
    for key in ("2026cur", None, "", "test", "evt_tags"):
        s = sn.season_for_event(key)
        assert s.tag_layout == "2026-rebuilt-welded", key
        assert (s.auto_s, s.teleop_s) == (20.0, 140.0)
        assert s.structures == sn.STRUCTURE_2026_HUB
    assert sn.season_for_event(None).source == "default"
    assert sn.season_for_event("2026cur").source == "year"


def test_2024_and_2023_have_layouts_but_no_structures_or_wireframe():
    s24 = sn.season_for_event("2024txho")
    assert s24.tag_layout == "2024-crescendo" and s24.structures is None
    assert s24.wireframe is None
    s23 = sn.season_for_event("2023mil")
    assert s23.tag_layout == "2023-chargedup" and s23.structures is None


@pytest.mark.parametrize("key", ["2022cmptx", "2019txho", "2030abc"])
def test_years_without_tags_or_unknown_have_no_layout(key):
    s = sn.season_for_event(key)
    assert s.tag_layout is None and not s.has_tags
    assert s.structures is None and s.known is False
    assert s.year == int(key[:4])
    assert sn.no_layout_reason(s) == f"no AprilTag layout for {s.year} — calibrate manually"


def test_every_registered_layout_and_wireframe_exists():
    for s in sn.SEASONS.values():
        fm = FieldModel.load(s.tag_layout)
        assert fm.season == s.tag_layout and fm.tags
        if s.tag_layout_andymark:
            assert FieldModel.load(s.tag_layout_andymark).tags
        if s.wireframe:
            assert sn.load_wireframe(s.wireframe)


def test_override_variant_and_year(tmp_path):
    sn.save_override("2025hop", variant="andymark")
    s = sn.season_for_event("2025hop")
    assert s.tag_layout == "2025-reefscape-andymark" and s.source == "override"
    assert s.variant == "andymark"
    assert (tmp_path / "field_season_2025hop.json").is_file()

    sn.save_override("2026demo", year=2025)
    assert sn.season_for_event("2026demo").tag_layout == "2025-reefscape-welded"

    assert sn.clear_override("2025hop") is True
    assert sn.season_for_event("2025hop").tag_layout == "2025-reefscape-welded"
    assert sn.clear_override("2025hop") is False


def test_override_rejects_bad_input():
    with pytest.raises(sn.SeasonError):
        sn.save_override("2025hop", year=2019)
    with pytest.raises(sn.SeasonError):
        sn.save_override("2025hop", variant="plywood")
    with pytest.raises(sn.SeasonError):
        sn.save_override("2024txho", year=2024, variant="andymark")


def test_corrupt_override_is_ignored(tmp_path):
    (tmp_path / "field_season_2025hop.json").write_text("{not json")
    assert sn.season_for_event("2025hop").tag_layout == "2025-reefscape-welded"


def test_season_for_layout_and_structure_kind():
    assert sn.season_for_layout("2026-rebuilt-welded").year == 2026
    am = sn.season_for_layout("2025-reefscape-andymark")
    assert am.year == 2025 and am.variant == "andymark"
    assert sn.season_for_layout("") is None
    assert sn.structure_kind(FieldModel.load("2026-rebuilt-welded")) == sn.STRUCTURE_2026_HUB
    assert sn.structure_kind(FieldModel.load("2025-reefscape-welded")) == sn.STRUCTURE_2025_REEF
    assert sn.structure_kind(FieldModel.load("2024-crescendo")) is None
    # a hand-built model keeps the historical (2026) behaviour
    hand = FieldModel(tags={}, length_m=16.5, width_m=8.0)
    assert hand.season == "" and sn.structure_kind(hand) == sn.STRUCTURE_2026_HUB


def test_wireframe_load_is_cached_and_grouped():
    wf = sn.wireframe_for(FieldModel.load("2025-reefscape-welded"))
    assert {"reef", "barge", "perimeter"} <= set(wf)
    assert wf["reef"].shape[1] == 6
    assert sn.wireframe_for(FieldModel.load("2025-reefscape-welded")) is wf
    assert sn.wireframe_for(FieldModel.load("2024-crescendo")) is None

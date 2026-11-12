"""The calibrate page's Lens selector default (lens_borrow.lens_options): joint with
a sibling corner > borrow a sibling's lens > own lens; main never shares."""
from tests.field.hop2025_fixtures import HOP_LEFT_SOURCE
from tests.field.test_lens_borrow import _lens
from web.field import lens_borrow as lb

CAMS = ["main", "corner_left", "corner_right", "corner_far"]


def _ver(vid, rms=1.0, source=None, start=None, mode="pinhole", created="2026-11-11T00"):
    cal = _lens(mode=mode)
    cal.rms_error = rms
    return {"calibration": cal, "start_match": start, "version_id": vid,
            "created_at": created, "label": None, "source": source}


def _opts(camera, versions, match="evt_qm3"):
    return lb.lens_options(camera, CAMS, lambda c: versions.get(c, []), match)


def test_defaults_to_joint_when_the_sibling_has_stored_tags():
    o = _opts("corner_right", {"corner_left": [_ver("L1", source=HOP_LEFT_SOURCE)],
                               "main": [_ver("M1", source=HOP_LEFT_SOURCE)]})
    assert o["default"] == "joint:corner_left"
    assert [x["value"] for x in o["options"]] == ["", "joint:corner_left", "corner_left"]
    assert "share a lens model" in o["hint"] and "corner_left" in o["hint"]
    assert o["options"][1]["version_id"] == "L1"


def test_defaults_to_borrow_when_the_sibling_has_no_stored_tags():
    o = _opts("corner_right", {"corner_left": [_ver("L1", source=None)],
                               "corner_far": [_ver("F1", source={"auto_tags": {"3": []},
                                                                 "manual_tags": []})]})
    assert o["default"] == "corner_left"      # one tag is not enough for a joint fit
    assert [x["kind"] for x in o["options"]] == ["own", "borrow", "borrow"]
    assert "share a lens model" in o["hint"]


def test_prefers_the_lowest_rms_sibling():
    o = _opts("corner_right", {"corner_left": [_ver("L1", rms=4.0, source=HOP_LEFT_SOURCE)],
                               "corner_far": [_ver("F1", rms=1.5, source=HOP_LEFT_SOURCE)]})
    assert o["default"] == "joint:corner_far"
    o = _opts("corner_right", {"corner_left": [_ver("L1", rms=4.0)],
                               "corner_far": [_ver("F1", rms=1.5)]})
    assert o["default"] == "corner_far"


def test_own_lens_when_no_sibling_covers_the_match():
    o = _opts("corner_right", {"main": [_ver("M1")],
                               "corner_left": [_ver("L1", mode="camera")]})
    assert o["default"] == "" and [x["kind"] for x in o["options"]] == ["own"]
    assert o["hint"]


def test_main_has_no_shared_options():
    o = _opts("main", {"corner_left": [_ver("L1", source=HOP_LEFT_SOURCE)]})
    assert o == {"options": [{"value": "", "kind": "own", "label": "solve own lens"}],
                 "default": "", "hint": None}


def test_uses_the_version_covering_the_match():
    vs = {"corner_left": [_ver("early", source=None, start=None, created="2026-11-10"),
                          _ver("late", source=HOP_LEFT_SOURCE, start="evt_qm5")]}
    assert _opts("corner_right", vs, "evt_qm2")["default"] == "corner_left"
    assert _opts("corner_right", vs, "evt_qm7")["default"] == "joint:corner_left"


def test_holdouts_do_not_count_as_joint_tags():
    src = dict(HOP_LEFT_SOURCE, holdout_ids=[12, 13, 20, 21])
    assert lb.source_tag_ids(src) == [22]
    assert _opts("corner_right", {"corner_left": [_ver("L1", source=src)]})["default"] \
        == "corner_left"


def test_suggestion_names_the_best_available_kind():
    o = _opts("corner_right", {"corner_left": [_ver("L1", source=HOP_LEFT_SOURCE)]})
    assert lb.lens_suggestion(o, "own") == "try the joint lens with corner_left"
    assert lb.lens_suggestion(o, "borrow") == "try the joint lens with corner_left"
    assert lb.lens_suggestion(o, "joint") is None
    o = _opts("corner_right", {"corner_left": [_ver("L1")]})
    assert lb.lens_suggestion(o, "own") == "try borrowing from corner_left"
    assert lb.lens_suggestion(_opts("corner_right", {}), "own") is None

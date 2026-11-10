"""Final labels and tag strings (web/hud_labels.py; spec section 2): every case
of the tag table and the correction rule."""
from web import hud_labels as hl
from web.hud_draw import slot_label

ROWS = {"B1": {"slot": "B1", "team_number": "3128", "confirmed": True},
        "B2": {"slot": "B2", "team_number": "6773", "confirmed": False},
        "B3": {"slot": "B3", "team_number": None, "confirmed": False},
        "R1": {"slot": "R1", "team_number": "AMBIGUOUS", "confirmed": True}}


def _robot(slot, score=2, ghost=False):
    return {"slot": slot, "slot_id": None, "alliance": "blue", "box": [0, 0, 1, 1],
            "ghost": ghost, "score": score, "mask_poly": None}


# --- the tag table ---

def test_confirmed_named_unlabelled():
    assert hl.entry_text(_robot("B1"), ROWS, {}) == ("robot", "3128 S2")
    assert hl.entry_text(_robot("B2"), ROWS, {}) == ("robot", "6773? S2")
    assert hl.entry_text(_robot("B3"), ROWS, {}) == ("robot", "no # S2")


def test_ambiguous_counts_as_unlabelled_and_never_confirmed():
    assert hl.entry_text(_robot("R1"), ROWS, {}) == ("robot", "no # S2")
    assert hl.norm_team("AMBIGUOUS") is None and hl.norm_team("") is None
    assert hl.norm_team(3128) == "3128"


def test_ghost_is_the_slot_prefixed_tag():
    assert hl.entry_text(_robot("B1", ghost=True), ROWS, {}) == ("ghost", "B1 3128 S2")
    assert hl.entry_text(_robot("B3", ghost=True), ROWS, {}) == ("ghost", "B3 no # S2")


def test_corner_entries():
    with_slot = {"slot": "B2", "alliance": "blue", "box": [0, 0, 1, 1], "score": 1,
                 "live_text": "6773? (40%) S1"}
    no_slot = {"slot": None, "alliance": "red", "box": [0, 0, 1, 1], "score": None,
               "live_text": "red bumper no #"}
    assert hl.entry_text(with_slot, ROWS, {}) == ("robot", "6773? S1")
    assert hl.entry_text(no_slot, ROWS, {}) == ("live", "red bumper no #")


def test_radar_label_is_team_or_slot_without_question_mark():
    assert hl.radar_text({"slot": "B2"}, ROWS, {}) == (True, "6773")
    assert hl.radar_text({"slot": "B3"}, ROWS, {}) == (True, "B3")
    assert hl.radar_text({"slot": "R1"}, ROWS, {}) == (True, "R1")
    assert hl.radar_text({"slot": "R3"}, ROWS, {}) == (False, "R3")


def test_slot_without_a_meta_row_is_a_phantom():
    assert hl.entry_text(_robot("R2"), ROWS, {}) == ("phantom", "R2")


def test_score_suffix_is_omitted_when_unknown():
    assert hl.final_tag("3128", True, None) == "3128"
    assert hl.final_tag("3128", False, None) == "3128?"
    assert hl.final_tag(None, False, None) == "no #"
    assert hl.entry_text(_robot("B2", score=None), ROWS, {}) == ("robot", "6773?")


# --- the correction rule ---

def test_old_team_frames_replaced_other_teams_kept():
    corr = {"B2": ("6773", "9066")}
    assert hl.entry_text(_robot("B2"), ROWS, corr) == ("robot", "9066 S2")   # confirmed now
    other = {"B2": {"slot": "B2", "team_number": "2637", "confirmed": False}}
    assert hl.entry_text(_robot("B2"), other, corr) == ("robot", "2637? S2")
    unl = {"B2": {"slot": "B2", "team_number": None, "confirmed": False}}
    assert hl.entry_text(_robot("B2"), unl, corr) == ("robot", "no # S2")


def test_old_none_fills_unlabelled_frames_only():
    corr = {"B3": (None, "4256")}
    assert hl.entry_text(_robot("B3"), ROWS, corr) == ("robot", "4256 S2")
    amb = {"B3": {"slot": "B3", "team_number": "AMBIGUOUS", "confirmed": False}}
    assert hl.entry_text(_robot("B3"), amb, corr) == ("robot", "4256 S2")
    named = {"B3": {"slot": "B3", "team_number": "6800", "confirmed": True}}
    assert hl.entry_text(_robot("B3"), named, corr) == ("robot", "6800 S2")


def test_first_old_and_latest_new_across_two_corrections():
    rows = [{"slot": 1, "old_team": "6773", "new_team": "1111"},
            {"slot": 1, "old_team": "1111", "new_team": "9066"},
            {"slot": 4, "old_team": "AMBIGUOUS", "new_team": "2637"}]
    assert hl.effective_corrections(rows) == {1: ("6773", "9066"), 4: (None, "2637")}


def test_new_equal_to_old_is_no_correction():
    rows = [{"slot": 1, "old_team": "6773", "new_team": "1111"},
            {"slot": 1, "old_team": "1111", "new_team": "6773"},
            {"slot": 2, "old_team": None, "new_team": ""}]
    assert hl.effective_corrections(rows) == {}


def test_replaced_frames_render_confirmed():
    assert hl.final_label("6773", False, ("6773", "9066")) == ("9066", True, True)
    assert hl.final_label(None, False, (None, "4256")) == ("4256", True, True)
    assert hl.final_label("2637", False, ("6773", "9066")) == ("2637", False, False)


def test_relabel_counts_per_corrected_slot():
    frames = [{"robots": [{"slot": "B2", "team_number": "6773"},
                          {"slot": "B3", "team_number": None}]},
              {"robots": [{"slot": "B2", "team_number": "2637"},
                          {"slot": "B3", "team_number": "AMBIGUOUS"}]},
              {"robots": []}]
    corr = {1: ("6773", "9066"), 2: (None, "4256"), 5: ("1", "2")}
    assert hl.relabel_counts(frames, corr, slot_label) == {"B2": 1, "B3": 2, "R3": 0}


def test_label_before_correction_prefers_slot_labels():
    summary = {"slot_labels": [{"slot": 0, "slot_label": "B1", "team_number": "AMBIGUOUS"},
                               {"slot": 1, "slot_label": "B2", "team_number": "6773"}]}
    assert hl.label_before_correction(summary, 1, "9999") == "6773"
    assert hl.label_before_correction(summary, 0, "9999") is None
    assert hl.label_before_correction(summary, 3, "2637") == "2637"
    assert hl.label_before_correction({}, 3, "AMBIGUOUS") is None

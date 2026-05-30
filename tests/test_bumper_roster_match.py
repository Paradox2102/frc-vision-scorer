from web.app import _match_bumper_read_to_roster


BLUE = ["3175", "1710", "3512"]
RED = ["8749", "2486", "4201"]
ALL = BLUE + RED


def test_exact_team_read_defines_match_alliance_even_if_color_hint_is_wrong():
    team, alliance, conf = _match_bumper_read_to_roster(
        "8749", model_color="blue", expected_alliance="blue",
        match_teams=ALL, blue_teams=BLUE, red_teams=RED,
    )

    assert team == "8749"
    assert alliance == "red"
    assert conf >= 0.9


def test_partial_read_is_constrained_by_observed_bumper_alliance():
    team, alliance, conf = _match_bumper_read_to_roster(
        "42", model_color="unknown", expected_alliance="red",
        match_teams=ALL, blue_teams=BLUE, red_teams=RED,
    )

    assert team == "4201"
    assert alliance == "red"
    assert conf > 0


def test_partial_read_does_not_cross_observed_bumper_alliance():
    team, alliance, conf = _match_bumper_read_to_roster(
        "35", model_color="unknown", expected_alliance="red",
        match_teams=ALL, blue_teams=BLUE, red_teams=RED,
    )

    assert team is None
    assert alliance == "red"
    assert conf == 0.0

"""Regression guard for the alliance flip-flop bug.

A robot with a tentative team number must report the roster alliance for that
team. No code path may override that with embedding clustering or anything else.
"""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from web.app import InferenceJob, RobotTrack
from web.diagnostic import NullRecorder


def _make_job():
    """A lightweight stand-in exposing only the attributes
    `assign_teams_from_votes` actually reads. Lets us call the method as an
    unbound function (`InferenceJob.assign_teams_from_votes(job)`) without
    paying the cost of `InferenceJob.__init__` (which loads YOLO weights)."""
    return SimpleNamespace(
        match_blue_teams=["2102", "9452", "2485"],
        match_red_teams=["1538", "294", "4322"],
        match_teams=["2102", "9452", "2485", "1538", "294", "4322"],
        robot_tracks={},
        scoring_events=[],
        _current_frame=0,
        diagnostic=NullRecorder(),
    )


def test_tentative_team_alliance_matches_roster():
    """A robot with strong votes for a red team must end up red, not blue."""
    job = _make_job()
    job.robot_tracks = {
        0: RobotTrack(robot_id=0, alliance="blue", team_number=None,
                      last_position=(0, 0), positions_history=deque(maxlen=30)),
    }
    # Strong VLM votes for a RED team
    job.robot_tracks[0].team_ocr_votes = {"4322": 5.0}

    InferenceJob.assign_teams_from_votes(job)

    r = job.robot_tracks[0]
    assert r.team_number == "4322"
    assert r.alliance == "red", (
        f"Robot has team 4322 (red) but alliance={r.alliance!r}; "
        "regression: alliance must follow the roster when a team is assigned."
    )


def test_no_embedding_alliance_calls_in_loop():
    """Structural guard: the loop must not call _assign_alliances_by_embedding
    or assign_identities_now (Component D removal). They caused the flip-flop."""
    src = Path("web/app.py").read_text()
    # Function bodies may exist, but no `self.<func>()` calls should remain in the loop.
    assert "self._assign_alliances_by_embedding()" not in src, (
        "_assign_alliances_by_embedding() must not be called from the loop."
    )
    assert "self.assign_identities_now()" not in src, (
        "assign_identities_now() must not be called from the loop."
    )

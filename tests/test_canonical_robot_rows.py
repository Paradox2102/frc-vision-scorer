from collections import deque

from web.app import InferenceJob, RobotTrack
from web.field.identity import IdentityRegistry, RobotIdentity


def _track(identity_id, alliance, scores):
    return RobotTrack(
        robot_id=identity_id,
        alliance=alliance,
        team_number=None,
        last_position=(0.0, 0.0),
        positions_history=deque(maxlen=30),
        scores=scores,
    )


def test_canonical_rows_ignore_retired_tracks_and_keep_fixed_slots():
    job = InferenceJob.__new__(InferenceJob)
    job.identity_registry = IdentityRegistry()
    job.identity_registry.identities = {
        10: RobotIdentity(10, "blue", active_slot=1, last_seen_frame=20),
        11: RobotIdentity(11, "blue", active_slot=None, last_seen_frame=10),
        20: RobotIdentity(20, "red", active_slot=4, last_seen_frame=20),
    }
    job.robot_tracks = {
        10: _track(10, "blue", 7),
        11: _track(11, "blue", 3),
        20: _track(20, "red", 5),
        99: _track(99, "blue", 183),  # retired historical identity
    }
    job._get_display_team_number = lambda robot: robot.team_number

    rows = job._canonical_robot_rows()

    assert [row["slot"] for row in rows] == ["B1", "B2", "B3", "R1", "R2", "R3"]
    assert len(rows) == 6
    assert all(row["id"] != 99 for row in rows)
    assert next(row for row in rows if row["slot"] == "B2")["scores"] == 7
    assert sum(row["scores"] for row in rows) == 15

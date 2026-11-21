"""export.robot_path / match_robots: whole-match per-robot paths."""
from web.scouting import export
from web.scouting.model import MatchData, Sample, TeamMatch


def _s(fn, t, period, x=1.0, y=2.0, confirmed=False, slot=None):
    return Sample(frame_num=fn, t_match=t, period=period, remaining=0.0, x=x, y=y,
                  ax=x, ay=y, dt=0.2, confirmed=confirmed, slot=slot)


def _tm(samples, auto_path=None, auto_source=None):
    tm = TeamMatch(team=254, alliance="red", station=1)
    tm.samples = samples
    tm.auto_path = auto_path or []
    tm.auto_source = auto_source
    return tm


def test_auto_path_replaces_auto_samples_and_teleop_follows():
    samples = [_s(10, 1.0, "auto", x=9.0), _s(200, 30.0, "teleop", confirmed=True),
               _s(201, 30.2, "teleop")]
    auto = [_s(10, 1.0, "auto", x=1.0, slot="R1"), _s(11, 1.1, "auto", x=1.1, slot="R1")]
    p = export.robot_path(_tm(samples, auto, "autonomous_paths"), 0.2)
    assert p["t"] == [1.0, 1.1, 30.0, 30.2]
    assert p["x"][:2] == [1.0, 1.1]                   # the auto path, not the auto sample
    assert p["frame"] == [10, 11, 200, 201]
    assert p["confirmed_share"] == round(1 / 3, 3)
    assert p["seen_s"]["teleop"] == 0.4


def test_frames_auto_source_keeps_labelled_auto_samples():
    samples = [_s(10, 1.0, "auto"), _s(11, 1.2, "auto")]
    p = export.robot_path(_tm(samples, list(samples), "frames"), 0.2)
    assert p["frame"] == [10, 11] and p["seen_s"]["auto"] == 0.4


def test_overlapping_auto_pieces_keep_one_point_per_frame():
    auto = [_s(10, 1.0, "auto", slot="R1"), _s(11, 1.1, "auto", slot="R1"),
            _s(11, 1.1, "auto", slot="R2"), _s(12, 1.2, "auto", slot="R2")]
    p = export.robot_path(_tm([], auto, "autonomous_paths"), 0.2)
    assert p["frame"] == [10, 11, 12]


def test_segments_split_on_gaps():
    samples = [_s(1, 21.0, "teleop"), _s(2, 21.2, "teleop"), _s(30, 25.0, "teleop"),
               _s(31, 25.2, "teleop"), _s(32, 25.4, "teleop")]
    p = export.robot_path(_tm(samples), 0.2)
    assert p["segments"] == [[0, 1], [2, 4]]
    assert (p["first_t"], p["last_t"]) == (21.0, 25.4)


def test_match_robots_lineup_order_then_unlisted_teams():
    md = MatchData(event_key="2026tst", match_key="2026tst_qm1", comp_level="qm",
                   match_number=1, set_number=1, year=2026, replay_id="r",
                   replay_created_at=None, legacy_labels=False,
                   alliances={"red": [1, None, 3], "blue": [4, 5, 6]},
                   hub_auto={}, frame_period_s=0.2)
    extra = _tm([_s(1, 21.0, "teleop")])
    extra.team, extra.alliance, extra.station = 99, "blue", None
    md.teams[99] = extra
    rows = export.match_robots(md)
    assert [r["team_number"] for r in rows] == [1, 3, 4, 5, 6, 99]
    assert rows[1]["alliance_position"] == 3
    assert rows[-1]["path"]["n_points"] == 1


def test_segments_split_on_impossible_jumps():
    samples = [_s(1, 21.0, "teleop", x=1.0), _s(2, 21.2, "teleop", x=1.5),
               _s(3, 21.4, "teleop", x=9.0), _s(4, 21.6, "teleop", x=9.5)]
    p = export.robot_path(_tm(samples), 0.2)
    assert p["segments"] == [[0, 1], [2, 3]] and p["n_jumps"] == 1

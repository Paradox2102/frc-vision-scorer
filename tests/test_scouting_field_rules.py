"""Tests for the 2026 auto-path field rules (start line + obstacles)."""
from __future__ import annotations

import numpy as np
import pytest

from web.scouting.field_rules import (auto_rules, constrain_auto_path, segment_enters)
from web.scouting.metrics.auto_paths import AutoPaths
from web.scouting.model import MatchData, Sample, TeamMatch

L, W = 16.541, 8.069
MARGIN = 0.25


def rules():
    return auto_rules(2026)


def inflated(m=MARGIN):
    return [(x0 - m, y0 - m, x1 + m, y1 + m) for x0, y0, x1, y1 in rules().obstacles]


def inside(x, y, r, tol=1e-6):
    return r[0] + tol < x < r[2] - tol and r[1] + tol < y < r[3] - tol


def assert_clear(out, rects):
    """No point and no densely sampled segment point inside any rect."""
    for (_, x, y, _k) in out:
        assert not any(inside(x, y, r) for r in rects), (x, y)
    for a, b in zip(out, out[1:]):
        for f in np.linspace(0, 1, 200):
            x, y = a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
            assert not any(inside(x, y, r) for r in rects), (a, b, f)


def test_rules_geometry_2026():
    r = rules()
    assert r.start_line_x == pytest.approx(4.00, abs=0.01)
    assert len(r.obstacles) == 6
    assert r.field_m == pytest.approx((L, W))
    hub = r.obstacles[0]
    assert hub == pytest.approx((4.029, 3.438, 5.222, 4.632), abs=2e-3)
    ys = sorted((o[1], o[3]) for o in r.obstacles if o[0] < L / 2)
    assert ys[0] == pytest.approx((1.265, 1.577), abs=2e-3)
    assert ys[2] == pytest.approx((6.492, 6.804), abs=2e-3)
    # symmetric under the 2026 rotation (x, y) -> (L - x, W - y)
    rot = sorted(tuple(round(v, 3) for v in (L - x1, W - y1, L - x0, W - y0))
                 for x0, y0, x1, y1 in r.obstacles)
    assert rot == pytest.approx(sorted(tuple(round(v, 3) for v in o) for o in r.obstacles), abs=2e-3)


def test_rules_none_for_other_years():
    assert auto_rules(2025) is None
    assert auto_rules(2024) is None


def test_segment_enters():
    r = (0.0, 0.0, 1.0, 1.0)
    assert segment_enters(-1, 0.5, 2, 0.5, r)
    assert not segment_enters(-1, 0.0, 2, 0.0, r)       # along a side
    assert not segment_enters(-1, 2, 2, 2, r)
    assert not segment_enters(1.0, 0.5, 2.0, 0.5, r)    # leaves from the boundary


def test_segment_through_hub_gets_detour():
    # 2026hop_qm84 1287: straight line across a coverage gap cuts the hub
    pts = [(6.54, 7.32, 3.00), (8.71, 3.61, 4.20)]
    out, rep = constrain_auto_path(pts, rules())
    assert rep["n_detour"] >= 1 and rep["start_check"] == "unseen"
    kinds = [k for *_, k in out]
    assert kinds[0] == "seen" and kinds[-1] == "seen" and "detour" in kinds
    ts = [t for t, *_ in out]
    assert ts == sorted(ts) and ts[0] == 6.54 and ts[-1] == 8.71
    assert_clear(out, inflated())


def test_point_in_trench_block_clamped():
    pts = [(0.0, 4.0, 2.5), (0.5, 4.6, 1.42), (1.0, 4.6, 0.6)]
    out, rep = constrain_auto_path(pts, rules())
    assert rep["n_clamped"] == 1
    clamped = [o for o in out if o[3] == "clamped"]
    assert len(clamped) == 1 and clamped[0][0] == 0.5
    assert_clear(out, inflated())


def test_start_ok():
    out, rep = constrain_auto_path([(0.0, 3.6, 2.5), (1.0, 3.0, 2.5)], rules())
    assert rep["start_check"] == "ok" and rep["start_offset_m"] == 0.0
    assert out[0][3] == "seen"


def test_start_inferred_prepends_on_line():
    r = rules()
    out, rep = constrain_auto_path([(0.5, 2.6, 2.5), (1.5, 2.0, 2.5)], r)
    assert rep["start_check"] == "inferred"
    t, x, y, kind = out[0]
    assert (t, kind) == (0.0, "start")
    assert abs(x - r.start_line_x) <= 0.8 + 1e-9 and y == pytest.approx(2.5)
    assert rep["start_offset_m"] == pytest.approx(0.6, abs=0.02)
    assert [o[3] for o in out[1:]] == ["seen", "seen"]


def test_start_inferred_at_t0_moves_first_point():
    out, rep = constrain_auto_path([(0.0, 2.9, 2.5), (1.0, 2.5, 2.5)], rules())
    assert rep["start_check"] == "inferred"
    assert len(out) == 2 and out[0][3] == "start" and out[0][0] == 0.0
    assert out[0][1] == pytest.approx(rules().start_line_x - 0.8)


def test_start_violation_far_wall():
    # a blue robot first seen at the red wall (another robot's slot)
    out, rep = constrain_auto_path([(0.3, 15.0, 4.0), (2.0, 13.5, 4.0)], rules())
    assert rep["start_check"] == "violation" and rep["start_offset_m"] > 9
    assert all(k != "start" for *_, k in out)


def test_start_unseen_late_first_sample():
    out, rep = constrain_auto_path([(13.0, 8.0, 2.0), (14.0, 8.5, 2.0)], rules())
    assert rep["start_check"] == "unseen" and rep["start_offset_m"] is None
    assert out[0][3] == "seen"


def test_inferred_start_segment_is_detoured():
    # first seen just past the hub at 0.6 s: the inferred start is in front of
    # the hub, and the line from it must not go through the hub
    out, rep = constrain_auto_path([(0.6, 5.55, 4.0), (1.5, 6.5, 4.0)], rules())
    assert rep["start_check"] == "inferred"
    assert out[0][3] == "start" and abs(out[0][1] - rules().start_line_x) <= 0.8 + 1e-9
    assert rep["n_detour"] >= 1
    assert_clear(out, inflated())


def test_start_under_trench_back_bumper_on_line_ok():
    # ~0.85 m robot in the trench lane, back bumper on the line: centre is
    # ~0.43 m past the line, 0.6 m off the wall — a legal start
    r = rules()
    out, rep = constrain_auto_path([(0.0, r.start_line_x + 0.43, 0.6),
                                    (1.0, r.start_line_x + 1.5, 0.6)], r)
    assert rep["start_check"] == "ok" and rep["n_clamped"] == 0
    out, rep = constrain_auto_path([(0.0, r.start_line_x + 0.43, W - 0.6),
                                    (1.0, r.start_line_x + 1.5, W - 0.6)], r)
    assert rep["start_check"] == "ok" and rep["n_clamped"] == 0


def test_start_snaps_sideways_into_trench_lane():
    # first seen in the lane past the trench (2026hop qm24 3506: (5.71, 1.04) at
    # 0.51 s). The nearest legal start is in the lane under the trench, not
    # back behind the trench block toward the wall.
    r = rules()
    out, rep = constrain_auto_path([(0.51, 5.71, 1.04), (1.5, 7.0, 1.0)], r)
    assert rep["start_check"] == "inferred"
    t, x, y, kind = out[0]
    assert kind == "start" and x == pytest.approx(r.start_line_x + 0.8)
    assert y <= 49.82 * 0.0254 - MARGIN + 1e-6   # in the lane, under the trench
    assert rep["start_offset_m"] < 1.0
    assert_clear(out, inflated())


def test_empty_path():
    out, rep = constrain_auto_path([], rules())
    assert out == [] and rep["start_check"] is None


# ------------------------------------------------------------------ AutoPaths end-to-end
def mk_path(fn, t0=0.0, t1=15.0, hz=5.0, red=False):
    out = []
    for i in range(int(round((t1 - t0) * hz)) + 1):
        t = t0 + i / hz
        ax, ay = fn(t)
        x, y = (L - ax, W - ay) if red else (ax, ay)
        out.append(Sample(frame_num=i, t_match=round(t, 4), period="auto", remaining=20 - t,
                          x=x, y=y, ax=ax, ay=ay, dt=0.2))
    return out


def legal(t):          # starts on the line, drives through the hub (gap-like straight)
    return 3.6 + 0.4 * min(t, 12.0), 4.0 + 0.05 * np.sin(t)


def mk_match(num, paths, blue=(), red=()):
    md = MatchData(event_key="2026test", match_key=f"2026test_qm{num}", comp_level="qm",
                   match_number=num, set_number=1, year=2026, replay_id=f"r{num}",
                   replay_created_at=None, legacy_labels=False,
                   alliances={"red": list(red) + [None] * (3 - len(red)),
                              "blue": list(blue) + [None] * (3 - len(blue))},
                   hub_auto={"red": None, "blue": None}, frame_period_s=0.2)
    for team, (path, alliance) in paths.items():
        md.teams[team] = TeamMatch(team=team, alliance=alliance, station=1, auto_path=path,
                                   auto_slot="B1", auto_label_share=0.9,
                                   auto_runner_up_share=0.1, auto_source="autonomous_paths")
    return md


def test_autopaths_end_to_end():
    far = lambda t: (15.0 - 0.3 * t, 4.0)            # noqa: E731  starts at the red wall
    mds = [mk_match(1, {100: (mk_path(legal), "blue")}, blue=[100]),
           mk_match(2, {100: (mk_path(legal, red=True), "red")}, red=[100]),
           mk_match(3, {100: (mk_path(far, t0=0.2), "blue")}, blue=[100])]
    team = AutoPaths().compute(mds, dict(AutoPaths.default_params))["teams"]["100"]
    m1, m2, m3 = team["matches"]
    assert m3["start_check"] == "violation" and m3["cluster"] is None
    assert m3["points"] and len(m3["point_kinds"]) == len(m3["points"])
    assert all(c["count"] for c in team["clusters"])
    assert "2026test_qm3" not in [k for c in team["clusters"] for k in c["members"]]
    assert m1["cluster"] is not None and m1["cluster"] == m2["cluster"]
    for m in (m1, m2):
        assert m["start_check"] == "ok"
        assert m["n_detour"] >= 1 and "d" in m["point_kinds"]
        assert len(m["point_kinds"]) == len(m["points"]) == len(m["points_norm"])
        assert m["n_samples"] == len(mk_path(legal))
        pts = [(t, x, y, "s") for t, x, y in m["points"]]
        assert_clear(pts, inflated(MARGIN - 0.02))       # points are rounded to 1 cm
    # red points come back in alliance perspective, identical to the blue match
    assert m1["points"] == m2["points"]


def test_autopaths_rules_off_and_other_year():
    md = mk_match(1, {100: (mk_path(legal), "blue")}, blue=[100])
    m = AutoPaths().compute([md], {**AutoPaths.default_params, "field_rules": False})["teams"]["100"]["matches"][0]
    assert m["start_check"] is None and set(m["point_kinds"]) == {"s"}
    md.year = 2025
    m = AutoPaths().compute([md], dict(AutoPaths.default_params))["teams"]["100"]["matches"][0]
    assert m["start_check"] is None and m["n_detour"] == 0


def test_field_rules_bool_param_parses():
    from web.scouting.registry import merged_params
    assert merged_params(AutoPaths(), {"field_rules": "false"})["field_rules"] is False
    with pytest.raises(ValueError):
        merged_params(AutoPaths(), {"obstacle_margin_m": 5.0})

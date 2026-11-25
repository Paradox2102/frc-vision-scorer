"""Finalize re-resolve keeps the live per-point payload (state / tracker_prob /
det_conf) on the rebuilt replay robot rows."""
from web.app import _carry_point_payload


def test_carry_point_payload_by_frame_and_slot():
    orig = [{"frame_num": 10, "robots": [
                {"slot": "R1", "state": "visible", "tracker_prob": 0.9, "det_conf": 0.8},
                {"slot": "B1", "state": "coasting", "tracker_prob": None, "det_conf": None}]},
            {"frame_num": 16, "robots": [{"slot": "R1", "state": "visible", "det_conf": 0.7}]}]
    rebuilt = [{"frame_num": 10, "robots": [{"slot": "B1", "team_number": 1},
                                            {"slot": "R1", "team_number": 2},
                                            {"slot": "R3", "team_number": 3}]},
               {"frame_num": 16, "robots": [{"slot": "R1"}]},
               {"frame_num": 22, "robots": [{"slot": "R1"}]}]
    out = _carry_point_payload(rebuilt, orig)
    r = {x["slot"]: x for x in out[0]["robots"]}
    assert r["R1"] == {"slot": "R1", "team_number": 2, "state": "visible",
                       "tracker_prob": 0.9, "det_conf": 0.8}
    assert r["B1"]["state"] == "coasting" and r["B1"]["det_conf"] is None
    assert "det_conf" not in r["R3"]                    # no recorded row
    assert out[1]["robots"][0] == {"slot": "R1", "state": "visible", "det_conf": 0.7}
    assert out[2]["robots"][0] == {"slot": "R1"}          # frame not recorded


def test_carry_point_payload_does_not_override_rebuilt_keys():
    orig = [{"frame_num": 1, "robots": [{"slot": "R1", "det_conf": 0.5}]}]
    rebuilt = [{"frame_num": 1, "robots": [{"slot": "R1", "det_conf": 0.6}]}]
    assert _carry_point_payload(rebuilt, orig)[0]["robots"][0]["det_conf"] == 0.6

"""alliance_by_territory knob: unknown-colour detections / identity births no
longer take their alliance from the field half when the knob is False.

qm8 hand GT: in the opponent's half a robot rode a wrong-alliance slot 63% of
its tracked time (own half 4%) — the territory rule hands an unknown-colour
detection of a red robot in the blue half to the BLUE alliance.
"""
import numpy as np
import pytest

from web.field.detect_schema import select_gap_fills
from web.field.identity import IdentityObservation, IdentityRegistry
from web.field.tracker import FieldDetection, FieldTracker
from web.job_config import JobConfig

L = 16.541          # 2026 field length; midline at 8.27


def _det(x, y, color, box=(0, 0, 10, 10), tid=None):
    return FieldDetection(pos=(x, y), color=color, box=box, score=0.9,
                          track_id=tid)


def _slot(trk, sid):
    return next(t for t in trk.tracks if t.slot_id == sid)


def _red_crosses_midfield(territory: bool):
    """A red robot (seeded red at x=9.4) drives into the blue half seen only as
    unknown-colour detections; a blue robot sits far away at x=3."""
    trk = FieldTracker(L, alliance_by_territory=territory)
    trk.update([_det(9.4, 4.0, "red"), _det(3.0, 2.0, "blue")], {}, 0.1)
    red_sid = trk.last_det_to_slot[0]
    assert _slot(trk, red_sid).alliance == "red"
    x = 9.4
    claims = []
    for _ in range(6):
        x -= 0.4                                  # 9.0 ... 7.0 (crosses 8.27)
        trk.update([_det(x, 4.0, "unknown"), _det(3.0, 2.0, "blue")], {}, 0.1)
        claims.append((x, trk.last_det_to_slot.get(0)))
    return trk, red_sid, claims


# ---------------------------------------------------------------- tracker

def test_default_is_territory():
    assert FieldTracker(L).alliance_by_territory is True
    assert FieldTracker(L).config_dict()["alliance_by_territory"] is True


def test_bug_documented_territory_hands_red_robot_to_blue():
    trk, red_sid, claims = _red_crosses_midfield(territory=True)
    blue_half = [(x, sid) for x, sid in claims if x < L / 2]
    assert blue_half
    # Every unknown det in the blue half is claimed by a BLUE slot (a fresh
    # seed — the red slot is never offered it) ...
    for _x, sid in blue_half:
        assert sid is not None and _slot(trk, sid).alliance == "blue"
    # ... and the red slot coasts.
    assert _slot(trk, red_sid).missed > 0


def test_territory_free_red_robot_stays_on_red_slot():
    trk, red_sid, claims = _red_crosses_midfield(territory=False)
    for _x, sid in claims:
        assert sid == red_sid
    red = _slot(trk, red_sid)
    assert red.missed == 0 and red.kf.pos[0] < L / 2
    # No blue slot was seeded off the red robot (only the x=3 blue robot).
    assert sum(t.active for t in trk.tracks if t.alliance == "blue") == 1


def test_territory_free_unknown_never_seeds():
    trk = FieldTracker(L, alliance_by_territory=False)
    out = trk.update([_det(3.0, 2.0, "unknown"), _det(12.0, 5.0, "unknown")],
                     {}, 0.1)
    assert out == [] and trk.last_det_to_slot == {}
    # With a live red slot, an out-of-gate unknown det still seeds nothing.
    trk.update([_det(12.0, 5.0, "red")], {}, 0.1)
    trk.update([_det(12.0, 5.0, "red"), _det(2.0, 7.0, "unknown")], {}, 0.1)
    assert sum(t.active for t in trk.tracks) == 1
    assert 1 not in trk.last_det_to_slot


def test_territory_free_unknown_with_trackid_never_seeds():
    trk = FieldTracker(L, alliance_by_territory=False)
    trk.update([_det(3.0, 2.0, "unknown", tid=7)], {}, 0.1)
    assert not any(t.active for t in trk.tracks)


def test_territory_free_one_unknown_claims_one_slot():
    """A red and a blue slot both within gate of ONE unknown detection: the
    joint pass gives it to exactly one (the nearer) and the other coasts."""
    trk = FieldTracker(L, alliance_by_territory=False)
    trk.update([_det(8.0, 4.0, "red"), _det(8.9, 4.0, "blue")], {}, 0.1)
    red_sid, blue_sid = trk.last_det_to_slot[0], trk.last_det_to_slot[1]
    trk.update([_det(8.2, 4.0, "unknown")], {}, 0.1)
    assert list(trk.last_det_to_slot.items()) == [(0, red_sid)]
    assert _slot(trk, red_sid).missed == 0
    assert _slot(trk, blue_sid).missed == 1


def test_territory_free_two_unknowns_pair_across_alliances():
    trk = FieldTracker(L, alliance_by_territory=False)
    trk.update([_det(7.0, 3.0, "red"), _det(7.5, 6.0, "blue")], {}, 0.1)
    red_sid, blue_sid = trk.last_det_to_slot[0], trk.last_det_to_slot[1]
    trk.update([_det(7.6, 5.8, "unknown"), _det(7.1, 3.1, "unknown")], {}, 0.1)
    assert trk.last_det_to_slot == {0: blue_sid, 1: red_sid}
    assert len(set(trk.last_det_to_slot.values())) == 2


def test_territory_free_known_colour_claims_first():
    """A slot that took its own known-colour detection is not also offered an
    unknown one (no double measurement)."""
    trk = FieldTracker(L, alliance_by_territory=False)
    trk.update([_det(12.0, 4.0, "red")], {}, 0.1)
    sid = trk.last_det_to_slot[0]
    trk.update([_det(12.1, 4.0, "red"), _det(12.3, 4.2, "unknown")], {}, 0.1)
    assert trk.last_det_to_slot == {0: sid}
    assert sum(t.active for t in trk.tracks) == 1


def _run(trk, frames):
    log = []
    for dets in frames:
        out = trk.update(dets, {}, 0.1)
        log.append((sorted((t.slot_id, t.alliance, round(t.pos[0], 6),
                            round(t.pos[1], 6), t.state) for t in out),
                    dict(trk.last_det_to_slot),
                    [t.missed for t in trk.tracks]))
    return log


def test_known_colour_only_sequence_identical_across_modes():
    """With no unknown-colour detections the territory-free path must be the
    legacy path exactly (seed, match, coast, expire)."""
    rng = np.random.default_rng(0)
    frames = []
    base = {"blue": [(2.0, 2.0), (4.0, 5.0), (6.0, 7.0)],
            "red": [(10.0, 2.0), (12.0, 5.0), (14.0, 7.0)]}
    for k in range(40):
        dets = []
        for col, pts in base.items():
            for j, (x, y) in enumerate(pts):
                if (k + j) % 7 == 3:          # periodic dropouts -> coasting
                    continue
                dets.append(_det(x + 0.05 * k + rng.normal(0, 0.05),
                                 y + rng.normal(0, 0.05), col,
                                 tid=(10 * j + (0 if col == "blue" else 5))))
        frames.append(dets)
    frames += [[]] * 12                        # everything coasts then expires
    a = _run(FieldTracker(L), frames)
    b = _run(FieldTracker(L, alliance_by_territory=False), frames)
    assert a == b


# --------------------------------------------------------------- identity

def test_identity_unknown_alliance_birth_deferred():
    reg = IdentityRegistry(alliance_by_territory=False)
    reg.update_frame(0, [IdentityObservation(slot_id=0, pos=(3.0, 2.0),
                                             alliance="unknown")],
                     alliance_side_fn=lambda p: "blue")
    assert reg.identities == {}
    assert reg.stats["births_deferred_no_alliance"] == 1


def test_identity_unknown_alliance_birth_default_uses_side():
    reg = IdentityRegistry()
    reg.update_frame(0, [IdentityObservation(slot_id=0, pos=(3.0, 2.0),
                                             alliance="unknown")],
                     alliance_side_fn=lambda p: "blue")
    assert [i.alliance for i in reg.identities.values()] == ["blue"]


def test_identity_unknown_alliance_still_matches_existing():
    reg = IdentityRegistry(alliance_by_territory=False)
    reg.update_frame(0, [IdentityObservation(slot_id=3, pos=(9.0, 4.0),
                                             alliance="red")])
    (rid,) = reg.identities
    reg.update_frame(1, [IdentityObservation(slot_id=3, pos=(8.9, 4.0),
                                             alliance="unknown")],
                     alliance_side_fn=lambda p: "blue")
    assert list(reg.identities) == [rid]
    assert reg.identities[rid].alliance == "red"
    assert reg.identities[rid].active_slot == 3


# ----------------------------------------------------- gap-fill roster cap

def test_gap_fills_default_per_half_cap():
    assert select_gap_fills(["blue", "blue"], ["blue", "blue", "red"]) == [0, 2]


def test_gap_fills_territory_free_whole_field_cap():
    # 4 robots in the blue half (a red robot crossed): legacy drops the 4th.
    assert select_gap_fills(["blue", "blue", "blue"], ["blue", None, "red"],
                            by_territory=False) == [0, 2]
    assert select_gap_fills(["blue"] * 3 + ["red"] * 2, ["blue", "red", "blue"],
                            by_territory=False) == [0]


# ------------------------------------------------------------- plumbing

def test_job_config_knob():
    assert JobConfig().alliance_by_territory is True
    assert JobConfig({"alliance_by_territory": False}).alliance_by_territory is False
    assert JobConfig({"alliance_by_territory": "0"}).alliance_by_territory is False


def test_tracker_config_dict_roundtrip():
    trk = FieldTracker(L, alliance_by_territory=False)
    cfg = trk.config_dict()
    assert FieldTracker(**cfg).alliance_by_territory is False


def test_app_identity_registry_receives_knob():
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig({"alliance_by_territory": False})
    assert job._build_identity_registry().alliance_by_territory is False
    job.config = JobConfig()
    assert job._build_identity_registry().alliance_by_territory is True


def test_app_tracker_constructor_passes_knob():
    """The FieldTracker in InferenceJob's field-tracking setup is built from
    JobConfig (needs calibration to run, so check the call site)."""
    import inspect
    import web.app as app
    src = inspect.getsource(app)
    assert "alliance_by_territory=self.config.alliance_by_territory)" in src
    assert src.count("by_territory=self.config.alliance_by_territory") >= 4

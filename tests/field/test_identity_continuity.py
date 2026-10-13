"""Identity churn: continuity gate (knob identity_continuity, default OFF),
young-identity appearance warm-up, and always-on churn telemetry.

qm11 full match (job 80a1cd4c): 575 identities in 168 s, median life 2
processed frames. 88% of teleop births happened with the slot moved < 1 m;
the only cost term that can push a same-slot match over the 0.75 association
threshold is the appearance veto (+0.5 when max cosine < app_floor), and a
young identity only ever compares against its birth embedding because the
EMA/exemplar update needs confidence >= 0.6.
"""
import numpy as np

from web.field.identity import IdentityObservation, IdentityRegistry, RobotIdentity


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


A = _e(1, 0, 0)
B = _e(0, 1, 0)          # cosine(A, B) = 0 -> below app_floor -> veto


def _obs(slot, pos, emb, **kw):
    return IdentityObservation(slot_id=slot, pos=pos, alliance="blue", embedding=emb, **kw)


def _two_frames(reg, second_obs):
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    first = reg.get_active_identities()[0].identity_id
    reg.update_frame(frame_idx=16, observations=[second_obs])
    return first


# --------------------------------------------------------------- continuity

def test_veto_splits_same_slot_when_continuity_off():
    """Today's behaviour (knob OFF): a dissimilar embedding on the SAME slot,
    moved 0.3 m, with no tracker event, is split into a new identity."""
    reg = IdentityRegistry()
    first = _two_frames(reg, _obs(0, (2.3, 2.0), B))
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first
    assert reg.stats["births"] == 2
    assert reg.stats["births_appearance_veto"] == 1


def test_continuity_keeps_identity_on_undisturbed_slot():
    reg = IdentityRegistry(continuity=True, continuity_max_m=1.0)
    first = _two_frames(reg, _obs(0, (2.3, 2.0), B))
    active = reg.get_active_identities()
    assert [i.identity_id for i in active] == [first]
    assert reg.stats["births"] == 1
    assert reg.stats["continuity_holds"] == 1


def test_continuity_still_splits_on_tracker_binding_event():
    reg = IdentityRegistry(continuity=True, continuity_max_m=1.0)
    first = _two_frames(reg, _obs(0, (2.3, 2.0), B, slot_disrupted=True))
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first


def test_continuity_still_splits_on_large_position_jump():
    reg = IdentityRegistry(continuity=True, continuity_max_m=1.0)
    first = _two_frames(reg, _obs(0, (3.5, 2.0), B))      # moved 1.5 m
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first


def test_continuity_still_splits_on_conflicting_team_evidence():
    reg = IdentityRegistry(continuity=True, continuity_max_m=1.0)
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    ident = reg.get_active_identities()[0]
    ident.team_number = "4786"
    reg.update_frame(frame_idx=16, observations=[
        _obs(0, (2.3, 2.0), B, team_number="2992")])
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != ident.identity_id
    assert reg.frame_rejections[0]["continuity_eligible"] is False


def test_continuity_only_for_previous_frame_holder_of_the_same_slot():
    """An identity that held a DIFFERENT slot, or wasn't observed last frame,
    gets no continuity pass."""
    reg = IdentityRegistry(continuity=True, continuity_max_m=1.0)
    reg.update_frame(frame_idx=8, observations=[_obs(1, (2.0, 2.0), A)])
    first = reg.get_active_identities()[0].identity_id
    reg.update_frame(frame_idx=16, observations=[_obs(0, (2.3, 2.0), B)])
    assert reg.get_active_identities()[0].identity_id != first

    reg2 = IdentityRegistry(continuity=True, continuity_max_m=1.0)
    reg2.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    first2 = reg2.get_active_identities()[0].identity_id
    reg2.update_frame(frame_idx=16, observations=[])          # unobserved frame
    reg2.update_frame(frame_idx=24, observations=[_obs(0, (2.3, 2.0), B)])
    assert reg2.get_active_identities()[0].identity_id != first2


def test_continuity_off_is_byte_identical_cost_path():
    """With the knob off, a similar-appearance continuation still matches and
    the veto still fires exactly as before (no hidden behaviour change)."""
    reg = IdentityRegistry()
    first = _two_frames(reg, _obs(0, (2.3, 2.0), _e(1, 0.1, 0)))
    assert reg.get_active_identities()[0].identity_id == first
    assert reg.stats["births"] == 1


# ------------------------------------------------------------------ warm-up

def test_young_identity_collects_exemplars_during_warmup():
    reg = IdentityRegistry(continuity=True, warmup_frames=4)
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    ident = reg.get_active_identities()[0]
    assert ident.confidence < reg.app_update_floor          # young: 0.5
    reg.update_frame(frame_idx=16, observations=[
        _obs(0, (2.1, 2.0), _e(1, 0.3, 0), relink_margin=0.5)])
    assert len(ident.recent_exemplars) == 2
    assert not np.allclose(ident.appearance_embedding, A)  # EMA moved
    assert np.allclose(ident.anchor_embedding, A)          # anchor never moves
    assert reg.stats["warmup_updates"] == 1


def test_warmup_window_closes_and_mature_gate_applies():
    reg = IdentityRegistry(continuity=True, warmup_frames=3)
    reg.update_frame(frame_idx=0, observations=[_obs(0, (2.0, 2.0), A)])
    ident = reg.get_active_identities()[0]
    for k in range(1, 6):
        ident.confidence = 0.5                               # stays young
        reg.update_frame(frame_idx=8 * k, observations=[
            _obs(0, (2.0, 2.0), _e(1, 0.05 * k, 0), relink_margin=0.5)])
    # birth sample + observations 2 and 3 only (warmup_frames=3)
    assert len(ident.recent_exemplars) == 3
    assert reg.stats["warmup_updates"] == 2


def test_warmup_respects_relink_margin():
    reg = IdentityRegistry(continuity=True, warmup_frames=4)
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    ident = reg.get_active_identities()[0]
    reg.update_frame(frame_idx=16, observations=[
        _obs(0, (2.1, 2.0), _e(1, 0.3, 0), relink_margin=0.01)])
    assert len(ident.recent_exemplars) == 1


def test_no_warmup_when_knob_off():
    reg = IdentityRegistry(warmup_frames=4)                  # continuity False
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    ident = reg.get_active_identities()[0]
    reg.update_frame(frame_idx=16, observations=[
        _obs(0, (2.1, 2.0), _e(1, 0.3, 0), relink_margin=0.5)])
    assert len(ident.recent_exemplars) == 1
    assert np.allclose(ident.appearance_embedding, A)


# ---------------------------------------------------------------- telemetry

def test_rejection_row_carries_sim_distance_cost_and_reason():
    reg = IdentityRegistry()
    first = _two_frames(reg, _obs(0, (2.3, 2.0), B))
    rows = reg.frame_rejections
    assert len(rows) == 1
    row = rows[0]
    assert row["slot_id"] == 0 and row["candidate_id"] == first
    assert row["reason"] == "appearance_veto"
    assert abs(row["max_sim"] - 0.0) < 1e-6
    assert abs(row["distance_m"] - 0.3) < 1e-6
    assert row["cost"] >= reg.association_threshold
    # the knob-OFF run reports whether continuity WOULD have held
    assert row["continuity_eligible"] is True and row["slot_disrupted"] is False
    assert row["outcome"] == "born"
    assert reg.stats["births_veto_continuity_eligible"] == 1


def test_rejections_reset_each_frame():
    reg = IdentityRegistry()
    _two_frames(reg, _obs(0, (2.3, 2.0), B))
    reg.update_frame(frame_idx=24, observations=[_obs(0, (2.3, 2.0), B)])
    assert reg.frame_rejections == []


def test_birth_reason_distance_and_no_candidate():
    reg = IdentityRegistry()
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    assert reg.stats["births_no_candidate"] == 1               # empty registry
    reg.update_frame(frame_idx=16, observations=[_obs(0, (8.0, 2.0), A)])
    assert reg.stats["births_distance"] == 1                   # 6 m jump


def test_birth_reason_team_conflict():
    reg = IdentityRegistry()
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    reg.get_active_identities()[0].team_number = "4786"
    reg.update_frame(frame_idx=16, observations=[
        _obs(0, (2.0, 2.0), A, team_number="2992")])
    assert reg.stats["births_team_conflict"] == 1


def test_evictions_counted_and_reported_per_frame():
    """Fourth blue identity evicts the oldest inactive provisional one."""
    reg = IdentityRegistry()
    embs = [_e(1, 0, 0), _e(0, 1, 0), _e(0, 0, 1)]
    for i in range(3):
        reg.update_frame(frame_idx=2 * i, observations=[
            _obs(i, (10.0 * i, 1.0), embs[i])])
        reg.update_frame(frame_idx=2 * i + 1, observations=[])
    oldest = min(reg.identities)
    reg.update_frame(frame_idx=10, observations=[_obs(5, (40.0, 1.0), _e(1, 1, 1))])
    assert reg.frame_evictions == [oldest]
    assert reg.stats["evictions"] == 1
    assert reg.frame_evictions == [oldest] and oldest not in reg.identities


def test_cap_rejected_creation_counted():
    reg = IdentityRegistry()
    reg.identities = {
        i: RobotIdentity(identity_id=i, alliance="blue", is_confirmed=True,
                         appearance_embedding=_e(*([0] * i + [1] + [0] * (3 - i))),
                         last_seen_pos=(float(i), 1.0))
        for i in range(3)
    }
    reg._next_id = 3
    reg.update_frame(frame_idx=10, observations=[_obs(9, (30.0, 1.0), _e(1, 1, 1, 1))])
    assert reg.stats["cap_rejected"] == 1
    assert reg.stats["births"] == 0


def test_born_frame_and_obs_count_tracked():
    reg = IdentityRegistry()
    reg.update_frame(frame_idx=8, observations=[_obs(0, (2.0, 2.0), A)])
    ident = reg.get_active_identities()[0]
    assert ident.born_frame == 8 and ident.obs_count == 1
    reg.update_frame(frame_idx=16, observations=[_obs(0, (2.0, 2.0), A)])
    assert ident.born_frame == 8 and ident.obs_count == 2


# ------------------------------------------------------------ end-of-job line

def test_churn_summary_line_and_payload():
    from collections import Counter
    from web.field.identity import churn_summary
    stats = Counter({"births": 575, "births_appearance_veto": 506,
                     "births_veto_continuity_eligible": 450, "births_distance": 12,
                     "births_no_candidate": 40, "births_other": 17,
                     "evictions": 569, "matches": 9000})
    line, payload = churn_summary(
        stats, final_ages_s={"B1": 0.4, "R1": 32.5}, span_s=168.0, continuity=False)
    assert line.startswith("[identity] births 575 (appearance_veto 506 "
                           "[continuity-eligible 450], distance 12, team_conflict 0, "
                           "no_candidate 40, other 17)")
    assert "205.4/min" in line and "evictions 569" in line
    assert "continuity=off" in line and "final ages B1 0.4s R1 32.5s" in line
    assert payload["births"] == 575 and payload["births_per_min"] == 205.4
    assert payload["births_by_reason"]["appearance_veto"] == 506
    assert payload["final_ages_s"] == {"B1": 0.4, "R1": 32.5}

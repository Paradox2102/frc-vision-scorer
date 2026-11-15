"""fused_evidence_input_order: FusedTrackRegistry.assign evidence alignment.

``assign`` builds its evidence list pass by pass (direct key matches, then
appearance relinks / new-with-embedding, then new-without-embedding), but its
callers (merge_same_id_observations, InferenceJob._sync_identity_registry_frame)
read it index-aligned to the input observations. With the knob ON the list is
returned in input order; OFF keeps the legacy pass order byte-identical."""
import numpy as np

from web.field.fused_track_id import FusedTrackRegistry
from web.field.fusion import FusedObservation, merge_same_id_observations
from web.field.tracker import FieldDetection
from web.job_config import KNOBS, JobConfig

MAIN_BOX = (0, 0, 10, 10)


def _emb(seed):
    v = np.random.default_rng(seed).normal(size=8).astype(np.float32)
    return v / np.linalg.norm(v)


def _cluster(members, pos, main=True, emb=None):
    det = FieldDetection(pos=pos, color="unknown",
                         box=MAIN_BOX if main else None, embedding=emb)
    return FusedObservation(detection=det, evidence=[], members=list(members))


def _seed(reg):
    """Frame 1: ids 0 (main 1, with embedding) and 1 (main 2)."""
    reg.assign([_cluster([("main", 1)], (1.0, 1.0), emb=_emb(1)),
                _cluster([("main", 2)], (5.0, 5.0), emb=_emb(2))])


def _mixed_frame():
    """Input order that differs from pass order: a new no-embedding cluster,
    a new with-embedding cluster, an appearance relink of id 0 (new key, same
    embedding), then a direct continuation of id 1."""
    return [
        _cluster([("corner_left", 7)], (8.0, 2.0)),                  # new, no emb
        _cluster([("corner_right", 8)], (3.0, 6.0), emb=_emb(99)),   # new, emb
        _cluster([("main", 11)], (1.1, 1.0), emb=_emb(1)),           # relink -> 0
        _cluster([("main", 2)], (5.0, 5.1)),                         # continued 1
    ]


def test_legacy_evidence_is_in_pass_order_not_input_order():
    # Confirms the bug the knob fixes: evidence[i] is NOT observation i's.
    reg = FusedTrackRegistry()
    _seed(reg)
    obs = _mixed_frame()
    ev = reg.assign(obs)
    assert [e.type for e in ev] == [
        "continued", "newly_created", "appearance_relinked", "newly_created"]
    assert [e.track_id for e in ev] != [o.track_id for o in obs]
    # The per-observation stamp is right; only the returned list is misaligned.
    assert [o.relink_evidence.type for o in obs] == [
        "newly_created", "newly_created", "appearance_relinked", "continued"]
    # Always-on diagnostic counts the mispaired positions (knob-independent).
    assert reg.stats["evidence_misaligned"] == sum(
        e is not o.relink_evidence for e, o in zip(ev, obs))
    assert reg.stats["evidence_misaligned_type"] == sum(
        (e.type, e.margin) != (o.relink_evidence.type, o.relink_evidence.margin)
        for e, o in zip(ev, obs))
    assert reg.stats["evidence_misaligned"] > 0


def test_input_order_evidence_aligns_with_observations():
    reg = FusedTrackRegistry(evidence_input_order=True)
    _seed(reg)
    obs = _mixed_frame()
    ev = reg.assign(obs)
    assert len(ev) == len(obs)
    for e, o in zip(ev, obs):
        assert e is o.relink_evidence
        assert e.track_id == o.track_id
    assert [e.type for e in ev] == [
        "newly_created", "newly_created", "appearance_relinked", "continued"]
    assert obs[2].track_id == 0 and obs[3].track_id == 1


def test_ids_and_state_identical_with_knob_on_or_off():
    # The knob changes only the returned list's order, never id assignment.
    outs = []
    for flag in (False, True):
        reg = FusedTrackRegistry(evidence_input_order=flag)
        _seed(reg)
        obs = _mixed_frame()
        reg.assign(obs)
        outs.append(([o.track_id for o in obs],
                     [(o.relink_evidence.type, o.relink_evidence.margin)
                      for o in obs],
                     dict(reg.stats), dict(reg._key_to_id)))
    assert outs[0] == outs[1]


def test_merge_keeps_retained_members_evidence():
    # A new cluster precedes a main obs + its corner duplicate (same id). The
    # merged observation must carry the continued id's own evidence.
    reg = FusedTrackRegistry(evidence_input_order=True)
    reg.assign([_cluster([("main", 1), ("corner_left", 4)], (2.0, 2.0))])  # id 0
    obs = [_cluster([("corner_right", 9)], (7.0, 7.0)),                     # new
           _cluster([("main", 1)], (2.0, 2.0)),                             # cont 0
           _cluster([("corner_left", 4)], (2.3, 2.1), main=False)]          # cont 0
    ev = reg.assign(obs)
    merged, mev = merge_same_id_observations(obs, ev)
    assert [o.track_id for o in merged] == [1, 0]
    assert [e.track_id for e in mev] == [1, 0]
    assert [e.type for e in mev] == ["newly_created", "continued"]


def test_knob_default_off_and_wired(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    from web.app import InferenceJob
    cfg = JobConfig()
    assert cfg.fused_evidence_input_order is False
    job = InferenceJob.__new__(InferenceJob)
    job.config = cfg
    assert InferenceJob._build_fused_registry(job).evidence_input_order is False
    job.config = JobConfig({"fused_evidence_input_order": "true"})
    assert InferenceJob._build_fused_registry(job).evidence_input_order is True

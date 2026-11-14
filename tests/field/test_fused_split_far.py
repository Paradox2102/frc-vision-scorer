"""fused_split_far: same-id fused observations far apart are different robots.

Covers the merge distance guard (merge_same_id_observations max_merge_m /
on_split), the registry far-key unbind pre-pass and split_off, and the knob
wiring. Knob off must be byte-identical to the legacy path."""
import copy

import numpy as np

from web.field.fused_track_id import FusedTrackRegistry
from web.field.fusion import FusedObservation, merge_same_id_observations
from web.field.tracker import FieldDetection
from web.job_config import KNOBS, JobConfig

MAIN_BOX = (0, 0, 10, 10)


def _obs(pos, box, tid, cam, sam_id=None, emb=None):
    det = FieldDetection(pos=pos, color="blue", box=box, track_id=tid,
                         embedding=emb)
    sid = tid if sam_id is None else sam_id
    return FusedObservation(detection=det, evidence=[(cam, b"crop", 1.0, pos)],
                            members=[(cam, sid)], track_id=tid)


def _cluster(members, pos, main=True, emb=None):
    det = FieldDetection(pos=pos, color="unknown",
                         box=MAIN_BOX if main else None, embedding=emb)
    return FusedObservation(detection=det, evidence=[], members=list(members))


def _emb(seed):
    v = np.random.default_rng(seed).normal(size=8).astype(np.float32)
    return v / np.linalg.norm(v)


def _snapshot(obs_list):
    return [(o.track_id, o.detection.track_id, tuple(o.detection.pos),
             o.detection.box, tuple(map(tuple, o.members)), len(o.evidence))
            for o in obs_list]


# ---- merge distance guard ---------------------------------------------------

def _qm2_group():
    # main robot X with id 1; a corner SAM object whose key still maps to id 1
    # now sits on robot Y 4.8 m away; plus an unrelated robot (id 3).
    a = _obs((6.0, 3.4), MAIN_BOX, 1, "main")
    b = _obs((10.8, 3.4), None, 1, "corner_left", sam_id=7)
    c = _obs((2.4, 5.1), MAIN_BOX, 3, "main")
    return a, b, c


def test_off_is_identical_to_legacy():
    for main_guard in (False, True):
        objs = list(_qm2_group()) + [_obs((6.4, 3.4), None, 1, "corner_right")]
        ev = ["eA", "eB", "eC", "eD"]
        legacy_in = copy.deepcopy(objs)
        legacy, legacy_ev = merge_same_id_observations(
            legacy_in, list(ev), main_guard=main_guard)
        off, off_ev = merge_same_id_observations(
            copy.deepcopy(objs), list(ev), main_guard=main_guard,
            max_merge_m=0.0, on_split=lambda *_: 1 / 0)
        assert _snapshot(off) == _snapshot(legacy)
        assert off_ev == legacy_ev


def test_far_corner_member_not_merged_and_drops_shared_id():
    a, b, c = _qm2_group()
    merged, ev = merge_same_id_observations([a, b, c], ["eA", "eB", "eC"],
                                            max_merge_m=2.5)
    assert len(merged) == 3 and len(ev) == 3
    kept = next(o for o in merged if o.track_id == 1)
    assert kept.detection.pos == (6.0, 3.4)
    assert set(kept.members) == {("main", 1)}           # corner not unioned in
    far = next(o for o in merged if o.detection.pos == (10.8, 3.4))
    assert far.track_id is None and far.detection.track_id is None
    assert ev == ["eA", "eB", "eC"]                     # index-aligned
    assert [o.detection.pos for o in merged] == [(6.0, 3.4), (10.8, 3.4), (2.4, 5.1)]


def test_near_corner_duplicate_still_merges():
    a = _obs((6.0, 3.4), MAIN_BOX, 1, "main")
    b = _obs((6.5, 3.4), None, 1, "corner_left", sam_id=7)   # 0.5 m
    merged, ev = merge_same_id_observations([a, b], ["eA", "eB"], max_merge_m=2.5)
    assert len(merged) == 1 and ev == ["eA"]
    assert merged[0].track_id == 1 and merged[0].detection.pos == (6.0, 3.4)
    assert set(merged[0].members) == {("main", 1), ("corner_left", 7)}


def test_far_members_merge_among_themselves():
    a = _obs((6.0, 3.4), MAIN_BOX, 1, "main")
    b = _obs((10.8, 3.4), None, 1, "corner_left", sam_id=7)
    d = _obs((11.2, 3.5), None, 1, "corner_right", sam_id=8)   # 0.4 m from b
    merged, ev = merge_same_id_observations([a, b, d], ["eA", "eB", "eD"],
                                            max_merge_m=2.5)
    assert len(merged) == 2 and ev == ["eA", "eB"]
    far = merged[1]
    assert far.track_id is None
    assert set(far.members) == {("corner_left", 7), ("corner_right", 8)}


def test_on_split_mints_id_and_replaces_evidence():
    reg = FusedTrackRegistry(split_far_m=2.5)
    a, b, c = _qm2_group()
    reg._next_id = 10
    reg._key_to_id.update({("main", 1): 1, ("corner_left", 7): 1, ("main", 3): 3})
    merged, ev = merge_same_id_observations(
        [a, b, c], ["eA", "eB", "eC"], max_merge_m=2.5, on_split=reg.split_off)
    far = merged[1]
    assert far.track_id == 10 and far.detection.track_id == 10
    assert ev[0] == "eA" and ev[2] == "eC"
    assert ev[1].track_id == 10 and ev[1].type == "newly_created"
    assert far.relink_evidence is ev[1]
    # Key now continues the new id; the kept id's position is the kept member's.
    assert reg._key_to_id[("corner_left", 7)] == 10
    assert reg._key_to_id[("main", 1)] == 1
    assert reg._id_to_pos[1] == (6.0, 3.4) and reg._id_to_pos[10] == (10.8, 3.4)
    assert reg.stats["split_far"] == 1
    assert reg.last_frame_split["split"] == 1


def test_main_guard_applies_per_subgroup():
    # Two main-backed far apart: split (far one gets a new id) rather than both
    # passing through with the shared id.
    a = _obs((6.0, 3.4), MAIN_BOX, 1, "main")
    b = _obs((9.4, 4.3), MAIN_BOX, 1, "main", sam_id=2)
    merged, _ = merge_same_id_observations([a, b], main_guard=True, max_merge_m=2.5)
    assert [o.track_id for o in merged] == [1, None]
    # Two main-backed NEAR each other: main_guard still passes both through.
    b2 = _obs((7.0, 3.4), MAIN_BOX, 1, "main", sam_id=2)
    merged, _ = merge_same_id_observations([a, b2], main_guard=True, max_merge_m=2.5)
    assert [o.track_id for o in merged] == [1, 1]


# ---- registry far-key unbind ------------------------------------------------

def _seed_pair(reg):
    """Frame 1: main SAM 1 + corner SAM 7 cross-view paired on robot X."""
    f1 = _cluster([("main", 1), ("corner_left", 7)], (6.0, 3.4))
    reg.assign([f1])
    return f1.track_id


def test_registry_off_far_key_inherits_shared_id():
    reg = FusedTrackRegistry()
    x = _seed_pair(reg)
    a = _cluster([("main", 1)], (6.0, 3.4))
    b = _cluster([("corner_left", 7)], (10.8, 3.4), main=False)
    reg.assign([a, b])
    assert a.track_id == b.track_id == x                 # the bug
    assert reg.stats["key_unbound_far"] == 0


def test_registry_unbinds_key_that_jumped_to_far_cluster():
    reg = FusedTrackRegistry(split_far_m=2.5)
    x = _seed_pair(reg)
    a = _cluster([("main", 1)], (6.1, 3.4))
    b = _cluster([("corner_left", 7)], (10.8, 3.4), main=False)
    reg.assign([a, b])
    assert a.track_id == x
    assert b.track_id is not None and b.track_id != x     # its own id
    assert reg._key_to_id[("corner_left", 7)] == b.track_id
    assert reg.stats["key_unbound_far"] == 1
    assert reg.last_frame_split == {"unbound": 1, "split": 0}
    # Next frame both continue their own ids, no further unbinds.
    a2 = _cluster([("main", 1)], (6.2, 3.4))
    b2 = _cluster([("corner_left", 7)], (10.7, 3.4), main=False)
    reg.assign([a2, b2])
    assert (a2.track_id, b2.track_id) == (a.track_id, b.track_id)
    assert reg.last_frame_split == {"unbound": 0, "split": 0}
    assert reg.stats["key_unbound_far"] == 1


def test_registry_owner_is_nearest_last_position_not_main():
    # The MAIN SAM object jumped robots; the corner stayed: main is unbound.
    reg = FusedTrackRegistry(split_far_m=2.5)
    x = _seed_pair(reg)
    a = _cluster([("main", 1)], (10.8, 3.4))                 # main jumped
    b = _cluster([("corner_left", 7)], (6.3, 3.4), main=False)
    reg.assign([a, b])
    assert b.track_id == x and a.track_id != x


def test_registry_near_claimants_keep_shared_id():
    reg = FusedTrackRegistry(split_far_m=2.5)
    x = _seed_pair(reg)
    a = _cluster([("main", 1)], (6.0, 3.4))
    b = _cluster([("corner_left", 7)], (7.0, 3.4), main=False)   # 1 m: same robot
    reg.assign([a, b])
    assert a.track_id == b.track_id == x
    assert reg.stats["key_unbound_far"] == 0


def test_registry_far_key_does_not_collapse_two_ids():
    # Robot Y has its own id; the stale corner key (-> X's id) joins Y's cluster.
    # Legacy collapses X and Y into one id; with the split Y keeps its id.
    for split, expect_same in ((0.0, True), (2.5, False)):
        reg = FusedTrackRegistry(split_far_m=split)
        x = _seed_pair(reg)
        y0 = _cluster([("main", 2)], (10.8, 3.4))
        reg.assign([_cluster([("main", 1)], (6.0, 3.4)), y0])
        y = y0.track_id
        a = _cluster([("main", 1)], (6.0, 3.4))
        b = _cluster([("main", 2), ("corner_left", 7)], (10.8, 3.4))
        reg.assign([a, b])
        assert a.track_id == x
        assert (b.track_id == x) is expect_same
        if not expect_same:
            assert b.track_id == y
            assert reg._key_to_id[("corner_left", 7)] == y


def test_registry_unbound_cluster_relinks_or_creates_with_embedding():
    reg = FusedTrackRegistry(split_far_m=2.5)
    f1 = _cluster([("main", 1), ("corner_left", 7)], (6.0, 3.4), emb=_emb(1))
    reg.assign([f1])
    a = _cluster([("main", 1)], (6.0, 3.4), emb=_emb(1))
    b = _cluster([("corner_left", 7)], (10.8, 3.4), main=False, emb=_emb(2))
    ev = reg.assign([a, b])
    assert b.track_id != a.track_id
    assert reg.stats["created_with_emb"] == 2   # f1 + b
    assert {e.track_id for e in ev} == {a.track_id, b.track_id}


def test_end_to_end_assign_then_merge_keeps_far_robot():
    reg = FusedTrackRegistry(split_far_m=2.5)
    _seed_pair(reg)
    a = _cluster([("main", 1)], (6.0, 3.4))
    b = _cluster([("corner_left", 7)], (10.8, 3.4), main=False)
    ev = reg.assign([a, b])
    merged, _ = merge_same_id_observations(
        [a, b], ev, main_guard=True, max_merge_m=reg.split_far_m,
        on_split=reg.split_off)
    assert len(merged) == 2
    assert len({o.track_id for o in merged}) == 2
    assert reg.stats["split_far"] == 0    # registry already separated them


# ---- knob wiring ------------------------------------------------------------

def test_knob_default_off_and_wired(monkeypatch):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    from web.app import InferenceJob
    cfg = JobConfig()
    assert cfg.fused_split_far is False and cfg.fused_split_far_m == 2.5
    job = InferenceJob.__new__(InferenceJob)
    job.config = cfg
    assert InferenceJob._build_fused_registry(job).split_far_m == 0.0
    job.config = JobConfig({"fused_split_far": "true"})
    assert InferenceJob._build_fused_registry(job).split_far_m == 2.5
    job.config = JobConfig({"fused_split_far": "true", "fused_split_far_m": "3.0"})
    assert InferenceJob._build_fused_registry(job).split_far_m == 3.0


def test_summary_gauge_adds_split_counts_only_when_on(monkeypatch, capsys):
    for knob in KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    from web.app import InferenceJob
    from web.field.identity_diagnostics import IdentityDiagnosticRecorder
    for on in (False, True):
        job = InferenceJob.__new__(InferenceJob)
        job.config = JobConfig({"fused_split_far": str(on).lower()})
        job.identity_diagnostics = IdentityDiagnosticRecorder()
        job.fused_registry = InferenceJob._build_fused_registry(job)
        job.fused_registry.stats.update({"key_unbound_far": 4, "split_far": 1})
        job._record_fused_ids_summary()
        gauge = job.identity_diagnostics.summary()["gauges"]["fused_ids"]
        out = capsys.readouterr().out
        if on:
            assert gauge["key_unbound_far"] == 4 and gauge["split_far"] == 1
            assert "key_unbound_far 4 split_far 1" in out
        else:
            assert "key_unbound_far" not in gauge and "key_unbound_far" not in out

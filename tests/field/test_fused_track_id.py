import numpy as np

from web.field.fused_track_id import FusedTrackRegistry
from web.field.fusion import FusedObservation
from web.field.tracker import FieldDetection


def _obs(members, pos=(0.0, 0.0)):
    det = FieldDetection(pos=pos, color="unknown", box=(0, 0, 1, 1))
    return FusedObservation(detection=det, evidence=[], members=list(members))


def test_distinct_clusters_get_distinct_ids():
    reg = FusedTrackRegistry()
    a = _obs([("main", 1)])
    b = _obs([("main", 2)])
    reg.assign([a, b])
    assert a.track_id != b.track_id
    assert {a.track_id, b.track_id} == {0, 1}


def test_same_members_keep_id_across_frames():
    reg = FusedTrackRegistry()
    f1 = _obs([("main", 1)])
    reg.assign([f1])
    first = f1.track_id
    f2 = _obs([("main", 1)])      # next frame, same tracklet
    reg.assign([f2])
    assert f2.track_id == first


def test_corner_only_then_joined_by_main_keeps_id():
    # The key SP-T.2 case: a robot first seen ONLY by a corner camera, then later
    # also by main, must keep the SAME fused id (so its slot is stable).
    reg = FusedTrackRegistry()
    f1 = _obs([("corner_left", 1)])
    reg.assign([f1])
    cid = f1.track_id
    f2 = _obs([("corner_left", 1), ("main", 3)])  # main joins the cluster
    reg.assign([f2])
    assert f2.track_id == cid


def test_id_stamped_onto_detection():
    # The fused id must land on the FieldDetection the tracker will consume.
    reg = FusedTrackRegistry()
    obs = _obs([("main", 1)])
    reg.assign([obs])
    assert obs.detection.track_id == obs.track_id


def test_new_member_set_gets_fresh_id():
    reg = FusedTrackRegistry()
    reg.assign([_obs([("main", 1)])])         # id 0
    reg.assign([_obs([("main", 1)])])         # still 0
    fresh = _obs([("corner_right", 9)])       # never seen
    reg.assign([_obs([("main", 1)]), fresh])
    assert fresh.track_id == 1


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=384)
    return v / np.linalg.norm(v)


def test_merge_into_embeddingless_winner_adopts_loser_embedding():
    # Regression: a track created from an observation WITHOUT an embedding
    # (winner, lower id) later merges with a track that HAS one (loser).
    # The collapse blended winner+loser embeddings assuming the winner had
    # one -> KeyError on the winner id. The winner must adopt the loser's
    # appearance state instead of crashing.
    reg = FusedTrackRegistry()
    no_emb = _obs([("main", 1)])            # frame 1 -> id 0, no embedding
    reg.assign([no_emb])
    with_emb = _obs([("corner_left", 1)])   # frame 2 -> id 1, has embedding
    with_emb.detection.embedding = _emb(1)
    reg.assign([with_emb])

    merged = _obs([("main", 1), ("corner_left", 1)])
    reg.assign([merged])
    assert merged.track_id == 0
    assert reg.get_embedding(0) is not None


def test_bridging_two_prior_ids_collapses_to_one():
    # Two tracklets that were separate clusters become one cluster (e.g. SAM had
    # split a robot across cameras, now they're within radius). They must collapse
    # to a single fused id, not keep two.
    reg = FusedTrackRegistry()
    reg.assign([_obs([("main", 1)]), _obs([("corner_left", 1)])])  # ids 0 and 1
    merged = _obs([("main", 1), ("corner_left", 1)])
    reg.assign([merged])
    assert merged.track_id in (0, 1)

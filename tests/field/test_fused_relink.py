import numpy as np
import pytest

from web.field.fused_track_id import FusedTrackRegistry
from web.field.fusion import FusedObservation
from web.field.tracker import FieldDetection


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _obs(members, embedding=None, pos=(0.0, 0.0)):
    det = FieldDetection(pos=pos, color="unknown", box=(0, 0, 1, 1), embedding=embedding)
    return FusedObservation(detection=det, evidence=[], members=list(members))


def test_relink_lost_track_by_embedding():
    reg = FusedTrackRegistry(app_floor=0.55, ema_alpha=0.3)
    emb_a = _e(1, 0, 0)

    # Frame 1: robot seen on main:1. Gets fused ID 0.
    f1 = _obs([("main", 1)], embedding=emb_a)
    reg.assign([f1])
    assert f1.track_id == 0

    # Frame 2: robot is not seen (main:1 is absent).

    # Frame 3: robot re-appears on corner_left:2 (new SAM2 ID).
    # Its embedding is highly similar to A (cosine = 1.0).
    f3 = _obs([("corner_left", 2)], embedding=emb_a)
    reg.assign([f3])
    # Should re-link to fused ID 0!
    assert f3.track_id == 0


def test_no_relink_when_similarity_below_floor():
    reg = FusedTrackRegistry(app_floor=0.55, ema_alpha=0.3)
    emb_a = _e(1, 0, 0)
    emb_b = _e(0, 1, 0)  # orthogonal, cosine = 0.0

    # Frame 1: main:1 seen with A. Gets ID 0.
    f1 = _obs([("main", 1)], embedding=emb_a)
    reg.assign([f1])
    assert f1.track_id == 0

    # Frame 2: main:1 absent.

    # Frame 3: corner_left:2 appears with B.
    f3 = _obs([("corner_left", 2)], embedding=emb_b)
    reg.assign([f3])
    # Should get a new ID (1) because cosine similarity is below floor.
    assert f3.track_id == 1


def test_no_relink_to_already_active_track():
    reg = FusedTrackRegistry(app_floor=0.55, ema_alpha=0.3)
    emb_a = _e(1, 0, 0)

    # Frame 1: main:1 seen with A. Gets ID 0.
    f1 = _obs([("main", 1)], embedding=emb_a)
    reg.assign([f1])
    assert f1.track_id == 0

    # Frame 2: both robots seen.
    # main:1 is still active (ID 0).
    # corner_left:2 appears with same embedding A.
    # It must NOT re-link to ID 0 because ID 0 is already active in this frame.
    m1 = _obs([("main", 1)], embedding=emb_a)
    c2 = _obs([("corner_left", 2)], embedding=emb_a)
    reg.assign([m1, c2])
    
    assert m1.track_id == 0
    assert c2.track_id == 1  # gets a fresh ID!


def test_ema_updates_running_embedding():
    reg = FusedTrackRegistry(app_floor=0.55, ema_alpha=0.3)
    emb_a = _e(1, 0, 0)
    emb_b = _e(1, 1, 0)  # similar enough to A (cosine = 1/sqrt(2) = 0.707 > 0.55)

    f1 = _obs([("main", 1)], embedding=emb_a)
    reg.assign([f1])
    assert f1.track_id == 0

    # Running embedding starts as emb_a
    np.testing.assert_array_almost_equal(reg._id_to_embedding[0], emb_a)

    # Re-link with emb_b
    f2 = _obs([("corner_left", 2)], embedding=emb_b)
    reg.assign([f2])
    assert f2.track_id == 0

    # Expected blended embedding:
    # blended = (1 - 0.3) * emb_a + 0.3 * emb_b
    # blended = 0.7 * [1, 0, 0] + 0.3 * [0.707, 0.707, 0] = [0.912, 0.212, 0]
    # normalize it
    expected = 0.7 * emb_a + 0.3 * emb_b
    expected = expected / np.linalg.norm(expected)

    np.testing.assert_array_almost_equal(reg._id_to_embedding[0], expected)


def test_get_embedding_returns_ema_and_none_for_unknown():
    # Regression: the appearance gallery reads per-track embeddings via this
    # accessor; without it, RobotTrack.appearance_embedding stays None on the
    # field-tracking path and the gallery silently never deposits/queries.
    reg = FusedTrackRegistry()
    f1 = _obs([("main", 1)], embedding=_e(1, 0, 0))
    reg.assign([f1])
    emb = reg.get_embedding(f1.track_id)
    assert emb is not None
    np.testing.assert_array_almost_equal(emb, _e(1, 0, 0))
    assert reg.get_embedding(99999) is None


def test_permutation_invariance_and_margins():
    # Assert identical optimal assignments regardless of input order, and check margins.
    emb_a = _e(1, 0, 0)
    emb_b = _e(0, 1, 0)
    emb_c = _e(0, 0, 1)

    reg = FusedTrackRegistry(app_floor=0.5)

    # Frame 1: Create active tracks for 0, 1, 2
    f1 = _obs([("main", 1)], embedding=emb_a)
    f2 = _obs([("main", 2)], embedding=emb_b)
    f3 = _obs([("main", 3)], embedding=emb_c)
    reg.assign([f1, f2, f3])
    assert {f1.track_id, f2.track_id, f3.track_id} == {0, 1, 2}

    # Frame 2: Inactive tracks exist.
    # Now we have observations corresponding to emb_a and emb_b but with different per-camera IDs.
    # Observation list order A then B
    reg_order1 = FusedTrackRegistry(app_floor=0.5)
    reg_order1._key_to_id = reg._key_to_id.copy()
    reg_order1._next_id = reg._next_id
    reg_order1._id_to_embedding = reg._id_to_embedding.copy()
    reg_order1._id_to_anchor = reg._id_to_anchor.copy()
    reg_order1._recent_exemplars = {k: list(v) for k, v in reg._recent_exemplars.items()}

    obs_a = _obs([("corner", 1)], embedding=emb_a)
    obs_b = _obs([("corner", 2)], embedding=emb_b)

    reg_order1.assign([obs_a, obs_b])
    assert obs_a.track_id == f1.track_id
    assert obs_b.track_id == f2.track_id
    
    # Check margins:
    # For obs_a: matches to f1 (sim = 1.0), next best is f2 (sim = 0.0). Margin should be 1.0.
    assert obs_a.relink_margin == 1.0
    assert obs_b.relink_margin == 1.0

    # Test order B then A
    reg_order2 = FusedTrackRegistry(app_floor=0.5)
    reg_order2._key_to_id = reg._key_to_id.copy()
    reg_order2._next_id = reg._next_id
    reg_order2._id_to_embedding = reg._id_to_embedding.copy()
    reg_order2._id_to_anchor = reg._id_to_anchor.copy()
    reg_order2._recent_exemplars = {k: list(v) for k, v in reg._recent_exemplars.items()}

    obs_a2 = _obs([("corner", 1)], embedding=emb_a)
    obs_b2 = _obs([("corner", 2)], embedding=emb_b)

    reg_order2.assign([obs_b2, obs_a2])
    assert obs_a2.track_id == f1.track_id
    assert obs_b2.track_id == f2.track_id
    assert obs_a2.relink_margin == 1.0
    assert obs_b2.relink_margin == 1.0


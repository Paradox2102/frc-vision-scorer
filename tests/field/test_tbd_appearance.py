"""Appearance term in the selective TBD tracker (blue-gap fix #3).

Pure numpy; embeddings are hand-built unit vectors. The crossing scenario:
two same-color robots swap positions between frames. Pure expanded-IoU
assigns each track the nearest box (identity flip); with the appearance
term the tracks follow their embeddings instead.
"""
import numpy as np

from web.field.selective_prop import SelectiveConfig, TbdTracker

BOX_A = (0, 0, 20, 20)
BOX_B = (24, 0, 44, 20)          # close enough that expanded IoU (0.25) > 0
E_A = np.zeros(384, dtype=np.float32); E_A[0] = 1.0
E_B = np.zeros(384, dtype=np.float32); E_B[1] = 1.0   # orthogonal to E_A


def _warmup(tracker, n=3):
    """Establish two stationary same-color tracks with stable embeddings."""
    for f in range(n):
        tracker.step(f, [(BOX_A, "blue", E_A), (BOX_B, "blue", E_B)])
    # oids are spawn-ordered: 1 <- BOX_A, 2 <- BOX_B
    return 1, 2


def test_weight_zero_is_exact_noop():
    """app_weight=0 with embeddings present == no embeddings at all."""
    swap = [(BOX_B, "blue", E_A), (BOX_A, "blue", E_B)]
    t_emb = TbdTracker(SelectiveConfig(app_weight=0.0))
    _warmup(t_emb)
    obs_emb = t_emb.step(3, swap)

    t_plain = TbdTracker(SelectiveConfig())
    for f in range(3):
        t_plain.step(f, [(BOX_A, "blue"), (BOX_B, "blue")])
    obs_plain = t_plain.step(3, [(BOX_B, "blue"), (BOX_A, "blue")])

    assert obs_emb == obs_plain


def test_pure_eiou_flips_identity_on_swap():
    """Documents the failing baseline the appearance term exists to fix."""
    t = TbdTracker(SelectiveConfig(app_weight=0.0))
    oid_a, oid_b = _warmup(t)
    obs = t.step(3, [(BOX_B, "blue", E_A), (BOX_A, "blue", E_B)])
    # eIoU alone keeps each track on its old position -> identities flip.
    assert obs[oid_a].box == BOX_A
    assert obs[oid_b].box == BOX_B


def test_appearance_holds_identity_through_swap():
    t = TbdTracker(SelectiveConfig(app_weight=1.0))
    oid_a, oid_b = _warmup(t)
    obs = t.step(3, [(BOX_B, "blue", E_A), (BOX_A, "blue", E_B)])
    # Track A follows its embedding to BOX_B (the robot moved there).
    assert obs[oid_a].box == BOX_B
    assert obs[oid_b].box == BOX_A


def test_appearance_skipped_for_cross_color_and_missing_emb():
    """Term applies only when colors match and both sides have vectors."""
    t = TbdTracker(SelectiveConfig(app_weight=1.0))
    t.step(0, [(BOX_A, "blue", E_A), (BOX_B, "red", E_B)])
    # None embedding + color mismatch: must not raise, must still assign.
    obs = t.step(1, [(BOX_A, "blue", None), (BOX_B, "red", E_A)])
    assert obs[1].box == BOX_A
    assert obs[2].box == BOX_B


def test_ema_updates_only_on_confident_matches():
    cfg = SelectiveConfig(app_weight=1.0, app_ema=0.5, margin_exit=0.10)
    t = TbdTracker(cfg)
    # Spawn: track takes the det embedding directly.
    t.step(0, [(BOX_A, "blue", E_A)])
    trk = t._tracks[1]
    assert np.allclose(trk.emb, E_A)
    # Confident (single det, margin=inf): EMA moves toward E_B, renormalized.
    t.step(1, [(BOX_A, "blue", E_B)])
    mixed = 0.5 * E_A + 0.5 * E_B
    expected = mixed / np.linalg.norm(mixed)
    assert np.allclose(trk.emb, expected, atol=1e-6)


def test_ambiguous_match_does_not_update_ema():
    # app_weight must be >0 here: with the no-op gate at 0.0 the warmup
    # spawns tracks with emb=None and there is no gallery to protect.
    cfg = SelectiveConfig(app_weight=1.0, app_ema=0.5, margin_exit=0.10)
    t = TbdTracker(cfg)
    oid_a, oid_b = _warmup(t)
    emb_before = t.emb_of(oid_a).copy()
    # Two dets each overlapping both tracks -> tiny margins (ambiguous).
    mid1 = (6, 0, 26, 20)
    mid2 = (12, 0, 32, 20)
    obs = t.step(3, [(mid1, "blue", E_B), (mid2, "blue", E_B)])
    a_obs = obs[oid_a]
    assert a_obs.margin < cfg.margin_exit          # scenario sanity check
    assert np.allclose(t.emb_of(oid_a), emb_before)


def test_two_tuple_dets_still_accepted():
    """All existing callers pass (box, color); must keep working."""
    t = TbdTracker(SelectiveConfig(app_weight=1.0))
    obs = t.step(0, [(BOX_A, "blue"), (BOX_B, "red")])
    assert set(obs) == {1, 2}

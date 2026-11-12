"""Per-camera appearance (knob identity_per_cam_appearance, default OFF).

2025hop qm2, S4 cosines on the same slot: main-vs-main (<=2 s) median 0.93,
corner-vs-corner median 0.92, but main-vs-corner SAME frame median 0.62 (42%
below the 0.59 identity floor). With corner cameras on, a corner-only
observation (robot hidden from main) was vetoed against an identity built
from main crops — births 2-3x (qm3 49 -> 134), mostly appearance_veto. With
the knob ON the identity keeps per-camera appearance and only compares
same-camera embeddings; no shared camera = neutral appearance.
"""
import numpy as np

from web.field.fusion import FusedObservation, fuse_detections, merge_same_id_observations
from web.field.identity import (
    IdentityObservation, IdentityRegistry, appearance_ema_similarity)
from web.field.tracker import FieldDetection


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


MAIN_A = _e(1, 0, 0, 0)
# cosine(MAIN_A, CORNER_A) = 0.4 — the same robot from another viewpoint.
CORNER_A = _e(0.4, np.sqrt(1 - 0.16), 0, 0)
OTHER = _e(0, 0, 1, 0)        # cosine 0 vs both


def _obs(slot, pos, emb, cam_embeddings=None, **kw):
    return IdentityObservation(slot_id=slot, pos=pos, alliance="blue",
                               embedding=emb, cam_embeddings=cam_embeddings, **kw)


def _main_obs(pos=(2.0, 2.0), emb=MAIN_A):
    return _obs(0, pos, emb, {"main": emb})


def _corner_obs(pos=(2.3, 2.0), emb=CORNER_A, cam="corner_left"):
    return _obs(0, pos, emb, {cam: emb})


def _seed_then(reg, second):
    reg.update_frame(frame_idx=8, observations=[_main_obs()])
    first = reg.get_active_identities()[0].identity_id
    reg.update_frame(frame_idx=16, observations=[second])
    return first


# ------------------------------------------------------------------ knob OFF

def test_off_corner_only_low_cross_view_cosine_still_vetoed():
    """Knob OFF: today's behaviour — the merged embedding is compared, a
    corner-only view at cosine 0.4 vs the main-built identity is vetoed."""
    reg = IdentityRegistry(app_floor=0.59)
    assert reg.per_cam_appearance is False
    first = _seed_then(reg, _corner_obs())
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first
    assert reg.stats["births_appearance_veto"] == 1
    # And no per-camera state is kept.
    assert all(not i.cam_appearance for i in reg.identities.values())


def test_off_ignores_cam_embeddings_entirely():
    """OFF with identical inputs must equal OFF without cam_embeddings."""
    def run(with_cams):
        reg = IdentityRegistry(app_floor=0.59)
        seq = [(MAIN_A, "main", (2.0, 2.0)), (CORNER_A, "corner_left", (2.3, 2.0)),
               (MAIN_A, "main", (2.5, 2.0)), (OTHER, "main", (2.6, 2.0))]
        for k, (emb, cam, pos) in enumerate(seq):
            reg.update_frame(frame_idx=8 * (k + 1), observations=[
                _obs(0, pos, emb, {cam: emb} if with_cams else None)])
        return ([i.identity_id for i in reg.get_active_identities()],
                dict(reg.stats))
    assert run(True) == run(False)


# ------------------------------------------------------------------- knob ON

def test_on_corner_only_obs_vs_main_only_identity_is_neutral_and_matches():
    reg = IdentityRegistry(app_floor=0.59, per_cam_appearance=True)
    first = _seed_then(reg, _corner_obs())
    active = reg.get_active_identities()
    assert [i.identity_id for i in active] == [first]
    assert reg.stats["births"] == 1
    assert reg.stats["births_appearance_veto"] == 0
    assert reg.stats["percam_neutral_matches"] == 1


def test_on_neutral_appearance_gets_no_bonus():
    """No shared camera = no veto AND no bonus: a near position alone
    (cost 0.5 + 0.1*d) decides, so at 2.6 m (cost 0.76 > 0.75) it splits."""
    reg = IdentityRegistry(app_floor=0.59, per_cam_appearance=True)
    first = _seed_then(reg, _corner_obs(pos=(4.6, 2.0)))
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first
    row = reg.frame_rejections[0]
    assert row["max_sim"] is None and row["reason"] == "other"


def test_on_same_camera_low_cosine_still_vetoed():
    reg = IdentityRegistry(app_floor=0.59, per_cam_appearance=True)
    first = _seed_then(reg, _obs(0, (2.3, 2.0), OTHER, {"main": OTHER}))
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first
    assert reg.stats["births_appearance_veto"] == 1


def test_on_uses_shared_camera_among_several():
    """Identity seen from main + corner_left; observation from corner_left +
    corner_right. Only corner_left is shared and it matches -> no veto even
    though the merged embedding is far from the identity's merged one."""
    reg = IdentityRegistry(app_floor=0.59, per_cam_appearance=True)
    merged0 = _e(1, 1, 0, 0)
    reg.update_frame(frame_idx=8, observations=[
        _obs(0, (2.0, 2.0), merged0, {"main": MAIN_A, "corner_left": CORNER_A})])
    first = reg.get_active_identities()[0].identity_id
    obs = _obs(0, (2.3, 2.0), OTHER,
               {"corner_left": CORNER_A, "corner_right": OTHER})
    reg.update_frame(frame_idx=16, observations=[obs])
    assert [i.identity_id for i in reg.get_active_identities()] == [first]
    assert reg.stats["births_appearance_veto"] == 0


def test_on_shared_camera_mismatch_vetoes_even_if_other_cam_unseen():
    """Shared corner_left disagrees -> veto, the unshared corner_right
    embedding never rescues it."""
    reg = IdentityRegistry(app_floor=0.59, per_cam_appearance=True)
    reg.update_frame(frame_idx=8, observations=[
        _obs(0, (2.0, 2.0), MAIN_A, {"main": MAIN_A, "corner_left": CORNER_A})])
    first = reg.get_active_identities()[0].identity_id
    reg.update_frame(frame_idx=16, observations=[
        _obs(0, (2.3, 2.0), OTHER, {"corner_left": OTHER, "corner_right": MAIN_A})])
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first
    assert reg.stats["births_appearance_veto"] == 1


def test_on_obs_without_cam_embeddings_falls_back_to_legacy():
    reg = IdentityRegistry(app_floor=0.59, per_cam_appearance=True)
    first = _seed_then(reg, _obs(0, (2.3, 2.0), CORNER_A, None))
    active = reg.get_active_identities()
    assert len(active) == 1 and active[0].identity_id != first
    assert reg.stats["births_appearance_veto"] == 1


def test_on_birth_seeds_per_cam_stores():
    reg = IdentityRegistry(per_cam_appearance=True)
    reg.update_frame(frame_idx=8, observations=[
        _obs(0, (2.0, 2.0), MAIN_A, {"main": MAIN_A, "corner_left": CORNER_A})])
    ident = reg.get_active_identities()[0]
    assert set(ident.cam_appearance) == {"main", "corner_left"}
    np.testing.assert_allclose(ident.cam_anchor["corner_left"], CORNER_A)
    assert len(ident.cam_exemplars["main"]) == 1
    # Legacy fields still seeded from the merged embedding.
    np.testing.assert_allclose(ident.appearance_embedding, MAIN_A)


def test_on_match_updates_per_cam_stores_with_same_gating():
    reg = IdentityRegistry(per_cam_appearance=True, ema_alpha=0.3,
                           app_update_floor=0.6)
    reg.update_frame(frame_idx=8, observations=[_main_obs()])
    ident = reg.get_active_identities()[0]
    # A camera seen for the first time on a match seeds that camera's store.
    reg.update_frame(frame_idx=16, observations=[_corner_obs()])
    assert ident.active_slot == 0
    np.testing.assert_allclose(ident.cam_anchor["corner_left"], CORNER_A)
    # Low confidence (0.5 < app_update_floor): no EMA / exemplar update.
    near_main = _e(0.95, 0.05, 0.3, 0)
    reg.update_frame(frame_idx=24, observations=[_main_obs(emb=near_main)])
    assert len(ident.cam_exemplars["main"]) == 1
    np.testing.assert_allclose(ident.cam_appearance["main"], MAIN_A)
    # Mature: EMA + exemplar FIFO update; anchor fixed.
    ident.confidence = 0.9
    reg.update_frame(frame_idx=32, observations=[_main_obs(emb=near_main)])
    assert len(ident.cam_exemplars["main"]) == 2
    expected = 0.7 * MAIN_A + 0.3 * near_main
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(ident.cam_appearance["main"], expected, rtol=1e-5)
    np.testing.assert_allclose(ident.cam_anchor["main"], MAIN_A)
    np.testing.assert_allclose(ident.cam_appearance["corner_left"], CORNER_A)
    # Low relink margin blocks the update (same gate as the legacy EMA).
    reg.update_frame(frame_idx=40, observations=[
        _obs(0, (2.0, 2.0), near_main, {"main": near_main}, relink_margin=0.0)])
    assert len(ident.cam_exemplars["main"]) == 2


def test_on_exemplar_fifo_capped_at_five_per_camera():
    reg = IdentityRegistry(per_cam_appearance=True)
    reg.update_frame(frame_idx=8, observations=[_main_obs()])
    ident = reg.get_active_identities()[0]
    ident.confidence = 0.9
    for k in range(8):
        reg.update_frame(frame_idx=16 + 8 * k, observations=[_main_obs()])
    assert len(ident.cam_exemplars["main"]) == 5


def test_ema_similarity_same_camera_rule():
    reg = IdentityRegistry(per_cam_appearance=True)
    reg.update_frame(frame_idx=8, observations=[_main_obs()])
    ident = reg.get_active_identities()[0]
    corner = _corner_obs()
    # Shared camera absent -> None (not divergent) when per-cam.
    assert appearance_ema_similarity(ident, corner, per_cam=True) is None
    # Legacy compares the merged embedding.
    assert abs(appearance_ema_similarity(ident, corner, per_cam=False) - 0.4) < 1e-5
    main = _main_obs(emb=OTHER)
    assert abs(appearance_ema_similarity(ident, main, per_cam=True)) < 1e-6
    assert appearance_ema_similarity(ident, None, per_cam=True) is None


# --------------------------------------------------------------------- fusion

def _det(pos, emb, tid=1, box=(0, 0, 10, 10)):
    return FieldDetection(pos=pos, color="blue", box=box, track_id=tid, embedding=emb)


def test_fusion_cluster_carries_per_camera_embeddings():
    per_cam = {"main": [(_det((3.0, 3.0), MAIN_A), b"c1")],
               "corner_left": [(_det((3.2, 3.1), CORNER_A, tid=2), b"c2")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 3.0})
    assert len(fused) == 1
    ce = fused[0].detection.cam_embeddings
    assert set(ce) == {"main", "corner_left"}
    np.testing.assert_allclose(ce["main"], MAIN_A)
    np.testing.assert_allclose(ce["corner_left"], CORNER_A)
    # The merged embedding is unchanged (weighted average).
    expected = 1.0 * MAIN_A + 3.0 * CORNER_A
    np.testing.assert_allclose(fused[0].detection.embedding,
                               expected / np.linalg.norm(expected), rtol=1e-5)


def test_fusion_same_camera_duplicates_keep_highest_weight_and_normalise():
    low = FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 10, 10),
                         track_id=1, embedding=OTHER, score=0.5)
    high = FieldDetection(pos=(3.1, 3.0), color="blue", box=(0, 0, 10, 10),
                          track_id=2, embedding=MAIN_A * 2.0, score=1.0)
    fused = fuse_detections({"main": [(low, b"a"), (high, b"b")]},
                            weights={"main": 1.0})
    assert len(fused) == 1
    np.testing.assert_allclose(fused[0].detection.cam_embeddings["main"], MAIN_A)


def test_fusion_no_embeddings_yields_none():
    per_cam = {"main": [(_det((3.0, 3.0), None), b"c1")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0})
    assert fused[0].detection.cam_embeddings is None


def test_merged_group_unions_per_cam_embeddings_ref_wins_ties():
    a = FusedObservation(
        detection=FieldDetection(pos=(1, 1), color="blue", box=(0, 0, 5, 5),
                                 embedding=MAIN_A,
                                 cam_embeddings={"main": MAIN_A}),
        evidence=[], members=[("main", 1)], track_id=7)
    b = FusedObservation(
        detection=FieldDetection(pos=(1.5, 1), color="blue", box=None,
                                 embedding=CORNER_A,
                                 cam_embeddings={"main": OTHER,
                                                 "corner_left": CORNER_A}),
        evidence=[], members=[("corner_left", 4)], track_id=7)
    merged, _ = merge_same_id_observations([b, a])
    assert len(merged) == 1
    ce = merged[0].detection.cam_embeddings
    assert set(ce) == {"main", "corner_left"}
    np.testing.assert_allclose(ce["main"], MAIN_A)       # ref (main-backed) wins
    np.testing.assert_allclose(ce["corner_left"], CORNER_A)

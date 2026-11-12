"""FusedTrackRegistry: outcome counters (always on) and the per-camera
appearance relink knob (fused_per_cam_relink, default OFF)."""
import numpy as np
import pytest

from web.embedder import cosine_similarity
from web.field.fused_track_id import FusedTrackRegistry
from web.field.fusion import FusedObservation
from web.field.tracker import FieldDetection


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _avg(*embs):
    m = np.sum(embs, axis=0)
    return m / np.linalg.norm(m)


def _obs(members, embedding=None, pos=(0.0, 0.0), cam_embeddings=None):
    det = FieldDetection(pos=pos, color="unknown", box=(0, 0, 1, 1),
                         embedding=embedding, cam_embeddings=cam_embeddings)
    return FusedObservation(detection=det, evidence=[], members=list(members))


def _cam_obs(members, cam_embeddings, pos=(0.0, 0.0)):
    """Fusion-shaped observation: merged embedding = mean of the views."""
    return _obs(members, embedding=_avg(*cam_embeddings.values()), pos=pos,
                cam_embeddings=cam_embeddings)


# ---------------------------------------------------------------- counters

def test_counters_continued_merged_relinked_created():
    reg = FusedTrackRegistry(app_floor=0.55)
    a, b = _e(1, 0, 0), _e(0, 1, 0)
    reg.assign([_obs([("main", 1)], a), _obs([("corner_left", 2)], b),
                _obs([("main", 9)])])                                   # 2 emb + 1 no-emb
    reg.assign([_obs([("main", 1)], a)])                                # continued
    reg.assign([_obs([("main", 1), ("corner_left", 2)], a)])            # merged
    reg.assign([_obs([("main", 7)], a)])                                # relinked onto 0
    s = reg.stats
    assert s["created_with_emb"] == 2
    assert s["created_no_emb"] == 1
    assert s["continued"] == 1
    assert s["merged"] == 1
    assert s["relinked"] == 1


def test_counter_crossview_only_when_inactive_id_shares_no_camera():
    reg = FusedTrackRegistry(app_floor=0.55)
    c = _e(0, 1, 0)
    reg.assign([_cam_obs([("corner_left", 1)], {"corner_left": c})])
    # A main-only cluster whose appearance does not relink (cos 0) -> created;
    # the only inactive id was built from corner_left alone -> cross-view only.
    reg.assign([_cam_obs([("main", 5)], {"main": _e(1, 0, 0)})])
    assert reg.stats["created_with_emb"] == 2
    assert reg.stats["created_crossview_only"] == 1
    assert reg.stats["created_samecam_below_floor"] == 0


def test_counter_crossview_respects_spatial_gate():
    reg = FusedTrackRegistry(app_floor=0.55, relink_max_dist_m=3.0)
    reg.assign([_cam_obs([("corner_left", 1)], {"corner_left": _e(0, 1, 0)},
                         pos=(0.0, 0.0))])
    # Far away (10 m): the corner-only id is no candidate at all.
    reg.assign([_cam_obs([("main", 5)], {"main": _e(1, 0, 0)}, pos=(10.0, 0.0))])
    assert reg.stats["created_crossview_only"] == 0


def test_counter_samecam_below_floor():
    reg = FusedTrackRegistry(app_floor=0.55)
    reg.assign([_cam_obs([("main", 1)], {"main": _e(1, 0, 0)})])
    reg.assign([_cam_obs([("main", 5)], {"main": _e(0, 1, 0)})])
    assert reg.stats["created_with_emb"] == 2
    assert reg.stats["created_samecam_below_floor"] == 1
    assert reg.stats["created_crossview_only"] == 0


def test_counters_need_no_cam_embeddings():
    # Legacy observations (no cam_embeddings): counted, never classified.
    reg = FusedTrackRegistry(app_floor=0.55)
    reg.assign([_obs([("main", 1)], _e(1, 0, 0))])
    reg.assign([_obs([("main", 5)], _e(0, 1, 0))])
    assert reg.stats["created_with_emb"] == 2
    assert reg.stats["created_crossview_only"] == 0
    assert reg.stats["created_samecam_below_floor"] == 0


# ---------------------------------------------------------------- knob OFF

def _random_sequence(**kw):
    """60 frames of 1-4 robots seen by 1-2 of 3 cameras, member ids that roll
    over every 15 frames (+ random fresh ids), corner views biased away from
    main, ~15% observations without embeddings."""
    rng = np.random.default_rng(7)
    protos = [rng.normal(size=8) for _ in range(6)]
    cams = ["main", "corner_left", "corner_right"]
    reg = FusedTrackRegistry(app_floor=0.55, relink_max_dist_m=4.0, **kw)
    out = []
    for f in range(60):
        obs = []
        for r in rng.choice(6, size=rng.integers(1, 5), replace=False):
            cs = list(rng.choice(cams, size=rng.integers(1, 3), replace=False))
            members = [(c, int(r * 10 + (f // 15) + (rng.random() < 0.2) * 100)) for c in cs]
            ce = {}
            for c in cs:
                v = (protos[r] + 0.6 * rng.normal(size=8)
                     + (2.0 if c != "main" else 0) * np.eye(8)[cams.index(c)])
                ce[c] = v / np.linalg.norm(v)
            emb = None
            if rng.random() > 0.15:
                m = sum(ce.values())
                emb = m / np.linalg.norm(m)
            else:
                ce = None
            det = FieldDetection(pos=(float(r * 2 + rng.normal()), float(rng.normal())),
                                 color="unknown", box=(0, 0, 1, 1), embedding=emb,
                                 cam_embeddings=ce)
            obs.append(FusedObservation(detection=det, evidence=[], members=members))
        reg.assign(obs)
        out.append([o.track_id for o in obs])
    return out, reg


# Recorded from FusedTrackRegistry at main 29fe1dbb (before counters / knob).
_FROZEN_IDS = [
    [0, 1, 2], [0], [3], [0, 1, 4, 2], [0], [3, 1, 4], [2], [5], [2, 6, 0, 4],
    [2, 0, 5, 3], [7, 1, 0, 5], [2, 0, 6, 4], [2, 5, 3, 8], [2, 1], [4, 0, 2, 1],
    [6, 1, 4], [1, 2, 9, 0], [6, 4], [2, 1, 9, 6], [0, 5], [6, 1], [0, 2],
    [0, 6, 9, 2], [2], [1], [10, 6], [4], [1, 5, 0, 2], [2, 0], [5, 2, 6], [6],
    [5, 6, 0, 1], [5, 0, 2, 1], [5, 4, 1], [0, 5, 6], [2, 1, 6, 5], [2], [1, 0, 6],
    [0, 3, 4, 6], [6, 4, 1], [0, 3], [4], [0, 3, 0], [3], [4], [11, 12, 3, 13],
    [4, 2, 3], [0, 13, 4], [0, 3, 2], [2, 3], [4, 3], [13, 2, 4, 3], [2, 4, 13, 6],
    [13], [4, 13, 3], [3, 13], [13, 3, 2, 4], [3, 6, 13], [6, 12, 2, 4], [4, 3],
]


@pytest.mark.parametrize("kw", [{}, {"per_cam_relink": False}])
def test_knob_off_matches_pre_change_ids_and_embeddings(kw):
    ids, reg = _random_sequence(**kw)
    assert ids == _FROZEN_IDS
    assert sorted(reg._id_to_embedding) == [0, 2, 3, 4, 6, 10, 12, 13]
    total = sum(float(np.asarray(reg.get_embedding(i)).sum())
                for i in sorted(reg._id_to_embedding))
    assert total == pytest.approx(-3.3786083443076884, abs=1e-6)
    s = reg.stats
    assert (s["continued"] + s["merged"] + s["relinked"] + s["created_with_emb"]
            + s["created_no_emb"]) == sum(len(f) for f in ids)


def test_knob_on_changes_the_random_sequence_but_keeps_counting():
    ids, reg = _random_sequence(per_cam_relink=True)
    assert ids != _FROZEN_IDS
    s = reg.stats
    assert (s["continued"] + s["merged"] + s["relinked"] + s["created_with_emb"]
            + s["created_no_emb"]) == sum(len(f) for f in ids)


# ---------------------------------------------------------------- knob ON

A = _e(1, 0, 0, 0)                         # main view of robot X
A2 = _e(0.9, np.sqrt(1 - 0.81), 0, 0)      # later main view, cos(A, A2) = 0.9
C = _e(-0.3, 0, 1, 0)                      # corner view of X: looks nothing like main


def test_knob_on_relinks_on_same_camera_where_average_fails():
    merged = _avg(A2, C)
    assert cosine_similarity(A2, A) == pytest.approx(0.9, abs=1e-5)
    assert cosine_similarity(merged, A) < 0.55        # averaged view is below floor
    for on, expect in ((False, 1), (True, 0)):
        reg = FusedTrackRegistry(app_floor=0.55, per_cam_relink=on)
        reg.assign([_cam_obs([("main", 1)], {"main": A})])
        o = _cam_obs([("main", 5), ("corner_left", 2)], {"main": A2, "corner_left": C})
        reg.assign([o])
        assert o.track_id == expect, f"per_cam_relink={on}"
        if on:
            assert reg.stats["relinked"] == 1
            assert o.relink_evidence.type == "appearance_relinked"
        else:
            assert reg.stats["created_samecam_below_floor"] == 0
            assert reg.stats["created_samecam_above_floor"] == 1


def test_knob_on_no_shared_camera_is_no_relink_candidate():
    # Identical merged embedding (cos 1.0) but the inactive id was only ever
    # seen by main and the new cluster only by a corner: no appearance
    # evidence -> new id under ON; the legacy path relinks.
    for on, expect in ((False, 0), (True, 1)):
        reg = FusedTrackRegistry(app_floor=0.55, per_cam_relink=on)
        reg.assign([_cam_obs([("main", 1)], {"main": A})])
        o = _cam_obs([("corner_left", 2)], {"corner_left": A})
        reg.assign([o])
        assert o.track_id == expect, f"per_cam_relink={on}"
    assert reg.stats["created_crossview_only"] == 1


def test_knob_on_falls_back_to_legacy_without_cam_embeddings():
    reg = FusedTrackRegistry(app_floor=0.55, per_cam_relink=True)
    reg.assign([_obs([("main", 1)], A)])
    o = _obs([("corner_left", 2)], A)
    reg.assign([o])
    assert o.track_id == 0
    # Observation with views, id without any: legacy too.
    reg2 = FusedTrackRegistry(app_floor=0.55, per_cam_relink=True)
    reg2.assign([_obs([("main", 1)], A)])
    o2 = _cam_obs([("corner_left", 2)], {"corner_left": A})
    reg2.assign([o2])
    assert o2.track_id == 0


def test_merge_carries_per_camera_stores():
    reg = FusedTrackRegistry(app_floor=0.55, per_cam_relink=True)
    reg.assign([_cam_obs([("main", 1)], {"main": A}),
                _cam_obs([("corner_left", 2)], {"corner_left": C})])
    # Both ids continue together -> collapse to 0, which inherits corner_left.
    reg.assign([_cam_obs([("main", 1), ("corner_left", 2)], {"main": A, "corner_left": C})])
    assert reg.stats["merged"] == 1
    assert set(reg._cam_appearance[0]) == {"main", "corner_left"}
    assert 1 not in reg._cam_appearance
    assert set(reg._cam_anchor[0]) == {"main", "corner_left"}
    assert len(reg._cam_exemplars[0]["corner_left"]) >= 1
    # Lost, then re-seen by the corner alone with a fresh SAM id: relinks on
    # the corner view the merged id inherited.
    o = _cam_obs([("corner_left", 8)], {"corner_left": C})
    reg.assign([o])
    assert o.track_id == 0


def test_relink_quarantine_skips_per_camera_update():
    # Two inactive ids that look alike from main: the relink margin is ~0,
    # so neither the legacy nor the per-camera EMA moves.
    reg = FusedTrackRegistry(app_floor=0.55, per_cam_relink=True)
    reg.assign([_cam_obs([("main", 1)], {"main": A}),
                _cam_obs([("main", 2)], {"main": A})])
    before = {i: reg._cam_appearance[i]["main"].copy() for i in (0, 1)}
    o = _cam_obs([("main", 3)], {"main": A2})
    reg.assign([o])
    assert o.track_id in (0, 1) and o.relink_margin < 0.05
    for i in (0, 1):
        np.testing.assert_allclose(reg._cam_appearance[i]["main"], before[i])

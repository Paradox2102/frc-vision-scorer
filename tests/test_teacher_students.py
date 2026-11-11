"""Unit tests for the S2/S3 student heads and predict helper (synthetic input; no gold data)."""
import numpy as np
import pytest

from scripts.teacher.students import features as F
from scripts.teacher.students.heads import (
    Head, S2_CLASSES, s2_label, s3_label, thr_for_recall, auc, size_feats_from_wh,
)
from scripts.teacher.students.predict import Students, pad_crop


class FakeBackbone:
    """Deterministic stand-in for DINOv3: mean colour per channel, tiled to 384 dims."""

    def embed_crops(self, crops):
        out = []
        for c in crops:
            x = F.preprocess(c)
            out.append(np.tile(x.mean((1, 2)), F.FEAT_DIM // 3))
        return np.asarray(out, np.float32)


def _ckpts(use_size=False):
    rng = np.random.default_rng(0)
    d = F.FEAT_DIM + (3 if use_size else 0)
    X = rng.normal(size=(60, d)).astype(np.float32)
    feat = {"model": F.MODEL_NAME, "mode": F.DEFAULT_MODE, "size": F.DEFAULT_SIZE, "cls": "x"}
    h2 = Head(kind="mlp", n_out=3, steps=20).fit(X, np.arange(60) % 3)
    h3 = Head(kind="lr", n_out=2, steps=20).fit(X, np.arange(60) % 2)
    return ({"head": h2.to_ckpt(), "classes": list(S2_CLASSES), "use_size": use_size,
             "threshold_keep95": 0.3, "feature": feat},
            {"head": h3.to_ckpt(), "use_size": use_size, "threshold_recall995": 0.1, "feature": feat})


@pytest.mark.parametrize("use_size", [False, True])
def test_predict_shapes_and_probabilities(use_size):
    ck2, ck3 = _ckpts(use_size)
    st = Students(backbone=FakeBackbone(), s2_ckpt=ck2, s3_ckpt=ck3)
    rng = np.random.default_rng(1)
    crops = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for h, w in ((54, 81), (120, 90), (8, 30))]
    out = st.predict(crops)
    assert out["s2"].shape == (3, 3) and out["s3"].shape == (3,)
    assert np.allclose(out["s2"].sum(1), 1.0, atol=1e-5)
    assert ((out["s3"] >= 0) & (out["s3"] <= 1)).all()
    assert st.s2_threshold == 0.3 and st.s3_threshold == 0.1
    empty = st.predict([])
    assert empty["s2"].shape == (0, 3) and empty["s3"].shape == (0,)


def test_checkpoint_roundtrip_with_weights_only(tmp_path):
    import torch
    ck2, _ = _ckpts()
    torch.save(ck2, tmp_path / "s2.pt")
    back = torch.load(tmp_path / "s2.pt", map_location="cpu", weights_only=True)
    X = np.random.default_rng(2).normal(size=(5, F.FEAT_DIM)).astype(np.float32)
    a = Head.from_ckpt(ck2["head"]).predict_proba(X)
    b = Head.from_ckpt(back["head"]).predict_proba(X)
    assert np.allclose(a, b)


def test_head_learns_separable_classes():
    rng = np.random.default_rng(3)
    X = np.concatenate([rng.normal(-2, 1, (50, 8)), rng.normal(2, 1, (50, 8))]).astype(np.float32)
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    p = Head(kind="lr", n_out=2, steps=200).fit(X, y).predict_proba(X)
    assert (p.argmax(1) == y).mean() > 0.95


def test_preprocess_modes_keep_whole_crop():
    crop = np.zeros((40, 120, 3), np.uint8)
    crop[:, :10] = 255                                  # white stripe at the left edge
    lb = F.preprocess(crop, "letterbox")
    assert lb.shape == (3, F.DEFAULT_SIZE, F.DEFAULT_SIZE)
    assert lb[:, :, :5].max() > 2.0                     # letterbox keeps the edge stripe
    ce = F.preprocess(crop, "center")
    assert ce.shape == (3, 256, 256) and ce[:, :, :5].max() < 0     # centre crop cuts it
    assert F.preprocess(None).shape == (3, F.DEFAULT_SIZE, F.DEFAULT_SIZE)


def test_pad_crop_matches_harvest_padding():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert pad_crop(frame, (100, 100, 200, 140), "bumper").shape[:2] == (60, 150)
    assert pad_crop(frame, (0, 0, 100, 100), "robot").shape[:2] == (110, 110)   # clamped at 0


def test_gold_label_rules():
    assert s2_label({"kind": "bumper", "legible": "full", "team": "unsure"}) == "legible"
    assert s2_label({"kind": "robot", "legible": "partial", "team": "254", "conf": 0.9}) == "legible"
    assert s2_label({"kind": "robot", "legible": "partial", "team": "254", "conf": 0.5}) is None
    assert s2_label({"kind": "bumper", "legible": "none"}) == "illegible"
    assert s2_label({"kind": "person", "legible": "none"}) == "not_bumper"
    assert s3_label({"kind": "bumper"}) == 1 and s3_label({"kind": "game_piece"}) == 0


def test_threshold_and_auc_helpers():
    s = np.linspace(0, 1, 1000)
    t = thr_for_recall(s, 0.995)
    assert (s >= t).mean() >= 0.995
    assert auc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert size_feats_from_wh([[100, 100]]).shape == (1, 3)

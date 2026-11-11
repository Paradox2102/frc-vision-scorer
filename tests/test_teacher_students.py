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


# ---------------------------------------------------------------- v2: b3 gold + equipment negatives

def test_equipment_geometry_helpers():
    from scripts.teacher.students.equipment import _ioa, _label_names
    assert _ioa((10, 10, 20, 20), (0, 0, 100, 100)) == 1.0
    assert abs(_ioa((0, 0, 10, 10), (5, 0, 100, 100)) - 0.5) < 1e-9
    names = _label_names([{"label": "hub"}, {"label": "hub"}, {"label": "outpost"}, {"label": "hub"}])
    assert names == ["hub", "hub2", "outpost", "hub3"]     # crop_negatives_from_boxes.py naming


def test_equipment_manifest_uses_reviewed_crops_only(tmp_path):
    import json
    from scripts.teacher.students.equipment import load_manifest
    rows = [{"path": f"crops/s/{i}.jpg", "split": "train"} for i in range(3)]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_manifest(tmp_path) == []                   # nothing unreviewed is used
    (tmp_path / "review.json").write_text(json.dumps({"keep": ["crops/s/1.jpg"], "drop": ["crops/s/0.jpg"]}))
    assert [r["path"] for r in load_manifest(tmp_path)] == ["crops/s/1.jpg"]
    assert len(load_manifest(tmp_path, reviewed_only=False)) == 3


def test_folds_by_source_groups_and_is_seeded():
    from scripts.teacher.students.heads import folds_by_source
    rows = [{"source": s} for s in ("a", "b", "c", "a", "d", "e", "f")]
    f0, f1 = folds_by_source(rows, k=5, seed=0), folds_by_source(rows, k=5, seed=0)
    assert f0 == f1 and set(f0) == {"a", "b", "c", "d", "e", "f"}
    assert set(f0.values()) <= set(range(5))


def test_kind_and_equipment_rejection_counts():
    from scripts.teacher.students.heads import equip_rejection, kind_rejection
    rows = [{"kind": "person"}, {"kind": "person"}, {"kind": "field"}, {"kind": "robot"}]
    s = np.array([0.01, 0.5, 0.02, 0.9])
    y = np.array([0, 0, 0, 1])
    assert kind_rejection(rows, s, y, 0.1) == {"field": "1/1", "person": "1/2"}
    er = equip_rejection([{"origin": "detector", "source": "x"}, {"origin": "region", "source": "x"}],
                         np.array([0.01, 0.5]), 0.1, keys=("origin",))
    assert er == {"all": "1/2", "origin=detector": "1/1", "origin=region": "0/1"}


def test_s3_threshold_rules():
    from scripts.teacher.students.heads import s3_thresholds
    use = [{"s3": 1, "batch": "b1"}, {"s3": 1, "batch": "b3"}, {"s3": 0, "batch": "b1"}]
    cvs = [{"use": use, "P": np.array([[0.5, 0.5], [0.9, 0.1], [0.99, 0.01]]), "Pw": np.array([0.8, 0.9])},
           {"use": use, "P": np.array([[0.6, 0.4], [0.8, 0.2], [0.99, 0.01]]), "Pw": np.array([])}]
    th = s3_thresholds(cvs)
    assert abs(th["t_min_b1b3"] - 0.1 * 0.999) < 1e-9      # hard b3 positive sets the min rule
    assert abs(th["t_min_b1"] - 0.4 * 0.999) < 1e-9
    assert th["t_q995"] <= 0.5


def test_train_variant_gold_filter_and_equipment_negatives():
    from scripts.teacher.students import heads as Hm
    rng = np.random.default_rng(4)

    class Store:
        def __init__(self):
            self.index, self.X = {}, []

        def add(self, path, x):
            self.index[path] = len(self.X)
            self.X.append(x)

        def get(self, paths):
            return np.asarray([self.X[self.index[p]] for p in paths], np.float32)

    st = Store()
    gold = []
    for i in range(40):
        y = i % 2
        p = f"g{i}"
        st.add(p, rng.normal(2 * y - 1, 0.3, 8))
        gold.append({"path": p, "s3": y, "s2": None, "batch": "b3" if i >= 30 else "b1"})
    eq = []
    for i in range(10):
        p = f"e{i}"
        st.add(p, rng.normal(4, 0.3, 8))     # equipment cluster far from both gold classes
        eq.append({"path": p, "y": 0})
    seen = {}
    orig_fit = Hm.Head.fit

    def spy(self, X, y, w=None):
        seen["n"], seen["neg"] = len(y), int((y == 0).sum())
        return orig_fit(self, X, y, w)

    Hm.Head.fit = spy
    try:
        v = Hm.Variant("t", "s3", "lr", 1e-3, gold=("b1",))
        Hm.train_variant(v, st, gold, [], equip_train=eq)
        assert seen["n"] == 30                               # b3 rows and equipment both excluded
        v = Hm.Variant("t", "s3", "lr", 1e-3, gold=("b1", "b3"), equip=True)
        Hm.train_variant(v, st, gold, [], equip_train=eq)
        assert seen["n"] == 50 and seen["neg"] == 30         # 20 gold negatives + 10 equipment
    finally:
        Hm.Head.fit = orig_fit

"""Fast CPU-only tests for the S4 same-robot embedder (scripts/teacher/students/embedder.py)."""
import json
from collections import Counter

import numpy as np
import pytest

from scripts.teacher.students import embedder as S4


def _rec(event, mk, team, det="robot", trk="r0", alliance="red", t=0.0, i=0):
    roster = {"red": ["1", "2", "3"], "blue": ["4", "5", "6"]}
    return {"path": f"{event}/{mk}/crops/{trk}_{i}.jpg", "event": event, "match_key": mk,
            "detector": det, "team": team, "t_sec": t, "tracklet_id": trk, "roster": roster,
            "alliance": S4.team_alliance(roster, team) or alliance}


def _toy_records():
    recs = []
    i = 0
    for m in ("2026xx_qm1", "2026xx_qm2", "2026xx_qm3"):
        for team in ("1", "2", "3", "4", "5", "6"):
            for det in ("robot", "bumper"):
                for c in range(3):
                    recs.append(_rec("2026xx", m, team, det, trk=f"{det[0]}{team}", i=i))
                    i += 1
    return recs


def test_record_split_holds_out_test_gt_and_val_events():
    assert S4.record_split("2026alhu_qm1") == "test"
    assert S4.record_split("2025casd_qm6") == "test"
    assert S4.record_split("2026cur_qm8") == "excluded"
    assert S4.record_split(f"{S4.VAL_EVENTS[0]}_qm1") == "val"
    assert S4.record_split("2026hop_qm1") == "train"


def test_identity_key_is_event_and_team():
    a = _rec("2025cur", "2025cur_qm1", "254")
    b = _rec("2026cur", "2026cur_qm22", "254")
    assert S4.ident(a) != S4.ident(b)


def test_load_match_records_propagates_and_drops_off_alliance(tmp_path):
    md = tmp_path / "2026xx" / "2026xx_qm1"
    md.mkdir(parents=True)
    roster = {"red": ["1", "2", "3"], "blue": ["4", "5", "6"]}
    rows = [
        {"event": "2026xx", "match_key": "2026xx_qm1", "detector": "bumper", "tracklet_id": "b0",
         "t_sec": 1.0, "path": "2026xx/2026xx_qm1/crops/b0_1.jpg", "roster": roster},
        {"event": "2026xx", "match_key": "2026xx_qm1", "detector": "bumper", "tracklet_id": "b0",
         "t_sec": 2.0, "path": "2026xx/2026xx_qm1/crops/b0_2.jpg", "roster": roster},
        {"event": "2026xx", "match_key": "2026xx_qm1", "detector": "bumper", "tracklet_id": "b1",
         "t_sec": 2.0, "path": "2026xx/2026xx_qm1/crops/b1_2.jpg", "roster": roster},
        {"event": "2026xx", "match_key": "2026xx_qm1", "detector": "robot", "tracklet_id": "r0",
         "t_sec": 1.0, "path": "2026xx/2026xx_qm1/crops/r0_1.jpg", "roster": roster},
        {"event": "2026xx", "match_key": "2026xx_qm1", "detector": "robot", "tracklet_id": "r0",
         "t_sec": 3.0, "path": "2026xx/2026xx_qm1/crops/r0_3.jpg", "roster": roster},
    ]
    (md / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (md / "labels_jev.jsonl").write_text("\n".join(json.dumps(d) for d in [
        {"tracklet_id": "b0", "decision": "team", "team": "2", "off_alliance": False},
        {"tracklet_id": "b1", "decision": "team", "team": "5", "off_alliance": True},
    ]))
    (md / "labels_robot.jsonl").write_text("\n".join(json.dumps(d) for d in [
        {"path": rows[3]["path"], "team": "2", "off_alliance": False},
        {"path": rows[4]["path"], "team": None, "off_alliance": False},
    ]))
    recs = S4.load_match_records(md)
    assert [(r["detector"], r["team"]) for r in recs] == [("bumper", "2"), ("bumper", "2"), ("robot", "2")]
    assert all(r["alliance"] == "red" for r in recs)


def test_sampler_batches_are_p_by_k_with_cooccurring_identities():
    recs = _toy_records()
    robots = [r for r in recs if r["detector"] == "robot"]
    s = S4.IdentityBatchSampler(robots, p=6, k=4, steps=5, seed=1)
    for b in s:
        ids = Counter(S4.ident(robots[i]) for i in b)
        assert len(ids) == 6 and set(ids.values()) == {4}
        # K crops of an identity spread over its matches (3 matches available)
        for idk in ids:
            ms = {robots[i]["match_key"] for i in b if S4.ident(robots[i]) == idk}
            assert len(ms) == 3


def test_sampler_bumper_fraction():
    recs = _toy_records()
    s = S4.IdentityBatchSampler(recs, p=6, k=4, steps=200, bumper_frac=0.25, seed=0)
    dets = Counter(recs[i]["detector"] for b in s for i in b)
    frac = dets["bumper"] / sum(dets.values())
    assert 0.18 < frac < 0.32
    s0 = S4.IdentityBatchSampler(recs, p=6, k=4, steps=20, bumper_frac=0.0, seed=0)
    assert all(recs[i]["detector"] == "robot" for b in s0 for i in b)


def test_supcon_loss_prefers_clustered_embeddings():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    good = torch.nn.functional.normalize(torch.eye(3)[labels] + 0.01 * torch.randn(6, 3), dim=1)
    bad = torch.nn.functional.normalize(torch.randn(6, 3), dim=1)
    assert float(S4.supcon_loss(good, labels)) < float(S4.supcon_loss(bad, labels))
    # no positives at all -> zero loss, finite gradient
    z = torch.nn.functional.normalize(torch.randn(3, 4), dim=1).requires_grad_()
    S4.supcon_loss(z, torch.tensor([0, 1, 2])).backward()
    assert torch.isfinite(z.grad).all()


def test_auc_fast():
    y = np.array([1, 1, 0, 0])
    assert S4.auc_fast(np.array([0.9, 0.8, 0.1, 0.2]), y) == 1.0
    assert S4.auc_fast(np.array([0.1, 0.2, 0.9, 0.8]), y) == 0.0
    assert S4.auc_fast(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    assert np.isnan(S4.auc_fast(np.array([0.1, 0.2]), np.array([1, 1])))


def test_metrics_perfect_and_random_embeddings():
    recs = [r for r in _toy_records() if r["detector"] == "robot"]
    team_vec = {t: np.eye(8)[int(t)] for t in "123456"}
    E = np.stack([team_vec[r["team"]] for r in recs]).astype(np.float32)
    m = S4.evaluate(E, recs, per_event=False)["all"]
    assert m["within_match_auc"] == 1.0 and m["cross_match_auc"] == 1.0
    assert m["lineup_crop6_top1"] == 1.0 and m["lineup_trk3_top1"] == 1.0
    assert m["lineup_mean_candidates6"] == 6
    rng = np.random.default_rng(0)
    m2 = S4.evaluate(rng.normal(size=(len(recs), 16)), recs, per_event=False)["all"]
    assert 0.3 < m2["cross_match_auc"] < 0.7


def test_lineup_gallery_excludes_own_match():
    # team "1" appears only in qm1: no gallery from other matches -> not scored
    recs = [_rec("2026xx", "2026xx_qm1", "1", i=0), _rec("2026xx", "2026xx_qm1", "2", i=1),
            _rec("2026xx", "2026xx_qm2", "2", i=2)]
    E = np.eye(3, dtype=np.float32)
    m = S4.lineup_top1(E, recs)
    assert m["lineup_crop6_n"] == 2  # only the two team-"2" crops have a gallery


def test_augment_keeps_valid_image():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)
    for _ in range(20):
        out = S4.augment(img, rng)
        assert out.dtype == np.uint8 and out.ndim == 3 and min(out.shape[:2]) >= 4


def test_net_forward_cpu_random_init():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    net = S4.build_net(pretrained=False, train_blocks=2, dim=16)
    trainable = {n.split(".")[1] for n, p in net.backbone.named_parameters()
                 if p.requires_grad and n.startswith("blocks.")}
    assert trainable == {"10", "11"}
    z = net(torch.randn(2, 3, 64, 64))
    assert z.shape == (2, 16)
    assert torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-4)

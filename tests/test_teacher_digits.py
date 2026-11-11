"""Unit tests for the S1 digit reader (tiny randomly initialised PARSeq on CPU; no data, no weights)."""
import numpy as np
import pytest
import torch

from scripts.teacher.students import digits as D
from scripts.teacher.students import synth
from scripts.teacher.students.parseq_plm import plm_loss
from web.students.parseq import PARSeq, DigitTokenizer

TINY = dict(max_label_length=6, embed_dim=48, enc_num_heads=2, enc_mlp_ratio=2, enc_depth=1,
            dec_num_heads=2, dec_mlp_ratio=2, dec_depth=1, dropout=0.0)
SOFT = {"mode": "softmax", "T": 2.0, "floor": -1e9, "t_p": 0.0, "t_m": 0.0}
ROSTER = {"red": ["254", "1678", "118"], "blue": ["10254", "971", "6800"]}


def _ckpt(seed=0):
    torch.manual_seed(seed)
    m = PARSeq(**TINY)
    rules = {"plain": dict(SOFT, colour_eps=None), "colour": dict(SOFT, colour_eps=0.05),
             "colour_hard": dict(SOFT, colour_eps=0.0)}
    return {"config": TINY, "state_dict": {k: v.half() for k, v in m.state_dict().items()},
            "rules": rules, "colour_eps": 0.05, "use_colour": True}


def test_tokenizer_roundtrip():
    tok = DigitTokenizer()
    t = tok.encode(["254", "10254"])
    assert t.shape == (2, 7) and t[0, 0] == tok.bos_id and t[0, 4] == tok.eos_id and t[0, 5] == tok.pad_id
    assert tok.decode_ids(t[1, 1:].tolist()) == "10254"


def test_pretrained_slicing_keeps_digit_rows():
    full = PARSeq(**TINY).state_dict()
    n = 94
    sd = dict(full)
    sd["text_embed.embedding.weight"] = torch.randn(n + 3, 48)
    sd["head.weight"] = torch.randn(n + 1, 48)
    sd["head.bias"] = torch.randn(n + 1)
    sd["pos_queries"] = torch.randn(1, 26, 48)
    m = PARSeq.from_pretrained(sd, **{k: v for k, v in TINY.items() if k != "max_label_length"})
    # upstream charset starts with 0-9 -> rows 1..10; [E]=0; [B],[P] = 95, 96
    assert torch.equal(m.head.weight[1:11], sd["head.weight"][1:11])
    assert torch.equal(m.text_embed.embedding.weight[11], sd["text_embed.embedding.weight"][95])
    assert m.pos_queries.shape == (1, 7, 48)


def test_scores_are_sequence_log_probs_and_sum_below_one():
    m = PARSeq(**TINY).eval()
    X = torch.randn(2, 3, 32, 128)
    with torch.no_grad():
        mem = m.encode(X)
        lp = m.score_strings(mem, [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]] * 2)
        reads, lp_read = m.greedy(mem)
    assert lp.shape == (2, 10) and (lp <= 0).all()
    assert lp.exp().sum(1).max() <= 1.0 + 1e-5          # disjoint strings: total mass <= 1
    with torch.no_grad():                               # greedy string's own teacher-forced score
        again = m.score_strings(mem, [[r] for r in reads])[:, 0]
    assert torch.allclose(again, lp_read, atol=1e-4)


def test_plm_loss_backprops():
    m = PARSeq(**TINY)
    loss = plm_loss(m, torch.randn(4, 3, 32, 128), ["254", "10254", "7", "1678"],
                    rng=np.random.default_rng(0))
    loss.backward()
    assert torch.isfinite(loss) and m.head.weight.grad is not None


def test_decide_abstains_on_other_low_p_small_margin_and_floor():
    P = np.array([[0.9, 0.05, 0.05],     # confident team 0
                  [0.1, 0.1, 0.8],       # "other" on top -> abstain
                  [0.5, 0.45, 0.05],     # small margin
                  [0.4, 0.3, 0.3]])      # low top-1
    assert D.decide(P, 0.6, 0.2).tolist() == [0, -1, -1, -1]
    assert D.decide(P, 0.0, 0.0).tolist() == [0, -1, 0, 0]
    assert D.decide(P, 0.0, 0.0, np.array([-1.0, -1, -9, -1]), floor=-5).tolist() == [0, -1, -1, 0]


def test_posterior_modes():
    LP = np.log(np.array([[0.30, 0.02, 0.01], [0.9, 0.05, 0.01]]))
    pm = D.posterior(LP, np.zeros_like(LP), {"mode": "mass"})
    assert np.allclose(pm.sum(1), 1) and pm[0, -1] > pm[0, 0]          # most mass off-roster -> other
    ps = D.posterior(LP, np.zeros_like(LP), {"mode": "softmax", "T": 1.0})
    assert ps[0, -1] == 0 and ps[0, 0] > 0.9                           # nearest roster number wins
    hot = D.posterior(LP, np.zeros_like(LP), {"mode": "softmax", "T": 4.0})
    assert hot[0, 0] < ps[0, 0]                                        # temperature flattens


def test_tally_and_rule_tuning():
    P = np.array([[0.99, 0.0, 0.01], [0.6, 0.39, 0.01], [0.7, 0.29, 0.01]])
    teams = [["1", "2"]] * 3
    truth = ["1", "2", "1"]                              # row 1 is wrong unless t_p > 0.6
    r = D.tally(D.decide(P, 0.0, 0.0), teams, truth)
    assert (r["right"], r["wrong"], r["abstain"]) == (2, 1, 0)
    LP = np.log(P[:, :2] + 1e-9)
    rosters, colours = [["1", "2"]] * 3, [None] * 3
    tuned = D.tune_rule({"val": (LP, teams, rosters, colours, truth), "prop": None}, None, max_wrong=0.0)
    pred, _ = D.apply_rule(LP, teams, rosters, colours, tuned["rule"])
    t = D.tally(pred, teams, truth)
    assert t["wrong"] == 0 and t["right"] >= 1


def test_colour_prior_downweights_off_alliance():
    m = PARSeq(**TINY).eval()
    teams, LP, _ = D.score_batch(m, torch.randn(1, 3, 32, 128), [ROSTER])
    assert LP.shape == (1, 6) and (LP <= 0).all()
    lp0 = D.colour_logprior(teams, [ROSTER], ["red"], None)
    lp1 = D.colour_logprior(teams, [ROSTER], ["red"], 0.0)
    P0 = D.posterior(LP, lp0, SOFT)
    P1 = D.posterior(LP, lp1, SOFT)
    assert np.isclose(P1.sum(), 1.0) and np.allclose(P1[0, 3:6], 0.0) and P1[0, :3].sum() >= P0[0, :3].sum()


def test_reader_api_shapes(tmp_path):
    torch.save(_ckpt(), tmp_path / "s1.pt")
    rd = D.DigitReader(tmp_path / "s1.pt", device="cpu")
    rng = np.random.default_rng(1)
    crops = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for h, w in ((54, 81), (20, 90), (8, 30))]
    out = rd.read(crops, [ROSTER, ROSTER, ["1", "22"]], ["red", None, "blue"])
    assert len(out) == 3
    for o in out:
        assert set(o) == {"probs", "team", "p", "abstain", "read", "roster_mass"}
        assert np.isclose(sum(o["probs"].values()), 1.0)
        assert o["abstain"] == (o["team"] is None)
    assert set(out[2]["probs"]) == {"1", "22", "other"}
    assert rd.read([], [], []) == []


def test_students_read_digits_delegates(tmp_path):
    from scripts.teacher.students.predict import Students
    torch.save(_ckpt(), tmp_path / "s1_digits.pt")
    st = Students.__new__(Students)                     # skip S2/S3 + DINOv3 loading
    st.models_dir, st.device, st._s1 = tmp_path, "cpu", None
    out = st.read_digits([np.zeros((40, 100, 3), np.uint8)], [ROSTER], ["blue"])
    assert len(out) == 1 and "other" in out[0]["probs"]


def test_preprocess_shape_and_range():
    x = D.preprocess(np.full((50, 90, 3), 255, np.uint8))
    assert x.shape == (3, 32, 128) and np.allclose(x, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_render(seed):
    rng = np.random.default_rng(seed)
    img, label = synth.render(rng, np.zeros((60, 100, 3), np.uint8))
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[0] >= 12
    assert label.isdigit() and 1 <= len(label) <= 5 and label[0] != "0"
    comb = synth.interlace_comb(np.tile(np.arange(20, dtype=np.uint8), (4, 1))[..., None], rng, shift=1)
    assert (comb[0] == np.arange(20)[:, None]).all() and not (comb[1] == np.arange(20)[:, None]).all()

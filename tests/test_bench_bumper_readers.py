"""Pure-function tests for scripts/bench_bumper_readers.py (no network, no web.app)."""
import math

import pytest

from scripts.bench_bumper_readers import (
    average_distributions,
    blank_calibrate,
    build_options,
    calibrated_view,
    clef_image_tokens,
    decoy_teams,
    ece,
    hint_alliance,
    letter_distribution,
    load_env_file,
    permutation_orders,
    score_file_variants,
    score_predictions,
    select_rows,
    shuffle_options,
    target_option,
    top_choice,
    true_team,
    truncate_payload,
)

ROW = {"id": "m_f00010_r001", "red_teams": ["1", "2", "3"], "blue_teams": ["4", "5", "6"],
       "baseline_alliance_read": "red", "baseline_matched_team": "2",
       "label_status": "unlabelled", "label_team": None, "label_alliance": None,
       "stratum": "matched"}


def _ids(opts):
    return [o["id"] for o in opts]


# ---- option sets ------------------------------------------------------------

def test_hint_sources():
    assert hint_alliance(ROW, "baseline") == "red"
    assert hint_alliance(ROW, "none") == "unknown"
    assert hint_alliance({**ROW, "label_alliance": "blue"}, "label") == "blue"
    assert hint_alliance({**ROW, "baseline_alliance_read": "unknown"}) == "unknown"


def test_alliance_options_and_fallback():
    assert _ids(build_options(ROW, "alliance", "red")) == ["1", "2", "3", "none", "other"]
    assert _ids(build_options(ROW, "alliance", "unknown")) == \
        ["1", "2", "3", "4", "5", "6", "none", "other"]
    opts = build_options(ROW, "alliance", "blue")
    assert [o["alliance"] for o in opts if o["kind"] == "team"] == ["blue"] * 3


def test_decoy_options_and_decoys():
    assert _ids(build_options(ROW, "decoy", "red")) == ["1", "2", "3", "4", "5", "6", "none", "other"]
    assert decoy_teams(ROW, "red") == ["4", "5", "6"]
    assert decoy_teams({**ROW, "label_alliance": "blue"}, "red") == ["1", "2", "3"]  # label wins
    assert decoy_teams(ROW, "unknown") == []


def test_missing_true_removes_true_team():
    # Unlabelled: baseline matched team is removed.
    assert true_team(ROW) == "2"
    assert _ids(build_options(ROW, "missing-true", "red")) == ["1", "3", "none", "other"]
    # Labelled legible: the label wins over the baseline.
    lab = {**ROW, "label_status": "legible", "label_team": "3", "label_alliance": "red"}
    assert _ids(build_options(lab, "missing-true", "red")) == ["1", "2", "none", "other"]
    # Unknown hint: all six minus the true team.
    assert _ids(build_options(ROW, "missing-true", "unknown")) == \
        ["1", "3", "4", "5", "6", "none", "other"]
    # Neither label nor baseline team: skipped by select_rows, refused by build_options.
    bare = {**ROW, "baseline_matched_team": None}
    assert true_team(bare) is None
    with pytest.raises(ValueError):
        build_options(bare, "missing-true", "red")
    assert select_rows([ROW, {**bare, "id": "x"}], False, None, "missing-true") == [ROW]
    # The correct answer becomes an escape.
    assert target_option(lab, _ids(build_options(lab, "missing-true", "red"))) == "other"


def test_shuffle_is_deterministic_per_crop():
    opts = build_options(ROW, "decoy", "red")
    a = _ids(shuffle_options(opts, "crop-a"))
    assert a == _ids(shuffle_options(opts, "crop-a"))
    assert sorted(a) == sorted(_ids(opts))
    # Different crops get different orders (overwhelmingly likely for 8 options)
    assert any(_ids(shuffle_options(opts, f"crop-{i}")) != a for i in range(5))


# ---- permutations / distributions --------------------------------------------

def test_permutation_orders_are_cyclic_rotations():
    base = ["a", "b", "c", "d", "e"]
    perms = permutation_orders(base, 3)
    assert perms[0] == base and len(perms) == 3
    for p in perms:
        i = base.index(p[0])
        assert p == base[i:] + base[:i]
    assert len({p[0] for p in perms}) == 3  # distinct first slots
    assert len(permutation_orders(base, 99)) == 5  # capped at n
    assert permutation_orders(base, 1) == [base]


def test_average_distributions():
    avg = average_distributions([{"a": 0.8, "b": 0.2}, {"a": 0.4, "b": 0.6}, None])
    assert avg == pytest.approx({"a": 0.6, "b": 0.4})
    assert average_distributions([]) is None


def test_letter_distribution_renormalises_and_merges_variants():
    tl = [{"token": "B", "logprob": math.log(0.5)}, {"token": " B", "logprob": math.log(0.1)},
          {"token": "A", "logprob": math.log(0.2)}, {"token": "The", "logprob": math.log(0.15)},
          {"token": "Z", "logprob": math.log(0.05)}]
    dist, mass = letter_distribution(tl, "ABC")
    assert mass == pytest.approx(0.8)
    assert dist == pytest.approx({"A": 0.25, "B": 0.75, "C": 0.0})
    assert letter_distribution([{"token": "x", "logprob": 0.0}], "AB") == (None, 0.0)


def test_blank_calibrate():
    p = {"a": 0.5, "b": 0.3, "none": 0.2}
    pb = {"a": 0.25, "b": 0.25, "none": 0.5}
    cal = blank_calibrate(p, pb)
    raw = {"a": 2.0, "b": 1.2, "none": 0.4}
    s = sum(raw.values())
    assert cal == pytest.approx({k: v / s for k, v in raw.items()})
    # A uniform blank leaves p unchanged; a zero blank prob is clipped, not inf.
    assert blank_calibrate(p, {k: 1 / 3 for k in p}) == pytest.approx(p)
    clipped = blank_calibrate(p, {"a": 0.0, "b": 0.5, "none": 0.5}, eps=1e-3)
    assert all(math.isfinite(v) for v in clipped.values())
    assert sum(clipped.values()) == pytest.approx(1.0)


def test_top_choice():
    assert top_choice({"a": 0.2, "b": 0.7, "c": 0.1}) == ("b", 0.7)
    assert top_choice(None) == (None, 0.0)


# ---- scoring -----------------------------------------------------------------

def test_target_option():
    leg = {"label_status": "legible", "label_team": "5"}
    assert target_option(leg, ["4", "5", "none", "other"]) == "5"
    assert target_option(leg, ["1", "2", "none", "other"]) == "other"
    assert target_option(leg, None) == "5"
    assert target_option({"label_status": "illegible"}, ["1"]) == "none"
    assert target_option({"label_status": "unlabelled"}, ["1"]) is None


def test_ece_perfect_and_overconfident():
    assert ece([0.95, 0.95], [True, True]) == pytest.approx(0.05)
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)
    assert ece([], []) is None


def _manifest():
    rows = {
        "c1": {**ROW, "id": "c1", "label_status": "legible", "label_team": "1", "label_alliance": "red"},
        "c2": {**ROW, "id": "c2", "label_status": "legible", "label_team": "2", "label_alliance": "red"},
        "c3": {**ROW, "id": "c3", "label_status": "illegible"},
        "c4": {**ROW, "id": "c4", "label_status": "legible", "label_team": "3", "label_alliance": "red"},
    }
    return rows


def _choice_pred(cid, probs, cal=None):
    top, p = top_choice(probs)
    ids = list(probs)
    pred = {"id": cid, "arm": "vllm-choice", "options_mode": "decoy", "hint_alliance": "red",
            "option_ids": ids, "probs": probs, "top": top, "top_p": p,
            "pred_team": None if top in ("none", "other") else top,
            "perms": [{"option_ids": ids, "probs": probs, "probs_cal": cal}]}
    if cal is not None:
        pred["probs_cal"] = cal
    return pred


def test_score_predictions_probabilistic():
    man = _manifest()
    base = {"1": 0, "2": 0, "3": 0, "4": 0, "none": 0, "other": 0}
    preds = [
        _choice_pred("c1", {**base, "1": 0.9, "4": 0.1}),            # correct, conf .9
        _choice_pred("c2", {**base, "4": 0.6, "2": 0.4}),            # wrong (decoy), conf .6
        _choice_pred("c3", {**base, "none": 0.8, "1": 0.2}),         # true negative
        _choice_pred("c4", {**base, "none": 0.7, "3": 0.3}),         # abstain
    ]
    m = score_predictions(preds, man)
    assert (m["n_legible"], m["correct"], m["wrong"], m["abstain"]) == (3, 1, 1, 1)
    assert m["fp_rate"] == 0.0
    assert m["at"]["0.5"]["precision"] == 0.5 and m["at"]["0.5"]["coverage"] == pytest.approx(2 / 3)
    assert m["at"]["0.7"]["precision"] == 1.0
    assert m["decoy_mass"] == pytest.approx((0.1 + 0.6 + 0 + 0) / 4)
    assert m["assert_rate"] == 0.5
    # ECE over 4 rows: confs .9 ok, .6 wrong, .8 ok, .7 wrong
    assert m["ece"] == pytest.approx(0.25 * (0.1 + 0.6 + 0.2 + 0.7))


def test_score_blank_calibrated_variant():
    man = _manifest()
    base = {"1": 0, "2": 0, "3": 0, "4": 0, "none": 0, "other": 0}
    preds = [_choice_pred("c4", {**base, "none": 0.7, "3": 0.3},
                          cal={**base, "none": 0.2, "3": 0.8})]
    out = dict(score_file_variants(preds, man, "vllm-choice:decoy"))
    assert out["vllm-choice:decoy"]["abstain"] == 1
    assert out["vllm-choice:decoy+cal"]["correct"] == 1
    assert calibrated_view(preds[0])["top"] == "3"


def test_score_freeread_missing_true():
    man = _manifest()
    # True team 1 was removed from the roster: reading "1" and NOT matching is right;
    # forcing a listed team is wrong.
    preds = [
        {"id": "c1", "arm": "freeread", "options_mode": "missing-true", "pred_team": None,
         "roster_offered": ["2", "3", "4", "5", "6"], "freeread": {"raw_read": "1"}},
        {"id": "c2", "arm": "freeread", "options_mode": "missing-true", "pred_team": "3",
         "roster_offered": ["1", "3", "4", "5", "6"], "freeread": {"raw_read": "3"}},
    ]
    m = score_predictions(preds, man)
    assert (m["correct"], m["wrong"]) == (1, 1)
    assert m["assert_rate"] == 0.5
    assert m["ece"] is None  # not probabilistic


def test_slot_a_bias_metric():
    man = _manifest()
    p = _choice_pred("c3", {"none": 0.9, "1": 0.1})
    p["perms"] = [{"option_ids": ["none", "1"], "probs": {"none": 0.9, "1": 0.1}},
                  {"option_ids": ["1", "none"], "probs": {"none": 0.9, "1": 0.1}}]
    m = score_predictions([p], man)
    assert m["slot_a_top"] == 0.5 and m["slot_a_expected"] == 0.5


# ---- misc --------------------------------------------------------------------

def test_load_env_file_never_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nexport CLOUDFLARE_ACCOUNT_ID='acct'\nCLOUDFLARE_API_TOKEN=tok # note\n"
                   "OTHER=x\n")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "already")
    monkeypatch.delenv("OTHER", raising=False)
    loaded = load_env_file(env)
    import os
    assert loaded == ["CLOUDFLARE_ACCOUNT_ID"]
    assert os.environ["CLOUDFLARE_ACCOUNT_ID"] == "acct"
    assert os.environ["CLOUDFLARE_API_TOKEN"] == "already"
    assert "OTHER" not in os.environ
    assert load_env_file(tmp_path / "missing.env") == []


def test_clef_image_tokens():
    assert clef_image_tokens(256, 640) == 8 * 20 + 3
    assert clef_image_tokens(4000, 4000) == 1024


def test_truncate_payload():
    p = {"images": ["data:image/jpeg;base64," + "A" * 500], "state": "short"}
    t = truncate_payload(p, keep=30)
    assert t["state"] == "short"
    assert t["images"][0].startswith("data:image/jpeg;base64,") and t["images"][0].endswith("chars>")

"""The `jev` arm of scripts/bench_bumper_readers.py: image prep into the API's
280-1568 px window, payload shape, response parsing, the `+subj` gate, dry-run.
No network."""
import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import bench_bumper_readers as B

ROW = {"id": "m_f00010_r001", "file": "crops/m_f00010_r001.jpg", "width": 96, "height": 48,
       "red_teams": ["1", "2", "3"], "blue_teams": ["4", "5", "6"],
       "baseline_alliance_read": "red", "baseline_matched_team": "2",
       "label_status": "legible", "label_team": "2", "label_alliance": "red",
       "stratum": "matched"}


def _jpg(w, h, colour=(200, 30, 30), fmt="JPEG"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format=fmt, quality=95)
    return buf.getvalue()


def _size(jpg):
    with Image.open(io.BytesIO(jpg)) as im:
        return im.size


# ---- image prep ---------------------------------------------------------------

@pytest.mark.parametrize("w,h", [(60, 20), (940, 400), (2000, 150), (20, 60), (150, 2000),
                                 (280, 280), (3000, 3000), (279, 900)])
def test_jev_image_sides_in_range(w, h):
    out, ow, oh, padded = B.jev_image(_jpg(w, h))
    assert _size(out) == (ow, oh)
    assert 280 <= ow <= 1568 and 280 <= oh <= 1568
    assert 78_400 <= ow * oh <= 2_458_624


def test_jev_image_scaling_and_letterbox():
    # tiny: scaled up to min side 280, no pad needed
    _o, w, h, padded = B.jev_image(_jpg(60, 20))
    assert (w, h) == (840, 280) and not padded
    # normal in-range JPEG: passed through byte-identical
    raw = _jpg(940, 400)
    out, w, h, padded = B.jev_image(raw)
    assert out == raw and (w, h) == (940, 400) and not padded
    # very wide: long side shrinks to 1568, short side gray-padded to 280
    out, w, h, padded = B.jev_image(_jpg(2000, 150, (255, 0, 0)))
    assert (w, h) == (1568, 280) and padded
    with Image.open(io.BytesIO(out)) as im:
        top = im.getpixel((784, 2))
        mid = im.getpixel((784, 140))
    assert all(abs(c - 128) < 8 for c in top)
    assert mid[0] > 200 and mid[1] < 60


def test_jev_image_converts_png_to_jpeg():
    out, w, h, _ = B.jev_image(_jpg(400, 400, fmt="PNG"))
    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "JPEG"


# ---- payload shape --------------------------------------------------------------

def _payload(mode="alliance", hint="red"):
    opts = B.shuffle_options(B.jev_options(ROW, mode, hint), ROW["id"])
    img = _jpg(300, 300)
    return B.jev_payload(ROW, img, opts, mode, hint), img


def test_payload_shape():
    p, img = _payload()
    assert p["model"] == "jev-latest"
    ref = B.JEV_ASSET_REF
    assert p["state"]["crop"] == ref
    assert set(p["assets"]) == {ref}
    a = p["assets"][ref]
    assert a["type"] == "image" and a["format"] == "jpg"
    assert not a["data_base64"].startswith("data:")
    assert base64.b64decode(a["data_base64"]) == img
    assert "images" not in p
    # subject is asked BEFORE team
    assert list(p["questions"]) == ["subject", "team"]
    assert all(q["type"] == "choice" for q in p["questions"].values())
    assert set(p["questions"]["subject"]["criteria"]) == set(B.JEV_SUBJECTS)
    team = p["questions"]["team"]["criteria"]
    assert {"not_robot", "unreadable", "printed", "other"} <= set(team)
    assert "none" not in team
    assert {"1", "2", "3"} == set(team) - set(B.JEV_ESCAPES)  # red alliance hint
    assert team["other"].startswith("Legible bumper digits")
    assert "Teams on the red alliance in this match: 1, 2, 3." in p["state"]["context"]
    assert "`crop`" in p["questions"]["team"]["instructions"]


def test_payload_decoy_and_missing_true_rosters():
    p, _ = _payload("decoy", "red")
    assert set(p["questions"]["team"]["criteria"]) - set(B.JEV_ESCAPES) == set("123456")
    assert "Teams in this match: 1, 2, 3, 4, 5, 6." in p["state"]["context"]
    p, _ = _payload("missing-true", "red")
    assert set(p["questions"]["team"]["criteria"]) - set(B.JEV_ESCAPES) == {"1", "3"}


# ---- arm: parsing, scoring, dry run ----------------------------------------------

def _ctx(bench, **over):
    a = dict(frames="single", options="alliance", hint_source="baseline", option_seed=0,
             permutations=None, retries=0, blank_calibrate_clef=False)
    a.update(over)
    return SimpleNamespace(args=SimpleNamespace(**a), bench_dir=bench,
                           perms_for=lambda arm: a["permutations"] or B.DEFAULT_PERMS[arm],
                           crop=lambda row: (bench / row["file"]).read_bytes())


@pytest.fixture
def bench(tmp_path, monkeypatch):
    (tmp_path / "crops").mkdir()
    (tmp_path / ROW["file"]).write_bytes(_jpg(96, 48))
    monkeypatch.setenv("JEV_TYPESAFE_API_TOKEN", "test-token")
    return tmp_path


def _answers(team_top, subject_top, team_p=0.9):
    team = {k: 0.01 for k in ("1", "2", "3", *B.JEV_ESCAPES)}
    team[team_top] = team_p
    subj = {k: 0.01 for k in B.JEV_SUBJECTS}
    subj[subject_top] = 0.97
    return {"model": "jev-1.14.0",
            "answers": {"subject": {"type": "choice", "choice": subject_top,
                                    "confidence": 0.97, "probabilities": subj},
                        "team": {"type": "choice", "choice": team_top,
                                 "confidence": team_p, "probabilities": team}},
            "usage": {"input_tokens": 938, "output_tokens": 3}}


def _run(bench, monkeypatch, body, **ctx_over):
    seen = []

    def fake_post(url, payload, headers=None, timeout=120.0, retries=5):
        seen.append((url, payload, headers))
        return body, 0.2, 1
    monkeypatch.setattr(B, "post_json", fake_post)
    arm = B.Jev(_ctx(bench, **ctx_over), False)
    pred, _ = arm.predict(ROW)
    return pred, seen


def test_live_team_top(bench, monkeypatch):
    pred, seen = _run(bench, monkeypatch, _answers("2", "bumper"))
    url, payload, headers = seen[0]
    assert url == B.JEV_URL and len(seen) == 1
    assert headers["TypeSafe-Danger-Zone-Images"] == "true"
    assert headers["Authorization"] == "Bearer test-token"
    assert pred["pred_team"] == "2" and pred["top"] == "2" and pred["error"] is None
    assert pred["model"] == "jev-1.14.0"
    assert pred["subject_top"] == "bumper" and pred["subject_probs"]["bumper"] == 0.97
    assert pred["usage"] == {"input_tokens": 938, "output_tokens": 3}
    assert pred["est_cost_usd"] == pytest.approx(938 * 0.042 / 1e6)
    assert pred["latency_s"] == 0.2 and pred["escape_ids"] == list(B.JEV_ESCAPES)
    assert B._pred_team(pred) == "2"
    assert "test-token" not in json.dumps(pred)


@pytest.mark.parametrize("esc", B.JEV_ESCAPES)
def test_escape_top_gives_no_team(bench, monkeypatch, esc):
    pred, _ = _run(bench, monkeypatch, _answers(esc, "not_robot"))
    assert pred["top"] == esc and pred["pred_team"] is None
    assert B._pred_team(pred) is None
    m = B.score_predictions([pred], {ROW["id"]: ROW})
    assert m["abstain"] == 1 and m["assert_rate"] == 0.0
    assert m["listed_mass"] == pytest.approx(0.03)  # only 1,2,3 count as listed
    neg = dict(ROW, label_status="illegible", label_team=None)
    m = B.score_predictions([pred], {ROW["id"]: neg})
    assert m["fp"] == 0 and m["ece"] is not None


def test_permutations_default_and_override(bench, monkeypatch):
    _pred, seen = _run(bench, monkeypatch, _answers("2", "bumper"))
    assert len(seen) == 1
    pred, seen = _run(bench, monkeypatch, _answers("2", "bumper"), permutations=3)
    assert len(seen) == 3 and len(pred["perms"]) == 3
    orders = {tuple(pm["option_ids"]) for pm in pred["perms"]}
    assert len(orders) == 3


def test_subj_variant_gates_team(bench, monkeypatch):
    ok, _ = _run(bench, monkeypatch, _answers("2", "bumper"))
    blocked, _ = _run(bench, monkeypatch, _answers("2", "no_digits"))
    man = {ROW["id"]: ROW}
    variants = dict(B.score_file_variants([ok], man, "jev:alliance"))
    assert set(variants) == {"jev:alliance", "jev:alliance+subj"}
    assert variants["jev:alliance+subj"]["correct"] == 1
    variants = dict(B.score_file_variants([blocked], man, "jev:alliance"))
    assert variants["jev:alliance"]["correct"] == 1
    gated = variants["jev:alliance+subj"]
    assert gated["correct"] == 0 and gated["abstain"] == 1 and gated["assert_rate"] == 0.0
    # on a negative crop the gate turns a false positive into a pass
    neg = dict(ROW, label_status="not_bumper", label_team=None)
    variants = dict(B.score_file_variants([blocked], {ROW["id"]: neg}, "jev:alliance"))
    assert variants["jev:alliance"]["fp"] == 1 and variants["jev:alliance+subj"]["fp"] == 0
    # the original row is untouched by the gated view
    assert blocked["top"] == "2"


def test_existing_arms_have_no_subj_variant_and_keep_escapes():
    row = {"id": "x", "arm": "clef-flash", "probs": {"2": 0.1, "none": 0.8, "other": 0.1},
           "top": "none", "top_p": 0.8, "option_ids": ["2", "none", "other"],
           "pred_team": None, "error": None}
    assert [n for n, _ in B.score_file_variants([row], {"x": dict(ROW, id="x")}, "l")] == ["l"]
    assert B._pred_team(row) is None


def test_frames_other_than_single_raises(bench):
    with pytest.raises(B.ArmError, match="single frame only"):
        B.Jev(_ctx(bench, frames="multi4"), True)


def test_missing_token_raises_live_but_not_dry(bench, monkeypatch):
    monkeypatch.delenv("JEV_TYPESAFE_API_TOKEN")
    monkeypatch.setattr(B, "ENV_FILE", bench / "no.env")
    with pytest.raises(B.ArmError, match="JEV_TYPESAFE_API_TOKEN"):
        B.Jev(_ctx(bench), False)
    assert B.Jev(_ctx(bench), True).missing == ["JEV_TYPESAFE_API_TOKEN"]


def test_dry_run_makes_no_network_call(bench, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call in dry-run")
    monkeypatch.setattr(B, "post_json", boom)
    arm = B.Jev(_ctx(bench), True)
    pred, payload = arm.predict(ROW)
    assert pred["error"] is None and pred["probs"] is None and pred["pred_team"] is None
    assert pred["est_input_tokens"] > 0 and pred["requests_per_crop"] == 1
    assert pred["image_wh"] == [560, 280]  # 96x48 scaled so the short side is 280
    assert list(payload["questions"]) == ["subject", "team"]
    assert B.make_arm("jev", _ctx(bench), True).__class__ is B.Jev

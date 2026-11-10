"""Multi-frame (--frames multi4|video) parts of scripts/bench_bumper_readers.py:
frame selection, letterboxing, payload shapes with a fake tracklet, and the
`score --subset tracklet` filter. No network; the video test needs ffmpeg."""
import base64
import io
import json
import os
import shutil
import tempfile

import pytest
from PIL import Image

from scripts.bench_bumper_readers import (
    FFMPEG,
    VIDEO_FPS,
    build_frame_media,
    build_options,
    clef_image,
    clef_media_payload,
    clef_questions,
    filter_subset,
    letterbox,
    letterbox_dims,
    load_tracklets,
    score_predictions,
    select_multi_frames,
    tracklet_subset_ids,
    usable_tracklet,
)

ROW = {"id": "m_f00010_r001", "file": "crops/m_f00010_r001.jpg",
       "red_teams": ["1", "2", "3"], "blue_teams": ["4", "5", "6"],
       "baseline_alliance_read": "red", "baseline_matched_team": "2",
       "label_status": "legible", "label_team": "2", "label_alliance": "red",
       "stratum": "matched"}


def _fr(off, sharp):
    return {"offset": off, "file": f"tracklets/x/{off}.jpg", "match_score": 0.9,
            "anchor_score": 0.9, "sharpness": sharp}


# ---- multi4 frame selection --------------------------------------------------

def test_select_multi_frames_one_per_third_sharpest():
    frames = [_fr(-15, 5), _fr(-10, 50), _fr(-5, 1), _fr(0, 999), _fr(5, 70),
              _fr(10, 3), _fr(15, 30)]
    picks = select_multi_frames(frames)
    # others = -15,-10 | -5,5 | 10,15 -> sharpest per chunk; centre never picked
    assert [f["offset"] for f in picks] == [-10, 5, 15]


def test_select_multi_frames_few_and_time_order():
    frames = [_fr(10, 1), _fr(0, 9), _fr(-5, 2)]
    assert [f["offset"] for f in select_multi_frames(frames)] == [-5, 10]
    assert select_multi_frames([_fr(0, 1)]) == []


def test_select_multi_frames_tie_prefers_centre_and_asymmetric():
    # Ties on sharpness -> the frame nearer the centre.
    frames = [_fr(o, 10) for o in (-30, -25, -20, -15, -10, -5, 0)]
    picks = select_multi_frames(frames)
    assert [f["offset"] for f in picks] == [-25, -15, -5]


# ---- letterboxing --------------------------------------------------------------

def test_letterbox_dims_even_long_side():
    assert letterbox_dims(84, 36) == (512, 220)
    assert letterbox_dims(36, 84) == (220, 512)
    assert letterbox_dims(100, 100) == (512, 512)
    for w, h in [(83, 37), (20, 30), (1476, 396), (31, 7)]:
        cw, ch = letterbox_dims(w, h)
        assert cw % 2 == 0 and ch % 2 == 0 and max(cw, ch) == 512


def test_letterbox_pads_black_and_centres():
    img = Image.new("RGB", (50, 50), (255, 0, 0))
    out = letterbox(img, 512, 220)
    assert out.size == (512, 220)
    assert out.getpixel((0, 0)) == (0, 0, 0) and out.getpixel((511, 219)) == (0, 0, 0)
    r, g, b = out.getpixel((256, 110))
    assert r > 200 and g < 40 and b < 40


# ---- payloads with a fake tracklet ----------------------------------------------

def _jpg(w, h, colour):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def fake_bench(tmp_path):
    (tmp_path / "crops").mkdir()
    crop = _jpg(96, 48, (200, 30, 30))
    (tmp_path / ROW["file"]).write_bytes(crop)
    tdir = tmp_path / "tracklets" / ROW["id"]
    tdir.mkdir(parents=True)
    frames = []
    for i, off in enumerate((-15, -10, -5, 0, 5, 10, 15)):
        (tdir / f"{off}.jpg").write_bytes(_jpg(32, 16, (10 * i, 40, 200)))
        frames.append({"offset": off, "file": f"tracklets/{ROW['id']}/{off}.jpg",
                       "match_score": 0.9, "anchor_score": 0.9, "sharpness": float(i)})
    trk = {"id": ROW["id"], "status": "ok", "fps": 60.0, "box": [100, 200, 32, 16],
           "frames": frames}
    (tmp_path / "tracklets.jsonl").write_text(
        json.dumps(trk) + "\n"
        + json.dumps({"id": "nope", "status": "loc_failed", "frames": []}) + "\n")
    return tmp_path, crop, trk


def test_load_tracklets_keeps_ok_rows(fake_bench):
    bench, _crop, trk = fake_bench
    got = load_tracklets(bench)
    assert list(got) == [ROW["id"]] and got[ROW["id"]]["frames"] == trk["frames"]
    assert load_tracklets(bench / "missing") == {}


def test_multi4_payload_shape(fake_bench):
    bench, crop, trk = fake_bench
    media = build_frame_media("multi4", crop, trk, bench)
    assert media["kind"] == "multi" and media["frames_used"] == 4
    assert media["offsets"] == sorted(media["offsets"]) and 0 in media["offsets"]
    assert media["span_s"] == pytest.approx((max(media["offsets"]) - min(media["offsets"])) / 60.0)
    # Every image prepped like the single crop (upscaled to >= 256 px min side).
    assert all(min(w, h) >= 256 for _b, w, h in media["images"])
    centre = media["images"][media["offsets"].index(0)]
    assert centre == clef_image(crop)
    opts = build_options(ROW, "decoy", "unknown")
    p = clef_media_payload("clef-flash", ROW, media, opts, "decoy", "unknown")
    assert set(p) == {"model", "state", "questions", "images"}
    assert p["model"] == "clef-flash" and len(p["images"]) == 4
    assert all(u.startswith("data:image/jpeg;base64,") for u in p["images"])
    assert "SAME robot" in p["state"] and "ordered in time" in p["state"]
    assert "4 images" in p["state"] and "Match roster" in p["state"]
    assert p["questions"] == clef_questions(opts)


def test_fallback_to_single_without_usable_tracklet(fake_bench):
    bench, crop, trk = fake_bench
    for t in (None, {**trk, "frames": trk["frames"][:1]}, {**trk, "status": "loc_failed"}):
        media = build_frame_media("multi4", crop, t, bench)
        assert media["kind"] == "single" and media["frames_used"] == 1
        assert media["images"] == [clef_image(crop)] and media["video"] is None
    assert build_frame_media("single", crop, trk, bench)["kind"] == "single"
    assert build_frame_media("single", crop, trk, bench)["tracklet_frames"] == 7


@pytest.mark.skipif(not os.path.exists(FFMPEG), reason="ffmpeg not installed")
def test_video_payload_shape_and_clip(fake_bench):
    import cv2
    bench, crop, trk = fake_bench
    media = build_frame_media("video", crop, trk, bench)
    assert media["kind"] == "video" and media["images"] == []
    assert media["frames_used"] == 7 and media["offsets"] == [-15, -10, -5, 0, 5, 10, 15]
    assert media["video_wh"] == letterbox_dims(32, 16) == (512, 256)
    p = clef_media_payload("clef-omni", ROW, media, build_options(ROW, "decoy", "unknown"),
                           "decoy", "unknown")
    assert set(p) == {"model", "state", "questions", "videos"} and p["model"] == "clef-omni"
    assert len(p["videos"]) == 1 and p["videos"][0].startswith("data:video/mp4;base64,")
    assert "video" in p["state"] and "SAME robot" in p["state"]
    raw = base64.b64decode(p["videos"][0].split(",", 1)[1])
    td = tempfile.mkdtemp()
    try:
        path = os.path.join(td, "v.mp4")
        with open(path, "wb") as fh:
            fh.write(raw)
        cap = cv2.VideoCapture(path)
        n = 0
        while cap.read()[0]:
            n += 1
        assert n == 7  # one encoded frame per tracklet crop
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(VIDEO_FPS)
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (512, 256)
        cap.release()
    finally:
        shutil.rmtree(td)
    still = build_frame_media("video", crop, trk, bench, with_still=True)
    p2 = clef_media_payload("clef-omni", ROW, still, build_options(ROW, "decoy", "unknown"),
                            "decoy", "unknown")
    assert len(p2["images"]) == 1 and len(p2["videos"]) == 1
    assert "still image" in p2["state"]


# ---- score --subset tracklet ---------------------------------------------------

def test_tracklet_subset_and_filter():
    tr = {"a": {"status": "ok", "frames": [1, 2, 3]},
          "b": {"status": "ok", "frames": [1, 2]},
          "c": {"status": "ok", "frames": list(range(9))}}
    ids = tracklet_subset_ids(tr)
    assert ids == {"a", "c"}
    assert usable_tracklet(tr["b"]) and not usable_tracklet(tr["b"], 3)
    preds = [{"id": i, "arm": "clef", "probs": {"2": 0.9, "none": 0.1}, "top": "2",
              "top_p": 0.9, "option_ids": ["2", "none", "other"]} for i in ("a", "b", "c", "d")]
    kept = filter_subset(preds, ids)
    assert [p["id"] for p in kept] == ["a", "c"]
    assert filter_subset(preds, None) == preds
    manifest = {i: {**ROW, "id": i} for i in ("a", "b", "c", "d")}
    assert score_predictions(kept, manifest)["n"] == 2
    assert score_predictions(preds, manifest)["n"] == 4


def test_quota_error_detection_and_run_subset():
    from scripts.bench_bumper_readers import is_quota_error, select_rows
    msg = ('{"errors":[{"message":"AiError: AiError: you have used up your daily free '
           'allocation of 10,000 neurons, please upgrade to Cloudflare\'s Workers Paid plan"}]}')
    assert is_quota_error(429, msg)
    assert not is_quota_error(429, '{"errors":[{"message":"rate limited"}]}')
    assert not is_quota_error(500, msg)
    rows = [{**ROW, "id": i} for i in ("a", "b", "c")]
    assert [r["id"] for r in select_rows(rows, True, None, "decoy", {"a", "c"})] == ["a", "c"]
    assert len(select_rows(rows, True, None, "decoy")) == 3


# ---- freeread --frames multi4|multi8 ---------------------------------------------

def test_freeread_hint_and_multi_prompt():
    from scripts.bench_bumper_readers import freeread_hint_line, freeread_multi_prompt
    red, blue = ["1", "2", "3"], ["4", "5", "6"]
    h = freeread_hint_line("red", red, blue, red + blue)
    assert h == ("Computer vision sees RED bumper pixels, so partial or uncertain reads should "
                 "be matched only against these red alliance teams: 1, 2, 3.\n")
    assert freeread_hint_line("unknown", red, blue, red + blue) == \
        "Possible current-match teams are: 1, 2, 3, 4, 5, 6.\n"
    p = freeread_multi_prompt(4, 1.5, h)
    assert p.startswith("These 4 images show the SAME FRC robot at 4 moments over about 1.5 "
                        "seconds, ordered in time")
    assert h in p and "give ONE reading for this robot" in p
    assert p.endswith('or "printed" if they are printed on a crate, bin, sign, or other '
                      'field structure}.')


def test_multi_image_payload_swaps_content_only():
    from scripts.bench_bumper_readers import multi_image_payload
    base = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,A"}},
        {"type": "text", "text": "old"}]}],
        "temperature": 0, "max_tokens": 64, "logprobs": True,
        "response_format": {"type": "json_schema", "json_schema": {"name": "bumper_read",
                                                                   "schema": {"x": 1}}}}
    uris = [f"data:image/jpeg;base64,{c}" for c in "ABCD"]
    p = multi_image_payload(base, uris, "new prompt")
    content = p["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url"] * 4 + ["text"]
    assert [c["image_url"]["url"] for c in content[:4]] == uris and content[4]["text"] == "new prompt"
    assert {k: v for k, v in p.items() if k != "messages"} == \
        {k: v for k, v in base.items() if k != "messages"}
    assert base["messages"][0]["content"][1]["text"] == "old"  # base not mutated


def test_parse_bumper_reply():
    from scripts.bench_bumper_readers import parse_bumper_reply
    assert parse_bumper_reply('{"team": "6497", "color": "red", "surface": "bumper"}') == \
        ("6497", "red", "bumper")
    assert parse_bumper_reply('```json\n{"team": 254, "color": "blue"}\n```') == ("254", "blue", None)
    assert parse_bumper_reply('{"team": null, "color": "red"}') == (None, "red", None)
    assert parse_bumper_reply("garbage") == (None, "unknown", None)


@pytest.mark.parametrize("mode,n", [("multi4", 4), ("multi8", 8)])
def test_freeread_multi_payload_via_app(fake_bench, mode, n):
    """Real web.app._vlm_chat_payload (import only, no server): ONE request,
    n time-ordered images with the incumbent prep, guided-JSON schema kept."""
    from types import SimpleNamespace
    import web.app as appmod
    from scripts.bench_bumper_readers import (MULTI_IMAGES, FreeRead, app_style_data_uri,
                                              multi_frame_sequence)
    bench, crop, trk = fake_bench
    fake = SimpleNamespace(app=appmod, ctx=SimpleNamespace(bench_dir=bench))
    payload, offsets, span = FreeRead.build_multi_payload(
        fake, crop, trk, ["1", "2", "3"], ["4", "5", "6"], "red", MULTI_IMAGES[mode])
    n = min(n, len(trk["frames"]))  # the fake tracklet has 7 frames: multi8 sends all 7
    assert len(offsets) == n and offsets == sorted(offsets) and 0 in offsets
    content = payload["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url"] * n + ["text"]
    seq = multi_frame_sequence(crop, trk, bench, n)
    assert [c["image_url"]["url"] for c in content[:n]] == \
        [app_style_data_uri(d["bytes"]) for d in seq]
    with Image.open(io.BytesIO(base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1]))) as im:
        assert min(im.size) >= 512  # incumbent 512-px min-dim upscale
    assert f"These {n} images show the SAME FRC robot" in content[-1]["text"]
    assert "Computer vision sees RED bumper pixels" in content[-1]["text"]
    assert payload["max_tokens"] == 64 and payload["temperature"] == 0
    if appmod.VLM_GUIDED_JSON:
        assert payload["response_format"]["json_schema"] == \
            {"name": "bumper_read", "schema": appmod._BUMPER_READ_SCHEMA}
    assert span == pytest.approx((max(offsets) - min(offsets)) / 60.0)


# ---- freeread --prompt escape ----------------------------------------------------

def _escape_ctx(bench, **over):
    from types import SimpleNamespace
    a = dict(server_url="http://127.0.0.1:1/v1", frames="single", prompt="escape",
             options="alliance", hint_source="baseline", no_thinking=False)
    a.update(over)
    return SimpleNamespace(args=SimpleNamespace(**a), bench_dir=bench,
                           vlm_model="test-model", vlm_extra={}, tracklet=lambda _i: None)


@pytest.fixture
def freeread_app(monkeypatch):
    """web.app restored after FreeRead.__init__ patches it (module globals)."""
    import web.app as appmod
    for name in ("_post_vlm_chat", "VLM_SERVER_URL", "_active_vlm_payload_model",
                 "_active_vlm_extra_body", "_vllm_read_bumpers"):
        monkeypatch.setattr(appmod, name, getattr(appmod, name))
    return appmod


def test_escape_prompt_and_hint():
    from scripts.bench_bumper_readers import freeread_escape_hint_line, freeread_escape_prompt
    red, blue = ["1", "2", "3"], ["4", "5", "6"]
    h = freeread_escape_hint_line("red", red, blue, red + blue)
    assert h == "Teams on the red alliance in this match: 1, 2, 3."
    assert freeread_escape_hint_line("blue", red, blue, red + blue) == \
        "Teams on the blue alliance in this match: 4, 5, 6."
    assert freeread_escape_hint_line("unknown", red, blue, red + blue) == \
        "Teams in this match: 1, 2, 3, 4, 5, 6."
    p = freeread_escape_prompt(h)
    assert h in p
    for opt in ('"bumper"', '"no_digits"', '"printed"', '"not_robot"'):
        assert opt in p
    assert "ONE FRC robot" not in p and "should be matched only" not in p
    assert p.rstrip().endswith('"color": "red" | "blue" | "none"}')


def test_escape_schema_subject_first():
    from scripts.bench_bumper_readers import ESCAPE_READ_SCHEMA
    props = list(ESCAPE_READ_SCHEMA["properties"])
    assert props[0] == "subject" and props == ["subject", "team", "color"]
    assert ESCAPE_READ_SCHEMA["properties"]["subject"]["enum"] == \
        ["bumper", "no_digits", "printed", "not_robot"]


@pytest.mark.parametrize("subject,surface", [("not_robot", "unknown"), ("no_digits", "unknown"),
                                             ("printed", "printed")])
def test_parse_escape_reply_forces_team_none(subject, surface):
    from scripts.bench_bumper_readers import parse_escape_reply
    text = json.dumps({"subject": subject, "team": "2485", "color": "red"})
    assert parse_escape_reply(text) == (None, "red", surface, subject)


def test_parse_escape_reply_bumper_keeps_digits():
    from scripts.bench_bumper_readers import parse_escape_reply
    assert parse_escape_reply('{"subject": "bumper", "team": "2485", "color": "blue"}') == \
        ("2485", "blue", "bumper", "bumper")
    assert parse_escape_reply('```json\n{"subject": "bumper", "team": 254, "color": "red"}\n```') == \
        ("254", "red", "bumper", "bumper")
    # missing / garbage subject is treated as not-a-bumper
    assert parse_escape_reply('{"team": "254", "color": "red"}')[0] is None
    assert parse_escape_reply("garbage") == (None, "unknown", "unknown", None)


def test_escape_with_multiframe_raises(fake_bench, freeread_app):
    from scripts.bench_bumper_readers import ArmError, FreeRead
    bench, _crop, _trk = fake_bench
    with pytest.raises(ArmError, match="--prompt escape supports --frames single only"):
        FreeRead(_escape_ctx(bench, frames="multi4"), True)


def test_escape_dry_predict_row_no_network(fake_bench, freeread_app, monkeypatch):
    from scripts.bench_bumper_readers import FreeRead
    bench, _crop, _trk = fake_bench
    calls = []
    monkeypatch.setattr(freeread_app, "_vllm_read_bumpers",
                        lambda *a, **k: calls.append("incumbent") or [{}])
    ctx = _escape_ctx(bench)
    ctx.crop = lambda row: (bench / row["file"]).read_bytes()
    arm = FreeRead(ctx, True)
    arm.orig_post = lambda p: calls.append("post") or {}
    pred, payload = arm.predict(ROW)
    assert calls == []  # no incumbent read, no network post in dry-run
    assert pred["prompt_variant"] == "escape" and pred["error"] is None
    assert pred["pred_team"] is None
    assert pred["freeread"]["subject"] == "not_robot" and pred["freeread"]["surface"] == "unknown"
    assert pred["freeread"]["raw_text"] and pred["latencies_s"] and pred["attempts"] == 1
    assert payload["response_format"]["json_schema"]["name"] == "bumper_read_escape"
    text = payload["messages"][0]["content"][1]["text"]
    assert "Teams on the red alliance in this match: 1, 2, 3." in text


def test_escape_live_path_posts_and_applies_extra_body(fake_bench, freeread_app, monkeypatch):
    from scripts.bench_bumper_readers import FreeRead
    bench, _crop, _trk = fake_bench
    ctx = _escape_ctx(bench)
    ctx.vlm_extra = {"chat_template_kwargs": {"enable_thinking": False}}
    ctx.crop = lambda row: (bench / row["file"]).read_bytes()
    arm = FreeRead(ctx, False)
    seen = []

    def fake_post(payload):
        seen.append(payload)
        return {"choices": [{"message": {"content":
                '{"subject": "bumper", "team": "2", "color": "red"}'}, "logprobs": None}],
                "usage": {"prompt_tokens": 1}}
    arm.orig_post = fake_post
    pred, _ = arm.predict(ROW)
    assert len(seen) == 1 and seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert pred["pred_team"] == "2" and pred["freeread"]["subject"] == "bumper"
    assert pred["usage"] == {"prompt_tokens": 1}


def test_default_prompt_keeps_incumbent_path(fake_bench, freeread_app, monkeypatch):
    from scripts.bench_bumper_readers import FreeRead, _label
    bench, _crop, _trk = fake_bench
    calls = []
    monkeypatch.setattr(freeread_app, "_vllm_read_bumpers",
                        lambda *a, **k: calls.append(k) or [{"team_number": None}])
    ctx = _escape_ctx(bench, prompt="incumbent")
    ctx.crop = lambda row: (bench / row["file"]).read_bytes()
    arm = FreeRead(ctx, True)
    pred, _ = arm.predict(ROW)
    assert len(calls) == 1 and pred["prompt_variant"] == "incumbent"
    assert _label(pred) == "freeread" and "subject" not in pred["freeread"]
    assert _label({"arm": "freeread", "options_mode": "missing-true", "frames_mode": "single",
                   "prompt_variant": "escape"}) == "freeread:missing-true+escape"

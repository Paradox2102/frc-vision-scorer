"""web/bumper_register.py (multi-frame registration + fusion of a bumper
tracklet) and its bench wiring (--frames registered|anchor, --labels) in
scripts/bench_bumper_readers.py. Synthetic images only; no network."""
import json

import cv2
import numpy as np
import pytest

from web.bumper_register import (
    comb_ratio,
    psnr,
    register_and_fuse,
    register_pair,
    split_fields,
    warp_plausibility,
)

H, W = 36, 84          # native bumper box, like the bench's small crops
SS = 8                 # supersampling of the clean master image
UP = 4


def _master() -> np.ndarray:
    """Clean high-res bumper: white digits on red, SS x native."""
    img = np.zeros((H * SS, W * SS, 3), np.uint8)
    img[:] = (40, 40, 200)
    cv2.putText(img, "4256", (30, int(H * SS * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, SS * 0.95,
                (255, 255, 255), SS * 2, cv2.LINE_AA)
    return img


MASTER = _master()
CLEAN_UP = cv2.resize(MASTER, (W * UP, H * UP), interpolation=cv2.INTER_AREA)


def _degrade(dx: float, dy: float, rng, blur: float = 0.3, noise: float = 8.0,
             angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Native frame of the master shifted by (dx, dy) native px (optionally
    rotated / scaled about the centre), optically blurred, box-downsampled,
    noised and JPEG'd."""
    c = (MASTER.shape[1] / 2, MASTER.shape[0] / 2)
    M = cv2.getRotationMatrix2D(c, angle, scale)
    M[0, 2] += dx * SS
    M[1, 2] += dy * SS
    s = cv2.warpAffine(MASTER, M, (MASTER.shape[1], MASTER.shape[0]), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    if blur:
        s = cv2.GaussianBlur(s, (0, 0), blur * SS)
    n = cv2.resize(s, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    n += rng.normal(0, noise, n.shape)
    n = np.clip(n, 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", n, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], 1)


def _tracklet(n: int = 7, seed: int = 0, **kw):
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0)] + [tuple(rng.uniform(-1.5, 1.5, 2)) for _ in range(n - 1)]
    return [_degrade(dx, dy, rng, **kw) for dx, dy in shifts], shifts


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #

def test_register_pair_recovers_subpixel_shift():
    rng = np.random.default_rng(1)
    a = _degrade(0, 0, rng, noise=3)
    for dx, dy in [(0.4, -0.7), (-1.3, 1.1), (0.9, 0.25)]:
        f = _degrade(dx, dy, rng, noise=3)
        r = register_pair(a, f)
        assert r["ok"] and r["ecc"] > 0.95
        # warp maps anchor -> frame coords: frame(x + d) = anchor(x)
        assert r["warp"][0, 2] == pytest.approx(dx, abs=0.12)
        assert r["warp"][1, 2] == pytest.approx(dy, abs=0.12)


def test_warp_plausibility_limits():
    ok = np.array([[1.02, 0.01, 0.5], [-0.01, 0.99, -0.3]], np.float32)
    assert warp_plausibility(ok, W, H, 0.12, 0.10, 8.0, 0.30) is None
    big_scale = np.array([[1.3, 0, 0], [0, 1.3, 0]], np.float32)
    assert "scale" in warp_plausibility(big_scale, W, H, 0.12, 0.10, 8.0, 0.30)
    shear = np.array([[1.0, 0.3, 0], [0, 1.0, 0]], np.float32)
    assert "shear" in warp_plausibility(shear, W, H, 0.12, 0.10, 8.0, 0.30)
    far = np.array([[1, 0, 40], [0, 1, 0]], np.float32)
    assert "shift" in warp_plausibility(far, W, H, 0.12, 0.10, 8.0, 0.30)
    flip = np.array([[-1, 0, W], [0, 1, 0]], np.float32)
    assert warp_plausibility(flip, W, H, 0.12, 0.10, 8.0, 0.30) == "flip"


# --------------------------------------------------------------------------- #
# fusion quality
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fuse", ["robust", "median", "mean"])
def test_fused_closer_to_clean_than_anchor(fuse):
    frames, _ = _tracklet(7, seed=0, noise=10)
    fused, d = register_and_fuse(frames, 0, UP, deinterlace="none", fuse=fuse)
    single, ds = register_and_fuse(frames[:1], 0, UP, deinterlace="none")
    plain = cv2.resize(frames[0], (W * UP, H * UP), interpolation=cv2.INTER_CUBIC)
    assert d["fused"] and d["n_used"] == 7 and not d["rejected"]
    assert not ds["fused"] and ds["n_used"] == 1
    assert fused.shape == (H * UP, W * UP, 3)
    p_f, p_s, p_p = psnr(fused, CLEAN_UP), psnr(single, CLEAN_UP), psnr(plain, CLEAN_UP)
    assert p_f > p_s + 0.5, (p_f, p_s)
    assert p_f > p_p + 0.5, (p_f, p_p)


def test_garbage_frames_rejected_not_fused():
    frames, _ = _tracklet(6, seed=2)
    rng = np.random.default_rng(9)
    junk = np.clip(rng.normal(128, 60, frames[0].shape), 0, 255).astype(np.uint8)
    rotated = _degrade(0, 0, rng, angle=35.0)            # implausible warp / poor ECC
    zoomed = _degrade(0, 0, rng, scale=1.45)             # scale far outside the limit
    other = np.zeros_like(frames[0])
    other[:] = (200, 60, 30)
    cv2.putText(other, "9917", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    allf = frames + [junk, rotated, zoomed, other]
    fused, d = register_and_fuse(allf, 0, UP, deinterlace="none", max_samples=len(allf))
    bad = {6, 7, 8, 9}
    used = set(d["used"])
    assert not (used & bad), d
    rej = {s[0] for s, _why in d["rejected"]}
    assert bad <= rej, d["rejected"]
    clean_only, _ = register_and_fuse(frames, 0, UP, deinterlace="none")
    # the garbage changed nothing: same frames fused -> same image
    assert np.abs(fused.astype(int) - clean_only.astype(int)).max() <= 1


def test_robust_merge_suppresses_a_locally_wrong_frame():
    """A frame that aligns globally but whose digits differ (an occluder / a
    part that moved): robust fusion must keep the anchor's digits, not ghost."""
    frames, _ = _tracklet(5, seed=4, noise=4)
    bad = frames[1].copy()
    cv2.rectangle(bad, (30, 8), (52, 30), (40, 40, 200), -1)   # paint over 2 digits
    fr = [frames[0], bad] + frames[2:]
    rob, _ = register_and_fuse(fr, 0, UP, deinterlace="none", fuse="robust")
    mean, _ = register_and_fuse(fr, 0, UP, deinterlace="none", fuse="mean")
    roi = (slice(8 * UP, 30 * UP), slice(30 * UP, 52 * UP))
    err_r = np.abs(rob[roi].astype(float) - CLEAN_UP[roi]).mean()
    err_m = np.abs(mean[roi].astype(float) - CLEAN_UP[roi]).mean()
    assert err_r < err_m * 0.8, (err_r, err_m)


def test_single_frame_and_auto_upscale():
    frames, _ = _tracklet(1)
    out, d = register_and_fuse(frames, 0)          # upscale=None -> auto (160 px target)
    assert not d["fused"] and d["n_used"] == 1 and d["upscale"] == 4
    assert out.shape == (H * 4, W * 4, 3) and out.dtype == np.uint8
    big = cv2.resize(frames[0], (W * 2, H * 2))
    _, d2 = register_and_fuse([big], 0)
    assert d2["upscale"] == 3


def test_fast_enough():
    frames, _ = _tracklet(16, seed=5)
    register_and_fuse(frames, 0, deinterlace="none")      # warm-up
    _, d = register_and_fuse(frames, 0, deinterlace="none")
    assert d["ms"] < 150, d["ms"]                          # ~10-30 ms typical


# --------------------------------------------------------------------------- #
# interlace
# --------------------------------------------------------------------------- #

def _woven(dx: float, rng) -> np.ndarray:
    """Interlaced frame: even rows from one instant, odd rows from the next,
    the bumper having moved ``dx`` px in between."""
    a = _degrade(0, 0, rng, noise=2)
    b = _degrade(dx, 0, rng, noise=2)
    out = a.copy()
    out[1::2] = b[1::2]
    return out


def test_comb_ratio_separates_interlaced_from_progressive():
    rng = np.random.default_rng(6)
    prog = _degrade(0, 0, rng, noise=2)
    assert comb_ratio(prog) < 0.9          # the auto threshold
    assert comb_ratio(_woven(2.5, rng)) > 1.2


def test_split_fields_geometry():
    """Fields of a progressive frame resample back onto true row positions:
    each is close to the frame itself, and they agree with each other."""
    rng = np.random.default_rng(7)
    prog = _degrade(0, 0, rng, noise=0, blur=0.6)
    f0, f1 = split_fields(prog)
    assert f0.shape == prog.shape == f1.shape
    assert psnr(f0, prog) > 26 and psnr(f1, prog) > 26
    # a half-row misplacement would make the fields disagree far more
    assert psnr(f0, f1) > 24


def test_auto_deinterlace_beats_woven_anchor():
    rng = np.random.default_rng(8)
    woven = [_woven(2.0, rng) for _ in range(4)]
    out, d = register_and_fuse(woven, 0, UP, deinterlace="auto")
    assert d["split_fields"] and d["fused"] and d["n_samples"] >= 4
    plain, dn = register_and_fuse(woven, 0, UP, deinterlace="none")
    assert not dn["split_fields"]
    assert comb_ratio(out) < comb_ratio(plain)


# --------------------------------------------------------------------------- #
# bench wiring
# --------------------------------------------------------------------------- #

def _fake_bench(tmp_path, n_frames=6):
    rng = np.random.default_rng(11)
    tdir = tmp_path / "tracklets" / "m_f00010_r001"
    tdir.mkdir(parents=True)
    frames = []
    for k, off in enumerate(range(-5 * (n_frames // 2), 5 * (n_frames - n_frames // 2), 5)):
        img = _degrade(*(rng.uniform(-1, 1, 2) if off else (0, 0)), rng)
        p = tdir / f"{off}.jpg"
        cv2.imwrite(str(p), img)
        frames.append({"offset": off, "file": f"tracklets/m_f00010_r001/{off}.jpg",
                       "match_score": 0.9, "sharpness": 100.0, "box": [10, 10, W, H]})
    (tmp_path / "crops").mkdir()
    crop = cv2.resize(_degrade(0, 0, rng), (W * 11, H * 11))
    cv2.imwrite(str(tmp_path / "crops" / "m_f00010_r001.jpg"), crop)
    trk = {"id": "m_f00010_r001", "status": "ok", "box": [10, 10, W, H], "fps": 60.0,
           "frames": frames}
    (tmp_path / "tracklets.jsonl").write_text(json.dumps(trk) + "\n")
    return trk


def test_registered_media_and_fallback(tmp_path):
    from scripts.bench_bumper_readers import app_bumper_jpeg, registered_media
    trk = _fake_bench(tmp_path)
    crop = (tmp_path / "crops" / "m_f00010_r001.jpg").read_bytes()
    jpg, info = registered_media("registered", crop, trk, tmp_path)
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1)
    # app prep: fused 144 px tall -> integer x3 -> 432 >= 384
    assert img.shape[0] == H * UP * 3 and img.shape[1] == W * UP * 3
    assert info["reg_fused"] and info["frames_used"] >= 3 and info["reg_fallback"] is None
    assert info["media_kind"] == "registered" and 0 in info["frame_offsets"]
    jpg_a, info_a = registered_media("anchor", crop, trk, tmp_path)
    assert info_a["frames_used"] == 1 and not info_a["reg_fused"]
    same, info_n = registered_media("registered", crop, None, tmp_path)
    assert same == crop and info_n["reg_fallback"] == "no_tracklet"
    # app prep mirrors web/app.py _bumper_crop_b64: integer factor, >= 384 tall
    out = cv2.imdecode(np.frombuffer(app_bumper_jpeg(np.zeros((50, 90, 3), np.uint8)),
                                     np.uint8), 1)
    assert out.shape[:2] == (400, 720)


def test_ctx_crop_serves_registered_image(tmp_path):
    import argparse
    from scripts.bench_bumper_readers import Ctx
    _fake_bench(tmp_path)
    row = {"id": "m_f00010_r001", "file": "crops/m_f00010_r001.jpg"}
    ctx = Ctx(argparse.Namespace(frames="registered"), tmp_path)
    assert ctx.crop(row) != (tmp_path / row["file"]).read_bytes()
    assert ctx.media_info(row)["frames_mode"] == "registered"
    ctx1 = Ctx(argparse.Namespace(frames="single"), tmp_path)
    assert ctx1.crop(row) == (tmp_path / row["file"]).read_bytes()
    assert ctx1.media_info(row)["frames_mode"] == "single"


def test_apply_label_file(tmp_path):
    from scripts.bench_bumper_readers import apply_label_file
    manifest = {"a": {"id": "a", "label_status": "illegible", "label_team": None},
                "b": {"id": "b", "label_status": "legible", "label_team": "1"}}
    p = tmp_path / "truth.jsonl"
    p.write_text(json.dumps({"id": "a", "status": "legible", "team": 254,
                             "alliance": "red"}) + "\n")
    assert apply_label_file(manifest, str(p), tmp_path) == 1
    assert manifest["a"]["label_status"] == "legible" and manifest["a"]["label_team"] == "254"
    assert manifest["a"]["label_alliance"] == "red"
    assert manifest["b"]["label_status"] == "unlabelled"     # not covered -> not scored
    assert apply_label_file(manifest, "manifest", tmp_path) == 0
    # track_truth rows: identity only
    q = tmp_path / "track_truth.jsonl"
    q.write_text(json.dumps({"id": "a", "truth_team": "1477", "truth_source": "gt"}) + "\n"
                 + json.dumps({"id": "b", "truth_team": None}) + "\n")
    assert apply_label_file(manifest, "truth", tmp_path) == 2
    assert (manifest["a"]["label_status"], manifest["a"]["label_team"]) == ("legible", "1477")
    assert manifest["b"]["label_status"] == "unlabelled"

"""S4 embedder as a per-job backend (robot_embedder knob): fast CPU tests, no weights.

Covers the knob default, backend resolution + fallback, the S4 crop rule, the per-model
threshold table and its wiring into a job, and embed_model scoping of gallery rows.
"""
import asyncio
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from web import database
from web import embedder as E
from web.job_config import KNOBS, JobConfig
from web.students import s4_embedder as S4


# ----------------------------------------------------------------------------- knob
def test_robot_embedder_knob_default_s4(monkeypatch):
    monkeypatch.delenv("ROBOT_EMBEDDER", raising=False)
    cfg = JobConfig()
    assert cfg.robot_embedder == "s4" and "robot_embedder" not in cfg.non_defaults()
    assert JobConfig({"robot_embedder": "dinov3"}).non_defaults() == {"robot_embedder": "dinov3"}
    with pytest.raises(ValueError):
        JobConfig({"robot_embedder": "clip"})
    monkeypatch.setenv("ROBOT_EMBEDDER", "s4")
    assert JobConfig().robot_embedder == "s4"
    knob = next(k for k in KNOBS if k.name == "robot_embedder")
    assert knob.choices == ("dinov3", "s4") and knob.section == "Identity & votes"


# ----------------------------------------------------------------------------- resolution
def test_resolve_dinov3_and_unknown():
    assert E.resolve_embedder_backend("dinov3") == ("dinov3", "")
    assert E.resolve_embedder_backend(None) == ("dinov3", "")
    b, why = E.resolve_embedder_backend("clip")
    assert b == "dinov3" and "unknown" in why


def test_resolve_s4_falls_back_without_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(S4, "DEFAULT_CKPT", tmp_path / "missing.pt")
    b, why = E.resolve_embedder_backend("s4")
    assert b == "dinov3" and "checkpoint missing" in why


def test_resolve_s4_falls_back_without_cuda(monkeypatch, tmp_path):
    ck = tmp_path / "s4.pt"
    ck.write_bytes(b"x")
    monkeypatch.setattr(S4, "DEFAULT_CKPT", ck)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b, why = E.resolve_embedder_backend("s4")
    assert b == "dinov3" and "CUDA" in why
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert E.resolve_embedder_backend("s4") == ("s4", "")


def test_make_robot_embedder_backends(monkeypatch, tmp_path):
    ck = tmp_path / "s4_embedder.pt"
    ck.write_bytes(b"weights")
    monkeypatch.setattr(S4, "DEFAULT_CKPT", ck)
    s4 = E.make_robot_embedder("s4")
    assert isinstance(s4, S4.S4RobotEmbedder) and s4.backend == "s4"
    assert s4.model_name.startswith("s4:s4_embedder:") and s4._impl is None  # lazy
    assert E.embedder_model_name("s4") == s4.model_name
    d = E.make_robot_embedder("dinov3")
    assert isinstance(d, E.RobotEmbedder) and d.backend == "dinov3"
    assert d.model_name == E.ACTIVE_MODEL == E.embedder_model_name("dinov3")


def test_model_tag_tracks_checkpoint_content(tmp_path):
    a, b = tmp_path / "s4_embedder.pt", tmp_path / "other" / "s4_embedder.pt"
    b.parent.mkdir()
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert S4.model_tag(a) != S4.model_tag(b)
    assert E.backend_of_model(S4.model_tag(a)) == "s4"
    assert E.backend_of_model("timm/vit_small_patch16_dinov3.lvd1689m") == "dinov3"
    assert E.backend_of_model(None) == "dinov3"


# ----------------------------------------------------------------------------- crop rule
def test_robot_crop_is_harvester_pad_not_inset():
    frame = np.zeros((200, 300, 3), np.uint8)
    c = S4.robot_crop(frame, (100, 50, 200, 150))
    assert c.shape[:2] == (120, 120)            # 100 px box + 10% each side
    c = S4.robot_crop(frame, (0, 0, 50, 40))    # clamped at the frame edge
    assert c.shape[:2] == (44, 55)
    assert S4.robot_crop(frame, (10, 10, 15, 100)) is None   # < MIN_CROP_PX wide
    assert S4.pad_box((100, 50, 200, 150), 0.10, 300, 200) == (90, 40, 210, 160)


def test_letterbox_matches_training_preprocess():
    from scripts.teacher.students import features as F
    rng = np.random.default_rng(0)
    for shape in ((60, 130, 3), (300, 90, 3), (224, 224, 3)):
        crop = rng.integers(0, 255, shape, dtype=np.uint8)
        np.testing.assert_array_equal(S4.letterbox(crop), F.preprocess(crop, "letterbox", 224))
    assert S4.letterbox(np.zeros((0, 0, 3), np.uint8)).shape == (3, 224, 224)


def test_field_deinterlace_only_on_combing():
    rng = np.random.default_rng(1)
    flat = np.full((40, 40, 3), 120, np.uint8)
    assert S4.field_deinterlace(flat) is flat
    comb = np.zeros((40, 40, 3), np.uint8)
    comb[::2] = 255                              # odd rows black: maximal combing
    comb[::2] = rng.integers(200, 255, comb[::2].shape, dtype=np.uint8)
    out = S4.field_deinterlace(comb)
    assert out.shape == comb.shape and out.min() >= 150   # even field stretched


class _FakeImpl:
    def __init__(self):
        self.calls = []

    def embed_crops(self, crops, batch=64):
        self.calls.append([c.shape for c in crops])
        v = np.ones((len(crops), 128), np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_embed_batch_crop_rule_masks_ignored_and_invalid_none(tmp_path):
    emb = S4.S4RobotEmbedder(ckpt=tmp_path / "x.pt", deinterlace=False, model_name="s4:test")
    emb._impl = _FakeImpl()
    frame = np.zeros((200, 300, 3), np.uint8)
    mask = np.zeros((100, 100), bool)
    out = emb.embed_batch(frame, [(100, 50, 200, 150), (10, 10, 12, 12), None],
                          masks=[mask, None, None])
    assert out[0].shape == (128,) and out[1] is None and out[2] is None
    assert emb._impl.calls == [[(120, 120, 3)]]   # padded, mask NOT applied
    assert emb.embed(frame, (100, 50, 200, 150)).dtype == np.float32
    emb.release()
    assert emb._impl is None


def test_embed_batch_load_failure_latches_to_none(tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("no weights")

    monkeypatch.setattr(S4, "S4Embedder", boom)
    emb = S4.S4RobotEmbedder(ckpt=tmp_path / "x.pt", model_name="s4:test")
    frame = np.zeros((100, 100, 3), np.uint8)
    assert emb.embed_batch(frame, [(10, 10, 60, 60)]) == [None]
    assert emb.embed_batch(frame, [(10, 10, 60, 60)]) == [None]
    assert calls["n"] == 1


# ----------------------------------------------------------------------------- thresholds
def test_dinov3_thresholds_are_todays_constants():
    from web import app
    t = E.thresholds_for("dinov3")
    assert t is E.DINOV3_THRESHOLDS
    assert t.app_floor == app.APP_FLOOR == 0.55 and t.relink_floor == 0.55
    assert t.identity_floor == 0.55 and t.update_margin == 0.05
    assert t.confident_margin == 0.10 and t.divergence_sim == 0.70
    assert (t.gallery_confirm_sim, t.gallery_confirm_gap, t.gallery_supplement_sim,
            t.gallery_dedup_sim) == (0.88, 0.06, 0.80, 0.92)
    assert t.selective_app_scale == 1.0
    assert E.thresholds_for("facebook/dinov2-small") is t
    assert E.thresholds_for(None) is t


def test_s4_thresholds_table():
    t = E.thresholds_for("s4")
    assert t is E.S4_THRESHOLDS and E.thresholds_for("s4:s4_embedder:abc") is t
    for name in ("app_floor", "relink_floor", "identity_floor", "divergence_sim",
                 "gallery_confirm_sim", "gallery_supplement_sim", "gallery_dedup_sim"):
        assert 0.0 < getattr(t, name) < 1.0, name
    assert 0.0 <= t.update_margin < t.confident_margin < 0.5
    assert t.gallery_supplement_sim < t.gallery_confirm_sim < t.gallery_dedup_sim
    assert 0.0 < t.selective_app_scale <= 1.0


def _job(backend):
    from web.app import InferenceJob
    with patch("web.app.YOLO"), \
            patch("web.app.resolve_embedder_backend", return_value=(backend, "")):
        return InferenceJob("j", "/mock/video.mp4",
                            config=JobConfig({"robot_embedder": backend}))


def test_job_wires_s4_thresholds():
    job = _job("s4")
    t = E.S4_THRESHOLDS
    assert job.embed_backend == "s4" and job._emb_thr() is t
    assert job.identity_registry.app_floor == t.identity_floor
    assert job.identity_registry.update_margin == t.update_margin
    assert job.confidence_config.confident_margin == t.confident_margin
    assert job._gallery_vote_thresholds() == (t.gallery_confirm_sim, t.gallery_confirm_gap,
                                              t.gallery_supplement_sim)
    assert job._gallery_dedup_sim() == t.gallery_dedup_sim
    assert job._gallery_deposits.dedup_sim == t.gallery_dedup_sim


def test_job_default_keeps_dinov3_values():
    from web import app
    job = _job("dinov3")
    assert job.embed_backend == "dinov3" and job._emb_thr() is E.DINOV3_THRESHOLDS
    assert job.identity_registry.app_floor == 0.55 and job.identity_registry.update_margin == 0.05
    assert job.confidence_config.confident_margin == 0.10
    assert job._gallery_vote_thresholds() == (app.GALLERY_CONFIRM_SIM, app.GALLERY_CONFIRM_GAP,
                                              app.GALLERY_SUPPLEMENT_SIM)
    assert job._embed_model_tag() == E.ACTIVE_MODEL


def test_job_falls_back_with_warning(capsys):
    from web.app import InferenceJob
    with patch("web.app.YOLO"), patch(
            "web.app.resolve_embedder_backend",
            return_value=("dinov3", "S4 checkpoint missing (x); using dinov3")):
        job = InferenceJob("j", "/mock/video.mp4", config=JobConfig({"robot_embedder": "s4"}))
    assert job.embed_backend == "dinov3" and job._emb_thr() is E.DINOV3_THRESHOLDS
    assert "S4 checkpoint missing" in capsys.readouterr().out


def test_release_embedder_drops_weights():
    job = _job("dinov3")

    class _Emb:
        released = False

        def release(self):
            _Emb.released = True

    job.robot_embedder = _Emb()
    job._release_embedder()
    assert job.robot_embedder is None and _Emb.released


def test_identity_registry_update_margin_gates_ema():
    from web.field.identity import IdentityRegistry
    assert IdentityRegistry().update_margin == 0.05
    assert IdentityRegistry(update_margin=0.023).update_margin == 0.023


def test_confidence_margin_param():
    from web.field.identity_confidence import (ConfidenceConfig, ConfidenceSignals,
                                               ConfidenceState, calculate_confidence_update)
    sig = ConfidenceSignals(is_observed=True, relink_margin=0.09)
    st = ConfidenceState(confidence=0.5, is_confirmed=False)
    assert "low_margin_drop" in calculate_confidence_update(st, sig, ConfidenceConfig()).reasons
    up = calculate_confidence_update(st, sig, ConfidenceConfig(confident_margin=0.081))
    assert "confident_observation" in up.reasons


# ----------------------------------------------------------------------------- embed_model scoping
@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _seed_two_spaces(db):
    ev = asyncio.run(db.upsert_event("2026xx", "Test", 2026, "X", "2026-03-01", "2026-03-03"))
    for n in (1, 2):
        asyncio.run(db.upsert_match(ev, f"2026xx_qm{n}", "qm", n))
    for model, dim in ((E.ACTIVE_MODEL, 384), ("s4:s4_embedder:abc", 128)):
        asyncio.run(db.insert_gallery_exemplar({
            "event_key": "2026xx", "source_match_key": "2026xx_qm1", "team": "254",
            "alliance_color": "red", "embedding": np.ones(dim, np.float32).tobytes(),
            "legibility": 0.5, "verified": 1, "embed_model": model}))


def test_gallery_rows_scoped_by_embed_model(db):
    _seed_two_spaces(db)
    dino = asyncio.run(db.get_gallery_exemplars("2026xx"))
    s4 = asyncio.run(db.get_gallery_exemplars("2026xx", embed_model="s4:s4_embedder:abc"))
    assert [r["embed_model"] for r in dino] == [E.ACTIVE_MODEL]
    assert [r["embed_model"] for r in s4] == ["s4:s4_embedder:abc"]
    prev = asyncio.run(db.get_previous_gallery_exemplars(
        "2026xx", "2026xx_qm2", verified_only=True, embed_model="s4:s4_embedder:abc"))
    assert [r["embed_model"] for r in prev] == ["s4:s4_embedder:abc"]
    prev_d = asyncio.run(db.get_previous_gallery_exemplars("2026xx", "2026xx_qm2"))
    assert [r["embed_model"] for r in prev_d] == [E.ACTIVE_MODEL]

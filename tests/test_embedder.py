import importlib
import sys
import types

import numpy as np
import pytest

from web.embedder import RobotEmbedder, cosine_similarity, _apply_mask, _clamp_box

EMBED_DIM = 384


def _synthetic_frame():
    # Self-contained: unlike sample_frame/second_frame, needs no dataset on
    # disk, so the timm-backend tests below always run (never skip).
    rng = np.random.default_rng(0)
    return (rng.random((64, 64, 3)) * 255).astype(np.uint8)


def test_apply_mask_zeros_background():
    crop = np.full((10, 10, 3), 255, np.uint8)
    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    out = _apply_mask(crop, mask)
    assert (out[mask] == 255).all()       # robot pixels kept
    assert (out[~mask] == 0).all()        # background blacked out
    assert crop[0, 0, 0] == 255           # original untouched (copy made)


def test_apply_mask_resizes_to_crop():
    crop = np.full((20, 20, 3), 200, np.uint8)
    mask = np.ones((5, 5), bool)          # smaller than crop -> resized up
    out = _apply_mask(crop, mask)
    assert out.shape == crop.shape
    assert (out == 200).all()             # all-true mask keeps everything


def test_apply_mask_none_is_passthrough():
    crop = np.full((4, 4, 3), 7, np.uint8)
    assert _apply_mask(crop, None) is crop


def test_clamp_box_rejects_tiny():
    assert _clamp_box((0, 0, 3, 3), 100, 100) is None
    assert _clamp_box((0, 0, 50, 50), 100, 100) == (0, 0, 50, 50)


def test_embed_with_mask_returns_none_for_tiny_box(sample_frame):
    emb = RobotEmbedder()
    m = np.ones((3, 3), bool)
    assert emb.embed(sample_frame, (0, 0, 3, 3), mask=m) is None  # no model load


def test_cosine_identical_vectors_is_one():
    v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    v = v / np.linalg.norm(v)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_handles_none():
    assert cosine_similarity(None, np.array([1.0])) == 0.0
    assert cosine_similarity(np.array([1.0]), None) == 0.0


def test_embed_returns_none_for_tiny_box(sample_frame):
    emb = RobotEmbedder()
    assert emb.embed(sample_frame, (0, 0, 3, 3)) is None  # < min crop size, no model load


@pytest.mark.model
def test_embed_returns_normalized_vector(sample_frame):
    emb = RobotEmbedder()
    h, w = sample_frame.shape[:2]
    v = emb.embed(sample_frame, (w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    assert v is not None
    assert v.shape == (EMBED_DIM,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.model
def test_same_crop_more_similar_than_different(sample_frame, second_frame):
    emb = RobotEmbedder()
    h, w = sample_frame.shape[:2]
    box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    v1 = emb.embed(sample_frame, box)
    v1b = emb.embed(sample_frame, box)
    v2 = emb.embed(second_frame, box)
    assert cosine_similarity(v1, v1b) == pytest.approx(1.0, abs=1e-3)
    assert cosine_similarity(v1, v2) < cosine_similarity(v1, v1b)


@pytest.mark.model
def test_embed_batch_matches_embed(sample_frame):
    emb = RobotEmbedder()
    h, w = sample_frame.shape[:2]
    boxes = [(w // 4, h // 4, 3 * w // 4, 3 * h // 4), (0, 0, w // 2, h // 2)]
    batched = emb.embed_batch(sample_frame, boxes)
    assert len(batched) == 2
    single = emb.embed(sample_frame, boxes[0])
    assert cosine_similarity(batched[0], single) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.model
def test_masked_embedding_differs_from_unmasked(sample_frame):
    # Masking out half the box must change the appearance signature: the
    # background pixels it removes genuinely affect DINOv2's CLS token.
    emb = RobotEmbedder()
    h, w = sample_frame.shape[:2]
    box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    bw, bh = box[2] - box[0], box[3] - box[1]
    mask = np.zeros((bh, bw), bool)
    mask[: bh // 2, :] = True            # keep only the top half
    plain = emb.embed(sample_frame, box)
    masked = emb.embed(sample_frame, box, mask=mask)
    assert masked is not None
    assert masked.shape == (EMBED_DIM,)
    assert np.linalg.norm(masked) == pytest.approx(1.0, abs=1e-4)
    assert cosine_similarity(plain, masked) < 0.999


# --- ROBOT_EMBEDDER_MODEL / timm backend ---

def test_env_var_resolves_active_model(monkeypatch):
    # ACTIVE_MODEL is resolved once at import time so RobotEmbedder() (no
    # args) picks up a DINOv3 rollout with zero code changes; reload the
    # module to re-run that resolution under a patched environment.
    monkeypatch.setenv("ROBOT_EMBEDDER_MODEL", "timm/vit_small_patch16_dinov3.lvd1689m")
    import web.embedder as embedder_mod
    try:
        importlib.reload(embedder_mod)
        assert embedder_mod.ACTIVE_MODEL == "timm/vit_small_patch16_dinov3.lvd1689m"
        assert embedder_mod.RobotEmbedder().model_name == "timm/vit_small_patch16_dinov3.lvd1689m"
        # explicit ctor arg still wins over the env-resolved default
        explicit = embedder_mod.RobotEmbedder(model_name="facebook/dinov2-small")
        assert explicit.model_name == "facebook/dinov2-small"
    finally:
        monkeypatch.delenv("ROBOT_EMBEDDER_MODEL", raising=False)
        importlib.reload(embedder_mod)  # restore ACTIVE_MODEL for later tests


def _install_fake_timm(monkeypatch):
    """Stub the `timm` module so the timm backend can be exercised without
    downloading real weights.

    forward_features() returns 3 tokens: a distinctive CLS token (all 9s) at
    index 0, and two all-zero patch tokens. Avg-pooling all tokens would blend
    in the zeros and shift the result away from the CLS direction, so a test
    that asserts the output equals the (normalized) CLS token also proves the
    avg-pool path was NOT taken.
    """
    import torch

    class FakeTimmModel:
        def __init__(self):
            self.pooled_forward_called = False
            self.forward_features_called = False

        def to(self, device):
            return self

        def eval(self):  # torch.nn.Module.eval() (inference mode), not the builtin
            return self

        def parameters(self):
            # embed_batch reads next(self._model.parameters()).dtype to pin
            # the input batch's dtype to the model's (here: float32, like
            # the real timm weights).
            yield torch.zeros(1)

        def forward_features(self, batch):
            self.forward_features_called = True
            n = batch.shape[0]
            cls = torch.full((n, 1, 4), 9.0)
            patches = torch.zeros(n, 2, 4)
            return torch.cat([cls, patches], dim=1)

        def __call__(self, *args, **kwargs):
            # This is the avg-pool trap: timm's default pooled forward() must
            # never be called by embed_batch. Surface it loudly if it is.
            self.pooled_forward_called = True
            raise AssertionError(
                "RobotEmbedder must call forward_features()[:, 0], not the "
                "model's pooled forward()")

    class FakeTimmData:
        @staticmethod
        def resolve_model_data_config(model):
            return {}

        @staticmethod
        def create_transform(is_training=False, **kwargs):
            return lambda pil_img: torch.zeros(3, 4, 4)

    fake_model = FakeTimmModel()
    fake_timm = types.SimpleNamespace(
        create_model=lambda name, pretrained, num_classes: fake_model,
        data=FakeTimmData(),
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    return fake_model


def test_timm_path_extracts_cls_token_not_avg_pool(monkeypatch):
    fake_model = _install_fake_timm(monkeypatch)
    emb = RobotEmbedder(model_name="timm/vit_small_patch16_dinov3.lvd1689m")
    frame = _synthetic_frame()
    v = emb.embed(frame, (0, 0, 64, 64))

    assert v is not None
    assert fake_model.forward_features_called
    assert not fake_model.pooled_forward_called
    # CLS token was a constant [9, 9, 9, 9] -> L2-normalized is [0.5]*4.
    # If avg-pool over all 3 tokens had been used instead, the all-zero patch
    # tokens would pull this well away from 0.5.
    assert np.allclose(v, 0.5, atol=1e-5)


def test_timm_import_failure_degrades_gracefully(monkeypatch, capsys):
    # Simulate "timm isn't installed": a None entry in sys.modules makes the
    # next `import timm` raise ImportError, same as a genuinely missing package.
    monkeypatch.setitem(sys.modules, "timm", None)
    emb = RobotEmbedder(model_name="timm/vit_small_patch16_dinov3.lvd1689m")
    frame = _synthetic_frame()

    v = emb.embed(frame, (0, 0, 64, 64))

    assert v is None
    assert emb._load_failed is True
    out = capsys.readouterr().out
    assert "failed to load" in out
    assert "position-only re-ID" in out

    # Latched: a second call doesn't retry the (still-failing) load.
    v2 = emb.embed(frame, (0, 0, 64, 64))
    assert v2 is None

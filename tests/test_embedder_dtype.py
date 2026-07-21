"""Regression test: embed_batch must not let its input batch's dtype drift
to whatever torch's *ambient* default dtype happens to be.

Production incident (dev log job 4e6a28fe, 2026-07-20): the job pipeline's
local bumper-reader fallback loads a transformers VLM via
AutoModel.from_pretrained(..., torch_dtype=torch.float16) on another thread.
transformers temporarily flips torch's global default dtype to float16 while
building that model. If RobotEmbedder.embed_batch's transform/stack runs
inside that window, the crops get stacked into a float16 batch even though
the timm model's weights are float32, and the forward pass dies with:

    RuntimeError: Input type (c10::Half) and bias type (float) should be
    the same

(web/app.py -> web/embedder.py:181 -> timm/layers/patch_embed.py -> F.conv2d)

This test reproduces the poisoned-default-dtype window deterministically,
with no model download and no GPU: RobotEmbedder's lazy-loaded internals are
stubbed directly (bypassing _ensure_model(), which no-ops once self._model
is already set -- see RobotEmbedder._ensure_model in web/embedder.py), and
torch.set_default_dtype(torch.float16) stands in for the transformers race
window.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from web.embedder import RobotEmbedder


class _StubTimmModel(torch.nn.Module):
    """Mimics the real timm model just enough for embed_batch's timm path:
    forward_features() returning a [N, tokens, dim] tensor that gets sliced
    at index 0 for the CLS token (see embed_batch's `cls = feats[:, 0]`).

    Carries one real float32 parameter so that
    `next(model.parameters()).dtype` -- what the fix queries -- resolves to
    float32, exactly like the real (already loaded, always-float32) timm
    weights: the weights aren't touched by the ambient-default-dtype race,
    only tensors freshly constructed while it's in effect are.
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.seen_dtype = None  # records the dtype embed_batch actually fed it

    def forward_features(self, x):
        self.seen_dtype = x.dtype
        n = x.shape[0]
        return torch.zeros((n, 2, 384), dtype=torch.float32)


def _stub_transform(pil_img):
    # torch.ones() honors torch's *current* default dtype -- standing in for
    # whatever op in the real timm transform pipeline would get poisoned by
    # a concurrent transformers default-dtype flip.
    return torch.ones(3, 16, 16) * 0.5


def _make_stubbed_embedder() -> tuple[RobotEmbedder, _StubTimmModel]:
    emb = RobotEmbedder(model_name="timm/vit_small_patch16_dinov3.lvd1689m")
    model = _StubTimmModel()
    emb._model = model      # non-None -> _ensure_model() treats it as already loaded
    emb._torch = torch      # normally cached by _ensure_model(); set directly since
                            # _ensure_model() no-ops (and never sets _torch) when
                            # self._model is already set
    emb._transform = _stub_transform
    emb._image_cls = Image
    emb._device = "cpu"
    return emb, model


def _synthetic_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((64, 64, 3)) * 255).astype(np.uint8)


def test_embed_batch_pins_input_dtype_to_model_dtype_during_default_dtype_race():
    """While torch's global default dtype is float16 (mid-init of an
    unrelated transformers model on another thread), embed_batch must still
    hand the timm model a float32 batch -- matching the model's own
    (always float32) parameter dtype -- not a float16 batch that would blow
    up in the model's first conv/linear layer.
    """
    emb, model = _make_stubbed_embedder()
    frame = _synthetic_frame()
    box = (0, 0, 64, 64)

    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        result = emb.embed_batch(frame, [box])
    finally:
        torch.set_default_dtype(prev_default)

    assert result[0] is not None
    assert model.seen_dtype == torch.float32

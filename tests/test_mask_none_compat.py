"""tracker_raw entries may carry mask=None in selective mode (Design-decision-3):
outside ambiguous windows, entries are pure-TBD boxes with no mask. These
tests pin the None-tolerant contract for the app.py consumers of
entry["mask"] -- _restore_mask_to_view and the extracted ground_uv
uv-selection helper -- plus a smoke check that the embedder's embed_batch
(already None-aware) doesn't choke when handed a None mask.
"""
import numpy as np

from web.app import _restore_mask_to_view, ground_uv
from web.field.mask_tracker import mask_bottom_center


def test_restore_mask_to_view_none_input_returns_none():
    assert _restore_mask_to_view(None, (10, 12, 3)) is None


def test_restore_mask_to_view_real_mask_still_works():
    mask = np.ones((2, 3), dtype=bool)
    restored = _restore_mask_to_view(mask, (4, 6, 3))
    assert restored.all()


def test_ground_uv_box_bottom_center_when_mask_none():
    box = (10.0, 20.0, 30.0, 50.0)
    assert ground_uv(box, None, use_footprint=True) == (20.0, 50.0)


def test_ground_uv_box_bottom_center_when_footprint_disabled():
    mask = np.ones((10, 12), dtype=bool)
    box = (0.0, 0.0, 12.0, 10.0)
    assert ground_uv(box, mask, use_footprint=False) == (6.0, 10.0)


def test_ground_uv_uses_mask_footprint_when_present_and_enabled():
    # Asymmetric mask: wide top (extended intake) + narrow bottom contact band,
    # same shape used in test_mask_tracker's mask_bottom_center coverage.
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:7, 0:11] = True
    mask[7:9, 8:11] = True
    box = (0.0, 2.0, 11.0, 9.0)
    expected = mask_bottom_center(mask)
    assert expected is not None
    assert ground_uv(box, mask, use_footprint=True) == expected


def test_ground_uv_falls_back_to_box_when_mask_footprint_is_empty():
    # Empty mask -> mask_bottom_center returns None -> falls back to box
    # bottom-center, exactly like the pre-extraction inline code did.
    mask = np.zeros((10, 12), dtype=bool)
    box = (0.0, 0.0, 12.0, 10.0)
    assert ground_uv(box, mask, use_footprint=True) == (6.0, 10.0)


def test_embed_batch_accepts_none_mask_without_crash():
    """embed_batch already supports masks=[None, ...] (embedder.py:131 ->
    unmasked inset crop); confirm nothing on the path does mask.shape etc.
    before that None check. Tiny box means no model load is attempted."""
    from web.embedder import RobotEmbedder

    emb = RobotEmbedder()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = emb.embed_batch(frame, [(0, 0, 3, 3)], masks=[None])
    assert out == [None]

import numpy as np
import pytest

from web.field.tracker import FieldDetection
from web.field.fusion import fuse_detections


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_fusion_single_view_carries_embedding():
    emb = _e(1, 2, 3)
    det = FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 10, 10), track_id=1, embedding=emb)
    per_cam = {"main": [(det, b"crop")]}
    fused = fuse_detections(per_cam, weights={"main": 1.0})
    assert len(fused) == 1
    assert fused[0].detection.embedding is not None
    np.testing.assert_array_almost_equal(fused[0].detection.embedding, emb)


def test_fusion_multi_view_averages_and_normalizes_embeddings():
    emb1 = _e(1, 0, 0)
    emb2 = _e(0, 1, 0)
    det1 = FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 10, 10), track_id=1, embedding=emb1)
    det2 = FieldDetection(pos=(3.2, 3.1), color="blue", box=(0, 0, 10, 10), track_id=2, embedding=emb2)
    
    # Weighted average: main is 1.0, corner is 3.0. 
    # Weighted sum = 1.0 * [1, 0, 0] + 3.0 * [0, 1, 0] = [1.0, 3.0, 0.0]
    # Normalized = [1, 3, 0] / sqrt(10)
    expected = np.array([1.0, 3.0, 0.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)

    per_cam = {
        "main": [(det1, b"crop1")],
        "corner_left": [(det2, b"crop2")]
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 3.0})
    assert len(fused) == 1
    assert fused[0].detection.embedding is not None
    np.testing.assert_array_almost_equal(fused[0].detection.embedding, expected)


def test_fusion_mix_of_embeddings_and_nones():
    emb = _e(1.0, 0.0, 0.0)
    # det1 has embedding, det2 has None embedding
    det1 = FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 10, 10), track_id=1, embedding=emb)
    det2 = FieldDetection(pos=(3.2, 3.1), color="blue", box=(0, 0, 10, 10), track_id=2, embedding=None)

    per_cam = {
        "main": [(det1, b"crop1")],
        "corner_left": [(det2, b"crop2")]
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 3.0})
    assert len(fused) == 1
    assert fused[0].detection.embedding is not None
    np.testing.assert_array_almost_equal(fused[0].detection.embedding, emb)


def test_fusion_all_nones_yields_none_embedding():
    det1 = FieldDetection(pos=(3.0, 3.0), color="blue", box=(0, 0, 10, 10), track_id=1, embedding=None)
    det2 = FieldDetection(pos=(3.2, 3.1), color="blue", box=(0, 0, 10, 10), track_id=2, embedding=None)

    per_cam = {
        "main": [(det1, b"crop1")],
        "corner_left": [(det2, b"crop2")]
    }
    fused = fuse_detections(per_cam, weights={"main": 1.0, "corner_left": 3.0})
    assert len(fused) == 1
    assert fused[0].detection.embedding is None

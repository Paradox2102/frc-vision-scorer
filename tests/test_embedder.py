import numpy as np
import pytest

from web.embedder import RobotEmbedder, cosine_similarity

EMBED_DIM = 384


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

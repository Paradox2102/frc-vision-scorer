import numpy as np

from web.app import InferenceJob


class _Stub:
    pass


def _cap(stub, slot_id, crop_b64, quality, embedding):
    return InferenceJob._capture_slot_crop(stub, slot_id, crop_b64, quality, embedding)


def test_capture_keeps_top_3_by_quality_descending():
    s = _Stub()
    s._slot_crops = {}
    for q in [0.1, 0.9, 0.5, 0.7, 0.3]:
        _cap(s, 0, f"crop{q}", q, np.array([1, 0, 0], dtype=np.float32))
    bucket = s._slot_crops[0]
    assert len(bucket) == 3
    assert [c["quality"] for c in bucket] == [0.9, 0.7, 0.5]  # top-3, sorted desc


def test_capture_noop_on_empty_crop():
    s = _Stub()
    s._slot_crops = {}
    _cap(s, 0, "", 0.9, np.array([1, 0, 0], dtype=np.float32))
    _cap(s, 0, None, 0.9, np.array([1, 0, 0], dtype=np.float32))
    assert s._slot_crops.get(0, []) == []


def test_capture_stores_embedding_copy_not_reference():
    s = _Stub()
    s._slot_crops = {}
    emb = np.array([1, 0, 0], dtype=np.float32)
    _cap(s, 0, "crop", 0.9, emb)
    emb[0] = 99.0  # mutate the original
    stored = s._slot_crops[0][0]["embedding"]
    assert stored[0] == 1.0  # snapshot unaffected


def test_capture_handles_none_embedding():
    s = _Stub()
    s._slot_crops = {}
    _cap(s, 0, "crop", 0.9, None)
    assert s._slot_crops[0][0]["embedding"] is None

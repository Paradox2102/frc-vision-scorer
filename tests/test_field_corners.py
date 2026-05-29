import numpy as np
from web.field.field_model import FieldModel, TAG_SIZE_M

def test_corners_centered_on_translation():
    fm = FieldModel.load("2026-rebuilt-welded")
    c = fm.tag_corners_3d(1)            # (4,3)
    assert c.shape == (4, 3)
    np.testing.assert_allclose(c.mean(axis=0), fm.tag_translation(1), atol=1e-6)

def test_corners_lie_in_vertical_plane_of_size_s():
    # Each edge length equals TAG_SIZE_M; corners are coplanar.
    fm = FieldModel.load("2026-rebuilt-welded")
    c = fm.tag_corners_3d(1)
    edges = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    for e in edges:
        assert abs(e - TAG_SIZE_M) < 1e-6

def test_identity_rotation_spans_local_y_z():
    # A synthetic tag with identity quaternion: corners vary only in y and z.
    fm = FieldModel.load("2026-rebuilt-welded")
    fm.tags[900] = {"t": np.array([5.0, 4.0, 1.0]),
                    "q": np.array([1.0, 0.0, 0.0, 0.0])}
    c = fm.tag_corners_3d(900)
    assert np.allclose(c[:, 0], 5.0, atol=1e-9)        # x constant (normal axis)
    assert set(np.round(c[:, 1] - 4.0, 5)) == {-TAG_SIZE_M/2, TAG_SIZE_M/2}
    assert set(np.round(c[:, 2] - 1.0, 5)) == {-TAG_SIZE_M/2, TAG_SIZE_M/2}

import math
import numpy as np
from web.field.field_model import FieldModel, TAG_SIZE_M


def test_loads_2026_layout():
    fm = FieldModel.load("2026-rebuilt-welded")
    assert len(fm.tags) == 32
    assert math.isclose(fm.length_m, 16.541, abs_tol=1e-3)
    assert math.isclose(fm.width_m, 8.069, abs_tol=1e-3)


def test_tag_pose_matches_known_value():
    fm = FieldModel.load("2026-rebuilt-welded")
    t = fm.tag_translation(1)
    np.testing.assert_allclose(t, [11.8779798, 7.4247756, 0.889], atol=1e-6)


def test_alliance_side_split_at_midline():
    fm = FieldModel.load("2026-rebuilt-welded")
    assert fm.alliance_side((1.0, 4.0)) == "blue"   # near x=0 (blue corner)
    assert fm.alliance_side((15.0, 4.0)) == "red"    # near far wall
    assert fm.alliance_side((fm.length_m / 2 - 0.01, 4.0)) == "blue"
    assert fm.alliance_side((fm.length_m / 2 + 0.01, 4.0)) == "red"


def test_unknown_tag_raises():
    fm = FieldModel.load("2026-rebuilt-welded")
    import pytest
    with pytest.raises(KeyError):
        fm.tag_translation(999)

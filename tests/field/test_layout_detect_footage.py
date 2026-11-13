"""Opt-in regression of broadcast-layout detection on real match footage.

Slow (~5-12 s per video, decoding included) and needs data/test_footage/, so it
only runs with FRC_LAYOUT_FOOTAGE=1:

    FRC_LAYOUT_FOOTAGE=1 .venv/bin/python -m pytest tests/field/test_layout_detect_footage.py -v

The reference layouts and the per-camera delta report live in
scripts/layout_detect_regression.py (run it directly for a table).
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import layout_detect_regression as reg  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("FRC_LAYOUT_FOOTAGE") != "1",
                                reason="set FRC_LAYOUT_FOOTAGE=1 to run footage regressions")

CASES = [n for n in reg.REFERENCE if reg.expected_for(n)]


@pytest.mark.parametrize("name", CASES)
def test_layout_matches_reference(name):
    path = reg.find_video(name)
    if path is None:
        pytest.skip(f"no footage for {name}")
    out = reg.run_one(path)
    exp = reg.expected_for(name)
    cams = out["cameras"]
    assert set(cams) >= set(exp), (sorted(cams), out["diag"].get("selection"))
    for cam, e in exp.items():
        assert reg.delta(cams[cam]["crop"], e) <= 6, (cam, cams[cam]["crop"], e)

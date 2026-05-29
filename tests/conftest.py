"""Shared test fixtures for the re-ID test suite."""
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGES = sorted((PROJECT_ROOT / "data" / "roboflow_frc" / "valid" / "images").glob("*.jpg"))


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A real FRC frame (BGR) from the roboflow dataset."""
    if not SAMPLE_IMAGES:
        pytest.skip("no sample images available")
    img = cv2.imread(str(SAMPLE_IMAGES[0]))
    assert img is not None
    return img


@pytest.fixture
def second_frame() -> np.ndarray:
    """A different real FRC frame (BGR)."""
    if len(SAMPLE_IMAGES) < 2:
        pytest.skip("need at least two sample images")
    img = cv2.imread(str(SAMPLE_IMAGES[1]))
    assert img is not None
    return img

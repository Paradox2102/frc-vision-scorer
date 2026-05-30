import numpy as np

from web.field.number_match import match_numbers_to_robots


def test_containment():
    # Setup a 100x100 dummy frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:30, 20:30] = 255  # paint the area to make sure crop is not empty

    # Robot 1 mask is at [10:50, 10:50]
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:50, 10:50] = True

    # Number box is at [20, 20, 30, 30] which is fully inside Robot 1
    boxes = np.array([[20, 20, 30, 30]])

    crops = match_numbers_to_robots(frame, boxes, {1: mask})
    assert 1 in crops
    assert crops[1].shape == (10, 10, 3)
    assert np.all(crops[1] == 255)


def test_no_match():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:50, 10:50] = True

    # Number box is at [60, 60, 70, 70] which doesn't overlap at all
    boxes = np.array([[60, 60, 70, 70]])

    crops = match_numbers_to_robots(frame, boxes, {1: mask})
    assert len(crops) == 0


def test_ambiguous_overlap():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Two overlapping robot masks
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[10:50, 10:50] = True
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[15:55, 15:55] = True

    # Bounding box that is fully inside BOTH mask1 and mask2
    boxes = np.array([[20, 20, 30, 30]])

    crops = match_numbers_to_robots(frame, boxes, {1: mask1, 2: mask2})
    # Should be rejected because it matches multiple robots by >= 0.9 IoS
    assert len(crops) == 0


def test_multiple_matches():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[10:40, 10:40] = True
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[60:90, 60:90] = True

    # Number boxes matching both
    boxes = np.array([[15, 15, 25, 25], [70, 70, 80, 80]])

    crops = match_numbers_to_robots(frame, boxes, {1: mask1, 2: mask2})
    assert 1 in crops
    assert 2 in crops
    assert crops[1].shape == (10, 10, 3)
    assert crops[2].shape == (10, 10, 3)

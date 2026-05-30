import numpy as np
from web.field import apriltag_detect as ad


def test_roi_maps_corners_back_to_full_frame(monkeypatch):
    # Fake inner detector: in the (upscaled) ROI it "finds" tag 17 at a known spot.
    def fake_detect(frame_bgr):
        return {17: np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float)}
    monkeypatch.setattr(ad, "detect_tags", fake_detect)
    frame = np.zeros((400, 400, 3), np.uint8)
    # click near (200,200); ROI half-size 50, upscale 4
    out = ad.detect_tags_in_roi(frame, click_xy=(200, 200), target_id=17,
                                half=50, upscale=4)
    assert out is not None
    # corner (10,10) in 4x-upscaled ROI -> (10/4,10/4) + ROI offset (150,150)
    assert np.allclose(out[0], [150 + 2.5, 150 + 2.5], atol=1e-6)


def test_roi_rejects_id_mismatch(monkeypatch):
    monkeypatch.setattr(ad, "detect_tags", lambda f: {99: np.zeros((4, 2))})
    out = ad.detect_tags_in_roi(np.zeros((400, 400, 3), np.uint8),
                                click_xy=(200, 200), target_id=17)
    assert out is None  # decoded a different tag -> reject


def test_roi_none_when_nothing_decodes(monkeypatch):
    monkeypatch.setattr(ad, "detect_tags", lambda f: {})
    out = ad.detect_tags_in_roi(np.zeros((400, 400, 3), np.uint8),
                                click_xy=(200, 200), target_id=17)
    assert out is None

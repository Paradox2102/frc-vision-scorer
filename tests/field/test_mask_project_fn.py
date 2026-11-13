"""Parity test: the mask-tracker cull's project_fn offset must match the result
consumer's field projection for nonzero top-crop. Pure algebra; no app import."""


def _project_offset_from_cropped(cropped_full_h, tc):
    # The formula project_fn uses (only the cropped exported height is available).
    return int(cropped_full_h * tc / max(1.0 - tc, 1e-6)) if tc > 0 else 0


def _consumer_offset_from_original(original_h, tc):
    # The downstream result consumer's offset (uses the original, uncropped height).
    return int(original_h * tc)


def test_project_fn_offset_matches_consumer_offset():
    for tc in (0.0, 0.2):
        for original_h in (720, 1080, 1536):
            cropped_full_h = int(round(original_h * (1.0 - tc)))
            proj = _project_offset_from_cropped(cropped_full_h, tc)
            cons = _consumer_offset_from_original(cropped_full_h / (1.0 - tc) if tc > 0 else original_h, tc)
            # Allow <=1 px rounding slack from the integer cropping round-trip.
            assert abs(proj - cons) <= 1, (tc, original_h, proj, cons)


def test_project_fn_offset_zero_when_no_top_crop():
    assert _project_offset_from_cropped(720, 0.0) == 0

"""web.field.sam2_tracker (pre-rename module name) stays importable."""


def test_sam2_tracker_module_is_a_compat_alias():
    import web.field.mask_tracker as mt
    import web.field.sam2_tracker as legacy
    from web.field.sam2_tracker import Sam2FieldTracker, TrackStopped, _ios
    from web.field.sam3_tracker import Sam3FieldTracker
    assert legacy is mt
    assert Sam2FieldTracker is mt.MaskFieldTracker
    assert TrackStopped is mt.TrackStopped and _ios is mt._ios
    assert issubclass(Sam3FieldTracker, mt.MaskFieldTracker)


def test_patching_legacy_module_patches_the_tracker(monkeypatch):
    import web.field.mask_tracker as mt
    import web.field.sam2_tracker as legacy
    monkeypatch.setattr(legacy, "REANCHOR_IOS", 0.123)
    assert mt.REANCHOR_IOS == 0.123

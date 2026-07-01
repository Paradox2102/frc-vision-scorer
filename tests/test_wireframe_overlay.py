"""InferenceJob._maybe_draw_field_wireframe: the per-job field-wireframe overlay.

When the field_wireframe_overlay knob is on, the reprojected field is drawn onto
the main-view frame (which is streamed live AND recorded to the replay) so the
operator can eyeball whether the calibration/results line up with the real field.
The draw is delegated to the shared _draw_field_overlay; here we only test the
gating (knob on/off, calibration present) so it stays cheap and crash-proof.
"""
from types import SimpleNamespace

import web.app as app


def _job(*, enabled, calib=object(), fm=object()):
    return SimpleNamespace(
        config=SimpleNamespace(field_wireframe_overlay=enabled),
        calib=calib, field_model=fm)


def test_wireframe_drawn_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_draw_field_overlay",
                        lambda f, c, m: calls.append((f, c, m)))
    job = _job(enabled=True)
    app.InferenceJob._maybe_draw_field_wireframe(job, "FRAME")
    assert len(calls) == 1
    assert calls[0] == ("FRAME", job.calib, job.field_model)


def test_wireframe_not_drawn_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_draw_field_overlay", lambda *a: calls.append(a))
    app.InferenceJob._maybe_draw_field_wireframe(_job(enabled=False), "FRAME")
    assert calls == []


def test_wireframe_skipped_without_calibration(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_draw_field_overlay", lambda *a: calls.append(a))
    app.InferenceJob._maybe_draw_field_wireframe(_job(enabled=True, calib=None), "FRAME")
    assert calls == []


def test_wireframe_draw_failure_is_swallowed(monkeypatch):
    """A projection blowup in the overlay must never sink inference."""
    def boom(*_a):
        raise ValueError("projection blew up")
    monkeypatch.setattr(app, "_draw_field_overlay", boom)
    # Must not raise.
    app.InferenceJob._maybe_draw_field_wireframe(_job(enabled=True), "FRAME")

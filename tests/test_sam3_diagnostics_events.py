"""SAM3 tracker diagnostics must not be schema-rejected.

Sam3FieldTracker emits events under the "sam3." prefix (DIAG_PREFIX), but the
DiagnosticEvents schema only listed "sam2.*" names — so every SAM3 seeding /
re-prompt / cull decision record was silently counted as a dropped log
(qm10: dropped_logs=201 while we debugged corner seeding blind)."""
from web.field.identity_diagnostics import IdentityDiagnosticRecorder
from web.field.sam2_tracker import Sam2FieldTracker
from web.field.sam3_tracker import Sam3FieldTracker


def test_sam3_prefixed_events_are_accepted():
    rec = IdentityDiagnosticRecorder()
    for suffix in ("object_created", "reprompt_rejected", "object_culled",
                   "mask_gap", "object_reanchored"):
        rec.record(f"{Sam3FieldTracker.DIAG_PREFIX}.{suffix}",
                   details={"object_id": 1})
    summary = rec.summary() if hasattr(rec, "summary") else None
    assert rec._dropped_logs == 0, (
        f"sam3.* events were schema-rejected ({rec._dropped_logs} dropped)")
    assert len(rec._events) == 5


def test_every_sam2_event_has_a_sam3_mirror():
    """Any event the shared tracker code can emit under either prefix must be
    valid under both, so a new sam2.* addition can't silently break sam3."""
    from web.field.identity_diagnostics import ALL_EVENTS
    sam2 = {e.split(".", 1)[1] for e in ALL_EVENTS if e.startswith("sam2.")}
    sam3 = {e.split(".", 1)[1] for e in ALL_EVENTS if e.startswith("sam3.")}
    assert sam2 == sam3, f"unmirrored events: {sam2 ^ sam3}"

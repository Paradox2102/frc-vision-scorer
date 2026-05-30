"""Regression: a wrong confirmed lock must not deadlock the re-read audit.

In SP2, a team identity is locked to a field-tracker slot, and locked teams are
removed from the assignable pool (reid.assign_teams_alliance_masked). So if a
slot wrongly confirms team T, the *real* robot for T can never be assigned T and
stays perpetually unconfirmed. That means `unconfirmed_count` can never reach 0.

If re-reading confirmed slots is gated solely on `unconfirmed_count == 0`, the
bad lock is never re-read, never contradicted, and never overturned: the early
(wrong) crop "sticks around" for the whole match. Re-reads must therefore also
fire on a fixed period regardless of how many slots remain unconfirmed.
"""
from web.app import InferenceJob, REREAD_EVERY_N_PASSES


def _job():
    # Bypass the heavy __init__ (model loading); _is_reread_pass only needs
    # self._claude_vision_calls.
    return InferenceJob.__new__(InferenceJob)


def test_confirmed_slots_are_reread_even_while_a_slot_stays_unconfirmed():
    job = _job()
    rereads = []
    # A wrong lock guarantees >=1 perpetually-unconfirmed slot.
    for call in range(1, REREAD_EVERY_N_PASSES * 3 + 1):
        job._claude_vision_calls = call
        rereads.append(job._is_reread_pass(unconfirmed_count=1))
    assert any(rereads), (
        "confirmed slots are never re-read while any slot stays unconfirmed — "
        "a wrong lock deadlocks forever"
    )


def test_periodic_audit_recurs_so_a_bad_lock_eventually_self_corrects():
    job = _job()
    # Across a long match the audit must keep recurring, not fire once.
    hits = 0
    for call in range(1, REREAD_EVERY_N_PASSES * 10 + 1):
        job._claude_vision_calls = call
        if job._is_reread_pass(unconfirmed_count=2):
            hits += 1
    assert hits >= 5


def test_all_confirmed_always_rereads():
    job = _job()
    job._claude_vision_calls = 1  # not on the periodic boundary
    assert job._is_reread_pass(unconfirmed_count=0) is True

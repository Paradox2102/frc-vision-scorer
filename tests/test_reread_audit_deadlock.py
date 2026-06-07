"""Regression: a wrong confirmed lock must not deadlock the re-read audit.

In the new confidence-driven scheduler, we schedule VLM audits based on match clock
seconds. This ensures that even if there are unconfirmed identities (meaning the
reread isn't gated on unconfirmed_count == 0), confirmed identities will still be
scheduled for periodic audits and won't be starved.
"""
from web.field.identity_confidence import ConfidenceConfig, decide_vlm_read


def test_confirmed_slots_are_reread_even_while_a_slot_stays_unconfirmed():
    config = ConfidenceConfig(audit_floor_seconds=30.0)
    
    # Across time, a confirmed robot should become due for audit
    # even when there are unconfirmed robots.
    due_at_some_point = False
    for t in range(0, 100):
        decision = decide_vlm_read(
            is_confirmed=True,
            confidence=0.9,
            last_vlm_time=0.0,
            last_success_time=0.0,
            backoff_until=0.0,
            needs_immediate_read=False,
            current_match_time=float(t),
            unconfirmed_count=1,
            config=config
        )
        if decision.due:
            due_at_some_point = True
            break
            
    assert due_at_some_point, "confirmed slots are never audited while unconfirmed slots exist"


def test_periodic_audit_recurs_so_a_bad_lock_eventually_self_corrects():
    config = ConfidenceConfig(audit_floor_seconds=30.0)
    
    # Across a long match, if we update the last success time,
    # it should become due again after another interval.
    last_success = 0.0
    audit_count = 0
    for t in range(0, 200):
        decision = decide_vlm_read(
            is_confirmed=True,
            confidence=0.9,
            last_vlm_time=last_success,
            last_success_time=last_success,
            backoff_until=0.0,
            needs_immediate_read=False,
            current_match_time=float(t),
            unconfirmed_count=2,
            config=config
        )
        if decision.due:
            audit_count += 1
            last_success = float(t)
            
    assert audit_count >= 5, f"periodic audit did not recur enough times (got {audit_count})"

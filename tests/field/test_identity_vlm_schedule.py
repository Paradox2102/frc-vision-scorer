import pytest
from web.field.identity_confidence import ConfidenceConfig, decide_vlm_read

def test_decide_vlm_read_immediate_read():
    decision = decide_vlm_read(
        is_confirmed=True,
        confidence=0.9,
        last_vlm_time=10.0,
        last_success_time=10.0,
        backoff_until=50.0,
        needs_immediate_read=True,
        current_match_time=20.0,
        unconfirmed_count=0
    )
    assert decision.due is True
    assert decision.reason == "immediate_read_requested"
    assert decision.priority == 10.0

def test_decide_vlm_read_backed_off():
    # If backed off, shouldn't schedule even if elapsed time is long
    decision = decide_vlm_read(
        is_confirmed=False,
        confidence=0.5,
        last_vlm_time=10.0,
        last_success_time=10.0,
        backoff_until=30.0,
        needs_immediate_read=False,
        current_match_time=25.0,
        unconfirmed_count=1
    )
    assert decision.due is False
    assert decision.reason == "backed_off"

def test_decide_vlm_read_confirmed_audit():
    config = ConfidenceConfig(audit_floor_seconds=30.0)
    
    # Audit not due yet
    decision1 = decide_vlm_read(
        is_confirmed=True,
        confidence=0.9,
        last_vlm_time=10.0,
        last_success_time=10.0,
        backoff_until=0.0,
        needs_immediate_read=False,
        current_match_time=35.0,
        unconfirmed_count=0,
        config=config
    )
    assert decision1.due is False
    assert decision1.reason == "audit_not_due"

    # Audit due
    decision2 = decide_vlm_read(
        is_confirmed=True,
        confidence=0.9,
        last_vlm_time=10.0,
        last_success_time=10.0,
        backoff_until=0.0,
        needs_immediate_read=False,
        current_match_time=45.0,
        unconfirmed_count=0,
        config=config
    )
    assert decision2.due is True
    assert decision2.reason == "audit_due"
    assert decision2.priority >= 1.0

def test_decide_vlm_read_unconfirmed_retry():
    config = ConfidenceConfig(
        unconfirmed_fast_interval_sec=0.75,
        unconfirmed_med_interval_sec=1.25,
        unconfirmed_slow_interval_sec=2.0
    )

    # Fast retry due (low confidence)
    decision_fast = decide_vlm_read(
        is_confirmed=False,
        confidence=0.3,
        last_vlm_time=10.0,
        last_success_time=10.0,
        backoff_until=0.0,
        needs_immediate_read=False,
        current_match_time=10.8,
        unconfirmed_count=1,
        config=config
    )
    assert decision_fast.due is True
    assert decision_fast.reason == "unconfirmed_retry"
    assert decision_fast.priority >= 5.0

    # Med retry not due (medium confidence, elapsed 1.0s < 1.25s)
    decision_med = decide_vlm_read(
        is_confirmed=False,
        confidence=0.5,
        last_vlm_time=10.0,
        last_success_time=10.0,
        backoff_until=0.0,
        needs_immediate_read=False,
        current_match_time=11.0,
        unconfirmed_count=1,
        config=config
    )
    assert decision_med.due is False

    # Slow retry due (high confidence but unconfirmed, elapsed 2.1s >= 2.0s)
    decision_slow = decide_vlm_read(
        is_confirmed=False,
        confidence=0.8,
        last_vlm_time=10.0,
        last_success_time=10.0,
        backoff_until=0.0,
        needs_immediate_read=False,
        current_match_time=12.1,
        unconfirmed_count=1,
        config=config
    )
    assert decision_slow.due is True
    assert decision_slow.reason == "unconfirmed_retry"
    assert decision_slow.priority >= 3.0

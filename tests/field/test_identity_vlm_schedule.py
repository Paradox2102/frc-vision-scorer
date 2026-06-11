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


# ---------------------------------------------------------------------------
# Novelty gate: an unconfirmed identity whose last read FAILED and which has
# not moved since is showing us the same unreadable view — skip the retry
# (bounded: after novelty_skip_max_sec it reads anyway, so a parked robot is
# never frozen out — see the re-read-audit-deadlock lesson).
# ---------------------------------------------------------------------------

def _novelty_kwargs(**over):
    base = dict(
        is_confirmed=False, confidence=0.5,
        last_vlm_time=10.0, last_success_time=5.0,   # failed after success
        backoff_until=-999.0, needs_immediate_read=False,
        current_match_time=13.0, unconfirmed_count=2,
        last_read_failed=True, moved_since_read=False,
    )
    base.update(over)
    return base


def test_novelty_gate_skips_unmoved_failed_retry():
    from web.field.identity_confidence import decide_vlm_read
    d = decide_vlm_read(**_novelty_kwargs())
    assert d.due is False
    assert d.reason == "no_new_view"


def test_novelty_gate_allows_after_movement():
    from web.field.identity_confidence import decide_vlm_read
    d = decide_vlm_read(**_novelty_kwargs(moved_since_read=True))
    assert d.due is True
    assert d.reason == "unconfirmed_retry"


def test_novelty_gate_allows_after_successful_read():
    from web.field.identity_confidence import decide_vlm_read
    d = decide_vlm_read(**_novelty_kwargs(last_read_failed=False))
    assert d.due is True


def test_novelty_gate_expires_after_max_seconds():
    """Periodic escape hatch: even a parked, unreadable robot is re-read
    eventually (never gate forever — wrong assumptions must stay falsifiable)."""
    from web.field.identity_confidence import decide_vlm_read, ConfidenceConfig
    cfg = ConfidenceConfig()
    d = decide_vlm_read(**_novelty_kwargs(
        current_match_time=10.0 + cfg.novelty_skip_max_sec + 0.1))
    assert d.due is True


def test_novelty_gate_never_blocks_immediate_or_audit():
    from web.field.identity_confidence import decide_vlm_read
    d = decide_vlm_read(**_novelty_kwargs(needs_immediate_read=True))
    assert d.due is True and d.reason == "immediate_read_requested"
    d = decide_vlm_read(**_novelty_kwargs(
        is_confirmed=True, last_success_time=-50.0))
    assert d.due is True and d.reason == "audit_due"


def test_novelty_gate_disabled_by_config():
    from web.field.identity_confidence import decide_vlm_read, ConfidenceConfig
    cfg = ConfidenceConfig(novelty_gate_enabled=False)
    d = decide_vlm_read(**_novelty_kwargs(), config=cfg)
    assert d.due is True


# ---------------------------------------------------------------------------
# Occlusion gate helper: never schedule a read whose bumper is occluded in
# every camera — at best a wasted call, at worst a garbage crop whose
# contradicting read drops confidence on a CORRECT lock.
# ---------------------------------------------------------------------------

def test_occlusion_blocks_read_on_bumper_occluded():
    from web.field.identity_confidence import occlusion_blocks_read
    assert occlusion_blocks_read(
        {"view_occluded": False, "bumper_occluded": True, "cause": "hub"}) is True


def test_occlusion_allows_read_when_bumper_visible_somewhere():
    from web.field.identity_confidence import occlusion_blocks_read
    assert occlusion_blocks_read(
        {"view_occluded": False, "bumper_occluded": False, "cause": None}) is False


def test_occlusion_allows_read_when_unknown():
    from web.field.identity_confidence import occlusion_blocks_read
    assert occlusion_blocks_read(None) is False
    assert occlusion_blocks_read({}) is False


# ---------------------------------------------------------------------------
# Logprob confidence helpers (pure parts of VLM improvement #4).
# ---------------------------------------------------------------------------

def test_digit_token_conf_min_over_digit_tokens():
    from web.field.identity_confidence import digit_token_conf
    # JSON scaffolding tokens are high-prob; digits carry the uncertainty.
    probs = [0.99, 0.98, 0.95, 0.60, 0.97, 0.90]
    texts = ['{"', 'team', '":', '417', '4', '"}']
    assert digit_token_conf(probs, texts) == 0.60


def test_digit_token_conf_falls_back_to_mean_without_digits():
    from web.field.identity_confidence import digit_token_conf
    probs = [0.8, 0.6]
    texts = ["null", "}"]
    assert abs(digit_token_conf(probs, texts) - 0.7) < 1e-9


def test_digit_token_conf_empty_returns_none():
    from web.field.identity_confidence import digit_token_conf
    assert digit_token_conf([], []) is None


def test_blend_confidence_sqrt_damps_shaky_reads():
    from web.field.identity_confidence import blend_read_confidence
    # solid digits keep an exact roster read above the 0.85 full-vote bar
    assert blend_read_confidence(0.95, 0.9) >= 0.85
    # shaky digits demote it below the bar
    assert blend_read_confidence(0.95, 0.5) < 0.85
    # missing digit_prob leaves base unchanged
    assert blend_read_confidence(0.95, None) == 0.95

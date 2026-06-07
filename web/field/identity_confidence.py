from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ConfidenceConfig:
    decay_rate: float = 0.005          # decay per frame when occluded/coasting
    observation_rise: float = 0.02     # rise per frame when observed confidently
    low_margin_drop: float = 0.05      # drop when matched with low margin
    vlm_reinforce: float = 0.15        # rise when VLM read reinforces
    vlm_contradict: float = 0.30       # drop when VLM contradicts
    initial_confidence: float = 0.5    # initial confidence for new identities
    confirmed_floor: float = 0.85      # confidence floor once confirmed
    hysteresis_rise: float = 0.80      # threshold to rise to high confidence
    hysteresis_fall: float = 0.40      # threshold to fall to low confidence
    audit_floor_seconds: float = 30.0  # min seconds between audits for confirmed identity
    unconfirmed_fast_interval_sec: float = 0.75
    unconfirmed_med_interval_sec: float = 1.25
    unconfirmed_slow_interval_sec: float = 2.0
    backoff_base_sec: float = 2.0

@dataclass
class ConfidenceSignals:
    is_observed: bool = False
    relink_margin: float = 1.0
    has_vlm_read: bool = False
    vlm_agrees: bool = False
    vlm_contradicts: bool = False
    is_confirmed: bool = False
    roster_uniqueness_ok: bool = True

@dataclass
class ConfidenceState:
    confidence: float
    is_confirmed: bool

@dataclass
class ConfidenceUpdate:
    new_confidence: float
    reasons: List[str]

@dataclass
class ReadDecision:
    due: bool
    reason: str
    priority: float
    is_audit: bool

def calculate_confidence_update(
    state: ConfidenceState,
    signals: ConfidenceSignals,
    config: Optional[ConfidenceConfig] = None
) -> ConfidenceUpdate:
    if config is None:
        config = ConfidenceConfig()
        
    conf = state.confidence
    reasons = []
    
    if signals.is_confirmed:
        conf = max(conf, config.confirmed_floor)
        reasons.append("confirmed_status_floor")
        
    if signals.is_observed:
        if signals.relink_margin >= 0.1:
            conf += config.observation_rise
            reasons.append("confident_observation")
        else:
            conf -= config.low_margin_drop
            reasons.append("low_margin_drop")
    else:
        conf -= config.decay_rate
        reasons.append("occlusion_decay")
        
    if signals.has_vlm_read:
        if signals.vlm_agrees:
            conf += config.vlm_reinforce
            reasons.append("vlm_agreement")
        elif signals.vlm_contradicts:
            conf -= config.vlm_contradict
            reasons.append("vlm_contradiction")
            
    if not signals.roster_uniqueness_ok:
        conf -= 0.2
        reasons.append("roster_conflict")
        
    # Clamp confidence
    conf = max(0.0, min(1.0, conf))
    
    # If confirmed, maintain confirmed floor unless heavily contradicted
    if signals.is_confirmed and not signals.vlm_contradicts:
        conf = max(conf, config.confirmed_floor)
        
    return ConfidenceUpdate(new_confidence=float(conf), reasons=reasons)

def decide_vlm_read(
    is_confirmed: bool,
    confidence: float,
    last_vlm_time: float,
    last_success_time: float,
    backoff_until: float,
    needs_immediate_read: bool,
    current_match_time: float,
    unconfirmed_count: int,
    config: Optional[ConfidenceConfig] = None
) -> ReadDecision:
    if config is None:
        config = ConfidenceConfig()

    if needs_immediate_read:
        return ReadDecision(due=True, reason="immediate_read_requested", priority=10.0, is_audit=False)

    if current_match_time < backoff_until:
        return ReadDecision(due=False, reason="backed_off", priority=0.0, is_audit=is_confirmed)

    if is_confirmed:
        elapsed = current_match_time - last_success_time
        if elapsed >= config.audit_floor_seconds:
            priority = 1.0 + (elapsed - config.audit_floor_seconds) * 0.01
            return ReadDecision(due=True, reason="audit_due", priority=priority, is_audit=True)
        else:
            return ReadDecision(due=False, reason="audit_not_due", priority=0.0, is_audit=True)
    else:
        elapsed = current_match_time - last_vlm_time
        # Choose interval based on confidence level or unconfirmed count
        if unconfirmed_count >= 4 or confidence < 0.4:
            interval = config.unconfirmed_fast_interval_sec
            base_priority = 5.0
        elif unconfirmed_count >= 2 or confidence < 0.7:
            interval = config.unconfirmed_med_interval_sec
            base_priority = 4.0
        else:
            interval = config.unconfirmed_slow_interval_sec
            base_priority = 3.0

        if elapsed >= interval:
            priority = base_priority + (elapsed - interval) * 0.1
            return ReadDecision(due=True, reason="unconfirmed_retry", priority=priority, is_audit=False)
        else:
            return ReadDecision(due=False, reason="unconfirmed_not_due", priority=0.0, is_audit=False)


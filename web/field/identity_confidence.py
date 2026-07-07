import os
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
    # Novelty gate: skip an unconfirmed retry when the last read FAILED and
    # the robot hasn't moved since — same pose, same unreadable view, same
    # outcome. Bounded by novelty_skip_max_sec so a parked robot still gets
    # periodic reads (a gate that can hold forever would freeze a wrong
    # assumption — the re-read-audit-deadlock lesson).
    novelty_gate_enabled: bool = field(default_factory=lambda: os.environ.get(
        "VLM_NOVELTY_GATE", "1") not in ("0", "false", "False"))
    novelty_min_move_m: float = field(default_factory=lambda: float(
        os.environ.get("VLM_NOVELTY_MIN_MOVE_M", "0.5")))
    novelty_skip_max_sec: float = field(default_factory=lambda: float(
        os.environ.get("VLM_NOVELTY_SKIP_MAX_SEC", "8.0")))
    # Immediate-read escalations gain nothing within a second of the last
    # read/queue — the unconfirmed retry cadence already covers that window.
    immediate_read_cooldown_sec: float = field(default_factory=lambda: float(
        os.environ.get("VLM_IMMEDIATE_COOLDOWN_SEC", "1.0")))

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
    config: Optional[ConfidenceConfig] = None,
    last_read_failed: bool = False,
    moved_since_read: bool = True,
) -> ReadDecision:
    if config is None:
        config = ConfidenceConfig()

    if needs_immediate_read:
        return ReadDecision(due=True, reason="immediate_read_requested", priority=10.0, is_audit=False)

    if current_match_time < backoff_until:
        return ReadDecision(due=False, reason="backed_off", priority=0.0, is_audit=is_confirmed)

    # Novelty gate (unconfirmed retries only — audits and immediate reads are
    # exempt): a failed read of an unmoved robot will fail again; wait for a
    # new view or the periodic escape.
    if (config.novelty_gate_enabled
            and not is_confirmed
            and last_read_failed
            and not moved_since_read
            and (current_match_time - last_vlm_time) < config.novelty_skip_max_sec):
        return ReadDecision(due=False, reason="no_new_view", priority=0.0, is_audit=False)

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


def should_queue_immediate_read(
    current_match_time: float,
    last_vlm_time: float,
    last_immediate_queue_time: float,
    cooldown_sec: float,
) -> bool:
    """Rate-limit priority-10 immediate-read escalations per identity.

    The escalation triggers (re-attachment, low margin, appearance
    divergence, roster conflict) re-fire every frame their condition
    holds; an identity read or queued within the cooldown gains nothing
    from another escalation. Sentinel -999 times fail open so a fresh
    identity's first re-attachment still reads immediately.
    """
    if (current_match_time - last_vlm_time) < cooldown_sec:
        return False
    if (current_match_time - last_immediate_queue_time) < cooldown_sec:
        return False
    return True


def occlusion_blocks_read(occ_entry) -> bool:
    """True when a slot's merged occlusion entry says its bumper is hidden in
    EVERY camera. Reading it is at best a wasted call; at worst the garbage
    crop yields a contradicting read that drops confidence on a correct lock.
    Missing/empty entries fail open (no occlusion info -> allow the read)."""
    if not occ_entry:
        return False
    return bool(occ_entry.get("bumper_occluded"))


def digit_token_conf(token_probs, token_texts):
    """Model certainty of a VLM bumper read, from per-token probabilities.

    The JSON scaffolding tokens are near-deterministic; the team digits carry
    the real uncertainty, so the MIN probability over digit-bearing tokens is
    the read's weakest link. Falls back to the mean over all tokens when the
    answer contains no digits ("team": null), and None for empty output.
    """
    if not token_probs:
        return None
    digit_probs = [p for p, t in zip(token_probs, token_texts)
                   if any(ch.isdigit() for ch in (t or ""))]
    if digit_probs:
        return float(min(digit_probs))
    return float(sum(token_probs) / len(token_probs))


def blend_read_confidence(base_conf: float, digit_prob) -> float:
    """Fold the model's own digit certainty into the roster-match confidence.

    sqrt damping: a read needs digit_prob >= ~0.8 for an exact roster match
    (base 0.95) to keep its full-weight vote (the 0.85 bar in app.py); a
    50/50 read drops to half weight. None (logprobs unavailable) leaves the
    base unchanged so the feature degrades gracefully.
    """
    if digit_prob is None:
        return base_conf
    return round(base_conf * (max(0.0, min(1.0, float(digit_prob))) ** 0.5), 3)



def dedupe_batch_team_reads(reads):
    """Allow at most ONE vote per team value within a single VLM batch.

    ``reads`` is ``[(team, conf, source_weight), ...]`` in batch order; the
    return is a parallel list of booleans (True = this read may cast its
    vote). One physical robot photographed by two crops in the same batch
    (or its number cross-read from an adjacent crop) otherwise votes the
    same team into two different slots simultaneously — the qm11 duplicate
    "2992?" label. The strongest read (conf x weight, first on ties) keeps
    the vote; ``None`` teams are never blocked (they cast no vote anyway).
    """
    best_idx = {}
    for i, (team, conf, weight) in enumerate(reads):
        if team is None:
            continue
        score = float(conf) * float(weight)
        if team not in best_idx or score > best_idx[team][1]:
            best_idx[team] = (i, score)
    winners = {i for i, _s in best_idx.values()}
    return [team is None or i in winners for i, (team, _c, _w) in enumerate(reads)]


def tentative_claim_winners(vote_maps):
    """For each team, the single robot allowed to display it as a tentative
    "TEAM?" claim: the one with the highest accumulated vote (ties break to
    the first key in sorted order, for stable display).

    ``vote_maps`` is ``{robot_id: {team: vote_total}}``. Accumulated votes
    across batches can put the same team's top vote on two robots (qm11: two
    slots both showed "2992?"); the HUD consults this map so only the
    strongest claimant renders the label.
    """
    winners = {}
    for rid in sorted(vote_maps):
        for team, total in (vote_maps[rid] or {}).items():
            cur = winners.get(team)
            if cur is None or total > cur[1]:
                winners[team] = (rid, total)
    return {team: rid for team, (rid, _t) in winners.items()}


def vote_ray_cross_m(cam_center_xy, cam_det_pos, fused_pos):
    """Perpendicular (cross-ray) distance in meters of ``fused_pos`` from the
    ground ray ``cam_center_xy`` -> ``cam_det_pos``.

    Bearing is the trustworthy component of every camera's projection (range
    is not, especially for low corner cameras — see fusion._ray_gate_score).
    A bumper crop is misattributed when the slot position it votes into does
    not sit near the ray its own camera saw it on. Returns None when the
    camera and detection coincide (no defined ray)."""
    cx, cy = float(cam_center_xy[0]), float(cam_center_xy[1])
    dx, dy = float(cam_det_pos[0]) - cx, float(cam_det_pos[1]) - cy
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return None
    ux, uy = dx / n, dy / n
    rx, ry = float(fused_pos[0]) - cx, float(fused_pos[1]) - cy
    return abs(rx * uy - ry * ux)


def best_owned_team(votes, robot_id, winners):
    """The robot's highest-voted team among those it actually OWNS per
    ``tentative_claim_winners`` — ``(team, vote)`` or ``(None, 0.0)``."""
    best = (None, 0.0)
    for team, total in (votes or {}).items():
        if winners.get(team) == robot_id and total > best[1]:
            best = (team, total)
    return best

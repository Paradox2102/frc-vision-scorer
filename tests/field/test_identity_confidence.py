import pytest
from web.field.identity_confidence import (
    ConfidenceConfig,
    ConfidenceSignals,
    ConfidenceState,
    calculate_confidence_update,
    vote_ray_cross_m,
)

def test_decay_on_occlusion():
    config = ConfidenceConfig(decay_rate=0.01)
    state = ConfidenceState(confidence=0.5, is_confirmed=False)
    signals = ConfidenceSignals(is_observed=False)
    
    update = calculate_confidence_update(state, signals, config)
    assert update.new_confidence == 0.49
    assert "occlusion_decay" in update.reasons

def test_gradual_rising():
    config = ConfidenceConfig(observation_rise=0.02)
    state = ConfidenceState(confidence=0.5, is_confirmed=False)
    signals = ConfidenceSignals(is_observed=True, relink_margin=0.5)
    
    update = calculate_confidence_update(state, signals, config)
    assert update.new_confidence == 0.52
    assert "confident_observation" in update.reasons

def test_low_margin_drop():
    config = ConfidenceConfig(low_margin_drop=0.05)
    state = ConfidenceState(confidence=0.5, is_confirmed=False)
    signals = ConfidenceSignals(is_observed=True, relink_margin=0.02)
    
    update = calculate_confidence_update(state, signals, config)
    assert update.new_confidence == 0.45
    assert "low_margin_drop" in update.reasons

def test_vlm_reinforce():
    config = ConfidenceConfig(vlm_reinforce=0.15)
    state = ConfidenceState(confidence=0.5, is_confirmed=False)
    signals = ConfidenceSignals(is_observed=True, has_vlm_read=True, vlm_agrees=True)
    
    update = calculate_confidence_update(state, signals, config)
    assert update.new_confidence == 0.5 + config.observation_rise + 0.15

def test_vlm_contradiction():
    config = ConfidenceConfig(vlm_contradict=0.30)
    state = ConfidenceState(confidence=0.85, is_confirmed=True)
    signals = ConfidenceSignals(is_observed=True, has_vlm_read=True, vlm_contradicts=True, is_confirmed=True)
    
    update = calculate_confidence_update(state, signals, config)
    # 0.85 + 0.02 (obs) - 0.30 (contradict) = 0.57
    assert update.new_confidence == pytest.approx(0.57)

def test_roster_conflict():
    state = ConfidenceState(confidence=0.6, is_confirmed=False)
    signals = ConfidenceSignals(is_observed=True, roster_uniqueness_ok=False)
    
    update = calculate_confidence_update(state, signals)
    assert update.new_confidence == pytest.approx(0.42) # 0.6 + 0.02 - 0.2

def test_confirmed_floor():
    config = ConfidenceConfig(confirmed_floor=0.85)
    state = ConfidenceState(confidence=0.5, is_confirmed=True)
    signals = ConfidenceSignals(is_observed=True, is_confirmed=True)

    update = calculate_confidence_update(state, signals, config)
    assert update.new_confidence == 0.87 # max(0.5, 0.85) + 0.02


def test_vote_ray_cross_on_ray_is_zero():
    # camera at origin, detection at (4,0): fused pos on the same ray -> ~0
    assert vote_ray_cross_m((0, 0), (4.0, 0.0), (6.0, 0.0)) < 1e-9


def test_vote_ray_cross_lateral_miss():
    # fused position 2m laterally off the camera->det ray
    assert abs(vote_ray_cross_m((0, 0), (4.0, 0.0), (4.0, 2.0)) - 2.0) < 1e-9


def test_vote_ray_cross_degenerate_returns_none():
    assert vote_ray_cross_m((3.0, 1.0), (3.0, 1.0), (5.0, 5.0)) is None

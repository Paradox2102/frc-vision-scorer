"""Immediate-read escalations must be rate-limited per identity.

The four needs_immediate_read triggers (re-attachment, low margin,
appearance divergence, roster conflict) fire on every frame their
condition holds, so a churning identity re-escalates to priority 10
moments after a read just completed — ~570 escalations per auto job.
An identity read within the cooldown gains nothing from another
priority-10 read: the unconfirmed retry cadence (0.75-2s) already
covers it. The gate must fail open for identities never read or never
queued (sentinel -999) so a genuine first re-attachment still reads
immediately.
"""
from web.field.identity import RobotIdentity
from web.field.identity_confidence import (
    ConfidenceConfig,
    should_queue_immediate_read,
)


def test_fresh_identity_queues_immediately():
    assert should_queue_immediate_read(
        current_match_time=30.0, last_vlm_time=-999.0,
        last_immediate_queue_time=-999.0, cooldown_sec=1.0)


def test_recent_read_suppresses_escalation():
    assert not should_queue_immediate_read(
        current_match_time=30.0, last_vlm_time=29.7,
        last_immediate_queue_time=-999.0, cooldown_sec=1.0)


def test_stale_read_allows_escalation():
    assert should_queue_immediate_read(
        current_match_time=30.0, last_vlm_time=28.5,
        last_immediate_queue_time=-999.0, cooldown_sec=1.0)


def test_recent_queue_suppresses_requeue():
    assert not should_queue_immediate_read(
        current_match_time=30.0, last_vlm_time=20.0,
        last_immediate_queue_time=29.5, cooldown_sec=1.0)


def test_zero_cooldown_disables_gate():
    assert should_queue_immediate_read(
        current_match_time=30.0, last_vlm_time=30.0,
        last_immediate_queue_time=30.0, cooldown_sec=0.0)


def test_config_default_cooldown_is_one_second():
    assert ConfidenceConfig().immediate_read_cooldown_sec == 1.0


def test_identity_tracks_last_immediate_queue_time():
    ident = RobotIdentity(identity_id=1, alliance="blue")
    assert ident.last_immediate_queue_time == -999.0

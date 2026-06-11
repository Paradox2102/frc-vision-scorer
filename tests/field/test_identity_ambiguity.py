import pytest
from web.field.identity_ambiguity import (
    AmbiguityResolution,
    AssignmentCandidate,
    CoOcclusionEvent,
    CoOcclusionTracker,
    resolve_group_assignment,
)

def test_co_occlusion_grouping_same_alliance():
    tracker = CoOcclusionTracker(re_emergence_window=30)
    
    # Mock identities going inactive (occluded)
    # Blue alliance identities 0 and 1 go inactive at frame 10
    tracker.record_occlusion(frame_idx=10, identity_id=0, alliance="blue")
    tracker.record_occlusion(frame_idx=12, identity_id=1, alliance="blue")
    
    # Red alliance identity 2 goes inactive at frame 11 (should not be grouped with blue)
    tracker.record_occlusion(frame_idx=11, identity_id=2, alliance="red")
    
    # Verify events created
    active_events = tracker.get_active_events()
    assert len(active_events) == 2 # One blue event, one red event
    
    blue_event = next(e for e in active_events if e.alliance == "blue")
    assert blue_event.identity_ids == {0, 1}
    
    red_event = next(e for e in active_events if e.alliance == "red")
    assert red_event.identity_ids == {2}

def test_re_emergence_window_expiry():
    tracker = CoOcclusionTracker(re_emergence_window=30)
    tracker.record_occlusion(frame_idx=10, identity_id=0, alliance="blue")
    tracker.record_occlusion(frame_idx=12, identity_id=1, alliance="blue")
    
    # Reappearance happens at frame 50 (> 30 frames window)
    event = tracker.handle_reemergence(frame_idx=50, identity_id=0, slot_id=2, margin=0.1)
    # Event should be expired/ignored due to window
    assert event is None

def test_margin_check_ambiguous_vs_decisive():
    tracker = CoOcclusionTracker(re_emergence_window=30)
    tracker.record_occlusion(frame_idx=10, identity_id=0, alliance="blue")
    tracker.record_occlusion(frame_idx=12, identity_id=1, alliance="blue")
    
    # Reappearance with high margin (decisive) -> does not trigger ambiguity
    event_high = tracker.handle_reemergence(frame_idx=20, identity_id=0, slot_id=2, margin=0.8)
    assert event_high is None
    
    # Reappearance with low margin (ambiguous) -> triggers ambiguity
    event_low = tracker.handle_reemergence(frame_idx=22, identity_id=1, slot_id=3, margin=0.15)
    assert event_low is not None
    assert event_low.identity_ids == {0, 1}

def test_resolution_lifecycle():
    tracker = CoOcclusionTracker(re_emergence_window=30)
    tracker.record_occlusion(frame_idx=10, identity_id=0, alliance="blue")
    tracker.record_occlusion(frame_idx=12, identity_id=1, alliance="blue")
    
    tracker.handle_reemergence(frame_idx=20, identity_id=0, slot_id=2, margin=0.1)
    event = tracker.handle_reemergence(frame_idx=21, identity_id=1, slot_id=3, margin=0.1)
    
    assert event is not None
    assert event.resolved is False
    
    # Resolve the event with team number permutation mapping from VLM
    tracker.resolve_event(event.event_id, {0: 2, 1: 3})
    
    assert event.resolved is True
    assert event.resolved_mapping == {0: 2, 1: 3}


def test_repeated_occlusion_does_not_duplicate_events():
    tracker = CoOcclusionTracker()
    first = tracker.record_occlusion(10, 0, "blue")
    second = tracker.record_occlusion(11, 0, "blue")
    assert first.event_id == second.event_id
    assert len(tracker.get_active_events()) == 1


def test_single_identity_event_is_not_ambiguous():
    tracker = CoOcclusionTracker()
    tracker.record_occlusion(10, 0, "blue")
    assert tracker.handle_reemergence(12, 0, 4, margin=0.01) is None
    assert tracker.ambiguous_identity_ids() == set()
    assert tracker.get_active_events() == []


def test_decisive_group_reemergence_closes_event():
    tracker = CoOcclusionTracker()
    tracker.record_occlusion(10, 0, "blue")
    tracker.record_occlusion(11, 1, "blue")
    assert tracker.handle_reemergence(12, 0, 4, margin=0.8) is None
    assert tracker.handle_reemergence(13, 1, 5, margin=0.7) is None
    assert tracker.get_active_events() == []


def test_group_assignment_uses_global_margin():
    result = resolve_group_assignment(
        [0, 1],
        [3, 4],
        [
            AssignmentCandidate(0, 3, 0.9),
            AssignmentCandidate(0, 4, 0.1),
            AssignmentCandidate(1, 3, 0.2),
            AssignmentCandidate(1, 4, 0.8),
        ],
    )
    assert result.status == "resolved"
    assert result.mapping == {0: 3, 1: 4}
    assert result.margin > 0.3

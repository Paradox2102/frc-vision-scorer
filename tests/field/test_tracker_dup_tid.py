"""Duplicate same-tid detections must not ping-pong slot bindings.

When the fused registry mis-assigns one track_id to two distinct robots,
merge_main_guard deliberately passes both detections through so each keeps its
own slot. But the sticky pass bound the first to the id's mapped slot and the
distance pass then OVERWROTE _id_to_slot with the second robot's slot — so the
sticky target flip-flopped every frame (qm11: tid=4 bounced across R1/R2/R3,
563+ rebinds). A tid that appears on 2+ detections in one frame is ambiguous:
no sticky binding, no mapping updates; distance matching keeps continuity."""
from web.field.tracker import FieldTracker, FieldDetection


def _red(x, y, tid=None):
    return FieldDetection((x, y), "red", (0, 0, 10, 10), track_id=tid)


def _seed_two(tr):
    outs = tr.update([_red(11.6, 6.6, tid=4), _red(14.9, 2.4, tid=2)], {}, 0.1)
    assert len(outs) == 2
    a = tr.last_det_to_slot[0]   # robot A slot (seeded by tid=4)
    b = tr.last_det_to_slot[1]   # robot B slot (seeded by tid=2)
    return a, b


def test_duplicate_tid_frames_keep_stable_slot_assignment():
    tr = FieldTracker(16.541)
    a, b = _seed_two(tr)
    # Registry collapses: BOTH robots now report tid=4, several frames running.
    for _ in range(6):
        tr.update([_red(11.6, 6.6, tid=4), _red(14.9, 2.4, tid=4)], {}, 0.1)
        assert tr.last_det_to_slot == {0: a, 1: b}, (
            f"dup-tid frame flipped bindings: {tr.last_det_to_slot}")


def test_ambiguous_tid_does_not_poison_mapping_for_later_frames():
    tr = FieldTracker(16.541)
    a, b = _seed_two(tr)
    for _ in range(3):
        tr.update([_red(11.6, 6.6, tid=4), _red(14.9, 2.4, tid=4)], {}, 0.1)
    # Collapse resolves: robot B gets a fresh id, tid=4 stays on robot A.
    tr.update([_red(11.6, 6.6, tid=4), _red(14.9, 2.4, tid=9)], {}, 0.1)
    assert tr.last_det_to_slot == {0: a, 1: b}
    # tid=4's sticky mapping still points at robot A's slot.
    tr.update([_red(11.8, 6.6, tid=4)], {}, 0.1)
    assert tr.last_det_to_slot == {0: a}


def test_single_tid_sticky_behavior_unchanged():
    """No duplicates -> sticky binding works exactly as before."""
    tr = FieldTracker(16.541)
    a, b = _seed_two(tr)
    tr.update([_red(12.0, 6.4, tid=4), _red(14.7, 2.6, tid=2)], {}, 0.1)
    assert tr.last_det_to_slot == {0: a, 1: b}

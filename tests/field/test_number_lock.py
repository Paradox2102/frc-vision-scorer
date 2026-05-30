from web.field.number_lock import ConsecutiveValueTracker


def test_lock_after_n_consecutive():
    tracker = ConsecutiveValueTracker(n_consecutive=3)
    assert tracker.locked_value is None
    assert tracker.confirmed is False

    # First read
    val, conf = tracker.update("2102")
    assert val is None
    assert conf is False

    # Second read (consistent)
    val, conf = tracker.update("2102")
    assert val is None
    assert conf is False

    # Third read (consistent -> locks!)
    val, conf = tracker.update("2102")
    assert val == "2102"
    assert conf is True

    # Reads after lock are ignored or return the locked value
    val, conf = tracker.update("2102")
    assert val == "2102"
    assert conf is True


def test_reset_on_disagreement():
    tracker = ConsecutiveValueTracker(n_consecutive=3)

    # Two reads of 2102
    tracker.update("2102")
    tracker.update("2102")
    assert tracker.consecutive_count == 2

    # Disagreement read: resets count to 1 for the new value
    val, conf = tracker.update("3128")
    assert val is None
    assert conf is False
    assert tracker.current_value == "3128"
    assert tracker.consecutive_count == 1

    # Keep reading 3128
    tracker.update("3128")
    val, conf = tracker.update("3128")
    assert val == "3128"
    assert conf is True


def test_none_ignored():
    tracker = ConsecutiveValueTracker(n_consecutive=3)

    tracker.update("2102")
    assert tracker.consecutive_count == 1

    # None read does not reset or increment
    val, conf = tracker.update(None)
    assert val is None
    assert conf is False
    assert tracker.consecutive_count == 1
    assert tracker.current_value == "2102"

    # Empty string read also ignored
    val, conf = tracker.update(" ")
    assert val is None
    assert conf is False
    assert tracker.consecutive_count == 1
    assert tracker.current_value == "2102"

    # Second read
    tracker.update("2102")
    # Third read -> locks
    val, conf = tracker.update("2102")
    assert val == "2102"
    assert conf is True


def test_unlock_on_contradiction():
    tracker = ConsecutiveValueTracker(n_consecutive=3)

    tracker.update("2102")
    tracker.update("2102")
    tracker.update("2102")
    assert tracker.confirmed is True
    assert tracker.locked_value == "2102"

    # Direct contradiction unlock
    tracker.unlock()
    assert tracker.confirmed is False
    assert tracker.locked_value is None
    assert tracker.current_value is None
    assert tracker.consecutive_count == 0

    # Start tracking new team after unlock
    tracker.update("3328")
    assert tracker.current_value == "3328"
    assert tracker.consecutive_count == 1

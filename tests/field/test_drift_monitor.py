from web.field.drift_monitor import DriftMonitor


def test_due_respects_interval():
    m = DriftMonitor(interval_s=20.0, max_m=0.5)
    assert m.due(0.0)
    m.record(0.0, 0.1, 4)
    assert not m.due(10.0)
    assert m.due(20.1)


def test_drifted_needs_sustained_window():
    m = DriftMonitor(interval_s=20.0, max_m=0.5, window=3)
    m.record(0.0, 0.9, 4)
    m.record(20.0, 0.9, 4)
    assert not m.drifted          # only 2 samples
    m.record(40.0, 0.8, 4)
    assert m.drifted


def test_one_bad_sample_does_not_trip():
    m = DriftMonitor(interval_s=20.0, max_m=0.5, window=3)
    m.record(0.0, 0.1, 4)
    m.record(20.0, 0.9, 4)       # single outlier (a robot covering tags)
    m.record(40.0, 0.1, 4)
    assert not m.drifted


def test_low_tag_samples_ignored():
    m = DriftMonitor(interval_s=20.0, max_m=0.5, window=3, min_tags=2)
    for t in (0.0, 20.0, 40.0):
        m.record(t, 5.0, 1)      # unusable: too few tags
    assert not m.drifted

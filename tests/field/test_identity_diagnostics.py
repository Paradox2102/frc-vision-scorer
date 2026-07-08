import json
import numpy as np
import pytest
from web.field.identity_diagnostics import (
    IdentityDiagnosticRecorder,
    DiagnosticEvents,
    DiagnosticReasons,
)


def test_numpy_normalization():
    recorder = IdentityDiagnosticRecorder()
    data = {
        "int_val": np.int32(42),
        "float_val": np.float64(3.14),
        "array_val": np.array([1, 2, 3]),
    }
    normalized = recorder._normalize(data)
    assert isinstance(normalized["int_val"], int)
    assert isinstance(normalized["float_val"], float)
    assert isinstance(normalized["array_val"], list)


def test_schema_validation():
    recorder = IdentityDiagnosticRecorder()
    
    # Valid event and reason
    recorder.record(DiagnosticEvents.IDENTITY_CREATED, reason=DiagnosticReasons.REATTACH_CONFIDENT)
    assert len(recorder._events) == 1
    assert recorder._events[0]["event"] == DiagnosticEvents.IDENTITY_CREATED
    assert recorder._events[0]["reason"] == DiagnosticReasons.REATTACH_CONFIDENT

    # Invalid event
    recorder.record("unknown.event")
    assert recorder._dropped_logs == 1

    # Invalid reason
    recorder.record(DiagnosticEvents.IDENTITY_CREATED, reason="invalid_reason")
    assert recorder._dropped_logs == 2


def test_bounded_ring_buffer_and_dropping(tmp_path):
    recorder = IdentityDiagnosticRecorder(max_events=3)
    for _ in range(5):
        recorder.record(DiagnosticEvents.IDENTITY_CREATED)
    
    assert len(recorder._events) == 3
    assert recorder._dropped_logs == 2

    # Verify flush
    path = tmp_path / "test.jsonl"
    recorder.flush_jsonl(str(path))
    
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        data = json.loads(line)
        assert data["event"] == DiagnosticEvents.IDENTITY_CREATED


def test_exception_safe_stage_timer():
    fake_clock_time = 0.0
    def fake_clock():
        return fake_clock_time

    recorder = IdentityDiagnosticRecorder(clock=fake_clock)
    
    # Success path
    with recorder.stage_timer("test_stage"):
        fake_clock_time += 0.5
    
    # Exception path
    with pytest.raises(ValueError):
        with recorder.stage_timer("test_stage"):
            fake_clock_time += 0.25
            raise ValueError("Stop")
            
    summary = recorder.summary()
    assert "test_stage" in summary["timings"]
    assert summary["timings"]["test_stage"]["count"] == 2
    assert summary["timings"]["test_stage"]["total"] == pytest.approx(0.75)


def test_slow_stage_detection():
    fake_clock_time = 0.0
    def fake_clock():
        return fake_clock_time

    recorder = IdentityDiagnosticRecorder(clock=fake_clock)
    
    with recorder.stage_timer("heavy_inference", slow_threshold=0.1):
        fake_clock_time += 0.05  # below threshold -> no slow log
        
    with recorder.stage_timer("heavy_inference", slow_threshold=0.1):
        fake_clock_time += 0.15  # above threshold -> triggers performance.slow_stage
        
    events = [e for e in recorder._events if e["event"] == DiagnosticEvents.PERFORMANCE_SLOW_STAGE]
    assert len(events) == 1
    assert events[0]["details"]["stage"] == "heavy_inference"
    assert events[0]["details"]["duration"] == pytest.approx(0.15)


def test_timings_percentile_summaries():
    recorder = IdentityDiagnosticRecorder()
    recorder._timings["inference"] = [float(x) for x in range(1, 101)] # 1 to 100
    
    summary = recorder.summary()
    inference = summary["timings"]["inference"]
    assert inference["count"] == 100
    assert inference["mean"] == pytest.approx(50.5)
    assert inference["max"] == pytest.approx(100.0)
    assert inference["p50"] == pytest.approx(50.5)
    assert inference["p90"] == pytest.approx(90.1)
    assert inference["p95"] == pytest.approx(95.05)
    assert inference["p99"] == pytest.approx(99.01)


def test_counters_and_gauges():
    recorder = IdentityDiagnosticRecorder()
    recorder.increment("matches_processed", 2)
    recorder.gauge("active_slots", 4)

    summary = recorder.summary()
    assert summary["counters"]["matches_processed"] == 2
    assert summary["gauges"]["active_slots"] == 4


def test_set_context_merges_into_details():
    d = IdentityDiagnosticRecorder()
    d.set_context(view="corner1")
    d.record("sam2.object_culled", reason="size_prior_violation",
             details={"object_id": 3, "frame_idx": 12, "box": [1, 2, 3, 4]})
    d.clear_context()
    d.record("sam2.object_culled", reason="person_overlap",
             details={"object_id": 4})
    evs = d._events
    assert evs[-2]["details"]["view"] == "corner1"
    assert "view" not in evs[-1]["details"]

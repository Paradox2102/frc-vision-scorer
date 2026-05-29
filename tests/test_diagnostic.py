"""Tests for web.diagnostic — the inference recorder and replay reader."""
import json
from pathlib import Path

import pytest

from web.diagnostic import DiagnosticRecorder, NullRecorder, replay_iter


def _snapshot_robots():
    return [
        {"robot_id": 3, "alliance": "red", "team": "1538", "confirmed": True,
         "votes": {"1538": 4.0}, "red_votes": 12, "blue_votes": 0,
         "last_seen_frame": 11,
         "embedding": {"age_frames": 0, "last_match_similarity": 0.82,
                       "last_match_gap": 0.11}},
    ]


def test_recorder_writes_one_of_each_event_type(tmp_path):
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    rec.record_vlm_read(frame=10, robot_id=3, crop_b64="abc", legibility=0.5,
                        raw_read="1538", matched_team="1538", alliance_read="red",
                        confidence=0.9, vote_weight=1.0)
    rec.record_vlm_filtered(frame=10, robot_id=1, crop_b64="def", legibility=0.001,
                            reason="below_legibility_floor")
    rec.record_assignment(frame=12, robot_id=3, team="1538", alliance="red",
                          confirmed=True, votes_for_assigned=4.0, votes_total=4.5,
                          top_team_share=0.89)
    rec.record_elimination(frame=12, robot_id=5, team="4322", alliance="red")
    rec.record_overturn(frame=20, robot_id=3, old_team="1538",
                        challenger_team="294", current_votes=3.0, challenger_votes=7.5)
    rec.record_reid_match(frame=15, track_id=7, canonical_id=3,
                          similarity=0.82, second_best=0.71)
    rec.record_reid_drop(frame=15, track_id=9, reason="claimed_by_lower_index",
                         would_have_matched=3, similarity=0.79)
    rec.record_snapshot(frame=12,
                        robots_state=_snapshot_robots(),
                        roster={"red": ["1538"], "blue": ["2102"]},
                        identified={"red": ["1538"], "blue": []})
    rec.close()

    lines = out.read_text().splitlines()
    assert len(lines) == 8
    parsed = [json.loads(l) for l in lines]
    assert [p["type"] for p in parsed] == [
        "vlm_read", "vlm_filtered", "assignment", "elimination",
        "overturn", "reid_match", "reid_drop", "snapshot",
    ]
    for p in parsed:
        assert "frame" in p and "t_ms" in p and "type" in p
    by_type = {p["type"]: p for p in parsed}
    assert by_type["reid_match"]["gap"] == pytest.approx(0.11)
    assert by_type["snapshot"]["robots"][0]["embedding"]["age_frames"] == 0


def test_recorder_survives_io_error(tmp_path, monkeypatch):
    """If the JSONL fails to open, the recorder must degrade to null mode and
    subsequent calls must not raise."""
    real_open = open

    def fail_open(*args, **kwargs):
        path = str(args[0]) if args else ""
        if path.endswith("diag.jsonl") and "a" in (args[1] if len(args) > 1 else kwargs.get("mode", "")):
            raise OSError("simulated: no space left")
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_open)
    rec = DiagnosticRecorder(tmp_path / "diag.jsonl", ws_send=None)
    assert rec.is_null
    rec.record_elimination(frame=1, robot_id=0, team="1538", alliance="red")  # must not raise
    rec.flush()
    rec.close()


def test_recorder_drops_input_with_missing_required_fields(tmp_path):
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    rec.record_vlm_read(frame=None, robot_id=3, crop_b64="abc", legibility=0.5,
                        raw_read="1538", matched_team="1538", alliance_read="red",
                        confidence=0.9, vote_weight=1.0)
    rec.close()
    assert out.read_text() == ""


def test_replay_iter_skips_malformed_lines(tmp_path):
    p = tmp_path / "diag.jsonl"
    p.write_text(
        '{"frame":1,"t_ms":0.0,"type":"elimination","robot_id":0,"team":"1538","alliance":"red"}\n'
        '{"frame":2,"t_ms":1.0,"type":"trunc\n'
        '{"frame":3,"t_ms":2.0,"type":"elimination","robot_id":1,"team":"4322","alliance":"red"}\n'
    )
    records = list(replay_iter(p))
    assert len(records) == 2
    assert records[0]["robot_id"] == 0
    assert records[1]["robot_id"] == 1


def test_replay_iter_handles_missing_file(tmp_path):
    assert list(replay_iter(tmp_path / "nope.jsonl")) == []


def test_snapshot_round_trips_embedding_age(tmp_path):
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    rec.record_snapshot(
        frame=50,
        robots_state=[
            {"robot_id": 3, "alliance": "red", "team": "1538", "confirmed": True,
             "votes": {"1538": 4.0}, "red_votes": 12, "blue_votes": 0,
             "last_seen_frame": 49,
             "embedding": {"age_frames": 42, "last_match_similarity": 0.55,
                           "last_match_gap": 0.04}},
        ],
        roster={"red": ["1538"], "blue": []},
        identified={"red": ["1538"], "blue": []},
    )
    rec.close()
    parsed = json.loads(out.read_text().strip())
    assert parsed["robots"][0]["embedding"]["age_frames"] == 42


def test_null_recorder_swallows_known_methods():
    rec = NullRecorder()
    # All record_* methods and flush/close must be callable on the null recorder.
    rec.record_vlm_read(frame=1, robot_id=2, crop_b64="x", legibility=0.1,
                        raw_read="x", matched_team=None, alliance_read="red",
                        confidence=0.5, vote_weight=0.5)
    rec.record_snapshot(frame=1, robots_state=[], roster={}, identified={})
    rec.flush()
    rec.close()


def test_recorder_degrades_to_null_on_midstream_write_failure(tmp_path):
    """If write() fails after the recorder is happily writing, it must close
    its FP, flip is_null=True, and silently swallow subsequent calls."""
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    # First record writes fine.
    rec.record_elimination(frame=1, robot_id=0, team="1538", alliance="red")
    assert not rec.is_null
    # Sabotage the file handle so the next write raises.
    import io
    class BrokenWriter:
        def write(self, *_a, **_kw): raise OSError("disk full")
        def flush(self): pass
        def close(self): pass
    rec._fp = BrokenWriter()
    rec.record_elimination(frame=2, robot_id=1, team="294", alliance="red")
    assert rec.is_null, "recorder must degrade after write failure"
    # Further calls must not raise.
    rec.record_elimination(frame=3, robot_id=2, team="4322", alliance="red")
    rec.flush()
    rec.close()
    # The first record made it to disk before the sabotage.
    lines = out.read_text().splitlines()
    assert len(lines) == 1


def test_recorder_drops_unserializable_record(tmp_path):
    """A record containing a non-JSON-serializable value must be logged and
    dropped, not raised back into the caller."""
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    # bytes are not JSON-serializable; raw_read happens to be the field that
    # passes through without numeric coercion (so the bytes survive to json.dumps).
    rec.record_vlm_read(frame=10, robot_id=3, crop_b64="abc", legibility=0.5,
                        raw_read=b"\x80\x81",   # un-serializable
                        matched_team="1538", alliance_read="red",
                        confidence=0.9, vote_weight=1.0)
    rec.close()
    # Either zero lines (record dropped) — the recorder must not crash.
    assert out.read_text() == ""

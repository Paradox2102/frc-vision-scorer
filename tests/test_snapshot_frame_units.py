"""Diagnostic snapshot rows must be stamped in SOURCE video frames.

`process_video` samples source frames every `frame_skip` across
`processing_ranges` (with gaps), so the main loop carries two counters:
`frame_num` (a source frame number) and `processed_frame_num` (a processed-
frame INDEX). Every other diagnostic row type (vlm_read, vlm_filtered,
overturn, adjudication, elimination, ...) and `_identity_log` use source
frames. The snapshot row once mixed the two: `frame` was a processed index
while `last_seen_frame` / `last_embedding_frame` are source frames, so
`embedding.age_frames` subtracted a source frame from a processed index, and
the `positions_history` fallback (whose tuples are `(processed_frame_num, ...)`)
injected a processed index into a source-frame field.
"""
import ast
import inspect
import json
from collections import deque

import web.app as app_module
from web.app import InferenceJob, RobotTrack
from web.diagnostic import DiagnosticRecorder


def _is_emit_snapshot_call(node):
    """True for a `self._emit_diagnostic_snapshot(...)` call expression."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "_emit_diagnostic_snapshot"
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    )


def test_main_loop_emits_snapshot_with_source_frame():
    source_path = inspect.getsourcefile(app_module)
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=source_path)

    calls = [n for n in ast.walk(tree) if _is_emit_snapshot_call(n)]
    assert calls, (
        "Could not find any self._emit_diagnostic_snapshot(...) call in "
        "web/app.py -- has the snapshot emit been refactored?")

    for call in calls:
        first = call.args[0] if call.args else None
        assert isinstance(first, ast.Name) and first.id == "frame_num", (
            f"self._emit_diagnostic_snapshot(...) at web/app.py:{call.lineno} "
            "must be passed `frame_num` (a SOURCE video frame number), not "
            "`processed_frame_num` (a processed-frame index). Snapshot rows "
            "share a timeline with every other diagnostic row and with "
            "last_seen_frame / last_embedding_frame, all of which are source "
            "frames.")


def _make_job(tmp_path, robots):
    job = InferenceJob.__new__(InferenceJob)
    job.robot_tracks = {r.robot_id: r for r in robots}
    job.match_red_teams = ["1538"]
    job.match_blue_teams = ["2102"]
    job.diagnostic = DiagnosticRecorder(tmp_path / "diag.jsonl", ws_send=None)
    return job


def _robot(robot_id, history=(), **kwargs):
    return RobotTrack(
        robot_id=robot_id, alliance="red", team_number=None,
        last_position=(0.0, 0.0), positions_history=deque(history), **kwargs)


def _read_snapshot_rows(job, tmp_path):
    job.diagnostic.flush()
    rows = [json.loads(line)
            for line in (tmp_path / "diag.jsonl").read_text().splitlines()]
    return [r for r in rows if r["type"] == "snapshot"]


def test_snapshot_last_seen_fallback_never_uses_processed_index(tmp_path):
    # positions_history tuples are (processed_frame_num, cx, cy): 125 is a
    # processed INDEX, which must never leak into the source-frame field.
    robot = _robot(1, history=[(125, 1.0, 2.0)],
                   last_seen_frame=None, last_embedding_frame=None)
    job = _make_job(tmp_path, [robot])

    job._emit_diagnostic_snapshot(1000)

    (snap,) = _read_snapshot_rows(job, tmp_path)
    (row,) = snap["robots"]
    assert row["last_seen_frame"] == 1000, (
        f"last_seen_frame fell back to positions_history's processed index "
        f"({row['last_seen_frame']}) instead of the snapshot's source frame")


def test_snapshot_ages_and_last_seen_are_source_frame_differences(tmp_path):
    # Characterization of the all-source-frame contract (passes before the
    # fix too: it pins behaviour the fix must preserve).
    fresh = _robot(1, last_embedding_frame=992, last_seen_frame=1000)
    never_embedded = _robot(2, last_embedding_frame=None, last_seen_frame=990)
    frame_zero = _robot(3, last_embedding_frame=0, last_seen_frame=0)
    job = _make_job(tmp_path, [fresh, never_embedded, frame_zero])

    job._emit_diagnostic_snapshot(1000)

    (snap,) = _read_snapshot_rows(job, tmp_path)
    assert snap["frame"] == 1000
    by_id = {r["robot_id"]: r for r in snap["robots"]}
    assert by_id[1]["embedding"]["age_frames"] == 8
    assert by_id[1]["last_seen_frame"] == 1000
    assert by_id[2]["embedding"]["age_frames"] is None
    assert by_id[2]["last_seen_frame"] == 990
    # Frame 0 is a real frame number: the `is None` guard must keep it 0
    # rather than treating it as "never seen".
    assert by_id[3]["last_seen_frame"] == 0
    assert by_id[3]["embedding"]["age_frames"] == 1000

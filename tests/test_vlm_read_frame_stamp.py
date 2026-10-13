"""vlm_read diagnostic rows must name the CROP frame and the crop slot's
identity AT that frame — not the frame/identity at read completion.

The fused bumper-read pass runs as a background asyncio task. Its body only
starts at the next loop yield (mid identity re-binding of a later frame) and
finishes at another, so the live self._current_frame and the identity bound
to the slot at completion are routinely 21-149 source frames late and
frequently a different identity. web/field/identity_resolve.build_slot_evidence
joins each vlm_read row on (frame, robot_id) against _identity_log as an
exact-frame misattribution guard, so a completion-time stamp drops the read
as misattributed or credits it to the wrong slot.

Also: the main loop assigns self._current_frame at the END of each
iteration, so at schedule time it still holds the PREVIOUS processed frame;
the caller must pass the frame `fused`'s crops were cut from (frame_num).
"""
import ast
import asyncio
import inspect
import json
import threading
import types
from unittest.mock import MagicMock

import numpy as np

import web.app as app_module
from web.app import InferenceJob
from web.diagnostic import DiagnosticRecorder
from web.field.identity import IdentityRegistry, RobotIdentity
from web.field.identity_resolve import build_slot_evidence
from web.job_config import JobConfig

_RESULT = {"team_number": "4174", "confidence": 0.9,
           "alliance": "blue", "raw_read": "4174"}


def _make_job(diag_path, current_frame=100):
    """Bare InferenceJob wired with a REAL IdentityRegistry and the REAL
    _get_or_create_slot_track (a live slot->identity lookup, exactly as in
    production), plus a real DiagnosticRecorder so the test reads back the
    actual JSONL row."""
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig()
    job._camera_centers_xy = {}
    job.match_teams = ["4174", "1538"]
    job.match_blue_teams = ["4174"]
    job.match_red_teams = ["1538"]
    job.vlm_status = {}
    job._vlm_task = None
    job._claude_vision_calls = 0
    job._current_frame = current_frame
    job._evidence_frame = current_frame
    job.video_fps = 30.0
    job.field_tracker = types.SimpleNamespace(last_det_to_slot={0: 1})
    reg = IdentityRegistry()
    reg.identities[7] = RobotIdentity(identity_id=7, alliance="blue", active_slot=1)
    reg.identities[9] = RobotIdentity(identity_id=9, alliance="red", active_slot=4)
    reg._next_id = 10
    job.identity_registry = reg
    job.robot_tracks = {}
    job._last_bindings = {}
    job._slot_source = {}
    job._slot_crops = {}
    job.scoring_events = []
    job.user_seed_zones_raw = []
    job._occ = {}
    job.confidence_config = types.SimpleNamespace(backoff_base_sec=2.0)
    job.team_reader = "local"
    job.co_occlusion_tracker = types.SimpleNamespace(get_active_events=lambda: [])
    job.apply_gallery_matches = lambda crops: None
    job.identity_diagnostics = MagicMock()
    job.diagnostic = DiagnosticRecorder(diag_path, ws_send=None)
    return job


def _fused_obs():
    return [types.SimpleNamespace(
        evidence=[("main", "fakecropb64", 1.0, (4.0, 3.0))],
        detection=types.SimpleNamespace(pos=(4.0, 3.0)),
    )]


def _rows(job, diag_path, row_type):
    job.diagnostic.flush()
    rows = [json.loads(line) for line in diag_path.read_text().splitlines()]
    return [r for r in rows if r["type"] == row_type]


def _install_blocking_read(monkeypatch):
    """Patch the VLM read with one that blocks until released, so the test
    can advance the tracker / swap identities while the read is in flight."""
    entered = threading.Event()
    release = threading.Event()

    def blocking_read(crops, teams, blue, red, expected_alliances=None, provider="local"):
        entered.set()
        release.wait(timeout=5.0)  # bounded: a bug can't hang the suite
        return [dict(_RESULT) for _ in crops]

    monkeypatch.setattr(app_module, "_vision_read_bumpers", blocking_read)
    return entered, release


async def _swap_while_in_flight(job, entered, release):
    """Simulate the tracker advancing 8 frames and identity 9 re-binding
    onto slot 1 (identity 7 losing it) while the read is in flight."""
    await asyncio.sleep(0.05)
    assert await asyncio.to_thread(entered.wait, 5.0), "read never started"
    assert not job._vlm_task.done(), "pass should still be in flight"
    job._current_frame = 108
    job.identity_registry.identities[9].active_slot = 1
    job.identity_registry.identities[7].active_slot = None
    release.set()
    await job._vlm_task


def _run_swap_scenario(monkeypatch, diag_path):
    entered, release = _install_blocking_read(monkeypatch)

    async def run():
        job = _make_job(diag_path, current_frame=100)
        job._schedule_vlm_pass(_fused_obs(), {7: 1.0})
        await _swap_while_in_flight(job, entered, release)
        return job

    return asyncio.run(run())


def test_async_read_recorded_at_schedule_frame_not_completion_frame(
        monkeypatch, tmp_path):
    diag_path = tmp_path / "diag.jsonl"
    job = _run_swap_scenario(monkeypatch, diag_path)

    reads = _rows(job, diag_path, "vlm_read")
    assert len(reads) == 1, reads
    row = reads[0]
    assert row["matched_team"] == "4174"
    assert row["frame"] == 100, (
        f"vlm_read stamped frame {row['frame']}; the crop was cut at frame 100 "
        f"(completion-time _current_frame leaked into the row)")
    assert row["robot_id"] == 7, (
        f"vlm_read stamped robot_id {row['robot_id']}; identity 7 held slot 1 "
        f"when the crop was cut (completion-time slot owner leaked into the row)")


def test_async_read_attributed_to_crop_slot_by_build_slot_evidence(
        monkeypatch, tmp_path):
    diag_path = tmp_path / "diag.jsonl"
    job = _run_swap_scenario(monkeypatch, diag_path)
    reads = _rows(job, diag_path, "vlm_read")

    identity_log = [
        {"frame_num": 100, "identity_id": 7, "slot_id": 1},
        {"frame_num": 100, "identity_id": 9, "slot_id": 4},
        {"frame_num": 108, "identity_id": 7, "slot_id": 1},
        {"frame_num": 108, "identity_id": 9, "slot_id": 4},
        {"frame_num": 116, "identity_id": 9, "slot_id": 1},
    ]
    evidence, stats = build_slot_evidence(
        reads, identity_log, {}, {},
        slot_alliance=lambda s: "blue" if s < 3 else "red")

    assert stats["reads_used"] == 1, stats
    assert stats["reads_dropped_misattributed"] == 0, stats
    assert stats["fallback_used"] is False, stats
    assert 1 in evidence and "4174" in evidence[1].read_scores, (
        f"read not credited to crop slot 1: "
        f"{ {s: e.read_scores for s, e in evidence.items()} }")
    assert 4 not in evidence or not evidence[4].read_scores, (
        f"read wrongly credited to slot 4: {evidence[4].read_scores}")


def test_schedule_explicit_crop_frame_wins_over_lagging_current_frame(
        monkeypatch, tmp_path):
    """Main-loop semantics: _current_frame still holds the PREVIOUS processed
    frame (92) when the pass is scheduled for frame_num 100."""
    monkeypatch.setattr(
        app_module, "_vision_read_bumpers",
        lambda crops, teams, blue, red, expected_alliances=None, provider="local":
            [dict(_RESULT) for _ in crops],
    )
    diag_path = tmp_path / "diag.jsonl"

    async def run():
        job = _make_job(diag_path, current_frame=92)
        job._schedule_vlm_pass(_fused_obs(), {7: 1.0}, crop_frame=100)
        await job._vlm_task
        return job

    job = asyncio.run(run())
    reads = _rows(job, diag_path, "vlm_read")
    assert len(reads) == 1, reads
    assert reads[0]["frame"] == 100
    assert reads[0]["robot_id"] == 7


def _app_tree():
    source_path = inspect.getsourcefile(app_module)
    with open(source_path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=source_path)


def _is_self_call(node, attr):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self")


def _kw(node, name):
    return next((k.value for k in node.keywords if k.arg == name), None)


def test_main_loop_schedules_vlm_pass_with_frame_num():
    calls = [n for n in ast.walk(_app_tree())
             if _is_self_call(n, "_schedule_vlm_pass")]
    assert calls, "no self._schedule_vlm_pass(...) call site found in web/app.py"
    for call in calls:
        val = _kw(call, "crop_frame")
        assert isinstance(val, ast.Name) and val.id == "frame_num", (
            f"self._schedule_vlm_pass(...) at line {call.lineno} must pass "
            f"crop_frame=frame_num (_current_frame still holds the previous "
            f"processed frame at schedule time)")


def test_early_pass_passes_crop_frame():
    calls = []
    for n in ast.walk(_app_tree()):
        if not _is_self_call(n, "vision_identify_robots"):
            continue
        arg_names = {a.id for a in n.args if isinstance(a, ast.Name)}
        if "early_overhead" in arg_names or "early_robots" in arg_names:
            calls.append(n)
    assert calls, "early-OCR self.vision_identify_robots(...) call not found"
    for call in calls:
        val = _kw(call, "crop_frame")
        assert isinstance(val, ast.Name) and val.id == "early_ocr_source_frame", (
            f"early-OCR vision_identify_robots at line {call.lineno} must pass "
            f"crop_frame=early_ocr_source_frame")


def test_sync_identify_stamps_explicit_crop_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(
        app_module, "_vision_read_bumpers",
        lambda crops, teams, blue, red, expected_alliances=None, provider="local":
            [dict(_RESULT) for _ in crops],
    )
    diag_path = tmp_path / "diag.jsonl"
    job = _make_job(diag_path, current_frame=492)
    job.config.det_gated_vlm_reads = True
    job.config.det_gate_max_age_s = 4.0
    job.detection_archive = MagicMock()
    job.detection_archive.corroborates = MagicMock(return_value=True)
    job._bumper_crop_b64 = MagicMock(return_value=("crop-b64", 0.5))

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    box = (20, 20, 180, 180)
    job._vision_identify_robots_impl(frame, [(1, box)], crop_frame=500)

    job.detection_archive.corroborates.assert_called_once()
    args, _kwargs = job.detection_archive.corroborates.call_args
    assert args[0] == "main"
    assert args[1] == 500, f"det-gate looked up frame {args[1]}, crop is frame 500"
    assert job._bumper_crop_b64.call_args.kwargs.get("crop_frame") == 500

    reads = _rows(job, diag_path, "vlm_read")
    assert len(reads) == 1, reads
    assert reads[0]["frame"] == 500
    assert reads[0]["robot_id"] == 7


def test_bumper_crop_filtered_row_uses_crop_frame(tmp_path):
    diag_path = tmp_path / "diag.jsonl"
    job = _make_job(diag_path, current_frame=492)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)  # legibility 0.0 < floor
    out = job._bumper_crop_b64(frame, (20, 20, 180, 180), robot_id=3,
                               enforce_floor=True, crop_frame=500)
    assert out is None

    filtered = _rows(job, diag_path, "vlm_filtered")
    assert len(filtered) == 1, filtered
    assert filtered[0]["frame"] == 500
    assert filtered[0]["robot_id"] == 3

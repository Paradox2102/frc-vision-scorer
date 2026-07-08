"""Detection-gated VLM reads (det_gated_vlm_reads knob, Task 8).

Focused unit test on the candidate-building loop inside
_vision_identify_robots_impl: verifies (a) a candidate whose box is NOT
corroborated by the detection archive is skipped and increments the
vlm_read.det_gate_skip diagnostics counter, (b) a candidate with no
archive coverage (None) or a corroborated one (True) proceeds, and
(c) with the knob at its default (False), the gate never touches the
archive at all, regardless of what corroborates() would return.

Also includes a source-structural regression test (see
test_current_frame_set_before_early_ocr_read_regression) guarding the
frame-units wiring for the early-OCR vision_identify_robots call.
"""
import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import web.app as app_module
from web.app import InferenceJob
from web.job_config import JobConfig


def _make_job(det_gated_vlm_reads, corroborate_return):
    """Bare InferenceJob with just enough stubbed to reach the candidate
    loop's occlusion + det-gate checks without touching real identity /
    tracking / VLM machinery."""
    job = InferenceJob.__new__(InferenceJob)
    job.config = JobConfig()
    job.config.det_gated_vlm_reads = det_gated_vlm_reads
    job.config.det_gate_max_age_s = 4.0
    job.video_fps = 10.0
    job._current_frame = 500
    job.match_teams = ["1", "2"]
    job.match_blue_teams = ["1"]
    job.match_red_teams = ["2"]
    job._occ = {}
    job.robot_tracks = {}

    stub_robot = SimpleNamespace(
        team_number_confirmed=False, team_number=None, alliance="blue",
    )
    job._get_or_create_slot_track = MagicMock(return_value=stub_robot)
    job._bumper_crop_b64 = MagicMock(return_value=None)  # short-circuit before VLM

    job.detection_archive = MagicMock()
    job.detection_archive.corroborates = MagicMock(return_value=corroborate_return)

    job.identity_diagnostics = MagicMock()

    return job


def _run(job, box):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    job._vision_identify_robots_impl(frame, [(0, box)])


def test_det_gate_skips_false_and_increments_counter():
    job = _make_job(det_gated_vlm_reads=True, corroborate_return=False)
    _run(job, (1, 2, 3, 4))

    job.detection_archive.corroborates.assert_called_once()
    args, kwargs = job.detection_archive.corroborates.call_args
    assert args[0] == "main"
    assert args[1] == 500
    assert args[2] == (1, 2, 3, 4)
    # box was never handed to the crop builder -> candidate was skipped
    job._bumper_crop_b64.assert_not_called()
    job.identity_diagnostics.increment.assert_called_once_with(
        "vlm_read.det_gate_skip")


def test_det_gate_allows_none_coverage_to_proceed():
    job = _make_job(det_gated_vlm_reads=True, corroborate_return=None)
    _run(job, (1, 2, 3, 4))

    job.detection_archive.corroborates.assert_called_once()
    job._bumper_crop_b64.assert_called_once()
    job.identity_diagnostics.increment.assert_not_called()


def test_det_gate_allows_true_corroboration_to_proceed():
    job = _make_job(det_gated_vlm_reads=True, corroborate_return=True)
    _run(job, (1, 2, 3, 4))

    job.detection_archive.corroborates.assert_called_once()
    job._bumper_crop_b64.assert_called_once()
    job.identity_diagnostics.increment.assert_not_called()


def test_det_gate_default_off_never_touches_archive():
    # Default False: even though corroborates() would say "skip", the gate
    # must not execute at all when the knob is off.
    job = _make_job(det_gated_vlm_reads=False, corroborate_return=False)
    _run(job, (1, 2, 3, 4))

    job.detection_archive.corroborates.assert_not_called()
    job._bumper_crop_b64.assert_called_once()
    job.identity_diagnostics.increment.assert_not_called()


def test_det_gate_fails_open_on_archive_exception():
    # A corroborates() that raises must FAIL OPEN (proceed to the read),
    # mirroring detection_archive.record()'s "telemetry must never break
    # detection" contract — never skip and never crash the identify pass.
    job = _make_job(det_gated_vlm_reads=True, corroborate_return=None)
    job.detection_archive.corroborates = MagicMock(
        side_effect=RuntimeError("corrupt index"))
    _run(job, (1, 2, 3, 4))

    job.detection_archive.corroborates.assert_called_once()
    # Read proceeded despite the error ...
    job._bumper_crop_b64.assert_called_once()
    # ... and the error was recorded (not the skip counter).
    job.identity_diagnostics.increment.assert_called_once_with(
        "vlm_read.det_gate_error")


def _is_self_current_frame_assign_from_early_ocr(node):
    """True if `node` is `self._current_frame = early_ocr_source_frame`."""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not (isinstance(target, ast.Attribute) and target.attr == "_current_frame"):
        return False
    if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
        return False
    return isinstance(node.value, ast.Name) and node.value.id == "early_ocr_source_frame"


def _is_early_vision_identify_robots_call(node):
    """True if `node` is a `self.vision_identify_robots(...)` call site fed
    early_overhead/early_robots (the early-OCR pass, not the impl's own
    definition and not other call sites e.g. re-read passes)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "vision_identify_robots"):
        return False
    if not (isinstance(func.value, ast.Name) and func.value.id == "self"):
        return False
    arg_names = {a.id for a in node.args if isinstance(a, ast.Name)}
    return "early_overhead" in arg_names or "early_robots" in arg_names


def test_current_frame_set_before_early_ocr_read_regression():
    """Regression guard for a shipped bug in the early vision-OCR pass.

    process_video runs an "early vision OCR pass" at the autonomous start,
    before the main per-frame loop begins (the main loop is what normally
    advances self._current_frame). The early pass calls
    self.vision_identify_robots(early_overhead, early_robots, ...), and
    that call's identity/diagnostics machinery (record_vlm_read, and the
    det_gated_vlm_reads archive lookup in _vision_identify_robots_impl)
    keys off self._current_frame.

    The shipped bug: without explicitly assigning
    self._current_frame = early_ocr_source_frame before that call,
    self._current_frame stayed at its initial value (0) during the entire
    early-OCR identify pass, corrupting diagnostics and the detection-gate
    frame-age lookup. The fix assigns self._current_frame just before the
    early vision_identify_robots(...) call.

    A full integration test through process_video is impractical (video
    I/O, models, huge pipeline), so this is a source-structural seam test:
    it parses web/app.py's own AST and asserts the assignment appears
    (by line number) strictly before the early-OCR call site. This catches
    a future reorder or removal of that assignment without requiring a
    production test seam.
    """
    source_path = inspect.getsourcefile(app_module)
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=source_path)

    assign_lines = []
    call_lines = []
    for node in ast.walk(tree):
        if _is_self_current_frame_assign_from_early_ocr(node):
            assign_lines.append(node.lineno)
        elif _is_early_vision_identify_robots_call(node):
            call_lines.append(node.lineno)

    assert assign_lines, (
        "Could not find `self._current_frame = early_ocr_source_frame` "
        "in web/app.py -- has the early-OCR pass been refactored?")
    assert call_lines, (
        "Could not find the early-OCR self.vision_identify_robots(...) "
        "call site (fed early_overhead/early_robots) in web/app.py -- "
        "has the early-OCR pass been refactored?")

    # Disambiguate via proximity: pair each assignment with the nearest
    # candidate call within a bounded window, since both the impl's own
    # definition and other call sites (e.g. re-read passes) may also
    # exist elsewhere in the file.
    paired = [
        (a, c) for a in assign_lines for c in call_lines
        if abs(c - a) <= 200
    ]
    assert paired, (
        "Found candidate assignment line(s) %r and call line(s) %r, but "
        "none within the expected ~200-line proximity window; the "
        "early-OCR pass may have been restructured." % (assign_lines, call_lines)
    )

    for assign_line, call_line in paired:
        assert assign_line < call_line, (
            "self._current_frame = early_ocr_source_frame (line %d) must "
            "be set BEFORE the early-OCR self.vision_identify_robots(...) "
            "call (line %d); otherwise self._current_frame stays stale "
            "(0) during the early-OCR identify pass, breaking "
            "record_vlm_read diagnostics and the det_gated_vlm_reads "
            "archive lookup." % (assign_line, call_line)
        )

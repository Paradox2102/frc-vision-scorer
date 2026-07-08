"""Detection-gated VLM reads (det_gated_vlm_reads knob, Task 8).

Focused unit test on the candidate-building loop inside
_vision_identify_robots_impl: verifies (a) a candidate whose box is NOT
corroborated by the detection archive is skipped and increments the
vlm_read.det_gate_skip diagnostics counter, (b) a candidate with no
archive coverage (None) or a corroborated one (True) proceeds, and
(c) with the knob at its default (False), the gate never touches the
archive at all, regardless of what corroborates() would return.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

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

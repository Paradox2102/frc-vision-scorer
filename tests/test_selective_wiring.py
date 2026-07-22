"""app.py wiring for selective mask propagation (Task 5).

Covers the two new seams:

* :meth:`InferenceJob.make_frame_detect_fn` -- the per-frame detector fed to
  ``track_selective``. It reads the exported view JPEG, runs the SAME cascade /
  field-filter / roster-cap logic as ``make_detect_fn`` but returns firm boxes
  in EXPORT coords, and mirrors the ``detection_archive`` telemetry with the
  full-res boxes.
* :meth:`InferenceJob._run_view_tracking` -- the ``selective_mask_prop`` gate.
  Knob OFF must run the historical ``track`` call untouched and never reference
  ``track_selective``; knob ON runs ``track_selective`` and, on ANY failure,
  logs ``[selective] FALLBACK`` and falls through to the identical ``track``
  call so a bad selective run can never lose a view.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from web.app import InferenceJob


def _make_job():
    with patch("web.app.YOLO"):
        job = InferenceJob("test_job", "/mock/video.mp4")
    return job


# ---------------------------------------------------------------------------
# make_frame_detect_fn
# ---------------------------------------------------------------------------
def test_make_frame_detect_fn_returns_export_boxes_and_records(tmp_path):
    job = _make_job()
    job.field_dims = (16.54, 8.21)
    job.field_model = None                       # -> half is None, no roster cap
    job.detection_archive = MagicMock()
    # The cascade already runs on the export-scale JPEG, so its box is in
    # EXPORT coords. One firm (strong=True) candidate.
    job._sam2_robot_candidates = MagicMock(
        return_value=[((10.0, 20.0, 30.0, 40.0), "red", True)])

    view = SimpleNamespace(name="main", calibration=MagicMock())
    view_plan = SimpleNamespace(source_frames=[7, 14, 21])

    jpg = tmp_path / "00000.jpg"
    cv2.imwrite(str(jpg), np.zeros((100, 200, 3), dtype=np.uint8))

    with patch("web.app.filter_field_robot_boxes", return_value=[0]):
        detect = job.make_frame_detect_fn(view, view_plan, 0.5, 0)
        out = detect(0, str(jpg))

    # EXPORT box passed straight through to the TBD tracker, paired with color.
    assert out == [((10.0, 20.0, 30.0, 40.0), "red")]

    # Archive mirrors make_detect_fn: FULL-RES box (export / 0.5 == *2),
    # corrob_only False, under this view + its source frame.
    job.detection_archive.record.assert_called_once()
    args = job.detection_archive.record.call_args.args
    assert args[0] == "main"          # view name
    assert args[1] == 0               # local_idx
    assert args[2] == 7               # source_frame from view_plan map
    assert args[3] == [((20.0, 40.0, 60.0, 80.0), "red", False)]


def test_make_frame_detect_fn_applies_top_crop_y_offset(tmp_path):
    """y_off is added to the full-res y-coords (the inverse of the export
    top-crop), and the export box is still returned unscaled."""
    job = _make_job()
    job.field_dims = (16.54, 8.21)
    job.field_model = None
    job.detection_archive = MagicMock()
    job._sam2_robot_candidates = MagicMock(
        return_value=[((10.0, 20.0, 30.0, 40.0), "blue", True)])

    view = SimpleNamespace(name="corner", calibration=MagicMock())
    view_plan = SimpleNamespace(source_frames=[3])

    jpg = tmp_path / "00000.jpg"
    cv2.imwrite(str(jpg), np.zeros((100, 200, 3), dtype=np.uint8))

    with patch("web.app.filter_field_robot_boxes", return_value=[0]):
        detect = job.make_frame_detect_fn(view, view_plan, 0.5, 12)
        out = detect(0, str(jpg))

    assert out == [((10.0, 20.0, 30.0, 40.0), "blue")]   # export box unchanged
    recorded = job.detection_archive.record.call_args.args[3]
    # full-res: x/0.5, y/0.5 + 12
    assert recorded == [((20.0, 52.0, 60.0, 92.0), "blue", False)]


def test_make_frame_detect_fn_swallows_errors_with_first_only_warning(capsys):
    job = _make_job()
    job.field_dims = (16.54, 8.21)
    job.field_model = None
    job.detection_archive = MagicMock()
    # imread returns None for a missing path -> the cascade never runs.
    job._sam2_robot_candidates = MagicMock(side_effect=RuntimeError("cascade boom"))

    view = SimpleNamespace(name="main", calibration=MagicMock())
    view_plan = SimpleNamespace(source_frames=[1, 2])

    with patch("web.app.cv2.imread", return_value=np.zeros((10, 20, 3), np.uint8)):
        detect = job.make_frame_detect_fn(view, view_plan, 0.5, 0)
        assert detect(0, "x.jpg") == []      # never raises into Phase A
        assert detect(1, "y.jpg") == []

    out = capsys.readouterr().out
    # First-error-only: the warning is printed once, not per frame.
    assert out.count("[selective] detect_fn error") == 1


def test_make_frame_detect_fn_out_of_range_local_idx_returns_empty():
    job = _make_job()
    view = SimpleNamespace(name="main", calibration=MagicMock())
    view_plan = SimpleNamespace(source_frames=[0, 1])
    detect = job.make_frame_detect_fn(view, view_plan, 0.5, 0)
    assert detect(5, "whatever.jpg") == []
    assert detect(None, "whatever.jpg") == []


# ---------------------------------------------------------------------------
# make_embed_frame_fn
# ---------------------------------------------------------------------------
def test_make_embed_frame_fn_batches_boxes(tmp_path):
    job = _make_job()
    vec = np.ones(384, dtype=np.float32)
    job.robot_embedder = SimpleNamespace(
        embed_batch=MagicMock(return_value=[vec, None]))
    jpg = tmp_path / "00000.jpg"
    cv2.imwrite(str(jpg), np.zeros((50, 50, 3), dtype=np.uint8))

    fn = job.make_embed_frame_fn()
    out = fn(0, str(jpg), [(0, 0, 10, 10), (20, 20, 40, 40)])
    assert out[0] is vec and out[1] is None
    assert job.robot_embedder.embed_batch.call_count == 1


def test_make_embed_frame_fn_none_embedder_returns_nones():
    job = _make_job()
    job.robot_embedder = None
    fn = job.make_embed_frame_fn()
    assert fn(0, "/nonexistent.jpg", [(0, 0, 10, 10)]) == [None]


def test_make_embed_frame_fn_swallows_errors_first_warning_only(capsys):
    job = _make_job()
    job.robot_embedder = SimpleNamespace(
        embed_batch=MagicMock(side_effect=RuntimeError("boom")))
    fn = job.make_embed_frame_fn()
    # Unreadable path also exercises the imread-None branch harmlessly.
    boxes = [(0, 0, 10, 10)]
    assert fn(0, "/nonexistent.jpg", boxes) == [None]
    assert fn(1, "/nonexistent.jpg", boxes) == [None]
    err = capsys.readouterr().out
    assert err.count("embed_frame_fn") <= 1


# ---------------------------------------------------------------------------
# _run_view_tracking gate
# ---------------------------------------------------------------------------
def _tracker_stub(track_return):
    tracker = MagicMock()
    tracker.track.return_value = track_return
    return tracker


def test_gate_off_never_references_track_selective():
    job = _make_job()
    job.config.selective_mask_prop = False
    job.timings = {}
    sentinel = {0: {}}
    tracker = _tracker_stub(sentinel)
    tracker.track_selective.side_effect = AssertionError("track_selective ran")

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames", "prompt_idx": 3}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    result = asyncio.run(run())
    assert result is sentinel
    tracker.track_selective.assert_not_called()
    tracker.track.assert_called_once_with(**track_kwargs)
    # Timing still accrues to the uniform path.
    assert "sam2_tracking" in job.timings


def test_gate_on_fallback_runs_track_and_logs(capsys):
    job = _make_job()
    job.config.selective_mask_prop = True
    job.timings = {}
    job.sam2_obj_color = {}
    sentinel = {0: {}, 1: {}}
    tracker = _tracker_stub(sentinel)
    tracker.track_selective.side_effect = RuntimeError("boom")

    view = SimpleNamespace(name="corner")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    result = asyncio.run(run())
    assert result is sentinel                     # fell back to uniform track
    tracker.track_selective.assert_called_once()
    tracker.track.assert_called_once_with(**track_kwargs)
    out = capsys.readouterr().out
    assert "[selective] FALLBACK view=corner: boom" in out


def test_gate_on_success_logs_stats_and_registers_colors(capsys):
    job = _make_job()
    job.config.selective_mask_prop = True
    job.timings = {}
    job.sam2_obj_color = {}
    tracks = {0: {1: {}}, 1: {1: {}}}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.return_value = tracks
    tracker.selective_stats = {
        "frames": 100, "windows": 3, "sam_frames": 25,
        "outcomes": {"clean": 2, "swap": 1}, "renames": 1, "skipped": 0}
    tracker.selective_obj_colors = {1: "red", 2: None, 3: "blue"}

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    result = asyncio.run(run())
    assert result is tracks
    tracker.track.assert_not_called()             # selective path stood on its own
    # Selective mode REPLACES the per-view dict with the tracker's colors, keyed
    # by TBD oid and INCLUDING None values (so a colorless oid reads back as
    # "unknown" via evidence_build's ``.get(oid) or "unknown"`` rather than a
    # stale prompt-seed color left at a colliding integer key).
    assert job.sam2_obj_color["main"] == {1: "red", 2: None, 3: "blue"}

    out = capsys.readouterr().out
    assert ("[selective] view=main: frames=100 windows=3 sam_frames=25 "
            "(25% of uniform)") in out
    assert "outcomes={'clean': 2, 'swap': 1} renames=1 skipped=0" in out


def test_gate_on_passes_sam_budget_frac_override_to_selective_config():
    """selective_sam_budget_frac (Task C knob) must reach the SelectiveConfig
    built for track_selective, not just sit unused on JobConfig."""
    job = _make_job()
    job.config.selective_mask_prop = True
    job.config.selective_sam_budget_frac = 0.8   # non-default override
    job.timings = {}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.side_effect = RuntimeError("boom")  # short-circuit to fallback

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    tracker.track_selective.assert_called_once()
    cfg = tracker.track_selective.call_args.kwargs["cfg"]
    assert cfg.sam_budget_frac == 0.8


def test_gate_on_passes_mask_overlap_exit_override_to_selective_config():
    """selective_mask_overlap_exit (Task 1 knob) must reach the SelectiveConfig
    built for track_selective, not just sit unused on JobConfig."""
    job = _make_job()
    job.config.selective_mask_prop = True
    job.config.selective_mask_overlap_exit = 0.95   # non-default override
    job.timings = {}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.side_effect = RuntimeError("boom")  # short-circuit to fallback

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    tracker.track_selective.assert_called_once()
    cfg = tracker.track_selective.call_args.kwargs["cfg"]
    assert cfg.mask_overlap_exit == 0.95


def test_gate_on_passes_kill_coast_overrides_to_selective_config():
    """selective_kill_coast / _frames (Task 2 knobs) must reach the
    SelectiveConfig built for track_selective, not just sit unused on
    JobConfig."""
    job = _make_job()
    job.config.selective_mask_prop = True
    job.config.selective_kill_coast = True          # non-default overrides
    job.config.selective_kill_coast_frames = 48
    job.timings = {}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.side_effect = RuntimeError("boom")  # short-circuit to fallback

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    tracker.track_selective.assert_called_once()
    cfg = tracker.track_selective.call_args.kwargs["cfg"]
    assert cfg.kill_coast is True
    assert cfg.kill_coast_frames == 48


def test_gate_on_passes_starve_overrides_to_selective_config():
    """selective_starve_windows / _trigger / _max_frames (Task 3 knobs) must
    reach the SelectiveConfig built for track_selective, not just sit unused on
    JobConfig."""
    job = _make_job()
    job.config.selective_mask_prop = True
    job.config.selective_starve_windows = True      # non-default overrides
    job.config.selective_starve_trigger = 11
    job.config.selective_starve_max_frames = 200
    job.timings = {}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.side_effect = RuntimeError("boom")  # short-circuit to fallback

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    tracker.track_selective.assert_called_once()
    cfg = tracker.track_selective.call_args.kwargs["cfg"]
    assert cfg.starve_windows is True
    assert cfg.starve_trigger == 11
    assert cfg.starve_max_frames == 200


def test_gate_on_passes_starve_min_obs_to_selective_config():
    """selective_starve_min_obs (junk gate knob) must reach the SelectiveConfig
    built for track_selective, not just sit unused on JobConfig."""
    job = _make_job()
    job.config.selective_mask_prop = True
    job.config.selective_starve_min_obs = 55        # non-default override
    job.timings = {}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.side_effect = RuntimeError("boom")  # short-circuit to fallback

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    tracker.track_selective.assert_called_once()
    cfg = tracker.track_selective.call_args.kwargs["cfg"]
    assert cfg.starve_min_obs == 55


def test_gate_on_passes_app_overrides_to_selective_config():
    """selective_app_weight / selective_app_ema (Task 2 knobs) must reach the
    SelectiveConfig built for track_selective, not just sit unused on
    JobConfig."""
    job = _make_job()
    job.config.selective_mask_prop = True
    job.config.selective_app_weight = 0.6           # non-default overrides
    job.config.selective_app_ema = 0.7
    job.timings = {}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.side_effect = RuntimeError("boom")  # short-circuit to fallback

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    tracker.track_selective.assert_called_once()
    cfg = tracker.track_selective.call_args.kwargs["cfg"]
    assert cfg.app_weight == 0.6
    assert cfg.app_ema == 0.7


def test_gate_on_passes_embed_frame_fn_only_when_weight_positive():
    """weight 0 -> embed_frame_fn=None; weight >0 -> a callable."""
    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    def _run(weight):
        job = _make_job()
        job.config.selective_mask_prop = True
        job.config.selective_app_weight = weight
        job.timings = {}
        tracker = _tracker_stub({"unused": True})
        tracker.track_selective.side_effect = RuntimeError("boom")  # short-circuit to fallback

        async def run():
            return await job._run_view_tracking(
                tracker, view, track_kwargs,
                lambda i, p: [], lambda a, b: None)

        asyncio.run(run())
        tracker.track_selective.assert_called_once()
        return tracker.track_selective.call_args.kwargs["embed_frame_fn"]

    assert _run(0.0) is None
    assert callable(_run(0.6))


def test_gate_on_success_log_includes_episodes_and_dropped(capsys):
    job = _make_job()
    job.config.selective_mask_prop = True
    job.timings = {}
    job.sam2_obj_color = {}
    tracks = {0: {1: {}}, 1: {1: {}}}
    tracker = _tracker_stub({"unused": True})
    tracker.track_selective.return_value = tracks
    tracker.selective_stats = {
        "frames": 100, "windows": 3, "sam_frames": 25,
        "outcomes": {"clean": 2, "swap": 1}, "renames": 1, "skipped": 0,
        "episodes": 2, "dropped_windows": 1}
    tracker.selective_obj_colors = {1: "red"}

    view = SimpleNamespace(name="main")
    track_kwargs = {"frames_dir": "/frames"}

    async def run():
        return await job._run_view_tracking(
            tracker, view, track_kwargs,
            lambda i, p: [], lambda a, b: None)

    asyncio.run(run())
    out = capsys.readouterr().out
    assert " episodes=2" in out
    assert " dropped=1" in out

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import cv2

from web.app import InferenceJob, _restore_mask_to_view
from web.job_config import JobConfig

# These tests exercise the SAM2 path specifically; pin the backend so the
# sam3 default (June 10 A/B) cannot route them to the real SAM3 model.
SAM2_CFG = JobConfig({"tracking_backend": "sam2"})
from web.field.tracker import FieldDetection


class MockTensor:
    def __init__(self, val):
        self.val = val
    def cpu(self):
        return self
    def numpy(self):
        return self.val


class MockXYXY:
    def __init__(self, val_2d):
        self.val_2d = val_2d
    def cpu(self):
        return self
    def numpy(self):
        return self.val_2d
    def __getitem__(self, idx):
        return MockTensor(self.val_2d[idx])
    @property
    def shape(self):
        return self.val_2d.shape
    @property
    def ndim(self):
        return self.val_2d.ndim


class MockCls:
    def __init__(self, val_1d):
        self.val_1d = val_1d
    def cpu(self):
        return self
    def numpy(self):
        return self.val_1d
    def __getitem__(self, idx):
        return self.val_1d[idx]
    @property
    def shape(self):
        return self.val_1d.shape
    @property
    def ndim(self):
        return self.val_1d.ndim


class MockBoxes:
    def __init__(self, cls_1d, xyxy_2d, conf_1d):
        self.cls = MockCls(cls_1d)
        self.xyxy = MockXYXY(xyxy_2d)
        self.conf = MockCls(conf_1d)
        self.id = None  # To satisfy supervision.Detections.from_ultralytics
    def __len__(self):
        return len(self.cls.val_1d)


def test_inference_job_defaults_vram_budget_from_env_constant():
    """Per-job VRAM frame budget defaults to the module env constant."""
    import web.app as app_mod
    job = app_mod.InferenceJob("j", "/mock/video.mp4")
    assert job.sam3_max_gpu_frames == app_mod.SAM3_MAX_GPU_FRAMES


def test_restore_mask_reinserts_top_crop_without_vertical_shift():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True

    restored = _restore_mask_to_view(
        mask,
        (10, 12, 3),
        top_offset_px=2,
    )

    assert restored.shape == (10, 12)
    assert not restored[:2].any()
    ys, xs = np.where(restored)
    assert ys.min() == 4
    assert ys.max() == 7
    assert xs.min() == 4
    assert xs.max() == 9


def test_restore_mask_without_crop_fills_full_view():
    mask = np.ones((2, 3), dtype=bool)

    restored = _restore_mask_to_view(mask, (4, 6, 3))

    assert restored.all()


class TestSAM2Streaming(unittest.TestCase):
    @patch("web.app.YOLO")
    def test_sam2_fallback_on_import_error(self, mock_yolo):
        """If SAM2 fails to load, the job gracefully falls back to Kalman tracking."""
        job = InferenceJob("test_job", "/mock/path/video.mp4", config=SAM2_CFG)
        
        # Setup mock views and tracking ready
        job.views = [MagicMock(name="main")]
        job.views[0].name = "main"
        job.total_frames = 300
        job.video_fps = 20.0
        
        # Mock _init_field_tracking to bypass calibration loading
        job._init_field_tracking = MagicMock(return_value=True)
        job.model = MagicMock()
        job.person_model = MagicMock()
        
        # Mock VideoCapture to return empty/no frames to end loop early
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 20.0,
            cv2.CAP_PROP_FRAME_COUNT: 300
        }.get(prop, 0.0)
        
        # Mock open log file to avoid creating files
        job._open_log_file = MagicMock()
        job._close_log_file = MagicMock()
        job.extract_team_numbers = MagicMock()
        job._scan_match_clock = MagicMock(return_value=[])
        
        # Simulate MaskFieldTracker raising an ImportError (e.g. no torch/sam2 installed)
        with patch("web.field.mask_tracker.MaskFieldTracker", side_effect=ImportError("no torch")):
            with patch("cv2.VideoCapture", return_value=mock_cap):
                import asyncio
                asyncio.run(job.process_video())
                
                # Check that use_mask_tracker is False and it ran fallback path
                assert job.use_mask_tracker is False
                assert not hasattr(job, "fused_registry")

    @patch("web.app.YOLO")
    @patch("web.app.RobotEmbedder")
    @patch("web.field.mask_tracker.MaskFieldTracker")
    def test_sam2_precomputation_and_lookup(self, mock_sam2_tracker_cls, mock_embedder_cls, mock_yolo):
        """Verify the SAM2 pre-computation pass runs and lookups are performed in the loop."""
        # Force YOLO calls to return our MockBoxes
        mock_yolo_result = MagicMock()
        mock_yolo_result.boxes = MockBoxes(
            cls_1d=np.array([1.0], dtype=np.float32),
            xyxy_2d=np.array([[10, 10, 50, 50]], dtype=np.float32),
            conf_1d=np.array([0.9], dtype=np.float32)
        )
        mock_yolo_result.obb = None
        mock_yolo_result.masks = None
        mock_yolo_result.keypoints = None
        
        # Mock YOLO model constructor
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = [mock_yolo_result]
        mock_model_instance.names = {0: "blue_bumper", 1: "red_bumper"}
        mock_yolo.return_value = mock_model_instance
        
        job = InferenceJob("test_job", "/mock/path/video.mp4", config=SAM2_CFG)
        
        # Mock views
        mock_view = MagicMock()
        mock_view.name = "main"
        mock_view.calibration.image_to_field.return_value = (5.0, 5.0)
        mock_view.calibration.camera_center = (0.0, 0.0, 5.0)
        mock_view.calibration.robot_track_top_crop = 0.0  # match real CameraCalibration contract
        mock_view.masks = []  # real CameraView.masks is an iterable of (x,y,w,h)
        
        # view.extract returns a mock frame
        mock_crop = np.zeros((300, 300, 3), dtype=np.uint8)
        mock_view.extract.return_value = mock_crop
        
        job.views = [mock_view]
        job.total_frames = 210
        job.video_fps = 20.0
        job.target_tracking_fps = 10.0
        job.field_dims = (16.0, 8.0)
        
        # Mock methods
        job._init_field_tracking = MagicMock(return_value=True)
        # Mock the field tracker so we don't get NoneType AttributeError on update
        job.field_tracker = MagicMock()
        # Mock tracking outputs so we get a list of active tracks
        mock_track = MagicMock()
        mock_track.slot_id = 0
        mock_track.alliance = "blue"
        mock_track.pos = (5.0, 5.0)
        mock_track.box = (10, 10, 50, 50)
        mock_track.state = "visible"
        job.field_tracker.update.return_value = [mock_track]
        
        # Mock last_det_to_slot
        job.field_tracker.last_det_to_slot = {0: 0}
        job.field_tracker._id_to_slot = {0: 0, 1: 0}
        
        # Mock radar
        job.radar = MagicMock()
        job.radar.render.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        job._compose_hud_frame = MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8)
        )
        
        job._open_log_file = MagicMock()
        job._close_log_file = MagicMock()
        job.extract_team_numbers = MagicMock()
        job._scan_match_clock = MagicMock(return_value=[])
        
        mock_person_result = MagicMock()
        mock_person_result.boxes = MockBoxes(
            cls_1d=np.zeros(0, dtype=np.float32),
            xyxy_2d=np.zeros((0, 4), dtype=np.float32),
            conf_1d=np.zeros(0, dtype=np.float32)
        )
        mock_person_result.obb = None
        mock_person_result.masks = None
        mock_person_result.keypoints = None
        
        # Setup person model
        mock_person_model_instance = MagicMock()
        mock_person_model_instance.return_value = [mock_person_result]
        job.person_model = mock_person_model_instance
        
        # Mock SAM2 tracker returning a track dictionary
        # Legacy fallback starts at frame 180, so frame 200 maps to local index 10.
        mock_sam2_tracker = MagicMock()
        mock_sam2_tracker.track.return_value = {
            10: {0: {"box": (5.0, 5.0, 25.0, 25.0), "mask": np.ones((150, 150), dtype=bool)}}
        }
        mock_sam2_tracker_cls.return_value = mock_sam2_tracker
        
        # Mock embedder returning one embedding per box (batched API)
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros(384, dtype=np.float32)
        mock_embedder.embed_batch.side_effect = (
            lambda frame, boxes, masks=None:
                [np.zeros(384, dtype=np.float32)] * len(boxes))
        mock_embedder_cls.return_value = mock_embedder
        
        # Setup VideoCapture to run for a few frames
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 20.0,
            cv2.CAP_PROP_FRAME_COUNT: 210
        }.get(prop, 0.0)
        
        # Let's mock a separate counter for each file capture instance
        # Since VideoCapture is instantiated inside process_video, we can return fresh instances
        # and limit the reads on each
        read_counter = 0
        def mock_read():
            nonlocal read_counter
            read_counter += 1
            # All VideoCapture() instances share this one mock, so this counter bounds
            # the whole pipeline's reads.  The SAM2 frame export now does a single
            # SEQUENTIAL decode pass (reading every frame between title-screen and end),
            # rather than one seek+read per kept frame, so it consumes more reads up
            # front.  Budget must still leave the main loop enough reads to reach
            # local_idx 10 (frame 200), where the DINOv2 embed() is exercised.
            if read_counter > 120:
                return False, None
            return True, np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.side_effect = mock_read
        
        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("web.app.ReplayRecorder") as replay_cls, \
             patch(
                 "web.app.database.upsert_match_robot_identity",
                 new=AsyncMock(),
             ), \
             patch(
                 "web.app.database.insert_match_robot_crop",
                 new=AsyncMock(),
             ), \
             patch(
                 "web.app.database.insert_match_robot_binding",
                 new=AsyncMock(),
             ), \
             patch(
                 "web.app.asyncio.to_thread",
                 new=AsyncMock(
                     side_effect=lambda func, **kwargs: func(**kwargs)
                 ),
             ):
            replay_cls.return_value.is_open = False
            import asyncio
            asyncio.run(job.process_video())
            
            # Print state for debugging if it fails
            print(f"DEBUG: use_mask_tracker = {job.use_mask_tracker}")
            if hasattr(job, "tracker_raw"):
                print(f"DEBUG: tracker_raw keys = {list(job.tracker_raw.keys())}")
                if "main" in job.tracker_raw:
                    print(f"DEBUG: tracker_raw['main'] keys = {list(job.tracker_raw['main'].keys())}")
            
            # Check that use_mask_tracker is True
            assert job.use_mask_tracker is True
            assert hasattr(job, "fused_registry")
            # Verify SAM2 tracker was called
            mock_sam2_tracker.track.assert_called_once()
            # Verify periodic re-prompting was wired in: a YOLO detect callback,
            # an integer chunk interval, and the FRC 6-robot cap.
            _, track_kwargs = mock_sam2_tracker.track.call_args
            assert callable(track_kwargs.get("detect_fn"))
            assert isinstance(track_kwargs.get("reprompt_every"), int)
            assert track_kwargs["reprompt_every"] >= 1
            assert isinstance(track_kwargs.get("object_metadata"), dict)
            assert track_kwargs.get("max_objects") == 6
            # Verify DINOv2 embedding was extracted during the loop
            # (one batched forward pass per view instead of per-object embed()).
            mock_embedder.embed_batch.assert_called()


def test_debug_tasks_endpoint_reports_running_task_stacks():
    """/api/debug/tasks dumps every live asyncio task with a coroutine stack —
    the diagnostic hook for event-loop hangs (job stalls with all threads idle)."""
    import asyncio
    import web.app as app_mod

    async def scenario():
        async def parked():
            await asyncio.sleep(30)

        t = asyncio.create_task(parked(), name="parked-task")
        await asyncio.sleep(0)  # let it start and suspend
        payload = await app_mod.debug_tasks()
        t.cancel()
        return payload

    payload = asyncio.run(scenario())
    names = {t["name"] for t in payload["tasks"]}
    assert "parked-task" in names
    parked_stacks = [t["stack"] for t in payload["tasks"] if t["name"] == "parked-task"]
    assert any("parked" in frame for frame in parked_stacks[0])


def test_ws_first_attach_claims_and_runs_job():
    """The first websocket connection to a job starts its run (and marks the
    job claimed so later connections won't run it again)."""
    from unittest.mock import patch
    from starlette.testclient import TestClient
    import web.app as app_mod

    job = app_mod.InferenceJob("ws-first", "/mock/video.mp4")
    app_mod.jobs["ws-first"] = job
    ran = []

    async def fake_run(j):
        ran.append(j.job_id)

    try:
        with patch.object(app_mod, "_run_job_serialized", fake_run):
            client = TestClient(app_mod.app)
            with client.websocket_connect("/ws/infer/ws-first"):
                pass
        assert ran == ["ws-first"]
        assert job._run_claimed is True
    finally:
        app_mod.jobs.pop("ws-first", None)


def test_ws_reattach_does_not_rerun_running_job():
    """Reconnecting to a job that is already running must attach for
    streaming only — never queue a second process_video run."""
    from unittest.mock import patch
    from starlette.testclient import TestClient
    import web.app as app_mod

    job = app_mod.InferenceJob("ws-re", "/mock/video.mp4")
    job._run_claimed = True
    job.status = "processing"
    app_mod.jobs["ws-re"] = job
    ran = []

    async def fake_run(j):
        ran.append(j.job_id)

    try:
        with patch.object(app_mod, "_run_job_serialized", fake_run):
            client = TestClient(app_mod.app)
            with client.websocket_connect("/ws/infer/ws-re") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "status"
                assert msg["status"] == "processing"
                assert "attach" in msg["message"].lower()
        assert ran == []
    finally:
        app_mod.jobs.pop("ws-re", None)


def test_ws_attach_to_finished_job_reports_terminal_status():
    """Connecting to an already-finished job reports its terminal status
    instead of re-running it."""
    from unittest.mock import patch
    from starlette.testclient import TestClient
    import web.app as app_mod

    job = app_mod.InferenceJob("ws-done", "/mock/video.mp4")
    job._run_claimed = True
    job.status = "completed"
    app_mod.jobs["ws-done"] = job
    ran = []

    async def fake_run(j):
        ran.append(j.job_id)

    try:
        with patch.object(app_mod, "_run_job_serialized", fake_run):
            client = TestClient(app_mod.app)
            with client.websocket_connect("/ws/infer/ws-done") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "status"
                assert msg["status"] == "completed"
        assert ran == []
    finally:
        app_mod.jobs.pop("ws-done", None)


def test_active_job_endpoint_reports_running_job():
    """GET /api/job/active lets a reloaded page rediscover and rejoin a
    running match (the page loses the job id on reload)."""
    from starlette.testclient import TestClient
    import web.app as app_mod

    job = app_mod.InferenceJob("active-1", "/mock/video.mp4")
    job._run_claimed = True
    job.status = "processing"
    app_mod.jobs["active-1"] = job
    try:
        client = TestClient(app_mod.app)
        body = client.get("/api/job/active").json()
        assert body["job_id"] == "active-1"
        assert body["status"] == "processing"
        assert isinstance(body.get("started_at"), (int, float))
    finally:
        app_mod.jobs.pop("active-1", None)


def test_active_job_endpoint_none_when_idle():
    from starlette.testclient import TestClient
    import web.app as app_mod

    job = app_mod.InferenceJob("done-1", "/mock/video.mp4")
    job.status = "completed"
    app_mod.jobs["done-1"] = job
    try:
        client = TestClient(app_mod.app)
        body = client.get("/api/job/active").json()
        assert body["job_id"] is None
    finally:
        app_mod.jobs.pop("done-1", None)

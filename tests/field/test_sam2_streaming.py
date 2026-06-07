import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import cv2

from web.app import InferenceJob
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
    assert job.sam2_max_gpu_frames == app_mod.SAM2_MAX_GPU_FRAMES


class TestSAM2Streaming(unittest.TestCase):
    @patch("web.app.YOLO")
    def test_sam2_fallback_on_import_error(self, mock_yolo):
        """If SAM2 fails to load, the job gracefully falls back to Kalman tracking."""
        job = InferenceJob("test_job", "/mock/path/video.mp4")
        
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
        
        # Simulate Sam2FieldTracker raising an ImportError (e.g. no torch/sam2 installed)
        with patch("web.field.sam2_tracker.Sam2FieldTracker", side_effect=ImportError("no torch")):
            with patch("cv2.VideoCapture", return_value=mock_cap):
                import asyncio
                asyncio.run(job.process_video())
                
                # Check that use_sam2 is False and it ran fallback path
                assert job.use_sam2 is False
                assert not hasattr(job, "fused_registry")

    @patch("web.app.YOLO")
    @patch("web.app.RobotEmbedder")
    @patch("web.field.sam2_tracker.Sam2FieldTracker")
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
        
        job = InferenceJob("test_job", "/mock/path/video.mp4")
        
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
        
        # Mock embedder returning mock embedding
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros(384, dtype=np.float32)
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
                 "web.app.asyncio.to_thread",
                 new=AsyncMock(
                     side_effect=lambda func, **kwargs: func(**kwargs)
                 ),
             ):
            replay_cls.return_value.is_open = False
            import asyncio
            asyncio.run(job.process_video())
            
            # Print state for debugging if it fails
            print(f"DEBUG: use_sam2 = {job.use_sam2}")
            if hasattr(job, "sam2_raw"):
                print(f"DEBUG: sam2_raw keys = {list(job.sam2_raw.keys())}")
                if "main" in job.sam2_raw:
                    print(f"DEBUG: sam2_raw['main'] keys = {list(job.sam2_raw['main'].keys())}")
            
            # Check that use_sam2 is True
            assert job.use_sam2 is True
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
            mock_embedder.embed.assert_called()

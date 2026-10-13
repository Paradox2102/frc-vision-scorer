from unittest.mock import MagicMock

import numpy as np

from web.app import InferenceJob


def test_vision_identify_threads_bumper_box_mode_to_impl():
    job = InferenceJob.__new__(InferenceJob)
    job.timings = {}
    job._vision_identify_robots_impl = MagicMock()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    tracked = [(0, (1, 2, 3, 4))]

    job.vision_identify_robots(frame, tracked, boxes_are_bumpers=True)

    job._vision_identify_robots_impl.assert_called_once_with(
        frame,
        tracked,
        boxes_are_bumpers=True,
        crop_frame=None,
    )

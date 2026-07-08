import cv2
import numpy as np
from web.field.hardneg_miner import mine_hard_negatives


def _write_video(path, n_frames=30, w=64, h=48):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n_frames):
        vw.write(np.full((h, w, 3), i * 5 % 255, np.uint8))
    vw.release()


def test_miner_exports_frames_for_cull_events(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    events = [
        {"event": "sam2.object_culled", "reason": "size_prior_violation",
         "details": {"view": "main", "frame_idx": 4, "box": [2, 2, 10, 10]}},
        {"event": "sam2.reprompt_rejected", "reason": "static_structure",
         "details": {"view": "main", "frame_idx": 6, "box": [4, 4, 12, 12]}},
        {"event": "sam2.object_culled", "reason": "evicted_for_new_detection",
         "details": {"view": "main", "frame_idx": 8, "box": [1, 1, 5, 5]}},  # not a mined reason
        {"event": "sam2.object_culled", "reason": "person_overlap",
         "details": {"frame_idx": 9, "box": [1, 1, 5, 5]}},  # no view -> skipped
    ]
    view_infos = {"main": {
        "source_frames": list(range(0, 30)), "scale": 0.5, "y_off": 0,
        "extract": lambda fr: fr,
    }}
    out = tmp_path / "hn"
    n = mine_hard_negatives(events, str(video), view_infos, str(out))
    assert n == 2
    jsons = sorted(out.glob("*.json"))
    assert len(jsons) == 2
    import json
    m = json.loads(jsons[0].read_text())
    # SAM2-space box [2,2,10,10] at scale 0.5 -> full-res [4,4,20,20]
    assert m["rejected"][0]["box"] == [4.0, 4.0, 20.0, 20.0]
    assert m["rejected"][0]["reason"] == "size_prior_violation"

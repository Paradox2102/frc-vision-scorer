"""Post-job hard-negative export from tracker diagnostics.

Equipment hijacks are detector-corroborated (YOLO false-fires on the
equipment itself — see the size-prior comment in mask_tracker.py), so culls
can only ever treat symptoms; retraining the detector on its own confusers is
the fix. The diagnostics stream already names them: size-prior culls, person
vetoes, static-structure and implausible-aspect seed rejections all carry a
frame_idx and a box. This module decodes those frames post-job and writes
HardNegativeLogger artifacts ready for scripts/build_hardneg_dataset.py."""
from __future__ import annotations

import cv2

from .hard_negatives import HardNegativeLogger

MINE_REASONS = {
    "size_prior_violation",
    "person_overlap",
    "static_structure",
    "implausible_seed_box",
}


def _to_fullres(box, scale, y_off):
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1 / scale, y1 / scale + y_off, x2 / scale, y2 / scale + y_off]


def mine_hard_negatives(events, video_path, view_infos, out_dir,
                        max_files: int = 200) -> int:
    """Export one hard-negative artifact set per mined event. Returns the
    number of frames written. Events without a view tag, with an unknown
    view, an unmappable frame_idx, or a non-mined reason are skipped."""
    logger = HardNegativeLogger(out_dir, enabled=True, log_every=0,
                                max_files=max_files)
    # (view, source_frame) -> list of {"box", "reason"} in FULL-RES view coords
    wanted: dict[tuple, list] = {}
    for ev in events or []:
        if ev.get("reason") not in MINE_REASONS:
            continue
        det = ev.get("details") or {}
        view = det.get("view")
        info = view_infos.get(view)
        box = det.get("box")
        fidx = det.get("frame_idx")
        if info is None or box is None or fidx is None:
            continue
        sf_list = info["source_frames"]
        if not (0 <= int(fidx) < len(sf_list)):
            continue
        key = (view, int(sf_list[int(fidx)]))
        wanted.setdefault(key, []).append({
            "box": _to_fullres(box, info["scale"], info["y_off"]),
            "reason": ev.get("reason"),
        })
    if not wanted:
        return 0
    written = 0
    cap = cv2.VideoCapture(str(video_path))
    try:
        for (view, source_frame), rejected in sorted(wanted.items(),
                                                     key=lambda t: t[0][1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            crop = view_infos[view]["extract"](frame)
            if logger.maybe_log(crop, [], rejected, view_name=view,
                                frame_num=source_frame,
                                require_rejected=True) is not None:
                written += 1
    finally:
        cap.release()
    return written

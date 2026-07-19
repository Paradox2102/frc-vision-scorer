"""
FRC Vision Scorer - YOLO26s Hub-Zone Model
Clean rebuild focused on per-robot scoring attribution
"""

# Set the CUDA caching-allocator config BEFORE torch initializes CUDA (the var is
# read at the first CUDA allocation). expandable_segments lets freed blocks
# coalesce, cutting fragmentation-driven OOMs on the 24GB card. setdefault so an
# explicit shell/.env value still wins.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio
import base64
import cv2
import datetime
import hashlib
import io
import json
import math
import numpy as np
import os
import pytesseract
import re
import shutil
from PIL import Image
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yt_dlp
from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import supervision as sv

# This file is launched as `python web/app.py`, which puts web/ (not the project
# root) on sys.path — so the `web` package below would not resolve. Put the
# project root on the path so `web.embedder`/`web.reid` import the same way they
# do under pytest (which runs from the root). Harmless when already present.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.embedder import RobotEmbedder, cosine_similarity, _apply_mask
from web.job_config import JobConfig
from web import job_config as job_config_registry
from web.legibility import legibility_score
from web import reid
from web.diagnostic import DiagnosticRecorder, NullRecorder
from web.field import store as field_store
from web.field.sam2_tracker import TrackStopped
from web.field.match_order import select_epoch
from web.field.field_model import FieldModel
from web.field.calibration import (CameraCalibration, Calibrator, CalibrationError,
                                   holdout_field_error, tag_alignment_error, MIN_POSE_TAGS)
from web.field.pose_autofix import resolve_pose_from_video, PoseFixConfig
from web.field.apriltag_detect import detect_tags
from web.field.drift_monitor import DriftMonitor
from web.field.composite_layout import load_layout
from web.field.tracker import FieldTracker, FieldDetection
from web.field.occlusion import (
    _merge_occlusion,
    hub_prisms,
    occluded_by_hub,
    resolve_occlusion,
)
from web.field.occlusion_portal import predict_exit, path_crosses_footprint
from web.field.overlay_geometry import (
    bump_wireframe_edges,
    hub_wireframe_edges,
    structure_footprints,
    tower_wireframe_edges,
    trench_wireframe_edges,
)
from web.field.detect_bridge import (
    to_field_detections,
    field_adjacent_persons,
    box_exceeds_frac,
    corner_detection_weight,
    filter_field_robot_boxes,
)
from web.field.detect_schema import (
    DetectionSchema,
    bumper_admissible,
    expand_bumper_to_robot,
    blob_window_open,
    box_ios,
    dedupe_against,
    dedupe_seed_robots,
    diff_box_energy,
    merge_robot_detections,
    motion_blob_boxes,
    motion_diff,
    partition_low_conf_movers,
    plausible_bumper_box,
    reject_ball_pile_candidates,
    reject_person_candidates,
    select_gap_fills,
)
from web.field.hard_negatives import HardNegativeLogger
from web.field.detection_archive import DetectionArchive
from web.field.hardneg_miner import mine_hard_negatives
from web.field.number_lock import ConsecutiveValueTracker
from web.field.radar import Radar, RadarTrack
from web.field.manta import overlay_paths_for_job
from web.field.appearance_gallery import AppearanceGallery, classify_match
from web.field.camera_view import load_views
from web.field.view_stride import build_view_frame_plan, scale_frame_knob
from web.field.fusion import fuse_detections, merge_same_id_observations
from web.field.fused_track_id import FusedTrackRegistry
from web.hud_util import select_coasting_ghosts, draw_dashed_box
from web.field.bumper import bumper_band_box, bumper_crop
from web.field.zones import (
    inflate_zones,
    load_zones,
    point_in_zones,
    prism_bounding_zones,
    ray_enters_zones,
    save_zones,
    start_zones,
)
from web.replay_recorder import ReplayRecorder
from web import database
from web.field.identity import IdentityRegistry, RobotIdentity, IdentityObservation
from web.field.identity_confidence import (
    ConfidenceConfig,
    best_owned_team,
    blend_read_confidence,
    decide_vlm_read,
    dedupe_batch_team_reads,
    digit_token_conf,
    normalize_surface,
    occlusion_blocks_read,
    prob_vote_multiplier,
    surface_vote_multiplier,
    tentative_claim_winners,
    vote_ray_cross_m,
)
from web.field.identity_ambiguity import CoOcclusionTracker
from web.field.identity_diagnostics import IdentityDiagnosticRecorder, DiagnosticEvents, DiagnosticReasons
from web.field.track_evidence import MatchEvidence
from web.field.field_qc import track_is_static, parse_object_class

# yt-dlp needs a JavaScript runtime (deno) to unlock YouTube's high-res (720p/1080p)
# formats — without it YouTube only serves a 360p progressive stream. The deno
# installer drops the binary in ~/.deno/bin, which usually isn't on the PATH of the
# shell that launches this app, so put it there ourselves if present.
_deno_bin = os.path.expanduser("~/.deno/bin")
if os.path.isdir(_deno_bin) and _deno_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _deno_bin + os.pathsep + os.environ.get("PATH", "")


def jsonify(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # any other numpy scalar -> native Python
        return obj.item()
    return obj


def _box_iou(box_a, box_b):
    """Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union == 0.0:
        return 0.0
    return inter / union


def _on_field(xy, dims, margin=1.0):
    """Check if field coordinate (x, y) is on the field bounds plus some margin."""
    L, W = dims
    return -margin <= xy[0] <= L + margin and -margin <= xy[1] <= W + margin


def _restore_sam2_mask_to_view(
    mask2d: np.ndarray,
    view_shape,
    top_offset_px: int = 0,
) -> np.ndarray:
    """Map a SAM2 mask from its top-cropped frame back into full-view pixels."""
    full_h, full_w = int(view_shape[0]), int(view_shape[1])
    top_offset_px = max(0, min(int(top_offset_px), full_h))
    tracked_h = full_h - top_offset_px
    restored = np.zeros((full_h, full_w), dtype=bool)
    if tracked_h <= 0 or mask2d is None or np.asarray(mask2d).size == 0:
        return restored
    resized = cv2.resize(
        np.asarray(mask2d, dtype=np.uint8),
        (full_w, tracked_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    restored[top_offset_px:, :] = resized
    return restored



# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
# Prefer the 2026cur hard-negative distill (kills the OVERFLOW-bin robot-class
# false positive that corroborated the qm10/qm11 equipment hijacks: bin FP
# 44/47 -> 1/47 sampled auto frames, parked-robot recall preserved), with the
# base model as fallback. Override per-process via FRC_HUB_MODEL.
_HUB_MODEL_HARDNEG = PROJECT_ROOT / "runs/train/hub_zone_yolo26s_2026cur_v2/weights/best.pt"
_HUB_MODEL_BASE = PROJECT_ROOT / "runs/train/hub_zone_yolo26s/weights/best.pt"
MODEL_PATH = (
    Path(os.environ["FRC_HUB_MODEL"]).expanduser() if os.environ.get("FRC_HUB_MODEL")
    else _HUB_MODEL_HARDNEG if _HUB_MODEL_HARDNEG.exists()
    else _HUB_MODEL_BASE
)
BUMPER_MODEL_HARDNEG_PATH = (
    PROJECT_ROOT / "runs/train/frc_bumpers_equipment_hardneg/weights/best.pt"
)
BUMPER_MODEL_BASE_PATH = PROJECT_ROOT / "runs/train/frc_bumpers/weights/best.pt"


def _resolve_bumper_model_path(
    override: Optional[str] = None,
    hardneg_path: Path = BUMPER_MODEL_HARDNEG_PATH,
    base_path: Path = BUMPER_MODEL_BASE_PATH,
) -> Path:
    """Prefer the latest hard-negative retrain, with a base-model fallback."""
    if override:
        return Path(override).expanduser()
    if hardneg_path.exists():
        return hardneg_path
    return base_path


BUMPER_MODEL_PATH = _resolve_bumper_model_path(os.environ.get("FRC_BUMPER_MODEL"))
ROBOT_BOX_FROM_BUMPER_FACTOR = float(
    os.environ.get("ROBOT_BOX_FROM_BUMPER_FACTOR", "3.0")
)
STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
# Frames where the detector fired on likely non-robots (gated by HARD_NEG_LOG=1),
# accumulated per event/run for hard-negative retraining.
HARD_NEG_DIR = Path(__file__).resolve().parents[1] / "data" / "hard_negatives"
REPLAYS_DIR = Path(__file__).parent / "replays"
REPLAYS_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
SCORE_REGIONS_FILE = PROJECT_ROOT / "score_regions.json"
CALIB_TEST_VIDEO = Path(__file__).resolve().parents[1] / "data/test_footage/qual1_first_champs_2026.mp4"
# Per-camera crop regions now come from data/field/composite_layout.json (load_layout).

# --- Per-match logging -------------------------------------------------------
# Mirror everything this module print()s into any open match-log file(s) so a
# full, timestamped record of the pipeline's actions and errors is saved for
# later review. Terminal output is unchanged; files only receive copies while a
# job's log is open. (Designed for one match at a time — concurrent jobs would
# interleave into each other's logs.)
_real_print = print            # capture the builtin before shadowing it below
_active_job_logs = []          # open log file objects to mirror prints into


def _logging_print(*args, **kwargs):
    _real_print(*args, **kwargs)
    if not _active_job_logs:
        return
    sep, end = kwargs.get("sep", " "), kwargs.get("end", "\n")
    stamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{stamp}] " + sep.join(str(a) for a in args) + end
    for f in list(_active_job_logs):
        try:
            f.write(line)
            f.flush()
        except Exception:
            pass


print = _logging_print  # all bare print() in this module now mirrors to job logs
# -----------------------------------------------------------------------------

OVERHEAD_CROP_RATIO = 0.67
FRAME_SKIP = 2
LOST_TRACK_BUFFER = 150
MAX_ROBOTS_PER_ALLIANCE = 3
MAX_ROBOTS_TOTAL = 6  # Absolute max: 3 blue + 3 red on the field

# OCR regions defined at 1920x1080 reference resolution (x, y, w, h)
# Scaled proportionally to actual video resolution at runtime
# These are the default regions; can be overridden per event via API
OCR_REGIONS_1080P = {
    "blue_center": (775, 63, 120, 66),
    "red_center": (1015, 59, 124, 68),
    "match_clock": (905, 68, 118, 56),
}

# Custom OCR regions per event (set via API)
custom_ocr_regions: Optional[Dict] = None

# Hub positions per event (set during setup, static for entire event)
# Format: {"blue": [x, y], "red": [x, y]} in overhead crop coordinates at reference resolution
event_hub_positions: Optional[Dict] = None

def scale_ocr_regions(video_width: int, overhead_height: int) -> dict:
    """Scale OCR regions from reference overhead crop resolution to actual overhead crop.
    
    OCR regions are drawn on the overhead crop (top 67% of frame) in the setup UI.
    They must be applied to the overhead crop frame, NOT the full frame.
    Reference: 1920x723 (67% of 1920x1080).
    """
    regions = {**OCR_REGIONS_1080P, **(custom_ocr_regions or {})}
    sx = video_width / 1920.0
    sy = overhead_height / 723.0  # Scale to actual overhead crop height
    scaled = {}
    for key, (x, y, w, h) in regions.items():
        scaled[key] = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
    return scaled
OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"
MATCH_CLOCK_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789:"
MATCH_CLOCK_SAMPLE_SECONDS = 0.5
MATCH_CLOCK_LEAD_SECONDS = 0.5
MATCH_CLOCK_TAIL_SECONDS = 3.0
TITLE_SCREEN_FRAMES = 180
EARLY_OCR_FRAME = 200
OCR_FRAME_INTERVAL = 5  # OCR every 5th processed frame
BUMPER_OCR_FRAME_INTERVAL = 15  # Run bumper OCR every 15th processed frame (slower, GPU-heavy)
OCR_TEAM_CONFIRM_VOTES = 3  # Votes needed to lock in a team number from continuous OCR
APPEARANCE_CONFIRM_VOTES = 8  # Votes needed to confirm via appearance (higher bar — histograms are noisier)

# Embedding re-ID (DINOv2)
EMBED_MODEL_NAME = "facebook/dinov2-small"
APP_FLOOR = 0.55          # min cosine to merge a new detection into a track
POS_GATE_PX = 200.0       # max pixel distance to consider a merge
W_POS, W_APP = 0.4, 0.6   # continuity score weights
ASSIGN_MAX_COST = 0.45    # max Hungarian cost (≈ min cosine 0.55) to accept a label
IDENTITY_INTERVAL = 15    # run joint identity assignment every Nth processed frame
EMBED_EMA_ALPHA = 0.3     # weight of the new embedding in the running EMA

LEGIBILITY_FLOOR = 0.005       # crops below this are not worth a Qwen call
VLM_TOP_K = 6                  # max crops to send to Qwen per pass (whole field)
# One padded generate over all due crops instead of one generate per crop
# (2-4x read throughput; same model, same prompts). Kill-switch for A/B.
VLM_BATCH_READS = os.environ.get("VLM_BATCH_READS", "1") not in ("0", "false", "False")
# Fold the model's own per-digit token probability into read confidence
# (sqrt-damped in blend_read_confidence): a wavering read loses its
# full-weight vote instead of riding the 0.95 exact-roster-match ladder.
VLM_LOGPROB_CONFIDENCE = os.environ.get("VLM_LOGPROB_CONFIDENCE", "1") not in ("0", "false", "False")
# Server-side structured output for vLLM reads: constrain generation to the
# reply schema (OpenAI response_format json_schema) so a read can never be
# lost to malformed JSON. Automatic single retry without the schema on an
# HTTP 400 (older vLLM builds). Kill-switch for A/B.
VLM_GUIDED_JSON = os.environ.get("VLM_GUIDED_JSON", "1") not in ("0", "false", "False")
# Live-view frame streaming: display sends are rate-capped and pushed through
# a latest-only queue so a slow websocket client never back-pressures the
# inference loop. The replay recorder keeps its own full-rate copy.
FRAME_STREAM_INTERVAL_SEC = float(os.environ.get("FRAME_STREAM_INTERVAL_SEC", "0.1"))
FRAME_STREAM_JPEG_QUALITY = int(os.environ.get("FRAME_STREAM_JPEG_QUALITY", "70"))
# Upper bound on any single websocket send. uvicorn's ws send can await
# indefinitely on a stalled transport — even across a client disconnect —
# which left a finished job wedged in its completion drain for 60+ minutes
# (status already "completed", `complete` never sent, replay never saved).
# On timeout the socket is treated as failed, same as a transport error.
WS_SEND_TIMEOUT = float(os.environ.get("WS_SEND_TIMEOUT", "30"))
# Scheduling slack over WS_SEND_TIMEOUT when the completion drain waits for a
# sender task: at most ONE in-flight send resolves at the timeout (it detaches
# the socket, so every queued payload behind it short-circuits on None).
WS_STOP_GRACE = float(os.environ.get("WS_STOP_GRACE", "5"))
# Never schedule a read for a slot whose bumper is occluded in EVERY camera.
VLM_OCCLUSION_GATE = os.environ.get("VLM_OCCLUSION_GATE", "1") not in ("0", "false", "False")
AUDIT_FLOOR_SECONDS = 30.0     # min match seconds between audits for a confirmed identity
VLM_UNCONFIRMED_FAST_INTERVAL_SEC = 0.75
VLM_UNCONFIRMED_MED_INTERVAL_SEC = 1.25
VLM_UNCONFIRMED_SLOW_INTERVAL_SEC = 2.0
VLM_CONFIRMED_AUDIT_INTERVAL_SEC = 8.0
OVERTURN_VOTE_MARGIN = 3.0     # competitor team's votes must exceed current by this to overturn
VLM_DIRECT_CONTRADICTION_CONF = 0.9  # exact/high-confidence read that unlocks a wrong confirmed label
CORNER_VOTE_WEIGHT = 2.5       # corner-camera reads out-weigh a same-modality main read
CORNER_READ_INTERVAL = 15      # process-frames between corner-camera bumper read passes


def parse_match_clock_text(text: str) -> Optional[int]:
    """Parse a broadcast match clock into seconds, rejecting implausible values."""
    cleaned = re.sub(r"\s+", "", text or "")
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", cleaned)
    if match:
        minutes, seconds = map(int, match.groups())
        if seconds >= 60:
            return None
        total = minutes * 60 + seconds
    elif re.fullmatch(r"\d{1,3}", cleaned):
        total = int(cleaned)
    else:
        return None
    return total if 0 <= total <= 180 else None


def _find_clock_period(
    samples: List[Tuple[int, Optional[int]]],
    target_start: int,
    after_frame: int = -1,
) -> Optional[dict]:
    """Find one decreasing countdown, tolerating a single bad OCR sample."""
    candidates = [
        i for i, (frame, value) in enumerate(samples)
        if frame > after_frame
        and value is not None
        and target_start - 10 <= value <= target_start + 5
    ]
    for start_idx in candidates:
        start_frame, start_value = samples[start_idx]
        last_value = start_value
        previous_frame = start_frame
        running_start_frame = start_frame
        running_start_value = start_value
        end_frame = start_frame
        decreases = 0
        bad_samples = 0
        reached_near_zero = False

        def result():
            return {
                "start_frame": running_start_frame,
                "end_frame": end_frame,
                "start_seconds": running_start_value,
                "end_seconds": last_value,
            }

        for frame, value in samples[start_idx + 1:]:
            if value is None or value > last_value + 2:
                bad_samples += 1
                if bad_samples > 1:
                    if reached_near_zero:
                        return result()
                    break
                continue
            bad_samples = 0
            if value < last_value:
                if decreases == 0:
                    running_start_frame = previous_frame
                    running_start_value = last_value
                decreases += last_value - value
            last_value = min(last_value, value)
            end_frame = frame
            previous_frame = frame
            if value <= 2:
                if decreases >= max(8, int(target_start * 0.6)):
                    reached_near_zero = True
                    if value == 0:
                        return result()
                else:
                    break
        if reached_near_zero:
            return result()
    return None


def _merge_frame_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[List[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _pad_clock_period(
    label: str,
    period: dict,
    fps: float,
    total_frames: int,
    lead_frames: int,
    tail_frames: int,
) -> dict:
    """Label a raw clock period and add its padded processing window.

    Extends the end by any residual clock seconds (the countdown is sampled
    sparsely, so the true end is `end_frame` + `end_seconds`), then pads
    `lead_frames` before the countdown start and `tail_frames` after.
    """
    item = dict(period, label=label)
    if item["end_seconds"] > 0:
        item["end_frame"] = min(
            total_frames - 1,
            item["end_frame"] + int(round(item["end_seconds"] * fps)),
        )
        item["end_seconds"] = 0
    item["processing_start_frame"] = max(0, period["start_frame"] - lead_frames)
    item["processing_end_frame"] = min(
        total_frames - 1, item["end_frame"] + tail_frames
    )
    return item


def compute_active_windows(
    samples: List[Tuple[int, Optional[int]]],
    fps: float,
    total_frames: int,
    lead_seconds: float = MATCH_CLOCK_LEAD_SECONDS,
    tail_seconds: float = MATCH_CLOCK_TAIL_SECONDS,
    auto_only: bool = False,
) -> Tuple[List[Tuple[int, int]], List[dict], bool, Optional[str]]:
    """Return merged processing ranges, labeled core periods, and fallback state.

    When ``auto_only`` is set, restrict processing to the autonomous window and
    skip the teleop search entirely — auto-only ground-truth runs (vs. Manta)
    don't need teleop, and a clip that ends in auto still locks cleanly.
    """
    full_range = [(0, max(0, total_frames - 1))]
    if not samples or fps <= 0 or total_frames <= 0:
        return full_range, [], True, "missing clock samples or video metadata"

    auto = _find_clock_period(samples, 20)
    if auto is None:
        return full_range, [], True, "autonomous countdown was not found"

    lead_frames = int(round(lead_seconds * fps))
    tail_frames = int(round(tail_seconds * fps))

    if auto_only:
        item = _pad_clock_period("auto", auto, fps, total_frames, lead_frames, tail_frames)
        return (
            [(item["processing_start_frame"], item["processing_end_frame"])],
            [item],
            False,
            None,
        )

    teleop = _find_clock_period(samples, 140, after_frame=auto["end_frame"])
    if teleop is None:
        return full_range, [dict(auto, label="auto")], True, "teleop countdown was not found"

    periods = []
    padded = []
    for label, period in (("auto", auto), ("teleop", teleop)):
        item = _pad_clock_period(label, period, fps, total_frames, lead_frames, tail_frames)
        periods.append(item)
        padded.append((item["processing_start_frame"], item["processing_end_frame"]))
    return _merge_frame_ranges(padded), periods, False, None


def frame_in_ranges(frame: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in ranges)

# Appearance gallery (per-event re-ID exemplar memory)
GALLERY_ENABLED = os.environ.get("GALLERY_ENABLED", "1") not in ("0", "false", "False")
GALLERY_CONFIRM_SIM = float(os.environ.get("GALLERY_CONFIRM_SIM", "0.88"))
GALLERY_CONFIRM_GAP = float(os.environ.get("GALLERY_CONFIRM_GAP", "0.06"))
GALLERY_SUPPLEMENT_SIM = float(os.environ.get("GALLERY_SUPPLEMENT_SIM", "0.80"))
GALLERY_MAX_EXEMPLARS = int(os.environ.get("GALLERY_MAX_EXEMPLARS", "12"))
GALLERY_DEDUP_SIM = float(os.environ.get("GALLERY_DEDUP_SIM", "0.92"))
GALLERY_STRONG_VOTE_WEIGHT = float(os.environ.get("GALLERY_STRONG_VOTE_WEIGHT", "1.0"))
GALLERY_WEAK_VOTE_WEIGHT = float(os.environ.get("GALLERY_WEAK_VOTE_WEIGHT", "0.4"))
# Human-verified exemplars (created when a user corrects an identity) out-weigh
# auto-deposited ones so corrections reliably override bad OCR.
GALLERY_VERIFIED_VOTE_WEIGHT = float(os.environ.get("GALLERY_VERIFIED_VOTE_WEIGHT", "2.0"))

# SAM2 periodic re-prompting: re-run the detector every REPROMPT_SECONDS of
# tracked footage to catch robots that start occluded (e.g. behind the hub) and
# emerge mid-match. This is the new-robot ACQUISITION LATENCY: a robot is
# invisible until the next boundary. Env-tunable for A/B; note the cull and
# eviction evidence windows are counted in boundaries, so halving this also
# halves their time horizons.
REPROMPT_SECONDS = float(os.environ.get("REPROMPT_SECONDS", "2.0") or "2.0")
# Seed after robots have begun moving out of their often-obscured starting
# positions, then run the initial SAM propagation backward to the active start.
SAM2_PROMPT_OFFSET_SECONDS = max(
    0.0,
    float(os.environ.get("SAM2_PROMPT_OFFSET_SECONDS", "3.0") or "3.0"),
)
MAX_TRACKED_ROBOTS = 6         # FRC match maximum (3 per alliance); caps phantom tracks

# Resolution of the JPEG frames exported for SAM2 tracking, as a fraction of the
# view-crop resolution. SAM2 resizes every input to a fixed 1024² square, so this
# does NOT change SAM2's compute or VRAM — but 0.5 hands SAM2 a ~960px-wide image,
# *below* its native input, discarding detail for small/far robots (sliver-prone
# masks past midfield). Raising it costs only export disk/decode time plus mask
# RAM in sam2_raw (output masks come back at export resolution: 1.0 is 4x the
# bool-mask memory of 0.5). Env-tunable for A/B.
SAM2_EXPORT_SCALE = float(os.environ.get("SAM2_EXPORT_SCALE", "0.5") or "0.5")

# Honor the per-camera calibrated top exclusion by default. This prevents stands,
# outposts, and other above-field structures from becoming robot tracks. It remains
# disableable for camera layouts where cropping materially hurts SAM2.
ROBOT_TRACK_TOP_CROP_ENABLED = os.environ.get(
    "ROBOT_TRACK_TOP_CROP_ENABLED", "1"
) not in ("0", "false", "False")

# User-drawn static-structure zones are inflated by this margin (meters) at every
# seed-gating site. Corner cams project a structure's ground contact with enough
# range error to land just outside a tightly drawn rect (2026hop qm28: the red
# outpost projected 17cm outside its zone and held a phantom slot for ~60s).
ZONE_SEED_MARGIN_M = float(os.environ.get("ZONE_SEED_MARGIN_M", "0.4") or "0.4")

# Cameras at or below this height get the ray-beyond-contact zone test at SAM2/3
# seed sites: a low camera sees a structure's base occluded (balls, wall lip), so
# the contact projects SHORT of the structure — qm28 corner_left put the blue
# outpost 1.9m inside the field, beyond any sane point margin. High cameras have
# reliable contacts and robots legitimately park flush to hubs, so they keep the
# plain point test. 0 disables ray gating entirely.
ZONE_RAY_GATE_CAM_Z_M = float(os.environ.get("ZONE_RAY_GATE_CAM_Z_M", "1.5") or "1.5")

# SAM2 model size — the single biggest speed lever.  Robots are large, high-contrast
# blobs, so the "small" Hiera encoder tracks them well at ~3-5x the throughput of the
# "large" model (which ran ~650-730 ms/frame, i.e. ~hours per match).  Override via
# `SAM2_MODEL_SIZE=large python web/app.py` when maximum mask fidelity is needed.
# All variants resize every frame to a 1024² square, so per-frame VRAM (and thus the
# SAM2_MAX_GPU_FRAMES budget below) is identical across sizes — only compute differs.
# sam3 default since the June 10 A/B: better identity (qm24 6/6, qm72 5/6 vs
# 4/3 and 4/4), identical scores, half the wall-clock on both matches.
TRACKING_BACKEND = os.environ.get("TRACKING_BACKEND", "sam3").lower()
if TRACKING_BACKEND not in {"sam2", "sam3"}:
    TRACKING_BACKEND = "sam2"
SAM2_MODEL_SIZE = os.environ.get("SAM2_MODEL_SIZE", "small").lower()
# Config paths are resolved by SAM2's hydra search path (package-relative); the
# checkpoint paths are OURS and must be absolute — a CWD-relative path silently
# fell back to Kalman tracking when the server was started from web/ (start.sh).
_SAM2_MODELS = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  str(PROJECT_ROOT / "checkpoints/sam2.1_hiera_tiny.pt")),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  str(PROJECT_ROOT / "checkpoints/sam2.1_hiera_small.pt")),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", str(PROJECT_ROOT / "checkpoints/sam2.1_hiera_base_plus.pt")),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  str(PROJECT_ROOT / "checkpoints/sam2.1_hiera_large.pt")),
}
SAM2_CONFIG, SAM2_CHECKPOINT = _SAM2_MODELS.get(SAM2_MODEL_SIZE, _SAM2_MODELS["small"])
# Display metadata for the UI model picker (ordered fast -> precise).
_SAM2_SIZE_INFO = [
    ("tiny",      "Tiny",      "~39M params · fastest, coarsest masks"),
    ("small",     "Small",     "~46M params · default, fast"),
    ("base_plus", "Base+",     "~81M params · balanced"),
    ("large",     "Large",     "~224M params · slowest, finest masks"),
]


def _resolve_sam2_model(size: str):
    """Map a UI size string to (config, checkpoint), falling back to 'small'."""
    return _SAM2_MODELS.get((size or "").lower(), _SAM2_MODELS["small"])


def _sam2_checkpoint_present(size: str) -> bool:
    """True if the .pt checkpoint for a size is downloaded (paths are repo-relative)."""
    cfg = _SAM2_MODELS.get((size or "").lower())
    return bool(cfg) and os.path.exists(cfg[1])


# EXPERIMENTAL: torch.compile the SAM2 image encoder via the VOS-optimized predictor
# for a further ~2x.  Disabled by default — its CUDA-graph capture is incompatible with
# our multi-chunk re-prompting flow (we read mask tensors every frame and re-enter
# propagate_in_video per chunk, which trips "accessing tensor output of CUDAGraphs that
# has been overwritten").  Only safe with reprompt disabled.  Enable via SAM2_VOS_OPTIMIZED=1.
SAM2_VOS_OPTIMIZED = os.environ.get("SAM2_VOS_OPTIMIZED", "0") not in ("0", "false", "False")
SAM3_USE_FA3 = os.environ.get("SAM3_USE_FA3", "0") not in ("0", "false", "False")
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT") or None

# SAM2 tracking frame rate — the single biggest *controllable* speed lever.  Total SAM2
# time is near-linear in the number of tracked frames, and frame count = source_fps /
# this value.  Robots move slowly relative to frame rate and the field tracker
# interpolates between samples, so 8 fps tracks them fine while a 60 fps match drops from
# ~4223 to ~1100 tracked frames.  Override via SAM2_TRACK_FPS.  (Previously this was
# implicitly forced to 20 fps whenever VRAM >= 16 GB — exactly backwards for speed.)
SAM2_TRACK_FPS = float(os.environ.get("SAM2_TRACK_FPS", "8.0") or "8.0")

# VRAM-fit tunable knob (speed/quality experiment).  When > 0, the SAM2 pass keeps
# all video + state tensors on the GPU (no CPU offload, which removes the per-frame
# host->device copy that dominates runtime) and raises frame_skip so the tracked
# segment stays within this many frames.  Trades temporal resolution for speed.
# Set via env, e.g. `SAM2_MAX_GPU_FRAMES=1200 python web/app.py`.  0 = disabled.
SAM2_MAX_GPU_FRAMES = int(os.environ.get("SAM2_MAX_GPU_FRAMES", "0") or "0")
# SAM2 stores each frame as float32 (3 x 1024 x 1024) = 12.0 MiB of GPU memory.
SAM2_FRAME_GPU_MIB = 12.0


def _hud_flag(name: str, default: str) -> bool:
    """Read a boolean HUD overlay toggle from the environment ('0'/'false' = off)."""
    return os.environ.get(name, default) not in ("0", "false", "False")


# HUD overlay toggles.  Now that SAM2 masks highlight each robot's silhouette, the
# extra per-robot bounding box, ground-contact plane, and velocity arrow are redundant
# clutter (see the busy overlay), so they default OFF.  The mask highlight, alliance
# bumper band, and team/slot label carry the useful information and stay ON.  Flip any
# of these back via env (e.g. `HUD_DRAW_ROBOT_BOX=1 python web/app.py`).
HUD_DRAW_ROBOT_MASK     = _hud_flag("HUD_DRAW_ROBOT_MASK", "1")      # SAM2 silhouette highlight
HUD_DRAW_BUMPER_BAND    = _hud_flag("HUD_DRAW_BUMPER_BAND", "1")     # alliance-color bumper band
HUD_DRAW_ROBOT_LABEL    = _hud_flag("HUD_DRAW_ROBOT_LABEL", "1")     # team #/slot/confidence text
HUD_DRAW_ROBOT_BOX      = _hud_flag("HUD_DRAW_ROBOT_BOX", "0")       # full bounding box (redundant w/ mask)
HUD_DRAW_GROUND_PLANE   = _hud_flag("HUD_DRAW_GROUND_PLANE", "0")    # bumper ground-contact polygon
HUD_DRAW_VELOCITY_ARROW = _hud_flag("HUD_DRAW_VELOCITY_ARROW", "0")  # motion-direction arrow
HUD_DRAW_BALL_BOXES     = _hud_flag("HUD_DRAW_BALL_BOXES", "0")      # per-ball detection boxes (hundreds = noise)
HUD_DRAW_COASTING_GHOST = _hud_flag("HUD_DRAW_COASTING_GHOST", "1")  # dashed last-known box for coasting tracks

# Bumper-number reading VLM. Loaded via AutoModelForImageTextToText, so any
# transformers-supported image-text-to-text model works.  Override via VLM_MODEL_NAME
# env to A/B a smaller/faster reader without code edits — the Qwen3-VL family ships
# 2B and 4B Instruct variants that are drop-in (same processor/chat-template API) and
# much faster than 8B for digit transcription, e.g.:
#   VLM_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct python web/app.py   # fall back to 8B
# Default is the 4B Instruct variant: drop-in with the 8B (same processor/chat API),
# much faster, and adequate for bumper-digit transcription.
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")
# team_reader="vllm": same model served by a vLLM OpenAI-compatible server
# (web/start_vllm.sh). Continuous batching replaces VLM_BATCH_READS, and the
# weights live in the vLLM process, so job pruning here never reloads them.
VLM_SERVER_URL = os.environ.get("VLM_SERVER_URL", "http://127.0.0.1:8006/v1")

# Match-start calibration drift check: sample a few in-match frames from the main
# view, auto-detect AprilTags, and verify the selected calibration epoch still
# reprojects them accurately. If the camera moved/zoomed since calibration the job
# aborts (needs recalibration) BEFORE the expensive SAM2 pass.
CALIB_CHECK_ENABLED = os.environ.get("CALIB_CHECK_ENABLED", "1") not in ("0", "false", "False")
CALIB_DRIFT_MAX_M = float(os.environ.get("CALIB_DRIFT_MAX_M", "0.5"))
CALIB_CHECK_MIN_TAGS = int(os.environ.get("CALIB_CHECK_MIN_TAGS", "2"))
CALIB_CHECK_SAMPLE_FRAMES = int(os.environ.get("CALIB_CHECK_SAMPLE_FRAMES", "5"))
# On drift, attempt an automatic pose re-solve (lens fixed, only pose re-solved from
# AprilTags) and stage it for HUMAN APPROVAL rather than aborting outright. The
# operator approves (promotes to a new epoch) or rejects (falls back to today's
# manual recalibration). Set to 0 to keep the old abort-straight-to-manual behavior.
CALIB_AUTOFIX_ENABLED = os.environ.get("CALIB_AUTOFIX_ENABLED", "1") not in ("0", "false", "False")

# Mid-match drift monitor: low-cadence AprilTag re-checks on frames the main
# loop already decoded. Sustained drift WARNS and records (never aborts — the
# match is already running); the calibrate page picks the record up like a
# match-start failure. ~8 detect_tags calls per 150s match at the default.
CALIB_MONITOR_ENABLED = os.environ.get("CALIB_MONITOR_ENABLED", "1") not in ("0", "false", "False")
CALIB_MONITOR_INTERVAL_S = float(os.environ.get("CALIB_MONITOR_INTERVAL_S", "20"))

# Minimum YOLO detection confidence. Raised from the 0.25 default to cut false-positive
# "robot" detections (people, carts) that otherwise steal canonical slots and inherit labels.
DET_CONF_THRESHOLD = 0.35

PERSON_OVERLAP_IOS = float(os.environ.get("PERSON_OVERLAP_IOS", "0.3"))
PERSON_ROBOT_REJECT_COVERAGE = float(
    os.environ.get("PERSON_ROBOT_REJECT_COVERAGE", "0.6")
)

BACKWARD_CONFIRM_ENABLED = os.environ.get("BACKWARD_CONFIRM_ENABLED", "1") == "1"
# Re-resolution must retain at least this fraction of the live replay rows, else it
# falls back to the slice-2 prune (guards against identity-churn gutting the replay).
RERESOLVE_MIN_KEEP_FRAC = float(os.environ.get("RERESOLVE_MIN_KEEP_FRAC", "0.5"))
# Per-segment resolution: split a slot's track at gap-jumps and drop phantom /
# wrong-robot segments inside an otherwise-robot slot (do-no-harm). Nested under
# backward-confirm; off -> exactly the prior per-slot behavior.
SEGMENT_RESOLVE_ENABLED = os.environ.get("SEGMENT_RESOLVE_ENABLED", "1") == "1"
# Debug: when set, dump a per-frame fused-observation + slot-assignment trace to
# /tmp/frc-tracker-trace.json to diagnose multi-camera slot churn. Off by default.
TRACKER_TRACE = os.environ.get("TRACKER_TRACE", "") == "1"

# Whole-robot gap-fill parameters
ROBOT_FILL_ENABLED = os.environ.get("ROBOT_FILL_ENABLED", "1") == "1"
ROBOT_FILL_CONF = float(os.environ.get("ROBOT_FILL_CONF", str(DET_CONF_THRESHOLD)))
ROBOT_FILL_IOS = float(os.environ.get("ROBOT_FILL_IOS", "0.5"))
# Motion-tier subordination: a motion hint is deduped against YOLO-tier
# boxes inflated by MOTION_TIER_INFLATE per side, at this (deliberately low)
# IoS — adjacency to a YOLO robot counts as redundancy for a hint.
MOTION_TIER_DEDUP_IOS = float(os.environ.get("MOTION_TIER_DEDUP_IOS", "0.1"))
MOTION_TIER_INFLATE = float(os.environ.get("MOTION_TIER_INFLATE", "0.25"))
# Colorless candidates >= this fraction covered by ball detections are ball
# piles, not robots (the whole-robot class fires on dense piles).
BALL_PILE_COVERAGE = float(os.environ.get("BALL_PILE_COVERAGE", "0.5"))
ROBOT_FILL_MAX_BOX_FRAC = float(os.environ.get("ROBOT_FILL_MAX_BOX_FRAC", "0.5"))
# Relaxed cap used when det_frac_backstop="uncalibrated": a robot close to a
# low corner camera legitimately spans most of the short panel's height, and
# 0.5 made the whole-robot fill blind exactly where corners matter (qm10 4027).
ROBOT_FILL_MAX_BOX_FRAC_CLOSE = float(
    os.environ.get("ROBOT_FILL_MAX_BOX_FRAC_CLOSE", "0.9"))

# Scoring parameters
BALL_PROXIMITY_FRAMES = 30  # Look back this many frames for ball near robot
HUB_DISTANCE_HIGH_CONF = 150  # px
HUB_DISTANCE_MED_CONF = 300   # px

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(title="FRC Vision Scorer")

# Global state for inference jobs
jobs: Dict[str, "InferenceJob"] = {}

# One job runs at a time by default: SAM2 + the VLM saturate the GPU, and a
# concurrent job forces SAM2 state offload to CPU (5-6x slower corners) and
# confounds A/B timing. Extra submissions queue FIFO-ish and the waiting
# client gets "queued" heartbeats. Raise for multi-GPU / spare-VRAM setups.
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1") or "1")
_job_slots = asyncio.Semaphore(max(1, MAX_CONCURRENT_JOBS))
_QUEUE_POLL_SECONDS = 2.0
# Finished jobs kept in `jobs` for inspection; older ones are pruned at the
# next submit so completed jobs stop pinning their YOLO models and frame
# state in memory across a long event day.
JOB_RETENTION = int(os.environ.get("JOB_RETENTION", "4") or "4")


def _prune_finished_jobs() -> None:
    """Drop all but the newest JOB_RETENTION finished jobs (insertion order).
    Jobs that are still initializing/queued/processing are never pruned."""
    finished = [job_id for job_id, job in jobs.items()
                if job.status in ("completed", "error")]
    for job_id in finished[:max(0, len(finished) - JOB_RETENTION)]:
        jobs.pop(job_id, None)


async def _run_job_serialized(job: "InferenceJob") -> None:
    """Run process_video once a job slot frees up.

    While queued, the client gets a heartbeat every poll; a stop request or a
    dead websocket abandons the run before it starts. With multiple waiters,
    ordering is best-effort (each poll rejoins the semaphore queue)."""
    waited = False
    while True:
        try:
            await asyncio.wait_for(_job_slots.acquire(), timeout=_QUEUE_POLL_SECONDS)
            break
        except asyncio.TimeoutError:
            waited = True
            if job._stopped:
                job.status = "error"
                job.error = "stopped while queued"
                await job._send_ws({"type": "error", "error": job.error})
                return
            job.status = "queued"
            sent = await job._send_ws({
                "type": "status", "status": "queued",
                "message": "Waiting for a free job slot…",
            })
            if not sent and job.websocket is None:
                job.status = "error"
                job.error = "client disconnected while queued"
                print(f"  [queue] job {job.job_id} abandoned: client disconnected")
                return
    try:
        if job._stopped:
            job.status = "error"
            job.error = "stopped while queued"
            await job._send_ws({"type": "error", "error": job.error})
            return
        if waited:
            await job._send_ws({"type": "status", "status": "processing",
                                "message": "Job slot free — starting…",
                                "progress": 0.0})
        await job.process_video()
    finally:
        _job_slots.release()


# Global state for current event session
current_event: Optional[Dict] = None


@dataclass
class DetectedHub:
    """Hub position detected from hub_opening class"""
    x: float
    y: float
    alliance: str  # 'blue' or 'red'
    # Bounding box for overlap detection
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


@dataclass
class RobotTrack:
    """Canonical robot track after consolidation"""
    robot_id: int
    alliance: str
    team_number: Optional[str]
    last_position: Tuple[float, float]
    positions_history: deque  # deque of (frame_num, x, y)
    scores: int = 0
    # Alliance voting: only flip after consistent detections
    blue_votes: int = 0
    red_votes: int = 0
    # Continuous OCR voting: accumulate team number reads over time
    team_ocr_votes: Optional[Dict[str, float]] = None  # team_number -> vote weight
    team_number_confirmed: bool = False  # True once OCR or calibration locked it in
    # Appearance embedding for re-ID (DINOv2, EMA-updated)
    appearance_embedding: Optional[np.ndarray] = None
    appearance_votes: Optional[Dict[str, float]] = None  # team_number -> vote score
    # Latest masked whole-robot crop (base64 JPEG) — the image behind the
    # appearance signature, shown in the gallery / replay-review UI.
    appearance_crop_b64: Optional[str] = None
    # Diagnostic / embedding-freshness state. Set when a re-ID match resolves
    # to this canonical; read by _emit_diagnostic_snapshot. None sentinels are
    # load-bearing — frame 0 is a real, valid frame number, so we cannot use 0
    # to mean "never set" without colliding with legitimate frame-0 values.
    last_embedding_frame: Optional[int] = None
    last_reid_similarity: Optional[float] = None
    last_reid_gap: Optional[float] = None
    last_seen_frame: Optional[int] = None
    # Field-space position (meters) from the FieldTracker. last_position stays
    # pixel-space for the (unchanged) pixel-space scoring/ball math.
    field_position: Optional[Tuple[float, float]] = None
    # Consecutive-agreement team number lock state
    consecutive_tracker: Optional[Any] = None


@dataclass
class BallTrajectory:
    """Tracked ball trajectory across frames"""
    ball_id: int
    positions: List[Tuple[int, float, float]]  # [(frame_num, x, y), ...]
    alive: bool = True  # Still being tracked
    scored: bool = False  # Entered a hub
    scored_alliance: Optional[str] = None
    shooter_robot_id: Optional[int] = None
    frames_missing: int = 0  # Consecutive frames not detected


@dataclass
class ScoringEvent:
    """A scoring attribution event"""
    frame_num: int
    timestamp: float
    alliance: str
    robot_id: int
    team_number: Optional[str]
    confidence: str  # 'high', 'medium', 'low'
    distance_to_hub: float


def _apply_gallery_vote(robot, gallery,
                        confirm_sim=GALLERY_CONFIRM_SIM, confirm_gap=GALLERY_CONFIRM_GAP,
                        supplement_sim=GALLERY_SUPPLEMENT_SIM,
                        strong_w=GALLERY_STRONG_VOTE_WEIGHT, weak_w=GALLERY_WEAK_VOTE_WEIGHT):
    """Cast a gallery-derived vote into a track's existing vote/lock state.

    Returns (verdict, team, sim) where verdict is 'confirm'|'supplement'|'none'.
    Skips confirmed tracks (never overrides a lock)."""
    if gallery is None or robot.team_number_confirmed or robot.appearance_embedding is None:
        return ("none", None, 0.0)
    color = robot.alliance if robot.alliance in ("red", "blue") else None
    ranked = gallery.query(robot.appearance_embedding, color)
    verdict, team, sim = classify_match(
        ranked, confirm_sim=confirm_sim, confirm_gap=confirm_gap, supplement_sim=supplement_sim)
    if verdict == "none" or not team:
        return ("none", None, sim)
    # A human-verified top match out-weighs the normal strong/weak vote so that
    # user corrections reliably override bad OCR. ranked holds (team, sim, verified).
    best_verified = bool(ranked[0][2]) if ranked else False
    if best_verified:
        weight = GALLERY_VERIFIED_VOTE_WEIGHT
    else:
        weight = strong_w if verdict == "confirm" else weak_w
    if robot.team_ocr_votes is None:
        robot.team_ocr_votes = {}
    robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + weight
    if verdict == "confirm":
        if robot.consecutive_tracker is None:
            robot.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
        locked_team, confirmed = robot.consecutive_tracker.update(team)
        if confirmed and locked_team:
            robot.team_number = locked_team
            robot.team_number_confirmed = True
    return (verdict, team, sim)


class _LazyCrop:
    """Deferred bumper-crop encode. Detections carry one of these instead of an
    eager base64 string; the VLM dispatcher resolves it only for the top-K
    candidates it actually sends (most processed frames dispatch no read at
    all, and each pass sends at most VLM_TOP_K of the ~6-18 per-frame crops).
    Memoized so status payloads / crop capture reuse the first encode."""
    __slots__ = ("_fn", "_done", "_val")

    def __init__(self, fn):
        self._fn = fn
        self._done = False
        self._val = None

    def __call__(self):
        if not self._done:
            self._val = self._fn()
            self._done = True
            self._fn = None  # release the captured view frame
        return self._val


def _resolve_crop(crop):
    """Evidence crops are either eager base64 strings (tests, corner sweep) or
    _LazyCrop thunks (per-frame detections). Returns str or None."""
    return crop() if callable(crop) else crop


class JobStopped(Exception):
    """Raised inside ``process_video`` when the user's STOP flag is seen.

    Every long phase runs on a worker thread (``asyncio.to_thread``) that can't
    be killed from outside, so the pipeline polls ``self._stopped`` at each
    phase boundary and inside the hot loops and raises this to unwind. The
    ``process_video`` handler turns it into a terminal ``error`` state (unless
    the analysis loop has already started, which keeps its partial-results
    behavior) and always runs the ``finally`` cleanup.
    """


class InferenceJob:
    """Manages a single video inference job"""

    def __init__(self, job_id: str, video_path: Path, team_reader: str = "local",
                 match_key: Optional[str] = None,
                 config: Optional[JobConfig] = None):
        self.job_id = job_id
        self.video_path = video_path
        # Per-job knob values (override > env > default), validated at submit.
        # Legacy callers that don't pass a config get pure env/defaults.
        self.config = config if config is not None else JobConfig()
        if config is not None:
            self.team_reader = self.config.team_reader
        else:
            self.team_reader = team_reader if team_reader in ("local", "deepseek", "vllm") else "local"
        self.status = "initializing"
        self.error: Optional[str] = None
        # Match this job processes; drives calibration-epoch selection. The
        # __init__ plumbing that populates this from the job request lands in a
        # separate change; None here means select_epoch() falls back to latest.
        self.match_key = getattr(self, "match_key", None)
        self._calib_version_id = None
        self.needs_calibration = None
        # Mid-match drift monitor: real instance built in _init_field_tracking once
        # calibration is loaded. None here is also the correct disabled state for
        # any caller that bypasses that method (e.g. mocks it out in tests).
        self._drift_monitor = None
        self._drift_warned = False
        self._log_file = None
        self._log_path: Optional[Path] = None
        # Wall-clock creation time, surfaced by /api/job/active so a
        # reloaded page can show truthful start/elapsed clocks on rejoin.
        self.started_at = time.time()
        self.websocket: Optional[WebSocket] = None
        self._websocket_send_lock = asyncio.Lock()
        # First ws connection claims the run; later connections only attach
        # for streaming and wait on _done_event (no second process_video).
        self._run_claimed = False
        self._done_event = asyncio.Event()
        self._paused = False
        self._stopped = False
        self._pause_frame = 0
        self._current_frame = 0
        
        # Model and tracking. SP2 replaced ByteTrack + DINOv2 re-ID with the
        # field-space FieldTracker (built in process_video once the calibration
        # loads), so neither the ByteTracker nor the embedder is instantiated here.
        self.model = YOLO(str(MODEL_PATH))
        self.bumper_model = None
        self.det_schema = None
        if BUMPER_MODEL_PATH.exists():
            self.bumper_model = YOLO(str(BUMPER_MODEL_PATH))
            self.det_schema = DetectionSchema.from_names(self.bumper_model.names)
            print(
                f"[detect] bumper model={BUMPER_MODEL_PATH.name} "
                f"classes={self.bumper_model.names} "
                f"robots={self.det_schema.robot_class_color}"
            )
        else:
            print(
                f"[detect] bumper model not found at {BUMPER_MODEL_PATH}; "
                "falling back to existing model robot detections with unknown color"
            )

        # Resolve the whole-robot class ID once
        self.whole_robot_class_id = 1
        if self.det_schema is not None:
            for cid, col in self.det_schema.robot_class_color.items():
                if col is None:
                    self.whole_robot_class_id = cid
                    break
        # Person detector for occlusion reasoning (run per processed frame, classes=[0]).
        # Absolute path so it doesn't depend on CWD (a bare name would trigger a
        # hub auto-download attempt for a non-existent release).
        self.person_model = YOLO(str(PROJECT_ROOT / "yolo26n.pt"))

        # Field-space tracking — bound to the loaded event/camera calibration.
        # Snapshot the active event key at construction; calibration + field model
        # are loaded at the top of process_video() (decision 7 gate).
        self.event_key = (current_event or {}).get('event_key')
        self.match_key = match_key
        self.camera = "main"
        self.calib = None
        self.field_model = None
        self.hub_prisms = []
        self.hub_no_robot_zones = []
        self.static_seed_zones = []
        self.user_seed_zones = []
        self.field_tracker: Optional[FieldTracker] = None
        self.radar: Optional[Radar] = None
        self._radar_trails: dict = {}   # slot_id -> deque of recent filtered positions
        self.field_tracking = False
        # Per-frame occlusion state. `_view_flags` feeds the NEXT frame's tracker
        # (view_occluded -> longer coast); `_occ` is this frame's full result and
        # gates VLM reads (bumper_occluded). Both keyed by slot_id.
        self._view_flags: Dict[int, dict] = {}
        self._occ: Dict[int, dict] = {}
        self._occ_by_cam: Dict[str, Dict[int, dict]] = {}
        # Occlusion exit portals (knob: occlusion_portal). slot_id ->
        # (exit_x, exit_y, radius_m). Set when a slot transitions into
        # hub-occluded coasting; radius grows with coast uncertainty; cleared
        # on re-acquire/expire. Fed into the NEXT frame's tracker update.
        self._portal_state: Dict[int, tuple] = {}
        # slot_id -> last-known (x, y), for teleport-suspect logging.
        self._portal_prev_pos: Dict[int, tuple] = {}

        # Video metadata
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 0
        self._match_evidence = MatchEvidence()
        self._binding_log: list[dict] = []   # [{"frame_num": int, "bindings": {track_id: slot_id}}]
        self._identity_log: list[dict] = []   # per-frame identity snapshots for re-resolution
        self._evidence_frame = 0             # current processed frame, set at loop TOP
        self.total_frames = 0
        self.overhead_height = 0
        self.frame_skip = FRAME_SKIP
        self.target_tracking_fps = None
        self.processing_ranges: List[Tuple[int, int]] = []
        self.match_periods: List[dict] = []
        self.timer_gate_fallback = True
        self.timer_gate_fallback_reason: Optional[str] = None
        self.sam2_source_to_local: Dict[int, int] = {}
        # Per-view source-frame -> tracker-local-index maps. Corner views may
        # be decimated (corner_track_stride), so each view has its own index
        # space; main's map equals sam2_source_to_local.
        self.sam2_view_source_to_local: Dict[str, Dict[int, int]] = {}
        self.autonomous_paths: Dict[str, List[dict]] = defaultdict(list)
        # Analysis-loop stage accumulators. Also (re)initialized in the run loop
        # next to self.timings, but declared here so crop-encode timers work when
        # _bumper_crop_b64 / _masked_robot_crop_b64 fire on early pre-loop paths.
        self._loop_stage_s = defaultdict(float)
        # Per-job A/B knobs, all resolved through self.config (override > env >
        # default). The attributes survive because downstream plumbing and the
        # job-control endpoints read them directly.
        self.sam2_max_gpu_frames = self.config.sam2_max_gpu_frames
        self.model_size = self.config.sam2_model_size
        self.sam2_export_scale = self.config.sam2_export_scale
        # JobConfig consults the same env vars the tracker defaults read, so
        # passing these explicitly is equivalent to the old None-defers logic.
        self.cull_hijack_enabled: Optional[bool] = self.config.cull_hijack_enabled
        self.reanchor_enabled: Optional[bool] = self.config.reanchor_enabled
        # SAM3 remains opt-in until it has been benchmarked on the event hardware.
        self.tracking_backend = self.config.tracking_backend
        # Display tag for log lines / status messages / degradation
        # components, so SAM3 runs no longer label themselves "[SAM2]".
        self.tracker_tag = "SAM3" if self.tracking_backend == "sam3" else "SAM2"

        # Detection state
        self.hubs: List[DetectedHub] = []
        self.identity_registry = IdentityRegistry()
        self.confidence_config = ConfidenceConfig()
        self.co_occlusion_tracker = CoOcclusionTracker()
        # max_events raised well above a full multi-view match's event volume so
        # late-match diagnostics (e.g. sam2.object_culled) aren't truncated — the
        # detailed record needed to compare cull on/off runs. The uncapped
        # "sam2_objects_culled" counter is the truncation-proof backstop.
        self.identity_diagnostics = IdentityDiagnosticRecorder(max_events=100000)
        self.identity_diagnostics.set_context(job_id=self.job_id)
        self.robot_identities = self.identity_registry.identities
        self._last_bindings: Dict[int, int] = {}
        self.robot_tracks: Dict[int, RobotTrack] = {}
        self.gallery: Optional[AppearanceGallery] = None  # loaded in process_video if enabled
        self._gallery_deposits = AppearanceGallery(
            dedup_sim=GALLERY_DEDUP_SIM,
            max_exemplars=GALLERY_MAX_EXEMPLARS,
        )
        self._pending_gallery_rows: list = []  # buffered exemplar rows, flushed (awaited) at finalize
        self._slot_crops: dict = {}   # slot_id -> list of {quality, crop_b64, embedding} (top-3)
        self._slot_source: dict = {}  # slot_id -> "ocr" | "gallery" (last confirm source)
        self.sam2_obj_color: dict[str, dict[int, str | None]] = {}
        self.bytetrack_to_canonical: Dict[int, int] = {}
        self.next_canonical_id = 0
        
        # Scoring state
        self.alliance_scores = {"blue": 0, "red": 0}
        self.last_alliance_scores = {"blue": 0, "red": 0}
        self.scoring_events: List[ScoringEvent] = []
        self.deferred_attributions: List[dict] = []
        self.recent_ball_detections: deque = deque(maxlen=BALL_PROXIMITY_FRAMES)
        self.last_ocr_crops: list = []     # crops fed to the VLM + reads, for UI spot-check
        self._ocr_crops_dirty = False      # only stream crops on frames where a new batch exists
        self.vlm_status = {
            "provider": self.team_reader,
            "calls": 0,
            "last_frame": None,
            "last_candidates": 0,
            "last_sent": 0,
            "last_interval_sec": None,
            "unconfirmed": 0,
            "last_results": [],
            "last_error": None,
        }
        
        # OCR state (raw reads for authoritative score sync)
        self._last_ocr = {"blue": None, "red": None}
        self._ball_score_cooldown = {}
        
        # Ball trajectory tracking
        self.ball_trajectories: List[BallTrajectory] = []
        self.next_ball_id = 0
        self.scored_balls: List[BallTrajectory] = []  # Balls that entered hub with trajectory info
        # Max distance to link a ball detection to an existing trajectory
        self.BALL_LINK_DISTANCE = 60  # px — balls are small and fast
        self.BALL_MAX_MISSING = 3  # frames before trajectory is considered dead
        
        # Team numbers from OCR
        self.team_numbers = {"blue": [], "red": []}

        # Known match teams from TBA (set via WebSocket before inference)
        self.match_teams: List[str] = []
        self.match_blue_teams: List[str] = []
        self.match_red_teams: List[str] = []

        # Appearance re-ID: reference embeddings from calibration
        self._reference_embeddings: Dict[str, np.ndarray] = {}  # team_number -> DINOv2 embedding

        # Claude Vision: track when we've called it during inference
        self._claude_vision_calls = 0
        # Single-flight background handle for the VLM bumper-read pass. The read
        # runs as an event-loop task (create_task) so frame processing never
        # blocks on the 0.6-1.5s generate; only one pass is in flight at a time.
        self._vlm_task = None
        # Single-flight background handle + per-identity position history for
        # the observe-only equipment-adjudication pass (equipment_adjudicate).
        self._adjudicate_task = None
        self._ident_pos_history: dict[int, deque] = {}
        self._claude_vision_max_calls = 20  # Max API calls per match during inference
        self._claude_vision_interval_sec = 5  # Call every 5 seconds until all identified
        self._claude_vision_last_time = 0.0  # Timestamp of last call

        # Diagnostic recorder — writes JSONL during inference and pushes
        # live records over the existing websocket (multiplex by type).
        # Constructor already swallows OSError internally and falls back to
        # a null state (is_null=True), so no outer try/except is needed.
        self.diagnostic = DiagnosticRecorder(
            UPLOAD_DIR / f"diagnostic_{self.job_id}.jsonl",
            ws_send=None,  # wired in Task 7
        )
        # Hard-negative capture: no-op unless HARD_NEG_LOG=1. Bucketed per event
        # (falls back to job_id) so negatives accumulate across runs of an event.
        self.hard_neg_logger = HardNegativeLogger(
            HARD_NEG_DIR / (self.event_key or self.job_id)
        )
        # Per-boundary detection archive: every re-prompt boundary's YOLO
        # detections (all views, full-res coords), beside diagnostics.
        # Pure telemetry — best-effort, never alters detection/tracking.
        self.detection_archive = (
            DetectionArchive(str(UPLOAD_DIR / f"detections_{self.job_id}.jsonl"))
            if self.config.detection_archive else DetectionArchive(None)
        )

    def extract_team_numbers(self, frame: np.ndarray) -> None:
        """Extract team numbers from scoreboard using OCR on early frame"""
        if frame is None or frame.size == 0:
            return
        fh, fw = frame.shape[:2]
        # Try each OCR region
        for alliance in ["blue", "red"]:
            for region_key in [f"{alliance}_corner", f"{alliance}_center"]:
                if region_key not in self.ocr_regions:
                    continue
                x, y, w, h = self.ocr_regions[region_key]
                if x + w > fw or y + h > fh or w <= 0 or h <= 0:
                    continue
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                # Preprocess for OCR
                roi_resized = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                _, roi_thresh = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
                
                text = pytesseract.image_to_string(roi_thresh, config=OCR_CONFIG).strip()
                
                # Parse team numbers (could be multiple digits)
                if text and text.isdigit():
                    # Split into potential team numbers (4 digits each typically)
                    for i in range(0, len(text), 4):
                        team_num = text[i:i+4]
                        if len(team_num) >= 3 and team_num not in self.team_numbers[alliance]:
                            self.team_numbers[alliance].append(team_num)
                            if len(self.team_numbers[alliance]) >= MAX_ROBOTS_PER_ALLIANCE:
                                break
                
                if len(self.team_numbers[alliance]) >= MAX_ROBOTS_PER_ALLIANCE:
                    break

    def ocr_robot_bumpers(self, frame: np.ndarray, tracked_robots: list) -> None:
        """Run EasyOCR on tracked robot bumpers during inference.
        Accumulates team number votes per robot. When a team number gets enough
        votes (OCR_TEAM_CONFIRM_VOTES), it's locked in and retroactively applied.

        Partial reads are matched against known match teams:
        - "31" uniquely matches 3128 → full vote for 3128
        - "28" matches both 3128 and 3328 → half vote to each
        - Exact matches always get a full vote

        tracked_robots: list of (canonical_id, box) tuples for robots detected this frame.
        """
        if not self.match_teams:
            return  # No known teams to match against

        # Filter to teams not yet confirmed on another robot
        confirmed_teams = {
            r.team_number for r in self.robot_tracks.values()
            if r.team_number_confirmed and r.team_number
        }
        available_teams = [t for t in self.match_teams if t not in confirmed_teams]

        for canonical_id, box in tracked_robots:
            robot = self._get_slot_track_if_exists(canonical_id)
            if not robot or robot.team_number_confirmed:
                continue  # Already confirmed, skip OCR
            if self._occ.get(canonical_id, {}).get("bumper_occluded"):
                continue  # bumper hidden by hub/person/robot — an unreliable read

            result = _ocr_bumper_number(frame, box, self.match_teams)
            raw = result.get("ocr_raw", "").strip()
            if not raw or len(raw) < 2:
                continue

            # Initialize vote dict if needed
            if robot.team_ocr_votes is None:
                robot.team_ocr_votes = {}

            # Find all available teams that contain this partial read
            # "31" in "3128" ✓, "28" in "3128" ✓ and "3328" ✓
            matching_teams = [t for t in available_teams if raw in t]

            # Also check if the raw read IS a team number exactly
            if raw in available_teams and raw not in matching_teams:
                matching_teams.append(raw)

            # If _ocr_bumper_number found an exact/fuzzy match, include it too
            matched = result.get("matched_team")
            if matched and matched in available_teams and matched not in matching_teams:
                matching_teams.append(matched)

            if not matching_teams:
                continue

            # EasyOCR votes are weighted lower (0.3) since Qwen VL is more accurate
            EASYOCR_VOTE_WEIGHT = 0.3
            if len(matching_teams) == 1:
                # Unique partial match
                team = matching_teams[0]
                robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0) + EASYOCR_VOTE_WEIGHT
                
                # Update consecutive tracker
                if robot.consecutive_tracker is None:
                    robot.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
                locked_team, confirmed = robot.consecutive_tracker.update(team)
                if confirmed and locked_team and locked_team != robot.team_number:
                    already_assigned = any(
                        r.team_number == locked_team and r.team_number_confirmed
                        for r in self.robot_tracks.values() if r.robot_id != canonical_id
                    )
                    if not already_assigned:
                        old_team = robot.team_number
                        robot.team_number = locked_team
                        robot.team_number_confirmed = True
                        self._clear_team_from_other_robots(canonical_id, locked_team)
                        for event in self.scoring_events:
                            if event.robot_id == canonical_id:
                                event.team_number = locked_team
                        print(f"  OCR confirmed: Robot {canonical_id} = Team {locked_team} (was {old_team})")
            else:
                # Ambiguous partial match — split vote among candidates
                vote_share = EASYOCR_VOTE_WEIGHT / len(matching_teams)
                for team in matching_teams:
                    robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0) + vote_share

    def _raise_if_stopped(self) -> None:
        """Raise ``JobStopped`` if the user has requested a stop.

        Called at every phase boundary in ``process_video`` so a STOP set
        during a long synchronous phase takes effect promptly instead of only
        at the queue-wait / analysis-loop poll points.
        """
        if self._stopped:
            raise JobStopped()

    @contextmanager
    def _stage(self, name):
        """Accumulate wall time for one analysis-loop stage. Timers may nest
        (sub-stages are also inside their parent's span); the report labels
        which keys overlap. ~100ns overhead per use — negligible at 8fps."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            # Lazy-init: direct callers of _stage-wrapped methods (e.g. the
            # VLM read pass exercised by unit tests) build the job via __new__
            # and never enter the analysis loop that sets up the accumulator.
            acc = getattr(self, "_loop_stage_s", None)
            if acc is None:
                acc = self._loop_stage_s = defaultdict(float)
            acc[name] += time.perf_counter() - t0

    def _bumper_crop_b64(
        self,
        frame: np.ndarray,
        box,
        robot_id=None,
        enforce_floor: bool = True,
        box_is_bumper: bool = False,
    ):
        """Crop the bumper band (lower part of the robot box), upscale, and JPEG/base64
        encode it for the VLM. Returns (crop_b64, legibility_score) for usable crops, or
        None when the crop is empty or, when enforce_floor=True, below LEGIBILITY_FLOOR.

        When a non-None robot_id is supplied and the crop is below the legibility floor,
        the rejected crop is recorded as a vlm_filtered diagnostic (preserving the prior
        behavior of vision_identify_robots).
        """
        with self._stage("loop.crop_encode"):
            x1, y1, x2, y2 = map(int, box)
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                return None

            crop = (
                frame[y1:y2, x1:x2]
                if box_is_bumper
                else bumper_crop(frame, (x1, y1, x2, y2))
            )
            if crop is None or crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            score = legibility_score(crop)
            if enforce_floor and score < LEGIBILITY_FLOOR:
                # Record the rejected crop so the diagnostic view can show why this robot
                # went unread.
                if robot_id is not None:
                    try:
                        _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        rejected_b64 = base64.b64encode(buf).decode('utf-8')
                    except Exception as e:
                        print(f"  vlm_filtered: imencode failed for robot {robot_id}: {e}")
                        rejected_b64 = ""
                    self.diagnostic.record_vlm_filtered(
                        frame=self._current_frame, robot_id=robot_id,
                        crop_b64=rejected_b64, legibility=score,
                        reason="below_legibility_floor",
                    )
                return None

            # Upscale so the bumper crop is at least ~384px tall (existing behavior).
            if crop.shape[0] < 384:
                scale = max(2, int(np.ceil(384 / max(1, crop.shape[0]))))
                # LINEAR over CUBIC: 3-6x faster and the VLM is insensitive to
                # upscale interpolation (unlike the OCR preprocess paths).
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buf).decode('utf-8')
            return crop_b64, score

    def _masked_robot_crop_b64(self, frame: np.ndarray, box, mask=None):
        """Crop the whole SAM2-detected robot (background blacked out via `mask`),
        upscale, and JPEG/base64 encode it for the appearance gallery image.

        This is the image *behind* the appearance signature — the same region
        DINOv2 embeds — so the gallery shows the whole robot, not the bumper band.
        Returns a base64 string, or None when the box is empty.
        """
        with self._stage("loop.crop_encode"):
            x1, y1, x2, y2 = map(int, box)
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = _apply_mask(frame[y1:y2, x1:x2], mask)
            if crop.size == 0:
                return None
            if crop.shape[0] < 192:
                scale = max(2, int(np.ceil(192 / max(1, crop.shape[0]))))
                # LINEAR over CUBIC: runs per detection per frame; display-only image.
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            return base64.b64encode(buf).decode('utf-8')

    def _detect_whole_robots(self, frame: np.ndarray, with_balls: bool = False):
        """Run YOLO model on the frame, select only the whole-robot class, and
        filter out boxes that exceed ROBOT_FILL_MAX_BOX_FRAC. With
        ``with_balls``, also return the fuel-ball boxes from the same pass
        (motion-attribution exclusions)."""
        if not ROBOT_FILL_ENABLED:
            return ([], []) if with_balls else []
        results = self.model(frame, conf=ROBOT_FILL_CONF, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)

        max_frac = (ROBOT_FILL_MAX_BOX_FRAC_CLOSE
                    if self.config.det_frac_backstop == "uncalibrated"
                    else ROBOT_FILL_MAX_BOX_FRAC)
        schema = getattr(self, "det_schema", None)
        fuel_cls = (schema.fuel_class
                    if schema is not None and schema.fuel_class is not None
                    else 0)
        valid_boxes = []
        ball_boxes = []
        for i in range(len(dets)):
            cid = int(dets.class_id[i])
            box = tuple(float(v) for v in dets.xyxy[i])
            if cid == self.whole_robot_class_id:
                if not box_exceeds_frac(box, frame.shape, max_frac):
                    valid_boxes.append(box)
            elif with_balls and cid == fuel_cls:
                ball_boxes.append(box)
        return (valid_boxes, ball_boxes) if with_balls else valid_boxes

    def _sam2_robot_candidates(self, frame: np.ndarray, conf: float = 0.4,
                               motion_grays=None, blobs_allowed: bool = False,
                               zone_probe=None, corrob_floor: float = 0.0,
                               with_corrob_band: bool = False):
        """Shared SAM2 seeding/re-prompt detection cascade on a full-res view
        frame (already top-cropped): bumper detections — admitted by shape OR
        by whole-robot corroboration — expanded to robot boxes, merged with
        whole-robot gap fills, person candidates vetoed.

        ``motion_grays``: optional (prev_gray, cur_gray) pair in this frame's
        pixel space, enabling the motion knobs. Motion is attributed before
        it is trusted: person and ball detections are zeroed from the diff,
        so a ref walking past the bump (or balls rolling under a misread)
        cannot back a candidate. ``blobs_allowed`` gates the diff-blob
        candidate source (early-match window; the caller knows the clock).

        ``zone_probe(box)->bool``: optional predicate in this frame's pixel
        space; when it (or ``motion_probe``) admits a sub-conf box, that box
        rides the SUBORDINATE (strong=False) motion tier. Used for the
        starting-zone seed prior (boxes whose ground contact projects into a
        legal opening starting band).

        ``corrob_floor``: when > 0, the detector runs down to this
        confidence floor (below ``det_conf_floor`` if lower) and detections
        in [corrob_floor, conf) that neither the motion nor zone probe
        admitted are captured separately as a corroborate-only band —
        presence evidence that can never seed/match/evict. Returned only
        when ``with_corrob_band`` is True (as a second return value); 0
        (the default) produces an empty band and leaves behavior unchanged.

        Returns [(box_xyxy, color, strong)] in the input frame's pixel
        coords, in strict priority order: strong YOLO evidence first
        (bumpers at/above conf + whole-robot fills), then the motion tier
        (low-conf movers, diff blobs) deduped against it. ``strong=False``
        entries are fallback hints — callers must never let them displace a
        strong candidate from a seat. Field filtering and any coordinate
        scaling are the caller's concern (they differ between the
        prompt-seeding and re-prompt paths).
        """
        whole_boxes, ball_boxes = self._detect_whole_robots(frame, with_balls=True)
        person_res = self.person_model(
            frame, classes=[0], conf=DET_CONF_THRESHOLD, verbose=False
        )[0]
        person_dets = sv.Detections.from_ultralytics(person_res)
        person_boxes = [
            tuple(person_dets.xyxy[i]) for i in range(len(person_dets))
        ]

        motion_probe = None
        motion_boxes = None
        want_probe = self.config.motion_gated_conf
        want_blobs = self.config.motion_seed_blobs and blobs_allowed
        if motion_grays is not None and (want_probe or want_blobs):
            prev_gray, cur_gray = motion_grays
            if (prev_gray is not None and cur_gray is not None
                    and prev_gray.shape == cur_gray.shape):
                exclude = list(person_boxes) + list(ball_boxes)
                diff = motion_diff(prev_gray, cur_gray, exclude)
                if want_probe:
                    def motion_probe(box, _d=diff,
                                     _t=self.config.motion_energy_thresh):
                        return diff_box_energy(_d, box) >= _t
                if want_blobs:
                    motion_boxes = motion_blob_boxes(
                        prev_gray, cur_gray, exclude_boxes=exclude)

        strong_bumpers = []
        motion_bumpers = []
        corrob_bumpers = []
        if self.bumper_model is not None:
            # Motion-gated low-conf admission: with a motion probe available,
            # detect down to det_conf_floor; the sub-conf band survives only
            # where pixels actually moved — and only as the SUBORDINATE tier.
            gated = motion_probe is not None or zone_probe is not None
            run_conf = min(conf, self.config.det_conf_floor) if gated else conf
            if corrob_floor > 0:
                run_conf = min(run_conf, corrob_floor)
            res = self.bumper_model(frame, conf=run_conf, verbose=False)[0]
            robots, _balls, _hubs = self.det_schema.split(
                sv.Detections.from_ultralytics(res)
            )

            def _admit_probe(box, _m=motion_probe, _z=zone_probe):
                return bool((_m is not None and _m(box))
                            or (_z is not None and _z(box)))
            strong_dets, motion_dets = partition_low_conf_movers(
                robots, conf, _admit_probe if gated else None)

            def _expand(dets):
                out = []
                for box, color, _confidence in dets:
                    if not bumper_admissible(box, whole_boxes):
                        continue
                    out.append((expand_bumper_to_robot(
                        box, frame.shape, ROBOT_BOX_FROM_BUMPER_FACTOR), color))
                return out

            strong_bumpers = _expand(strong_dets)
            motion_bumpers = _expand(motion_dets)
            corrob_dets = []
            if corrob_floor > 0:
                # Exclude detections already promoted to the strong/motion
                # tiers, keyed by box VALUE (not object identity):
                # partition_low_conf_movers rebuilds fresh tuples, so
                # id(d) never matches across strong_dets/motion_dets and
                # robots. The later dedupe_against(corrob_bumpers, ...)
                # still handles expanded-box overlap, but this keeps the
                # exclusion set doing real, direct work.
                _key = lambda d: tuple(round(float(v), 3) for v in d[0])
                taken = {_key(d) for d in strong_dets} | {_key(d) for d in motion_dets}
                corrob_dets = [d for d in robots
                               if _key(d) not in taken and d[2] >= corrob_floor]
            corrob_bumpers = _expand(corrob_dets)
            # Cross-class double-reads of one blurred bumper (per-class NMS)
            # must not inflate corrob-band presence-evidence count either;
            # mirror the strong-tier dedupe (dedupe_against(..., tiered)
            # below won't catch these since neither box is in `tiered`).
            corrob_bumpers = dedupe_seed_robots(corrob_bumpers, ROBOT_FILL_IOS)
        # Cross-class double-reads of one blurred bumper (per-class NMS)
        # must not seed duplicate objects.
        strong_bumpers = dedupe_seed_robots(strong_bumpers, ROBOT_FILL_IOS)
        # YOLO tier: strong bumpers + whole-robot fills.
        merged = merge_robot_detections(
            strong_bumpers, whole_boxes, ROBOT_FILL_IOS,
            union_extent=self.config.seed_union_extent)
        tiered = [(box, color, True) for box, color in merged]
        # Motion tier (STRICTLY subordinate): low-conf movers, then diff
        # blobs — each deduped against everything the YOLO tier produced,
        # with the YOLO boxes INFLATED first: a hint adjacent to a tracked
        # robot is usually its own spillover (ejected balls, blur trail),
        # not a new robot (qm11: a second box seeded beside 3181 and two
        # slots fought over one robot). These are hints for when YOLO has
        # nothing for a slot, never competitors for a seat YOLO can fill.
        yolo_boxes = [box for box, _c in merged]
        motion_tier = dedupe_seed_robots(
            dedupe_against(motion_bumpers, yolo_boxes,
                           MOTION_TIER_DEDUP_IOS, inflate=MOTION_TIER_INFLATE),
            ROBOT_FILL_IOS)
        if motion_boxes:
            covered = yolo_boxes + [box for box, _c in motion_tier]
            motion_tier += dedupe_against(
                [(mb, "unknown") for mb in motion_boxes], covered,
                MOTION_TIER_DEDUP_IOS, inflate=MOTION_TIER_INFLATE)
        tiered += [(box, color, False) for box, color in motion_tier]
        # Corroborate-only band: sub-seed-confidence detections the motion/
        # zone probes did not already admit into `tiered`. Deduped against
        # the final tier list so a strong (or motion) candidate is never
        # shadowed by its own weaker corrob-band echo. These boxes skip the
        # pile/person vetoes below: they cast no seeds, and vetoing presence
        # evidence recreates the false-absence problem the band exists to
        # prevent.
        corrob_band = dedupe_against(
            corrob_bumpers, [box for box, _c, _s in tiered],
            MOTION_TIER_DEDUP_IOS, inflate=MOTION_TIER_INFLATE)
        # Ball-pile veto: the whole-robot class fires on dense ball piles;
        # a colorless candidate mostly covered by ball detections is not a
        # robot (bumper-colored candidates are exempt — robots carry balls).
        tiered = reject_ball_pile_candidates(
            tiered, ball_boxes, BALL_PILE_COVERAGE)
        result = reject_person_candidates(
            tiered, person_boxes, PERSON_ROBOT_REJECT_COVERAGE
        )
        if with_corrob_band:
            return result, corrob_band
        return result

    def _make_start_zone_probe(self, view, top_crop_px, elapsed_s):
        """zone_probe(box)->bool for _sam2_robot_candidates: True when the
        box's ground contact projects inside a starting band, only while the
        opening window is open. Boxes are in the view's top-cropped space."""
        if (not self.config.start_zone_seed
                or elapsed_s > self.config.start_zone_window_s
                or view.calibration is None or self.field_model is None):
            return None
        zones = start_zones(
            (self.field_model.length_m, self.field_model.width_m),
            self.config.start_zone_depth_m)

        def _probe(box):
            uv = ((box[0] + box[2]) / 2.0, box[3] + top_crop_px)
            xy = view.calibration.image_to_field(uv)
            return xy is not None and point_in_zones(xy, zones)
        return _probe

    def _ray_zone_list(self, view):
        """Inflated user zones for the ray-beyond-contact seed gate, or None
        when it doesn't apply: high cameras (reliable ground contacts), pose-
        free calibrations (no camera center), or ZONE_RAY_GATE_CAM_Z_M=0."""
        if ZONE_RAY_GATE_CAM_Z_M <= 0 or not self.user_seed_zones:
            return None
        cc = getattr(view.calibration, "camera_center", None)
        if cc is None or float(cc[2]) > ZONE_RAY_GATE_CAM_Z_M:
            return None
        return self.user_seed_zones

    @staticmethod
    def _below_robot_tracking_top(view, box, frame_height: int) -> bool:
        if not ROBOT_TRACK_TOP_CROP_ENABLED:
            return True
        frac = (
            getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0
        )
        if frac <= 0.0:
            return True
        return float(box[3]) >= float(frame_height) * frac

    def _detect_view(self, view, composite_frame, precomputed_dets=None, frame_num=0):
        """Run YOLO + person detection on one camera view; returns
        (paired, person_boxes, person_boxes_all), where ``person_boxes`` are
        field-filtered for occlusion geometry and ``person_boxes_all`` retain
        near-wall people for negative filtering and backward evidence.

        precomputed_dets: an sv.Detections already run on this view's frame (the main
        loop computes robot/ball detections on the main overhead frame for hub/ball/
        annotation, so we reuse them here instead of re-running YOLO on the same crop)."""
        frame = view.extract(composite_frame)
        cal = view.calibration
        if precomputed_dets is not None:
            dets = precomputed_dets
        else:
            results = self.model(frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(results)
        person_results = self.person_model(
            frame, classes=[0], conf=DET_CONF_THRESHOLD, verbose=False
        )[0]
        pdets = sv.Detections.from_ultralytics(person_results)
        person_boxes_all = [tuple(pdets.xyxy[i]) for i in range(len(pdets))]
        if self.bumper_model is not None:
            bumper_results = self.bumper_model(
                frame, conf=DET_CONF_THRESHOLD, verbose=False
            )[0]
            bumper_dets = sv.Detections.from_ultralytics(bumper_results)
            robots, _balls, _hubs = self.det_schema.split(bumper_dets)
        else:
            robots = [
                (
                    tuple(dets.xyxy[i]),
                    None,
                    (
                        float(dets.confidence[i])
                        if dets.confidence is not None
                        else 1.0
                    ),
                )
                for i in range(len(dets))
                if dets.class_id[i] == 1
            ]
        robots = reject_person_candidates(
            robots, person_boxes_all, PERSON_ROBOT_REJECT_COVERAGE
        )
        robot_boxes = []
        robot_colors = []
        for box, color, _confidence in robots:
            if self.bumper_model is not None and not plausible_bumper_box(box):
                continue
            if not self._below_robot_tracking_top(view, box, frame.shape[0]):
                continue
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            if any(
                mx <= cx <= mx + mw and my <= cy <= my + mh
                for (mx, my, mw, mh) in view.masks
            ):
                continue
            robot_boxes.append(np.asarray(box))
            robot_colors.append(color)
        rejected = []
        field_dets = to_field_detections(
            robot_boxes, cal, self.field_dims, colors=robot_colors,
            frame_shape=frame.shape,
            rejected_out=rejected,
            frac_backstop=self.config.det_frac_backstop,
        )
        # Corner-camera range falloff: attenuate (via det.score, which scales
        # the fusion weight) instead of hard-dropping, so center-field robots
        # keep corner coverage; only the far half (weight 0) is discarded.
        weighted = []
        for det in field_dets:
            w = corner_detection_weight(view.name, det.pos, self.field_dims)
            if w <= 0.0:
                continue
            det.score = w
            weighted.append(det)
        field_dets = weighted
        if rejected:
            self.hard_neg_logger.maybe_log(
                frame, [d.box for d in field_dets], rejected,
                view_name=view.name, frame_num=frame_num,
            )
        paired = []
        for d in field_dets:
            # Defer the JPEG encode: only the top-K candidates the VLM actually
            # dispatches get resolved (see read_fused_bumper_numbers). Bind loop
            # vars via default args to avoid late-binding capture bugs.
            def _enc(_f=frame, _b=d.box, _bm=self.bumper_model is not None):
                cb = self._bumper_crop_b64(_f, _b, enforce_floor=False, box_is_bumper=_bm)
                return cb[0] if cb is not None else None
            paired.append((d, _LazyCrop(_enc)))
        person_boxes = field_adjacent_persons(person_boxes_all, cal, self.field_dims)
        return paired, person_boxes, person_boxes_all

    def _remaining_rosters(self):
        """(remaining_all, remaining_blue, remaining_red) — match roster minus
        already-confirmed teams. Shared by the main and corner read paths."""
        confirmed = {r.team_number for r in self.robot_tracks.values()
                     if r.team_number_confirmed and r.team_number}
        return (
            [t for t in self.match_teams if t not in confirmed],
            [t for t in self.match_blue_teams if t not in confirmed],
            [t for t in self.match_red_teams if t not in confirmed],
        )

    def _apply_bumper_read(self, slot_id, crop_b64, remaining, remaining_blue,
                           remaining_red, source_weight):
        """Read one bumper crop with the VLM and add a (confidence x source_weight)
        vote to the slot's team tally. source_weight > 1 for corner cameras."""
        robot = self._get_slot_track_if_exists(slot_id)
        if not robot:
            return
        alliance = getattr(robot, "alliance", "unknown")
        expected = alliance if alliance in ("blue", "red") else "unknown"
        try:
            res = _vision_read_bumpers(
                [crop_b64], getattr(self, "match_teams", None) or remaining, remaining_blue, remaining_red,
                expected_alliances=[expected],
                provider=getattr(self, "team_reader", "local"),
            )[0]
        except TypeError:
            # Older tests monkeypatch the previous 4-arg helper shape.
            res = _vision_read_bumpers([crop_b64], remaining, remaining_blue, remaining_red)[0]
        team = res.get("team_number")
        if not team:
            return
        if surface_vote_multiplier(
                res.get("surface"),
                getattr(getattr(self, "config", None), "surface_gate", False)) == 0.0:
            print(f"  VLM vote dropped [printed-surface]: slot {slot_id} += {team}")
            return
        conf = res.get("confidence", 0.0)
        weight = (1.0 if conf >= 0.85 else 0.5) * float(source_weight)
        weight *= prob_vote_multiplier(
            getattr(robot, "last_tracker_prob", None),
            getattr(getattr(self, "config", None), "vote_prob_weight", False))
        if robot.team_ocr_votes is None:
            robot.team_ocr_votes = {}
        robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + weight
        print(f"  Corner read: slot {slot_id} += {team} (w={weight:.1f}, "
              f"total={robot.team_ocr_votes[team]:.1f})")


    def _handle_confirmed_vlm_contradiction(self, robot, new_team: Optional[str],
                                            confidence: float, raw_read: Optional[str],
                                            source: str) -> None:
        """Unlock a confirmed label when a strong current VLM read contradicts it.

        A confirmed team is a state lock, not ground truth. If the current bumper
        crop reads a different roster team with high confidence, keep the vote and
        clear the lock so the normal one-to-one assignment can correct the HUD and
        scoring attribution on the next identity pass.

        NOTE: this override is intentionally NOT damped by ``vote_prob_weight``
        (the tracker_prob vote weighting applied to the additive vote paths).
        It is the escape hatch for a *wrong* confirmed lock — including one
        caused by a hijack — and tracker_prob runs lowest exactly during a
        hijack, so damping the unlock by it would trap a track in the bad lock
        it most needs to leave. Weighting reduces routine vote *accumulation*
        on shaky tracks; this high-bar (>=0.9, exact-roster) correction is a
        separate mechanism.
        """
        if not robot or not robot.team_number_confirmed or not robot.team_number:
            return
        if not new_team or new_team == robot.team_number:
            return
        raw_digits = "".join(ch for ch in str(raw_read or "") if ch.isdigit())
        # Treat exact roster reads as strong contradictions. Fuzzy/partial reads
        # still accumulate votes and can overturn through the margin audit.
        if confidence < VLM_DIRECT_CONTRADICTION_CONF or raw_digits != str(new_team):
            return

        old_team = robot.team_number
        self.diagnostic.record_overturn(
            frame=self._current_frame, robot_id=robot.robot_id,
            old_team=old_team, challenger_team=new_team,
            current_votes=float((robot.team_ocr_votes or {}).get(old_team, 0.0)),
            challenger_votes=float((robot.team_ocr_votes or {}).get(new_team, 0.0)),
        )
        print(f"  VLM contradiction [{source}]: {self._slot_label(robot.robot_id)} "
              f"was locked as {old_team}, current crop reads {raw_digits}->{new_team}; unlocking")
        if robot.team_ocr_votes is None:
            robot.team_ocr_votes = {}
        robot.team_ocr_votes.pop(old_team, None)
        robot.team_ocr_votes[new_team] = max(
            float(robot.team_ocr_votes.get(new_team, 0.0)),
            float(OCR_TEAM_CONFIRM_VOTES),
        )
        if robot.consecutive_tracker is None:
            robot.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
        robot.consecutive_tracker.unlock()
        robot.consecutive_tracker.update(new_team)
        robot.team_number = None
        robot.team_number_confirmed = False
        ident = self.identity_registry.identities.get(robot.robot_id)
        if ident:
            ident.team_number = None
            ident.is_confirmed = False
            ident.confidence = 0.4
            ident.needs_immediate_read = True

    def _capture_slot_crop(self, slot_id, crop_b64, quality, embedding, robot_crop_b64=None):
        """Keep the top-3 (by quality) crops per slot for post-match review.
        Defensive copy: the live track's appearance_embedding is rebound by the EMA
        each frame, so we snapshot a stable copy here to persist at finalize.

        `crop_b64` is the bumper-band crop (legibility-ranked); `robot_crop_b64`
        is the masked whole-robot image behind the embedding. The review UI / gallery
        prefer the whole-robot image, so it matches the appearance signature."""
        if not crop_b64:
            return
        emb = None if embedding is None else np.array(embedding, dtype=np.float32, copy=True)
        bucket = self._slot_crops.setdefault(slot_id, [])
        bucket.append({"quality": float(quality), "crop_b64": crop_b64,
                       "robot_crop_b64": robot_crop_b64, "embedding": emb})
        bucket.sort(key=lambda c: c["quality"], reverse=True)
        del bucket[3:]

    def _deposit_to_gallery(self, robot, legibility: float = 0.0, image_bytes=None) -> None:
        """Stage a confirmed robot appearance for future matches.

        The query gallery is a frozen snapshot of previous matches. Current-match
        deposits must never be added to it, or a robot can confirm itself.
        """
        if self.gallery is None or robot is None or not getattr(self, "use_sam2", False):
            return
        color = robot.alliance if robot.alliance in ("red", "blue") else None
        if color is None or not robot.team_number or robot.appearance_embedding is None:
            return
        if not self.event_key or not self.match_key:
            return
        deposits = getattr(self, "_gallery_deposits", None)
        if deposits is None:
            deposits = AppearanceGallery(
                dedup_sim=GALLERY_DEDUP_SIM,
                max_exemplars=GALLERY_MAX_EXEMPLARS,
            )
            self._gallery_deposits = deposits
        if not deposits.add(
            robot.team_number,
            color,
            robot.appearance_embedding,
            legibility,
        ):
            return
        self._pending_gallery_rows.append({
            "event_key": self.event_key,
            "source_match_key": self.match_key,
            "team": robot.team_number,
            "alliance_color": color,
            "embedding": np.asarray(robot.appearance_embedding, dtype=np.float32).tobytes(),
            "legibility": legibility,
            "image": image_bytes,
            "verified": 0,
            "job_id": self.job_id,
            # Tag with the model that actually produced this vector, so a later
            # switch to a different embedder can't cosine-compare across spaces.
            "embed_model": getattr(getattr(self, "robot_embedder", None), "model_name", None),
        })

    def _record_frame_evidence(self, fused, frame_num, main_person_boxes):
        """Slice-1 whole-match evidence. No behavior change; consumed by the
        backward-confirmation classifier (slice 2+). All keys are fused track_ids.
        Position is the RAW fused measurement (det.pos): a coasting track is absent
        from `fused`, so still frames contribute no motion sample — intended."""
        for obs in fused:
            tid = obs.track_id
            if tid is None:
                continue
            det = obs.detection
            fxy = det.pos if det is not None else None
            box = det.box if det is not None else None
            color = det.color if det is not None else None
            overlapped = bool(box is not None and main_person_boxes and any(
                box_ios(tuple(box), tuple(pb)) >= PERSON_OVERLAP_IOS
                for pb in main_person_boxes))
            self._match_evidence.observe(tid, frame_num, fxy, overlapped_person=overlapped)
            if color in ("red", "blue"):
                self._match_evidence.confirm(tid, frame_num, bumper_color=True)
        self._binding_log.append({
            "frame_num": frame_num,
            "bindings": dict(self.field_tracker._id_to_slot),  # {track_id: slot_id}
        })

    def _track_id_for_slot(self, slot_id):
        # _id_to_slot is {track_id: slot_id}; invert. Returns None when the slot is
        # not currently bound to a fused track (confirmation then drops — acceptable
        # for slice 1; these sites normally run on frames where the slot IS bound).
        return next((tid for tid, s in self.field_tracker._id_to_slot.items()
                     if s == slot_id), None)

    def apply_gallery_matches(self, ocr_crops: list) -> None:
        """Query the gallery for unconfirmed tracks, cast votes, and annotate any
        matching ocr_crops cards so the UI distinguishes appearance recovery."""
        if self.gallery is None or not getattr(self, "use_sam2", False):
            return
        by_slot = {c["robot_id"]: c for c in ocr_crops}
        for slot_id, robot in self.robot_tracks.items():
            if robot.team_number_confirmed:
                continue
            verdict, team, sim = _apply_gallery_vote(robot, self.gallery)
            if verdict == "none" or not team:
                continue
            card = by_slot.get(slot_id)
            if card is not None and not card.get("matched"):
                card["matched"] = team
                card["via"] = "gallery"
                card["gallery_sim"] = round(sim, 3)
            if robot.team_number_confirmed:
                # Mirror the OCR lock path: a confirmed identity is exclusive, so
                # release this team from any other track that was tentatively holding it.
                self._slot_source[slot_id] = "gallery"
                self._clear_team_from_other_robots(slot_id, robot.team_number)
                for event in self.scoring_events:
                    if event.robot_id == slot_id:
                        event.team_number = robot.team_number
                try:
                    tid = self._track_id_for_slot(slot_id)
                    if tid is not None:
                        self._match_evidence.confirm(tid, self._evidence_frame, appearance=True)
                except Exception:
                    pass
                print(f"  Gallery-confirmed: Robot {slot_id} = Team {robot.team_number} "
                      f"(appearance {sim:.2f})")

    async def read_fused_bumper_numbers(
        self, fused, requested_priorities: dict[int, float], det_to_slot=None
    ) -> None:
        """Read team numbers from the per-camera bumper crops attached to fusion.

        Accepts requested_priorities: dict mapping identity_id -> priority.

        The VLM generate runs on a worker thread (CUDA releases the GIL) so the
        event loop keeps servicing websocket keepalives and pause/stop during a
        100-500ms read; all identity-state mutation stays on the loop.

        The loop now schedules this as a background task (asyncio.create_task),
        so the body starts running at the NEXT loop yield point — possibly after
        the field tracker has advanced to a newer frame. det_to_slot lets the
        caller pin the slot mapping to the scheduling frame (snapshotting
        field_tracker.last_det_to_slot). When None (direct/synchronous callers,
        incl. tests) it falls back to the live tracker attr.

        The whole body is wrapped in the vlm.pass_async stage so the true
        read-pass wall cost stays visible in the profile without implying it
        blocks the loop.
        """
        with self._stage("vlm.pass_async"):
            await self._read_fused_bumper_numbers_impl(
                fused, requested_priorities, det_to_slot)

    def _schedule_vlm_pass(self, fused, due_priorities) -> None:
        """Single-flight schedule of the background bumper-read pass.

        Only one pass runs at a time. If a pass is already in flight this is a
        no-op: the due identities stay due and the next frame re-checks — no
        queueing, no pileup on the vLLM server. A finished previous pass has
        its exception surfaced (logged) here so one failed read can't silently
        swallow errors, and never kills the loop.
        """
        _t = self._vlm_task
        if _t is not None and _t.done():
            exc = _t.exception()  # retrieve + surface the previous pass
            if exc:
                print(f"  VLM pass failed: {exc}")
        if _t is not None and not _t.done():
            return  # a pass is in flight — stay due, re-check next frame
        self._claude_vision_calls += 1
        # Pin the detection->slot mapping to THIS frame: the task body starts at
        # the next loop yield, by which point last_det_to_slot may describe a
        # newer frame whose indices don't line up with this `fused` snapshot.
        _det_to_slot = dict(self.field_tracker.last_det_to_slot)
        self._vlm_task = asyncio.create_task(
            self.read_fused_bumper_numbers(
                fused, due_priorities, det_to_slot=_det_to_slot))

    async def _drain_vlm_task(self) -> None:
        """Await any in-flight background VLM read so its votes land before
        final team assignment. A failed pass is logged, never raised — a dead
        read must not abort finalization. Awaits (does not cancel) so the last
        pass's identity mutations complete on the loop."""
        t = self._vlm_task
        if t is None:
            return
        try:
            if not t.done():
                await t
            else:
                exc = t.exception()
                if exc:
                    print(f"  VLM pass failed (drain): {exc}")
        except Exception as e:
            print(f"  VLM pass failed (drain): {e}")
        finally:
            self._vlm_task = None

    def _maybe_schedule_adjudication(self, now: float) -> None:
        """Observe-only equipment adjudication: ask the VLM whether long-static
        tracks are actually robots. Single-flight like the bumper pass; verdicts
        go to diagnostics ONLY — no eviction, no confidence change. Equipment
        hijacks are detector-corroborated, so this is the independent modality
        the culls lack; grade it from replays before wiring any action."""
        t = self._adjudicate_task
        if t is not None and t.done():
            exc = t.exception()
            if exc:
                print(f"  Adjudication pass failed: {exc}")
        if t is not None and not t.done():
            return
        cfg = self.config
        batch = []  # (identity_id, slot_id, crop_b64)
        for ident in self.identity_registry.get_active_identities():
            if (now - ident.last_adjudicate_time) < cfg.adjudicate_cooldown_sec:
                continue
            hist = self._ident_pos_history.get(ident.identity_id)
            if not hist or not track_is_static(
                    hist, now, cfg.adjudicate_static_sec, cfg.adjudicate_static_move_m):
                continue
            robot = self._get_slot_track_if_exists(ident.active_slot)
            crop = getattr(robot, "appearance_crop_b64", None) if robot else None
            if not crop:
                continue
            batch.append((ident.identity_id, ident.active_slot, crop))
        if not batch:
            return
        for identity_id, _slot, _crop in batch:
            ident = self.identity_registry.identities.get(identity_id)
            if ident:
                ident.last_adjudicate_time = now
        self._adjudicate_task = asyncio.create_task(self._run_adjudication(batch))

    async def _run_adjudication(self, batch) -> None:
        results = await asyncio.to_thread(
            _vllm_classify_crops, [crop for (_i, _s, crop) in batch])
        for (identity_id, slot_id, _crop), res in zip(batch, results):
            ident = self.identity_registry.identities.get(identity_id)
            if ident is None or res.get("error"):
                continue
            verdict = res.get("object", "unclear")
            if verdict == "robot":
                ident.equipment_streak = 0
            elif verdict in ("equipment", "person"):
                ident.equipment_streak += 1
            self.diagnostic.record_adjudication(
                frame=self._current_frame, robot_id=slot_id,
                verdict=verdict, streak=ident.equipment_streak)
            print(f"  VLM adjudicate: {self._slot_label(slot_id)} -> {verdict} "
                  f"(streak={ident.equipment_streak})")

    async def _read_fused_bumper_numbers_impl(
        self, fused, requested_priorities: dict[int, float], det_to_slot=None
    ) -> None:
        if not self.match_teams:
            self.vlm_status.update({
                "provider": getattr(self, "team_reader", "local"),
                "last_frame": self._current_frame,
                "last_candidates": 0,
                "last_sent": 0,
                "last_results": [],
                "last_error": "no match roster",
            })
            self._ocr_crops_dirty = True
            return

        current_match_time = self._current_frame / (self.video_fps or 10.0)

        # Map slot_id -> obs. det_to_slot is the frame-indexed detection->slot
        # mapping; when the pass runs as a background task it MUST be the
        # snapshot captured at scheduling time (see caller) so indices into
        # this frame's `fused` don't mismatch a newer frame's live mapping.
        if det_to_slot is None:
            det_to_slot = self.field_tracker.last_det_to_slot
        slot_to_obs = {}
        for i, obs in enumerate(fused):
            slot_id = det_to_slot.get(i)
            if slot_id is not None:
                slot_to_obs[slot_id] = obs

        candidates = []  # (priority, source_weight, identity_id, slot_id, cam, crop_b64, crop_pos)
        identities_with_crops = set()

        for identity_id, priority in requested_priorities.items():
            ident = self.identity_registry.identities.get(identity_id)
            if ident is None or ident.active_slot is None:
                continue
            slot_id = ident.active_slot
            obs = slot_to_obs.get(slot_id)
            if obs is None:
                continue

            # Occlusion check. crop_b64 here may be a lazy _LazyCrop thunk (not yet
            # encoded); a thunk is truthy, so this preserves today's behavior where
            # the eager encode virtually never returned None (enforce_floor=False).
            # Truthiness means "a corner crop is available", not "resolves non-None".
            # NOTE: with the background-task schedule self._occ may be one frame
            # stale relative to `fused`. That is acceptable: these flags only gate
            # crop preference (which camera to trust), not read correctness.
            if self._occ.get(slot_id, {}).get("bumper_occluded") and not any(
                cam != "main" and crop_b64 for cam, crop_b64, _w, _pos in obs.evidence
            ):
                continue

            has_any_crop = False
            for cam, crop_b64, source_weight, cam_pos in obs.evidence:
                if crop_b64 is None:
                    continue
                if self.config.vote_ray_gate and cam_pos is not None:
                    center = self._camera_centers_xy.get(cam)
                    if center is not None:
                        miss = vote_ray_cross_m(center, cam_pos, obs.detection.pos)
                        if miss is not None and miss > self.config.vote_ray_cross_m:
                            print(f"  VLM crop dropped [ray-miss]: slot {slot_id} "
                                  f"cam={cam} cross={miss:.2f}m")
                            continue
                has_any_crop = True
                candidates.append((
                    priority,
                    float(source_weight),
                    identity_id,
                    slot_id,
                    cam,
                    crop_b64,
                    (float(obs.detection.pos[0]), float(obs.detection.pos[1]))
                ))

            if has_any_crop:
                identities_with_crops.add(identity_id)
            else:
                # No crops found: backoff
                ident.last_vlm_time = current_match_time
                ident.vlm_fail_count += 1
                ident.backoff_until = current_match_time + self.confidence_config.backoff_base_sec * (1.5 ** ident.vlm_fail_count)
                print(f"  VLM backoff: Identity {identity_id} (slot {slot_id}) has no crops. Backoff until {ident.backoff_until:.2f}")

        # For any requested identity that didn't even have an active slot or obs
        for identity_id in requested_priorities:
            if identity_id not in identities_with_crops:
                ident = self.identity_registry.identities.get(identity_id)
                if ident and ident.identity_id not in identities_with_crops:
                    ident.last_vlm_time = current_match_time
                    ident.vlm_fail_count += 1
                    ident.backoff_until = current_match_time + self.confidence_config.backoff_base_sec * (1.5 ** ident.vlm_fail_count)
                    print(f"  VLM backoff: Identity {identity_id} not visible/active. Backoff until {ident.backoff_until:.2f}")

        if not candidates:
            print("  VLM skipped: no legible fused bumper crops this pass")
            self.vlm_status.update({
                "provider": getattr(self, "team_reader", "local"),
                "last_frame": self._current_frame,
                "last_candidates": 0,
                "last_sent": 0,
                "last_results": [],
                "last_error": "no fused bumper crops",
            })
            self._ocr_crops_dirty = True
            return

        # Sort by (priority, source_weight) descending
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        chosen = []
        chosen_identities = set()
        for candidate in candidates:
            identity_id = candidate[2]
            if identity_id in chosen_identities:
                continue
            chosen.append(candidate)
            chosen_identities.add(identity_id)
            if len(chosen) >= VLM_TOP_K:
                break

        # Resolve lazy crops for only the chosen candidates — this is the point
        # of _LazyCrop: at most VLM_TOP_K encodes per pass instead of one per
        # detection per frame. A None resolution (degenerate box) is dropped.
        resolved = []
        for cand in chosen:
            crop = _resolve_crop(cand[5])
            if crop:
                resolved.append(cand[:5] + (crop,) + cand[6:])
        chosen = resolved
        if not chosen:
            print("  VLM skipped: chosen crops resolved empty")
            self.vlm_status.update({
                "provider": getattr(self, "team_reader", "local"),
                "last_frame": self._current_frame,
                "last_candidates": len(candidates),
                "last_sent": 0,
                "last_results": [],
                "last_error": "chosen crops resolved empty",
            })
            self._ocr_crops_dirty = True
            return

        crops = [c[5] for c in chosen]
        expected = []
        for _priority, _w, identity_id, _slot_id, _cam, _crop_b64, *_extra in chosen:
            ident = self.identity_registry.identities.get(identity_id)
            alliance = getattr(ident, "alliance", "unknown")
            expected.append(alliance if alliance in ("blue", "red") else "unknown")

        provider = getattr(self, "team_reader", "local")
        self.vlm_status.update({
            "provider": provider,
            "calls": int(self.vlm_status.get("calls", 0)) + 1,
            "last_frame": self._current_frame,
            "last_candidates": len(candidates),
            "last_sent": len(chosen),
            "last_results": [],
            "last_error": None,
        })
        try:
            results = await asyncio.to_thread(
                _vision_read_bumpers,
                crops, self.match_teams, self.match_blue_teams, self.match_red_teams,
                expected_alliances=expected,
                provider=provider,
            )
        except Exception as e:
            print(f"  VLM failed [{provider}]: {e}")
            self.vlm_status["last_error"] = str(e)
            return

        ocr_crops = []
        status_results = []
        # Within-batch team dedupe: one physical robot photographed by two
        # crops (or its number cross-read from an adjacent crop) must not
        # vote the same team into two slots at once (qm11: one batch put
        # "2992" on B1 AND B3). The strongest read keeps the vote.
        _vote_ok = dedupe_batch_team_reads([
            (r.get("team_number"), float(r.get("confidence", 0.0)), c[1])
            for c, r in zip(chosen, results)
        ])
        for _read_idx, ((_priority, source_weight, identity_id, slot_id, cam, crop_b64, *_extra), res) in enumerate(zip(chosen, results)):
            robot = self._get_or_create_slot_track(slot_id)
            ident = self.identity_registry.identities.get(identity_id)
            if not robot or not ident:
                continue

            # Update VLM attempt time + position (novelty-gate reference)
            ident.last_vlm_time = current_match_time
            ident.last_vlm_pos = getattr(robot, "field_position", None)

            team = res.get("team_number")
            conf = float(res.get("confidence", 0.0))
            self._capture_slot_crop(robot.robot_id, crop_b64, conf, robot.appearance_embedding,
                                    robot_crop_b64=robot.appearance_crop_b64)
            vote_weight = (1.0 if conf >= 0.85 else 0.5) * source_weight
            vote_weight *= prob_vote_multiplier(
                getattr(robot, "last_tracker_prob", None),
                getattr(getattr(self, "config", None), "vote_prob_weight", False))
            status_results.append({
                "slot": self._slot_label(slot_id),
                "source": cam,
                "raw_read": res.get("raw_read"),
                "team_number": team,
                "confidence": conf,
                "error": res.get("error"),
                "surface": res.get("surface", "unknown"),
            })
            ocr_crops.append({
                "robot_id": robot.robot_id,
                "slot": self._slot_label(slot_id),
                "alliance": res.get("alliance", "unknown"),
                "source": cam,
                "crop": crop_b64,
                "raw_read": res.get("raw_read"),
                "matched": team,
                "confidence": conf,
                "error": res.get("error"),
                "legibility": None,
                "via": "ocr",
                "surface": res.get("surface", "unknown"),
            })
            self.diagnostic.record_vlm_read(
                frame=self._current_frame, robot_id=robot.robot_id,
                crop_b64=crop_b64,
                legibility=0.0,
                raw_read=res.get("raw_read"),
                matched_team=team,
                alliance_read=res.get("alliance", "unknown"),
                confidence=conf,
                vote_weight=vote_weight,
                surface=res.get("surface", "unknown"),
            )

            # Update success/backoff based on whether read succeeded
            if res.get("error") or not team:
                ident.vlm_fail_count += 1
                ident.backoff_until = current_match_time + self.confidence_config.backoff_base_sec * (1.5 ** ident.vlm_fail_count)
            else:
                previous_team = ident.team_number
                ident.last_success_time = current_match_time
                ident.vlm_fail_count = 0
                ident.backoff_until = -999.0
                ident.has_vlm_read = True
                ident.vlm_agrees = previous_team is not None and team == previous_team
                ident.vlm_contradicts = (
                    ident.is_confirmed
                    and previous_team is not None
                    and team != previous_team
                )
                if not ident.is_ambiguous:
                    ident.needs_immediate_read = False

            if (robot.team_number_confirmed and team == robot.team_number
                    and conf >= 0.85):
                self._deposit_to_gallery(robot, legibility=conf,
                                         image_bytes=_thumbnail_bytes(robot.appearance_crop_b64 or crop_b64))
            if not team:
                continue
            if not _vote_ok[_read_idx]:
                print(f"  VLM vote dropped [batch-dup]: "
                      f"{self._slot_label(slot_id)} += {team} suppressed "
                      f"(stronger read of the same team in this batch)")
                continue
            if surface_vote_multiplier(
                    res.get("surface"),
                    getattr(getattr(self, "config", None), "surface_gate", False)) == 0.0:
                print(f"  VLM vote dropped [printed-surface]: "
                      f"{self._slot_label(slot_id)} += {team} "
                      f"(digits printed on structure, not a bumper)")
                continue
            # Zone-blind voting: a read taken while the track sits inside a
            # static-structure footprint is printed equipment text, not a
            # bumper (qm11: R2 read the OVERFLOW crate's "1477" at prob 1.0
            # from [11.3,2.1] — inside the bin zone — and it tied the one
            # real "4256" read so nothing ever locked).
            # robot.field_position is the track's LIVE position: with the
            # background-task schedule it may be ~0.1-0.3 match-seconds newer
            # than the crop this vote came from. That sub-metre drift is well
            # inside the inflate_zones(0.3) margin below, so the zone-blind
            # gate stays correct; no reconciliation needed.
            _pos = getattr(robot, "field_position", None)
            try:
                _pos = (float(_pos[0]), float(_pos[1]))
            except (TypeError, ValueError, IndexError):
                _pos = None
            # Small margin over the raw footprints: the SAM box reaches the
            # equipment (and the crop shows its printed text) slightly before
            # the track's ground contact enters the tight rect.
            _vote_zones = inflate_zones(
                getattr(self, "user_seed_zones_raw", None) or [], 0.3)
            if (getattr(getattr(self, "config", None), "zone_blind_votes", False)
                    and _pos is not None
                    and point_in_zones(_pos, _vote_zones)):
                print(f"  VLM vote dropped [static-zone]: "
                      f"{self._slot_label(slot_id)} += {team} at "
                      f"({_pos[0]:.1f},{_pos[1]:.1f})")
                continue
            # Stale-position gate: votes are applied by a background task, so
            # the slot's LIVE position (_pos) can have moved since the crop
            # this vote came from was captured. A move farther than a robot
            # can plausibly drive means the slot was re-bound between crop and
            # vote — the vote belongs to whatever now occupies the old spot,
            # not this identity.
            crop_pos = _extra[0] if _extra else None
            if (self.config.vote_ray_gate and crop_pos is not None
                    and _pos is not None):
                moved = ((_pos[0] - crop_pos[0]) ** 2
                         + (_pos[1] - crop_pos[1]) ** 2) ** 0.5
                if moved > self.config.vote_stale_max_m:
                    print(f"  VLM vote dropped [stale-pos]: "
                          f"{self._slot_label(slot_id)} += {team} moved "
                          f"{moved:.1f}m since crop")
                    continue
            if robot.team_ocr_votes is None:
                robot.team_ocr_votes = {}
            robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + vote_weight
            vote_total = robot.team_ocr_votes[team]
            self._handle_confirmed_vlm_contradiction(
                robot, team, conf, res.get("raw_read"), cam
            )
            if robot.consecutive_tracker is None:
                robot.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
            locked_team, confirmed = robot.consecutive_tracker.update(team)
            if confirmed and locked_team and locked_team != robot.team_number:
                old_team = robot.team_number
                if old_team and robot.team_ocr_votes:
                    # Drop the overridden team's stale votes so a later un-confirm
                    # can't resurrect the wrong identity from leftover tallies.
                    robot.team_ocr_votes.pop(old_team, None)
                robot.team_number = locked_team
                robot.team_number_confirmed = True
                ident.team_number = locked_team
                ident.is_confirmed = True
                self._slot_source[identity_id] = "ocr"
                self._clear_team_from_other_robots(robot.robot_id, locked_team)
                for event in self.scoring_events:
                    if event.robot_id == identity_id:
                        event.team_number = locked_team
                verb = "overridden" if old_team else "confirmed"
                print(f"  Consecutive-lock {verb}: Robot {slot_id} = Team {locked_team} (was {old_team})")
                try:
                    tid = self._track_id_for_slot(slot_id)
                    if tid is not None:
                        self._match_evidence.confirm(tid, self._evidence_frame, team_read=True)
                except Exception:
                    pass
                self._deposit_to_gallery(robot, legibility=conf,
                                         image_bytes=_thumbnail_bytes(robot.appearance_crop_b64 or crop_b64))
            print(f"  VLM read [{cam}]: {self._slot_label(slot_id)} += {team} "
                  f"(w={vote_weight:.1f}, total={vote_total:.1f}"
                  + (f", at {_pos[0]:.2f},{_pos[1]:.2f}" if _pos is not None else ", at ?")
                  + ")")

        # Co-occlusion resolution. Reads identify the robot occupying each slot;
        # translate those team labels back to durable identities before changing
        # any binding.
        pass_reads = {}
        for (_priority, source_weight, identity_id, slot_id, cam, crop_b64, *_extra), res in zip(chosen, results):
            team_val = res.get("team_number")
            conf_val = float(res.get("confidence", 0.0))
            if team_val and conf_val >= 0.85:
                pass_reads[identity_id] = team_val

        for event in list(self.co_occlusion_tracker.get_active_events()):
            if event.status != "pending_vlm":
                continue
            event_reads = {id: pass_reads[id] for id in event.identity_ids if id in pass_reads}
            expected_teams = {
                ident.team_number: ident.identity_id
                for ident in (
                    self.identity_registry.identities.get(identity_id)
                    for identity_id in event.identity_ids
                )
                if ident is not None and ident.team_number
            }
            complete = len(event_reads) == len(event.identity_ids)
            unique = len(set(event_reads.values())) == len(event_reads)
            roster_valid = set(expected_teams).issubset(set(event_reads.values()))
            if not (complete and unique and roster_valid):
                self.identity_diagnostics.record(
                    DiagnosticEvents.OCCLUSION_GROUP_UNRESOLVED,
                    reason=(DiagnosticReasons.UNREADABLE_RESULT
                            if not complete else DiagnosticReasons.ALLIANCE_CONFLICT),
                    event_id=event.event_id,
                    reads=event_reads,
                )
                continue

            resolved_slots = {}
            resolved_teams = {}
            for observed_identity_id, team_num in event_reads.items():
                observed_ident = self.identity_registry.identities.get(observed_identity_id)
                if observed_ident is None or observed_ident.active_slot is None:
                    resolved_slots = {}
                    break
                target_identity_id = expected_teams.get(team_num, observed_identity_id)
                if target_identity_id in resolved_slots:
                    resolved_slots = {}
                    break
                resolved_slots[target_identity_id] = observed_ident.active_slot
                resolved_teams[target_identity_id] = team_num
            if len(resolved_slots) != len(event.identity_ids):
                continue

            for identity_id, slot_id in resolved_slots.items():
                ident_obj = self.identity_registry.identities[identity_id]
                ident_obj.active_slot = slot_id
                ident_obj.is_ambiguous = False
                ident_obj.needs_immediate_read = False
                ident_obj.confidence = max(ident_obj.confidence, 0.85)
                if ident_obj.team_number is None:
                    ident_obj.team_number = resolved_teams[identity_id]
                    ident_obj.is_confirmed = True
                robot_track = self._get_or_create_identity_track(
                    identity_id, ident_obj.alliance
                )
                robot_track.team_number = ident_obj.team_number
                robot_track.team_number_confirmed = ident_obj.is_confirmed
                self._slot_source[slot_id] = "ocr"

            self._last_bindings.update(resolved_slots)
            self.co_occlusion_tracker.resolve_event(event.event_id, resolved_slots)
            self.identity_diagnostics.increment("occlusion_groups_resolved")
            self.identity_diagnostics.record(
                DiagnosticEvents.OCCLUSION_GROUP_RESOLVED,
                event_id=event.event_id,
                mapping=resolved_slots,
                reads=event_reads,
            )
            print(f"  Co-occlusion event {event.event_id} resolved: {resolved_slots}")

            remaining_deferred = []
            for attrib in self.deferred_attributions:
                if attrib["alliance"] != event.alliance:
                    remaining_deferred.append(attrib)
                    continue
                matched_ident_id = next(
                    (
                        identity_id
                        for identity_id, slot_id in resolved_slots.items()
                        if slot_id == attrib["slot_id"]
                    ),
                    None,
                )
                if matched_ident_id is None:
                    remaining_deferred.append(attrib)
                    continue
                ident_obj = self.identity_registry.identities[matched_ident_id]
                robot_track = self._get_or_create_identity_track(
                    matched_ident_id, ident_obj.alliance
                )
                robot_track.scores += 1
                self.scoring_events.append(ScoringEvent(
                    frame_num=attrib["frame_num"],
                    timestamp=attrib["timestamp"],
                    alliance=attrib["alliance"],
                    robot_id=matched_ident_id,
                    team_number=ident_obj.team_number,
                    confidence="high",
                    distance_to_hub=attrib["distance_to_hub"],
                ))
                self.identity_diagnostics.increment("scoring_backfilled")
                self.identity_diagnostics.record(
                    DiagnosticEvents.SCORING_BACKFILLED,
                    identity_id=matched_ident_id,
                    event_id=event.event_id,
                )
            self.deferred_attributions = remaining_deferred

        self.apply_gallery_matches(ocr_crops)
        self.last_ocr_crops = ocr_crops
        self._ocr_crops_dirty = True
        self.vlm_status["last_results"] = status_results
        errors = [r.get("error") for r in status_results if r.get("error")]
        if errors:
            self.vlm_status["last_error"] = errors[0]

    def vision_identify_robots(
        self, frame: np.ndarray, tracked_robots, boxes_are_bumpers: bool = False
    ) -> None:
        """Time-tracking wrapper around the bumper-VLM read (accumulates into
        timings['bumper_vlm'], a sub-component of the analysis loop)."""
        _t = time.perf_counter()
        try:
            return self._vision_identify_robots_impl(
                frame,
                tracked_robots,
                boxes_are_bumpers=boxes_are_bumpers,
            )
        finally:
            t = getattr(self, "timings", None)
            if t is not None:
                t["bumper_vlm"] = t.get("bumper_vlm", 0.0) + (time.perf_counter() - _t)

    def _vision_identify_robots_impl(
        self,
        frame: np.ndarray,
        tracked_robots,
        boxes_are_bumpers: bool = False,
    ) -> None:
        """Use Qwen-VL to read bumper numbers on a legibility-ranked subset of
        the field. Confirmed robots are included at a reduced cadence so a wrong
        identity can be overturned by the audit (assign_teams_from_votes +
        audit_overturn_confirmations). tracked_robots: list of (canonical_id, box)."""
        if not self.match_teams:
            return

        # Roster minus already-confirmed teams (the OCR matcher still sees the
        # full per-alliance lists; confirmed teams are pre-removed to bias reads).
        confirmed_teams = {r.team_number for r in self.robot_tracks.values()
                           if r.team_number_confirmed and r.team_number}
        remaining_teams = [t for t in self.match_teams if t not in confirmed_teams]
        if not remaining_teams:
            return  # everything confirmed
        remaining_blue = [t for t in self.match_blue_teams if t not in confirmed_teams]
        remaining_red = [t for t in self.match_red_teams if t not in confirmed_teams]

        # Build candidate crops. Confirmed robots only on re-read passes.
        is_reread_pass = True
        candidates = []  # list of (score, canonical_id, box, crop_b64)
        for canonical_id, box in tracked_robots:
            robot = self._get_or_create_slot_track(canonical_id)
            if not robot:
                continue
            if robot.team_number_confirmed and not is_reread_pass:
                continue  # skip confirmed unless this is a re-read pass
            if self._occ.get(canonical_id, {}).get("bumper_occluded"):
                continue  # bumper hidden by hub/person/robot — don't read it
            if (getattr(self.config, "det_gated_vlm_reads", False)
                    and getattr(self, "detection_archive", None) is not None):
                _gap = int(self.config.det_gate_max_age_s
                           * (self.video_fps or 10.0))
                try:
                    _ok = self.detection_archive.corroborates(
                        "main", self._current_frame, box, max_gap_frames=_gap)
                except Exception:
                    # Mirror detection_archive.record()'s "telemetry must
                    # never break detection" contract: a gate failure must
                    # fail open, exactly like no-coverage (None), not skip
                    # or crash the identify pass.
                    _ok = None
                    self.identity_diagnostics.increment(
                        "vlm_read.det_gate_error")
                if _ok is False:  # None = no coverage -> fail open
                    self.identity_diagnostics.increment(
                        "vlm_read.det_gate_skip")
                    continue

            cb = self._bumper_crop_b64(
                frame,
                box,
                robot_id=canonical_id,
                box_is_bumper=boxes_are_bumpers,
            )
            if cb is None:
                continue  # empty crop or below legibility floor (diagnostic recorded)
            crop_b64, score = cb
            candidates.append((score, canonical_id, box, crop_b64))

        if not candidates:
            return

        # Send the top-K legible crops to Qwen.
        candidates.sort(key=lambda c: c[0], reverse=True)
        chosen = candidates[:VLM_TOP_K]

        results = []
        ocr_crops = []
        for score, canonical_id, box, crop_b64 in chosen:
            expected = "unknown"
            robot = self._get_or_create_slot_track(canonical_id)
            if robot and robot.alliance in ("blue", "red"):
                expected = robot.alliance
            res = _vision_read_bumpers(
                [crop_b64], self.match_teams, self.match_blue_teams, self.match_red_teams,
                expected_alliances=[expected],
                provider=getattr(self, "team_reader", "local"),
            )[0]
            results.append((canonical_id, box, res))
            ocr_crops.append({
                "robot_id": robot.robot_id,
                "slot": self._slot_label(canonical_id),
                "alliance": res.get("alliance", "unknown"),
                "source": "main",
                "crop": crop_b64,
                "raw_read": res.get("raw_read"),
                "matched": res.get("team_number"),
                "confidence": float(res.get("confidence", 0.0)),
                "error": res.get("error"),
                "legibility": round(score, 4),
                "surface": res.get("surface", "unknown"),
            })
            self.diagnostic.record_vlm_read(
                frame=self._current_frame, robot_id=robot.robot_id,
                crop_b64=crop_b64,
                legibility=float(score),
                raw_read=res.get("raw_read"),
                matched_team=res.get("team_number"),
                alliance_read=res.get("alliance", "unknown"),
                confidence=float(res.get("confidence", 0.0)),
                vote_weight=1.0 if float(res.get("confidence", 0.0)) >= 0.85 else 0.5,
                surface=res.get("surface", "unknown"),
            )
        self.last_ocr_crops = ocr_crops
        self._ocr_crops_dirty = True

        for canonical_id, box, result in results:
            team = result.get("team_number")
            if not team:
                continue
            robot = self._get_or_create_slot_track(canonical_id)
            if not robot:
                continue
            if surface_vote_multiplier(
                    result.get("surface"),
                    getattr(getattr(self, "config", None), "surface_gate", False)) == 0.0:
                print(f"  VLM vote dropped [printed-surface]: "
                      f"{self._slot_label(canonical_id)} += {team}")
                continue
            conf = result.get("confidence", 0.0)
            weight = 1.0 if conf >= 0.85 else 0.5
            weight *= prob_vote_multiplier(
                getattr(robot, "last_tracker_prob", None),
                getattr(getattr(self, "config", None), "vote_prob_weight", False))
            if robot.team_ocr_votes is None:
                robot.team_ocr_votes = {}
            robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + weight
            self._handle_confirmed_vlm_contradiction(
                robot, team, float(conf), result.get("raw_read"), "main"
            )
            if robot.consecutive_tracker is None:
                robot.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
            locked_team, confirmed = robot.consecutive_tracker.update(team)
            if confirmed and locked_team and locked_team != robot.team_number:
                old_team = robot.team_number
                if old_team and robot.team_ocr_votes:
                    robot.team_ocr_votes.pop(old_team, None)
                robot.team_number = locked_team
                robot.team_number_confirmed = True
                self._clear_team_from_other_robots(robot.robot_id, locked_team)
                for event in self.scoring_events:
                    if event.robot_id == robot.robot_id:
                        event.team_number = locked_team
                verb = "overridden" if old_team else "confirmed"
                print(f"  Consecutive-lock {verb} (vision): Robot {canonical_id} = Team {locked_team} (was {old_team})")
            tag = "RE-READ" if robot.team_number_confirmed else "read"
            print(f"  Vision {tag}: Robot {canonical_id} += {team} "
                  f"(w={weight}, total={robot.team_ocr_votes[team]:.1f})")

    def _enforce_alliance_caps_by_color(self) -> None:
        """Ensure no alliance holds more than 3 robots. If one does, move the robots
        with the weakest color evidence to the other alliance (which has room, since
        total robots are capped at 6). Confirmed robots are kept (strongest evidence)."""
        for alliance, other in (("red", "blue"), ("blue", "red")):
            members = [r for r in self.robot_tracks.values() if r.alliance == alliance]
            if len(members) <= MAX_ROBOTS_PER_ALLIANCE:
                continue

            def color_margin(r):
                m = (r.red_votes - r.blue_votes) if alliance == "red" else (r.blue_votes - r.red_votes)
                return (1_000_000 if r.team_number_confirmed else 0) + m  # never move confirmed

            members.sort(key=color_margin)  # weakest evidence first
            for r in members[:len(members) - MAX_ROBOTS_PER_ALLIANCE]:
                print(f"  Alliance rebalance: Robot {r.robot_id} {alliance} -> {other} "
                      f"(weakest color evidence; B{r.blue_votes}/R{r.red_votes})")
                r.alliance = other
                r.team_number = None          # its old-alliance team no longer applies
                r.team_number_confirmed = False

    def _assign_alliances_by_embedding(self) -> None:
        """Split robots into the two alliances by clustering their appearance
        embeddings into 2 groups (a tiny cosine 2-means), then label the clusters
        red/blue. More robust than per-frame bumper-color detection because red vs
        blue bumpers form a strong, stable appearance split (the basketball-style
        team-clustering trick, supervised by the known roster). Only UNCONFIRMED
        robots are (re)assigned; confirmed robots keep their team-derived alliance
        and act as cluster anchors. Falls back to color-cap enforcement if there
        aren't enough embeddings to cluster yet."""
        robots = [r for r in self.robot_tracks.values() if r.appearance_embedding is not None]
        # A 2-way split is only meaningful with enough robots; with just 1-3 a cluster
        # could separate same-alliance robots (and arbitrarily label them). Below the
        # threshold, fall back to each robot's aggregate color-vote majority (stable
        # over the match, unlike per-frame HSV).
        if len(robots) < 4:
            for r in robots:
                if r.team_number_confirmed:
                    continue
                if r.red_votes > r.blue_votes:
                    r.alliance = "red"
                elif r.blue_votes > r.red_votes:
                    r.alliance = "blue"
            self._enforce_alliance_caps_by_color()
            return

        embs = np.stack([np.asarray(r.appearance_embedding, dtype=np.float64) for r in robots])
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

        # 2-means with farthest-pair seeding (few points -> converges in a couple steps)
        sims = embs @ embs.T
        i0, j0 = np.unravel_index(int(np.argmin(sims)), sims.shape)
        c0, c1 = embs[i0].copy(), embs[j0].copy()
        labels = np.zeros(len(robots), dtype=int)
        for step in range(10):
            new = (embs @ c1 > embs @ c0).astype(int)
            if step > 0 and np.array_equal(new, labels):
                break
            labels = new
            m0, m1 = embs[labels == 0], embs[labels == 1]
            if len(m0):
                c0 = m0.mean(0); c0 /= np.linalg.norm(c0) + 1e-9
            if len(m1):
                c1 = m1.mean(0); c1 /= np.linalg.norm(c1) + 1e-9

        # Label the two clusters red/blue. Confirmed robots anchor strongly (their
        # team's alliance); unconfirmed contribute their accumulated color-vote margin.
        def cluster_score(k: int) -> float:
            s = 0.0
            for r, l in zip(robots, labels):
                if l != k:
                    continue
                if r.team_number_confirmed and r.team_number:
                    s += 100.0 if r.team_number in self.match_red_teams else -100.0
                else:
                    s += float(r.red_votes - r.blue_votes)
            return s

        cluster_alliance = ({0: "red", 1: "blue"} if cluster_score(0) >= cluster_score(1)
                            else {0: "blue", 1: "red"})
        for r, l in zip(robots, labels):
            if not r.team_number_confirmed:
                r.alliance = cluster_alliance[l]

        self._enforce_alliance_caps_by_color()

    def assign_teams_from_votes(self) -> None:
        """Assign teams from accumulated VLM read votes using an alliance-masked
        Hungarian, plus process-of-elimination for the last unread robot on an
        alliance. Alliance is roster-authoritative when a team is set and
        bumper-color-derived otherwise. Re-runs each interval; early misreads get
        outvoted over time.
        """
        if not self.match_teams:
            return

        locked_teams = {r.team_number for r in self.robot_tracks.values()
                        if r.team_number_confirmed and r.team_number}
        locked_robots = {r.robot_id for r in self.robot_tracks.values()
                         if r.team_number_confirmed}

        # Per-robot alliance for the constraint is the field-tracker SLOT alliance
        # (bumper color), set every frame by the loop sync — authoritative in SP2.
        # Team assignment is masked to the slot's alliance; it never flips it.
        alliance_of_robot = {
            r.robot_id: (r.alliance if r.alliance in ("red", "blue") else "unknown")
            for r in self.robot_tracks.values()
        }

        votes = {r.robot_id: dict(r.team_ocr_votes or {})
                 for r in self.robot_tracks.values()}

        assignment = reid.assign_teams_alliance_masked(
            votes, alliance_of_robot,
            self.match_blue_teams, self.match_red_teams,
            locked_teams, locked_robots,
        )

        # Apply Hungarian assignments.
        for rid, team in assignment.items():
            r = self.robot_tracks[rid]
            if r.team_number_confirmed:
                continue
            if team is None:
                r.team_number = None
                continue
            v = (r.team_ocr_votes or {}).get(team, 0.0)
            total = sum((r.team_ocr_votes or {}).values()) or 1.0
            r.team_number = team
            r.team_number_confirmed = False
            self.diagnostic.record_assignment(
                frame=self._current_frame, robot_id=rid, team=team,
                alliance=r.alliance, confirmed=r.team_number_confirmed,
                votes_for_assigned=float(v), votes_total=float(total),
                top_team_share=float(v / total) if total else 0.0,
            )
            for event in self.scoring_events:
                if event.robot_id == rid:
                    event.team_number = team

        # Process-of-elimination: last roster team + last unidentified bumper-colored
        # robot → assign and auto-confirm. Subject to overturn by the audit (Task 6).
        team_of_robot = {rid: self.robot_tracks[rid].team_number
                         for rid in self.robot_tracks}
        eliminated = reid.eliminate_last_team(
            alliance_of_robot, team_of_robot,
            self.match_blue_teams, self.match_red_teams,
        )
        for rid, team in eliminated.items():
            r = self.robot_tracks[rid]
            if r.team_number_confirmed:
                continue
            r.team_number = team
            r.team_number_confirmed = True
            if r.consecutive_tracker is None:
                r.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
            r.consecutive_tracker.locked_value = team
            r.consecutive_tracker.confirmed = True
            self._clear_team_from_other_robots(rid, team)
            self.diagnostic.record_elimination(
                frame=self._current_frame, robot_id=rid, team=team, alliance=r.alliance,
            )
            for event in self.scoring_events:
                if event.robot_id == rid:
                    event.team_number = team

    def _emit_diagnostic_snapshot(self, frame: int) -> None:
        """Build a snapshot dict from current state and hand to the recorder.

        Reads embedding-freshness fields defensively via getattr — those fields
        are populated by the re-ID instrumentation in Task 6; before that lands,
        the snapshot still works with sensible defaults.
        """
        robots_state = []
        for r in self.robot_tracks.values():
            last_embed = getattr(r, "last_embedding_frame", None)
            embedding_info = {
                # None when no embedding ever recorded for this canonical.
                "age_frames": (frame - last_embed) if last_embed is not None else None,
                "last_match_similarity": getattr(r, "last_reid_similarity", None),
                "last_match_gap": getattr(r, "last_reid_gap", None),
            }
            # Use `is None` not `or` — frame 0 is a real frame number.
            last_seen = getattr(r, "last_seen_frame", None)
            if last_seen is None:
                last_seen = r.positions_history[-1][0] if r.positions_history else frame
            robots_state.append({
                "robot_id": r.robot_id,
                "alliance": r.alliance,
                "team": r.team_number,
                "confirmed": bool(r.team_number_confirmed),
                "votes": dict(r.team_ocr_votes or {}),
                "red_votes": int(r.red_votes),
                "blue_votes": int(r.blue_votes),
                "last_seen_frame": int(last_seen),
                "embedding": embedding_info,
            })
        identified = {
            "red": [r.team_number for r in self.robot_tracks.values()
                    if r.team_number and r.alliance == "red"],
            "blue": [r.team_number for r in self.robot_tracks.values()
                     if r.team_number and r.alliance == "blue"],
        }
        roster = {"red": list(self.match_red_teams), "blue": list(self.match_blue_teams)}
        self.diagnostic.record_snapshot(
            frame=frame, robots_state=robots_state,
            roster=roster, identified=identified,
        )

    def _recent_events_data(self) -> list:
        """Display rows for the last 10 scoring events."""
        events_data = []
        for event in self.scoring_events[-10:]:
            robot = self.robot_tracks.get(event.robot_id)
            ident = self.identity_registry.identities.get(event.robot_id)
            slot_id = ident.active_slot if ident else None
            slot_label = self._slot_label(slot_id) if slot_id is not None else f"Robot {event.robot_id}"
            events_data.append({
                "frame": event.frame_num,
                "timestamp": event.timestamp,
                "alliance": event.alliance,
                "robot_id": event.robot_id,
                "slot": slot_label,
                "team_number": event.team_number,
                "team_candidate": (robot.team_number if robot else None),
                "robot_label": (
                    event.team_number
                    or (robot.team_number if robot and robot.team_number else None)
                    or slot_label
                ),
                "confidence": event.confidence
            })
        return events_data

    def _frame_stream_due(self, now: float) -> bool:
        """Rate-cap live-view frame sends to FRAME_STREAM_INTERVAL_SEC."""
        last = getattr(self, "_last_frame_stream_time", None)
        if last is not None and (now - last) < self.config.frame_stream_interval_sec:
            return False
        self._last_frame_stream_time = now
        return True

    # Websocket send channels. Each kind gets a bounded queue drained by a
    # background sender so a slow client never back-pressures inference:
    #   frame  — latest-only; the newest frame supersedes any unsent one
    #   status — latest-only; progress is latest-wins
    #   diag   — bounded backlog, sheds oldest under pressure; the JSONL file
    #            is the canonical record, the websocket copy is best-effort
    # Terminal messages (complete / error / final status) bypass the queues
    # via _send_ws after _stop_ws_senders has drained them.
    _WS_QUEUE_DEPTH = {"frame": 1, "status": 1, "diag": 64}

    @property
    def _frame_send_queue(self):
        return getattr(self, "_ws_queues", {}).get("frame")

    @property
    def _frame_sender_task(self):
        return getattr(self, "_ws_sender_tasks", {}).get("frame")

    def _ws_queue(self, kind: str) -> asyncio.Queue:
        queues = getattr(self, "_ws_queues", None)
        if queues is None:
            queues = self._ws_queues = {}
            self._ws_sender_tasks = {}
            self._ws_drop_counts = {}
        queue = queues.get(kind)
        if queue is None:
            queue = queues[kind] = asyncio.Queue(maxsize=self._WS_QUEUE_DEPTH[kind])
        task = self._ws_sender_tasks.get(kind)
        if task is None or task.done():
            self._ws_sender_tasks[kind] = asyncio.get_running_loop().create_task(
                self._ws_sender_loop(queue)
            )
        return queue

    def _enqueue_ws(self, kind: str, payload) -> None:
        """Queue a payload for its kind's sender without ever blocking the
        inference loop; when the queue is full the oldest payload is shed."""
        queue = self._ws_queue(kind)
        while True:
            try:
                queue.put_nowait(payload)
                return
            except asyncio.QueueFull:
                try:
                    dropped = queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                self._ws_drop_counts[kind] = self._ws_drop_counts.get(kind, 0) + 1
                # A dropped frame's unsent VLM crops ride along on the
                # replacement so a slow client still sees every crops batch.
                if (kind == "frame" and payload is not None and dropped is not None
                        and dropped.get("ocr_crops") is not None
                        and payload.get("ocr_crops") is None):
                    payload["ocr_crops"] = dropped["ocr_crops"]

    def _enqueue_frame_payload(self, payload) -> None:
        self._enqueue_ws("frame", payload)

    async def _ws_sender_loop(self, queue: asyncio.Queue) -> None:
        while True:
            payload = await queue.get()
            if payload is None:  # stop sentinel
                return
            await self._send_ws(payload)

    async def _stop_ws_senders(self, drain: bool = True) -> None:
        """Stop every sender task. With drain=True pending payloads are sent
        first (call before the `complete` message so ordering holds); with
        drain=False the tasks are cancelled immediately."""
        tasks = getattr(self, "_ws_sender_tasks", None)
        if not tasks:
            return
        queues = self._ws_queues
        self._ws_sender_tasks = {}
        pending = {k: t for k, t in tasks.items() if t is not None and not t.done()}
        for kind, task in pending.items():
            if drain:
                # Hand off the stop sentinel WITHOUT ever blocking completion
                # forever: an unbounded put() on these maxsize-1 queues wedged
                # a finished job for 60+ minutes when its sender was stuck
                # inside a stalled send. Fast path first; a full queue means
                # the sender is mid-send (or not yet scheduled), so give it
                # one bounded-send window (bounded _send_ws frees the slot
                # within WS_SEND_TIMEOUT) — the sentinel still lands BEHIND
                # any pending payload, preserving drain-before-complete
                # ordering for healthy senders — then give up and cancel.
                try:
                    queues[kind].put_nowait(None)
                except asyncio.QueueFull:
                    try:
                        await asyncio.wait_for(
                            queues[kind].put(None),
                            timeout=WS_SEND_TIMEOUT + WS_STOP_GRACE)
                    except asyncio.TimeoutError:
                        task.cancel()
            else:
                task.cancel()
        for task in pending.values():
            try:
                # Bounded for the same reason: at most one in-flight send
                # resolves at WS_SEND_TIMEOUT (detaching the socket, so the
                # rest of the backlog short-circuits); a task still alive
                # past that is wedged — cancel it rather than hang the job.
                await asyncio.wait_for(task, timeout=WS_SEND_TIMEOUT + WS_STOP_GRACE)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        diag_drops = getattr(self, "_ws_drop_counts", {}).get("diag", 0)
        if diag_drops:
            print(f"  [ws] shed {diag_drops} diagnostic record(s) under "
                  f"backpressure (JSONL on disk is complete)")

    async def _stop_frame_sender(self, drain: bool = True) -> None:
        await self._stop_ws_senders(drain=drain)

    async def _warn_degraded(self, component: str, message: str) -> None:
        """Surface a quality-degrading failure (SAM2 fallback, gallery disabled,
        evidence loss) to the client. One websocket message per component;
        repeats only bump the count so a per-frame failure can't spam."""
        warnings = getattr(self, "degradation_warnings", None)
        if warnings is None:
            warnings = self.degradation_warnings = []
        existing = next((w for w in warnings if w["component"] == component), None)
        if existing is not None:
            existing["count"] += 1
            return
        warnings.append({"component": component, "message": message, "count": 1})
        print(f"  [DEGRADED] {component}: {message}")
        await self._send_ws({
            "type": "degradation_warning",
            "component": component,
            "message": message,
        })

    async def _send_ws(self, payload: dict) -> bool:
        """Serialize websocket writes and isolate transport failures from inference."""
        websocket = self.websocket
        if websocket is None:
            return False
        try:
            async with self._websocket_send_lock:
                if self.websocket is not websocket:
                    return False
                # Bounded: uvicorn's send can await forever on a stalled
                # transport (even after the client disconnects), wedging any
                # sender under the send lock. wait_for CANCELS the underlying
                # send on timeout, which also releases the send lock for the
                # other senders; the socket is treated as failed below.
                await asyncio.wait_for(
                    websocket.send_json(payload), timeout=WS_SEND_TIMEOUT)
            return True
        except asyncio.TimeoutError:
            # Same treatment as a transport failure: a send that can't finish
            # within the bound is a dead/stalled socket, not a slow client
            # (queue shedding already absorbs slow-client backpressure).
            if self.websocket is websocket:
                self.websocket = None
            print(f"  WebSocket send timed out after {WS_SEND_TIMEOUT}s; "
                  f"detaching socket")
            return False
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            if self.websocket is websocket:
                self.websocket = None
            return False
        except Exception as e:
            if self.websocket is websocket:
                self.websocket = None
            print(f"  WebSocket send failed: {type(e).__name__}: {e}")
            return False

    async def _diagnostic_ws_send(self, record: dict) -> None:
        """Push one diagnostic record over the existing websocket. Best-effort:
        the JSONL file is the canonical record, so records ride the bounded
        diag queue and shed oldest if a slow client falls behind."""
        msg_type = ("diagnostic_snapshot"
                   if record.get("type") == "snapshot"
                   else "diagnostic_event")
        self._enqueue_ws("diag", {"type": msg_type, "record": record})

    def audit_overturn_confirmations(self) -> None:
        """Let a wrong locked identity self-correct. For each confirmed robot,
        if a different roster team's accumulated VLM votes exceed the current
        label's by `OVERTURN_VOTE_MARGIN`, unlock the robot — the next
        `assign_teams_from_votes` will re-assign it from the now-broader pool."""
        for r in self.robot_tracks.values():
            if not r.team_number_confirmed or not r.team_number:
                continue
            new_team = reid.audit_overturn(
                current_team=r.team_number,
                votes=r.team_ocr_votes or {},
                margin=OVERTURN_VOTE_MARGIN,
            )
            if new_team is None:
                continue
            self.diagnostic.record_overturn(
                frame=self._current_frame, robot_id=r.robot_id,
                old_team=r.team_number, challenger_team=new_team,
                current_votes=float((r.team_ocr_votes or {}).get(r.team_number, 0.0)),
                challenger_votes=float((r.team_ocr_votes or {}).get(new_team, 0.0)),
            )
            r.team_number_confirmed = False

    def assign_identities_now(self) -> None:
        """Periodic one-to-one assignment of unlocked tracks to reference teams.

        OCR/Qwen-confirmed tracks (and their teams) are locked out. The Hungarian
        match fills the rest with a uniqueness guarantee; results accumulate as
        votes and confirm once stable.
        """
        if not self._reference_embeddings:
            return
        track_embeds = {r.robot_id: r.appearance_embedding for r in self.robot_tracks.values()}
        alliance_of_track = {r.robot_id: r.alliance for r in self.robot_tracks.values()}
        alliance_of_team = {t: "blue" for t in self.match_blue_teams}
        alliance_of_team.update({t: "red" for t in self.match_red_teams})
        locked_tracks = {r.robot_id for r in self.robot_tracks.values() if r.team_number_confirmed}
        locked_teams = {r.team_number for r in self.robot_tracks.values()
                        if r.team_number_confirmed and r.team_number}

        assignment = reid.assign_identities(
            track_embeds, self._reference_embeddings,
            alliance_of_track, alliance_of_team,
            locked_tracks, locked_teams, ASSIGN_MAX_COST,
        )

        for robot_id, team in assignment.items():
            if team is None:
                continue
            robot = self.robot_tracks.get(robot_id)
            if robot is None or robot.team_number_confirmed:
                continue
            if robot.appearance_votes is None:
                robot.appearance_votes = {}
            robot.appearance_votes[team] = robot.appearance_votes.get(team, 0) + 1
            top = max(robot.appearance_votes, key=robot.appearance_votes.get)
            if robot.appearance_votes[top] >= APPEARANCE_CONFIRM_VOTES:
                robot.team_number = top
                robot.team_number_confirmed = True
                # Alliance stays the field-tracker slot (bumper) alliance (SP2).
                for event in self.scoring_events:
                    if event.robot_id == robot_id:
                        event.team_number = top

    def _clear_team_from_other_robots(self, confirmed_robot_id: int, team: str) -> None:
        """When a team is confirmed on one robot, clear it from all other robots.
        Removes tentative team_number assignments and purges the team from
        OCR/appearance vote dicts so no other robot can claim it."""
        for r in self.robot_tracks.values():
            if r.robot_id == confirmed_robot_id:
                continue
            # Clear this team from any other robot (tentative OR confirmed)
            # A team can only physically be on one robot
            if r.team_number == team:
                print(f"  Cleared team {team} from robot {r.robot_id} "
                      f"(confirmed={r.team_number_confirmed}, now confirmed on robot {confirmed_robot_id})")
                r.team_number = None
                r.team_number_confirmed = False
            # Purge from vote dicts so it can't accumulate votes elsewhere
            if r.team_ocr_votes and team in r.team_ocr_votes:
                del r.team_ocr_votes[team]
            if r.appearance_votes and team in r.appearance_votes:
                del r.appearance_votes[team]
            ident = self.identity_registry.identities.get(r.robot_id)
            if ident is not None and ident.team_number == team:
                ident.team_number = None
                ident.is_confirmed = False
                ident.confidence = min(ident.confidence, 0.4)
                ident.needs_immediate_read = True

    def ocr_score(self, overhead_frame: np.ndarray, alliance: str) -> Optional[int]:
        """OCR a single alliance score from the overhead crop of the scoreboard.
        Regions are in overhead crop coordinate space (drawn by user on overhead crop).
        Returns the raw OCR value if it's a valid number (0-999).
        Caller is responsible for calibration logic."""
        if overhead_frame is None or overhead_frame.size == 0:
            return None
        fh, fw = overhead_frame.shape[:2]
        region_key = f"{alliance}_center"
        if region_key not in self.ocr_regions:
            return None
        x, y, w, h = self.ocr_regions[region_key]
        if x + w > fw or y + h > fh or w <= 0 or h <= 0:
            return None
        roi = overhead_frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        # Preprocess
        roi_resized = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY)
        
        text = pytesseract.image_to_string(roi_thresh, config=OCR_CONFIG).strip()
        
        if text and text.isdigit():
            score = int(text)
            # Basic sanity: FRC scores don't exceed a few hundred
            if 0 <= score <= 999:
                return score
        
        return None

    def read_match_clock(self, overhead_frame: np.ndarray) -> Optional[int]:
        """Read the configured match-clock ROI from an overhead-crop frame."""
        if overhead_frame is None or overhead_frame.size == 0:
            return None
        region = self.ocr_regions.get("match_clock")
        if region is None:
            return None
        x, y, w, h = region
        fh, fw = overhead_frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
            return None
        roi = overhead_frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(
            thresholded, config=MATCH_CLOCK_OCR_CONFIG
        ).strip()
        return parse_match_clock_text(text)

    def _clock_scan_cache_path(self, sample_step: int) -> Path:
        """Cache key = video CONTENT fingerprint, not filename: every job
        re-uploads the same bytes under a fresh uuid name, so the name is
        useless as a key while size+head-hash is stable and cheap."""
        h = hashlib.md5()
        with open(self.video_path, "rb") as f:
            h.update(f.read(1 << 20))
        h.update(f"{self.video_path.stat().st_size}:{sample_step}:{self.overhead_height}".encode())
        return UPLOAD_DIR / f"clockscan_{h.hexdigest()}.json"

    def _scan_match_clock(self) -> List[Tuple[int, Optional[int]]]:
        """Sparse-decode the source video and OCR the clock every half second.

        Results are cached per video content: the scan is deterministic given
        the same bytes and sampling params, and at ~60s per scan it dominated
        repeat-analysis (A/B) turnaround."""
        sample_step = max(1, int(round(self.video_fps * MATCH_CLOCK_SAMPLE_SECONDS)))
        cache_path = self._clock_scan_cache_path(sample_step)
        try:
            cached = json.loads(cache_path.read_text())
            print(f"  Clock scan: cache hit ({cache_path.name})")
            return [(int(i), None if v is None else int(v)) for i, v in cached]
        except (OSError, ValueError):
            pass
        samples = []
        scan_cap = cv2.VideoCapture(str(self.video_path))
        try:
            for frame_idx in range(0, self.total_frames, sample_step):
                scan_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = scan_cap.read()
                if not ok:
                    continue
                overhead = frame[:self.overhead_height, :]
                samples.append((frame_idx, self.read_match_clock(overhead)))
        finally:
            scan_cap.release()
        try:
            cache_path.write_text(json.dumps(samples))
        except OSError:
            pass
        return samples
    
    def update_hub_positions(self, detections: sv.Detections) -> None:
        """Update hub positions and bounding boxes from hub_opening detections.
        If event_hub_positions are set, use them for alliance assignment (static).
        Otherwise fall back to midpoint-based initial assignment."""
        # Hub detections intentionally come from the existing 3-class model.
        hub_mask = detections.class_id == 2
        if not np.any(hub_mask):
            return
        
        hub_boxes = detections.xyxy[hub_mask]
        
        for box in hub_boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            found = False
            for hub in self.hubs:
                if abs(hub.x - x_center) < 100:
                    # Smooth with EMA
                    a = 0.3
                    hub.x = hub.x * (1 - a) + x_center * a
                    hub.y = hub.y * (1 - a) + y_center * a
                    hub.x1 = hub.x1 * (1 - a) + float(box[0]) * a
                    hub.y1 = hub.y1 * (1 - a) + float(box[1]) * a
                    hub.x2 = hub.x2 * (1 - a) + float(box[2]) * a
                    hub.y2 = hub.y2 * (1 - a) + float(box[3]) * a
                    found = True
                    break
            
            if not found and len(self.hubs) < 2:
                # Determine alliance from saved hub positions or fallback to midpoint
                alliance = self._assign_hub_alliance(x_center, y_center)
                self.hubs.append(DetectedHub(
                    x=x_center, y=y_center, alliance=alliance,
                    x1=float(box[0]), y1=float(box[1]),
                    x2=float(box[2]), y2=float(box[3])
                ))
        
        if len(self.hubs) == 2:
            self.hubs.sort(key=lambda h: h.x)

    def _assign_hub_alliance(self, x: float, y: float) -> str:
        """Assign alliance to a detected hub based on saved positions or midpoint."""
        if event_hub_positions:
            # Use saved hub positions — assign to closest saved position
            sx = self.video_width / 1920.0
            blue_pos = event_hub_positions.get('blue')
            red_pos = event_hub_positions.get('red')
            if blue_pos and red_pos:
                # Scale saved positions to current video resolution
                blue_x = blue_pos[0] * sx
                red_x = red_pos[0] * sx
                dist_blue = abs(x - blue_x)
                dist_red = abs(x - red_x)
                return "blue" if dist_blue < dist_red else "red"
        # Fallback: left = red, right = blue
        midpoint = self.video_width / 2
        return "red" if x < midpoint else "blue"

    def _team_hud_label(self, robot, fallback: str) -> tuple[str, bool]:
        """Return a display label and whether the team is confirmed.

        VLM/OCR votes may exist before the interval assignment step promotes a
        team into `robot.team_number`; show those leading votes so the HUD makes
        active team-number reads visible immediately.
        """
        if robot is None:
            return f"{fallback} no #", False
        if robot.team_number and robot.team_number_confirmed:
            # A confirmed team is the current best identity, never a permanent
            # lock — a sustained contradicting read will still override it. Drop
            # the old "(100%)" so the HUD never implies certainty; the absence of
            # a trailing "?" already distinguishes confirmed from a leading guess.
            # (cv2 Hershey fonts are ASCII-only, so no unicode marker here.)
            return f"{robot.team_number} S{robot.scores}", True

        best_team = robot.team_number
        best_vote = 0.0
        if robot.team_ocr_votes:
            # Cross-slot claim arbitration: a tentative "TEAM?" renders only
            # on the robot with the highest accumulated vote for that team —
            # votes gathered across batches otherwise put the same guess on
            # two slots at once (qm11: B1 and B3 both showed "2992?").
            winners = tentative_claim_winners({
                rid: getattr(r, "team_ocr_votes", None) or {}
                for rid, r in self.robot_tracks.items()
            })
            owned_team, owned_vote = best_owned_team(
                robot.team_ocr_votes, robot.robot_id, winners)
            if owned_team:
                best_team, best_vote = owned_team, owned_vote
            elif best_team and winners.get(best_team) not in (None, robot.robot_id):
                # its assigned team_number is also claimed elsewhere — show no #
                best_team = None
        if best_team:
            if best_vote:
                conf = min(99, int(best_vote / OCR_TEAM_CONFIRM_VOTES * 100))
            else:
                conf = 50
            return f"{best_team}? ({conf}%) S{robot.scores}", False
        return f"{fallback} no # S{robot.scores}", False

    def _nearest_track_for_detection(self, det, track_outs, max_dist_m: float = 0.85):
        best = None
        best_d = max_dist_m
        dp = np.array(det.pos, float)
        for t in track_outs:
            if t.state == "empty":
                continue
            d = float(np.linalg.norm(dp - np.array(t.pos, float)))
            if d <= best_d:
                best = t
                best_d = d
        return best

    def _draw_bumper_hud(self, img: np.ndarray, box, alliance: str, label: str,
                         slot_id: Optional[int] = None, cal=None, field_pos=None,
                         vel=None, mask: Optional[np.ndarray] = None) -> None:
        """Draw bumper-specific evidence over a camera view."""
        color = (255, 80, 20) if alliance == "blue" else (30, 30, 255) if alliance == "red" else (190, 190, 190)

        SLOT_COLORS = {
            0: (0, 220, 0),       # slot 0 (B1): Green
            1: (255, 220, 0),     # slot 1 (B2): Cyan/Light Blue
            2: (0, 220, 220),     # slot 2 (B3): Yellow
            3: (30, 30, 255),     # slot 3 (R1): Red
            4: (225, 50, 225),    # slot 4 (R2): Magenta/Pink
            5: (0, 165, 255),     # slot 5 (R3): Orange
        }
        mask_color = SLOT_COLORS.get(slot_id, color) if slot_id is not None else color

        x1, y1, x2, y2 = map(int, box)
        
        # Draw SAM2 mask overlay if present
        if mask is not None and HUD_DRAW_ROBOT_MASK:
            overlay = img.copy()
            overlay[mask] = mask_color
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        if HUD_DRAW_BUMPER_BAND:
            band = bumper_band_box(img, box)
            if band is not None:
                bx1, by1, bx2, by2 = band
                overlay = img.copy()
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
                cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
                cv2.line(img, (bx1, by2), (bx2, by2), (255, 255, 255), 2)

        if HUD_DRAW_ROBOT_BOX:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if HUD_DRAW_ROBOT_LABEL:
            text = f"{self._slot_label(slot_id)} {label}" if slot_id is not None else label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 9)), (x1 + tw + 7, y1), (0, 0, 0), -1)
            cv2.putText(img, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)

        if cal is not None and field_pos is not None and (HUD_DRAW_GROUND_PLANE or HUD_DRAW_VELOCITY_ARROW):
            self._draw_projected_robot_plane(img, cal, field_pos, vel, color, box)

    def _draw_projected_robot_plane(self, img: np.ndarray, cal, field_pos, vel, color, box=None) -> None:
        """Draw bumper-ground evidence and a separate motion vector.

        Robot bumper/chassis orientation and robot travel direction are different
        signals. The debug plane is anchored to the detected bumper bottom edge
        in image space so it visualizes the measurement instead of a potentially
        lagged/smoothed track center.
        """
        try:
            x, y = float(field_pos[0]), float(field_pos[1])
            vx, vy = (float(vel[0]), float(vel[1])) if vel is not None else (0.0, 0.0)

            if box is not None and HUD_DRAW_GROUND_PLANE:
                band = bumper_band_box(img, box)
                if band is not None:
                    bx1, by1, bx2, by2 = band
                    bottom_y = by2
                    # A visual ground contact strip: lower edge is the bumper
                    # bottom, upper edge is just above it. This stays attached to
                    # the detected robot even when the field track is smoothed.
                    height = max(10, int((by2 - by1) * 0.45))
                    skew = int((bx2 - bx1) * 0.12)
                    poly = np.array([
                        (bx1, bottom_y),
                        (bx2, bottom_y),
                        (bx2 - skew, bottom_y + height),
                        (bx1 + skew, bottom_y + height),
                    ], np.int32)
                    h, w = img.shape[:2]
                    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
                    cv2.polylines(img, [poly], True, color, 2)

            speed = float(np.hypot(vx, vy))
            center = cal.field_to_image((x, y, 0.0))
            if HUD_DRAW_VELOCITY_ARROW and speed >= 0.15 and center is not None:
                n = max(speed, 1e-6)
                arrow_len = min(1.0, 0.35 + speed * 0.20)
                tip = cal.field_to_image((x + arrow_len * vx / n, y + arrow_len * vy / n, 0.0))
                if tip is None:
                    return
                cv2.arrowedLine(img, (int(center[0]), int(center[1])),
                                (int(tip[0]), int(tip[1])), (255, 255, 255), 2,
                                tipLength=0.35)
        except Exception:
            return

    def _camera_panel(self, view, composite_frame, paired, track_outs=None, max_width=720) -> np.ndarray:
        panel = view.extract(composite_frame)
        track_outs = track_outs or []
        for det, _crop_b64 in paired:
            t = self._nearest_track_for_detection(det, track_outs)
            robot = self._get_or_create_slot_track(t.slot_id) if t is not None else None
            fallback = f"{det.color} bumper" if det.color != "unknown" else "bumper?"
            label, _confirmed = self._team_hud_label(robot, fallback)
            alliance = robot.alliance if robot is not None else det.color
            self._draw_bumper_hud(
                panel, det.box, alliance, label,
                slot_id=t.slot_id if t is not None else None,
                cal=view.calibration,
                field_pos=t.pos if t is not None else det.pos,
                vel=t.vel if t is not None else None,
            )
        h, w = panel.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            panel = cv2.resize(panel, (max_width, int(h * scale)))
        header_h = 26
        out = np.full((panel.shape[0] + header_h, panel.shape[1], 3), 18, np.uint8)
        out[header_h:, :] = panel
        cv2.putText(out, view.name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
        return out

    def _maybe_draw_field_wireframe(self, frame) -> None:
        """When the field_wireframe_overlay knob is on, draw the reprojected field
        (perimeter, grid, hubs, bumps, towers, trenches) onto the main-view frame.
        ``frame`` is crop-local (origin at the main crop's top-left), matching what
        cal.field_to_image projects to. Best-effort: a poorly-conditioned calibration
        can make projection blow up, so a draw failure must never sink inference."""
        if not getattr(self.config, "field_wireframe_overlay", False):
            return
        if self.calib is None or self.field_model is None:
            return
        try:
            _draw_field_overlay(frame, self.calib, self.field_model)
        except Exception as e:
            print(f"  [wireframe] overlay draw skipped: {e}")

    def _compose_hud_frame(self, annotated_frame, radar_img, camera_panels) -> np.ndarray:
        """Stack a larger field/corner-camera dashboard below the main view."""
        ah, aw = annotated_frame.shape[:2]
        dash_h = max(300, min(560, int(ah * 0.48)))
        dash = np.full((dash_h, aw, 3), 16, np.uint8)
        pad = 12
        title_h = 30
        body_y = pad + title_h
        body_h = dash_h - body_y - pad

        def paste_fit(dst, src, x, y, w, h, max_upscale=1.8):
            sh, sw = src.shape[:2]
            if sh <= 0 or sw <= 0 or w <= 0 or h <= 0:
                return
            scale = min(w / sw, h / sh, max_upscale)
            out = cv2.resize(src, (max(1, int(sw * scale)), max(1, int(sh * scale))))
            ox = x + max(0, (w - out.shape[1]) // 2)
            oy = y + max(0, (h - out.shape[0]) // 2)
            dst[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out

        map_w = int(aw * 0.34) if camera_panels else aw - 2 * pad
        cv2.putText(dash, "FIELD MAP", (pad, pad + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (235, 235, 235), 1, cv2.LINE_AA)
        paste_fit(dash, radar_img, pad, body_y, map_w - pad, body_h, max_upscale=2.0)

        if camera_panels:
            cams_x = map_w + pad
            cams_w = aw - cams_x - pad
            cv2.putText(dash, "CORNER CAMERA HUD", (cams_x, pad + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
            n = len(camera_panels)
            gap = 10
            cell_w = max(1, (cams_w - gap * (n - 1)) // n)
            for i, panel in enumerate(camera_panels):
                x = cams_x + i * (cell_w + gap)
                cv2.rectangle(dash, (x, body_y), (x + cell_w, body_y + body_h),
                              (55, 55, 55), 1)
                paste_fit(dash, panel, x + 4, body_y + 4, cell_w - 8, body_h - 8,
                          max_upscale=1.7)

        cv2.line(dash, (0, 0), (aw, 0), (80, 80, 80), 1)
        return np.vstack([annotated_frame, dash])

    def detect_hub_alliance(self, frame: np.ndarray) -> None:
        """Detect hub alliance colors from the surrounding field area.
        Looks at the ramp/floor color around each hub to determine alliance."""
        if len(self.hubs) < 2:
            return
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for hub in self.hubs:
            # Sample area around hub (slightly expanded)
            x1 = max(0, int(hub.x1) - 30)
            y1 = max(0, int(hub.y1) - 30)
            x2 = min(frame.shape[1], int(hub.x2) + 30)
            y2 = min(frame.shape[0], int(hub.y2) + 30)
            
            region = hsv[y1:y2, x1:x2]
            if region.size == 0:
                continue
            
            red_mask1 = cv2.inRange(region, np.array([0, 60, 60]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(region, np.array([160, 60, 60]), np.array([180, 255, 255]))
            red_pixels = cv2.countNonZero(red_mask1 | red_mask2)
            
            blue_mask = cv2.inRange(region, np.array([100, 60, 60]), np.array([130, 255, 255]))
            blue_pixels = cv2.countNonZero(blue_mask)
            
            if red_pixels > blue_pixels * 1.5:
                hub.alliance = "red"
            elif blue_pixels > red_pixels * 1.5:
                hub.alliance = "blue"
            # Otherwise keep existing assignment

    def update_ball_trajectories(self, detections: sv.Detections, frame_num: int) -> None:
        """Track ball positions across frames using nearest-neighbor linking.
        
        Each detected ball is matched to the closest existing trajectory.
        Unmatched detections start new trajectories.
        Trajectories that haven't been seen for BALL_MAX_MISSING frames are marked dead.
        """
        # Fuel detections intentionally come from the existing 3-class model.
        ball_mask = detections.class_id == 0
        if not np.any(ball_mask):
            # No balls detected — increment missing counter on all active trajectories
            for traj in self.ball_trajectories:
                if traj.alive:
                    traj.frames_missing += 1
                    if traj.frames_missing > self.BALL_MAX_MISSING:
                        traj.alive = False
            return
        
        ball_boxes = detections.xyxy[ball_mask]
        ball_centers = []
        for box in ball_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            ball_centers.append((float(cx), float(cy), box))
        
        # Match each detection to the closest active trajectory
        matched_trajs = set()
        matched_dets = set()
        
        # Build cost matrix: distance from each detection to each active trajectory's last position
        active_trajs = [(i, t) for i, t in enumerate(self.ball_trajectories) if t.alive]
        
        if active_trajs:
            for det_idx, (cx, cy, box) in enumerate(ball_centers):
                best_dist = self.BALL_LINK_DISTANCE
                best_traj_idx = -1
                
                for traj_list_idx, (global_idx, traj) in enumerate(active_trajs):
                    if global_idx in matched_trajs:
                        continue
                    last_pos = traj.positions[-1]
                    dist = np.sqrt((cx - last_pos[1])**2 + (cy - last_pos[2])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_traj_idx = global_idx
                
                if best_traj_idx >= 0:
                    # Extend existing trajectory
                    self.ball_trajectories[best_traj_idx].positions.append((frame_num, cx, cy))
                    self.ball_trajectories[best_traj_idx].frames_missing = 0
                    matched_trajs.add(best_traj_idx)
                    matched_dets.add(det_idx)
        
        # Unmatched detections → new trajectories
        for det_idx, (cx, cy, box) in enumerate(ball_centers):
            if det_idx not in matched_dets:
                new_traj = BallTrajectory(
                    ball_id=self.next_ball_id,
                    positions=[(frame_num, cx, cy)]
                )
                self.ball_trajectories.append(new_traj)
                self.next_ball_id += 1
        
        # Mark dead trajectories — but only those that existed before this frame
        # (don't penalize freshly created trajectories)
        matched_ball_ids = {self.ball_trajectories[i].ball_id for i in matched_trajs} if matched_trajs else set()
        for traj in self.ball_trajectories:
            if traj.alive and len(traj.positions) > 1 and traj.ball_id not in matched_ball_ids:
                # Only increment missing if this trajectory existed before this frame
                # (trajectories with exactly 1 position were just created)
                traj.frames_missing += 1
                if traj.frames_missing > self.BALL_MAX_MISSING:
                    traj.alive = False
        
        # Check if any trajectories entered a hub
        # Requirements to score:
        #   1. At least 2 positions (must be tracked across frames, not a single detection)
        #   2. Must have moved at least MIN_SCORING_DISPLACEMENT px from first to last position
        #      (eliminates stationary ground balls near hubs)
        #   3. Must have minimum trajectory speed and move toward the center from alliance side
        #   3. Per-hub cooldown: no two scores at nearly the same location within SCORE_LOCATION_COOLDOWN frames
        MIN_SCORING_DISPLACEMENT = 8.0  # px — stationary balls move < 3-4px from jitter
        MIN_SPEED = 3.0  # px/frame — slow drifts are not shots
        SCORE_LOCATION_COOLDOWN = 10  # frames before same spot can score again
        SCORE_LOCATION_RADIUS = 25.0  # px — how close is "same location"
        MAX_SCORES_PER_FRAME_PER_ALLIANCE = 2  # cap runaway scoring
        
        # Track scores this frame to enforce per-frame cap
        scores_this_frame = {"blue": 0, "red": 0}
        
        # Initialize spatial cooldown tracking if not present
        if not hasattr(self, '_score_location_cooldown'):
            self._score_location_cooldown = []  # list of (frame, x, y, alliance)
        
        # Determine field midpoint from detected hubs when available.
        if len(self.hubs) >= 2:
            field_midpoint_x = (min(h.x for h in self.hubs) + max(h.x for h in self.hubs)) / 2.0
        else:
            field_midpoint_x = self.video_width / 2.0
        
        for traj in self.ball_trajectories:
            if traj.scored or len(traj.positions) < 2:
                continue
            
            # Check displacement: ball must have actually moved
            first_frame, first_x, first_y = traj.positions[0]
            last_frame, last_x, last_y = traj.positions[-1]
            frame_span = max(1, last_frame - first_frame)
            dx = last_x - first_x
            dy = last_y - first_y
            speed = np.sqrt(dx**2 + dy**2) / frame_span
            displacement = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
            if displacement < MIN_SCORING_DISPLACEMENT:
                continue
            if speed < MIN_SPEED:
                continue
            
            for hub in self.hubs:
                if hub.x2 == 0:
                    continue
                
                # Directional gate: shot must come from alliance side toward center hub opening.
                # Left-side hub expects +dx (left -> right), right-side hub expects -dx.
                if hub.x < field_midpoint_x:
                    if dx <= 0:
                        continue
                else:
                    if dx >= 0:
                        continue
                    
                # Per-frame cap
                if scores_this_frame.get(hub.alliance, 0) >= MAX_SCORES_PER_FRAME_PER_ALLIANCE:
                    continue
                
                # Check if ball center is within hub bbox (with margin)
                margin = 15.0
                if (last_x > hub.x1 - margin and last_x < hub.x2 + margin and
                    last_y > hub.y1 - margin and last_y < hub.y2 + margin):
                    
                    # Spatial cooldown: skip if a recent score happened at nearly the same spot
                    too_close = False
                    for (sc_frame, sc_x, sc_y, sc_alliance) in self._score_location_cooldown:
                        if sc_alliance == hub.alliance and (last_frame - sc_frame) < SCORE_LOCATION_COOLDOWN:
                            loc_dist = np.sqrt((last_x - sc_x)**2 + (last_y - sc_y)**2)
                            if loc_dist < SCORE_LOCATION_RADIUS:
                                too_close = True
                                break
                    if too_close:
                        traj.scored = True  # Mark scored to prevent re-checking
                        traj.alive = False
                        break
                    # Ball entered hub — trace trajectory back to find shooter
                    traj.scored = True
                    traj.scored_alliance = hub.alliance
                    traj.alive = False
                    
                    # Find the robot closest to the ball's ORIGIN (earliest position)
                    shooter = self._find_shooter_from_trajectory(traj, hub.alliance)
                    traj.shooter_robot_id = shooter
                    
                    self._ball_score_cooldown[hub.alliance] = last_frame
                    
                    # Record spatial cooldown for this scoring location
                    self._score_location_cooldown.append((last_frame, last_x, last_y, hub.alliance))
                    # Prune old entries
                    self._score_location_cooldown = [
                        e for e in self._score_location_cooldown 
                        if (last_frame - e[0]) < SCORE_LOCATION_COOLDOWN * 2
                    ]
                    scores_this_frame[hub.alliance] = scores_this_frame.get(hub.alliance, 0) + 1
                    shooter_name = None
                    if shooter is not None and shooter in self.robot_tracks:
                        shooter_name = self.robot_tracks[shooter].team_number or shooter
                    print(
                        f"  TRAJECTORY SCORE CANDIDATE: Ball {traj.ball_id} -> {hub.alliance} "
                        f"(shooter={shooter_name if shooter_name is not None else 'unknown'}, "
                        f"traj={len(traj.positions)} frames, speed={speed:.2f}px/frame, dx={dx:.1f})"
                    )
                    
                    self.scored_balls.append(traj)
                    break
        
        # Cleanup: remove very old dead trajectories (keep last 100)
        if len(self.ball_trajectories) > 200:
            self.ball_trajectories = [t for t in self.ball_trajectories if t.alive] + \
                                     [t for t in self.ball_trajectories if not t.alive][-100:]
    
    def _find_shooter_from_trajectory(self, traj: BallTrajectory, hub_alliance: str) -> Optional[int]:
        """Trace a ball's trajectory backward to find which robot shot it.
        
        Strategy:
        1. Look at the ball's earliest positions (origin)
        2. Find the robot closest to that origin point AT THAT TIME
        3. Prefer robots of the same alliance as the hub
        """
        if len(traj.positions) < 1:
            return None
        
        # Use the earliest 1-2 positions as the "origin zone"
        origin_positions = traj.positions[:min(2, len(traj.positions))]
        origin_x = np.mean([p[1] for p in origin_positions])
        origin_y = np.mean([p[2] for p in origin_positions])
        origin_frame = origin_positions[0][0]
        
        # Find robots and their positions near the origin time
        best_robot = None
        best_dist = float('inf')
        
        for robot in self.robot_tracks.values():
            # Find robot's position closest to the ball's origin frame
            robot_pos = robot.last_position
            
            if robot.positions_history:
                # positions_history stores (frame_num, x, y) — find closest frame
                closest_pos = None
                closest_frame_diff = float('inf')
                for pos in robot.positions_history:
                    frame_diff = abs(pos[0] - origin_frame)
                    if frame_diff < closest_frame_diff:
                        closest_frame_diff = frame_diff
                        closest_pos = (pos[1], pos[2])
                if closest_pos:
                    robot_pos = closest_pos
            
            dist = np.sqrt((robot_pos[0] - origin_x)**2 + (robot_pos[1] - origin_y)**2)
            
            # Prefer same-alliance robots (give them a distance bonus)
            effective_dist = dist
            if robot.alliance == hub_alliance:
                effective_dist *= 0.6  # Same alliance gets ~1.7x preference
            
            if effective_dist < best_dist:
                best_dist = effective_dist
                best_robot = robot.robot_id
        
        return best_robot

    def ball_overlaps_hub(self, ball_box: np.ndarray, hub: DetectedHub, margin: float = 10.0) -> bool:
        """Check if a ball bounding box overlaps with a hub bounding box (with margin)."""
        # Expand hub box by margin to catch balls entering the zone
        hx1 = hub.x1 - margin
        hy1 = hub.y1 - margin
        hx2 = hub.x2 + margin
        hy2 = hub.y2 + margin
        bx1, by1, bx2, by2 = ball_box[0], ball_box[1], ball_box[2], ball_box[3]
        return bx1 < hx2 and bx2 > hx1 and by1 < hy2 and by2 > hy1

    def detect_ball_in_hub_scores(self, detections: sv.Detections, frame_num: int) -> List[Tuple[str, float, float]]:
        """Detect balls overlapping with hub bounding boxes. 
        Returns list of (alliance, ball_cx, ball_cy) for new scoring events.
        Uses a cooldown to avoid counting the same ball multiple times."""
        
        ball_mask = detections.class_id == 0
        if not np.any(ball_mask):
            return []
        
        ball_boxes = detections.xyxy[ball_mask]
        scores = []
        
        for ball_box in ball_boxes:
            ball_cx = (ball_box[0] + ball_box[2]) / 2
            ball_cy = (ball_box[1] + ball_box[3]) / 2
            
            for hub in self.hubs:
                if hub.x2 == 0:  # Hub bbox not yet established
                    continue
                if self.ball_overlaps_hub(ball_box, hub, margin=15.0):
                    # Cooldown: don't count same alliance more than once every 10 processed frames
                    last_frame = self._ball_score_cooldown.get(hub.alliance, -999)
                    if frame_num - last_frame >= 10:
                        scores.append((hub.alliance, float(ball_cx), float(ball_cy)))
                        self._ball_score_cooldown[hub.alliance] = frame_num
        
        return scores
    
    def _alliance_full(self, alliance: str, exclude_robot_id: int) -> bool:
        """True if `alliance` already holds the max robots (excluding the given one).

        Used to keep alliance (re)assignment from ever exceeding 3 robots per
        alliance — the physical limit on an FRC field.
        """
        count = sum(1 for r in self.robot_tracks.values()
                    if r.alliance == alliance and r.robot_id != exclude_robot_id)
        return count >= MAX_ROBOTS_PER_ALLIANCE

    def get_canonical_robot_id(self, bytetrack_id: int, position: Tuple[float, float],
                               alliance: str, det_embed: Optional[np.ndarray],
                               claimed: set) -> Optional[int]:
        """Get or create the canonical robot ID for a ByteTrack ID.

        Enforces one canonical track per physical robot PER FRAME: a canonical
        already in `claimed` cannot be (re)used this frame. Cached ByteTrack
        mappings are re-validated by appearance so a ByteTrack ID-swap doesn't
        silently feed a different robot into the same track. Returns None when no
        canonical can be issued (caps full and all tracks claimed) — caller skips
        the detection.
        """
        cached = self.bytetrack_to_canonical.get(bytetrack_id)
        if cached is not None and cached in self.robot_tracks:
            track = self.robot_tracks[cached]
            appearance_ok = (
                det_embed is None or track.appearance_embedding is None
                or cosine_similarity(det_embed, track.appearance_embedding) >= APP_FLOOR
            )
            if appearance_ok and cached not in claimed:
                claimed.add(cached)
                track.last_seen_frame = self._current_frame
                return cached
            # Stale (appearance diverged) or contested (already claimed): drop the
            # mapping and re-resolve from scratch below.
            del self.bytetrack_to_canonical[bytetrack_id]

        tracks = [
            reid.TrackView(robot_id=r.robot_id, alliance=r.alliance,
                           position=r.last_position, embedding=r.appearance_embedding)
            for r in self.robot_tracks.values()
        ]
        action, rid = reid.choose_canonical_match(
            det_pos=position, det_embed=det_embed, det_alliance=alliance,
            tracks=tracks, app_floor=APP_FLOOR, pos_gate=POS_GATE_PX,
            w_pos=W_POS, w_app=W_APP,
            cap_per_alliance=MAX_ROBOTS_PER_ALLIANCE, cap_total=MAX_ROBOTS_TOTAL,
            claimed=claimed,
        )
        # Derive appearance similarities locally (choose_canonical_match does not
        # expose them). Top/second-best are over tracks with an embedding; the
        # "winning" similarity for the match record is the similarity vs the
        # chosen canonical when available, else the global top.
        candidate_sims: Dict[int, float] = {}
        if det_embed is not None:
            for t in tracks:
                if t.embedding is not None:
                    candidate_sims[t.robot_id] = float(cosine_similarity(det_embed, t.embedding))
        sorted_sims = sorted(candidate_sims.values(), reverse=True)
        global_top = sorted_sims[0] if sorted_sims else 0.0
        second_similarity = sorted_sims[1] if len(sorted_sims) >= 2 else None

        if action == "drop":
            # Caps full AND every track already claimed this frame. No candidate
            # remains to record as "would_have_matched"; leave it None.
            # TODO: if reid.choose_canonical_match gains new drop reasons,
            # widen its return tuple and thread the reason out instead of hardcoding.
            self.diagnostic.record_reid_drop(
                frame=self._current_frame, track_id=bytetrack_id,
                reason="caps_full_all_claimed",
                would_have_matched=None,
                similarity=None,
            )
            return None
        if action in ("merge", "assign"):
            self.bytetrack_to_canonical[bytetrack_id] = rid
            claimed.add(rid)
            top_similarity = candidate_sims.get(rid, global_top)
            track = self.robot_tracks.get(rid)
            if track is not None:
                track.last_embedding_frame = self._current_frame
                track.last_seen_frame = self._current_frame
                track.last_reid_similarity = float(top_similarity)
                track.last_reid_gap = (
                    float(top_similarity - second_similarity)
                    if second_similarity is not None else None
                )
            self.diagnostic.record_reid_match(
                frame=self._current_frame, track_id=bytetrack_id,
                canonical_id=rid,
                similarity=float(top_similarity),
                second_best=float(second_similarity) if second_similarity is not None else None,
            )
            return rid
        # action == "create"
        canonical_id = self.next_canonical_id
        self.next_canonical_id += 1
        # last_embedding_frame is set ONLY when an embedding actually exists;
        # otherwise leave it at the default None so age_frames reports
        # "never embedded" rather than "embedded at frame 0".
        new_track = RobotTrack(
            robot_id=canonical_id, alliance=alliance, team_number=None,
            last_position=position, positions_history=deque(maxlen=30),
            appearance_embedding=det_embed,
            last_seen_frame=self._current_frame,
        )
        if det_embed is not None:
            new_track.last_embedding_frame = self._current_frame
        self.robot_tracks[canonical_id] = new_track
        self.bytetrack_to_canonical[bytetrack_id] = canonical_id
        claimed.add(canonical_id)
        return canonical_id

    @staticmethod
    def _slot_alliance(slot_id: int) -> str:
        """Slots 0-2 are blue, 3-5 are red (matches FieldTracker)."""
        return "blue" if slot_id < 3 else "red"

    @staticmethod
    def _slot_label(slot_id: int) -> str:
        """Human label for a slot: B1..B3 / R1..R3."""
        return f"{'B' if slot_id < 3 else 'R'}{slot_id % 3 + 1}"

    def _update_portals(self, track_outs, dt) -> None:
        """Maintain occlusion exit portals across frames (occlusion_portal knob).

        - On a slot transitioning INTO hub-occluded coasting, project its
          constant-velocity ray through each hub prism and stash the first
          exit point as (x, y, 0.8) — the seed re-acquisition portal.
        - While it stays occ-coasting, grow the radius by 0.5*dt per frame
          (coast uncertainty) up to 2.5 m.
        - Clear the entry once the slot is re-acquired (visible) or gone.
        - Log-only teleport suspicion: a slot whose position jumped > 2.0 m
          this frame along a path crossing a hub, while NOT occ-coasting.
        """
        states = {t.slot_id: t for t in track_outs}
        # Teleport-suspect logging (log-only, never rejects an association).
        for sid, t in states.items():
            prev = self._portal_prev_pos.get(sid)
            new = (float(t.pos[0]), float(t.pos[1]))
            occ_coasting = (t.state == "coasting"
                            and self._occ.get(sid, {}).get("cause") == "hub")
            if (prev is not None and not occ_coasting
                    and float(np.hypot(new[0] - prev[0], new[1] - prev[1])) > 2.0
                    and any(path_crosses_footprint(prev, new, p.footprint)
                            for p in self.hub_prisms)):
                print(f"  [portal] teleport-suspect slot {sid}: "
                      f"{prev}->{new} crosses hub")
            self._portal_prev_pos[sid] = new

        for sid, t in states.items():
            hub_coasting = (t.state == "coasting"
                            and self._occ.get(sid, {}).get("cause") == "hub")
            if not hub_coasting:
                # Re-acquired or no longer hub-occluded: drop any portal.
                self._portal_state.pop(sid, None)
                continue
            if sid in self._portal_state:
                # Established portal: grow radius with coast uncertainty.
                px, py, pr = self._portal_state[sid]
                self._portal_state[sid] = (px, py, min(2.5, pr + 0.5 * dt))
                continue
            # New transition into hub-occluded coasting: predict the exit.
            for prism in self.hub_prisms:
                hit = predict_exit(t.pos, t.vel, prism.footprint)
                if hit is not None:
                    (ex, ey), _eta = hit
                    self._portal_state[sid] = (ex, ey, 0.8)
                    break

        # Drop portals for slots no longer present at all.
        for sid in [s for s in self._portal_state if s not in states]:
            self._portal_state.pop(sid, None)
            self._portal_prev_pos.pop(sid, None)

    def _identity_for_slot(self, slot_id: int) -> Optional[RobotIdentity]:
        """Find the active identity currently bound to a slot."""
        if not hasattr(self, "identity_registry") or self.identity_registry is None:
            self.identity_registry = IdentityRegistry()
            self.robot_identities = self.identity_registry.identities
            self._last_bindings = {}
        for ident in self.identity_registry.identities.values():
            if ident.active_slot == slot_id:
                return ident
        return None

    def _get_or_create_identity_track(self, identity_id: int, alliance: str) -> "RobotTrack":
        """Lazily create or retrieve the RobotTrack for a durable identity."""
        track = self.robot_tracks.get(identity_id)
        if track is None:
            track = RobotTrack(
                robot_id=identity_id,
                alliance=alliance,
                team_number=None,
                last_position=(0.0, 0.0),
                positions_history=deque(maxlen=30),
                consecutive_tracker=ConsecutiveValueTracker(n_consecutive=3),
            )
            self.robot_tracks[identity_id] = track
        elif not hasattr(track, "consecutive_tracker") or track.consecutive_tracker is None:
            try:
                track.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
            except AttributeError:
                pass
        return track

    def _get_slot_track_if_exists(self, slot_id: int) -> Optional[RobotTrack]:
        """Retrieve the RobotTrack for a slot if it already exists, without creating it lazily."""
        ident = self._identity_for_slot(slot_id)
        if ident is not None:
            return self.robot_tracks.get(ident.identity_id)
        return self.robot_tracks.get(slot_id)

    def _get_or_create_slot_track(self, slot_id: int) -> "RobotTrack":
        """Redirect slot lookup to the active identity's track.
        If no identity is currently bound to this slot, fall back to creating one."""
        ident = self._identity_for_slot(slot_id)
        if ident is not None:
            return self._get_or_create_identity_track(ident.identity_id, ident.alliance)
        
        alliance = self._slot_alliance(slot_id)
        ident = self.identity_registry.identities.get(slot_id)
        if ident is None:
            ident = RobotIdentity(
                identity_id=slot_id,
                alliance=alliance,
                active_slot=slot_id,
                confidence=0.5
            )
            self.identity_registry.identities[slot_id] = ident
            if slot_id >= self.identity_registry._next_id:
                self.identity_registry._next_id = slot_id + 1
        return self._get_or_create_identity_track(ident.identity_id, alliance)

    def _canonical_identity_slots(self) -> dict[int, int]:
        """Map at most three current identities per alliance onto the six slots.

        Active bindings win. Inactive identities occupy the remaining alliance
        slots so occlusion never creates an extra UI row.
        """
        assignments = {}
        identities = list(self.identity_registry.identities.values())
        for alliance, slots in (("blue", range(3)), ("red", range(3, 6))):
            slot_set = set(slots)
            alliance_ids = [i for i in identities if i.alliance == alliance]
            used_ids = set()
            for ident in alliance_ids:
                if ident.active_slot in slot_set and ident.active_slot not in assignments:
                    assignments[ident.active_slot] = ident.identity_id
                    used_ids.add(ident.identity_id)
            remaining = [i for i in alliance_ids if i.identity_id not in used_ids]
            remaining.sort(
                key=lambda i: (
                    not bool(getattr(i, "is_confirmed", False)),
                    -int(getattr(i, "last_seen_frame", 0)),
                    i.identity_id,
                )
            )
            free_slots = [slot for slot in slots if slot not in assignments]
            for slot, ident in zip(free_slots, remaining):
                assignments[slot] = ident.identity_id
        return assignments

    def _canonical_robot_rows(self) -> list[dict]:
        """Serialize exactly the fixed 3v3 roster slots for live/final results."""
        assignments = self._canonical_identity_slots()
        rows = []
        for slot_id in range(6):
            identity_id = assignments.get(slot_id)
            robot = self.robot_tracks.get(identity_id) if identity_id is not None else None
            best_vote_team = None
            best_vote = 0.0
            if robot is not None and robot.team_ocr_votes:
                best_vote_team, best_vote = max(
                    robot.team_ocr_votes.items(), key=lambda kv: kv[1]
                )
            rows.append({
                "id": identity_id,
                "slot": self._slot_label(slot_id),
                "slot_id": slot_id,
                "alliance": self._slot_alliance(slot_id),
                "team_number": (
                    self._get_display_team_number(robot) if robot is not None else None
                ),
                "confirmed": bool(robot and robot.team_number_confirmed),
                "team_candidate": best_vote_team,
                "team_candidate_votes": best_vote,
                "scores": robot.scores if robot is not None else 0,
                "position": robot.last_position if robot is not None else None,
            })
        return rows

    async def _bind_detection_identity(self, slot_id: int, identity_id: int):
        ident = self.identity_registry.identities.get(identity_id)
        if ident is not None:
            ident.active_slot = slot_id
            ident.last_seen_frame = self._current_frame
            if not getattr(self, "_mock_db", False):
                try:
                    await database.insert_match_robot_binding({
                        "job_id": self.job_id,
                        "identity_id": identity_id,
                        "slot": slot_id,
                        "start_frame": self._current_frame,
                    })
                except Exception as e:
                    print(f"Error inserting binding: {e}")

    async def _detach_expired_slots(self):
        if self.field_tracker is None:
            return
        import aiosqlite
        for event in self.field_tracker.binding_events:
            if event["event"] == "slot_expired":
                slot_id = event["slot_id"]
                # Snapshot: the await below yields the event loop, and a
                # concurrent coroutine can add/remove identities mid-iteration
                # (qm11's heavy identity churn hit "dict changed size during
                # iteration"). Iterating a list copy is stable across the await.
                for ident in list(self.identity_registry.identities.values()):
                    if ident.active_slot == slot_id:
                        ident.active_slot = None
                        if not getattr(self, "_mock_db", False):
                            try:
                                async with aiosqlite.connect(database.DB_PATH) as db:
                                    await db.execute(
                                        "UPDATE match_robot_binding SET end_frame = ? WHERE job_id = ? AND identity_id = ? AND slot = ? AND end_frame IS NULL",
                                        (self._current_frame, self.job_id, ident.identity_id, slot_id)
                                    )
                                    await db.commit()
                            except Exception as e:
                                print(f"Error updating end_frame: {e}")

    def _visible_identities(self) -> list[RobotIdentity]:
        """All active identities that are currently visible (not coasting)."""
        if self.field_tracker is None:
            return []
        visible_slots = {t.slot_id for t in self.field_tracker._emit() if t.status == "visible"}
        return [ident for ident in self.identity_registry.identities.values()
                if ident.active_slot in visible_slots]

    async def _sync_identity_registry_frame(self, fused: list, ev_list: list, track_outs: list, main_person_boxes=None, log_frame_num=None):
        self.identity_diagnostics.set_context(
            source_frame=self._current_frame,
            processed_frame=self._current_frame,
            match_sec=self._current_frame / (self.video_fps or 10.0),
        )
        # 1. Detach expired slots
        await self._detach_expired_slots()

        # 2. Gather observations
        observations = []
        slot_to_fused = {}
        slot_to_ev = {}

        # From assignment_evidence:
        for slot_id, evidence in self.field_tracker.assignment_evidence.items():
            det_idx = evidence["detection_idx"]
            if det_idx < len(fused):
                slot_to_fused[slot_id] = fused[det_idx]
                if det_idx < len(ev_list):
                    slot_to_ev[slot_id] = ev_list[det_idx]

        # From binding events:
        for event in self.field_tracker.binding_events:
            if event["event"] in ("slot_rebound", "slot_seeded"):
                slot_id = event["slot_id"]
                track_id = event["track_id"]
                fused_obs = next((o for o in fused if o.track_id == track_id), None)
                if fused_obs is not None:
                    slot_to_fused[slot_id] = fused_obs
                    for idx, o in enumerate(fused):
                        if o.track_id == track_id:
                            if idx < len(ev_list):
                                slot_to_ev[slot_id] = ev_list[idx]
                            break

        # Construct IdentityObservation
        for t in track_outs:
            slot_id = t.slot_id
            fused_obs = slot_to_fused.get(slot_id)
            ev = slot_to_ev.get(slot_id)
            
            pos = fused_obs.detection.pos if fused_obs is not None else t.pos
            alliance = t.alliance
            emb = fused_obs.detection.embedding if fused_obs is not None else None
            
            margin = ev.margin if ev is not None else 1.0
            ev_type = ev.type if ev is not None else "initial"
            
            obs = IdentityObservation(
                slot_id=slot_id,
                pos=pos,
                alliance=alliance,
                embedding=emb,
                relink_margin=margin,
                continuity_evidence=ev_type
            )
            observations.append(obs)

        # 3. Call registry update
        def alliance_side(pos):
            if self.field_model is not None:
                return self.field_model.alliance_side(pos)
            return "blue" if pos[0] < self.field_length / 2.0 else "red"

        # Record each disappearance once. Registry update clears active slots, so
        # compare against the prior frame's bindings rather than repeatedly
        # recording an identity throughout a long occlusion.
        active_slot_ids = {obs.slot_id for obs in observations}
        for ident in self.identity_registry.identities.values():
            if ident.active_slot is not None and ident.active_slot not in active_slot_ids:
                event = self.co_occlusion_tracker.record_occlusion(
                    self._current_frame, ident.identity_id, ident.alliance
                )
                self.identity_diagnostics.increment("identity_occlusions")
                if len(event.identity_ids) == 2:
                    self.identity_diagnostics.record(
                        DiagnosticEvents.OCCLUSION_GROUP_OPENED,
                        event_id=event.event_id,
                        alliance=event.alliance,
                        identity_ids=sorted(event.identity_ids),
                    )

        self.identity_registry.update_frame(
            frame_idx=self._current_frame,
            observations=observations,
            alliance_side_fn=alliance_side
        )

        # 3.5 Update live confidence signals
        from web.field.identity_confidence import (
            ConfidenceSignals,
            ConfidenceState,
            calculate_confidence_update,
            should_queue_immediate_read,
        )
        from web.embedder import cosine_similarity
        
        team_counts = {}
        for ident in self.identity_registry.identities.values():
            if ident.team_number:
                team_counts[ident.team_number] = team_counts.get(ident.team_number, 0) + 1

        for ident in self.identity_registry.identities.values():
            is_observed = (ident.active_slot is not None)
            
            # Find the observation relink_margin
            relink_margin = 1.0
            continuity_evidence = "initial"
            obs_emb = None
            if is_observed:
                # Find matching observation
                obs = next((o for o in observations if o.slot_id == ident.active_slot), None)
                if obs is not None:
                    relink_margin = obs.relink_margin
                    continuity_evidence = obs.continuity_evidence
                    obs_emb = obs.embedding

            # Check same-alliance co-occlusion re-emergence
            if is_observed:
                event = self.co_occlusion_tracker.handle_reemergence(
                    self._current_frame,
                    ident.identity_id,
                    ident.active_slot,
                    relink_margin
                )
                if event is not None:
                    self.identity_diagnostics.record(
                        DiagnosticEvents.OCCLUSION_GROUP_REEMERGED,
                        event_id=event.event_id,
                        identity_id=ident.identity_id,
                        slot_id=ident.active_slot,
                        margin=relink_margin,
                    )

            ambiguous_ids = self.co_occlusion_tracker.ambiguous_identity_ids()
            ident.is_ambiguous = ident.identity_id in ambiguous_ids
            if ident.is_ambiguous:
                ident.needs_immediate_read = True

            # Check for appearance divergence
            appearance_divergent = False
            if is_observed and obs_emb is not None and ident.appearance_embedding is not None:
                sim = cosine_similarity(obs_emb, ident.appearance_embedding)
                if sim < 0.7:
                    appearance_divergent = True

            roster_uniqueness_ok = True
            if ident.team_number:
                roster_uniqueness_ok = (team_counts.get(ident.team_number, 0) <= 1)

            signals = ConfidenceSignals(
                is_observed=is_observed,
                relink_margin=relink_margin,
                has_vlm_read=ident.has_vlm_read,
                vlm_agrees=ident.vlm_agrees,
                vlm_contradicts=ident.vlm_contradicts,
                is_confirmed=ident.is_confirmed,
                roster_uniqueness_ok=roster_uniqueness_ok
            )

            state = ConfidenceState(
                confidence=ident.confidence,
                is_confirmed=ident.is_confirmed
            )

            update = calculate_confidence_update(state, signals, self.confidence_config)

            # Log confidence changes crossing thresholds (from or to confirmed/high confidence)
            old_conf = ident.confidence
            new_conf = update.new_confidence
            ident.confidence = new_conf

            if (old_conf < 0.8 and new_conf >= 0.8) or (old_conf >= 0.4 and new_conf < 0.4):
                print(f"  Identity {ident.identity_id} confidence threshold crossed: {old_conf:.2f} -> {new_conf:.2f} (reasons: {', '.join(update.reasons)})")

            # Reset transient VLM flags
            ident.has_vlm_read = False
            ident.vlm_agrees = False
            ident.vlm_contradicts = False

            # Queue immediate reads on low confidence / bad signals — gated by
            # the escalation cooldown so a just-read identity isn't pinned at
            # priority 10 every frame its condition persists.
            if is_observed:
                current_match_time = self._current_frame / (self.video_fps or 10.0)
                allow_immediate = should_queue_immediate_read(
                    current_match_time,
                    ident.last_vlm_time,
                    ident.last_immediate_queue_time,
                    self.confidence_config.immediate_read_cooldown_sec,
                )
                queued = False
                prev_slot = self._last_bindings.get(ident.identity_id)
                # re-attachment
                if prev_slot is None and allow_immediate:
                    ident.needs_immediate_read = True
                    queued = True
                    print(f"  Immediate read queued for identity {ident.identity_id}: re-attachment to slot {ident.active_slot}")
                # low margins
                if relink_margin < 0.1 and allow_immediate:
                    ident.needs_immediate_read = True
                    queued = True
                    print(f"  Immediate read queued for identity {ident.identity_id}: low relink margin {relink_margin:.2f}")
                # appearance divergence
                if appearance_divergent and allow_immediate:
                    ident.needs_immediate_read = True
                    queued = True
                    print(f"  Immediate read queued for identity {ident.identity_id}: appearance divergent")
                # roster conflict
                if not roster_uniqueness_ok and allow_immediate:
                    ident.needs_immediate_read = True
                    queued = True
                    print(f"  Immediate read queued for identity {ident.identity_id}: roster conflict")
                if queued:
                    ident.last_immediate_queue_time = current_match_time

        # 4. Bind new slots to identities
        current_bindings = {}
        for ident in self.identity_registry.identities.values():
            if ident.active_slot is not None:
                current_bindings[ident.identity_id] = ident.active_slot

        for ident_id, slot_id in current_bindings.items():
            prev_slot = self._last_bindings.get(ident_id)
            if prev_slot != slot_id:
                await self._bind_detection_identity(slot_id, ident_id)

        self._last_bindings = current_bindings

        if log_frame_num is not None:
            try:
                for ident in self.identity_registry.identities.values():
                    slot = ident.active_slot
                    if slot is None:
                        continue
                    fobs = slot_to_fused.get(slot)
                    det = fobs.detection if fobs is not None else None
                    box = det.box if det is not None else None
                    overlap = bool(box is not None and main_person_boxes and any(
                        box_ios(tuple(box), tuple(pb)) >= PERSON_OVERLAP_IOS
                        for pb in main_person_boxes))
                    self._identity_log.append({
                        "frame_num": log_frame_num,
                        "identity_id": ident.identity_id,
                        "slot_id": slot,
                        "alliance": ident.alliance,
                        "pos": list(ident.last_seen_pos) if ident.last_seen_pos is not None else None,
                        "person_overlap": overlap,
                        "confirmed": bool(ident.is_confirmed or ident.team_number is not None),
                        # Per-frame appearance (DINOv2) of the slot's object, for the
                        # appearance-based slot-swap stitcher. In-memory only; the rerun
                        # script strips it before JSON and dumps embeddings to .npz.
                        "embedding": det.embedding if (det is not None and det.embedding is not None) else None,
                    })
            except Exception:
                pass

    def _get_display_team_number(self, robot) -> Optional[str]:
        ident = self.identity_registry.identities.get(robot.robot_id)
        if ident and ident.is_ambiguous:
            return "AMBIGUOUS"
        return robot.team_number

    def _is_robot_on_alliance_side(self, robot_x: float, alliance: str) -> bool:
        """Check if a robot is on its alliance's side of the field.
        Robots can only score when shooting from their side toward their hub.
        Robots in the center are just cycling balls, not scoring."""
        if len(self.hubs) < 2:
            return True  # Can't determine sides yet, allow it
        
        # Midpoint between the two hubs divides the field
        midpoint = (self.hubs[0].x + self.hubs[1].x) / 2
        
        # Find which hub belongs to this alliance
        hub = next((h for h in self.hubs if h.alliance == alliance), None)
        if not hub:
            return True
        
        # Robot should be on the same side as its hub (between hub and field edge)
        # With some tolerance past the hub toward center (robots approach hub to shoot)
        tolerance = abs(self.hubs[1].x - self.hubs[0].x) * 0.15  # 15% of hub-to-hub distance
        
        if hub.x < midpoint:
            # Hub is on left side — robot should be left of midpoint (+ tolerance)
            return robot_x < midpoint + tolerance
        else:
            # Hub is on right side — robot should be right of midpoint (- tolerance)
            return robot_x > midpoint - tolerance

    def attribute_score(self, alliance: str, frame_num: int, timestamp: float,
                        ball_position: Optional[Tuple[float, float]] = None) -> None:
        """Attribute an OCR-confirmed score to the best robot candidate.
        
        PRIMARY: Use ball trajectory data — trace scored balls back to their shooter.
        FALLBACK: If no trajectory data, use robot proximity to hub.
        
        Multiple robots CAN score simultaneously; each scored ball maps to a shooter.
        """
        all_robots = list(self.robot_tracks.values())
        if not all_robots:
            print(f"  WARN: No robots tracked at all for {alliance} score attribution at frame {frame_num}")
            return
        
        # PRIMARY: Check for recent scored balls with trajectory-identified shooters
        # Look for balls that scored for this alliance within recent frames
        trajectory_shooter = None
        for scored_ball in reversed(self.scored_balls):
            if (scored_ball.scored_alliance == alliance and 
                scored_ball.shooter_robot_id is not None and
                abs(scored_ball.positions[-1][0] - frame_num) <= 60):
                trajectory_shooter = scored_ball.shooter_robot_id
                # Remove this scored ball so it's not double-counted
                self.scored_balls.remove(scored_ball)
                break
        
        if trajectory_shooter is not None and trajectory_shooter in self.robot_tracks:
            robot = self.robot_tracks[trajectory_shooter]
            
            # Calculate distance to hub for confidence
            hub = next((h for h in self.hubs if h.alliance == alliance), None)
            dist = 0
            if hub:
                origin = robot.last_position
                dist = np.sqrt((origin[0] - hub.x)**2 + (origin[1] - hub.y)**2)
            
            # Defer score attribution if identity is ambiguous
            ident = self.identity_registry.identities.get(robot.robot_id)
            if ident and ident.is_ambiguous:
                slot_id = ident.active_slot if ident else None
                self.deferred_attributions.append({
                    "alliance": alliance,
                    "frame_num": frame_num,
                    "timestamp": timestamp,
                    "slot_id": slot_id,
                    "distance_to_hub": dist
                })
                print(f"  DEFERRED trajectory attribution [{alliance}]: slot {slot_id} is ambiguous")
                return

            robot.scores += 1
            confidence = "high"  # Trajectory-based attribution is high confidence
            event = ScoringEvent(
                frame_num=frame_num, timestamp=timestamp, alliance=alliance,
                robot_id=robot.robot_id, team_number=robot.team_number,
                confidence=confidence, distance_to_hub=dist
            )
            self.scoring_events.append(event)
            self._mirror_event_to_replay()
            print(f"  TRAJECTORY attribution [{alliance}]: Robot {robot.team_number or robot.robot_id} "
                  f"(trajectory-based, dist={dist:.0f})")
            return
        
        # FALLBACK: No trajectory data — use proximity to hub
        alliance_robots = [r for r in all_robots if r.alliance == alliance]
        if not alliance_robots:
            alliance_robots = all_robots  # Use any robot if none match alliance
        
        hub = next((h for h in self.hubs if h.alliance == alliance), None)
        if not hub:
            best_robot = alliance_robots[0]
            ident = self.identity_registry.identities.get(best_robot.robot_id)
            if ident and ident.is_ambiguous:
                slot_id = ident.active_slot if ident else None
                self.deferred_attributions.append({
                    "alliance": alliance,
                    "frame_num": frame_num,
                    "timestamp": timestamp,
                    "slot_id": slot_id,
                    "distance_to_hub": 999
                })
                print(f"  DEFERRED proximity fallback (no hub) [{alliance}]: slot {slot_id} is ambiguous")
                return
            best_robot.scores += 1
            self.scoring_events.append(ScoringEvent(
                frame_num=frame_num, timestamp=timestamp, alliance=alliance,
                robot_id=best_robot.robot_id,
                team_number=best_robot.team_number,
                confidence="low", distance_to_hub=999
            ))
            self._mirror_event_to_replay()
            return
        
        # Filter to robots on their alliance side
        valid_robots = [r for r in alliance_robots 
                       if self._is_robot_on_alliance_side(r.last_position[0], alliance)]
        if not valid_robots:
            valid_robots = alliance_robots
        
        # Pick closest to hub as fallback
        best_robot = min(valid_robots, 
                        key=lambda r: np.sqrt((r.last_position[0] - hub.x)**2 + 
                                             (r.last_position[1] - hub.y)**2))
        dist_to_hub = np.sqrt((best_robot.last_position[0] - hub.x)**2 + 
                              (best_robot.last_position[1] - hub.y)**2)
        
        # Defer score attribution if identity is ambiguous
        ident = self.identity_registry.identities.get(best_robot.robot_id)
        if ident and ident.is_ambiguous:
            slot_id = ident.active_slot if ident else None
            self.deferred_attributions.append({
                "alliance": alliance,
                "frame_num": frame_num,
                "timestamp": timestamp,
                "slot_id": slot_id,
                "distance_to_hub": dist_to_hub
            })
            print(f"  DEFERRED proximity fallback [{alliance}]: slot {slot_id} is ambiguous")
            return

        best_robot.scores += 1
        confidence = "low"  # Proximity-only is low confidence
        event = ScoringEvent(
            frame_num=frame_num, timestamp=timestamp, alliance=alliance,
            robot_id=best_robot.robot_id, team_number=best_robot.team_number,
            confidence=confidence, distance_to_hub=dist_to_hub
        )
        self.scoring_events.append(event)
        self._mirror_event_to_replay()
        print(f"  PROXIMITY fallback [{alliance}]: Robot {best_robot.team_number or best_robot.robot_id} "
              f"(dist={dist_to_hub:.0f})")
    
    def _adjust_robot_scores_down(self, alliance: str, target_total: int) -> None:
        """When OCR says we over-counted, adjust per-robot scores proportionally."""
        alliance_robots = [r for r in self.robot_tracks.values() if r.alliance == alliance]
        if not alliance_robots:
            return
        
        current_total = sum(r.scores for r in alliance_robots)
        if current_total <= target_total:
            return
        
        if current_total == 0:
            return
        
        # Scale each robot's scores proportionally
        for robot in alliance_robots:
            robot.scores = max(0, round(robot.scores * target_total / current_total))
        
        # Fix rounding: adjust highest scorer to match exact total
        new_total = sum(r.scores for r in alliance_robots)
        diff = target_total - new_total
        if diff != 0 and alliance_robots:
            best = max(alliance_robots, key=lambda r: r.scores)
            best.scores = max(0, best.scores + diff)

    def _open_log_file(self) -> None:
        """Open a per-match log file and register it so module prints mirror into it."""
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = LOGS_DIR / f"{ts}_{self.job_id}.log"
            self._log_file = open(self._log_path, "w", encoding="utf-8")
            _active_job_logs.append(self._log_file)
            self._log_file.write(
                "=== FRC Vision Scorer — match analysis log ===\n"
                f"job_id:       {self.job_id}\n"
                f"video:        {self.video_path}\n"
                f"match_teams:  blue={self.match_blue_teams} red={self.match_red_teams}\n"
                f"model:        {MODEL_PATH}\n"
                f"reid params:  APP_FLOOR={APP_FLOOR} POS_GATE_PX={POS_GATE_PX} "
                f"ASSIGN_MAX_COST={ASSIGN_MAX_COST} IDENTITY_INTERVAL={IDENTITY_INTERVAL}\n"
                f"started:      {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                + "=" * 60 + "\n"
            )
            self._log_file.flush()
            print(f"Logging this match to {self._log_path}")
        except Exception as e:
            self._log_file = None
            _real_print(f"  (could not open match log: {e})")

    def _close_log_file(self) -> None:
        """Write a summary footer and close/deregister the per-match log file."""
        if not self._log_file:
            return
        try:
            print("=" * 60)
            print(f"FINAL scores: {self.alliance_scores}")
            for r in sorted(self.robot_tracks.values(), key=lambda r: (r.alliance, r.robot_id)):
                print(f"  Robot {r.robot_id} [{r.alliance}] team={r.team_number} "
                      f"confirmed={r.team_number_confirmed} scores={r.scores}")
            print(f"status: {self.status}  error: {self.error}")
            print(f"finished: {datetime.datetime.now().isoformat(timespec='seconds')}")
        finally:
            try:
                _active_job_logs.remove(self._log_file)
            except ValueError:
                pass
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _init_field_tracking(self) -> bool:
        """Load calibration + field model and build the field-space tracker.

        Returns True if field tracking is ready. On any failure (no event loaded,
        missing/invalid calibration) returns False and records a clear error —
        the caller ends the job cleanly WITHOUT robot processing (decision 7).
        """
        if not self.event_key:
            self.error = ("No event loaded — load an event with a calibrated "
                          "camera before running inference.")
            print(f"  FIELD TRACKING DISABLED: {self.error}")
            return False
        versions = field_store.load_versions(self.event_key, self.camera)
        if not versions:
            self.error = (f"No calibration found for event '{self.event_key}' "
                          f"camera '{self.camera}'. Calibrate the camera first.")
            print(f"  FIELD TRACKING DISABLED: {self.error}")
            return False
        chosen = select_epoch(versions, self.match_key)
        cal = chosen["calibration"]
        self._calib_version_id = chosen["version_id"]
        if self.match_key is None:
            print(f"  [calib] no match_key — using latest version {chosen['version_id']}")
        else:
            print(f"  [calib] match {self.match_key}: epoch {chosen['version_id']} "
                  f"(start_match={chosen['start_match']})")
        if cal.camera_center is None:
            # Homography (or any pose-free) calibration has no camera center, so
            # occlusion ray-casting can't run. Require a pose-bearing solve.
            self.error = (f"Calibration for '{self.event_key}'/{self.camera} is "
                          f"mode '{cal.mode}' (no camera pose). Re-calibrate with the "
                          f"tag-anchored (pinhole) flow before running inference.")
            print(f"  FIELD TRACKING DISABLED: {self.error}")
            return False
        try:
            self.calib = cal
            self.field_model = FieldModel.load(cal.field_season)
            self.field_dims = (self.field_model.length_m, self.field_model.width_m)
            self.hub_prisms = hub_prisms(self.field_model)
            self.hub_no_robot_zones = prism_bounding_zones(self.hub_prisms)
            # One combined no-seed list for every gate: FieldTracker slot
            # seeding AND the SAM2/SAM3 prompt + re-prompt seed gates. User
            # zones are inflated to absorb corner-cam projection error; they
            # are also kept separately for the low-camera ray gate (hub zones
            # stay point-test only — robots park flush to hubs).
            # Raw (uninflated) rects kept separately for the corroboration
            # gate: blinding corroboration needs tight footprints so a robot
            # parked BESIDE the equipment keeps its detector evidence.
            self.user_seed_zones_raw = load_zones(self.event_key)
            self.user_seed_zones = inflate_zones(
                self.user_seed_zones_raw, ZONE_SEED_MARGIN_M
            )
            static_zones = self.user_seed_zones + self.hub_no_robot_zones
            self.static_seed_zones = static_zones
            if static_zones:
                print(f"  Static-structure zones (field m, user zones "
                      f"inflated {ZONE_SEED_MARGIN_M}m): {static_zones}")
            # Suspect zones for the occlusion re-acquisition guard: bounding
            # rects of every fixed structure footprint (bumps/hubs/trenches).
            # Unlike static_zones these are motion-conditioned in the tracker,
            # not hard no-seed blocks — robots legitimately cross bumps.
            suspect_zones = []
            if self.config.occlusion_rebind_guard:
                footprints = structure_footprints(self.field_model, self.hub_prisms)
                for polys in footprints.values():
                    for poly in polys:
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        suspect_zones.append(
                            (min(xs), min(ys), max(xs), max(ys)))
                print(f"  Occlusion rebind guard ON: {len(suspect_zones)} "
                      "suspect structure footprints")
            self.field_tracker = FieldTracker(
                self.field_model.length_m, zones=static_zones,
                corner_only_r_scale=self.config.corner_only_r_scale,
                coast_vel_decay=self.config.coast_vel_decay,
                sticky_max_dist_m=self.config.sticky_max_dist_m,
                suspect_zones=suspect_zones,
                occlusion_rebind_guard=self.config.occlusion_rebind_guard,
                kin_gate_enabled=self.config.kin_gate,
                kin_max_speed_mps=self.config.kin_max_speed_mps,
                record_history=self.config.rts_smooth)
            self.radar = Radar(self.field_model.length_m, self.field_model.width_m,
                               field_model=self.field_model)
            self._radar_trails = {}
            # Auto-only comparison runs overlay Manta's human-scouted auto paths on
            # the Field Map (cache-only; no network). Off unless auto_period_only AND
            # cached Manta data exist for this match.
            manta_ref = overlay_paths_for_job(self.config.auto_period_only, self.match_key)
            if manta_ref:
                self.radar.set_reference_paths(manta_ref)
                print(f"  [manta] overlaying {len(manta_ref)} scouted auto paths "
                      f"for {self.match_key}")
            elif self.config.auto_period_only and self.match_key:
                print(f"  [manta] auto-only run but no cached Manta paths for {self.match_key}")
            selection = {}
            for _name in load_layout().keys():
                _vs = field_store.load_versions(self.event_key, _name)
                _pick = select_epoch(_vs, self.match_key)
                if _pick is not None:
                    selection[_name] = _pick["version_id"]
            self.views = load_views(self.event_key, selection=selection)  # main first; corners if calibrated
            print(f"  Cameras active: {[v.name for v in self.views]}")
            # Per-camera ground position, for the vote ray-consistency gate
            # (identity_confidence.vote_ray_cross_m): a bumper crop's own
            # camera saw it on the ray camera_center -> detection, and a slot
            # position far off that ray is evidence of a wrong merge.
            # Homography-mode calibrations have no camera_center (no pose),
            # so those cameras are simply absent from the map and the gate
            # no-ops for their evidence (see call site's `center is not None`
            # guard).
            self._camera_centers_xy = {}
            for v in self.views:
                cc = v.calibration.camera_center if v.calibration else None
                if cc is not None:
                    self._camera_centers_xy[v.name] = (float(cc[0]), float(cc[1]))
        except Exception as e:
            self.error = f"Failed to initialize field tracking: {e}"
            print(f"  FIELD TRACKING DISABLED: {self.error}")
            return False
        self.field_tracking = True
        self._drift_monitor = (
            DriftMonitor(CALIB_MONITOR_INTERVAL_S, CALIB_DRIFT_MAX_M, CALIB_CHECK_MIN_TAGS)
            if (CALIB_MONITOR_ENABLED and CALIB_CHECK_ENABLED and self.calib is not None)
            else None)
        self._drift_warned = False
        print(f"  Field tracking ENABLED: event={self.event_key} camera={self.camera} "
              f"season={cal.field_season} mode={cal.mode}")
        return True

    def _mirror_event_to_replay(self) -> None:
        """Mirror the most recently appended scoring event into the replay recorder."""
        rec = getattr(self, "_replay", None)
        if rec is None or not self.scoring_events:
            return
        ev = self.scoring_events[-1]
        try:
            rec.add_event({
                "frame_num": ev.frame_num,
                "alliance": ev.alliance,
                "slot": self._slot_label(ev.robot_id),
                "team_number": ev.team_number,
                "confidence": ev.confidence,
                "robot_label": ev.team_number or self._slot_label(ev.robot_id),
            })
        except Exception:
            pass

    def _apply_track_verdicts(self) -> None:
        """Slice 2: after the whole match is processed, classify each track from the
        accumulated evidence and retroactively drop human/static-track rows from the
        recorded replay frames (refs / parked structures stop appearing as robots).
        Pruning only; does not re-bind slots (that is slice 3)."""
        from web.field.track_evidence import track_verdicts, prune_replay_frames
        rec = getattr(self, "_replay", None)
        if rec is None or not getattr(self, "_match_evidence", None):
            return
        verdicts = track_verdicts(self._match_evidence)
        rec.frames = prune_replay_frames(rec.frames, self._binding_log, verdicts)
        try:
            self.identity_diagnostics.gauge(
                "track_verdicts",
                {str(k): list(verdicts.values()).count(k)
                 for k in ("robot", "human", "static")})
        except Exception:
            pass

    def _reresolve_replay_frames(self) -> None:
        """Slice 3b: rebuild replay frames keyed by stable slot_id (identity_id churns
        — hundreds per match). Per-slot verdict excludes ref/static slots; retroactive
        team per slot. Slot-churn correction is the stitcher's job. Guarded fallback."""
        from web.field.track_evidence import slot_evidence, classify_track, reresolve_frames_by_slot
        rec = getattr(self, "_replay", None)
        if rec is None or not getattr(self, "_identity_log", None):
            return self._apply_track_verdicts()
        sev = slot_evidence(self._identity_log)
        slot_verdicts = {sid: classify_track(te) for sid, te in sev.items()}
        dropped = {}
        fps = getattr(self, "video_fps", None)
        if SEGMENT_RESOLVE_ENABLED and fps:
            # Without a real fps we can't turn frame gaps into seconds, so skip
            # per-segment resolution rather than split on bad time data.
            from web.field.track_segments import slot_dropped_ranges
            dropped = slot_dropped_ranges(self._identity_log, fps)
        assignments = self._canonical_identity_slots()        # {slot_id: identity_id}
        slot_meta = {}
        for slot_id, iid in assignments.items():
            robot = self.robot_tracks.get(iid)
            slot_meta[slot_id] = {
                "team_number": self._get_display_team_number(robot) if robot is not None else None,
                "confirmed": bool(robot and robot.team_number_confirmed),
                "alliance": self._slot_alliance(slot_id),
            }
        frame_nums = [f["frame_num"] for f in rec.frames]
        orig_rows = sum(len(f.get("robots", [])) for f in rec.frames)
        rebuilt = reresolve_frames_by_slot(self._identity_log, slot_meta, slot_verdicts, frame_nums)
        if dropped:
            # Drop phantom / wrong-robot segments (per-segment resolution) from the
            # per-slot rebuild. Rows are keyed by slot label; map back to slot_id.
            from web.field.track_evidence import slot_label_to_id
            for fr in rebuilt:
                fn = fr["frame_num"]
                fr["robots"] = [
                    r for r in fr["robots"]
                    if not any(lo <= fn <= hi
                               for (lo, hi) in dropped.get(slot_label_to_id(r["slot"]), []))
                ]
        rebuilt_rows = sum(len(f["robots"]) for f in rebuilt)
        if rebuilt_rows >= max(1, orig_rows) * RERESOLVE_MIN_KEEP_FRAC:
            orig = {f["frame_num"]: f for f in rec.frames}
            rec.frames = [{**orig.get(f["frame_num"], {}), **f} for f in rebuilt]
        else:
            print(f"[backward-confirm] re-resolution kept {rebuilt_rows}/{orig_rows} rows "
                  f"(< {RERESOLVE_MIN_KEEP_FRAC:.0%}); falling back to slice-2 prune")
            self._apply_track_verdicts()
        try:
            self.identity_diagnostics.gauge("reresolve_slot_verdicts",
                {k: list(slot_verdicts.values()).count(k) for k in ("robot", "human", "static")})
        except Exception:
            pass
        if dropped:
            # Apply the same per-segment drops to the auto-path data the Manta
            # comparison consumes, so phantom/alien segments stop corrupting it.
            from web.field.track_segments import filter_paths_by_dropped
            self.autonomous_paths = filter_paths_by_dropped(
                dict(self.autonomous_paths), dropped)

    def _verify_calibration_alignment(self) -> bool:
        """Validate self.calib against auto-detected AprilTags in sampled in-match
        frames (main view). Returns True to proceed. On drift, sets self.error +
        self.status='error' + self.needs_calibration and returns False. No tags -> warn
        + proceed. detect_tags must stay on one sequential path."""
        if not CALIB_CHECK_ENABLED or self.calib is None:
            return True
        main_view = next((v for v in self.views if v.name == "main"), None)
        if main_view is None or not self.processing_ranges:
            return True
        lo, hi = self.processing_ranges[0]
        n = max(1, CALIB_CHECK_SAMPLE_FRAMES)
        frame_idxs = [int(lo + (hi - lo) * i / (n - 1)) for i in range(n)] if n > 1 else [lo]
        best_err, best_n = float("inf"), 0
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            for fidx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ok, frame = cap.read()
                if not ok:
                    continue
                crop = main_view.extract(frame)
                detected = detect_tags(crop)
                err_m, ntags = tag_alignment_error(self.calib, detected, self.field_model)
                if ntags >= CALIB_CHECK_MIN_TAGS and err_m < best_err:
                    best_err, best_n = err_m, ntags
        finally:
            cap.release()
        if best_n < CALIB_CHECK_MIN_TAGS:
            print(f"  [calib] WARNING: could not validate calibration — <{CALIB_CHECK_MIN_TAGS} "
                  f"AprilTags detected across {n} sampled frames; proceeding unvalidated.")
            return True
        # best_err is finite here: the guard above returns when best_n < MIN_TAGS,
        # which is the only case best_err stays inf (no frame ever updated it).
        if best_err > CALIB_DRIFT_MAX_M:
            self.status = "error"
            # Before giving up, try to auto-recover the pose against the fixed lens
            # and STAGE it for approval (never auto-adopted). A staged fix flips the
            # abort message from "recalibrate manually" to "review/approve".
            autofix = None
            if (CALIB_AUTOFIX_ENABLED and self.match_key
                    and getattr(self.calib, "mode", None) in ("camera", "pinhole")):
                autofix = self._attempt_autofix(main_view, best_err)
            self.error, self.needs_calibration = self._drift_abort_state(best_err, autofix)
            if self.match_key:
                # Persist the failure so the calibrate page can flag this match
                # in its selector until a new epoch covers it (or a check passes).
                field_store.record_drift(self.event_key, self.camera, self.match_key,
                                         best_err, self._calib_version_id)
            print(f"  FIELD TRACKING ABORTED: {self.error}")
            return False
        if self.match_key:
            field_store.clear_drift(self.event_key, self.camera, self.match_key)
        print(f"  [calib] OK: {best_n} tags align to {best_err:.3f} m (epoch {self._calib_version_id})")
        return True

    def _attempt_autofix(self, main_view, best_err: float):
        """Re-solve the camera pose against the fixed lens from AprilTags across the
        match and, if it clears the guardrails, STAGE it for human approval. Returns
        the PoseFixResult (passed or not), or None if the solve raised. Never adopts
        the fix — approval happens via /api/calibration/autofix/approve."""
        cfg = PoseFixConfig(min_tags=max(MIN_POSE_TAGS, CALIB_CHECK_MIN_TAGS),
                            drift_max_m=CALIB_DRIFT_MAX_M)
        try:
            result = resolve_pose_from_video(
                str(self.video_path), main_view.crop, self.calib, self.field_model, cfg)
        except Exception as e:
            print(f"  [autofix] pose re-solve errored, falling back to manual: {e}")
            return None
        if result.passed:
            field_store.save_pending_autofix(
                self.event_key, self.camera, self.match_key,
                result.calibration, result.summary(), base_epoch=self._calib_version_id)
            print(f"  [autofix] staged a pose re-solve for approval: {result.reason}")
        else:
            print(f"  [autofix] no safe auto-fix ({result.reason}); manual recalibration needed")
        return result

    def _drift_abort_state(self, best_err: float, autofix):
        """PURE: build the (error message, needs_calibration dict) for a drift abort.
        ``autofix`` is the PoseFixResult from _attempt_autofix (or None). A passing
        fix routes the operator to review/approve; otherwise to manual recalibration."""
        calib_url = (f"/calibrate?match={self.match_key}&camera={self.camera}"
                     if self.match_key else "/calibrate")
        needs = {"event": self.event_key, "camera": self.camera,
                 "match_key": self.match_key, "error_m": round(best_err, 3),
                 "epoch": self._calib_version_id}
        head = (f"Calibration drift for '{self.camera}' on {self.match_key or 'this match'}: "
                f"AprilTags are {best_err:.2f} m off (epoch {self._calib_version_id}). ")
        if autofix is not None and autofix.passed:
            needs["autofix_pending"] = True
            needs["autofix_summary"] = autofix.summary()
            error = (head + "An automatic pose re-solve is available for review — "
                     f"approve or reject it at {calib_url}.")
        else:
            if autofix is not None:
                needs["autofix_attempted"] = True
                needs["autofix_reason"] = autofix.reason
            error = (head + f"The camera likely moved/zoomed — recalibrate at {calib_url} "
                     f"(saved as a new version applying from this match onward).")
        return error, needs

    async def process_video(self) -> None:
        """Main video processing pipeline"""
        cap = None
        # A/B provenance: knobs differing from code defaults (env or per-job).
        _nd = self.config.non_defaults()
        print(f"  [config] {_nd if _nd else 'all defaults'}")
        # Gate on calibration / loaded event (decision 7). No fallback to the old
        # pixel-space pipeline: if field tracking can't be initialized, end cleanly.
        if not self._init_field_tracking():
            self.status = "error"
            self._open_log_file()
            print(f"ABORTING JOB: {self.error}")
            self._close_log_file()
            await self._send_ws({"type": "error", "error": self.error})
            return
        try:
            self.status = "processing"
            self._open_log_file()
            # Load the per-event appearance gallery AFTER the log file is open so the
            # load line lands in the match log (not just the server console).
            if self.config.gallery_enabled and self.event_key:
                try:
                    roster_teams = list(dict.fromkeys(
                        str(team)
                        for team in (
                            self.match_teams
                            + self.match_blue_teams
                            + self.match_red_teams
                        )
                        if team is not None and str(team)
                    ))
                    rows = (
                        await database.get_previous_gallery_exemplars(
                            self.event_key,
                            self.match_key,
                            teams=roster_teams or None,
                            verified_only=self.config.gallery_verified_only,
                        )
                        if self.match_key else []
                    )
                    self.gallery = AppearanceGallery.from_rows(
                        rows, dedup_sim=GALLERY_DEDUP_SIM, max_exemplars=GALLERY_MAX_EXEMPLARS)
                    scope = (
                        f"match teams {roster_teams}"
                        if roster_teams else "all event teams (no match roster)"
                    )
                    trust = (
                        "verified-only" if self.config.gallery_verified_only
                        else "all deposits"
                    )
                    print(
                        f"  Appearance gallery loaded: {len(rows)} exemplars ({trust}) "
                        f"for {self.event_key} before {self.match_key or 'unknown match'}, {scope}"
                    )
                except Exception as e:
                    await self._warn_degraded(
                        "gallery",
                        f"Appearance gallery load failed ({e}); appearance recovery disabled",
                    )
                    self.gallery = None
            # Per-phase processing timers (seconds). `total` is filled at the end.
            self.timings = {}
            # Analysis-loop stage accumulators; folded into timings as
            # "loop.<stage>" keys so every run self-profiles its hot path.
            self._loop_stage_s = defaultdict(float)
            self._proc_t0 = time.perf_counter()

            # Open video
            cap = cv2.VideoCapture(str(self.video_path))
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.video_fps and self.video_fps > 0:
                self.frame_skip = max(1, int(round(self.video_fps / 10.0)))
            else:
                self.frame_skip = FRAME_SKIP
            print(f"Using dynamic frame_skip = {self.frame_skip} for video FPS = {self.video_fps}")
            self.overhead_height = int(self.video_height * OVERHEAD_CROP_RATIO)
            self.ocr_regions = scale_ocr_regions(self.video_width, self.overhead_height)
            print(f"Video: {self.video_width}x{self.video_height}, overhead: {self.video_width}x{self.overhead_height}")
            print(f"  OCR regions (overhead coords): {self.ocr_regions}")
            print(f"  Using {'custom' if custom_ocr_regions else 'default'} OCR regions")

            scan_t0 = time.perf_counter()
            # The clock scan is a ~minute-long synchronous decode+OCR loop. Run it
            # OFF the event loop (asyncio.to_thread) — running it inline blocks the
            # loop long enough that uvicorn can't service the WebSocket keepalive,
            # so the browser drops the connection ("connection error" in the UI)
            # even though the job keeps running. Tell the client what's happening
            # first, since no progress streams during the scan.
            await self._send_ws({
                "type": "status", "status": "processing",
                "message": "Scanning match clock…", "progress": 0.0,
            })
            clock_samples = await asyncio.to_thread(self._scan_match_clock)
            (
                self.processing_ranges,
                self.match_periods,
                self.timer_gate_fallback,
                self.timer_gate_fallback_reason,
            ) = compute_active_windows(
                clock_samples, self.video_fps, self.total_frames,
                auto_only=self.config.auto_period_only,
            )
            self.timings["clock_scan"] = round(time.perf_counter() - scan_t0, 2)
            self._raise_if_stopped()  # phase boundary: after clock scan
            if self.timer_gate_fallback:
                fallback_start = min(TITLE_SCREEN_FRAMES, max(0, self.total_frames - 1))
                self.processing_ranges = [
                    (fallback_start, max(0, self.total_frames - 1))
                ]
                if self.config.auto_period_only:
                    # Auto-only was requested but we never locked the auto
                    # countdown, so we can't isolate it. Surface this loudly —
                    # the run will cover the whole match and a silent full-match
                    # pass would quietly invalidate any Manta auto comparison.
                    await self._warn_degraded(
                        "auto_period_only",
                        "auto-period-only requested but the autonomous clock was "
                        f"not found ({self.timer_gate_fallback_reason}); processing "
                        "the full match instead.",
                    )
                print(
                    "  [timer] WARNING: using the legacy post-title-screen range: "
                    f"{self.timer_gate_fallback_reason}"
                )
            elif self.config.auto_period_only:
                print(
                    "  [timer] AUTO-PERIOD-ONLY: restricting processing to the "
                    f"autonomous window {self.processing_ranges}"
                )
            else:
                period_text = ", ".join(
                    f"{p['label']}={p['start_frame']}-{p['end_frame']}"
                    for p in self.match_periods
                )
                print(
                    f"  [timer] periods: {period_text}; "
                    f"processing ranges={self.processing_ranges}"
                )
            
            # Extract team numbers from early frame (using overhead crop)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
            ret, early_frame = cap.read()
            if ret:
                early_overhead = early_frame[:self.overhead_height, :]
                self.extract_team_numbers(early_overhead)

            # Run off the event loop (like the clock scan above): the seek+decode+
            # detect_tags work is synchronous and would otherwise block uvicorn's
            # WebSocket keepalive while it runs.
            if not await asyncio.to_thread(self._verify_calibration_alignment):
                # _verify_calibration_alignment set self.status="error", self.error,
                # self.needs_calibration. Notify the client and stop before the SAM2 pass
                # (mirrors the _init_field_tracking abort at the top of process_video).
                if self.websocket:
                    try:
                        await self.websocket.send_json({"type": "error", "error": self.error,
                                                        "needs_calibration": self.needs_calibration})
                    except Exception:
                        pass
                return
            self._raise_if_stopped()  # phase boundary: after calibration check
            # ----- SAM2 PRE-COMPUTATION PASS -----
            self.use_sam2 = False
            self.sam2_raw = {}
            self._view_mine_info = {}
            self.sam2_obj_color = {}
            self._replay = ReplayRecorder()
            self.robot_embedder = None
            
            active_frame_count = sum(
                end - start + 1 for start, end in self.processing_ranges
            )
            auto_period = next(
                (p for p in self.match_periods if p["label"] == "auto"), None
            )
            prompt_source_frame = (
                auto_period["start_frame"]
                if auto_period is not None and not self.timer_gate_fallback
                else EARLY_OCR_FRAME
            )

            if active_frame_count <= 1 or prompt_source_frame >= self.total_frames:
                print(
                    f"  [{self.tracker_tag}] Active video is too short ({active_frame_count} frames) "
                    f"for prompt frame {prompt_source_frame}. Disabling SAM2."
                )
            else:
                try:
                    import torch
                    
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    backend_label = "SAM3.1" if self.tracking_backend == "sam3" else "SAM2"

                    # Phase exclusivity: a resident vLLM server (~11GB pool)
                    # plus SAM3 propagation (~11GB peak) OOMs the 24GB card —
                    # and vLLM is idle through this whole phase. Take it down
                    # BEFORE the free-VRAM probe below so the offload
                    # heuristic sees the real budget; the server comes back
                    # right before the analysis loop's bumper reads.
                    if self.config.vllm_phase_exclusive and device == "cuda":
                        from web import vllm_lifecycle
                        await self._send_ws({
                            "type": "status", "status": "processing",
                            "message": "Pausing vLLM reader to free VRAM for tracking…",
                            "progress": 8.0,
                        })
                        # The in-process local reader squats ~8GB the same way
                        # the server does; evict it so the offload probe below
                        # sees the real budget. It re-loads lazily at the
                        # analysis loop's first read pass.
                        if _unload_qwen_vl():
                            print(f"  [{backend_label}] Unloaded in-process bumper reader to free VRAM for tracking")
                        if not await vllm_lifecycle.ensure_stopped():
                            await self._warn_degraded(
                                "vllm",
                                "Could not stop the vLLM server before tracking; "
                                "propagation may run out of GPU memory.",
                            )
                        else:
                            await asyncio.sleep(5)  # let CUDA reclaim the pool

                    offload_vid = True
                    offload_st = True
                    # Target FPS is a fixed speed knob (SAM2_TRACK_FPS), independent of
                    # VRAM.  A per-job override (self.target_tracking_fps) still wins.
                    # VRAM only decides whether tracker state can stay on the GPU (small
                    # latency win); it no longer inflates the frame rate.
                    target_tracking_fps = (
                        self.target_tracking_fps
                        if self.target_tracking_fps is not None
                        else self.config.sam2_track_fps
                    )
                    if device == "cuda":
                        try:
                            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(0)
                            free_vram_gb = free_mem_bytes / (1024 ** 3)
                            total_vram_gb = total_mem_bytes / (1024 ** 3)
                            print(f"  [{self.tracker_tag}] GPU VRAM: {free_vram_gb:.2f} GB free of {total_vram_gb:.2f} GB total")
                            if free_vram_gb >= 16.0:
                                offload_st = False
                                print(f"  [{self.tracker_tag}] Free VRAM >= 16GB: keeping tracker state on GPU.")
                            else:
                                print(f"  [{self.tracker_tag}] Free VRAM < 16GB: offloading tracker state to CPU.")
                        except Exception as e:
                            print(f"  [{self.tracker_tag}] Error querying GPU VRAM info: {e}. Using conservative defaults.")
                    
                    # Update dynamic frame_skip calculation based on adaptive target tracking FPS
                    if self.video_fps and self.video_fps > 0:
                        self.frame_skip = max(1, int(round(self.video_fps / target_tracking_fps)))
                    else:
                        self.frame_skip = FRAME_SKIP
                    print(f"  [{self.tracker_tag}] Recalculating dynamic frame_skip = {self.frame_skip} for target FPS = {target_tracking_fps}")

                    # VRAM-fit override: keep all tensors on the GPU and raise
                    # frame_skip so the tracked segment fits within the budget.
                    if self.sam2_max_gpu_frames > 0 and active_frame_count:
                        source_frames = max(1, active_frame_count)
                        self.frame_skip = max(
                            self.frame_skip,
                            math.ceil(source_frames / self.sam2_max_gpu_frames),
                        )
                        seg_frames = math.ceil(source_frames / self.frame_skip)
                        offload_vid = False
                        offload_st = False
                        est_gib = seg_frames * SAM2_FRAME_GPU_MIB / 1024.0
                        print(f"  [{self.tracker_tag}] VRAM-fit: budget={self.sam2_max_gpu_frames} frames -> "
                              f"frame_skip={self.frame_skip}, ~{seg_frames} tracked frames "
                              f"(~{est_gib:.1f} GiB video on GPU, no CPU offload)")

                    _t_load = time.perf_counter()

                    def _load_tracker_and_embedder():
                        # Runs on a worker thread so the frame-export decode
                        # pass overlaps the weight load. Nothing here is
                        # shared with the export path (it needs YOLO and
                        # calibration only); the barrier after the export
                        # awaits this task before seeding uses the tracker.
                        if self.tracking_backend == "sam3":
                            from web.field.sam3_tracker import Sam3FieldTracker

                            tracker = Sam3FieldTracker(
                                checkpoint=SAM3_CHECKPOINT,
                                device=device,
                                offload_video_to_cpu=offload_vid,
                                offload_state_to_cpu=offload_st,
                                compile_model=(self.config.sam3_compile and device == "cuda"),
                                max_objects=max(MAX_TRACKED_ROBOTS * 3, 20),
                                use_fa3=SAM3_USE_FA3,
                                chunk_frames=self.config.sam3_chunk_frames,
                            )
                            detail = (
                                f"video model w/ instance prompts, "
                                f"chunk={self.config.sam3_chunk_frames} frames, "
                                f"compile={self.config.sam3_compile}"
                            )
                        else:
                            from web.field.sam2_tracker import Sam2FieldTracker

                            # Per-job encoder size (UI override); falls back to the env default.
                            job_config, job_checkpoint = _resolve_sam2_model(self.model_size)
                            tracker = Sam2FieldTracker(
                                config=job_config,
                                checkpoint=job_checkpoint,
                                device=device,
                                offload_video_to_cpu=offload_vid,
                                offload_state_to_cpu=offload_st,
                                vos_optimized=(self.config.sam2_vos_optimized and device == "cuda"),
                            )
                            detail = (
                                f"{self.model_size} model, "
                                f"vos_optimized={self.config.sam2_vos_optimized and device == 'cuda'}"
                            )
                        embedder = RobotEmbedder()
                        self.timings["model_load"] = self.timings.get("model_load", 0.0) + (time.perf_counter() - _t_load)
                        return tracker, detail, embedder

                    model_load_task = asyncio.create_task(
                        asyncio.to_thread(_load_tracker_and_embedder))
                    # One tick hands the load to its worker thread before the
                    # export loop below starts decoding.
                    await asyncio.sleep(0)
                    # Optimistic: the plan/export phase needs no tracker; the
                    # barrier after the export downgrades on load failure.
                    self.use_sam2 = True
                except Exception as e:
                    backend_label = (
                        "SAM3.1" if self.tracking_backend == "sam3" else "SAM2"
                    )
                    await self._warn_degraded(
                        self.tracker_tag.lower(),
                        f"[{backend_label}] Failed to initialize tracker/embedder: "
                        f"{e}. Falling back to Kalman tracking.",
                    )
                    self.use_sam2 = False
                    self.robot_embedder = None

            if self.use_sam2:
                print(f"  [{self.tracker_tag}] Running pre-computation pass over {len(self.views)} views...")
                all_frames = [
                    frame
                    for start, end in self.processing_ranges
                    for frame in range(start, end + 1, self.frame_skip)
                ]
                self.sam2_source_to_local = {
                    source_frame: local_idx
                    for local_idx, source_frame in enumerate(all_frames)
                }
                prompt_target_frame = prompt_source_frame + int(
                    round(self.config.sam2_prompt_offset_seconds * max(self.video_fps, 0.0))
                )
                prompt_local_idx = min(
                    range(len(all_frames)),
                    key=lambda i: abs(all_frames[i] - prompt_target_frame),
                )
                prompt_frame = all_frames[prompt_local_idx]
                print(
                    f"  [{self.tracker_tag}] Seeding at source frame {prompt_frame} "
                    f"(~{self.config.sam2_prompt_offset_seconds:.1f}s after active start), "
                    "then propagating backward."
                )
                SAM2_SCALE = self.sam2_export_scale
                print(f"  [{self.tracker_tag}] Frame-export scale: {SAM2_SCALE}")

                # Corner-cam decimation: corner views are tracked at
                # track_fps/stride (less signal there, and propagation cost
                # scales with frames/view). Each view gets its own
                # local-index space, phase-aligned so the prompt frame is
                # always kept; main always runs at full rate.
                corner_stride = max(1, int(self.config.corner_track_stride))
                # Dense head: full-rate corners through the lock-formation
                # window (kept-frame units from the active start), strided
                # afterwards. The corner-stride A/B showed whole-timeline
                # decimation starves team confirmation.
                corner_dense_frames = 0
                if corner_stride > 1 and self.video_fps and self.frame_skip:
                    corner_dense_frames = int(round(
                        self.config.corner_dense_seconds
                        * self.video_fps / self.frame_skip))
                view_plans = {
                    v.name: build_view_frame_plan(
                        all_frames,
                        prompt_local_idx,
                        1 if v.name == "main" else corner_stride,
                        dense_frames=0 if v.name == "main" else corner_dense_frames,
                    )
                    for v in self.views
                }
                self.sam2_view_source_to_local = {
                    name: plan.source_to_local
                    for name, plan in view_plans.items()
                }
                if corner_stride > 1:
                    for v in self.views:
                        if v.name != "main":
                            print(
                                f"  [{self.tracker_tag}] View {v.name}: corner stride "
                                f"{corner_stride} (dense head {corner_dense_frames}) -> "
                                f"{len(view_plans[v.name].source_frames)}"
                                f"/{len(all_frames)} frames"
                            )

                N = len(self.views)
                loop = asyncio.get_running_loop()

                async def safe_send_status(msg: str, pct: float):
                    # Latest-wins queue: progress updates never wait on the client.
                    self._enqueue_ws("status", {
                        "type": "status",
                        "status": "processing",
                        "message": msg,
                        "progress": round(pct, 1)
                    })

                # ----- Phase 1: ONE sequential decode pass exports every view's frames.
                # Previously each view re-opened the video and random-seeked to every kept
                # frame (cap.set(POS_FRAMES, gf) per frame, repeated N times) — pathological
                # on H.264, where an arbitrary seek re-decodes from the nearest prior
                # keyframe.  Here we seek once to the first kept frame, then read forward
                # sequentially and fan each kept frame out to all views.  The kept frames
                # are exactly TITLE_SCREEN_FRAMES + k*frame_skip, byte-identical to before,
                # but decoding is linear instead of seek-thrashing and happens only once.
                view_dirs = {v.name: tempfile.TemporaryDirectory() for v in self.views}
                # Registered on the job so the process_video `finally` can sweep
                # any dirs a mid-phase JobStopped/TrackStopped left behind (the
                # per-view loop only cleans the view it just propagated).
                self._view_export_dirs = view_dirs
                view_prompt_boxes: dict[str, list] = {v.name: [] for v in self.views}
                view_prompt_colors: dict[str, list] = {v.name: [] for v in self.views}
                # Per-view grayscale of the previous kept frame, for the seed
                # frame's motion probe (motion-gated low-conf admission).
                view_prev_gray: dict[str, np.ndarray] = {}
                EXPORT_PCT = 25.0  # share of the progress bar spent exporting frames
                export_t0 = time.perf_counter()
                exported = 0
                cap_pre = cv2.VideoCapture(str(self.video_path))
                try:
                    first_export_frame = all_frames[0]
                    export_frame_set = set(all_frames)
                    cap_pre.set(cv2.CAP_PROP_POS_FRAMES, first_export_frame)
                    fpos = first_export_frame
                    local_idx = 0
                    last_export_frame = all_frames[-1]
                    while fpos <= last_export_frame:
                        ok, fr = cap_pre.read()
                        if not ok:
                            break
                        if fpos in export_frame_set:
                            is_prompt = (fpos == prompt_frame)
                            for view in self.views:
                                # Decimated views skip frames outside their
                                # plan (prompt frame is always in-plan).
                                view_local_idx = view_plans[view.name].source_to_local.get(fpos)
                                if view_local_idx is None:
                                    continue
                                try:
                                    crop = view.extract(fr)
                                    # Exclude the airspace at the top of the view from robot
                                    # tracking (robots can't be there) — a false-positive
                                    # filter.  NOTE: this does NOT save SAM2 VRAM/compute:
                                    # SAM2 hard-resizes every frame to a fixed image_size
                                    # square, so fewer input rows
                                    top_crop_frac = (getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0) if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                                    top_crop_px = int(crop.shape[0] * top_crop_frac) if top_crop_frac > 0 else 0
                                    if top_crop_px > 0:
                                        crop = crop[top_crop_px:, :]
                                    new_w = int(crop.shape[1] * SAM2_SCALE)
                                    new_h = int(crop.shape[0] * SAM2_SCALE)
                                    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                    fname = os.path.join(view_dirs[view.name].name, f"{view_local_idx:05d}.jpg")
                                    cv2.imwrite(fname, small, [cv2.IMWRITE_JPEG_QUALITY, 95])
 
                                    # Motion evidence for the seed frame: diff
                                    # vs this view's previous kept frame (low-
                                    # conf gate + detector-independent blobs).
                                    seed_motion_grays = None
                                    if (self.config.motion_gated_conf
                                            or self.config.motion_seed_blobs):
                                        if is_prompt:
                                            seed_motion_grays = (
                                                view_prev_gray.get(view.name),
                                                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                                        elif fpos < prompt_frame:
                                            view_prev_gray[view.name] = cv2.cvtColor(
                                                crop, cv2.COLOR_BGR2GRAY)

                                    # Seed SAM2 from expanded bumper detections so masks
                                    # cover whole robots while color stays class-derived.
                                    if is_prompt:
                                        merged = self._sam2_robot_candidates(
                                            crop, motion_grays=seed_motion_grays,
                                            blobs_allowed=blob_window_open(
                                                max(0.0,
                                                    (prompt_frame - prompt_source_frame)
                                                    / max(self.video_fps, 1e-6)),
                                                self.config.motion_blob_window_s),
                                            zone_probe=self._make_start_zone_probe(
                                                view, top_crop_px, 0.0))
                                        full_boxes = [
                                            (
                                                box[0],
                                                box[1] + top_crop_px,
                                                box[2],
                                                box[3] + top_crop_px,
                                            )
                                            for box, _color, _strong in merged
                                        ]
                                        keep_merged = filter_field_robot_boxes(
                                            full_boxes,
                                            view.calibration,
                                            self.field_dims,
                                            frame_shape=(
                                                crop.shape[0] + top_crop_px,
                                                crop.shape[1],
                                                crop.shape[2],
                                            ),
                                            blocked_zones=self.static_seed_zones,
                                            ray_blocked_zones=self._ray_zone_list(view),
                                            frac_backstop=self.config.det_frac_backstop,
                                        )
                                        merged = [merged[idx] for idx in keep_merged]

                                        # YOLO priority: strong colored
                                        # candidates seed directly; strong
                                        # colorless fills, then motion-tier
                                        # hints, share the per-half quota IN
                                        # THAT ORDER — a motion hint only
                                        # gets a seat YOLO left empty.
                                        bumper_halves = []
                                        pool_halves = []
                                        pool_entries = []
                                        for box, color, strong in merged:
                                            x1, y1, x2, y2 = box
                                            full_box = (x1, y1 + top_crop_px, x2, y2 + top_crop_px)
                                            uv = ((full_box[0] + full_box[2]) / 2.0, full_box[3])
                                            xy = view.calibration.image_to_field(uv)
                                            half = None
                                            if xy is not None and self.field_model is not None:
                                                half = self.field_model.alliance_side(xy)

                                            if strong and color != "unknown":
                                                if half is not None:
                                                    bumper_halves.append(half)
                                            else:
                                                pool_halves.append(half)
                                                pool_entries.append((box, color))

                                        keep_indices = select_gap_fills(bumper_halves, pool_halves, max_per_half=3)

                                        for box, color, strong in merged:
                                            if strong and color != "unknown":
                                                view_prompt_boxes[view.name].append(
                                                    tuple(v * SAM2_SCALE for v in box)
                                                )
                                                view_prompt_colors[view.name].append(color)

                                        for idx in keep_indices:
                                            box, color = pool_entries[idx]
                                            view_prompt_boxes[view.name].append(
                                                tuple(v * SAM2_SCALE for v in box)
                                            )
                                            view_prompt_colors[view.name].append(color)
                                except Exception as e:
                                    print(f"  [{self.tracker_tag}] Error exporting frame {local_idx} for view {view.name}: {e}")
                            local_idx += 1
                            exported += 1
                            if exported % 20 == 0:
                                pct = (exported / max(1, len(all_frames))) * EXPORT_PCT
                                msg = f"[{self.tracker_tag}] Exporting frames ({exported}/{len(all_frames)}) across {N} view(s)..."
                                await safe_send_status(msg, pct)
                                # Per-batch stop poll: abort export within ~20
                                # frames of a STOP instead of exporting the whole
                                # active window first. The `finally` below still
                                # releases cap_pre; JobStopped unwinds past it.
                                if self._stopped:
                                    raise JobStopped()
                        fpos += 1
                except JobStopped:
                    raise  # user stop: unwind to the process_video handler
                except Exception as e:
                    print(f"  [{self.tracker_tag}] Error during single-pass frame export: {e}")
                finally:
                    cap_pre.release()
                self.timings["frame_export"] = self.timings.get("frame_export", 0.0) + (time.perf_counter() - export_t0)
                print(f"  [{self.tracker_tag}] Exported {exported} frames/view across {N} view(s) in "
                      f"{time.perf_counter() - export_t0:.1f}s (single sequential decode pass)")

                self._raise_if_stopped()  # phase boundary: after frame export

                # Barrier: the overlapped tracker/embedder load must be
                # resident before seeding. On failure, degrade to Kalman
                # exactly as the old inline load did.
                try:
                    sam2_tracker, model_detail, self.robot_embedder = await model_load_task
                    print(f"  [{backend_label}] Loaded field tracker ({model_detail}) "
                          f"and RobotEmbedder on {device}")
                except Exception as e:
                    await self._warn_degraded(
                        self.tracker_tag.lower(),
                        f"[{backend_label}] Failed to initialize tracker/embedder: "
                        f"{e}. Falling back to Kalman tracking.",
                    )
                    self.use_sam2 = False
                    self.robot_embedder = None
            if self.use_sam2:
                self._raise_if_stopped()  # phase boundary: after model load
                # ----- Phase 2: per-view SAM2 propagation over the exported frames.
                for view_idx, view in enumerate(self.views):
                    tmp_dir_obj = view_dirs[view.name]
                    view_plan = view_plans[view.name]
                    # Effective average stride over the (possibly mixed-rate)
                    # timeline: 1.0 for main, the full stride for a pure
                    # decimated view, in between with a dense head. Used to
                    # rescale wall-clock-denominated tracker knobs.
                    view_stride = len(all_frames) / max(1, len(view_plan.source_frames))
                    prompt_boxes_scaled = view_prompt_boxes[view.name]
                    self.sam2_obj_color[view.name] = {
                        i + 1: color
                        for i, color in enumerate(view_prompt_colors[view.name])
                    }
                    base_pct = EXPORT_PCT + (view_idx / N) * (100.0 - EXPORT_PCT)
                    span_pct = (100.0 - EXPORT_PCT) / N
                    print(f"  [{self.tracker_tag}] View {view.name}: prompt frame detected {len(prompt_boxes_scaled)} robots")

                    # Tracking-phase visualization: send this view's seed frame
                    # (annotated with the prompt boxes) so the UI has something
                    # to show in the video area while propagation grinds.
                    try:
                        seed_path = os.path.join(
                            tmp_dir_obj.name, f"{view_plan.prompt_idx:05d}.jpg")
                        seed_img = cv2.imread(seed_path)
                        if seed_img is not None:
                            from web.hud_util import encode_seed_thumbnail
                            await self._send_ws({
                                "type": "tracking_view_seed",
                                "view": view.name,
                                "robots": len(prompt_boxes_scaled),
                                "image": encode_seed_thumbnail(
                                    seed_img, prompt_boxes_scaled),
                            })
                    except Exception as e:
                        print(f"  [{self.tracker_tag}] seed thumbnail for {view.name} failed: {e}")
                    if not prompt_boxes_scaled:
                        print(
                            f"  [{self.tracker_tag}] View {view.name}: no prompt-frame robots; "
                            "scanning later frames for detector bootstrap."
                        )
                        await safe_send_status(
                            f"[{self.tracker_tag}] View '{view.name}': waiting for first robot...",
                            base_pct,
                        )

                    _detect_caps: list = []  # detect_fn VideoCaptures, released in finally
                    with self.identity_diagnostics.scoped_context(view=view.name):
                        try:
                            def make_progress_cb(b_pct, sp_pct, v_name):
                                throttle = {"last": 0.0}

                                def progress_cb(current, total):
                                    pct = b_pct + (current / max(1, total)) * sp_pct
                                    msg = f"[{self.tracker_tag}] View '{v_name}': tracking frame {current}/{total}..."
                                    loop.call_soon_threadsafe(
                                        lambda: asyncio.create_task(safe_send_status(msg, pct))
                                    )
                                    # Structured per-view progress for the tracking
                                    # panel, throttled so a fast view can't flood
                                    # the socket.
                                    now_s = time.monotonic()
                                    if now_s - throttle["last"] >= 0.5 or current >= total:
                                        throttle["last"] = now_s
                                        payload = {
                                            "type": "tracking_progress",
                                            "view": v_name,
                                            "current": current,
                                            "total": total,
                                            "pct": round(pct, 1),
                                        }
                                        loop.call_soon_threadsafe(
                                            lambda p=payload: asyncio.create_task(
                                                self._send_ws(p))
                                        )
                                return progress_cb

                            # Periodic re-prompting: re-run the detector cascade at chunk
                            # boundaries so robots that start occluded (e.g. behind the
                            # hub) get picked up once they emerge.  Detection runs on the
                            # FULL-RESOLUTION source frame (decoded on demand — one seek
                            # per ~2s of footage), not the 0.5x JPEG export SAM2 tracks:
                            # the re-acquisition path exists for the hard cases, so it
                            # must not have the worst recall of the three detect paths.
                            # Returned boxes are scaled into SAM2 (0.5x, top-cropped)
                            # coords to match the masks.
                            def make_detect_fn(detect_caps):
                                state = {"cap": None, "pos": None}
                                detect_caps.append(state)

                                def detect_fn(local_idx, frame_path):
                                    # local_idx is in THIS view's (possibly
                                    # decimated) index space.
                                    if local_idx is None or local_idx >= len(view_plan.source_frames):
                                        return None
                                    source_frame = view_plan.source_frames[local_idx]
                                    if state["cap"] is None:
                                        state["cap"] = cv2.VideoCapture(str(self.video_path))
                                    cap_l = state["cap"]
                                    if state["pos"] != source_frame:
                                        cap_l.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                                    ok, fr = cap_l.read()
                                    state["pos"] = source_frame + 1
                                    if not ok or fr is None:
                                        return None
                                    crop = view.extract(fr)
                                    top_crop_frac = (
                                        getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0
                                    ) if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                                    top_crop_px = (
                                        int(crop.shape[0] * top_crop_frac)
                                        if top_crop_frac > 0 else 0
                                    )
                                    if top_crop_px:
                                        crop = crop[top_crop_px:, :]
                                    # Motion evidence: diff against the previous
                                    # kept frame in this view's plan (one extra
                                    # decode per boundary; the YOLO pass beside
                                    # it dwarfs the cost).
                                    motion_grays = None
                                    if ((self.config.motion_gated_conf
                                            or self.config.motion_seed_blobs)
                                            and local_idx > 0):
                                        prev_sf = view_plan.source_frames[local_idx - 1]
                                        cap_l.set(cv2.CAP_PROP_POS_FRAMES, prev_sf)
                                        ok_p, fr_p = cap_l.read()
                                        state["pos"] = prev_sf + 1
                                        if ok_p and fr_p is not None:
                                            pcrop = view.extract(fr_p)
                                            if top_crop_px:
                                                pcrop = pcrop[top_crop_px:, :]
                                            motion_grays = (
                                                cv2.cvtColor(pcrop, cv2.COLOR_BGR2GRAY),
                                                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                                    elapsed = max(
                                        0.0,
                                        (source_frame - prompt_source_frame)
                                        / max(self.video_fps, 1e-6))
                                    merged = self._sam2_robot_candidates(
                                        crop, motion_grays=motion_grays,
                                        blobs_allowed=blob_window_open(
                                            elapsed,
                                            self.config.motion_blob_window_s),
                                        zone_probe=self._make_start_zone_probe(
                                            view, top_crop_px, elapsed),
                                        corrob_floor=self.config.corrob_conf_floor,
                                        with_corrob_band=True)
                                    merged, corrob_band = merged
                                    # Field-filter in full-view coords (restore the
                                    # top-crop offset). No blocked_zones: detect_fn
                                    # output also corroborates existing masks (a
                                    # zone-parked robot must stay detector-present);
                                    # seeding inside zones is blocked by seed_gate_fn.
                                    full_boxes = [
                                        (b[0], b[1] + top_crop_px, b[2], b[3] + top_crop_px)
                                        for b, _c, _strong in merged
                                    ]
                                    keep = filter_field_robot_boxes(
                                        full_boxes,
                                        view.calibration,
                                        self.field_dims,
                                        frame_shape=(
                                            crop.shape[0] + top_crop_px,
                                            crop.shape[1],
                                            3,
                                        ),
                                        frac_backstop=self.config.det_frac_backstop,
                                    )
                                    # YOLO priority at boundaries too: motion
                                    # hints share the per-half quota (3 = slots
                                    # per alliance) AFTER every strong candidate
                                    # is counted — the seed-frame rule, applied
                                    # at every re-prompt. Tier order is also
                                    # preserved (strong first), so the tracker
                                    # hands roster seats to YOLO evidence first.
                                    strong_halves = []
                                    motion_entries = []   # (candidate_idx, half)
                                    for idx in keep:
                                        box, _color, strong = merged[idx]
                                        fb = full_boxes[idx]
                                        uv = ((fb[0] + fb[2]) / 2.0, fb[3])
                                        xy = view.calibration.image_to_field(uv)
                                        half = None
                                        if xy is not None and self.field_model is not None:
                                            half = self.field_model.alliance_side(xy)
                                        if strong:
                                            if half is not None:
                                                strong_halves.append(half)
                                        else:
                                            motion_entries.append((idx, half))
                                    kept_motion = set(
                                        motion_entries[i][0]
                                        for i in select_gap_fills(
                                            strong_halves,
                                            [h for _i, h in motion_entries],
                                            max_per_half=3))
                                    # Corroborate-only band: same field filter as
                                    # the main cascade, but these boxes never
                                    # compete for a roster seat — they ride as
                                    # corrob_only 3-tuples so the tracker counts
                                    # them as presence evidence without seeding,
                                    # matching, re-anchoring, or evicting on them.
                                    corrob_full = [
                                        (b[0], b[1] + top_crop_px, b[2], b[3] + top_crop_px)
                                        for b, _c in corrob_band
                                    ]
                                    corrob_keep = filter_field_robot_boxes(
                                        corrob_full, view.calibration, self.field_dims,
                                        frame_shape=(crop.shape[0] + top_crop_px,
                                                     crop.shape[1], 3),
                                        frac_backstop=self.config.det_frac_backstop,
                                    ) if corrob_full else []
                                    out = [
                                        (tuple(v * SAM2_SCALE for v in merged[idx][0]),
                                         merged[idx][1])
                                        for idx in keep
                                        if merged[idx][2] or idx in kept_motion
                                    ]
                                    out += [
                                        (tuple(v * SAM2_SCALE for v in corrob_band[i][0]),
                                         corrob_band[i][1], True)
                                        for i in corrob_keep
                                    ]
                                    try:
                                        self.detection_archive.record(
                                            view.name, local_idx, source_frame,
                                            [(full_boxes[idx], merged[idx][1], False)
                                             for idx in keep
                                             if merged[idx][2] or idx in kept_motion]
                                            + [(corrob_full[i], corrob_band[i][1], True)
                                               for i in corrob_keep])
                                    except Exception:
                                        pass  # telemetry must never break detection
                                    return out
                                return detect_fn

                            # Re-detect roughly every REPROMPT_SECONDS of tracked footage.
                            # Decimated views see 1/stride the frames, so divide to
                            # keep the same wall-clock cadence.
                            eff_fps = (self.video_fps / self.frame_skip) if (self.video_fps and self.frame_skip) else 10.0
                            eff_fps /= view_stride
                            reprompt_every = max(1, int(round(self.config.reprompt_seconds * eff_fps)))
                            # qm12 opening-window dense reprompt (same eff_fps local-frame
                            # space as reprompt_every). Both 0 -> feature off.
                            opening_reprompt_every = (max(1, int(round(self.config.opening_reprompt_seconds * eff_fps)))
                                                      if self.config.opening_reprompt_seconds > 0 else 0)
                            opening_window_frames = int(round(self.config.opening_window_seconds * eff_fps))
                            # Late-seed backward backfill (A/B knob): same
                            # seconds->frames conversion as reprompt_every,
                            # which already divides by view_stride above —
                            # no separate scale_frame_knob call needed here.
                            # backfill_min_lived_frames has no knob of its
                            # own; rescale the tracker's frame-denominated
                            # default the same way evict_dark_frames is below.
                            backfill_max_frames = max(1, int(round(
                                self.config.backfill_seconds * eff_fps)))
                            from web.field.sam2_tracker import BACKFILL_MIN_LIVED_FRAMES
                            backfill_min_lived_frames = scale_frame_knob(
                                BACKFILL_MIN_LIVED_FRAMES, view_stride)

                            # Build a projector mapping a SAM2 mask box (tracker/0.5x
                            # top-cropped pixel coords) to field meters, for the
                            # mask-hijack cull's stationarity test. Exported frames are
                            # ALREADY top-cropped, so we recover the CROPPED full-res
                            # height and reconstruct the consumer's offset
                            # (original_h * tc) via cropped_full_h * tc / (1 - tc).
                            _sam2_scale = self.sam2_export_scale
                            _top_crop = (
                                (getattr(view.calibration, "robot_track_top_crop", 0.0) or 0.0)
                                if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                            )
                            _exported = sorted(
                                f for f in os.listdir(tmp_dir_obj.name)
                                if f.endswith(".jpg") or f.endswith(".jpeg")
                            )
                            _cropped_full_h = None
                            if _exported:
                                _img0 = cv2.imread(os.path.join(tmp_dir_obj.name, _exported[0]))
                                if _img0 is not None:
                                    _cropped_full_h = int(round(_img0.shape[0] / _sam2_scale))
                            _y_off = (
                                int(_cropped_full_h * _top_crop / max(1.0 - _top_crop, 1e-6))
                                if (_cropped_full_h is not None and _top_crop > 0) else 0
                            )

                            def _make_project_fn(cal, y_off, scale):
                                def _project(box):
                                    x1, y1, x2, y2 = box
                                    fx1, fx2 = x1 / scale, x2 / scale
                                    fy2 = y2 / scale + y_off
                                    return cal.image_to_field(((fx1 + fx2) / 2.0, fy2))
                                return _project

                            _project_fn = _make_project_fn(view.calibration, _y_off, _sam2_scale)

                            # Hard-negative mining (post-job, best-effort): capture
                            # the pieces mine_hard_negatives needs to map a cull
                            # event's SAM2-space box back to a full-res frame —
                            # frame_idx -> source video frame, the SAM2 export
                            # scale/crop offset, and this view's extract() crop.
                            self._view_mine_info[view.name] = {
                                "source_frames": list(view_plan.source_frames),
                                "scale": _sam2_scale, "y_off": _y_off,
                                "extract": view.extract,
                            }

                            # Seed gate: a detection projecting into a static zone
                            # cannot found a NEW SAM2 object (a robot is never born
                            # inside a field element) but still corroborates masks.
                            # Low cameras also test the viewing ray beyond the
                            # contact — an occluded structure base projects short
                            # of its zone (qm28 outpost phantom).
                            def _make_seed_gate_fn(project, zones, ray_zones, cam_xy):
                                def _gate(box):
                                    xy = project(box)
                                    if xy is None:
                                        return True
                                    if point_in_zones(xy, zones):
                                        return False
                                    if ray_zones and cam_xy is not None:
                                        return not ray_enters_zones(cam_xy, xy, ray_zones)
                                    return True
                                return _gate

                            _cam_center = view.calibration.camera_center
                            _seed_gate_fn = _make_seed_gate_fn(
                                _project_fn,
                                self.static_seed_zones,
                                self._ray_zone_list(view),
                                None if _cam_center is None else (
                                    float(_cam_center[0]), float(_cam_center[1])
                                ),
                            )

                            # Size-prior hijack cull (A/B knob): expected robot pixel
                            # width at a field position, in SAM2 export pixel space.
                            # Width is invariant to the top-crop y-offset, so only
                            # _sam2_scale matters (unlike _project_fn, which needs
                            # _y_off). Independent of the corroboration machinery:
                            # equipment hijacks are detector-corroborated.
                            _expected_px_fn = None
                            if self.config.size_prior_cull:
                                from web.field.detect_bridge import expected_robot_pixel_width

                                def _make_expected_px_fn(cal, scale, robot_w=1.1):
                                    def _fn(xy):
                                        if xy is None:
                                            return None
                                        px = expected_robot_pixel_width(xy, cal, robot_w)
                                        return None if px is None else px * scale
                                    return _fn
                                _expected_px_fn = _make_expected_px_fn(
                                    view.calibration, _sam2_scale)

                            # Motion override (A/B knob): frame-diff motion probe
                            # over this view's exported frames; boxes at boundaries
                            # are in the same scaled/cropped pixel space as the
                            # jpgs, so no coordinate translation is needed.
                            _motion_fn = None
                            if self.config.motion_override:
                                from web.field.sam2_tracker import make_frame_diff_motion_fn
                                _motion_fn = make_frame_diff_motion_fn(
                                    tmp_dir_obj.name,
                                    self.config.motion_energy_thresh,
                                )

                            # Person-overlap veto (A/B knob): person YOLO on the
                            # exported tracking frame itself — boxes come back in
                            # the same scaled/cropped pixel space as the masks, so
                            # no coordinate translation and no extra video decode.
                            _person_fn = None
                            if self.config.person_veto_enabled:
                                def _person_fn(local_idx, frame_path):
                                    if not frame_path:
                                        return []
                                    img = cv2.imread(frame_path)
                                    if img is None:
                                        return []
                                    res = self.person_model(
                                        img, classes=[0], conf=DET_CONF_THRESHOLD,
                                        verbose=False)[0]
                                    dets = sv.Detections.from_ultralytics(res)
                                    return [tuple(dets.xyxy[i]) for i in range(len(dets))]

                            # Zone-blind corroboration (A/B knob): detections whose
                            # ground contact lands inside a raw static zone are
                            # invisible to the tracker's corroboration bookkeeping.
                            _corrob_gate_fn = None
                            if (self.config.zone_blind_corroboration
                                    and getattr(self, "user_seed_zones_raw", None)):
                                def _make_corrob_gate(project, zones):
                                    def _gate(box):
                                        xy = project(box)
                                        if xy is None:
                                            return True
                                        return not point_in_zones(xy, zones)
                                    return _gate
                                _corrob_gate_fn = _make_corrob_gate(
                                    _project_fn, self.user_seed_zones_raw)

                            # Main-view cull scoping: corner views are the qm24
                            # false-cull victims; None still defers to env knobs
                            # on the main view.
                            from web.field.sam2_tracker import cull_enabled_for_view
                            _cull_enabled = cull_enabled_for_view(
                                self.cull_hijack_enabled, view.name,
                                self.config.cull_main_view_only)

                            t0 = time.perf_counter()
                            view_tracks = await asyncio.to_thread(
                                sam2_tracker.track,
                                frames_dir=tmp_dir_obj.name,
                                prompt_idx=view_plan.prompt_idx,
                                prompt_boxes=prompt_boxes_scaled,
                                reprompt_every=reprompt_every,
                                detect_fn=make_detect_fn(_detect_caps),
                                motion_fn=_motion_fn,
                                person_fn=_person_fn,
                                object_metadata=self.sam2_obj_color[view.name],
                                max_objects=MAX_TRACKED_ROBOTS,
                                max_active_objects=MAX_TRACKED_ROBOTS,
                                # Capped-roster eviction (sam2_tracker) re-seeds robots
                                # under fresh ids, so lifetime id churn is higher than the
                                # old 3x budget; qm24 hit 20 ids by the final frame.
                                max_lifetime_objects=max(MAX_TRACKED_ROBOTS * 6, 36),
                                propagate_backward=True,
                                progress_cb=make_progress_cb(base_pct, span_pct, view.name),
                                diagnostics=self.identity_diagnostics,
                                project_fn=_project_fn,
                                seed_gate_fn=_seed_gate_fn,
                                corrob_gate_fn=_corrob_gate_fn,
                                # None defers to the tracker's env-knob defaults.
                                **({"cull_enabled": _cull_enabled}
                                   if _cull_enabled is not None else {}),
                                **({"reanchor_enabled": self.reanchor_enabled}
                                   if self.reanchor_enabled is not None else {}),
                                dark_reanchor_enabled=self.config.dark_reanchor_enabled,
                                # Frame-denominated wall-clock threshold: rescale
                                # for decimated views ("dark for ~1s" must not
                                # become ~2s on a stride-2 corner). Absence counts
                                # are per re-prompt cycle, already rescaled above.
                                evict_dark_frames=scale_frame_knob(
                                    self.config.evict_dark_frames, view_stride),
                                evict_min_absences=self.config.evict_min_absences,
                                # Misaligned-mask re-anchor: drift correction
                                # between dark and sliver re-anchor.
                                misalign_reanchor=self.config.misalign_reanchor,
                                misalign_ios_min=self.config.misalign_ios_min,
                                misalign_boundaries=self.config.misalign_boundaries,
                                # qm12 hub-face sliver: reject geometrically
                                # implausible seed boxes, and require motion before
                                # a static candidate may evict from the capped roster.
                                seed_box_max_aspect=self.config.seed_box_max_aspect,
                                evict_requires_motion=self.config.evict_requires_motion,
                                # qm11/qm10/qm12 gated leak cleanup: drop a mask
                                # component only if BOTH small AND detached, so a leak
                                # onto adjacent structure can't stretch the box /
                                # field position while an arm / occlusion-split half
                                # is preserved.
                                mask_leak_cleanup=self.config.mask_leak_cleanup,
                                # qm12 opening-window dense reprompt (0:20-0:16 stream-in).
                                opening_reprompt_every=opening_reprompt_every,
                                opening_window_frames=opening_window_frames,
                                # Size-prior hijack cull: geometric width violation
                                # is the one signal a detector false-fire can't
                                # corroborate (equipment hijacks are YOLO-backed).
                                expected_px_fn=_expected_px_fn,
                                size_prior_cull=self.config.size_prior_cull,
                                size_prior_hi=self.config.size_prior_hi,
                                size_prior_lo=self.config.size_prior_lo,
                                size_prior_boundaries=self.config.size_prior_boundaries,
                                # tracker_prob gates: LOW mean confidence gates the
                                # absence cull, HIGH mean confidence grants
                                # eviction immunity. Both default 0.0 = OFF.
                                cull_low_prob_max=self.config.cull_low_prob_max,
                                evict_prob_immune=self.config.evict_prob_immune,
                                # Negative-point prompting: label-0 points at
                                # neighbor detection centers, so SAM conditions
                                # against bleeding onto them from birth.
                                neg_point_prompts=self.config.neg_point_prompts,
                                # Late-seed backward backfill: recover a late-
                                # seeded robot's pre-seed history by bounded
                                # reverse propagation after the forward loop.
                                late_seed_backfill=self.config.late_seed_backfill,
                                backfill_max_frames=backfill_max_frames,
                                backfill_min_lived_frames=backfill_min_lived_frames,
                                # Cooperative stop: the tracker polls this once per
                                # propagated frame and raises TrackStopped so a user
                                # STOP aborts this (longest) phase within a frame's
                                # propagation time. Covers both SAM2 and SAM3 (the
                                # SAM3 tracker inherits this track() method).
                                should_stop=lambda: self._stopped,
                            )
                            elapsed = time.perf_counter() - t0
                            self.timings["sam2_tracking"] = self.timings.get("sam2_tracking", 0.0) + elapsed
                            n = len(view_tracks)
                            print(f"  [{self.tracker_tag}] View {view.name}: propagated {n} frames in {elapsed:.2f}s "
                                  f"({elapsed / max(1, n) * 1000:.1f} ms/frame)")
                            self.sam2_raw[view.name] = view_tracks

                            await safe_send_status(
                                f"[{self.tracker_tag}] View '{view.name}': tracking complete.",
                                base_pct + span_pct,
                            )
                        except TrackStopped:
                            # User stop — NOT a tracker failure. The finally below
                            # releases this view's caps/tempdir; re-raise so the
                            # process_video handler ends the job as a clean stop
                            # rather than a degraded "no tracks" warning.
                            raise
                        except Exception as e:
                            print(f"  [{self.tracker_tag}] Error during propagation for view {view.name}: {e}")
                            # A dead view means a mostly-empty overlay for that
                            # camera; without this warning the job still reports
                            # "completed" and the gap reads as a tracker bug
                            # (it has been a VRAM OOM all three times so far).
                            await self._warn_degraded(
                                self.tracker_tag.lower(),
                                f"View '{view.name}' produced no tracks "
                                f"({type(e).__name__}): {e}",
                            )
                        finally:
                            for _state in _detect_caps:
                                if _state.get("cap") is not None:
                                    _state["cap"].release()
                            tmp_dir_obj.cleanup()
                    # Phase boundary: after each view's propagation. A stop set
                    # DURING propagation already raised TrackStopped above; this
                    # catches one set in the window after track() returned so we
                    # never start seeding/propagating the next view.
                    self._raise_if_stopped()

                if not self.sam2_raw:
                    await self._warn_degraded(
                        self.tracker_tag.lower(),
                        "No views produced tracklets — falling back to Kalman tracking.",
                    )
                    self.use_sam2 = False
                else:
                    self.fused_registry = FusedTrackRegistry(
                        relink_max_dist_m=self.config.relink_max_dist_m)

                # Every tracklet the analysis loop will ever read is already
                # in self.sam2_raw (CPU); the SAM predictor weights and any
                # session state are dead VRAM from here on. Drop the last
                # reference so the cache release below can actually hand the
                # memory to the vLLM server (empty_cache cannot free live
                # tensors).
                del sam2_tracker

                if self.config.hardneg_mine:
                    try:
                        events = self.identity_diagnostics._events
                        n = await asyncio.to_thread(
                            mine_hard_negatives, events, str(self.video_path),
                            self._view_mine_info,
                            str(UPLOAD_DIR / f"hard_negatives_{self.job_id}"))
                        print(f"  [hardneg] mined {n} frames")
                    except Exception as e:
                        print(f"  [hardneg] mining failed (non-fatal): {e}")
            # -------------------------------------

            # Tracking precompute is done; the analysis loop (and the early
            # OCR pass just below) issue bumper reads, so bring the vLLM
            # server back now. On failure the read path falls back to the
            # in-process Qwen reader — degraded speed, not a dead job.
            self._raise_if_stopped()  # phase boundary: before vLLM startup
            if (self.config.vllm_phase_exclusive
                    and getattr(self, "team_reader", "local") == "vllm"):
                from web import vllm_lifecycle
                # Tracking leaves the CUDA caching allocator holding a large
                # RESERVED pool (observed 14.5GiB cached over ~4GiB live);
                # torch never returns it to the OS on its own, so vLLM's
                # ~10.6GB preallocation can't fit until we release the cache.
                try:
                    import gc
                    import torch
                    # Break any reference cycles inside the dropped tracker
                    # so its CUDA tensors are truly unreferenced before the
                    # cache is returned to the OS.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        free_gb = torch.cuda.mem_get_info(0)[0] / 2**30
                        print(f"  [{self.tracker_tag}] released CUDA cache "
                              f"before vLLM start ({free_gb:.1f} GB free)")
                except Exception as e:
                    print(f"  [vllm] cache release failed: {e}")
                await self._send_ws({
                    "type": "status", "status": "processing",
                    "message": "Starting vLLM reader for the analysis phase (~90s model load)…",
                    "progress": 55.0,
                })
                started = await vllm_lifecycle.ensure_started(
                    should_stop=lambda: self._stopped)
                # ensure_started returns early (False) on stop; distinguish a
                # user stop from a genuine startup failure so we don't waste a
                # degradation warning on a job we're about to end.
                self._raise_if_stopped()  # phase boundary: after vLLM startup
                if not started:
                    # The server never came up. Switch this job to the
                    # in-process reader for the rest of the run so every read
                    # batch skips the doomed vLLM round-trip (and its
                    # connection-refused fallback) entirely.
                    self.team_reader = "local"
                    await self._warn_degraded(
                        "vllm",
                        "vLLM reader did not come up after tracking; bumper "
                        "reads fall back to the slower in-process model.",
                    )

            # Early vision OCR pass at the autonomous start when timer gating succeeds.
            # Seeds the field tracker's slots from this frame so VLM reads bind to
            # slot ids (the same identity the main loop uses).
            early_ocr_source_frame = prompt_source_frame
            # The main per-frame loop doesn't advance self._current_frame
            # until it starts (below), so without this the early pass would
            # log/gate against a stale frame 0 (record_vlm_read diagnostics
            # and the det_gated_vlm_reads archive lookup both key off this).
            self._current_frame = early_ocr_source_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, early_ocr_source_frame)
            ret, early_ocr_frame = cap.read()
            if ret and self.match_teams:
                print(f"  Early vision OCR on frame {early_ocr_source_frame}...")
                early_overhead = early_ocr_frame[:self.overhead_height, :]
                if self.bumper_model is not None:
                    results = self.bumper_model(
                        early_overhead, conf=DET_CONF_THRESHOLD, verbose=False
                    )[0]
                    robots, _balls, _hubs = self.det_schema.split(
                        sv.Detections.from_ultralytics(results)
                    )
                    robots = [
                        robot for robot in robots
                        if plausible_bumper_box(robot[0])
                    ]
                else:
                    results = self.model(
                        early_overhead, conf=DET_CONF_THRESHOLD, verbose=False
                    )[0]
                    detections = sv.Detections.from_ultralytics(results)
                    robots = [
                        (
                            tuple(detections.xyxy[i]),
                            None,
                            (
                                float(detections.confidence[i])
                                if detections.confidence is not None
                                else 1.0
                            ),
                        )
                        for i in range(len(detections))
                        if detections.class_id[i] == 1
                    ]
                early_person_results = self.person_model(
                    early_overhead,
                    classes=[0],
                    conf=DET_CONF_THRESHOLD,
                    verbose=False,
                )[0]
                early_person_dets = sv.Detections.from_ultralytics(
                    early_person_results
                )
                early_person_boxes = [
                    tuple(early_person_dets.xyxy[i])
                    for i in range(len(early_person_dets))
                ]
                robots = reject_person_candidates(
                    robots,
                    early_person_boxes,
                    PERSON_ROBOT_REJECT_COVERAGE,
                )
                early_boxes = [np.asarray(box) for box, _color, _conf in robots]
                early_colors = [color for _box, color, _conf in robots]
                main_view = next((v for v in self.views if v.name == "main"), None)
                if main_view:
                    filtered_early_boxes = []
                    filtered_early_colors = []
                    for box, color in zip(early_boxes, early_colors):
                        if not self._below_robot_tracking_top(
                            main_view, box, early_overhead.shape[0]
                        ):
                            continue
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        is_masked = False
                        for (mx, my, mw, mh) in main_view.masks:
                            if mx <= cx <= mx + mw and my <= cy <= my + mh:
                                is_masked = True
                                break
                        if not is_masked:
                            filtered_early_boxes.append(box)
                            filtered_early_colors.append(color)
                    early_boxes = filtered_early_boxes
                    early_colors = filtered_early_colors
                early_rejected = []
                early_field_dets = to_field_detections(
                    early_boxes,
                    self.calib,
                    self.field_dims,
                    colors=early_colors,
                    frame_shape=early_overhead.shape,
                    rejected_out=early_rejected,
                    frac_backstop=self.config.det_frac_backstop,
                )
                if early_rejected:
                    self.hard_neg_logger.maybe_log(
                        early_overhead, [d.box for d in early_field_dets],
                        early_rejected, view_name="main",
                        frame_num=early_ocr_source_frame,
                    )
                early_dt = self.frame_skip / (self.video_fps or 30.0)
                early_outs = self.field_tracker.update(early_field_dets, {}, early_dt, frame_num=early_ocr_source_frame)
                from web.field.fusion import FusedObservation
                early_fused = [FusedObservation(detection=d, evidence=[]) for d in early_field_dets]
                await self._sync_identity_registry_frame(
                    early_fused, [], early_outs, log_frame_num=early_ocr_source_frame)
                early_robots = []
                for t in early_outs:
                    if t.state != "visible" or t.box is None:
                        continue
                    robot = self._get_or_create_slot_track(t.slot_id)
                    robot.alliance = t.alliance
                    robot.field_position = t.pos
                    cx = (t.box[0] + t.box[2]) / 2.0
                    cy = (t.box[1] + t.box[3]) / 2.0
                    robot.last_position = (float(cx), float(cy))
                    early_robots.append((t.slot_id, t.box))
                if early_robots:
                    self.vision_identify_robots(
                        early_overhead,
                        early_robots,
                        boxes_are_bumpers=self.bumper_model is not None,
                    )

            first_processing_frame = self.processing_ranges[0][0]
            last_processing_frame = self.processing_ranges[-1][1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_processing_frame)

            frame_num = first_processing_frame
            processed_frame_num = 0
            analysis_frame_set = {
                frame
                for start, end in self.processing_ranges
                for frame in range(start, end + 1, self.frame_skip)
            }
            auto_core = next(
                (
                    (p["start_frame"], p["end_frame"])
                    for p in self.match_periods
                    if p["label"] == "auto"
                ),
                None,
            )
            previous_processed_source_frame = None
            self._inference_start_time = time.time()

            _loop_t0 = time.perf_counter()
            while True:
                with self._stage("loop.decode"):
                    ret, frame = cap.read()
                if not ret:
                    break
                if frame_num > last_processing_frame:
                    break
                
                if frame_num not in analysis_frame_set:
                    frame_num += 1
                    continue

                self._evidence_frame = frame_num

                if (
                    previous_processed_source_frame is not None
                    and frame_num - previous_processed_source_frame > self.frame_skip * 2
                ):
                    self.ball_trajectories.clear()
                    self.recent_ball_detections.clear()
                    self._ball_score_cooldown.clear()
                    print(
                        f"  [timer] resumed processing at frame {frame_num}; "
                        "reset transient ball state across skipped gap"
                    )
                
                # Crop overhead portion
                overhead_frame = frame[:self.overhead_height, :]
                
                # Existing 3-class YOLO intentionally feeds fuel/hub detection
                # and their HUD annotations; robots come from the bumper model.
                with self._stage("loop.yolo_main"):
                    results = self.model(overhead_frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(results)
                
                # Update hub positions (alliance assignment is static from setup)
                self.update_hub_positions(detections)
                
                # ---- Multi-camera field-space tracking ----
                per_camera = {}            # cam_name -> [(FieldDetection, crop_b64), ...]
                person_boxes_by_cam = {}   # cam_name -> [boxes] in that cam's pixels
                person_evidence_boxes_by_cam = {}
                
                with self._stage("loop.evidence_build"):
                    if self.use_sam2:
                        SAM2_SCALE = self.sam2_export_scale
                        for view in self.views:
                            paired = []
                            view_raw = self.sam2_raw.get(view.name, {})
                            # Per-view index space: a decimated corner view has no
                            # entry for dropped frames -> clean gap (same as
                            # occlusion), never a stale-box reuse.
                            local_idx = self.sam2_view_source_to_local.get(
                                view.name, {}).get(frame_num)
                        
                            view_frame = view.extract(frame)
                            cal = view.calibration
                            # SAM2 tracked a top-cropped view; add the excluded rows back
                            # so box y-coords are in full-crop space (what cal/masks expect).
                            top_crop_frac = (getattr(cal, "robot_track_top_crop", 0.0) or 0.0) if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                            sam2_y_offset = int(view_frame.shape[0] * top_crop_frac) if top_crop_frac > 0 else 0
                        
                            # Prepare small frame for DINOv2 embedding if robot_embedder is loaded
                            small_frame = None
                            if self.robot_embedder is not None:
                                new_w = int(view_frame.shape[1] * SAM2_SCALE)
                                new_h = int(view_frame.shape[0] * SAM2_SCALE)
                                small_frame = cv2.resize(view_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                            with self._stage("loop.person_det"):
                                person_results = self.person_model(
                                    view_frame,
                                    classes=[0],
                                    conf=DET_CONF_THRESHOLD,
                                    verbose=False,
                                )[0]
                                pdets = sv.Detections.from_ultralytics(person_results)
                            person_boxes_all = [
                                tuple(pdets.xyxy[i]) for i in range(len(pdets))
                            ]

                            embed_candidates = []
                            for obj_id, info in view_raw.get(local_idx, {}).items():
                                x1, y1, x2, y2 = info["box"]  # 0.5x top-cropped coords
                                fx1, fy1, fx2, fy2 = (x1 / SAM2_SCALE, y1 / SAM2_SCALE + sam2_y_offset,
                                                      x2 / SAM2_SCALE, y2 / SAM2_SCALE + sam2_y_offset)
                                cx = (fx1 + fx2) / 2.0
                                cy = (fy1 + fy2) / 2.0
                                is_masked = False
                                for (mx, my, mw, mh) in view.masks:
                                    if mx <= cx <= mx + mw and my <= cy <= my + mh:
                                        is_masked = True
                                        break
                                if is_masked:
                                    continue
                                full_box = (fx1, fy1, fx2, fy2)
                                if not reject_person_candidates(
                                    [(full_box, None)],
                                    person_boxes_all,
                                    PERSON_ROBOT_REJECT_COVERAGE,
                                ):
                                    continue
                                # No blocked_zones here: these are tracked masks
                                # (post-seeding evidence); zones gate seeding only.
                                if not filter_field_robot_boxes(
                                    [full_box],
                                    cal,
                                    self.field_dims,
                                    frame_shape=view_frame.shape,
                                    frac_backstop=self.config.det_frac_backstop,
                                ):
                                    continue
                                uv = ((fx1 + fx2) / 2.0, fy2)
                                if self.config.mask_footprint_pos:
                                    # Use the mask's carpet-contact footprint as
                                    # the ray origin instead of the box bottom-
                                    # center: an extended intake/arm stretches
                                    # the box and shifts its bottom-center off
                                    # the robot's actual contact. The mask is in
                                    # 0.5x top-cropped coords (same space as the
                                    # box), so transform mask->full-res exactly
                                    # as the box is transformed above.
                                    from web.field.sam2_tracker import mask_bottom_center
                                    fmask = info.get("mask")
                                    if fmask is not None:
                                        mp = mask_bottom_center(fmask)
                                        if mp is not None:
                                            mx, my = mp
                                            uv = (mx / SAM2_SCALE,
                                                  my / SAM2_SCALE + sam2_y_offset)
                                xy = cal.image_to_field(uv)
                                corner_weight = corner_detection_weight(
                                    view.name, xy, self.field_dims
                                )
                                if corner_weight <= 0.0:
                                    continue
                            
                                color = (
                                    self.sam2_obj_color.get(view.name, {}).get(obj_id)
                                    or "unknown"
                                )

                                box_s = None
                                sub_mask = None
                                if small_frame is not None:
                                    # small_frame is the full-crop at 0.5x; index it with the
                                    # offset-corrected box scaled back to 0.5x (== raw box when
                                    # no top-crop is configured).
                                    box_s = (fx1 * SAM2_SCALE, fy1 * SAM2_SCALE,
                                             fx2 * SAM2_SCALE, fy2 * SAM2_SCALE)
                                    # SAM2 mask is in raw 0.5x top-cropped coords (same space
                                    # as info["box"]); slice the box region out as a box-local
                                    # mask so the signature covers the robot, not the field.
                                    mask2d = info.get("mask")
                                    if mask2d is not None:
                                        sub_mask = mask2d[int(y1):int(y2), int(x1):int(x2)]
                                        if sub_mask.size == 0:
                                            sub_mask = None
                                embed_candidates.append(
                                    (obj_id, full_box, xy, corner_weight, color,
                                     box_s, sub_mask))

                            # One DINOv2 forward pass for every robot in this view
                            # (embed() is embed_batch([box])[0], so semantics match).
                            det_embeds = [None] * len(embed_candidates)
                            if small_frame is not None and embed_candidates:
                                with self._stage("loop.embed"):
                                    det_embeds = self.robot_embedder.embed_batch(
                                        small_frame,
                                        [c[5] for c in embed_candidates],
                                        masks=[c[6] for c in embed_candidates])

                            for (obj_id, full_box, xy, corner_weight, color, box_s,
                                 sub_mask), det_embed in zip(embed_candidates, det_embeds):
                                robot_crop_b64 = None
                                if small_frame is not None:
                                    robot_crop_b64 = self._masked_robot_crop_b64(
                                        small_frame, box_s, sub_mask)

                                det = FieldDetection(
                                    pos=xy,
                                    color=color,
                                    box=full_box,
                                    score=corner_weight,
                                    track_id=obj_id,
                                    embedding=det_embed,
                                    robot_crop_b64=robot_crop_b64,
                                    # Payload only: SAM per-object confidence for
                                    # this frame (None on the legacy detector path).
                                    tracker_prob=(info.get("evidence") or {}).get(
                                        "tracker_prob"),
                                )

                                # Defer the JPEG encode; only top-K chosen crops resolve.
                                def _enc(_f=view_frame, _b=det.box):
                                    cb = self._bumper_crop_b64(_f, _b, enforce_floor=False)
                                    return cb[0] if cb is not None else None
                                paired.append((det, _LazyCrop(_enc)))


                            person_boxes = field_adjacent_persons(person_boxes_all, cal, self.field_dims)
                        
                            per_camera[view.name] = paired
                            person_boxes_by_cam[view.name] = person_boxes
                            person_evidence_boxes_by_cam[view.name] = person_boxes_all
                    else:
                        for view in self.views:
                            pre = detections if view.name == "main" else None
                            paired, pboxes, raw_pboxes = self._detect_view(
                                view,
                                frame,
                                precomputed_dets=pre,
                                frame_num=frame_num,
                            )
                            per_camera[view.name] = paired
                            person_boxes_by_cam[view.name] = pboxes
                            person_evidence_boxes_by_cam[view.name] = raw_pboxes

                with self._stage("loop.fuse_track"):
                    weights = {v.name: (CORNER_VOTE_WEIGHT if v.name != "main" else 1.0)
                               for v in self.views}
                    camera_centers = {
                        v.name: v.calibration.camera_center
                        for v in self.views
                        if v.calibration.camera_center is not None
                    }

                    def _hub_blocks_camera(camera_name, field_pos):
                        center = camera_centers.get(camera_name)
                        return center is not None and any(
                            occluded_by_hub(center, prism, field_pos)
                            for prism in self.hub_prisms
                        )

                    fused = fuse_detections(
                        per_camera,
                        weights=weights,
                        is_hub_occluded=_hub_blocks_camera,
                        camera_centers=(
                            {name: (float(c[0]), float(c[1]))
                             for name, c in camera_centers.items()}
                            if self.config.fuse_ray_gate else None),
                        ray_cross_m=self.config.fuse_ray_cross_m,
                    )   # list[FusedObservation]

                    # Cross-view merge rate is the health signal for multi-camera
                    # identity: silent fusion failure looks like per-view identity
                    # disagreement downstream (split fused tracks, slot churn).
                    if frame_num - getattr(self, "_last_fusion_log", -(10 ** 9)) >= 600:
                        self._last_fusion_log = frame_num
                        _multi = sum(
                            1 for o in fused if len({m[0] for m in o.members}) > 1)
                        print(f"  [fusion] frame {frame_num}: {len(fused)} fused obs, "
                              f"{_multi} cross-view, "
                              f"{sum(len(d) for d in per_camera.values())} raw dets")
                
                    ev_list = []
                    if self.use_sam2:
                        ev_list = self.fused_registry.assign(fused)
                        # A fused track_id is one physical robot; collapse any same-id
                        # split (e.g. a duplicate corner-camera mask whose projection
                        # slid away) into one observation so it can't seed a phantom
                        # slot (the B1<->B3 churn). Corner identity/bearing evidence is
                        # unioned in, and a lone corner obs (main occluded) still
                        # carries the track — so hand-off between cameras is preserved.
                        fused, ev_list = merge_same_id_observations(
                            fused, ev_list, main_guard=self.config.merge_main_guard)

                    field_dets = [obs.detection for obs in fused]
                    dt = self.frame_skip / (self.video_fps or 30.0)
                    portals = (self._portal_state
                               if self.config.occlusion_portal else None)
                    track_outs = self.field_tracker.update(
                        field_dets, self._view_flags, dt, portals=portals,
                        frame_num=frame_num)
                    if TRACKER_TRACE:
                        ft = self.field_tracker
                        self._tracker_trace = getattr(self, "_tracker_trace", [])
                        self._tracker_trace.append({
                            "frame": frame_num,
                            "fused": [{
                                "pos": [round(float(o.detection.pos[0]), 2),
                                        round(float(o.detection.pos[1]), 2)],
                                "cams": sorted({m[0] for m in o.members}),
                                "has_main_box": o.detection.box is not None,
                                "color": o.detection.color,
                                "tid": o.track_id,
                            } for o in fused],
                            "tracks": [{
                                "slot": t.slot_id, "label": self._slot_label(t.slot_id),
                                "pos": [round(float(t.pos[0]), 2), round(float(t.pos[1]), 2)],
                                "state": t.state, "corner_only": t.corner_only,
                            } for t in track_outs],
                            "det_to_slot": {int(k): int(v) for k, v in ft.last_det_to_slot.items()},
                            "events": [dict(e) for e in ft.binding_events],
                            # Raw per-camera detections BEFORE fusion — shows whether a
                            # robot vanished from detection itself (SAM3 gave up / occluded /
                            # left frame) vs was detected but lost in fusion/tracking.
                            "raw": {cam: [{
                                "pos": [round(float(d.pos[0]), 2), round(float(d.pos[1]), 2)],
                                "color": d.color, "has_box": d.box is not None,
                                "tid": d.track_id, "score": round(float(d.score or 1.0), 2),
                            } for (d, _crop) in dets] for cam, dets in per_camera.items()},
                        })
                    try:
                        self._record_frame_evidence(
                            fused,
                            frame_num,
                            person_evidence_boxes_by_cam.get("main", []),
                        )
                    except Exception as e:
                        await self._warn_degraded(
                            "frame_evidence",
                            f"Frame evidence recording failed ({e}); backward "
                            f"re-resolution will be incomplete for this run.",
                        )
                    await self._sync_identity_registry_frame(
                        fused, ev_list, track_outs,
                        main_person_boxes=person_evidence_boxes_by_cam.get("main", []),
                        log_frame_num=frame_num)

                    # 5. Resolve occlusion from every posed camera and merge: a track
                    #    is occluded only if occluded in ALL posed cameras.
                    occ_input = [{"slot_id": t.slot_id, "pos": t.pos, "box": t.box}
                                 for t in track_outs]
                    posed = [v for v in self.views
                             if v.calibration.camera_center is not None]
                    self._occ_by_cam = {
                        v.name: resolve_occlusion(
                            v.calibration, occ_input,
                            person_boxes_by_cam.get(v.name, []), self.hub_prisms)
                        for v in posed
                    }
                    self._occ = _merge_occlusion(list(self._occ_by_cam.values()))
                    self._view_flags = self._occ   # feed forward to next frame's tracker

                    if self.config.occlusion_portal:
                        self._update_portals(track_outs, dt)

                    # 6. Sync RobotTracks by slot_id and build the box list for HUD/VLM.
                    # Mirror each slot's fused-track appearance embedding (kept in the
                    # FusedTrackRegistry, keyed by fused id) onto its RobotTrack. Without
                    # this the per-slot appearance_embedding stays None on the field path,
                    # so the appearance gallery never deposits or queries anything.
                    slot_to_embedding = {}
                    slot_to_robot_crop = {}  # slot -> latest masked whole-robot crop (main cam)
                    if self.use_sam2 and self.field_tracker is not None:
                        for tid, slot in self.field_tracker._id_to_slot.items():
                            emb = self.fused_registry.get_embedding(tid)
                            if emb is not None:
                                slot_to_embedding[slot] = emb
                        # Mirror the main-camera robot crop behind that embedding so the
                        # gallery / replay-review UI can show the whole robot.
                        for obs in fused:
                            if obs.track_id is None:
                                continue
                            slot = self.field_tracker._id_to_slot.get(obs.track_id)
                            if slot is None:
                                continue
                            for cam, det_id in obs.members:
                                if cam != "main":
                                    continue
                                md = next((d for (d, _) in per_camera.get("main", [])
                                           if d.track_id == det_id), None)
                                if md is not None and md.robot_crop_b64:
                                    slot_to_robot_crop[slot] = md.robot_crop_b64
                                break

                tracked_robot_boxes = []  # (slot_id, box) for visible tracks
                for t in track_outs:
                    robot = self._get_or_create_slot_track(t.slot_id)
                    robot.alliance = t.alliance  # slot (bumper-color) alliance wins
                    robot.field_position = t.pos  # field meters, for consumers that want it
                    robot.last_tracker_prob = t.tracker_prob
                    _emb = slot_to_embedding.get(t.slot_id)
                    if _emb is not None:
                        robot.appearance_embedding = _emb
                    _crop = slot_to_robot_crop.get(t.slot_id)
                    if _crop:
                        robot.appearance_crop_b64 = _crop
                    if t.state == "visible" and t.box is not None:
                        cx = (t.box[0] + t.box[2]) / 2.0
                        cy = (t.box[1] + t.box[3]) / 2.0
                        robot.last_position = (float(cx), float(cy))  # pixel space (scoring)
                        robot.positions_history.append((processed_frame_num, float(cx), float(cy)))
                        robot.last_seen_frame = frame_num
                        tracked_robot_boxes.append((t.slot_id, t.box))
                        if auto_core and auto_core[0] <= frame_num <= auto_core[1]:
                            # Per-sample quality signals (payload only). `_occ` is
                            # the merged occlusion dict keyed by slot_id; it may be
                            # one frame stale (background-task schedule) — an honest
                            # approximation for path-quality metrics. Store the
                            # occlusion cause string when the bumper is occluded,
                            # else None; tracker_prob is the SAM confidence of the
                            # detection that measured this slot (None if absent).
                            _occ_slot = (self._occ or {}).get(t.slot_id) or {}
                            _tp = t.tracker_prob
                            self.autonomous_paths[self._slot_label(t.slot_id)].append({
                                "frame_num": frame_num,
                                "timestamp": round(
                                    frame_num / (self.video_fps or 1.0), 4
                                ),
                                "field_pos": [
                                    round(float(t.pos[0]), 4),
                                    round(float(t.pos[1]), 4),
                                ],
                                "corner_only": bool(t.corner_only),
                                "occluded": (_occ_slot.get("cause")
                                             if _occ_slot.get("bumper_occluded")
                                             else None),
                                "tracker_prob": (round(float(_tp), 4)
                                                 if _tp is not None else None),
                            })
                    # Coasting tracks keep their stale last_position untouched.

                with self._stage("loop.vlm"):
                    # VLM — read bumper numbers on a time interval. Qwen3-VL is LOCAL and
                    # free, so there's no per-match call cap (the old 20-call budget was a
                    # paid-API leftover). We schedule VLM reads per-identity based on match seconds.
                    current_match_time = processed_frame_num / (self.video_fps or 10.0)
                    unconfirmed_count = sum(
                        1 for ident in self.identity_registry.identities.values() if not ident.is_confirmed
                    )
                
                    due_priorities = {}
                    for ident in self.identity_registry.identities.values():
                        if ident.active_slot is None:
                            continue
                        # Occlusion gate: a bumper hidden in EVERY camera cannot
                        # be read — at best a wasted call, at worst a garbage-crop
                        # contradiction against a correct lock. The
                        # needs_immediate_read flag is NOT cleared, so the read
                        # fires on the first visible frame.
                        if VLM_OCCLUSION_GATE and occlusion_blocks_read(
                                (self._occ or {}).get(ident.active_slot)):
                            self.identity_diagnostics.increment(
                                "vlm_scheduled.occluded_skip")
                            continue
                        # Novelty inputs: a failed read of an unmoved robot is
                        # the same unreadable view again (decide_vlm_read gates
                        # it, bounded by novelty_skip_max_sec).
                        last_failed = (ident.last_vlm_time > ident.last_success_time)
                        moved = True
                        slot_track = self._get_slot_track_if_exists(ident.active_slot)
                        cur_pos = getattr(slot_track, "field_position", None)
                        if ident.last_vlm_pos is not None and cur_pos is not None:
                            moved = (
                                (cur_pos[0] - ident.last_vlm_pos[0]) ** 2
                                + (cur_pos[1] - ident.last_vlm_pos[1]) ** 2
                            ) ** 0.5 >= self.confidence_config.novelty_min_move_m
                        decision = decide_vlm_read(
                            is_confirmed=ident.is_confirmed,
                            confidence=ident.confidence,
                            last_vlm_time=ident.last_vlm_time,
                            last_success_time=ident.last_success_time,
                            backoff_until=ident.backoff_until,
                            needs_immediate_read=ident.needs_immediate_read,
                            current_match_time=current_match_time,
                            unconfirmed_count=unconfirmed_count,
                            config=self.confidence_config,
                            last_read_failed=last_failed,
                            moved_since_read=moved,
                        )
                        if decision.due:
                            due_priorities[ident.identity_id] = decision.priority
                            self.identity_diagnostics.increment(
                                f"vlm_scheduled.{decision.reason}"
                            )
                            self.identity_diagnostics.record(
                                DiagnosticEvents.VLM_SCHEDULED,
                                identity_id=ident.identity_id,
                                reason_type=decision.reason,
                                priority=decision.priority,
                                is_audit=decision.is_audit,
                            )

                    self.vlm_status.update({
                        "unconfirmed": unconfirmed_count,
                        "due_count": len(due_priorities),
                    })
                    if bool(fused) and len(due_priorities) > 0:
                        # Single-flight background schedule (see the method): the
                        # read pass no longer blocks frame processing — it runs
                        # as an event-loop task while the loop keeps decoding.
                        self._schedule_vlm_pass(fused, due_priorities)

                    if getattr(self.config, "equipment_adjudicate", False):
                        # Position history feeds track_is_static; only paid
                        # for when the adjudication knob is on.
                        for ident in self.identity_registry.get_active_identities():
                            _st = self._get_slot_track_if_exists(ident.active_slot)
                            _p = getattr(_st, "field_position", None)
                            if _p is not None:
                                self._ident_pos_history.setdefault(
                                    ident.identity_id, deque(maxlen=600)
                                ).append((current_match_time, float(_p[0]), float(_p[1])))
                        self._maybe_schedule_adjudication(current_match_time)

                with self._stage("loop.drift_monitor"):
                    if (self._drift_monitor is not None and not self._drift_warned
                            and self._drift_monitor.due(current_match_time)):
                        main_view = next((v for v in self.views if v.name == "main"), None)
                        if main_view is not None:
                            main_crop = main_view.extract(frame)
                            detected = detect_tags(main_crop)
                            err_m, ntags = tag_alignment_error(self.calib, detected, self.field_model)
                            self._drift_monitor.record(current_match_time, err_m, ntags)
                            if self._drift_monitor.drifted:
                                self._drift_warned = True
                                if self.match_key:
                                    field_store.record_drift(self.event_key, self.camera,
                                                             self.match_key, err_m,
                                                             self._calib_version_id)
                                await self._warn_degraded(
                                    "calib_drift",
                                    f"Calibration drifted mid-match (~{err_m:.2f} m at "
                                    f"t={current_match_time:.0f}s): positions after this point "
                                    "are suspect; recalibrate before the next match.")

                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()
                    self.audit_overturn_confirmations()
                    self._emit_diagnostic_snapshot(processed_frame_num)

                # Ball trajectory tracking — link ball detections across frames
                # and detect when balls enter hubs (with trajectory-based shooter ID)
                self.update_ball_trajectories(detections, processed_frame_num)
                
                # OCR AUTHORITATIVE SCORING:
                # a single valid OCR read (close to previous read) becomes the score.
                # Trajectories are used only for per-robot attribution.
                with self._stage("loop.ocr"):
                    if processed_frame_num >= 10 and processed_frame_num % OCR_FRAME_INTERVAL == 0:
                        if processed_frame_num <= 20:
                            print(f"  [DEBUG] OCR regions: {self.ocr_regions}")
                            print(f"  [DEBUG] overhead_frame shape: {overhead_frame.shape}")
                        for alliance in ["blue", "red"]:
                            ocr_val = self.ocr_score(overhead_frame, alliance)
                            if processed_frame_num <= 50:
                                print(f"  [DEBUG] OCR {alliance}: {ocr_val}")
                            if ocr_val is not None:
                                last_ocr = self._last_ocr.get(alliance)
                                if last_ocr is not None and abs(ocr_val - last_ocr) > 20:
                                    continue  # Likely a misread, skip
                                self._last_ocr[alliance] = ocr_val
                            
                                current = self.alliance_scores[alliance]
                                if ocr_val != current:
                                    if ocr_val > current:
                                        # OCR says we undercounted — add missing goals and attribute each.
                                        missing = ocr_val - current
                                        timestamp = frame_num / self.video_fps
                                        for _ in range(missing):
                                            self.attribute_score(alliance, processed_frame_num, timestamp)
                                        self.alliance_scores[alliance] = ocr_val
                                        print(f"  OCR authoritative [{alliance}]: {current} -> {ocr_val} (+{missing} attributed)")
                                    elif ocr_val < current:
                                        # OCR says we overcounted — adjust down immediately.
                                        self.alliance_scores[alliance] = ocr_val
                                        self._adjust_robot_scores_down(alliance, ocr_val)
                                        print(f"  OCR authoritative [{alliance}]: {current} -> {ocr_val} (corrected down)")
                
                with self._stage("loop.hud"):
                    # Draw annotations
                    annotated_frame = overhead_frame.copy()
                    # Optional verification pass: lay the reprojected field wireframe down
                    # first so robot/ball/hub HUD draws on top of it (and it's baked into
                    # both the live stream and the recorded replay).
                    self._maybe_draw_field_wireframe(annotated_frame)
                    track_by_slot = {t.slot_id: t for t in track_outs}
                
                    # Draw detections
                    for i in range(len(detections)):
                        box = detections.xyxy[i]
                        class_id = detections.class_id[i]
                    
                        x1, y1, x2, y2 = map(int, box)
                    
                        if class_id == 0:  # fuel_ball
                            color = (0, 255, 255)  # Yellow
                            label = "ball"
                        elif class_id == 1:  # robot
                            continue
                        else:  # hub_opening
                            color = (0, 255, 0)  # Green
                            label = "hub"

                        # Per-ball boxes are visual noise (hundreds of balls per frame); skip
                        # when disabled.  Hub boxes (class_id 2) are kept regardless.
                        if class_id == 0 and not HUD_DRAW_BALL_BOXES:
                            continue

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        # Label with background for readability
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Robots are drawn from visible tracker slots. Their boxes and
                    # alliance colors originate from the bumper-model path.
                    for slot_id, box in tracked_robot_boxes:
                        robot = self._get_slot_track_if_exists(slot_id)
                        if robot is None:
                            continue
                        label, _confirmed = self._team_hud_label(
                            robot, self._slot_label(slot_id)
                        )
                        track = track_by_slot.get(slot_id)
                        mask = None
                        if self.use_sam2:
                            local_idx = self.sam2_source_to_local.get(frame_num)
                            registry_track_id = next(
                                (
                                    track_id
                                    for track_id, mapped_slot in
                                    self.field_tracker._id_to_slot.items()
                                    if mapped_slot == slot_id
                                ),
                                None,
                            )
                            if registry_track_id is not None:
                                obs = next(
                                    (
                                        item for item in fused
                                        if item.track_id == registry_track_id
                                    ),
                                    None,
                                )
                                if obs is not None:
                                    for camera, det_id in obs.members:
                                        if camera != "main" or "main" not in self.sam2_raw:
                                            continue
                                        entry = self.sam2_raw["main"].get(
                                            local_idx, {}
                                        ).get(det_id)
                                        if entry is not None:
                                            mask2d = entry.get("mask")
                                            if mask2d is not None:
                                                main_view = next(
                                                    (
                                                        view
                                                        for view in self.views
                                                        if view.name == "main"
                                                    ),
                                                    None,
                                                )
                                                top_offset_px = 0
                                                if main_view is not None:
                                                    top_crop_frac = (
                                                        getattr(
                                                            main_view.calibration,
                                                            "robot_track_top_crop",
                                                            0.0,
                                                        )
                                                        or 0.0
                                                    ) if ROBOT_TRACK_TOP_CROP_ENABLED else 0.0
                                                    top_offset_px = int(
                                                        overhead_frame.shape[0]
                                                        * top_crop_frac
                                                    )
                                                mask = _restore_sam2_mask_to_view(
                                                    mask2d,
                                                    overhead_frame.shape,
                                                    top_offset_px,
                                                )
                                        break
                        self._draw_bumper_hud(
                            annotated_frame,
                            box,
                            robot.alliance,
                            label,
                            slot_id=slot_id,
                            cal=self.calib,
                            field_pos=(
                                track.pos if track else robot.field_position
                            ),
                            vel=track.vel if track else None,
                            mask=mask,
                        )

                    # Coasting tracks: position alive, mask gone. Without a ghost
                    # they are indistinguishable from never-seen robots in the
                    # camera view (they only show on the field map).
                    if HUD_DRAW_COASTING_GHOST:
                        drawn_slots = {sid for sid, _ in tracked_robot_boxes}
                        for slot_id, gbox in select_coasting_ghosts(track_outs):
                            if slot_id in drawn_slots:
                                continue
                            robot = self._get_slot_track_if_exists(slot_id)
                            if robot is None:
                                continue
                            label, _ = self._team_hud_label(
                                robot, self._slot_label(slot_id))
                            gcolor = ((153, 60, 0) if robot.alliance == "blue"
                                      else (0, 30, 153))
                            draw_dashed_box(annotated_frame, gbox, gcolor, thickness=2)
                            gx1, gy1 = int(gbox[0]), int(gbox[1])
                            cv2.putText(annotated_frame, f"{label} coast",
                                        (gx1, max(12, gy1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, gcolor, 1,
                                        cv2.LINE_AA)

                    # Draw hubs
                    for hub in self.hubs:
                        color = (255, 0, 0) if hub.alliance == "blue" else (0, 0, 255)
                        cv2.circle(annotated_frame, (int(hub.x), int(hub.y)), 20, color, 3)
                        cv2.putText(annotated_frame, f"{hub.alliance} hub",
                                   (int(hub.x) - 30, int(hub.y) - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Draw OCR score regions so user can verify placement
                    for region_key, (rx, ry, rw, rh) in self.ocr_regions.items():
                        if "blue" in region_key:
                            rect_color = (255, 0, 0)  # Blue in BGR
                        elif "red" in region_key:
                            rect_color = (0, 0, 255)  # Red in BGR
                        else:
                            rect_color = (0, 255, 255)  # Yellow in BGR
                        cv2.rectangle(annotated_frame, (rx, ry), (rx + rw, ry + rh), rect_color, 2)
                        cv2.putText(annotated_frame, region_key,
                                   (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rect_color, 1)

                    # Radar + corner camera HUD panels live in a right rail, not over the video.
                    radar_tracks = []
                    radar_seen_slots = set()
                    for t in track_outs:
                        if t.state == "empty":
                            continue
                        radar_seen_slots.add(t.slot_id)
                        trail = self._radar_trails.setdefault(
                            t.slot_id, deque(maxlen=16))
                        trail.append(t.pos)
                        slot_track = self._get_slot_track_if_exists(t.slot_id)
                        team = slot_track.team_number if slot_track is not None else None
                        radar_tracks.append(RadarTrack(
                            pos=t.pos, alliance=t.alliance,
                            team=team or self._slot_label(t.slot_id),
                            slot_id=t.slot_id, state=t.state,
                            corner_only=t.corner_only, trail=tuple(trail)))
                    # A slot that vanished re-seeds its trail from scratch.
                    for sid in list(self._radar_trails):
                        if sid not in radar_seen_slots:
                            del self._radar_trails[sid]
                    radar_img = self.radar.render(radar_tracks)
                    camera_panels = [
                        self._camera_panel(v, frame, per_camera.get(v.name, []), track_outs)
                        for v in self.views if v.name != "main"
                    ]
                    annotated_frame = self._compose_hud_frame(annotated_frame, radar_img, camera_panels)

                with self._stage("loop.replay"):
                    # --- Replay recording: capture the composited HUD frame + metadata
                    try:
                        if not self._replay.is_open:
                            eff_fps = (self.video_fps / self.frame_skip) if (self.video_fps and self.frame_skip) else 8.0
                            self._replay.open_lazy(annotated_frame, eff_fps, REPLAYS_DIR / self.job_id)
                        self._replay.add_frame(annotated_frame, {
                            "frame_num": frame_num,
                            "scores": dict(self.alliance_scores),
                            "robots": [
                                {
                                    "slot": self._slot_label(t.slot_id),
                                    "alliance": t.alliance,
                                    "team_number": self._get_display_team_number(self._get_or_create_slot_track(t.slot_id)),
                                    "confirmed": self._get_or_create_slot_track(t.slot_id).team_number_confirmed,
                                    "field_pos": [float(t.pos[0]), float(t.pos[1])] if t.pos is not None else None,
                                }
                                for t in track_outs if t.state != "empty"
                            ],
                        })
                    except Exception as e:
                        print(f"[replay] frame record error: {e}")

                # Stream frame data via the latest-only queue: build display
                # rows and encode only when a send is due, and never wait on
                # the client.
                with self._stage("loop.ws_emit"):
                    if self.websocket and self._frame_stream_due(time.perf_counter()):
                        robots_data = self._canonical_robot_rows()
                        events_data = self._recent_events_data()
                        _, buffer = cv2.imencode(
                            '.jpg', annotated_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.config.frame_stream_jpeg_quality])
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        # Only stream the VLM crops on frames where a fresh batch exists,
                        # to avoid resending the same images every frame.
                        ocr_crops_payload = self.last_ocr_crops if self._ocr_crops_dirty else None
                        self._ocr_crops_dirty = False
                        self._enqueue_frame_payload(jsonify({
                            "type": "frame",
                            "frame_num": frame_num,
                            "processed_frame_num": processed_frame_num,
                            "total_frames": self.total_frames,
                            "fps": self.video_fps,
                            "frame_data": frame_b64,
                            "backend": self.tracking_backend,
                            "scores": self.alliance_scores,
                            "robots": robots_data,
                            "scoring_events": events_data,
                            "ocr_crops": ocr_crops_payload,
                            "vlm_status": self.vlm_status,
                        }))

                self._current_frame = frame_num
                previous_processed_source_frame = frame_num

                if self._stopped:
                    break

                if self._paused:
                    self._pause_frame = self._current_frame
                    loop = asyncio.get_running_loop()
                    last_status_sent = 0.0
                    while self._paused and not self._stopped:
                        now = loop.time()
                        if self.websocket and (now - last_status_sent >= 1.0):
                            self._enqueue_ws("status", {
                                "type": "status",
                                "status": "paused",
                                "frame_num": self._pause_frame
                            })
                            last_status_sent = now
                        await asyncio.sleep(0.1)

                    if self._stopped:
                        break
                
                frame_num += 1
                processed_frame_num += 1
            
            # Drain the in-flight background VLM pass (if any) BEFORE finalizing:
            # its votes must land before assign_teams_from_votes/finalization
            # runs. Awaited, not cancelled — reached on normal break and on the
            # self._stopped break above.
            await self._drain_vlm_task()

            _adj = self._adjudicate_task
            if _adj is not None:
                try:
                    if not _adj.done():
                        await _adj
                    elif _adj.exception():
                        print(f"  Adjudication pass failed (drain): {_adj.exception()}")
                except Exception as e:
                    print(f"  Adjudication pass failed (drain): {e}")
                finally:
                    self._adjudicate_task = None

            # Fold the per-stage accumulators into timings as flat "loop.<stage>"
            # keys (survive the end-of-job round pass) and print one hot-path line.
            for _k, _v in sorted(self._loop_stage_s.items(), key=lambda kv: -kv[1]):
                self.timings[_k] = round(_v, 2)
            if self._loop_stage_s:
                _tot = time.perf_counter() - _loop_t0
                _acct = sum(v for k, v in self._loop_stage_s.items()
                            if k not in ("loop.person_det", "loop.embed", "loop.crop_encode"))  # nested inside evidence_build
                print("  [profile] analysis loop stages (s): "
                      + ", ".join(f"{k.split('.',1)[1]}={v:.1f}" for k, v in sorted(self._loop_stage_s.items(), key=lambda kv: -kv[1]))
                      + f" | accounted {_acct:.1f}/{_tot:.1f}s")
            self.timings["analysis_loop"] = round(time.perf_counter() - _loop_t0, 2)
            self.timings["total"] = round(time.perf_counter() - self._proc_t0, 2)
            self.detection_archive.close()
            self.status = "completed"

            # Offline RTS smoothing of autonomous_paths. Placed here — after the
            # tracking loop, before the WS "complete" send — so the post-match
            # summary the client receives and the persisted replay artifact are
            # smoothed from one place (no client-vs-meta.json divergence);
            # per-frame live rendering during the loop stayed causal. The later
            # gated _reresolve_replay_frames/filter_paths_by_dropped pass
            # composes correctly: it filters the already-smoothed dict in
            # place, so dropped ranges still get removed, interpolated samples
            # included.
            if self.config.rts_smooth and self.field_tracker is not None:
                # Offline pass: never let a smoothing failure kill a job that
                # already finished tracking.
                try:
                    from web.field.smoother import (smooth_history,
                                                    apply_smoothed_paths)
                    stats = apply_smoothed_paths(
                        self.autonomous_paths,
                        smooth_history(self.field_tracker.history),
                        self._slot_label,
                        self.video_fps or 1.0)
                    print(f"  [rts] smoothed autonomous_paths "
                          f"(label: replaced, inserted): {stats}")
                except Exception as e:
                    print(f"  [rts] smoothing failed, exporting causal paths: {e}")

            # Deliver any in-flight frame before the complete message.
            await self._stop_ws_senders(drain=True)
            await self._send_ws(jsonify({
                "type": "complete",
                "total_frames": self.total_frames,
                "robots": self._canonical_robot_rows(),
                "scoring_events": self._recent_events_data(),
                "final_scores": self.alliance_scores,
                "periods": self.match_periods,
                "autonomous_paths": dict(self.autonomous_paths),
                "processing_sec": self.timings.get("total"),
                "timings": self.timings,
                "degradation_warnings": getattr(self, "degradation_warnings", []),
                # A/B provenance: every knob that differed from the code default.
                "config": self.config.non_defaults(),
            }))
        
        except (JobStopped, TrackStopped):
            # User pressed STOP before the analysis loop began (every
            # _raise_if_stopped poll and the tracker's TrackStopped fire only
            # in the pre-loop phases; the analysis loop itself stops via its
            # own `if self._stopped: break`, which finalizes with partial
            # results and never raises). Terminal `error` state with the
            # standard error message, same websocket path as other job errors.
            # The finally below still runs all cleanup.
            self.status = "error"
            self.error = "Stopped by user"
            print("  process_video: stopped by user before the analysis loop")
            await self._send_ws({
                "type": "error",
                "error": self.error,
            })
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            import traceback
            print(f"ERROR in process_video: {e}")
            print(traceback.format_exc())
            await self._send_ws({
                "type": "error",
                "error": str(e)
            })
        finally:
            # Defensive drain for the exception path (the normal path already
            # drained before finalizing): don't orphan an in-flight background
            # VLM task. Idempotent — a no-op once _vlm_task is None.
            try:
                await self._drain_vlm_task()
            except Exception:
                pass
            # Same idea for the observe-only adjudication task: don't let an
            # exception-triggered teardown orphan it. This is the failure
            # path, so we cancel rather than await (awaiting could block
            # teardown up to the 120s HTTP timeout).
            try:
                _adj = self._adjudicate_task
                if _adj is not None:
                    if not _adj.done():
                        _adj.cancel()
                    else:
                        exc = _adj.exception()
                        if exc is not None:
                            print(f"  Adjudication pass failed (teardown): {exc}")
            except Exception:
                pass
            finally:
                self._adjudicate_task = None
            # Leave the GPU clean for whatever runs next: the vLLM server is
            # only needed during this job's analysis phase, and the next
            # job's tracking would have to stop it anyway.
            if getattr(self.config, "vllm_phase_exclusive", False):
                try:
                    from web import vllm_lifecycle
                    await vllm_lifecycle.ensure_stopped()
                except Exception as e:
                    print(f"  [vllm] post-job stop failed: {e}")
            try:
                await self._stop_ws_senders(drain=False)
            except Exception:
                pass
            try:
                self.diagnostic.close()
            except Exception:
                pass
            try:
                self.detection_archive.close()
            except Exception:
                pass
            self._close_log_file()
            if cap is not None:
                cap.release()
            # Sweep per-view frame-export tempdirs a mid-phase stop (or any
            # exception) left behind: the per-view propagation loop only cleans
            # the view it just finished, so a JobStopped/TrackStopped raised
            # during export or an early view would otherwise strand the rest
            # for GC-time finalizers. cleanup() on an already-cleaned dir is a
            # no-op failure we swallow.
            for _vd in (getattr(self, "_view_export_dirs", None) or {}).values():
                try:
                    _vd.cleanup()
                except Exception:
                    pass
            self._view_export_dirs = None
            # Finalize processing timings (covers partial/errored runs the success
            # path never reached) and round to 2 decimals for display/storage.
            timings = getattr(self, "timings", {}) or {}
            if "total" not in timings and getattr(self, "_proc_t0", None) is not None:
                timings["total"] = round(time.perf_counter() - self._proc_t0, 2)
            timings = {k: round(v, 2) for k, v in timings.items()}
            self.timings = timings
            try:
                rec = getattr(self, "_replay", None)
                if rec is not None and rec.is_open:
                    status = "complete" if self.status == "completed" else "partial"
                    blue_teams = [self.robot_tracks[s].team_number for s in sorted(self.robot_tracks)
                                  if self.robot_tracks[s].alliance == "blue"]
                    red_teams = [self.robot_tracks[s].team_number for s in sorted(self.robot_tracks)
                                 if self.robot_tracks[s].alliance == "red"]
                    summary = {
                        "match_label": Path(str(self.video_path)).stem,
                        "blue_score": self.alliance_scores.get("blue", 0),
                        "red_score": self.alliance_scores.get("red", 0),
                        "robots": [r for f in rec.frames[:1] for r in f["robots"]],
                        "processing_sec": timings.get("total"),
                        "timings": timings,
                        "periods": self.match_periods,
                        "processing_ranges": self.processing_ranges,
                        "timer_gate_fallback": self.timer_gate_fallback,
                        "timer_gate_fallback_reason": self.timer_gate_fallback_reason,
                        "autonomous_paths": dict(self.autonomous_paths),
                    }
                    self.identity_diagnostics.gauge(
                        "identity_count", len(self.identity_registry.identities)
                    )
                    self.identity_diagnostics.gauge(
                        "unresolved_occlusion_groups",
                        len(self.co_occlusion_tracker.get_active_events()),
                    )
                    self.identity_diagnostics.gauge(
                        "deferred_attributions", len(self.deferred_attributions)
                    )
                    if BACKWARD_CONFIRM_ENABLED:
                        try:
                            self._reresolve_replay_frames()
                            # Re-resolution may have dropped phantom/alien path
                            # segments; refresh the snapshot built before it ran.
                            summary["autonomous_paths"] = dict(self.autonomous_paths)
                        except Exception as e:
                            print(f"[backward-confirm] re-resolution skipped: {e}")
                    if TRACKER_TRACE and getattr(self, "_tracker_trace", None):
                        try:
                            Path("/tmp/frc-tracker-trace.json").write_text(
                                json.dumps(self._tracker_trace))
                            print(f"[tracker-trace] wrote {len(self._tracker_trace)} "
                                  "frames -> /tmp/frc-tracker-trace.json")
                        except Exception as e:
                            print(f"[tracker-trace] write failed: {e}")
                    artifacts = rec.finalize(
                        status=status,
                        summary=summary,
                        diagnostics=getattr(self, "identity_diagnostics", None),
                    )
                    import json as _json
                    from datetime import datetime as _dt
                    await database.insert_replay({
                        "id": self.job_id,
                        "created_at": _dt.now().isoformat(timespec="seconds"),
                        "label": Path(str(self.video_path)).stem,
                        "event_key": self.event_key, "match_key": self.match_key, "comp_level": None,
                        "blue_score": self.alliance_scores.get("blue", 0),
                        "red_score": self.alliance_scores.get("red", 0),
                        "blue_teams": _json.dumps(blue_teams),
                        "red_teams": _json.dumps(red_teams),
                        "source_video_path": str(self.video_path),
                        "processing_sec": timings.get("total"),
                        "timings": _json.dumps(timings),
                        **artifacts,
                    })
                    print(f"[replay] saved replay {self.job_id}: {artifacts['frame_count']} frames, status={status}")
            except Exception as e:
                print(f"[replay] finalize error: {e}")
            try:
                if self._pending_gallery_rows:
                    await database.insert_gallery_exemplars(self._pending_gallery_rows)
                    print(f"  Appearance gallery: persisted {len(self._pending_gallery_rows)} "
                          f"new exemplars for {self.event_key}")
                    self._pending_gallery_rows = []
            except Exception as e:
                print(f"  Appearance gallery flush failed ({e})")
            try:
                identity_rows = []
                crop_rows = []
                for slot_id, identity_id in self._canonical_identity_slots().items():
                    robot = self.robot_tracks.get(identity_id)
                    if robot is None:
                        continue
                    crops = self._slot_crops.get(identity_id, [])
                    identity_rows.append({
                        "job_id": self.job_id, "identity_id": identity_id, "slot": slot_id,
                        "slot_label": self._slot_label(slot_id),
                        "alliance": robot.alliance, "team_number": robot.team_number,
                        "confirmed": int(bool(robot.team_number_confirmed)),
                        "source": self._slot_source.get(identity_id, "ocr"),
                        "confidence": (crops[0]["quality"] if crops else None),
                    })
                    for c in crops:
                        has_embedding = c["embedding"] is not None
                        crop_rows.append({
                            "job_id": self.job_id, "identity_id": identity_id,
                            "legibility": c["quality"],
                            "embedding": (np.asarray(c["embedding"], dtype=np.float32).tobytes()
                                          if has_embedding else None),
                            "image": _thumbnail_bytes(c.get("robot_crop_b64") or c["crop_b64"]),
                            # Tag with the model that produced this vector (None
                            # alongside a None embedding — nothing to mismatch).
                            "embed_model": (getattr(getattr(self, "robot_embedder", None),
                                                     "model_name", None)
                                            if has_embedding else None),
                        })
                await database.upsert_match_robot_identities(identity_rows)
                await database.insert_match_robot_crops(crop_rows)
            except Exception as e:
                print(f"  Match-identity capture failed ({e})")


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    from database import init_db
    await init_db()


@app.get("/")
async def index():
    """Serve frontend"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/calibrate")
async def calibrate_page():
    return FileResponse(STATIC_DIR / "calibrate.html")


@app.get("/api/job/active")
async def active_job():
    """The currently running/queued job, if any.

    A page reload loses the job id, so the UI calls this on load to rejoin a
    running match (the websocket reattach path streams without re-running).
    """
    for job_id in reversed(jobs):
        job = jobs[job_id]
        if job.status not in ("completed", "error"):
            return {
                "job_id": job_id,
                "status": job.status,
                "started_at": getattr(job, "started_at", None),
                "match_key": getattr(job, "match_key", None),
            }
    return {"job_id": None}


@app.get("/api/settings")
async def get_settings():
    """Every registered knob with metadata and current resolution.

    `effective`/`source` reflect what the NEXT submitted job will use
    (per-job config_json still wins); `launch_value` is what a server
    restart returns to. Runtime overrides are in-memory by design.
    """
    return {"knobs": job_config_registry.describe_knobs()}


@app.post("/api/settings")
async def set_setting(body: dict):
    """Set (or clear, with value=null) one runtime knob override.

    Applies to jobs submitted after the change; the currently running job
    keeps the config it was created with.
    """
    name = body.get("name")
    if not name:
        return JSONResponse(status_code=400, content={"error": "missing 'name'"})
    try:
        job_config_registry.set_runtime_override(name, body.get("value"))
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    knob = next(k for k in job_config_registry.describe_knobs()
                if k["name"] == name)
    return {"ok": True, "knob": knob}


@app.post("/api/settings/reset")
async def reset_settings():
    """Clear every runtime override, returning to the launch profile."""
    job_config_registry.reset_runtime_overrides()
    return {"ok": True}


@app.get("/api/debug/tasks")
async def debug_tasks():
    """Stack of every live asyncio task.

    Diagnostic hook for event-loop hangs: a job that stalls with the log
    frozen, all OS threads idle and the loop still serving requests is a
    coroutine awaiting something that never resolves — py-spy can't see
    which one, this can. Hit it from a browser/curl while the hang is live.
    """
    out = []
    for task in asyncio.all_tasks():
        frames = [
            f"{f.f_code.co_filename}:{f.f_lineno} {f.f_code.co_name}"
            for f in task.get_stack(limit=30)
        ]
        out.append({
            "name": task.get_name(),
            "done": task.done(),
            "stack": frames,
        })
    return {"tasks": out}


@app.get("/api/status")
async def status():
    """Return model and system status"""
    latest_job = jobs[next(reversed(jobs))] if jobs else None
    bumper_classes = (
        dict(latest_job.bumper_model.names)
        if latest_job is not None and latest_job.bumper_model is not None
        else None
    )
    return {
        "model": "YOLO26s Hub-Zone + YOLOv8m Bumpers",
        "model_path": {
            "fuel_hub_model": str(MODEL_PATH),
            "bumper_model": str(BUMPER_MODEL_PATH),
        },
        "model_classes": {
            "fuel_hub_model": {0: "fuel_ball", 1: "robot", 2: "hub_opening"},
            "bumper_model": bumper_classes,
        },
        "status": "ready",
        "has_event": current_event is not None,
        "event_key": current_event['event_key'] if current_event else None,
        "event_name": current_event['event_name'] if current_event else None,
    }


@app.post("/api/ocr-regions")
async def set_ocr_regions(request: Request):
    """Set custom OCR regions for score reading. 
    Coordinates are in the overhead crop coordinate space (1920x723 for 1080p).
    Body keys: red_center, blue_center, and optional match_clock."""
    global custom_ocr_regions
    body = await request.json()
    # Build locally, assign once: a running job reads this global between
    # awaits and must never observe a partially-built region dict.
    regions_update = {}
    for key in ["red_center", "blue_center", "match_clock"]:
        if key in body:
            coords = body[key]
            regions_update[key] = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    custom_ocr_regions = regions_update

    # Persist to database for this event
    if current_event:
        from database import save_ocr_regions
        await save_ocr_regions(current_event['event_key'], custom_ocr_regions)
    
    return {"status": "ok", "regions": custom_ocr_regions}

@app.get("/api/ocr-regions")
async def get_ocr_regions():
    """Get current OCR regions."""
    regions = {**OCR_REGIONS_1080P, **(custom_ocr_regions or {})}
    return {"regions": {k: list(v) for k, v in regions.items()}, "custom": custom_ocr_regions is not None}


@app.post("/api/calibration/zones")
async def set_static_zones(request: Request):
    """Save static-structure zones for an event: field-coordinate rectangles
    (meters) marking known field elements the detector false-positives on. A
    detection inside a zone can never SEED a new robot track.
    Body: {"event": "<key>", "zones": [[xmin, ymin, xmax, ymax], ...]}"""
    body = await request.json()
    event_key = body.get("event") or (current_event['event_key'] if current_event else None)
    if not event_key:
        return JSONResponse(status_code=400,
                            content={"error": "No event — load an event or pass 'event'."})
    rects = []
    for rect in body.get("zones", []):
        if len(rect) != 4:
            return JSONResponse(status_code=400,
                                content={"error": "each zone must be [xmin, ymin, xmax, ymax]"})
        rects.append(tuple(float(v) for v in rect))
    save_zones(event_key, rects)
    return {"status": "ok", "event": event_key, "zones": [list(r) for r in rects]}


@app.get("/api/calibration/zones")
async def get_static_zones(event: Optional[str] = None):
    """Get static-structure zones for an event (falls back to the loaded event)."""
    event_key = event or (current_event['event_key'] if current_event else None)
    if not event_key:
        return {"zones": []}
    return {"event": event_key, "zones": [list(r) for r in load_zones(event_key)]}

@app.post("/api/hub-positions")
async def set_hub_positions(request: Request):
    """Set hub positions for the current event. Static for entire event.
    Body: {"blue": [x, y], "red": [x, y]} in overhead crop coordinates."""
    global event_hub_positions
    body = await request.json()
    event_hub_positions = {}
    for key in ["blue", "red"]:
        if key in body and body[key]:
            event_hub_positions[key] = [int(body[key][0]), int(body[key][1])]
    
    if current_event:
        from database import save_hub_positions
        await save_hub_positions(current_event['event_key'], event_hub_positions)
    
    return {"status": "ok", "positions": event_hub_positions}

@app.get("/api/hub-positions")
async def get_hub_positions_api():
    """Get current hub positions."""
    return {"positions": event_hub_positions, "has_positions": event_hub_positions is not None}

@app.post("/api/grab-frame")
async def grab_frame(request: Request):
    """Grab a single frame from a YouTube video for OCR region setup.
    Auto-picks a video from the loaded event if no URL provided.
    Uses ffmpeg to extract a frame directly from the stream URL — no full download.
    Returns the overhead crop as a base64 JPEG."""
    import subprocess
    
    body = await request.json()
    url = body.get("url", "")
    timestamp = body.get("timestamp", 60)  # Default 60 seconds in (mid-match)
    
    # Auto-pick a YouTube URL from the current event if none provided
    if not url and current_event:
        for m in current_event.get('matches', []):
            if m.get('video_key'):
                url = f"https://www.youtube.com/watch?v={m['video_key']}"
                break
    
    if not url:
        return JSONResponse(status_code=400, content={"error": "No video available — no matches with YouTube URLs found"})
    
    job_id = str(uuid.uuid4())[:8]
    frame_path = UPLOAD_DIR / f"frame_{job_id}.jpg"
    
    try:
        # Get direct stream URL from yt-dlp (no download)
        ydl_opts = {
            # Highest-res video up to 1080p, any codec — ext=mp4 was capping this at
            # ~480p (high-res YouTube streams are VP9/AV1, not mp4). Video-only here
            # since we only extract a single frame via ffmpeg.
            'format': 'bestvideo[height<=1080]/best[height<=1080]/bestvideo/best',
            'format_sort': ['res:1080', 'vcodec:h264'],
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info.get('url')
            if not stream_url:
                # Fallback: get from requested_formats (merged format)
                for fmt in info.get('requested_formats', []):
                    if fmt.get('vcodec') != 'none':
                        stream_url = fmt['url']
                        break
        
        if not stream_url:
            return JSONResponse(status_code=400, content={"error": "Could not resolve video stream URL"})
        
        # Extract single frame with ffmpeg — fast seek to timestamp
        result = subprocess.run([
            'ffmpeg', '-ss', str(timestamp), '-i', stream_url,
            '-frames:v', '1', '-q:v', '2',
            str(frame_path), '-y'
        ], capture_output=True, timeout=30)
        
        if not frame_path.exists():
            stderr = result.stderr.decode('utf-8', errors='replace')[-500:]
            return JSONResponse(status_code=400, content={"error": f"ffmpeg failed to extract frame: {stderr}"})
        
        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Could not read extracted frame"})
        
        # Crop to overhead
        h = frame.shape[0]
        overhead = frame[:int(h * OVERHEAD_CROP_RATIO), :]
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', overhead, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up
        frame_path.unlink(missing_ok=True)
        
        return {
            "frame": frame_b64,
            "width": overhead.shape[1],
            "height": overhead.shape[0]
        }
    except subprocess.TimeoutExpired:
        frame_path.unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": "Frame extraction timed out"})
    except Exception as e:
        frame_path.unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Calibration endpoints
# ---------------------------------------------------------------------------

@app.get("/api/calibration")
async def get_calibration(event: str, camera: str = "main"):
    cal = field_store.load(event, camera)
    return {
        "has_calibration": cal is not None,
        "rms_error": cal.rms_error if cal else None,
        "masks": cal.masks if (cal and hasattr(cal, "masks")) else [],
        "robot_track_top_crop": getattr(cal, "robot_track_top_crop", 0.0) if cal else 0.0,
    }


@app.post("/api/calibration/masks")
async def save_calibration_masks(request: Request):
    """Body: {event, camera?, masks: [[x,y,w,h], ...]}."""
    body = await request.json()
    event = body["event"]
    camera = body.get("camera", "main")
    masks = body.get("masks", [])

    version_id = body.get("version_id")
    versions = field_store.load_versions(event, camera)
    if not versions:
        # No calibration yet — create a default base version (mirrors today's
        # "no calibration -> default" branch).
        cal = CameraCalibration(image_size=(1920, 1080), field_season="2026-rebuilt-welded", mode="pinhole")
        cal.masks = [tuple(m) for m in masks]
        field_store.save_version(event, camera, cal, start_match=None)
        return {"status": "ok", "masks": cal.masks}
    target = next((v for v in versions if v["version_id"] == version_id), None) if version_id else None
    if target is None:
        target = max(versions, key=lambda v: v.get("created_at") or "")
    cal = target["calibration"]
    cal.masks = [tuple(m) for m in masks]
    field_store.update_version(event, camera, target["version_id"], cal)
    return {"status": "ok", "masks": cal.masks}


@app.post("/api/calibration/robot-track-top-crop")
async def save_robot_track_top_crop(request: Request):
    """Body: {event, camera?, robot_track_top_crop: float}.

    Fraction (0.0–0.5) of the top of this view excluded from the SAM2 robot
    tracking pass — robots cannot be in the airspace above the field.  Stored
    per camera/event; preserves any existing calibration (masks, intrinsics).
    """
    body = await request.json()
    event = body["event"]
    camera = body.get("camera", "main")
    crop = max(0.0, min(0.5, float(body.get("robot_track_top_crop", 0.0))))

    version_id = body.get("version_id")
    versions = field_store.load_versions(event, camera)
    if not versions:
        cal = CameraCalibration(image_size=(1920, 1080), field_season="2026-rebuilt-welded", mode="pinhole")
        cal.robot_track_top_crop = crop
        field_store.save_version(event, camera, cal, start_match=None)
        return {"status": "ok", "robot_track_top_crop": cal.robot_track_top_crop}
    target = next((v for v in versions if v["version_id"] == version_id), None) if version_id else None
    if target is None:
        target = max(versions, key=lambda v: v.get("created_at") or "")
    cal = target["calibration"]
    cal.robot_track_top_crop = crop
    field_store.update_version(event, camera, target["version_id"], cal)
    return {"status": "ok", "robot_track_top_crop": cal.robot_track_top_crop}


def _decode_b64_image(b64: str):
    """Base64 (optionally data-URL) JPEG/PNG -> BGR ndarray."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@app.post("/api/calibration/manual")
async def calibration_manual(request: Request):
    """Solve POSE from human-clicked points against saved intrinsics.
    Body: {event, camera?, points: [{u, v, x, y, z}, ...]}."""
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    base = field_store.load(event, camera)   # most-recent cal: intrinsics + lens mode
    if base is None or base.K is None:
        return JSONResponse(status_code=400, content={
            "error": "No saved intrinsics — run auto calibration first."})
    K, D, intr_mode = base.K, np.asarray(base.D).reshape(-1), base.mode
    pts = body.get("points", [])
    if len(pts) < 2:
        return JSONResponse(status_code=400, content={
            "error": "Need at least 2 clicked points."})
    corr = {f"pt{i}": (np.array([[p["u"], p["v"]]], float),
                       np.array([[p["x"], p["y"], p["z"]]], float))
            for i, p in enumerate(pts)}
    start_match = body.get("start_match")
    label = body.get("label")
    prev = select_epoch(field_store.load_versions(event, camera), start_match)
    existing = prev["calibration"] if prev else None
    image_size = existing.image_size if existing else tuple(body.get("image_size", (1920, 1080)))
    season = existing.field_season if existing else body.get("season", "2026-rebuilt-welded")
    try:
        cal = Calibrator.solve_pose(corr, K=K, D=D, image_size=image_size,
                                    field_season=season, mode=intr_mode)
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    if existing is not None:
        cal.masks = existing.masks
        cal.robot_track_top_crop = existing.robot_track_top_crop
    field_store.save_version(event, camera, cal, start_match=start_match, label=label)
    return {"status": "ok", "rms_error": cal.rms_error}


@app.post("/api/calibration/auto")
async def calibration_auto(request: Request):
    """Solve POSE from AprilTags detected in supplied frames, against saved
    intrinsics. Body: {event, camera?, season?, frames: [b64,...], image_size?}.
    Frames are pre-grabbed (e.g. via /api/grab-frame)."""
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    season = body.get("season", "2026-rebuilt-welded")
    base = field_store.load(event, camera)   # most-recent cal: intrinsics + lens mode
    if base is None or base.K is None:
        return JSONResponse(status_code=400, content={
            "error": "No intrinsics — import or run multi-viewpoint "
                     "intrinsics calibration first."})
    K, D, intr_mode = base.K, np.asarray(base.D).reshape(-1), base.mode
    fm = FieldModel.load(season)
    frames = body.get("frames", [])
    if not frames:
        return JSONResponse(status_code=400, content={"error": "No frames supplied."})
    image_size = None
    corr = {}
    for b64 in frames:
        img = _decode_b64_image(b64)
        if img is None:
            continue
        image_size = (img.shape[1], img.shape[0])
        for tid, corners_2d in detect_tags(img).items():
            if tid in fm.tags and tid not in corr:
                # corners_2d: detector order [TR, TL, BL, BR] (Task 3).
                # corners_3d: FieldModel order (Task 2). >>> CORNER-ORDER ALIGNMENT <<<
                # If real-frame RMS is high, fix the index mapping HERE (one place)
                # — validate during Task 10 manual calibration by watching rms_error.
                corr[tid] = (corners_2d, fm.tag_corners_3d(tid))
    try:
        cal = Calibrator.solve_pose(corr, K=K, D=D,
                                    image_size=image_size or (1920, 1080),
                                    field_season=season, mode=intr_mode)
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    start_match = body.get("start_match")
    label = body.get("label")
    prev = select_epoch(field_store.load_versions(event, camera), start_match)
    if prev is not None:
        cal.masks = prev["calibration"].masks
        cal.robot_track_top_crop = prev["calibration"].robot_track_top_crop
    field_store.save_version(event, camera, cal, start_match=start_match, label=label)
    return {"status": "ok", "rms_error": cal.rms_error, "n_tags": len(corr)}


# Stream URLs resolved for calibration frame grabs, keyed by YouTube video id.
# yt-dlp resolution takes seconds; URLs stay valid for hours, cache for 10 min.
_calib_stream_cache: dict = {}


def _match_stream_url(match_key: str):
    """Resolve a direct video stream URL for one of the loaded event's matches.
    Returns (stream_url, None) on success or (None, error_message)."""
    if not current_event:
        return None, "No event loaded — load the event first."
    m = next((mm for mm in current_event.get('matches', [])
              if mm.get('key') == match_key), None)
    if m is None:
        return None, f"Match '{match_key}' is not in the loaded event."
    vk = m.get('video_key')
    if not vk:
        return None, f"Match '{match_key}' has no YouTube video on TBA."
    cached = _calib_stream_cache.get(vk)
    if cached and cached[1] > time.time():
        return cached[0], None
    ydl_opts = {
        'format': 'bestvideo[height<=1080]/best[height<=1080]/bestvideo/best',
        'format_sort': ['res:1080', 'vcodec:h264'],
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vk}",
                                    download=False)
        stream_url = info.get('url')
        if not stream_url:
            for fmt in info.get('requested_formats', []):
                if fmt.get('vcodec') != 'none':
                    stream_url = fmt['url']
                    break
        if not stream_url:
            return None, "Could not resolve a video stream URL for the match."
        _calib_stream_cache[vk] = (stream_url, time.time() + 600)
        return stream_url, None
    except Exception as e:
        return None, f"Stream resolve failed: {e}"


def _read_composite_frame(body: dict):
    """Read the full composite frame a calibration request refers to.
    Source precedence: match_key (the loaded event's YouTube video for that
    match, single-frame ffmpeg extract — needed when recalibrating a moved
    camera against footage from that match) > video_path (local file restricted
    to data/test_footage/) > CALIB_TEST_VIDEO.
    Returns (frame_bgr, None) on success or (None, error_message)."""
    import subprocess
    ts = float(body.get("timestamp", 75))
    match_key = body.get("match_key")
    if match_key:
        stream_url, err = _match_stream_url(match_key)
        if err:
            return None, err
        frame_path = UPLOAD_DIR / f"calib_frame_{uuid.uuid4().hex[:8]}.jpg"
        try:
            subprocess.run(['ffmpeg', '-ss', str(ts), '-i', stream_url,
                            '-frames:v', '1', '-q:v', '2', str(frame_path), '-y'],
                           capture_output=True, timeout=45)
            frame = cv2.imread(str(frame_path)) if frame_path.exists() else None
        except subprocess.TimeoutExpired:
            return None, "Frame extraction from the match video timed out."
        finally:
            frame_path.unlink(missing_ok=True)
        if frame is None:
            return None, "ffmpeg could not extract a frame from the match video."
        return frame, None
    # video_path is restricted to data/test_footage/ with a video extension to
    # prevent arbitrary local-file / URL reads via cv2.VideoCapture.
    raw = body.get("video_path")
    if not raw:
        path = CALIB_TEST_VIDEO
    else:
        allowed_dir = (Path(__file__).resolve().parents[1] / "data/test_footage").resolve()
        candidate = Path(raw).resolve()
        try:
            candidate.relative_to(allowed_dir)
        except ValueError:
            return None, "video_path outside allowed directory"
        if candidate.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
            return None, "unsupported video extension"
        path = candidate
    if not path.exists():
        return None, "video not found"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, "could not read frame"
    return frame, None


@app.get("/api/calibration/versions")
async def get_calibration_versions(event: str, camera: str = "main"):
    """Calibration version metadata for (event, camera), oldest first."""
    versions = field_store.load_versions(event, camera)
    return {"versions": [
        {"version_id": v["version_id"], "start_match": v["start_match"],
         "label": v["label"], "created_at": v["created_at"],
         "rms_error": getattr(v["calibration"], "rms_error", None),
         "mode": getattr(v["calibration"], "mode", None)}
        for v in versions]}


@app.get("/api/calibration/drift")
async def get_calibration_drift(event: str):
    """Failed drift-check records for the event, keyed camera -> match_key.
    Each entry gains ``recalibrated``: True once a DIFFERENT epoch now covers
    that match (a new version was anchored at/before it), so the UI can stop
    flagging it without waiting for the match to be reprocessed."""
    raw = field_store.load_drift(event)
    out = {}
    for camera, entries in raw.items():
        versions = field_store.load_versions(event, camera)
        cam_out = {}
        for mk, rec in entries.items():
            cover = select_epoch(versions, mk)
            # A staged auto-fix awaiting approval lets the UI offer approve/reject
            # for this match instead of only manual recalibration.
            pending = field_store.load_pending_autofix(event, camera, mk)
            cam_out[mk] = {**rec, "recalibrated":
                           bool(cover and cover["version_id"] != rec.get("epoch")),
                           "autofix_pending": pending is not None}
        out[camera] = cam_out
    return {"drift": out}


@app.get("/api/calibration/autofix")
async def get_calibration_autofix(event: str, camera: str = "main",
                                  match: str | None = None):
    """The staged automatic pose re-solve awaiting approval for ``match`` (or None).

    Returned when the in-match drift check recovered a plausible pose by re-solving
    against the fixed lens (see web.field.pose_autofix). The operator reviews it via
    the preview overlay and then approves (promotes to a new epoch) or rejects
    (falls back to manual recalibration)."""
    if not match:
        return {"autofix": None}
    pending = field_store.load_pending_autofix(event, camera, match)
    if pending is None:
        return {"autofix": None}
    return {"autofix": {"summary": pending.get("summary"),
                        "base_epoch": pending.get("base_epoch"),
                        "created_at": pending.get("created_at")}}


@app.post("/api/calibration/autofix/preview")
async def calibration_autofix_preview(request: Request):
    """Render the staged auto-fix over an example frame for human review.

    Body: {event, camera?, match_key, timestamp?}. Draws the reprojected field
    wireframe under the re-solved pose plus every auto-detected AprilTag (its
    detected corners in green, the model tag reprojected under the fix in cyan) so
    a good fix shows green boxes sitting on cyan boxes. Returns {overlay: b64,
    width, height, summary}."""
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    match_key = body.get("match_key")
    pending = field_store.load_pending_autofix(event, camera, match_key)
    if pending is None:
        return JSONResponse(status_code=400, content={
            "error": "no pending auto-fix for this match"})
    cal = pending["calibration"]
    # Read the example frame from the drifted match's footage by default. An explicit
    # video_path (local file / manual override) wins so the preview isn't blocked when
    # the match's remote video is unavailable.
    frame_body = dict(body)
    if frame_body.get("video_path"):
        frame_body.pop("match_key", None)
    else:
        frame_body["match_key"] = match_key
    frame, err = await asyncio.to_thread(_read_composite_frame, frame_body)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    cl = load_layout().get(camera)
    if cl is None:
        return JSONResponse(status_code=400, content={"error": f"unknown camera '{camera}'"})
    cx, cy, cw, ch = cl.crop
    main = frame[cy:cy + ch, cx:cx + cw].copy()
    for (mx, my, mw_, mh_) in (cal.masks or cl.masks):
        main[my:my + mh_, mx:mx + mw_] = 0
    fm = FieldModel.load(cal.field_season or "2026-rebuilt-welded")
    try:
        _draw_field_overlay(main, cal, fm)
        for tid, corners in detect_tags(main).items():
            if tid not in fm.tags:
                continue
            pts = np.asarray(corners, float).reshape(-1, 2)
            cv2.polylines(main, [pts.astype(np.int32)], True, (0, 255, 0), 2)
            # model tag reprojected under the re-solved pose (should hug the green)
            proj = np.array([cal.field_to_image(tuple(p))
                             for p in fm.tag_corners_3d(tid)], float)
            if np.all(np.isfinite(proj)):
                cv2.polylines(main, [proj.astype(np.int32)], True, (255, 255, 0), 1)
    except Exception as e:
        print(f"  [autofix] preview overlay draw failed: {e}")
    okj, buf = cv2.imencode(".jpg", main, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not okj:
        return JSONResponse(status_code=500, content={"error": "jpeg encode failed"})
    return {"overlay": base64.b64encode(buf.tobytes()).decode(),
            "width": int(main.shape[1]), "height": int(main.shape[0]),
            "summary": pending.get("summary")}


@app.post("/api/calibration/autofix/approve")
async def calibration_autofix_approve(request: Request):
    """Adopt the staged auto-fix: promote it to a NEW calibration version anchored
    at the drifted match (so it applies from that match onward, earlier matches
    keep their old epoch), carry the base epoch's masks/crop settings, then clear
    the drift record and the pending fix. Body: {event, camera?, match_key, label?}."""
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    match_key = body.get("match_key")
    pending = field_store.load_pending_autofix(event, camera, match_key)
    if pending is None:
        return JSONResponse(status_code=400, content={
            "error": "no pending auto-fix for this match"})
    cal = pending["calibration"]
    # Carry the covering epoch's manual settings (masks, robot-track crop) so the
    # auto-fix only changes the pose, not the operator's per-camera tuning.
    prev = select_epoch(field_store.load_versions(event, camera), match_key)
    existing = prev["calibration"] if prev else None
    if existing is not None:
        cal.masks = existing.masks
        cal.robot_track_top_crop = existing.robot_track_top_crop
    version_id = field_store.save_version(
        event, camera, cal, start_match=match_key,
        label=body.get("label") or "auto-fix (pose re-solve)",
        source={"autofix": True, "base_epoch": pending.get("base_epoch"),
                "summary": pending.get("summary")})
    if match_key:
        field_store.clear_drift(event, camera, match_key)
        field_store.clear_pending_autofix(event, camera, match_key)
    return {"status": "ok", "version_id": version_id, "start_match": match_key,
            "rms_error": getattr(cal, "rms_error", None)}


@app.post("/api/calibration/autofix/reject")
async def calibration_autofix_reject(request: Request):
    """Discard the staged auto-fix without adopting it. The drift record is LEFT in
    place so the match stays flagged (❌) for manual recalibration. Returns a
    ``redirect`` to the calibrate page pre-scoped to this match/camera.
    Body: {event, camera?, match_key}."""
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    match_key = body.get("match_key")
    field_store.clear_pending_autofix(event, camera, match_key)
    q = f"?match={match_key}&camera={camera}" if match_key else f"?camera={camera}"
    return {"status": "ok", "redirect": f"/calibrate{q}"}


@app.get("/api/calibration/epoch")
async def get_calibration_epoch(event: str, camera: str = "main",
                                match: str | None = None):
    """The calibration version whose epoch covers ``match`` (latest version when
    match is omitted), including the stored solve inputs (``source``) so the
    calibrate page can restore correspondences and re-preview the solve."""
    versions = field_store.load_versions(event, camera)
    v = select_epoch(versions, match or None)
    if v is None:
        return {"version": None}
    return {"version": {
        "version_id": v["version_id"], "start_match": v["start_match"],
        "label": v["label"], "created_at": v["created_at"],
        "rms_error": getattr(v["calibration"], "rms_error", None),
        "mode": getattr(v["calibration"], "mode", None),
        "source": v.get("source")}}


@app.post("/api/calibration/frame")
async def calibration_frame(request: Request):
    """Grab a frame cropped to the requested camera's region from the composite
    layout (masks applied). Body: {timestamp?: seconds (default 75), match_key?,
    video_path?, camera?} — match_key reads the loaded event's video for that
    match (see _read_composite_frame for source precedence).
    Returns {frame: b64 jpeg, width, height} where width/height are the CROPPED size.
    Pixel coords are crop-local (origin at the camera crop's top-left)."""
    body = await request.json()
    frame, err = await asyncio.to_thread(_read_composite_frame, body)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    camera = body.get("camera", "main")
    layout = load_layout()
    cl = layout.get(camera)
    if cl is None:
        return JSONResponse(status_code=400, content={"error": f"unknown camera '{camera}'"})
    cx, cy, cw, ch = cl.crop
    main = frame[cy:cy + ch, cx:cx + cw].copy()
    for (mx, my, mw, mh) in cl.masks:
        main[my:my + mh, mx:mx + mw] = 0
    fm = FieldModel.load("2026-rebuilt-welded")
    det = detect_tags(main)
    detected = {str(tid): det[tid].tolist() for tid in det if tid in fm.tags}
    ok2, buf = cv2.imencode(".jpg", main, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok2:
        return JSONResponse(status_code=500, content={"error": "jpeg encode failed"})
    b64 = base64.b64encode(buf.tobytes()).decode()
    return {"frame": b64, "width": int(main.shape[1]), "height": int(main.shape[0]),
            "detected_tags": detected}


@app.post("/api/calibration/refine-tag")
async def calibration_refine_tag(request: Request):
    """Manual-locate -> auto-refine. Body: {camera?, timestamp?, match_key?, u, v, id}.
    Reads the same source frame the page grabbed (match video / video_path /
    CALIB_TEST_VIDEO), crops to the camera's composite region, runs tag detection
    in an upscaled ROI around (u,v), and returns {corners: [[u,v]x4]} in
    crop-pixel coords, or {corners: null}."""
    from web.field.apriltag_detect import detect_tags_in_roi
    body = await request.json()
    camera = body.get("camera", "main")
    tid = int(body["id"])
    frame, err = await asyncio.to_thread(_read_composite_frame, body)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    cl = load_layout().get(camera)
    if cl is None:
        return JSONResponse(status_code=400, content={"error": f"unknown camera '{camera}'"})
    cx, cy, cw, ch = cl.crop
    crop = frame[cy:cy + ch, cx:cx + cw]
    corners = detect_tags_in_roi(crop, (float(body["u"]), float(body["v"])), tid)
    return {"corners": corners.tolist() if corners is not None else None}


def _draw_field_overlay(frame, cal, fm):
    """Draw the reprojected field, hubs, and bumps using cal.field_to_image.
    Field lines are SAMPLED into polylines (a straight field line projects to a curved
    image line under lens distortion, so a 2-point segment can't trace it).
    Returns fp(x,y)->(int,int) so callers can add their own markers."""
    L, W = fm.length_m, fm.width_m
    H, Wd = frame.shape[:2]
    def fp(x, y):
        u, v = cal.field_to_image((x, y, 0.0))
        return (int(u), int(v))
    def _ok(u, v):
        # A poorly-conditioned calibration can project ground points to enormous
        # (or non-finite) pixel coords — points near/behind the camera blow up.
        # Keep only finite coords within a generous off-screen margin so the int32
        # polyline conversion can't overflow (and we don't draw garbage segments).
        return (np.isfinite(u) and np.isfinite(v)
                and -10000 <= u <= Wd + 10000 and -10000 <= v <= H + 10000)
    def seg3(p0, p1, color, thick, n=24):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        run = []
        def flush():
            if len(run) >= 2:
                cv2.polylines(frame, [np.array(run, np.int32)], False, color, thick)
            run.clear()
        for t in range(n + 1):
            u, v = cal.field_to_image((x0 + (x1 - x0) * t / n,
                                       y0 + (y1 - y0) * t / n,
                                       z0 + (z1 - z0) * t / n))
            if _ok(u, v):
                run.append((int(u), int(v)))
            else:
                flush()   # break the polyline across the blown-up region
        flush()
    def seg(p0, p1, color, thick, n=24):
        seg3((p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0),
             color, thick, n)
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0))]:
        seg(a, b, (0, 255, 0), 2)
    x = 0.0
    while x <= L + 1e-6:
        seg((x, 0), (x, W), (0, 180, 0), 1); x += 2.0
    y = 0.0
    while y <= W + 1e-6:
        seg((0, y), (L, y), (0, 180, 0), 1); y += 2.0
    prisms = hub_prisms(fm)
    for a, b in hub_wireframe_edges(prisms):
        seg3(a, b, (255, 220, 0), 2, n=12)
    for a, b in bump_wireframe_edges(prisms, W):
        seg3(a, b, (0, 140, 255), 2, n=8)
    for a, b in tower_wireframe_edges(fm):
        seg3(a, b, (255, 0, 255), 2, n=8)
    for a, b in trench_wireframe_edges(fm):
        seg3(a, b, (180, 80, 255), 2, n=8)
    return fp


@app.post("/api/calibration/solve-homography")
async def calibration_solve_homography(request: Request):
    """Body: {event?, camera?, timestamp?, image_size:[w,h], points:[{u,v,x,y},...]}.
    Fits a ground-plane homography from clicked (pixel<->field-xy) correspondences,
    saves it, and returns RMS + an overlay JPEG (reprojected field perimeter + 2 m
    grid drawn on the same frame) for visual verification."""
    body = await request.json()
    event = body.get("event", "test"); camera = body.get("camera", "main")
    pts = body.get("points", [])
    if len(pts) < 4:
        return JSONResponse(status_code=400, content={"error": "need >= 4 points"})
    img_pts = np.array([[p["u"], p["v"]] for p in pts], float)
    fld_xy = np.array([[p["x"], p["y"]] for p in pts], float)
    image_size = tuple(body.get("image_size", [1920, 1080]))
    try:
        cal = Calibrator.solve_ground_homography(img_pts, fld_xy, image_size,
                                                 "2026-rebuilt-welded")
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    start_match = body.get("start_match")
    label = body.get("label")
    prev = select_epoch(field_store.load_versions(event, camera), start_match)
    existing = prev["calibration"] if prev else None
    if existing is not None:
        cal.masks = existing.masks
        cal.robot_track_top_crop = existing.robot_track_top_crop
    field_store.save_version(event, camera, cal, start_match=start_match, label=label)

    # Build overlay on the same frame/timestamp (and same source video).
    fm = FieldModel.load("2026-rebuilt-welded")
    frame, _frame_err = await asyncio.to_thread(_read_composite_frame, body)
    overlay_b64 = None
    cl = load_layout().get(camera)
    if frame is not None and cl is not None:
        cx, cy, cw, ch = cl.crop
        main = frame[cy:cy + ch, cx:cx + cw].copy()
        for (mx, my, mw_, mh_) in cl.masks:
            main[my:my + mh_, mx:mx + mw_] = 0
        _draw_field_overlay(main, cal, fm)
        # clicked points
        for p in pts:
            cv2.circle(main, (int(p["u"]), int(p["v"])), 5, (0,0,255), -1)
        okj, buf = cv2.imencode(".jpg", main, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if okj:
            overlay_b64 = base64.b64encode(buf.tobytes()).decode()
    return {"status": "ok", "rms_error": cal.rms_error, "n_points": len(pts),
            "overlay": overlay_b64}


def _tag_id_of(key: str) -> int:
    return int(key[1:]) if key[:1] in ("a", "m") else int(key)


@app.post("/api/calibration/solve-tags")
async def calibration_solve_tags(request: Request):
    """Body: {event?, camera?, season?, image_size:[w,h], timestamp?,
    auto_tags:{id:[[u,v]x4]}, manual_tags:[{id,u,v}]}.
    Solves a single-view pinhole camera model from AprilTag correspondences
    (auto -> 4 corners, manual -> center), saves it, and returns RMS + overlay."""
    body = await request.json()
    event = body.get("event", "test"); camera = body.get("camera", "main")
    season = body.get("season", "2026-rebuilt-welded")
    image_size = tuple(body.get("image_size", [1920, 648]))
    fm = FieldModel.load(season)
    auto = body.get("auto_tags", {}) or {}
    manual = body.get("manual_tags", []) or []
    corr = {}
    for sid, corners in auto.items():
        tid = int(sid)
        if tid in fm.tags:
            corr[f"a{tid}"] = (np.array(corners, float).reshape(-1, 2),
                               fm.tag_corners_3d(tid))
    for m in manual:
        tid = int(m["id"])
        if tid in fm.tags:
            center3d = fm.tag_corners_3d(tid).mean(axis=0).reshape(1, 3)
            corr[f"m{tid}"] = (np.array([[float(m["u"]), float(m["v"])]], float),
                               center3d)
    holdout_ids = set(int(i) for i in body.get("holdout_ids", []))
    held = {k: v for k, v in corr.items() if _tag_id_of(k) in holdout_ids}
    solve_corr = {k: v for k, v in corr.items() if _tag_id_of(k) not in holdout_ids}
    try:
        cal = Calibrator.solve_camera_from_tags(solve_corr, image_size=image_size,
                                                field_season=season)
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    start_match = body.get("start_match")
    label = body.get("label")
    prev = select_epoch(field_store.load_versions(event, camera), start_match)
    existing = prev["calibration"] if prev else None
    if existing is not None:
        cal.masks = existing.masks
        cal.robot_track_top_crop = existing.robot_track_top_crop
    # dry_run solves + renders the overlay without persisting — used by the
    # calibrate page to re-preview a saved epoch's stored correspondences.
    if not body.get("dry_run"):
        field_store.save_version(
            event, camera, cal, start_match=start_match, label=label,
            source={"auto_tags": auto, "manual_tags": manual,
                    "holdout_ids": sorted(holdout_ids),
                    "timestamp": float(body.get("timestamp", 75)),
                    "match_key": body.get("match_key")})
    holdout_err = holdout_field_error(cal, held) if held else None

    used = {int(s) for s in auto.keys()} | {int(m["id"]) for m in manual}
    n_points = sum(len(c[0]) for c in corr.values())
    overlay_b64 = None
    frame, _frame_err = await asyncio.to_thread(_read_composite_frame, body)
    cl = load_layout().get(camera)
    # Overlay is a best-effort visualization. A poorly-conditioned calibration can
    # make projection blow up; never let drawing failures sink the solve result
    # (the caller still needs rms_error + holdout_error_m to judge the calibration).
    try:
        if frame is not None and cl is not None:
            cx, cy, cw, ch = cl.crop
            main = frame[cy:cy + ch, cx:cx + cw].copy()
            for (mx, my, mw_, mh_) in cl.masks:
                main[my:my + mh_, mx:mx + mw_] = 0
            _draw_field_overlay(main, cal, fm)
            mh, mw = main.shape[:2]
            for tid in fm.tags:
                t3 = fm.tag_translation(tid)
                u, v = cal.field_to_image((float(t3[0]), float(t3[1]), float(t3[2])))
                if 0 <= u < mw and 0 <= v < mh:
                    color = (0, 255, 0) if tid in used else (0, 165, 255)
                    cv2.circle(main, (int(u), int(v)), 5, color, -1)
                    cv2.putText(main, str(tid), (int(u) + 4, int(v) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            okj, buf = cv2.imencode(".jpg", main, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if okj:
                overlay_b64 = base64.b64encode(buf.tobytes()).decode()
    except Exception as e:
        print(f"  calibration overlay render failed ({camera}): {e}")
    return {"status": "ok", "rms_error": cal.rms_error, "n_tags": len(corr),
            "n_points": n_points, "overlay": overlay_b64,
            "holdout_error_m": holdout_err}


@app.get("/api/calibration/landmarks")
async def calibration_landmarks():
    import json
    p = Path(__file__).resolve().parents[1] / "data/field/2026_landmarks.json"
    return json.loads(p.read_text())


@app.get("/api/calibration/tag-map")
async def calibration_tag_map(season: str = "2026-rebuilt-welded"):
    """Top-down field tag layout for the calibration UI: tag (x,y,z) + field dims."""
    fm = FieldModel.load(season)
    tags = {str(tid): {"x": float(t["t"][0]), "y": float(t["t"][1]),
                       "z": float(t["t"][2])}
            for tid, t in fm.tags.items()}
    return {"tags": tags, "field": {"length": fm.length_m, "width": fm.width_m}}


def _vision_read_bumpers(robot_crops_b64: list, match_teams: list,
                          blue_teams: list, red_teams: list,
                          expected_alliances: Optional[list] = None,
                          provider: str = "local") -> list:
    """Read bumper numbers using best available method.
    provider="local" uses Qwen VL on the local GPU. provider="deepseek" uses
    local OCR digits plus DeepSeek V4 text/chat roster normalization."""
    print(f"  Team reader [{provider}]: sending {len(robot_crops_b64)} bumper crops")
    if provider == "deepseek":
        return _deepseek_read_bumpers(
            robot_crops_b64, match_teams, blue_teams, red_teams,
            expected_alliances=expected_alliances,
        )
    if provider == "vllm":
        try:
            return _vllm_read_bumpers(
                robot_crops_b64, match_teams, blue_teams, red_teams,
                expected_alliances=expected_alliances,
            )
        except Exception as e:
            print(f"  vLLM provider failed ({e!r}); falling back to in-process Qwen")
    try:
        return _qwen_vl_read_bumpers(
            robot_crops_b64, match_teams, blue_teams, red_teams,
            expected_alliances=expected_alliances,
        )
    except Exception as e:
        print(f"  Qwen VL failed: {e}")
        return [{
            "team_number": None,
            "alliance": "unknown",
            "confidence": 0,
            "raw_read": None,
            "error": str(e),
        }] * len(robot_crops_b64)

    # Claude Vision fallback (disabled for now)
    # return _claude_vision_read_bumpers(robot_crops_b64, match_teams, blue_teams, red_teams)


def _claude_vision_read_bumpers(robot_crops_b64: list, match_teams: list,
                                 blue_teams: list, red_teams: list) -> list:
    """Use Claude Vision to read bumper numbers from robot crop images.

    Sends all crops in a single API call for efficiency.
    Returns list of {"team_number": str|None, "alliance": str, "confidence": float}
    in the same order as input crops.

    Falls back gracefully if API key is missing or call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        return [{"team_number": None, "alliance": "unknown", "confidence": 0}] * len(robot_crops_b64)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build the message with all robot crop images
        content = []
        for i, crop_b64 in enumerate(robot_crops_b64):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": crop_b64,
                },
            })
            content.append({
                "type": "text",
                "text": f"Robot #{i + 1}",
            })

        teams_str = ", ".join(match_teams)
        blue_str = ", ".join(blue_teams)
        red_str = ", ".join(red_teams)

        content.append({
            "type": "text",
            "text": (
                f"These are cropped images of FRC (FIRST Robotics Competition) robots from an overhead camera. "
                f"Each robot has a bumper with a team number printed on it in white text. "
                f"The bumpers are at the bottom/perimeter of the robot and are either blue or red.\n\n"
                f"The ONLY possible teams are: {teams_str}\n"
                f"Blue alliance: {blue_str}\n"
                f"Red alliance: {red_str}\n\n"
                f"IMPORTANT RULES:\n"
                f"- Each team number can only appear ONCE across all robots (no duplicates)\n"
                f"- The bumper color tells you the alliance: blue bumpers = blue alliance team, red bumpers = red alliance team\n"
                f"- Even partial number reads help — match against the team list\n"
                f"- If the robot body is painted blue/red, ignore that — only the bumper color matters\n\n"
                f"For each robot image, identify:\n"
                f"1. The team number on the bumper (must be one of the teams listed above)\n"
                f"2. The bumper color (blue or red)\n\n"
                f"Respond with ONLY a JSON array, one entry per robot in order. Example:\n"
                f'[{{"team": "3128", "color": "blue"}}, {{"team": "4414", "color": "red"}}, ...]\n'
                f"If you cannot read a number, use null for team. No other text."
            ),
        })

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )

        # Parse response
        response_text = response.content[0].text.strip()
        # Extract JSON array from response (handle markdown code blocks)
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        results_raw = json.loads(response_text)

        results = []
        for i, entry in enumerate(results_raw):
            team = entry.get("team")
            color = entry.get("color", "unknown")
            if team is not None:
                team = str(team)
            # Validate team is in match
            confidence = 0.0
            if team and team in match_teams:
                confidence = 0.95
            elif team:
                # Claude returned a number not in our list — try to match
                team = None
                confidence = 0.0
            results.append({
                "team_number": team,
                "alliance": color if color in ("blue", "red") else "unknown",
                "confidence": confidence,
            })

        # Pad if Claude returned fewer results than crops
        while len(results) < len(robot_crops_b64):
            results.append({"team_number": None, "alliance": "unknown", "confidence": 0})

        print(f"  Claude Vision: identified {sum(1 for r in results if r['team_number'])}/"
              f"{len(robot_crops_b64)} robots: {results}")
        return results

    except Exception as e:
        print(f"  Claude Vision error: {e}")
        return [{"team_number": None, "alliance": "unknown", "confidence": 0}] * len(robot_crops_b64)


# Bumper-number reading VLM (VLM_MODEL_NAME, default Qwen3-VL-8B) — lazy-loaded, cached
_qwen_vl_model = None
_qwen_vl_processor = None


def _unload_qwen_vl() -> bool:
    """Drop the cached in-process bumper reader and return its VRAM to the pool.

    The local reader holds ~8GB for the life of the server, which pushes
    SAM3's offload heuristic below its 16GB-free threshold. Phase
    exclusivity calls this before tracking; the next read pass re-loads
    lazily via _get_qwen_vl(). Returns True when a model was unloaded.
    """
    global _qwen_vl_model, _qwen_vl_processor
    if _qwen_vl_model is None and _qwen_vl_processor is None:
        return False
    _qwen_vl_model = None
    _qwen_vl_processor = None
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def _get_qwen_vl():
    """Lazy-load the bumper-reading VLM (VLM_MODEL_NAME) on first use; cached after.

    Uses AutoModelForImageTextToText so the model is swappable via VLM_MODEL_NAME
    (Qwen3-VL today; InternVL3 / Nemotron later) with no code change here.
    """
    global _qwen_vl_model, _qwen_vl_processor
    if _qwen_vl_model is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"Loading VLM for bumper reading: {VLM_MODEL_NAME} ...")
        _qwen_vl_model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_NAME,
            dtype=torch.float16,
            device_map="auto",
        )
        # min/max_pixels are Qwen-family processor kwargs; fall back if unsupported.
        try:
            _qwen_vl_processor = AutoProcessor.from_pretrained(
                VLM_MODEL_NAME,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
        except TypeError:
            _qwen_vl_processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
        # Batched generation on a decoder-only model requires LEFT padding —
        # right-padded rows would generate from pad tokens.
        try:
            _qwen_vl_processor.tokenizer.padding_side = "left"
        except AttributeError:
            pass
        print(f"VLM loaded: {VLM_MODEL_NAME} ({torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM)")
    return _qwen_vl_model, _qwen_vl_processor


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two short strings (for fuzzy team-number matching)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _closest_team(read: str, teams: list):
    """Closest candidate team to the read digits. Returns (team, distance)."""
    best, best_d = None, 99
    for t in teams:
        d = _levenshtein(read, t)
        if d < best_d:
            best, best_d = t, d
    return best, best_d


def _match_bumper_read_to_roster(
    read: str,
    model_color: str,
    expected_alliance: str,
    match_teams: list,
    blue_teams: list,
    red_teams: list,
):
    """Validate a VLM bumper read against the current match roster.

    Exact roster reads are definitive and imply alliance from TBA for this
    match. Partial/fuzzy reads are constrained by the observed bumper alliance
    when available, otherwise by the model-reported bumper color.
    """
    digits = "".join(ch for ch in str(read or "") if ch.isdigit())
    if not digits:
        return None, "unknown", 0.0

    roster_alliance = {}
    roster_alliance.update({str(t): "blue" for t in blue_teams})
    roster_alliance.update({str(t): "red" for t in red_teams})
    all_teams = [str(t) for t in match_teams]

    # A clear full team read wins over color. Teams change alliance match to
    # match, so the current TBA roster is the source of truth for alliance.
    if digits in all_teams:
        return digits, roster_alliance.get(digits, "unknown"), 0.95

    alliance_hint = expected_alliance if expected_alliance in ("blue", "red") else model_color
    if alliance_hint == "red":
        candidates = [str(t) for t in red_teams]
    elif alliance_hint == "blue":
        candidates = [str(t) for t in blue_teams]
    else:
        candidates = all_teams

    substring_matches = [t for t in candidates if digits in t]
    if len(substring_matches) == 1:
        team = substring_matches[0]
        return team, roster_alliance.get(team, alliance_hint), 0.6

    cand, dist = _closest_team(digits, candidates)
    if cand and dist <= 1 and len(digits) >= 2:
        return cand, roster_alliance.get(cand, alliance_hint), 0.6
    return None, alliance_hint if alliance_hint in ("blue", "red") else "unknown", 0.0


def _thumbnail_bytes(crop_b64, max_h=128):
    """Decode a raw-base64 JPEG bumper crop and return a small JPEG thumbnail (bytes),
    bounded to max_h pixels tall. Returns None on bad/empty input."""
    if not crop_b64:
        return None
    try:
        raw = base64.b64decode(crop_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.height > max_h:
            w = max(1, int(img.width * max_h / img.height))
            img = img.resize((w, max_h))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=80)
        return out.getvalue()
    except Exception:
        return None


def _ocr_digits_from_crop_b64(crop_b64: str) -> str:
    try:
        raw = base64.b64decode(crop_b64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = max(2, int(np.ceil(96 / max(1, gray.shape[0]))))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(thresh, config=config)
        return "".join(ch for ch in text if ch.isdigit())
    except Exception as e:
        print(f"  DeepSeek OCR pre-read failed: {e}")
        return ""


def _deepseek_chat_json(prompt: str) -> dict:
    api_key = os.environ.get("DEEPSEEK_API", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API is not set")
    payload = {
        "model": os.environ.get("DEEPSEEK_TEAM_READER_MODEL", "deepseek-v4-flash"),
        "messages": [
            {
                "role": "system",
                "content": "Return only compact JSON. Do not include markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])
    return json.loads(content)


def _deepseek_read_bumpers(robot_crops_b64: list, match_teams: list,
                           blue_teams: list, red_teams: list,
                           expected_alliances: Optional[list] = None) -> list:
    """Cloud team-number reader using DeepSeek V4 text API.

    DeepSeek V4 official API docs do not document image input, so this provider
    does local OCR on the bumper crop and asks DeepSeek to normalize the digits
    against the current match roster and bumper-alliance hint.
    """
    expected_alliances = expected_alliances or ["unknown"] * len(robot_crops_b64)
    results = []
    for i, crop_b64 in enumerate(robot_crops_b64):
        read = _ocr_digits_from_crop_b64(crop_b64)
        expected = (
            expected_alliances[i]
            if i < len(expected_alliances) and expected_alliances[i] in ("blue", "red")
            else "unknown"
        )
        prompt = (
            "An OCR pass read digits from one FRC robot bumper crop.\n"
            f"OCR digits: {read or 'none'}\n"
            f"Observed bumper alliance: {expected}\n"
            f"All match teams: {', '.join(match_teams)}\n"
            f"Blue alliance teams: {', '.join(blue_teams)}\n"
            f"Red alliance teams: {', '.join(red_teams)}\n\n"
            "If the OCR digits exactly or partially identify one current-match team, return it. "
            "Use observed bumper alliance to disambiguate partial reads, but an exact current-match "
            "team number is authoritative. If no team is supported, return null.\n"
            'JSON schema: {"team": string|null, "confidence": number, "reason": string}'
        )
        try:
            parsed = _deepseek_chat_json(prompt)
            raw_team = parsed.get("team")
            raw_team = str(raw_team) if raw_team is not None else ""
            matched, alliance, base_conf = _match_bumper_read_to_roster(
                raw_team or read,
                model_color="unknown",
                expected_alliance=expected,
                match_teams=match_teams,
                blue_teams=blue_teams,
                red_teams=red_teams,
            )
            model_conf = float(parsed.get("confidence", base_conf) or 0.0)
            confidence = max(base_conf, min(0.95, model_conf))
            results.append({
                "team_number": matched,
                "alliance": alliance,
                "confidence": confidence if matched else 0.0,
                "raw_read": read,
            })
        except Exception as e:
            print(f"  DeepSeek team reader failed: {e}")
            results.append({
                "team_number": None,
                "alliance": expected,
                "confidence": 0.0,
                "raw_read": read,
            })
    identified = sum(1 for r in results if r["team_number"])
    print(f"  DeepSeek team reader: identified {identified}/{len(robot_crops_b64)} robots")
    return results


def _qwen_vl_read_bumpers(robot_crops_b64: list, match_teams: list,
                           blue_teams: list, red_teams: list,
                           expected_alliances: Optional[list] = None) -> list:
    """Use the local VLM (VLM_MODEL_NAME, default Qwen3-VL) to read bumper numbers from robot crops.

    All due crops go through ONE padded generate call (VLM_BATCH_READS) —
    prompts differ only in the alliance hint, so they batch cleanly; a batch
    failure falls back to per-crop calls. With VLM_LOGPROB_CONFIDENCE, the
    model's per-digit token probability is folded into the roster-match
    confidence (blend_read_confidence) so a wavering read loses vote weight.

    Returns list of {"team_number": str|None, "alliance": str,
    "confidence": float, "raw_read": str|None, "digit_prob": float|None}.
    """
    import torch
    try:
        model, processor = _get_qwen_vl()
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        print(f"  Qwen VL load error: {e}")
        return [{"team_number": None, "alliance": "unknown", "confidence": 0}] * len(robot_crops_b64)

    teams_str = ", ".join(match_teams)
    expected_alliances = expected_alliances or ["unknown"] * len(robot_crops_b64)

    def _error_entry(e):
        return {"team_number": None, "alliance": "unknown", "confidence": 0,
                "raw_read": None, "error": str(e), "surface": "unknown"}

    # ---- Phase 1: one (messages, context) item per decodable crop ----
    results: list = [None] * len(robot_crops_b64)
    items = []  # (idx, messages, expected_alliance, candidate_str, img_size)
    for idx, crop_b64 in enumerate(robot_crops_b64):
        try:
            img_bytes = base64.b64decode(crop_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Aggressively upscale small crops so Qwen can read the numbers
            min_dim = 512
            if img.width < min_dim or img.height < min_dim:
                scale = max(min_dim / img.width, min_dim / img.height)
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

            expected_alliance = (
                expected_alliances[idx]
                if idx < len(expected_alliances) and expected_alliances[idx] in ("blue", "red")
                else "unknown"
            )
            candidate_teams = (
                red_teams if expected_alliance == "red"
                else blue_teams if expected_alliance == "blue"
                else match_teams
            )
            candidate_str = ", ".join(candidate_teams)
            hint = (
                f"Computer vision sees {expected_alliance.upper()} bumper pixels, so partial or uncertain reads "
                f"should be matched only against these {expected_alliance} alliance teams: {candidate_str}.\n"
                if expected_alliance in ("blue", "red")
                else f"Possible current-match teams are: {teams_str}.\n"
            )
            prompt = (
                "This image shows ONE FRC robot from a side or elevated camera angle. "
                "Around the base of the robot is a bumper: a solid RED or BLUE band with a "
                "WHITE team number printed on it. FRC team numbers are 1 to 5 digits "
                "(for example 294, 2485, 4322).\n\n"
                f"{hint}"
                "Carefully read the WHITE digits on the bumper, left to right. Report the "
                "exact number you SEE. Do not use robot paint or mechanisms for alliance color; "
                "only the bumper band matters. If the digits are clear, report them even if the "
                "color looks surprising.\n\n"
                'Reply with ONLY compact JSON and nothing else: '
                '{"team": "<the digits, or null if you cannot read them>", '
                '"color": "red" or "blue", '
                '"surface": "bumper" if the digits are on a robot\'s bumper band, '
                'or "printed" if they are printed on a crate, bin, sign, or other '
                'field structure}.'
            )
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}
            ]
            items.append((idx, messages, expected_alliance, candidate_str, img.size))
        except Exception as e:
            print(f"  Qwen VL error on crop: {e}")
            results[idx] = _error_entry(e)

    # ---- Phase 2: one batched generate (per-crop fallback on failure) ----
    def _generate(batch):
        """batch: list of phase-1 items -> list of (output_text, digit_prob)."""
        msgs = [m for (_i, m, _a, _c, _s) in batch]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=64,
                return_dict_in_generate=True,
                output_scores=VLM_LOGPROB_CONFIDENCE,
            )
        # Left padding makes the prompt length uniform across rows.
        in_len = inputs.input_ids.shape[1]
        trimmed = gen.sequences[:, in_len:]
        outs = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        digit_probs = [None] * len(batch)
        if VLM_LOGPROB_CONFIDENCE and gen.scores:
            tok = processor.tokenizer
            stop_ids = {tid for tid in (tok.eos_token_id, tok.pad_token_id)
                        if tid is not None}
            for b in range(len(batch)):
                probs, tok_texts = [], []
                for t, step_logits in enumerate(gen.scores):
                    if t >= trimmed.shape[1]:
                        break
                    tid = int(trimmed[b, t])
                    if tid in stop_ids:
                        break
                    probs.append(
                        torch.softmax(step_logits[b].float(), dim=-1)[tid].item())
                    tok_texts.append(tok.decode([tid]))
                digit_probs[b] = digit_token_conf(probs, tok_texts)
        return [(o.strip(), dp) for o, dp in zip(outs, digit_probs)]

    decoded: list = [None] * len(items)
    if items:
        batched = VLM_BATCH_READS and len(items) > 1
        if batched:
            try:
                decoded = _generate(items)
            except Exception as e:
                print(f"  Qwen VL batched generate failed ({e!r}); falling back to per-crop")
        if any(d is None for d in decoded):
            for k, item in enumerate(items):
                if decoded[k] is not None:
                    continue
                try:
                    decoded[k] = _generate([item])[0]
                except Exception as e:
                    print(f"  Qwen VL error on crop: {e}")
                    results[item[0]] = _error_entry(e)

    # ---- Phase 3: parse + roster-match each output ----
    for item, out in zip(items, decoded):
        idx, _messages, expected_alliance, candidate_str, img_size = item
        if out is None:
            if results[idx] is None:
                results[idx] = _error_entry("no VLM output")
            continue
        output_text, digit_prob = out
        print(f"    Qwen input image: {img_size}, raw output: {output_text}")

        team = None
        color = "unknown"
        surface = "unknown"
        try:
            clean = output_text
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(clean[start:end])
                team = parsed.get("team")
                color = parsed.get("color", "unknown")
                surface = normalize_surface(parsed.get("surface"))
                if team is not None:
                    team = str(team)
        except (json.JSONDecodeError, Exception):
            pass

        print(f"    Qwen parsed: team={team}, color={color}, "
              f"expected={expected_alliance}, candidates={candidate_str}, "
              f"digit_prob={None if digit_prob is None else round(digit_prob, 3)}")

        read = "".join(ch for ch in team if ch.isdigit()) if team else ""
        model_color = color if color in ("blue", "red") else "unknown"
        matched, alliance, confidence = _match_bumper_read_to_roster(
            read, model_color, expected_alliance,
            match_teams, blue_teams, red_teams,
        )
        confidence = blend_read_confidence(confidence, digit_prob)
        if read and not matched:
            print(f"    Qwen read '{read}' — no roster match for alliance hint {expected_alliance}/{model_color}")

        results[idx] = {
            "team_number": matched,
            "alliance": alliance,
            "confidence": confidence,
            "raw_read": read,
            "digit_prob": digit_prob,
            "surface": surface,
        }

    results = [r if r is not None else _error_entry("skipped") for r in results]
    identified = sum(1 for r in results if r["team_number"])
    print(f"  Qwen VL: identified {identified}/{len(robot_crops_b64)} robots: "
          f"{[r['team_number'] for r in results]}")
    return results


_BUMPER_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "team": {"type": ["string", "null"]},
        "color": {"type": "string", "enum": ["red", "blue"]},
        "surface": {"type": "string", "enum": ["bumper", "printed", "unclear"]},
    },
    "required": ["team", "color"],
}

_OBJECT_CLASS_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string",
                   "enum": ["robot", "equipment", "person", "unclear"]},
    },
    "required": ["object"],
}


def _vlm_chat_payload(data_uri: str, prompt: str, *, max_tokens: int,
                      logprobs: bool = False, schema_name: str | None = None,
                      schema: dict | None = None, guided: bool | None = None) -> dict:
    """One vLLM chat/completions payload: image + text, optional logprobs and
    structured output. `guided=None` defers to the VLM_GUIDED_JSON kill-switch."""
    if guided is None:
        guided = VLM_GUIDED_JSON
    payload = {
        "model": VLM_MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
    if guided and schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name or "reply", "schema": schema},
        }
    return payload


def _post_vlm_chat(payload: dict) -> dict:
    """POST one chat payload to the vLLM server, returning the parsed body.

    If the server 400s a structured-output payload (older build without
    json_schema support), retry ONCE without response_format — a degraded
    read beats a dead one."""
    def _post(p: dict) -> dict:
        req = urllib.request.Request(
            f"{VLM_SERVER_URL}/chat/completions",
            data=json.dumps(p).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    try:
        return _post(payload)
    except urllib.error.HTTPError as e:
        if e.code == 400 and "response_format" in payload:
            retry = {k: v for k, v in payload.items() if k != "response_format"}
            print("  vLLM rejected response_format (400); retrying unguided")
            return _post(retry)
        raise


def _vllm_read_bumpers(robot_crops_b64: list, match_teams: list,
                       blue_teams: list, red_teams: list,
                       expected_alliances: Optional[list] = None) -> list:
    """Read bumper numbers via a vLLM OpenAI-compatible server (VLM_SERVER_URL).

    Same model and prompt as _qwen_vl_read_bumpers, but crops go out as
    concurrent HTTP requests and the server's continuous batching replaces the
    hand-rolled padded-generate batch. Per-token logprobs come back in the
    response, so digit_token_conf/blend_read_confidence apply unchanged.
    """
    teams_str = ", ".join(match_teams)
    expected_alliances = expected_alliances or ["unknown"] * len(robot_crops_b64)

    # Pre-flight reachability probe. The per-crop reads below catch their own
    # exceptions and return error entries, so a dead server would otherwise
    # yield a full batch of "connection refused" reads and the caller's
    # fallback (try/except around this function) would never fire. Raising here
    # on an unreachable server lets _vision_read_bumpers degrade to the
    # in-process Qwen reader — slower, but a live match instead of a dead one.
    try:
        probe = urllib.request.Request(f"{VLM_SERVER_URL}/models", method="GET")
        with urllib.request.urlopen(probe, timeout=3):
            pass
    except Exception as e:
        raise RuntimeError(
            f"vLLM server unreachable at {VLM_SERVER_URL}: {e}"
        ) from e

    def _error_entry(e):
        return {"team_number": None, "alliance": "unknown", "confidence": 0,
                "raw_read": None, "error": str(e), "surface": "unknown"}

    def _read_one(idx: int, crop_b64: str) -> dict:
        img_bytes = base64.b64decode(crop_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Same upscale as the in-process path so the model sees identical pixels.
        min_dim = 512
        if img.width < min_dim or img.height < min_dim:
            scale = max(min_dim / img.width, min_dim / img.height)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        expected_alliance = (
            expected_alliances[idx]
            if idx < len(expected_alliances) and expected_alliances[idx] in ("blue", "red")
            else "unknown"
        )
        candidate_teams = (
            red_teams if expected_alliance == "red"
            else blue_teams if expected_alliance == "blue"
            else match_teams
        )
        candidate_str = ", ".join(candidate_teams)
        hint = (
            f"Computer vision sees {expected_alliance.upper()} bumper pixels, so partial or uncertain reads "
            f"should be matched only against these {expected_alliance} alliance teams: {candidate_str}.\n"
            if expected_alliance in ("blue", "red")
            else f"Possible current-match teams are: {teams_str}.\n"
        )
        prompt = (
            "This image shows ONE FRC robot from a side or elevated camera angle. "
            "Around the base of the robot is a bumper: a solid RED or BLUE band with a "
            "WHITE team number printed on it. FRC team numbers are 1 to 5 digits "
            "(for example 294, 2485, 4322).\n\n"
            f"{hint}"
            "Carefully read the WHITE digits on the bumper, left to right. Report the "
            "exact number you SEE. Do not use robot paint or mechanisms for alliance color; "
            "only the bumper band matters. If the digits are clear, report them even if the "
            "color looks surprising.\n\n"
            'Reply with ONLY compact JSON and nothing else: '
            '{"team": "<the digits, or null if you cannot read them>", '
            '"color": "red" or "blue", '
            '"surface": "bumper" if the digits are on a robot\'s bumper band, '
            'or "printed" if they are printed on a crate, bin, sign, or other '
            'field structure}.'
        )
        payload = _vlm_chat_payload(
            data_uri, prompt, max_tokens=64,
            logprobs=VLM_LOGPROB_CONFIDENCE,
            schema_name="bumper_read", schema=_BUMPER_READ_SCHEMA,
        )
        body = _post_vlm_chat(payload)
        choice = body["choices"][0]
        output_text = (choice["message"]["content"] or "").strip()

        digit_prob = None
        lp_content = ((choice.get("logprobs") or {}).get("content")) or []
        if VLM_LOGPROB_CONFIDENCE and lp_content:
            probs = [math.exp(t["logprob"]) for t in lp_content]
            tok_texts = [t.get("token", "") for t in lp_content]
            digit_prob = digit_token_conf(probs, tok_texts)

        team = None
        color = "unknown"
        surface = "unknown"
        try:
            clean = output_text
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(clean[start:end])
                team = parsed.get("team")
                color = parsed.get("color", "unknown")
                surface = normalize_surface(parsed.get("surface"))
                if team is not None:
                    team = str(team)
        except (json.JSONDecodeError, Exception):
            pass

        read = "".join(ch for ch in team if ch.isdigit()) if team else ""
        model_color = color if color in ("blue", "red") else "unknown"
        matched, alliance, confidence = _match_bumper_read_to_roster(
            read, model_color, expected_alliance,
            match_teams, blue_teams, red_teams,
        )
        confidence = blend_read_confidence(confidence, digit_prob)
        print(f"    vLLM read: raw={output_text!r} -> team={matched}, "
              f"digit_prob={None if digit_prob is None else round(digit_prob, 3)}")
        return {
            "team_number": matched,
            "alliance": alliance,
            "confidence": confidence,
            "raw_read": read,
            "digit_prob": digit_prob,
            "surface": surface,
        }

    results: list = [None] * len(robot_crops_b64)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(robot_crops_b64), VLM_TOP_K)) as pool:
        futures = {pool.submit(_read_one, i, c): i
                   for i, c in enumerate(robot_crops_b64)}
        for fut, i in futures.items():
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"  vLLM read error on crop {i}: {e}")
                results[i] = _error_entry(e)

    identified = sum(1 for r in results if r["team_number"])
    print(f"  vLLM: identified {identified}/{len(robot_crops_b64)} robots: "
          f"{[r['team_number'] for r in results]}")
    return results


_CLASSIFY_PROMPT = (
    "This image shows ONE object cut out of video of an FRC (FIRST Robotics "
    "Competition) match. Decide what the object is:\n"
    '- "robot": a competition robot — a driving machine with wheels/mechanisms '
    "and a solid RED or BLUE bumper band around its base\n"
    '- "equipment": static field equipment — a crate, bin, tote, feeder rig, '
    "machine, or other structure that does not drive\n"
    '- "person": a human (referees wear striped or orange vests)\n'
    '- "unclear": you cannot tell\n\n'
    'Reply with ONLY compact JSON and nothing else: '
    '{"object": "robot" or "equipment" or "person" or "unclear"}.'
)


def _vllm_classify_crops(crops_b64: list) -> list:
    """Ask the vLLM server whether each crop is a robot/equipment/person.

    Same transport as _vllm_read_bumpers (probe, then one request per crop).
    Best-effort: any failure yields {"object": "unclear", "error": ...} —
    adjudication is an observe-only signal and must never raise into the loop.
    """
    try:
        probe = urllib.request.Request(f"{VLM_SERVER_URL}/models", method="GET")
        with urllib.request.urlopen(probe, timeout=3):
            pass
    except Exception as e:
        return [{"object": "unclear", "error": f"vLLM unreachable: {e}"} for _ in crops_b64]

    def _classify_one(crop_b64: str) -> dict:
        try:
            img_bytes = base64.b64decode(crop_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            min_dim = 512
            if img.width < min_dim or img.height < min_dim:
                scale = max(min_dim / img.width, min_dim / img.height)
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            payload = _vlm_chat_payload(
                data_uri, _CLASSIFY_PROMPT, max_tokens=32,
                schema_name="object_class", schema=_OBJECT_CLASS_SCHEMA,
            )
            body = _post_vlm_chat(payload)
            output_text = (body["choices"][0]["message"]["content"] or "").strip()
            return {"object": parse_object_class(output_text), "error": None}
        except Exception as e:
            return {"object": "unclear", "error": str(e)}

    return [_classify_one(c) for c in crops_b64]


_easyocr_reader = None

def _get_ocr_reader():
    """Lazy-init EasyOCR reader (uses GPU if available, cached after first call)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        _easyocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    return _easyocr_reader


def _ocr_bumper_number(frame: np.ndarray, box, match_teams: list) -> dict:
    """OCR the bumper number from a robot bounding box using EasyOCR.
    Tries multiple crop regions and preprocessing strategies.
    Returns {ocr_raw, matched_team, confidence}."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {"ocr_raw": "", "matched_team": None, "confidence": 0}

    bh = y2 - y1
    bw = x2 - x1
    reader = _get_ocr_reader()

    # Try multiple crop regions — bumper numbers can be at different heights
    # depending on camera angle (overhead vs side)
    crop_regions = [
        (0.50, 1.0),   # bottom half (typical overhead)
        (0.35, 0.85),  # middle-to-lower (angled view)
        (0.0, 1.0),    # full robot bbox (fallback)
    ]

    all_readings = []
    for top_frac, bot_frac in crop_regions:
        cy1 = y1 + int(bh * top_frac)
        cy2 = y1 + int(bh * bot_frac)
        crop = frame[cy1:cy2, x1:x2]
        if crop.size == 0:
            continue

        # Upscale small crops — EasyOCR works best at ~64px+ height
        crop_h = crop.shape[0]
        if crop_h < 64:
            scale = max(2, 64 // crop_h)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Run EasyOCR — allow digits only
        try:
            results = reader.readtext(
                crop,
                allowlist='0123456789',
                paragraph=False,
                min_size=5,
                text_threshold=0.3,
                low_text=0.3,
            )
            for (bbox, text, conf) in results:
                text = text.strip()
                if text:
                    all_readings.append((text, float(conf)))
        except Exception:
            pass

    # Deduplicate and pick the best reading
    raw = ""
    best_conf = 0
    for text, conf in all_readings:
        if conf > best_conf:
            raw = text
            best_conf = conf

    # Try to match against known teams
    best_match = None
    best_match_confidence = 0
    if raw:
        for team in match_teams:
            if raw == team:
                best_match = team
                best_match_confidence = 1.0
                break
            # Partial match — team number contained in OCR output or vice versa
            elif len(raw) >= 2 and (team in raw or raw in team):
                score = len(min(team, raw, key=len)) / len(max(team, raw, key=len))
                if score > best_match_confidence:
                    best_match = team
                    best_match_confidence = score
            # Edit distance for close matches (e.g. "2l02" vs "2102")
            elif len(raw) >= 3 and len(team) >= 3:
                matches = sum(a == b for a, b in zip(raw, team))
                score = matches / max(len(raw), len(team))
                if score >= 0.6 and score > best_match_confidence:
                    best_match = team
                    best_match_confidence = score * 0.8  # Penalize fuzzy matches

    return {"ocr_raw": raw, "matched_team": best_match, "confidence": best_match_confidence}


def _detect_robots_in_frame(
    frame: np.ndarray,
    model,
    match_teams: list,
    blue_teams: list,
    red_teams: list,
    bumper_model=None,
) -> list:
    """Detect robots in a single frame. Returns list of robot dicts."""
    if bumper_model is not None:
        results = bumper_model(frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
        schema = DetectionSchema.from_names(bumper_model.names)
        robot_dets, _balls, _hubs = schema.split(
            sv.Detections.from_ultralytics(results)
        )
    else:
        results = model(frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        robot_dets = [
            (
                tuple(detections.xyxy[i]),
                None,
                (
                    float(detections.confidence[i])
                    if detections.confidence is not None
                    else 1.0
                ),
            )
            for i in range(len(detections))
            if detections.class_id[i] == 1
        ]

    robots = []
    for box, class_color, conf in robot_dets:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        alliance = class_color or "unknown"

        if alliance == "blue":
            candidate_teams = blue_teams
        elif alliance == "red":
            candidate_teams = red_teams
        else:
            candidate_teams = match_teams

        ocr_result = _ocr_bumper_number(frame, box, candidate_teams)

        robot_crop = frame[max(0, y1):min(frame.shape[0], y2),
                          max(0, x1):min(frame.shape[1], x2)]
        _, crop_buf = cv2.imencode('.jpg', robot_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        crop_b64 = base64.b64encode(crop_buf).decode('utf-8')

        robots.append({
            "box": [x1, y1, x2, y2],
            "center": [cx, cy],
            "confidence": conf,
            "alliance": alliance,
            "ocr_raw": ocr_result["ocr_raw"],
            "matched_team": ocr_result["matched_team"],
            "ocr_confidence": ocr_result["confidence"],
            "crop": crop_b64,
            "frame_source": frame,  # Will be stripped before JSON response
        })
    return robots


def _deduplicate_robots(all_robots: list, merge_distance: float = 120.0) -> list:
    """Deduplicate robots detected across multiple frames.
    Keeps the detection with highest confidence for each unique position.
    If two detections from different frames are within merge_distance, they're the same robot."""
    if not all_robots:
        return []

    # Sort by confidence descending — best detections win
    all_robots.sort(key=lambda r: r["confidence"], reverse=True)

    unique = []
    for robot in all_robots:
        cx, cy = robot["center"]
        is_duplicate = False
        for existing in unique:
            ex, ey = existing["center"]
            dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if dist < merge_distance:
                is_duplicate = True
                # Keep existing (higher confidence), but update OCR if new one is better
                if robot["ocr_confidence"] > existing["ocr_confidence"]:
                    existing["ocr_raw"] = robot["ocr_raw"]
                    existing["matched_team"] = robot["matched_team"]
                    existing["ocr_confidence"] = robot["ocr_confidence"]
                # Keep the better crop (higher confidence detection)
                break
        if not is_duplicate:
            unique.append(robot)

    # Cap at 6 robots max (3 per alliance if possible)
    blue = [r for r in unique if r["alliance"] == "blue"][:MAX_ROBOTS_PER_ALLIANCE]
    red = [r for r in unique if r["alliance"] == "red"][:MAX_ROBOTS_PER_ALLIANCE]
    unknown = [r for r in unique if r["alliance"] == "unknown"]
    # Fill remaining slots with unknowns
    remaining_slots = MAX_ROBOTS_TOTAL - len(blue) - len(red)
    result = blue + red + unknown[:max(0, remaining_slots)]

    # Assign final indices
    for i, r in enumerate(result):
        r["index"] = i
        r["confidence"] = round(r["confidence"], 3)
        r["ocr_confidence"] = round(r["ocr_confidence"], 3)

    return result


@app.post("/api/detect-robots")
async def detect_robots(request: Request):
    """Detect robots across multiple calibration frames.
    Returns ALL detections (not deduplicated) so the user can tag each one.
    The frontend groups by team assignment on confirm."""
    body = await request.json()
    match_teams = body.get("match_teams", [])
    blue_teams = body.get("blue_teams", [])
    red_teams = body.get("red_teams", [])

    # Accept either a list of frames or single frame
    frames_b64 = body.get("frames", [])
    if not frames_b64 and body.get("frame"):
        frames_b64 = [body["frame"]]

    if not frames_b64:
        return JSONResponse(status_code=400, content={"error": "No frames provided"})

    model = YOLO(str(MODEL_PATH))
    bumper_model = (
        YOLO(str(BUMPER_MODEL_PATH)) if BUMPER_MODEL_PATH.exists() else None
    )
    all_robots = []
    frame_dims = None

    for frame_idx, frame_b64 in enumerate(frames_b64):
        frame_bytes = base64.b64decode(frame_b64)
        frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame_dims is None:
            frame_dims = (frame.shape[1], frame.shape[0])

        robots = _detect_robots_in_frame(
            frame,
            model,
            match_teams,
            blue_teams,
            red_teams,
            bumper_model=bumper_model,
        )
        for r in robots:
            r["frame_index"] = frame_idx
            r.pop("frame_source", None)
        all_robots.extend(robots)

    # Use Claude Vision for bumper number reading (much better than EasyOCR)
    if all_robots and match_teams:
        crops_b64 = [r["crop"] for r in all_robots]
        expected = [r["alliance"] if r.get("alliance") in ("blue", "red") else "unknown"
                    for r in all_robots]
        claude_results = _vision_read_bumpers(
            crops_b64, match_teams, blue_teams, red_teams,
            expected_alliances=expected,
        )
        for i, (robot, claude_result) in enumerate(zip(all_robots, claude_results)):
            if claude_result["team_number"]:
                robot["matched_team"] = claude_result["team_number"]
                robot["ocr_confidence"] = claude_result["confidence"]
                robot["ocr_raw"] = claude_result["team_number"]
                # Also update alliance if Claude is confident
                if claude_result["alliance"] in ("blue", "red"):
                    robot["alliance"] = claude_result["alliance"]

    # Assign indices and round floats
    for i, r in enumerate(all_robots):
        r["index"] = i
        r["confidence"] = round(r["confidence"], 3)
        r["ocr_confidence"] = round(r["ocr_confidence"], 3)

    w, h = frame_dims if frame_dims else (0, 0)
    return {
        "robots": all_robots,
        "frame_width": w,
        "frame_height": h,
        "frames_analyzed": len(frames_b64),
        "total_detections": len(all_robots),
    }


# --- TBA Event Session Endpoints ---

@app.post("/api/event/load")
async def load_event(request: Request):
    """Load an event from TBA by URL or event key."""
    global current_event
    from tba import load_event_data

    body = await request.json()
    url_or_key = body.get("url") or body.get("event_key", "")
    if not url_or_key:
        return JSONResponse(status_code=400, content={"error": "Provide a TBA event URL or event key"})

    try:
        event_data = await load_event_data(url_or_key)
        current_event = event_data

        # Persist event to DB and restore saved OCR regions
        from database import upsert_event, upsert_match, upsert_match_robot, get_ocr_regions as db_get_ocr_regions
        from tba import extract_hub_scores

        event_id = await upsert_event(
            event_key=event_data['event_key'],
            event_name=event_data['event_name'],
            year=event_data['year'],
            location=event_data.get('location'),
            start_date=event_data.get('start_date'),
            end_date=event_data.get('end_date'),
        )

        # Persist all matches and team assignments
        for m in event_data['matches']:
            blue_hub = m.get('blue_hub', {})
            red_hub = m.get('red_hub', {})

            match_id = await upsert_match(
                event_id=event_id,
                tba_match_key=m['key'],
                comp_level=m['comp_level'],
                match_number=m['match_number'],
                set_number=m.get('set_number', 1),
                tba_blue_score=m.get('blue_score'),
                tba_red_score=m.get('red_score'),
                tba_blue_hub_total=blue_hub.get('total'),
                tba_red_hub_total=red_hub.get('total'),
                tba_blue_hub_auto=blue_hub.get('auto'),
                tba_red_hub_auto=red_hub.get('auto'),
                tba_blue_hub_teleop=blue_hub.get('teleop'),
                tba_red_hub_teleop=red_hub.get('teleop'),
                tba_video_key=m.get('video_key'),
            )

            # Persist team assignments
            for i, team_num in enumerate(m['blue_teams']):
                await upsert_match_robot(match_id, 'blue', i + 1, team_num)
            for i, team_num in enumerate(m['red_teams']):
                await upsert_match_robot(match_id, 'red', i + 1, team_num)

        # Restore saved OCR regions and hub positions for this event
        saved_regions = await db_get_ocr_regions(event_data['event_key'])
        global custom_ocr_regions, event_hub_positions
        if saved_regions:
            custom_ocr_regions = saved_regions

        from database import get_hub_positions as db_get_hub_positions
        saved_hubs = await db_get_hub_positions(event_data['event_key'])
        if saved_hubs:
            event_hub_positions = saved_hubs

        return {
            "event_key": event_data['event_key'],
            "event_name": event_data['event_name'],
            "year": event_data['year'],
            "location": event_data.get('location'),
            "team_count": len(event_data['teams']),
            "match_count": len(event_data['matches']),
            "matches": event_data['matches'],
            "teams": event_data['teams'],
            "timezone": event_data.get('timezone'),
            "has_ocr_regions": saved_regions is not None,
            "has_hub_positions": saved_hubs is not None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/event/current")
async def get_current_event():
    """Get the currently loaded event data."""
    if not current_event:
        return JSONResponse(status_code=404, content={"error": "No event loaded"})
    return {
        "event_key": current_event['event_key'],
        "event_name": current_event['event_name'],
        "matches": current_event['matches'],
        "teams": current_event['teams'],
        "timezone": current_event.get('timezone'),
        "has_ocr_regions": custom_ocr_regions is not None,
        "has_hub_positions": event_hub_positions is not None,
    }


@app.get("/api/event/match/{match_key}")
async def get_match_details(match_key: str):
    """Get match details from DB (with robots and TBA ground truth)."""
    from database import get_match_by_key
    match = await get_match_by_key(match_key)
    if not match:
        return JSONResponse(status_code=404, content={"error": "Match not found"})
    return match


# --- Save / Update Results ---

@app.post("/api/match/save")
async def save_match_results(request: Request):
    """Save or update match results after inference and user corrections."""
    from database import (get_match_by_key, update_match_predictions,
                         update_robot_scores, save_scoring_events, update_match_notes)

    data = await request.json()
    match_key = data.get("match_key")
    if not match_key:
        return JSONResponse(status_code=400, content={"error": "match_key is required"})

    match = await get_match_by_key(match_key)
    if not match:
        return JSONResponse(status_code=404, content={"error": f"Match {match_key} not found in DB"})

    match_id = match['id']

    # Update predictions and video source
    if 'blue_predicted_score' in data or 'red_predicted_score' in data:
        await update_match_predictions(
            match_id,
            data.get('blue_predicted_score', match.get('blue_predicted_score', 0)),
            data.get('red_predicted_score', match.get('red_predicted_score', 0)),
            data.get('video_source'),
        )

    # Update notes
    if 'notes' in data:
        await update_match_notes(match_id, data['notes'])

    # Update robots (team numbers and corrected counts)
    robots = data.get('robots', [])
    for robot_data in robots:
        robot_id = robot_data.get('id')
        if not robot_id:
            continue
        await update_robot_scores(
            match_robot_id=robot_id,
            predicted_fuel_count=robot_data.get('predicted_fuel_count'),
            corrected_fuel_count=robot_data.get('corrected_fuel_count'),
            team_number=robot_data.get('team_number'),
        )

    # Save scoring events if provided
    if 'scoring_events' in data:
        await save_scoring_events(match_id, data['scoring_events'])

    # Return updated match
    updated = await get_match_by_key(match_key)
    return updated


# --- History Endpoints ---

@app.get("/api/history")
async def get_history():
    """Get all events with their scored matches."""
    from database import get_all_events, get_scored_matches

    events = await get_all_events()
    result = []
    for event in events:
        matches = await get_scored_matches(event['id'])
        if matches:
            event['matches'] = matches
            result.append(event)

    return result


@app.get("/api/history/all")
async def get_all_match_history():
    """Get all matches across all events (for history page)."""
    from database import get_all_events, get_event_matches

    events = await get_all_events()
    result = []
    for event in events:
        matches = await get_event_matches(event['id'])
        result.append({
            **event,
            'matches': matches,
        })

    return result


@app.get("/api/diagnostic/{key}")
async def get_diagnostic(key: str):
    """Stream the diagnostic JSONL for a live job or a saved match.
    Resolution order: diagnostic_<key>.jsonl (live) then diagnostic_match_<key>.jsonl (saved).
    """
    # Path-traversal guard: only allow safe key characters.
    if not key.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="invalid key")

    candidates = [
        UPLOAD_DIR / f"diagnostic_{key}.jsonl",
        UPLOAD_DIR / f"diagnostic_match_{key}.jsonl",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        raise HTTPException(status_code=404, detail="no diagnostic data")

    def iter_file():
        with target.open("rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(iter_file(), media_type="application/x-ndjson")


@app.get("/api/sam2_models")
async def sam2_models():
    """List selectable SAM2 encoder sizes and whether each checkpoint is downloaded.

    The UI uses `available` to disable sizes whose .pt file is missing, since picking
    one would silently degrade to Kalman tracking.
    """
    def _vllm_server_up() -> bool:
        try:
            req = urllib.request.Request(f"{VLM_SERVER_URL}/models", method="GET")
            with urllib.request.urlopen(req, timeout=1.5):
                return True
        except Exception:
            return False

    # Phase-exclusive lifecycle starts the server on demand during a job's
    # analysis phase, so "selectable" means installed, not currently serving.
    vllm_up = await asyncio.to_thread(_vllm_server_up)
    vllm_installed = (Path(__file__).parent.parent / ".venv-vllm" / "bin" / "vllm").exists()
    vllm_state = "online" if vllm_up else ("on-demand" if vllm_installed else "unavailable")

    return {
        "default": SAM2_MODEL_SIZE,
        "default_backend": TRACKING_BACKEND,
        "vllm_available": vllm_up or vllm_installed,
        "vllm_state": vllm_state,
        "backends": [
            {
                "id": "sam3",
                "label": "SAM 3",
                "detail": "instance tracking — best identity, default",
            },
            {
                "id": "sam2",
                "label": "SAM 2.1",
                "detail": "Hiera VOS — legacy fallback",
            },
        ],
        "models": [
            {
                "size": size,
                "label": label,
                "detail": detail,
                "available": _sam2_checkpoint_present(size),
            }
            for size, label, detail in _SAM2_SIZE_INFO
        ],
    }


async def _load_db_roster(match_key: Optional[str]) -> Optional[tuple[list, list, list]]:
    """Server-authoritative roster from the DB: ``(match_teams, blue, red)``.

    The browser pushes the roster over the websocket within a 0.5 s window
    (see ``websocket_infer``); on a timeout — or when the page had no roster
    loaded for the selected match — the job historically ran with NO roster,
    which disables the alliance gate in ``_match_bumper_read_to_roster`` and
    lets a misread of one alliance's team number land on the other alliance's
    robot. The same roster is already in ``match_robots``, so load it here and
    use it as the default; a non-empty client roster still overrides.
    """
    if not match_key:
        return None
    try:
        match = await database.get_match_by_key(match_key)
    except Exception as e:
        print(f"  [roster] DB lookup failed for {match_key}: {e}")
        return None
    if not match:
        return None
    blue, red = [], []
    for r in match.get("robots", []):
        team = r.get("team_number")
        if team is None:
            continue
        (blue if r.get("alliance") == "blue" else red).append(str(team))
    if not blue and not red:
        return None
    return [*blue, *red], blue, red


@app.post("/api/infer")
async def infer(
    video_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    team_reader: str = Form("local"),
    sam2_max_gpu_frames: Optional[int] = Form(None),
    tracking_backend: Optional[str] = Form(None),
    model_size: Optional[str] = Form(None),
    match_key: Optional[str] = Form(None),
    sam2_export_scale: Optional[float] = Form(None),
    cull_hijack_enabled: Optional[bool] = Form(None),
    reanchor_enabled: Optional[bool] = Form(None),
    config_json: Optional[str] = Form(None),
):
    """Start inference job.

    Per-job knobs arrive either as the legacy named form fields above or as
    `config_json`, a JSON object of {knob_name: value} validated against the
    JobConfig registry — a bad knob fails the submit with 400 instead of
    failing minutes into inference. config_json wins over named fields.
    """
    job_id = str(uuid.uuid4())

    # Resolve and validate the job config BEFORE touching the video source.
    overrides: dict = {}
    if team_reader and team_reader != "local":
        overrides["team_reader"] = team_reader
    if sam2_max_gpu_frames is not None:
        overrides["sam2_max_gpu_frames"] = sam2_max_gpu_frames
    if tracking_backend:
        overrides["tracking_backend"] = tracking_backend.lower()
    if model_size:
        overrides["sam2_model_size"] = model_size.lower()
    if sam2_export_scale is not None:
        overrides["sam2_export_scale"] = sam2_export_scale
    if cull_hijack_enabled is not None:
        overrides["cull_hijack_enabled"] = cull_hijack_enabled
    if reanchor_enabled is not None:
        overrides["reanchor_enabled"] = reanchor_enabled
    if config_json:
        try:
            extra = json.loads(config_json)
        except ValueError:
            return JSONResponse(status_code=400,
                                content={"error": "config_json is not valid JSON"})
        if not isinstance(extra, dict):
            return JSONResponse(status_code=400,
                                content={"error": "config_json must be a JSON object"})
        overrides.update(extra)
    try:
        job_config = JobConfig(overrides)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    if not _sam2_checkpoint_present(job_config.sam2_model_size):
        print(f"  [sam2] WARNING: '{job_config.sam2_model_size}' checkpoint not found at "
              f"{_SAM2_MODELS[job_config.sam2_model_size][1]}; "
              f"SAM2 will fall back to Kalman tracking.")

    try:
        # Handle video source
        if video_file:
            # Content-addressed storage: A/B reruns of the same match re-upload
            # the identical file under a new job_id every time. Hash while
            # streaming to a temp file, then dedupe on (digest, size) so repeat
            # uploads reuse the existing file instead of duplicating ~100MB+
            # videos on disk. Jobs are keyed by job_id, not filename, so
            # sharing one file across jobs is safe.
            suffix = Path(video_file.filename or "").suffix or ".mp4"
            temp_path = UPLOAD_DIR / f"{job_id}.part"
            md5 = hashlib.md5()
            try:
                with open(temp_path, "wb") as f:
                    while True:
                        chunk = video_file.file.read(1024 * 1024)
                        if not chunk:
                            break
                        md5.update(chunk)
                        f.write(chunk)
                final_path = UPLOAD_DIR / f"video_{md5.hexdigest()}{suffix}"
                if final_path.exists() and final_path.stat().st_size == temp_path.stat().st_size:
                    temp_path.unlink()
                    print(f"  [upload] dedupe hit: reusing {final_path.name}")
                else:
                    os.replace(temp_path, final_path)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
            video_path = final_path

        elif youtube_url:
            # Download YouTube video
            video_path = UPLOAD_DIR / f"{job_id}.mp4"
            ydl_opts = {
                # Select by RESOLUTION, not container. Forcing ext=mp4 on the video
                # stream caps quality at ~480p because YouTube serves 720p/1080p mostly
                # as VP9/AV1 (webm), not H.264 (mp4). Take the best video up to 1080p of
                # any codec, prefer H.264 at that resolution for clean OpenCV decode, and
                # remux to .mp4 so the output path stays {job_id}.mp4.
                'format': (
                    'bestvideo[height<=1080]+bestaudio[ext=m4a]/'
                    'bestvideo[height<=1080]+bestaudio/'
                    'best[height<=1080]/best'
                ),
                'format_sort': ['res:1080', 'vcodec:h264'],
                'merge_output_format': 'mp4',
                'outtmpl': str(video_path),
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
        
        else:
            return JSONResponse(status_code=400, content={"error": "No video file or YouTube URL provided"})
        
        # Create job; every per-job knob flows through the validated config.
        job = InferenceJob(
            job_id, video_path, match_key=match_key, config=job_config)
        # Seed the roster server-side from the DB so the alliance gate works
        # even if the browser never pushes one over the websocket. A non-empty
        # client roster still overrides this in websocket_infer.
        db_roster = await _load_db_roster(match_key)
        if db_roster is not None:
            job.match_teams, job.match_blue_teams, job.match_red_teams = db_roster
            print(f"  [roster] seeded from DB for {match_key}: "
                  f"blue={job.match_blue_teams} red={job.match_red_teams}")
        _prune_finished_jobs()  # finished jobs stop pinning models in memory
        jobs[job_id] = job
        
        return {"job_id": job_id}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.post("/api/job/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause an active inference job."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    if job.status in {"completed", "error"}:
        return JSONResponse(status_code=400, content={"error": f"Job already {job.status}"})
    job._paused = True
    job._pause_frame = job._current_frame
    return {"status": "paused", "job_id": job_id, "frame_num": job._pause_frame}


@app.post("/api/job/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused inference job."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    if job.status in {"completed", "error"}:
        return JSONResponse(status_code=400, content={"error": f"Job already {job.status}"})
    job._paused = False
    return {"status": "running", "job_id": job_id}


@app.post("/api/job/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop an active inference job and finalize with current results."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    if job.status in {"completed", "error"}:
        return JSONResponse(status_code=400, content={"error": f"Job already {job.status}"})
    job._stopped = True
    job._paused = False
    return {"status": "stopping", "job_id": job_id, "frame_num": job._current_frame}


@app.post("/api/job/{job_id}/retry")
async def retry_job(job_id: str):
    """Continue an aborted analysis WITHOUT re-uploading the video.

    A calibration-drift abort stops the job before any tracking. After the operator
    fixes the calibration (approves the staged auto-fix or recalibrates manually),
    this re-runs the SAME video, match, and per-job config under a fresh job id. The
    new run re-scans calibration against the now-current epoch and, if it passes,
    processes normally. The client opens a websocket to the returned job_id to start
    it (a fresh job is not yet run-claimed, so the socket drives process_video).
    """
    old = jobs.get(job_id)
    if old is None:
        return JSONResponse(status_code=404, content={
            "error": "Job not found (it may have been pruned) — re-submit the match."})
    video_path = Path(str(old.video_path))
    if not video_path.exists():
        return JSONResponse(status_code=400, content={
            "error": "The original video is no longer on the server — re-submit the match."})
    new_id = str(uuid.uuid4())
    job = InferenceJob(new_id, video_path,
                       match_key=getattr(old, "match_key", None),
                       config=getattr(old, "config", None))
    # Re-seed the roster from the DB exactly as /api/infer does; the client still
    # overrides with any roster it pushes over the new websocket.
    db_roster = await _load_db_roster(getattr(old, "match_key", None))
    if db_roster is not None:
        job.match_teams, job.match_blue_teams, job.match_red_teams = db_roster
    _prune_finished_jobs()
    jobs[new_id] = job
    print(f"  [retry] continuing {job_id} as {new_id} "
          f"(video + config reused; calibration will be re-scanned)")
    return {"job_id": new_id, "previous_job_id": job_id}


@app.get("/api/replays")
async def list_replays_endpoint():
    rows = await database.list_replays()
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "label": r["label"], "created_at": r["created_at"],
            "blue_score": r["blue_score"], "red_score": r["red_score"],
            # Official TBA rosters (empty when the match wasn't imported);
            # the card view shows these, not the detected-team lists.
            "blue_teams": r["tba_blue_teams"],
            "red_teams": r["tba_red_teams"],
            "event_name": r["event_name"],
            "comp_level": r["tba_comp_level"],
            "match_number": r["match_number"],
            "set_number": r["set_number"],
            "match_key": r["match_key"],
            "duration_sec": r["duration_sec"], "status": r["status"],
            "processing_sec": r["processing_sec"],
            "thumb_url": f"/api/replays/{r['id']}/thumb",
            "video_url": f"/api/replays/{r['id']}/video",
        })
    return out


@app.get("/api/replays/{replay_id}/thumb")
async def get_replay_thumb(replay_id: str):
    row = await database.get_replay(replay_id)
    if not row or not Path(row["thumb_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(row["thumb_path"], media_type="image/jpeg")


@app.get("/api/replays/{replay_id}/video")
async def get_replay_video(replay_id: str, request: Request):
    row = await database.get_replay(replay_id)
    if not row or not Path(row["video_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    path = Path(row["video_path"])
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header is None:
        return FileResponse(str(path), media_type="video/mp4")

    # Parse "bytes=START-END"
    units, _, rng = range_header.partition("=")
    start_s, _, end_s = rng.partition("-")
    try:
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    length = end - start + 1

    def iter_slice(chunk=64 * 1024):
        remaining = length
        with path.open("rb") as f:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iter_slice(), status_code=206,
                             media_type="video/mp4", headers=headers)


def _safe_filename(label: str, ext: str) -> str:
    """Filesystem-safe download filename derived from a replay label."""
    base = "".join(c if (c.isalnum() or c in "._-") else "_"
                   for c in (label or "replay")).strip("_") or "replay"
    return f"{base}.{ext}"


@app.get("/api/replays/{replay_id}/download/video")
async def download_replay_video(replay_id: str):
    """Serve the recorded mp4 as an attachment (forces a download with a clean name)."""
    row = await database.get_replay(replay_id)
    if not row or not Path(row["video_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(row["video_path"], media_type="video/mp4",
                        filename=_safe_filename(row["label"], "mp4"))


@app.get("/api/replays/{replay_id}/download/data")
async def download_replay_data(replay_id: str):
    """Serve the match data (meta.json: summary + per-frame tracking + events) as a download."""
    row = await database.get_replay(replay_id)
    if not row or not Path(row["meta_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(row["meta_path"], media_type="application/json",
                        filename=_safe_filename(row["label"], "json"))


@app.get("/api/replays/{replay_id}")
async def get_replay_endpoint(replay_id: str):
    row = await database.get_replay(replay_id)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not found"})
    meta_path = Path(row["meta_path"])
    if not meta_path.exists():
        return JSONResponse(status_code=404, content={"error": "metadata missing"})
    meta = json.loads(meta_path.read_text())
    # TBA match identity for the player header (the job GUID stays available
    # as `id` for troubleshooting).
    info = {"id": row["id"], "label": row["label"], "match_key": row["match_key"]}
    if row["match_key"]:
        match = await database.get_match_by_key(row["match_key"])
        if match:
            info.update({
                "event_name": match["event_name"],
                "comp_level": match["comp_level"],
                "match_number": match["match_number"],
                "set_number": match["set_number"],
            })
    meta["replay_info"] = info
    return JSONResponse(content=meta)


@app.get("/api/replays/{job_id}/identities")
async def get_replay_identities(job_id: str):
    """Robot-slot identities for a job, each with its captured bumper crops as
    data-URLs. Raw embeddings are NEVER exposed to the client."""
    ids = await database.get_match_identities(job_id)
    canonical = []
    for alliance, slots in (("blue", range(3)), ("red", range(3, 6))):
        candidates = [r for r in ids if r.get("alliance") == alliance]
        by_slot = {}
        used_identity_ids = set()
        for row in candidates:
            slot = row.get("slot")
            if slot in slots and slot not in by_slot:
                by_slot[slot] = row
                used_identity_ids.add(row.get("identity_id"))
        remaining = [
            row for row in candidates
            if row.get("identity_id") not in used_identity_ids
        ]
        remaining.sort(
            key=lambda row: (
                not bool(row.get("confirmed")),
                -(float(row.get("confidence") or 0.0)),
                int(row.get("identity_id") or 0),
            )
        )
        for slot, row in zip((s for s in slots if s not in by_slot), remaining):
            row = dict(row)
            row["slot"] = slot
            row["slot_label"] = InferenceJob._slot_label(slot)
            by_slot[slot] = row
        canonical.extend(by_slot[slot] for slot in slots if slot in by_slot)
    ids = canonical
    out = []
    for r in ids:
        identity_id = r.get("identity_id")
        if identity_id is None:
            identity_id = r["slot"]
        crops = await database.get_match_crops(job_id, identity_id)
        out.append({
            "slot": r["slot"] if r["slot"] is not None else -1,
            "slot_label": r["slot_label"] if r["slot_label"] is not None else "occluded",
            "alliance": r["alliance"],
            "team_number": r["team_number"], "confirmed": bool(r["confirmed"]),
            "source": r["source"], "confidence": r["confidence"],
            "identity_id": identity_id,
            "crops": [
                "data:image/jpeg;base64," + base64.b64encode(c["image"]).decode()
                for c in crops if c["image"] is not None
            ],
        })
    return {"job_id": job_id, "robots": out}


@app.post("/api/replays/{job_id}/correct")
async def correct_replay_identity(job_id: str, request: Request):
    """Correct one robot slot's team. Updates the stored identity, fixes the
    replay's displayed team list, and mints VERIFIED gallery exemplars from the
    slot's crops (only when the replay has an event_key)."""
    body = await request.json()
    identity_id = body.get("identity_id")
    slot = body.get("slot")
    if slot is not None:
        slot = int(slot)
    team = str(body["team_number"]).strip()
    
    ids = await database.get_match_identities(job_id)
    rec = None
    if identity_id is not None:
        rec = next((r for r in ids if r.get("identity_id") == int(identity_id)), None)
    elif slot is not None:
        rec = next((r for r in ids if r.get("slot") == slot), None)
        
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "identity not found"})
        
    resolved_identity_id = rec.get("identity_id")
    if resolved_identity_id is None:
        resolved_identity_id = rec["slot"]
    resolved_slot = rec.get("slot")
        
    alliance = rec["alliance"]
    # 1. update identity (preserve slot_label/alliance/confirmed/source/confidence)
    await database.upsert_match_robot_identity(
        {**rec, "job_id": job_id, "identity_id": resolved_identity_id, "team_number": team})
    # 2. update the replay's team lists for display + mint exemplars
    replay = await database.get_replay(job_id)
    minted = 0
    if replay is not None:
        key = "blue_teams" if alliance == "blue" else "red_teams"
        try:
            teams = json.loads(replay.get(key) or "[]")
        except Exception:
            teams = []
        # replace this slot's old team with the corrected one (best-effort positional)
        old = rec["team_number"]
        teams = [team if str(t) == str(old) else t for t in teams]
        if team not in [str(t) for t in teams]:
            teams.append(team)
        await database.update_replay_teams(job_id, key, json.dumps(teams))
        # 3. mint verified exemplars from this identity's crops (needs event_key + color)
        event_key = replay.get("event_key")
        color = alliance if alliance in ("red", "blue") else None
        if event_key and color:
            for c in await database.get_match_crops(job_id, resolved_identity_id):
                if c["embedding"] is None:
                    continue
                await database.insert_gallery_exemplar({
                    "event_key": event_key,
                    "source_match_key": replay.get("match_key"),
                    "team": team, "alliance_color": color,
                    "embedding": c["embedding"], "legibility": c["legibility"],
                    "image": c["image"], "verified": 1, "job_id": job_id,
                    # Carry the crop's own embed_model forward — it may have
                    # been captured by an older model than the current job.
                    "embed_model": c["embed_model"]})
                minted += 1
    return {"status": "ok", "slot": resolved_slot if resolved_slot is not None else -1,
            "identity_id": resolved_identity_id, "team_number": team,
            "verified_exemplars": minted}


@app.get("/api/gallery")
async def list_gallery(event_key: str):
    """List appearance-gallery exemplars for an event. Returns thumbnails as
    data-URLs; never serializes the raw embedding bytes."""
    rows = await database.get_gallery_exemplars(event_key)
    # Build a job_id -> roster ([blue+red team numbers]) map once, so each
    # exemplar can offer the 6 robots from the match it was pulled from.
    roster_by_job: dict[str, list] = {}
    for job_id in {r.get("job_id") for r in rows if r.get("job_id")}:
        replay = await database.get_replay(job_id)
        if not replay:
            continue
        teams = []
        for key in ("blue_teams", "red_teams"):
            try:
                teams.extend(str(t) for t in json.loads(replay.get(key) or "[]"))
            except (ValueError, TypeError):
                pass
        roster_by_job[job_id] = teams
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "team": r["team"], "alliance_color": r["alliance_color"],
            "verified": bool(r["verified"]), "legibility": r["legibility"],
            "job_id": r.get("job_id"),
            "match_teams": roster_by_job.get(r.get("job_id"), []),
            "image": ("data:image/jpeg;base64," + base64.b64encode(r["image"]).decode())
                      if r.get("image") else None,
        })
    return {"event_key": event_key, "exemplars": out}


@app.post("/api/gallery/{gid}")
async def update_gallery(gid: int, request: Request):
    """Relabel team/alliance_color or toggle verified on one gallery exemplar."""
    body = await request.json()
    color = body.get("alliance_color")
    if color is not None and color not in ("red", "blue"):
        return JSONResponse(status_code=400,
                            content={"error": "alliance_color must be 'red' or 'blue'"})
    team = body.get("team")
    if team is not None:
        team = str(team).strip()
    verified = body.get("verified")
    if verified is not None:
        verified = bool(verified)
    await database.update_gallery_exemplar(
        gid, team=team, alliance_color=color, verified=verified)
    return {"status": "ok", "id": gid}


@app.delete("/api/gallery/{gid}")
async def remove_gallery(gid: int):
    """Delete one gallery exemplar."""
    await database.delete_gallery_exemplar(gid)
    return {"status": "deleted", "id": gid}


@app.delete("/api/gallery")
async def clear_gallery(event_key: str):
    """Delete ALL gallery exemplars for an event (troubleshooting reset). The
    gallery regenerates from match processing, so this is a cache clear, not
    permanent data loss."""
    removed = await database.clear_gallery_exemplars(event_key)
    return {"status": "cleared", "event_key": event_key, "removed": removed}


@app.delete("/api/replays/{replay_id}")
async def delete_replay_endpoint(replay_id: str):
    row = await database.get_replay(replay_id)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not found"})
    replay_dir = Path(row["video_path"]).parent
    if replay_dir.exists():
        shutil.rmtree(replay_dir, ignore_errors=True)
    await database.delete_replay(replay_id)
    return {"status": "deleted", "id": replay_id}


@app.websocket("/ws/infer/{job_id}")
async def websocket_infer(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for streaming inference results"""
    await websocket.accept()
    
    if job_id not in jobs:
        await websocket.send_json({"type": "error", "error": "Job not found"})
        await websocket.close()
        return
    
    job = jobs[job_id]
    reattach = job._run_claimed
    job.websocket = websocket
    job.diagnostic.set_ws_send(job._diagnostic_ws_send)

    # Wait briefly for match team data from frontend
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
        data = json.loads(msg)
        if data.get("type") == "robot_identifications":
            # Store match teams for continuous bumper OCR during inference.
            # Only override the DB-seeded roster (set at job creation) when the
            # client actually supplies one — an empty client message must not
            # wipe a good server-side roster.
            client_blue = data.get("blue_teams") or []
            client_red = data.get("red_teams") or []
            if client_blue or client_red:
                job.match_teams = data.get("match_teams", []) or [*client_blue, *client_red]
                job.match_blue_teams = client_blue
                job.match_red_teams = client_red
                print(
                    "Received match roster for VLM: "
                    f"blue={job.match_blue_teams} red={job.match_red_teams}"
                )
    except (asyncio.TimeoutError, Exception):
        pass

    try:
        if reattach:
            # The job's run is owned by the connection that claimed it; this
            # one only attaches for streaming (job._send_ws already targets
            # the newest socket). Running _run_job_serialized again here used
            # to queue a SECOND process_video over the same job.
            if job.status in ("completed", "error"):
                await job._send_ws({
                    "type": "status", "status": job.status,
                    "error": job.error,
                    "message": "Job already finished — results are in Replays.",
                })
            else:
                await job._send_ws({
                    "type": "status", "status": job.status or "processing",
                    "message": "Re-attached to running job; streaming resumed.",
                })
                done_task = asyncio.create_task(job._done_event.wait())
                recv_task = asyncio.create_task(_drain_client(websocket))
                try:
                    finished, _pending = await asyncio.wait(
                        {done_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in finished:
                        t.exception()  # retrieve WebSocketDisconnect etc.
                finally:
                    done_task.cancel()
                    recv_task.cancel()
        else:
            job._run_claimed = True
            try:
                await _run_job_serialized(job)
            finally:
                job._done_event.set()
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        print(f"Error in websocket_infer: {e}")
        await job._send_ws({"type": "error", "error": str(e)})
    finally:
        # Only tear down job-side streaming if this socket is still the
        # job's current one — a stale handler exiting after a reattach must
        # not sever the new connection.
        if job.websocket is websocket:
            try:
                job.diagnostic.set_ws_send(None)
            except Exception:
                pass
            job.websocket = None
        try:
            await websocket.close()
        except:
            pass


async def _drain_client(websocket: WebSocket) -> None:
    """Consume messages from a reattached (read-only) viewer until it
    disconnects; raises WebSocketDisconnect to end the attachment."""
    while True:
        await websocket.receive_text()


# Mount static files
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if "/static/" in str(request.url):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

"""
FRC Vision Scorer - YOLO26s Hub-Zone Model
Clean rebuild focused on per-robot scoring attribution
"""

import asyncio
import base64
import cv2
import datetime
import json
import numpy as np
import os
import pytesseract
import shutil
import tempfile
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yt_dlp
from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import supervision as sv

# This file is launched as `python web/app.py`, which puts web/ (not the project
# root) on sys.path — so the `web` package below would not resolve. Put the
# project root on the path so `web.embedder`/`web.reid` import the same way they
# do under pytest (which runs from the root). Harmless when already present.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.embedder import RobotEmbedder, cosine_similarity
from web.legibility import legibility_score
from web import reid
from web.diagnostic import DiagnosticRecorder, NullRecorder
from web.field import store as field_store
from web.field.field_model import FieldModel
from web.field.calibration import Calibrator, CalibrationError, holdout_field_error
from web.field.apriltag_detect import detect_tags
from web.field.composite_layout import load_layout
from web.field.tracker import FieldTracker
from web.field.occlusion import hub_prisms, resolve_occlusion
from web.field.detect_bridge import to_field_detections, field_adjacent_persons
from web.field.radar import Radar

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


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "runs/train/hub_zone_yolo26s/weights/best.pt"
STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
SCORE_REGIONS_FILE = PROJECT_ROOT / "score_regions.json"
CALIB_TEST_VIDEO = Path(__file__).resolve().parents[1] / "data/test_footage/qual1_first_champs_2026.mp4"
CALIB_MAIN_CROP_FRAC = 0.60   # main camera = top 60% of the composite

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
    regions = custom_ocr_regions if custom_ocr_regions else OCR_REGIONS_1080P
    sx = video_width / 1920.0
    sy = overhead_height / 723.0  # Scale to actual overhead crop height
    scaled = {}
    for key, (x, y, w, h) in regions.items():
        scaled[key] = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
    return scaled
OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"
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
REREAD_EVERY_N_PASSES = 4      # include confirmed robots once every Nth pass (self-correction)
OVERTURN_VOTE_MARGIN = 3.0     # competitor team's votes must exceed current by this to overturn

# Bumper-number reading VLM. Swap models here (loaded via AutoModelForImageTextToText,
# so any transformers-supported image-text-to-text model works, e.g. InternVL3, Nemotron).
VLM_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Minimum YOLO detection confidence. Raised from the 0.25 default to cut false-positive
# "robot" detections (people, carts) that otherwise steal canonical slots and inherit labels.
DET_CONF_THRESHOLD = 0.35

# Scoring parameters
BALL_PROXIMITY_FRAMES = 30  # Look back this many frames for ball near robot
HUB_DISTANCE_HIGH_CONF = 150  # px
HUB_DISTANCE_MED_CONF = 300   # px

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(title="FRC Vision Scorer")

# Global state for inference jobs
jobs: Dict[str, "InferenceJob"] = {}

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
    team_ocr_votes: Optional[Dict[str, int]] = None  # team_number -> vote count
    team_number_confirmed: bool = False  # True once OCR or calibration locked it in
    # Appearance embedding for re-ID (DINOv2, EMA-updated)
    appearance_embedding: Optional[np.ndarray] = None
    appearance_votes: Optional[Dict[str, float]] = None  # team_number -> vote score
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


class InferenceJob:
    """Manages a single video inference job"""
    
    def __init__(self, job_id: str, video_path: Path):
        self.job_id = job_id
        self.video_path = video_path
        self.status = "initializing"
        self.error: Optional[str] = None
        self._log_file = None
        self._log_path: Optional[Path] = None
        self.websocket: Optional[WebSocket] = None
        self._paused = False
        self._stopped = False
        self._pause_frame = 0
        self._current_frame = 0
        
        # Model and tracking. SP2 replaced ByteTrack + DINOv2 re-ID with the
        # field-space FieldTracker (built in process_video once the calibration
        # loads), so neither the ByteTracker nor the embedder is instantiated here.
        self.model = YOLO(str(MODEL_PATH))
        # Person detector for occlusion reasoning (run per processed frame, classes=[0]).
        # Absolute path so it doesn't depend on CWD (a bare name would trigger a
        # hub auto-download attempt for a non-existent release).
        self.person_model = YOLO(str(PROJECT_ROOT / "yolo26n.pt"))

        # Field-space tracking — bound to the loaded event/camera calibration.
        # Snapshot the active event key at construction; calibration + field model
        # are loaded at the top of process_video() (decision 7 gate).
        self.event_key = (current_event or {}).get('event_key')
        self.camera = "main"
        self.calib = None
        self.field_model = None
        self.hub_prisms = []
        self.field_tracker: Optional[FieldTracker] = None
        self.radar: Optional[Radar] = None
        self.field_tracking = False
        # Per-frame occlusion state. `_view_flags` feeds the NEXT frame's tracker
        # (view_occluded -> longer coast); `_occ` is this frame's full result and
        # gates VLM reads (bumper_occluded). Both keyed by slot_id.
        self._view_flags: Dict[int, dict] = {}
        self._occ: Dict[int, dict] = {}

        # Video metadata
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 0
        self.total_frames = 0
        self.overhead_height = 0
        
        # Detection state
        self.hubs: List[DetectedHub] = []
        self.robot_tracks: Dict[int, RobotTrack] = {}
        self.bytetrack_to_canonical: Dict[int, int] = {}
        self.next_canonical_id = 0
        
        # Scoring state
        self.alliance_scores = {"blue": 0, "red": 0}
        self.last_alliance_scores = {"blue": 0, "red": 0}
        self.scoring_events: List[ScoringEvent] = []
        self.recent_ball_detections: deque = deque(maxlen=BALL_PROXIMITY_FRAMES)
        self.last_ocr_crops: list = []     # crops fed to the VLM + reads, for UI spot-check
        self._ocr_crops_dirty = False      # only stream crops on frames where a new batch exists
        
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
            robot = self.robot_tracks.get(canonical_id)
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
            else:
                # Ambiguous partial match — split vote among candidates
                vote_share = EASYOCR_VOTE_WEIGHT / len(matching_teams)
                for team in matching_teams:
                    robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0) + vote_share

            # Check if any team number has enough votes to confirm
            best_team = max(robot.team_ocr_votes, key=robot.team_ocr_votes.get)
            votes = robot.team_ocr_votes[best_team]

            if votes >= OCR_TEAM_CONFIRM_VOTES:
                # Verify this team isn't already assigned to another robot
                already_assigned = any(
                    r.team_number == best_team and r.team_number_confirmed
                    for r in self.robot_tracks.values() if r.robot_id != canonical_id
                )
                if already_assigned:
                    # Remove this candidate and check next best
                    del robot.team_ocr_votes[best_team]
                    continue

                old_team = robot.team_number
                robot.team_number = best_team
                robot.team_number_confirmed = True

                # Alliance is owned by the field-tracker slot (bumper color); a team
                # read records the number but never flips the slot's alliance (SP2).

                # Clear this team from other robots' tentative assignments
                self._clear_team_from_other_robots(canonical_id, best_team)

                # Update scoring events retroactively — fix team numbers on past events
                for event in self.scoring_events:
                    if event.robot_id == canonical_id:
                        event.team_number = best_team

                print(f"  OCR confirmed: Robot {canonical_id} = Team {best_team} "
                      f"(was {old_team}, {votes:.1f} votes, all: {dict(robot.team_ocr_votes)})")

    def vision_identify_robots(self, frame: np.ndarray, tracked_robots) -> None:
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
        if not remaining_teams and self._claude_vision_calls % REREAD_EVERY_N_PASSES != 0:
            return  # everything confirmed and not a re-read pass
        remaining_blue = [t for t in self.match_blue_teams if t not in confirmed_teams]
        remaining_red = [t for t in self.match_red_teams if t not in confirmed_teams]

        # Build candidate crops. Confirmed robots only on re-read passes.
        is_reread_pass = (self._claude_vision_calls % REREAD_EVERY_N_PASSES == 0)
        candidates = []  # list of (score, canonical_id, box, crop_b64)
        for canonical_id, box in tracked_robots:
            robot = self.robot_tracks.get(canonical_id)
            if not robot:
                continue
            if robot.team_number_confirmed and not is_reread_pass:
                continue  # skip confirmed unless this is a re-read pass
            if self._occ.get(canonical_id, {}).get("bumper_occluded"):
                continue  # bumper hidden by hub/person/robot — don't read it

            x1, y1, x2, y2 = map(int, box)
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Bumper band (lower part of the robot box).
            box_h = y2 - y1
            bumper_y1 = y1 + int(box_h * 0.35)
            crop = frame[bumper_y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            score = legibility_score(crop)
            if score < LEGIBILITY_FLOOR:
                # Record the rejected crop so the diagnostic view can show why
                # this robot went unread.
                try:
                    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    rejected_b64 = base64.b64encode(buf).decode('utf-8')
                except Exception as e:
                    print(f"  vlm_filtered: imencode failed for robot {canonical_id}: {e}")
                    rejected_b64 = ""
                self.diagnostic.record_vlm_filtered(
                    frame=self._current_frame, robot_id=canonical_id,
                    crop_b64=rejected_b64, legibility=score,
                    reason="below_legibility_floor",
                )
                continue

            # Upscale so the bumper crop is at least ~384px tall (existing behavior).
            if crop.shape[0] < 384:
                scale = max(2, int(np.ceil(384 / max(1, crop.shape[0]))))
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buf).decode('utf-8')
            candidates.append((score, canonical_id, box, crop_b64))

        if not candidates:
            return

        # Send the top-K legible crops to Qwen.
        candidates.sort(key=lambda c: c[0], reverse=True)
        chosen = candidates[:VLM_TOP_K]

        results = []
        ocr_crops = []
        for score, canonical_id, box, crop_b64 in chosen:
            res = _vision_read_bumpers([crop_b64], remaining_teams, remaining_blue, remaining_red)[0]
            results.append((canonical_id, box, res))
            ocr_crops.append({
                "robot_id": canonical_id,
                "alliance": res.get("alliance", "unknown"),
                "crop": crop_b64,
                "raw_read": res.get("raw_read"),
                "matched": res.get("team_number"),
                "legibility": round(score, 4),
            })
            self.diagnostic.record_vlm_read(
                frame=self._current_frame, robot_id=canonical_id,
                crop_b64=crop_b64,
                legibility=float(score),
                raw_read=res.get("raw_read"),
                matched_team=res.get("team_number"),
                alliance_read=res.get("alliance", "unknown"),
                confidence=float(res.get("confidence", 0.0)),
                vote_weight=1.0 if float(res.get("confidence", 0.0)) >= 0.85 else 0.5,
            )
        self.last_ocr_crops = ocr_crops
        self._ocr_crops_dirty = True

        for canonical_id, box, result in results:
            team = result.get("team_number")
            if not team:
                continue
            robot = self.robot_tracks.get(canonical_id)
            if not robot:
                continue
            conf = result.get("confidence", 0.0)
            weight = 1.0 if conf >= 0.85 else 0.5
            if robot.team_ocr_votes is None:
                robot.team_ocr_votes = {}
            robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + weight
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
            if team is None:
                r.team_number = None
                continue
            v = (r.team_ocr_votes or {}).get(team, 0.0)
            total = sum((r.team_ocr_votes or {}).values()) or 1.0
            r.team_number = team
            r.team_number_confirmed = bool(v >= 3.0 and v / total >= 0.6)
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
            r.team_number = team
            r.team_number_confirmed = True
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

    async def _diagnostic_ws_send(self, record: dict) -> None:
        """Push one diagnostic record over the existing websocket. Best-effort:
        the JSONL file is the canonical record. Disconnect-class exceptions are
        DEBUG-logged; anything unexpected is WARN-logged via print()."""
        if not self.websocket:
            return
        msg_type = ("diagnostic_snapshot"
                   if record.get("type") == "snapshot"
                   else "diagnostic_event")
        try:
            await self.websocket.send_json({"type": msg_type, "record": record})
        except WebSocketDisconnect:
            # Expected when the client tab closes mid-stream — quiet.
            pass
        except (RuntimeError, ConnectionError):
            # "WebSocket is not connected" / broken pipe — also expected on teardown.
            pass
        except Exception as e:
            print(f"  diagnostic_ws_send: unexpected error: {type(e).__name__}: {e}")

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
    
    def update_hub_positions(self, detections: sv.Detections) -> None:
        """Update hub positions and bounding boxes from hub_opening detections.
        If event_hub_positions are set, use them for alliance assignment (static).
        Otherwise fall back to midpoint-based initial assignment."""
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

    def detect_bumper_color(self, frame: np.ndarray, box: np.ndarray) -> str:
        """Detect robot alliance from bumper color using HSV analysis.
        Tuned for elevated camera angles (30-40 degrees) where bumpers may be
        partially occluded and arena lighting can wash out saturation.
        Returns 'blue', 'red', or 'unknown'."""
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        
        # Elevated spectator view: bumpers are usually in lower-middle, not extreme bottom.
        # Also trim side edges to reduce floor/reflection contamination.
        bw = x2 - x1
        bh = y2 - y1
        roi_x1 = x1 + int(bw * 0.10)
        roi_x2 = x2 - int(bw * 0.10)
        roi_y1 = y1 + int(bh * 0.35)
        roi_y2 = y1 + int(bh * 0.95)
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Wider HSV ranges for arena lighting conditions
        # Red bumper (red wraps around hue 0/180) — wider hue range, lower S/V minimums
        red_mask1 = cv2.inRange(hsv, np.array([0, 45, 40]), np.array([16, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 45, 40]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(red_mask1 | red_mask2)
        
        # Blue bumper — wider hue range, lower S/V minimums  
        blue_mask = cv2.inRange(hsv, np.array([88, 45, 40]), np.array([140, 255, 255]))
        blue_pixels = cv2.countNonZero(blue_mask)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return "unknown"
        red_ratio = red_pixels / total_pixels
        blue_ratio = blue_pixels / total_pixels
        
        # Lower threshold tuned for partially visible bumpers at elevated angles.
        if red_ratio > 0.015 and red_ratio > blue_ratio * 1.25:
            return "red"
        elif blue_ratio > 0.015 and blue_ratio > red_ratio * 1.25:
            return "blue"
        return "unknown"

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

    def _get_or_create_slot_track(self, slot_id: int) -> "RobotTrack":
        """Lazily create the RobotTrack for a field-tracker slot (6 max).

        FieldTracker owns position/alliance/box each frame; RobotTrack retains the
        team votes, scores, and diagnostic fields keyed by slot_id."""
        track = self.robot_tracks.get(slot_id)
        if track is None:
            track = RobotTrack(
                robot_id=slot_id,
                alliance=self._slot_alliance(slot_id),
                team_number=None,
                last_position=(0.0, 0.0),
                positions_history=deque(maxlen=30),
            )
            self.robot_tracks[slot_id] = track
        return track

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
            robot.scores += 1
            
            # Calculate distance to hub for confidence
            hub = next((h for h in self.hubs if h.alliance == alliance), None)
            dist = 0
            if hub:
                origin = robot.last_position
                dist = np.sqrt((origin[0] - hub.x)**2 + (origin[1] - hub.y)**2)
            
            confidence = "high"  # Trajectory-based attribution is high confidence
            event = ScoringEvent(
                frame_num=frame_num, timestamp=timestamp, alliance=alliance,
                robot_id=robot.robot_id, team_number=robot.team_number,
                confidence=confidence, distance_to_hub=dist
            )
            self.scoring_events.append(event)
            print(f"  TRAJECTORY attribution [{alliance}]: Robot {robot.team_number or robot.robot_id} "
                  f"(trajectory-based, dist={dist:.0f})")
            return
        
        # FALLBACK: No trajectory data — use proximity to hub
        alliance_robots = [r for r in all_robots if r.alliance == alliance]
        if not alliance_robots:
            alliance_robots = all_robots  # Use any robot if none match alliance
        
        hub = next((h for h in self.hubs if h.alliance == alliance), None)
        if not hub:
            alliance_robots[0].scores += 1
            self.scoring_events.append(ScoringEvent(
                frame_num=frame_num, timestamp=timestamp, alliance=alliance,
                robot_id=alliance_robots[0].robot_id,
                team_number=alliance_robots[0].team_number,
                confidence="low", distance_to_hub=999
            ))
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
        best_robot.scores += 1
        
        confidence = "low"  # Proximity-only is low confidence
        event = ScoringEvent(
            frame_num=frame_num, timestamp=timestamp, alliance=alliance,
            robot_id=best_robot.robot_id, team_number=best_robot.team_number,
            confidence=confidence, distance_to_hub=dist_to_hub
        )
        self.scoring_events.append(event)
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
        cal = field_store.load(self.event_key, self.camera)
        if cal is None:
            self.error = (f"No calibration found for event '{self.event_key}' "
                          f"camera '{self.camera}'. Calibrate the camera first.")
            print(f"  FIELD TRACKING DISABLED: {self.error}")
            return False
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
            self.field_tracker = FieldTracker(self.field_model.length_m)
            self.radar = Radar(self.field_model.length_m, self.field_model.width_m)
        except Exception as e:
            self.error = f"Failed to initialize field tracking: {e}"
            print(f"  FIELD TRACKING DISABLED: {self.error}")
            return False
        self.field_tracking = True
        print(f"  Field tracking ENABLED: event={self.event_key} camera={self.camera} "
              f"season={cal.field_season} mode={cal.mode}")
        return True

    async def process_video(self) -> None:
        """Main video processing pipeline"""
        cap = None
        # Gate on calibration / loaded event (decision 7). No fallback to the old
        # pixel-space pipeline: if field tracking can't be initialized, end cleanly.
        if not self._init_field_tracking():
            self.status = "error"
            self._open_log_file()
            print(f"ABORTING JOB: {self.error}")
            self._close_log_file()
            if self.websocket:
                try:
                    await self.websocket.send_json({"type": "error", "error": self.error})
                except Exception:
                    pass
            return
        try:
            self.status = "processing"
            self._open_log_file()
            
            # Open video
            cap = cv2.VideoCapture(str(self.video_path))
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.overhead_height = int(self.video_height * OVERHEAD_CROP_RATIO)
            self.ocr_regions = scale_ocr_regions(self.video_width, self.overhead_height)
            print(f"Video: {self.video_width}x{self.video_height}, overhead: {self.video_width}x{self.overhead_height}")
            print(f"  OCR regions (overhead coords): {self.ocr_regions}")
            print(f"  Using {'custom' if custom_ocr_regions else 'default'} OCR regions")
            
            # Extract team numbers from early frame (using overhead crop)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
            ret, early_frame = cap.read()
            if ret:
                early_overhead = early_frame[:self.overhead_height, :]
                self.extract_team_numbers(early_overhead)
            # Skip the first 180 frames (title screen / intro graphics)
            TITLE_SCREEN_FRAMES = 180

            # Early vision OCR pass on frame 200 to identify robots before main loop.
            # Seeds the field tracker's slots from this frame so VLM reads bind to
            # slot ids (the same identity the main loop uses).
            EARLY_OCR_FRAME = 200
            cap.set(cv2.CAP_PROP_POS_FRAMES, EARLY_OCR_FRAME)
            ret, early_ocr_frame = cap.read()
            if ret and self.match_teams:
                print(f"  Early vision OCR on frame {EARLY_OCR_FRAME}...")
                early_overhead = early_ocr_frame[:self.overhead_height, :]
                results = self.model(early_overhead, conf=DET_CONF_THRESHOLD, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                robot_mask = detections.class_id == 1
                early_boxes = [detections.xyxy[i] for i in range(len(detections))
                               if robot_mask[i]]
                early_field_dets = to_field_detections(
                    early_boxes, self.calib, early_overhead, self.field_dims,
                    color_fn=self.detect_bumper_color,
                )
                early_dt = FRAME_SKIP / (self.video_fps or 30.0)
                early_outs = self.field_tracker.update(early_field_dets, {}, early_dt)
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
                    self.vision_identify_robots(early_overhead, early_robots)

            cap.set(cv2.CAP_PROP_POS_FRAMES, TITLE_SCREEN_FRAMES)

            frame_num = TITLE_SCREEN_FRAMES
            processed_frame_num = 0
            self._inference_start_time = time.time()
            robots_data = []
            events_data = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames
                if frame_num % FRAME_SKIP != 0:
                    frame_num += 1
                    continue
                
                # Crop overhead portion
                overhead_frame = frame[:self.overhead_height, :]
                
                # Run YOLO
                results = self.model(overhead_frame, conf=DET_CONF_THRESHOLD, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Update hub positions (alliance assignment is static from setup)
                self.update_hub_positions(detections)
                
                # ---- Field-space robot tracking (replaces ByteTrack/canonical re-ID) ----
                # 1. Raw robot detection boxes from YOLO.
                robot_mask = detections.class_id == 1
                robot_boxes = [detections.xyxy[i] for i in range(len(detections))
                               if robot_mask[i]]

                # 2. Persons (occluders) — separate YOLO model, same skip cadence.
                person_results = self.person_model(
                    overhead_frame, classes=[0], conf=DET_CONF_THRESHOLD, verbose=False)[0]
                person_dets = sv.Detections.from_ultralytics(person_results)
                person_boxes_all = [tuple(person_dets.xyxy[i]) for i in range(len(person_dets))]
                person_boxes = field_adjacent_persons(
                    person_boxes_all, self.calib, self.field_dims)

                # 3. Bridge image boxes -> field-space detections (bumper color via
                #    detect_bumper_color, injected as color_fn).
                field_dets = to_field_detections(
                    robot_boxes, self.calib, overhead_frame, self.field_dims,
                    color_fn=self.detect_bumper_color,
                )

                # 4. Drive the field tracker. view_occluded flags from the PREVIOUS
                #    frame widen this frame's coast budget.
                dt = FRAME_SKIP / (self.video_fps or 30.0)
                track_outs = self.field_tracker.update(field_dets, self._view_flags, dt)

                # 5. Resolve occlusion once on the emitted tracks; gate VLM next.
                occ_input = [{"slot_id": t.slot_id, "pos": t.pos, "box": t.box}
                             for t in track_outs]
                self._occ = resolve_occlusion(
                    self.calib, occ_input, person_boxes, self.hub_prisms)
                self._view_flags = self._occ  # feed forward to next frame's tracker

                # 6. Sync RobotTracks by slot_id and build the box list for HUD/VLM.
                tracked_robot_boxes = []  # (slot_id, box) for visible tracks
                for t in track_outs:
                    robot = self._get_or_create_slot_track(t.slot_id)
                    robot.alliance = t.alliance  # slot (bumper-color) alliance wins
                    robot.field_position = t.pos  # field meters, for consumers that want it
                    if t.state == "visible" and t.box is not None:
                        cx = (t.box[0] + t.box[2]) / 2.0
                        cy = (t.box[1] + t.box[3]) / 2.0
                        robot.last_position = (float(cx), float(cy))  # pixel space (scoring)
                        robot.positions_history.append((processed_frame_num, float(cx), float(cy)))
                        robot.last_seen_frame = frame_num
                        tracked_robot_boxes.append((t.slot_id, t.box))
                    # Coasting tracks keep their stale last_position untouched.

                # Map detection index -> slot_id for the annotation pass (a track's
                # box is the same ndarray its field detection was built from).
                detection_to_tracked = {}  # det_index -> slot_id
                for slot_id, tbox in tracked_robot_boxes:
                    for di in range(len(detections)):
                        if detections.class_id[di] != 1:
                            continue
                        db = detections.xyxy[di]
                        if (abs(db[0] - tbox[0]) < 1 and abs(db[1] - tbox[1]) < 1 and
                                abs(db[2] - tbox[2]) < 1 and abs(db[3] - tbox[3]) < 1):
                            detection_to_tracked[di] = slot_id
                            break

                # VLM — read bumper numbers on a time interval. Qwen3-VL is LOCAL and
                # free, so there's no per-match call cap (the old 20-call budget was a
                # paid-API leftover); it runs the whole match until robots are identified.
                current_time = time.time()
                elapsed = current_time - self._inference_start_time if hasattr(self, '_inference_start_time') else 999
                vision_interval = 2 if elapsed < 60 else self._claude_vision_interval_sec
                unconfirmed_count = sum(
                    1 for r in self.robot_tracks.values() if not r.team_number_confirmed
                )
                # Let the call through even when everything is confirmed if the
                # next pass is a re-read pass — that's how a wrong lock self-corrects.
                is_reread_due = ((self._claude_vision_calls + 1) % REREAD_EVERY_N_PASSES == 0)
                if ((unconfirmed_count > 0 or is_reread_due) and
                        tracked_robot_boxes and
                        processed_frame_num >= 10 and
                        current_time - self._claude_vision_last_time >= vision_interval):
                    self._claude_vision_calls += 1
                    self._claude_vision_last_time = current_time
                    self.vision_identify_robots(overhead_frame, tracked_robot_boxes)

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
                
                # Draw annotations
                annotated_frame = overhead_frame.copy()
                
                # Draw detections
                for i in range(len(detections)):
                    box = detections.xyxy[i]
                    class_id = detections.class_id[i]
                    
                    x1, y1, x2, y2 = map(int, box)
                    
                    if class_id == 0:  # fuel_ball
                        color = (0, 255, 255)  # Yellow
                        label = "ball"
                    elif class_id == 1:  # robot
                        # Use the detection→tracked mapping built above
                        if i in detection_to_tracked:
                            slot_id = detection_to_tracked[i]
                            robot = self.robot_tracks[slot_id]
                            alliance = robot.alliance
                            confirmed = robot.team_number_confirmed
                            if confirmed:
                                color = (255, 50, 50) if alliance == "blue" else (50, 50, 255)
                                label = f"{robot.team_number} (100%)"
                            elif robot.team_number:
                                # Compute confidence from best vote ratio
                                conf = 0
                                if robot.team_ocr_votes:
                                    best = max(robot.team_ocr_votes.values())
                                    conf = max(conf, min(99, int(best / OCR_TEAM_CONFIRM_VOTES * 100)))
                                if robot.appearance_votes:
                                    best = max(robot.appearance_votes.values())
                                    conf = max(conf, min(99, int(best / APPEARANCE_CONFIRM_VOTES * 100)))
                                color = (180, 100, 100) if alliance == "blue" else (100, 100, 180)
                                label = f"{robot.team_number}? ({conf}%)"
                            else:
                                color = (150, 150, 150)
                                label = self._slot_label(slot_id)
                        else:
                            # Untracked robot detection — use bumper color for display only
                            alliance = self.detect_bumper_color(overhead_frame, detections.xyxy[i])
                            color = (255, 0, 0) if alliance == "blue" else (0, 0, 255) if alliance == "red" else (200, 200, 200)
                            confirmed = False
                            label = "robot"
                    else:  # hub_opening
                        color = (0, 255, 0)  # Green
                        label = "hub"
                    
                    if class_id == 1 and i in detection_to_tracked and confirmed:
                        # Confirmed robot: thick solid box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    elif class_id == 1 and i in detection_to_tracked:
                        # Tentative/unidentified robot: thin box with corner ticks
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    # Label with background for readability
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
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
                    else:
                        rect_color = (0, 0, 255)  # Red in BGR
                    cv2.rectangle(annotated_frame, (rx, ry), (rx + rw, ry + rh), rect_color, 2)
                    cv2.putText(annotated_frame, region_key,
                               (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rect_color, 1)

                # Radar picture-in-picture (bottom-right): top-down field with tracks.
                radar_img = self.radar.render(
                    [(t.pos, t.alliance) for t in track_outs if t.state != "empty"])
                rh, rw = radar_img.shape[:2]
                rscale = min(1.0, 360.0 / rw)
                rsz = cv2.resize(radar_img, (int(rw * rscale), int(rh * rscale)))
                ah, aw = annotated_frame.shape[:2]
                ry0, rx0 = ah - rsz.shape[0] - 8, aw - rsz.shape[1] - 8
                if ry0 > 0 and rx0 > 0:
                    annotated_frame[ry0:ry0 + rsz.shape[0], rx0:rx0 + rsz.shape[1]] = rsz

                # Encode frame
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Prepare robot data
                robots_data = []
                for robot in self.robot_tracks.values():
                    robots_data.append({
                        "id": robot.robot_id,
                        "alliance": robot.alliance,
                        "team_number": robot.team_number,
                        "confirmed": robot.team_number_confirmed,
                        "scores": robot.scores,
                        "position": robot.last_position
                    })
                
                # Prepare scoring events
                events_data = []
                for event in self.scoring_events[-10:]:  # Last 10 events
                    events_data.append({
                        "frame": event.frame_num,
                        "timestamp": event.timestamp,
                        "alliance": event.alliance,
                        "robot_id": event.robot_id,
                        "team_number": event.team_number,
                        "confidence": event.confidence
                    })
                
                # Send frame data via WebSocket
                if self.websocket:
                    # Only stream the VLM crops on frames where a fresh batch exists,
                    # to avoid resending the same images every frame.
                    ocr_crops_payload = self.last_ocr_crops if self._ocr_crops_dirty else None
                    self._ocr_crops_dirty = False
                    try:
                        await self.websocket.send_json(jsonify({
                            "type": "frame",
                            "frame_num": frame_num,
                            "processed_frame_num": processed_frame_num,
                            "total_frames": self.total_frames,
                            "fps": self.video_fps,
                            "frame_data": frame_b64,
                            "scores": self.alliance_scores,
                            "robots": robots_data,
                            "scoring_events": events_data,
                            "ocr_crops": ocr_crops_payload
                        }))
                        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                    except Exception as e:
                        print(f"WebSocket send error: {e}")
                        break

                self._current_frame = frame_num

                if self._stopped:
                    break

                if self._paused:
                    self._pause_frame = self._current_frame
                    loop = asyncio.get_running_loop()
                    last_status_sent = 0.0
                    while self._paused and not self._stopped:
                        now = loop.time()
                        if self.websocket and (now - last_status_sent >= 1.0):
                            try:
                                await self.websocket.send_json({
                                    "type": "status",
                                    "status": "paused",
                                    "frame_num": self._pause_frame
                                })
                            except Exception as e:
                                print(f"WebSocket pause status send error: {e}")
                                self._stopped = True
                                break
                            last_status_sent = now
                        await asyncio.sleep(0.1)

                    if self._stopped:
                        break
                
                frame_num += 1
                processed_frame_num += 1
            
            self.status = "completed"
            
            if self.websocket:
                await self.websocket.send_json(jsonify({
                    "type": "complete",
                    "total_frames": self.total_frames,
                    "robots": robots_data,
                    "scoring_events": events_data,
                    "final_scores": self.alliance_scores
                }))
        
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            import traceback
            print(f"ERROR in process_video: {e}")
            print(traceback.format_exc())
            if self.websocket:
                await self.websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
        finally:
            try:
                self.diagnostic.close()
            except Exception:
                pass
            self._close_log_file()
            if cap is not None:
                cap.release()


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


@app.get("/api/status")
async def status():
    """Return model and system status"""
    return {
        "model": "YOLO26s Hub-Zone",
        "model_path": str(MODEL_PATH),
        "model_classes": {0: "fuel_ball", 1: "robot", 2: "hub_opening"},
        "status": "ready",
        "has_event": current_event is not None,
        "event_key": current_event['event_key'] if current_event else None,
        "event_name": current_event['event_name'] if current_event else None,
    }


@app.post("/api/ocr-regions")
async def set_ocr_regions(request: Request):
    """Set custom OCR regions for score reading. 
    Coordinates are in the overhead crop coordinate space (1920x723 for 1080p).
    Body: {"red_center": [x, y, w, h], "blue_center": [x, y, w, h]}"""
    global custom_ocr_regions
    body = await request.json()
    custom_ocr_regions = {}
    for key in ["red_center", "blue_center"]:
        if key in body:
            coords = body[key]
            custom_ocr_regions[key] = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    
    # Persist to database for this event
    if current_event:
        from database import save_ocr_regions
        await save_ocr_regions(current_event['event_key'], custom_ocr_regions)
    
    return {"status": "ok", "regions": custom_ocr_regions}

@app.get("/api/ocr-regions")
async def get_ocr_regions():
    """Get current OCR regions."""
    regions = custom_ocr_regions if custom_ocr_regions else OCR_REGIONS_1080P
    return {"regions": {k: list(v) for k, v in regions.items()}, "custom": custom_ocr_regions is not None}

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
    return {"has_calibration": cal is not None,
            "rms_error": cal.rms_error if cal else None}


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
    K, D = field_store.load_intrinsics(event, camera)
    if K is None:
        return JSONResponse(status_code=400, content={
            "error": "No saved intrinsics — run auto calibration first."})
    pts = body.get("points", [])
    if len(pts) < 2:
        return JSONResponse(status_code=400, content={
            "error": "Need at least 2 clicked points."})
    corr = {f"pt{i}": (np.array([[p["u"], p["v"]]], float),
                       np.array([[p["x"], p["y"], p["z"]]], float))
            for i, p in enumerate(pts)}
    existing = field_store.load(event, camera)
    image_size = existing.image_size if existing else tuple(body.get("image_size", (1920, 1080)))
    season = existing.field_season if existing else body.get("season", "2026-rebuilt-welded")
    try:
        cal = Calibrator.solve_pose(corr, K=K, D=D, image_size=image_size, field_season=season)
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    field_store.save(event, camera, cal)
    return {"status": "ok", "rms_error": cal.rms_error}


@app.post("/api/calibration/auto")
async def calibration_auto(request: Request):
    """Solve POSE from AprilTags detected in supplied frames, against saved
    intrinsics. Body: {event, camera?, season?, frames: [b64,...], image_size?}.
    Frames are pre-grabbed (e.g. via /api/grab-frame)."""
    body = await request.json()
    event = body["event"]; camera = body.get("camera", "main")
    season = body.get("season", "2026-rebuilt-welded")
    K, D = field_store.load_intrinsics(event, camera)
    if K is None:
        return JSONResponse(status_code=400, content={
            "error": "No intrinsics — import or run multi-viewpoint "
                     "intrinsics calibration first."})
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
                                    field_season=season)
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    field_store.save(event, camera, cal)
    return {"status": "ok", "rms_error": cal.rms_error, "n_tags": len(corr)}


@app.post("/api/calibration/frame")
async def calibration_frame(request: Request):
    """Grab a frame from a LOCAL video, cropped to the requested camera's region
    from the composite layout (masks applied). Body: {timestamp?: seconds
    (default 75), video_path?, camera?}.
    Returns {frame: b64 jpeg, width, height} where width/height are the CROPPED size.
    Pixel coords are crop-local (origin at the camera crop's top-left)."""
    body = await request.json()
    ts = float(body.get("timestamp", 75))
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
            return JSONResponse(status_code=400,
                                content={"error": "video_path outside allowed directory"})
        if candidate.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
            return JSONResponse(status_code=400, content={"error": "unsupported video extension"})
        path = candidate
    if not path.exists():
        return JSONResponse(status_code=400, content={"error": "video not found"})
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return JSONResponse(status_code=400, content={"error": "could not read frame"})
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


def _draw_field_overlay(frame, cal, fm):
    """Draw the reprojected field perimeter + 2 m grid using cal.field_to_image.
    Field lines are SAMPLED into polylines (a straight field line projects to a curved
    image line under lens distortion, so a 2-point segment can't trace it).
    Returns fp(x,y)->(int,int) so callers can add their own markers."""
    L, W = fm.length_m, fm.width_m
    def fp(x, y):
        u, v = cal.field_to_image((x, y, 0.0))
        return (int(u), int(v))
    def seg(p0, p1, color, thick, n=24):
        (x0, y0), (x1, y1) = p0, p1
        pts = [fp(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n) for t in range(n + 1)]
        cv2.polylines(frame, [np.array(pts, np.int32)], False, color, thick)
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0))]:
        seg(a, b, (0, 255, 0), 2)
    x = 0.0
    while x <= L + 1e-6:
        seg((x, 0), (x, W), (0, 180, 0), 1); x += 2.0
    y = 0.0
    while y <= W + 1e-6:
        seg((0, y), (L, y), (0, 180, 0), 1); y += 2.0
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
    field_store.save(event, camera, cal)

    # Build overlay on the same frame/timestamp.
    fm = FieldModel.load("2026-rebuilt-welded")
    ts = float(body.get("timestamp", 75))
    cap = cv2.VideoCapture(str(CALIB_TEST_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    okf, frame = cap.read(); cap.release()
    overlay_b64 = None
    if okf and frame is not None:
        main = frame[: int(frame.shape[0] * CALIB_MAIN_CROP_FRAC), :]
        _draw_field_overlay(main, cal, fm)
        # clicked points
        for p in pts:
            cv2.circle(main, (int(p["u"]), int(p["v"])), 5, (0,0,255), -1)
        okj, buf = cv2.imencode(".jpg", main, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if okj:
            overlay_b64 = base64.b64encode(buf.tobytes()).decode()
    return {"status": "ok", "rms_error": cal.rms_error, "n_points": len(pts),
            "overlay": overlay_b64}


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
    try:
        cal = Calibrator.solve_camera_from_tags(corr, image_size=image_size,
                                                field_season=season)
    except CalibrationError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    field_store.save(event, camera, cal)

    used = {int(s) for s in auto.keys()} | {int(m["id"]) for m in manual}
    n_points = sum(len(c[0]) for c in corr.values())
    overlay_b64 = None
    ts = float(body.get("timestamp", 75))
    cap = cv2.VideoCapture(str(CALIB_TEST_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    okf, frame = cap.read(); cap.release()
    if okf and frame is not None:
        main = frame[: int(frame.shape[0] * CALIB_MAIN_CROP_FRAC), :]
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
    return {"status": "ok", "rms_error": cal.rms_error, "n_tags": len(corr),
            "n_points": n_points, "overlay": overlay_b64}


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
                          blue_teams: list, red_teams: list) -> list:
    """Read bumper numbers using best available method.
    Currently: Qwen VL only (local, free). Claude Vision fallback disabled for testing."""
    try:
        return _qwen_vl_read_bumpers(robot_crops_b64, match_teams, blue_teams, red_teams)
    except Exception as e:
        print(f"  Qwen VL failed: {e}")
        return [{"team_number": None, "alliance": "unknown", "confidence": 0}] * len(robot_crops_b64)

    # Claude Vision fallback (disabled for now)
    # return _claude_vision_read_bumpers(robot_crops_b64, match_teams, blue_teams, red_teams)


def _detect_bumper_color_static(frame: np.ndarray, box) -> str:
    """Static version of bumper color detection for use outside InferenceJob."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    bw = x2 - x1
    bh = y2 - y1
    roi_x1 = x1 + int(bw * 0.10)
    roi_x2 = x2 - int(bw * 0.10)
    roi_y1 = y1 + int(bh * 0.35)
    roi_y2 = y1 + int(bh * 0.95)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, np.array([0, 45, 40]), np.array([16, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 45, 40]), np.array([180, 255, 255]))
    red_pixels = cv2.countNonZero(red_mask1 | red_mask2)
    blue_mask = cv2.inRange(hsv, np.array([88, 45, 40]), np.array([140, 255, 255]))
    blue_pixels = cv2.countNonZero(blue_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return "unknown"
    red_ratio = red_pixels / total_pixels
    blue_ratio = blue_pixels / total_pixels
    if red_ratio > 0.015 and red_ratio > blue_ratio * 1.25:
        return "red"
    elif blue_ratio > 0.015 and blue_ratio > red_ratio * 1.25:
        return "blue"
    return "unknown"


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


def _qwen_vl_read_bumpers(robot_crops_b64: list, match_teams: list,
                           blue_teams: list, red_teams: list) -> list:
    """Use the local VLM (VLM_MODEL_NAME, default Qwen3-VL) to read bumper numbers from robot crops.

    Processes each crop individually (Qwen VL handles multi-image differently from Claude).
    Returns list of {"team_number": str|None, "alliance": str, "confidence": float}.
    """
    import torch
    try:
        model, processor = _get_qwen_vl()
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        print(f"  Qwen VL load error: {e}")
        return [{"team_number": None, "alliance": "unknown", "confidence": 0}] * len(robot_crops_b64)

    teams_str = ", ".join(match_teams)
    blue_str = ", ".join(blue_teams)
    red_str = ", ".join(red_teams)

    results = []
    for crop_b64 in robot_crops_b64:
        try:
            # Decode base64 to PIL Image
            import io
            from PIL import Image
            img_bytes = base64.b64decode(crop_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Aggressively upscale small crops so Qwen can read the numbers
            min_dim = 512
            if img.width < min_dim or img.height < min_dim:
                scale = max(min_dim / img.width, min_dim / img.height)
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

            prompt = (
                "This image shows ONE FRC robot from a side or elevated camera angle. "
                "Around the base of the robot is a bumper: a solid RED or BLUE band with a "
                "WHITE team number printed on it. FRC team numbers are 1 to 5 digits "
                "(for example 294, 2485, 4322).\n\n"
                "Carefully read the WHITE digits on the bumper, left to right. Report the "
                "exact number you SEE — do NOT guess, and do NOT choose from any list.\n\n"
                'Reply with ONLY compact JSON and nothing else: '
                '{"team": "<the digits, or null if you cannot read them>", '
                '"color": "red" or "blue"}.'
            )

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=64)

            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            print(f"    Qwen input image: {img.size}, raw output: {output_text}")

            # Parse JSON from response
            team = None
            color = "unknown"
            try:
                # Handle markdown code blocks
                clean = output_text
                if "```" in clean:
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                    clean = clean.strip()
                # Find JSON object
                start = clean.find("{")
                end = clean.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(clean[start:end])
                    team = parsed.get("team")
                    color = parsed.get("color", "unknown")
                    if team is not None:
                        team = str(team)
            except (json.JSONDecodeError, Exception):
                pass

            print(f"    Qwen parsed: team={team}, color={color}, candidates={teams_str}")

            # Validate what Qwen READ against the roster, in code — so the prompt can
            # never bias the model toward a particular team. Use the reported color to
            # constrain which alliance's teams we match against.
            read = "".join(ch for ch in team if ch.isdigit()) if team else ""
            alliance = color if color in ("blue", "red") else "unknown"
            matched, confidence = None, 0.0
            if read:
                # Match ONLY within the alliance the model read off the bumper. The
                # per-crop color is reliable, so a misread never maps cross-alliance
                # (a blue robot's "432" can't become red 4322).
                color_cands = (red_teams if alliance == "red"
                               else blue_teams if alliance == "blue"
                               else match_teams)
                if read in color_cands:
                    matched, confidence = read, 0.9
                else:
                    cand, dist = _closest_team(read, color_cands)
                    if cand and dist <= 1 and len(read) >= 2:
                        matched, confidence = cand, 0.6
                        print(f"    Qwen fuzzy match: read '{read}' ({alliance}) -> '{cand}' (edit dist {dist})")
                    else:
                        print(f"    Qwen read '{read}' ({alliance}) — no match in {color_cands}")

            results.append({
                "team_number": matched,
                "alliance": alliance,
                "confidence": confidence,
                "raw_read": read,
            })

        except Exception as e:
            print(f"  Qwen VL error on crop: {e}")
            results.append({"team_number": None, "alliance": "unknown", "confidence": 0})

    identified = sum(1 for r in results if r["team_number"])
    print(f"  Qwen VL: identified {identified}/{len(robot_crops_b64)} robots: "
          f"{[r['team_number'] for r in results]}")
    return results


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


def _detect_robots_in_frame(frame: np.ndarray, model, match_teams: list,
                            blue_teams: list, red_teams: list) -> list:
    """Detect robots in a single frame. Returns list of robot dicts."""
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    robot_mask = detections.class_id == 1
    robot_dets = detections[robot_mask]

    robots = []
    for i in range(len(robot_dets)):
        box = robot_dets.xyxy[i]
        conf = float(robot_dets.confidence[i])
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        alliance = _detect_bumper_color_static(frame, box)

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

        robots = _detect_robots_in_frame(frame, model, match_teams, blue_teams, red_teams)
        for r in robots:
            r["frame_index"] = frame_idx
            r.pop("frame_source", None)
        all_robots.extend(robots)

    # Use Claude Vision for bumper number reading (much better than EasyOCR)
    if all_robots and match_teams:
        crops_b64 = [r["crop"] for r in all_robots]
        claude_results = _vision_read_bumpers(crops_b64, match_teams, blue_teams, red_teams)
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


@app.post("/api/infer")
async def infer(
    video_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None)
):
    """Start inference job"""
    job_id = str(uuid.uuid4())
    
    try:
        # Handle video source
        if video_file:
            # Save uploaded file
            video_path = UPLOAD_DIR / f"{job_id}_{video_file.filename}"
            with open(video_path, "wb") as f:
                shutil.copyfileobj(video_file.file, f)
        
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
        
        # Create job
        job = InferenceJob(job_id, video_path)
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


@app.websocket("/ws/infer/{job_id}")
async def websocket_infer(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for streaming inference results"""
    await websocket.accept()
    
    if job_id not in jobs:
        await websocket.send_json({"type": "error", "error": "Job not found"})
        await websocket.close()
        return
    
    job = jobs[job_id]
    job.websocket = websocket
    job.diagnostic.set_ws_send(job._diagnostic_ws_send)

    # Wait briefly for match team data from frontend
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
        data = json.loads(msg)
        if data.get("type") == "robot_identifications":
            # Store match teams for continuous bumper OCR during inference
            job.match_teams = data.get("match_teams", [])
            job.match_blue_teams = data.get("blue_teams", [])
            job.match_red_teams = data.get("red_teams", [])
    except (asyncio.TimeoutError, Exception):
        pass

    try:
        await job.process_video()
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        print(f"Error in websocket_infer: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    finally:
        try:
            job.diagnostic.set_ws_send(None)
        except Exception:
            pass
        job.websocket = None
        try:
            await websocket.close()
        except:
            pass


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

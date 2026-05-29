# FRC Vision Scorer - YOLO26s Rebuild

**Clean rebuild focused on per-robot scoring attribution for the YOLO26s Hub-Zone model.**

## What Changed

This is a complete rebuild of the web app, removing all legacy v2/v4/v5 code paths and focusing exclusively on the YOLO26s model with intelligent per-robot scoring attribution.

### Previous Issues
- Messy codebase with multiple model versions
- Manual setup phase with drag-and-drop hub positioning
- No automated scoring attribution
- Small video canvas
- Loading overlays that blocked the video

### New Architecture

**Backend** (`web/app.py`):
- Single-file FastAPI application
- WebSocket-based real-time streaming
- Automatic hub detection from `hub_opening` class
- ByteTrack robot tracking with ID consolidation (max 3 per alliance)
- Intelligent scoring attribution based on:
  - Distance to hub
  - Movement direction
  - Ball proximity (robot and hub)
- OCR-based scoreboard reading with monotonicity validation
- Team number extraction from early frames

**Frontend** (`web/static/`):
- Clean dark theme UI
- Full-width video canvas
- Real-time HUD overlay (frame counter, FPS, alliance scores)
- Per-robot scoring tables (3 blue + 3 red robots)
- Chronological scoring timeline with confidence indicators
- Status bar instead of blocking overlays

## Running the Server

```bash
cd ~/development/projects/frc-vision-scorer
source .venv/bin/activate
uvicorn web.app:app --host 0.0.0.0 --port 8004
```

Access at: http://localhost:8004 (or http://frc-vision.dev if Caddy is configured)

## API Endpoints

### `GET /`
Serves the frontend HTML

### `GET /api/status`
Returns model information and status
```json
{
  "model": "YOLO26s Hub-Zone",
  "model_path": "...",
  "model_classes": {
    "0": "fuel_ball",
    "1": "robot", 
    "2": "hub_opening"
  },
  "status": "ready"
}
```

### `POST /api/infer`
Start inference job
- **Body**: FormData with either:
  - `video_file`: Video file upload
  - `youtube_url`: YouTube video URL
- **Returns**: `{"job_id": "..."}`

### `WebSocket /ws/infer/{job_id}`
Streaming inference results

**Message Types:**
- `frame`: Real-time frame data with detections, scores, robots, events
- `complete`: Final results when processing finishes
- `error`: Error information if processing fails

## Features

### Automatic Detection
- **Hubs**: Detected from `hub_opening` class, sorted by x-position (left=blue, right=red)
- **Team Numbers**: OCR'd from scoreboard in early frames
- **Alliance Assignment**: Based on field position (left half = blue, right = red)

### Robot Tracking
- ByteTrack with 150-frame lost track buffer
- ID consolidation to max 3 canonical robots per alliance
- 120px merge threshold for duplicate IDs
- Persistent tracking across frames

### Scoring Attribution
The **critical feature** — when an alliance score increases:

1. **Find all robots** of that alliance currently tracked
2. **Score each robot** by multiple factors:
   - Distance to their alliance hub (inverse relationship)
   - Recent movement direction (approaching hub gets bonus)
   - Ball proximity to robot (within 100px, last 30 frames)
   - Ball proximity to hub (within 100px, last 30 frames)
3. **Attribute to highest-scoring robot**
4. **Confidence levels**:
   - High: <150px from hub
   - Medium: 150-300px from hub
   - Low: >300px from hub
5. **Track per-robot scores** for display

### OCR Scoreboard Reading
- Reads every 5th processed frame
- Multiple region fallbacks (corner + center positions)
- 3x resize + thresholding for better accuracy
- Monotonicity validation (reject backwards/spike >20)
- Config: `--psm 7 -c tessedit_char_whitelist=0123456789`

## Testing

The app has been tested with:
- Test videos: `data/videos/2026/week0_f1m{1,2,3}.mp4`
- Video specs: 1280x720, 30fps
- Overhead crop ratio: 0.67 (top 67% of frame)

## Model Details

**YOLO26s Hub-Zone Model:**
- Path: `runs/train/hub_zone_yolo26s/weights/best.pt`
- Size: ~19MB
- Classes: `{0: 'fuel_ball', 1: 'robot', 2: 'hub_opening'}`
- Performance:
  - fuel_ball: 72.8% mAP50
  - robot: 89.2% mAP50
  - hub_opening: 99.5% mAP50

**Training:**
- Trained on 640x482 alliance-side crops
- Works on full overhead (YOLO auto-resizes to 640)
- No red/blue robot distinction in model (determined by field position)

## Configuration

Key parameters in `web/app.py`:

```python
OVERHEAD_CROP_RATIO = 0.67        # Top 67% of frame
FRAME_SKIP = 2                    # Process every 2nd frame
LOST_TRACK_BUFFER = 150           # ByteTrack buffer
ROBOT_MERGE_THRESHOLD = 120       # px distance for ID merging
MAX_ROBOTS_PER_ALLIANCE = 3       # Max canonical robots

OCR_FRAME_INTERVAL = 5            # OCR every 5th processed frame
BALL_PROXIMITY_FRAMES = 30        # Ball tracking lookback
HUB_DISTANCE_HIGH_CONF = 150      # px for high confidence
HUB_DISTANCE_MED_CONF = 300       # px for medium confidence
```

## Files

```
web/
├── app.py                 # FastAPI backend (complete rewrite)
├── static/
│   ├── index.html        # Single-page frontend
│   ├── style.css         # Dark theme styling
│   └── app.js            # Frontend logic with WebSocket handling
├── uploads/              # Temporary video storage
└── README_REBUILD.md     # This file
```

## Dependencies

All dependencies are already installed in `.venv/`:
- ultralytics (YOLO26s)
- supervision (ByteTrack)
- pytesseract (OCR)
- opencv-python (video processing)
- fastapi, uvicorn
- python-multipart
- yt-dlp (YouTube downloads)

## Future Improvements

Potential enhancements:
- Export results to JSON/CSV
- Video playback controls (pause, seek)
- Configurable OCR regions for different video resolutions
- Training data export for scoring events
- Historical comparison across multiple matches

## Notes

- The app uses **only** the YOLO26s model — no v2/v4/v5 support
- No setup phase or manual hub positioning required
- All detection is automatic from model output
- Focus is entirely on accurate per-robot scoring attribution
- Video canvas is full-width for easy viewing
- Status bar (not overlay) for non-blocking progress updates

---

**Built:** 2026-02-28  
**Model:** YOLO26s Hub-Zone  
**Purpose:** Per-robot scoring attribution for FRC match analysis

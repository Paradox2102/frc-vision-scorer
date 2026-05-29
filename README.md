# FRC Vision Scorer

Real-time computer vision system for FRC (FIRST Robotics Competition) match analysis. Uses a custom-trained YOLO model to detect robots, fuel balls, and hub openings from match video, then attributes scored points to individual robots using scoreboard OCR combined with ball trajectory analysis.

Built for the FRC 2026 REBUILT game — yellow Fuel balls (5.91" foam) scored into a central Hub, 1 point each.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)

---

## What It Does

- **Detects robots, fuel balls, and hub openings** in match video using a custom YOLO26s model
- **Tracks robots across frames** with ByteTrack, consolidating fragmented IDs into stable identities
- **Identifies robot alliances** (blue/red) from bumper colors using HSV color detection
- **Reads the scoreboard via OCR** — the authoritative source for alliance point totals
- **Tracks ball trajectories** — when the scoreboard shows a point increase, the nearest ball trajectory headed toward the hub identifies which robot scored
- **Streams results live** via WebSocket — watch detections, scores, and robot tracking in real-time
- **Integrates with The Blue Alliance** for event schedules, team lists, and ground-truth scores
- **Saves results** to a local SQLite database for review and correction

---

## How It Works

The scoring pipeline has two complementary systems:

1. **Scoreboard OCR (authoritative)** — Reads the red and blue alliance scores directly from the broadcast scoreboard overlay. When OCR detects a score increase, it sets the alliance total. This is the ground truth for point totals.

2. **Ball trajectory tracking (attribution)** — Tracks fuel balls frame-to-frame and detects when a ball moves from an alliance's side of the field toward the central hub. When OCR reports a new point, the system checks recent ball trajectories to determine which robot likely scored. This provides **per-robot** attribution that OCR alone cannot.

Directional gates ensure only balls moving toward the hub count (not stationary ground balls or balls rolling away). A minimum speed threshold filters out slow-drifting objects.

---

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| **CPU** | Any modern x86_64 or ARM64 (Intel i5+, AMD Ryzen 5+, Apple M1+) |
| **RAM** | 8 GB |
| **Storage** | 3 GB free (Python environment + model weights) |
| **GPU** | None required — runs on CPU |
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 10+ (WSL2 recommended) |

### Recommended

| Component | Recommendation |
|-----------|----------------|
| **CPU** | Apple M1/M2/M3/M4 or Intel i7+ / AMD Ryzen 7+ |
| **RAM** | 16 GB |
| **GPU** | Apple Silicon (MPS) or NVIDIA GPU with CUDA |
| **Storage** | 5 GB free |

### Performance

- **Apple Silicon (M1+):** Best consumer experience. PyTorch MPS acceleration gives near-real-time processing. Mac mini M4 processes a 3-minute match in ~2 minutes.
- **NVIDIA GPU (CUDA):** Fastest. Any GPU with 4+ GB VRAM (GTX 1060+, RTX 2060+). Model is only 19 MB.
- **CPU-only:** Works but 5-10x slower. A 3-minute match may take 10-20 minutes.

---

## Prerequisites

Install these **before** setting up the application.

### 1. Python 3.12+

```bash
# macOS
brew install python@3.12

# Ubuntu/Debian
sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev

# Verify
python3.12 --version
```

### 2. Tesseract OCR

Required for scoreboard reading.

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Verify
tesseract --version
# Should print version 4.x or 5.x
```

### 3. ffmpeg

Required for YouTube video frame extraction.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### 4. Git + Git LFS

The trained model weights (19 MB) are stored via [Git Large File Storage](https://git-lfs.com/).

```bash
# macOS
brew install git git-lfs

# Ubuntu/Debian
sudo apt install git git-lfs

# Initialize LFS (one-time, required before cloning)
git lfs install
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ohmeow-ship/frc-vision-scorer.git
cd frc-vision-scorer
```

> **Important:** You must have `git lfs install` run before cloning. If you cloned without LFS, the model file will be a tiny pointer instead of the actual weights. Fix it with: `git lfs pull`

### Step 2: Create Python Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip

pip install \
  torch torchvision \
  ultralytics \
  opencv-python-headless \
  fastapi \
  "uvicorn[standard]" \
  python-multipart \
  pytesseract \
  yt-dlp \
  supervision \
  numpy \
  Pillow \
  aiosqlite \
  httpx \
  python-dotenv
```

**Platform-specific PyTorch options:**

| Platform | Install Command |
|----------|----------------|
| **Apple Silicon (default, includes MPS)** | `pip install torch torchvision` |
| **NVIDIA GPU (CUDA 12.1)** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **CPU-only (smaller download)** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

### Step 4: Verify Model Weights

```bash
# Check the model file exists and is ~19 MB
ls -lh runs/train/hub_zone_yolo26s/weights/best.pt

# If the file is tiny (< 1 KB), LFS didn't pull the actual weights:
git lfs pull

# Verify the model loads correctly
source .venv/bin/activate
python -c "from ultralytics import YOLO; m = YOLO('runs/train/hub_zone_yolo26s/weights/best.pt'); print('Model loaded:', m.names)"
```

Expected output:
```
Model loaded: {0: 'fuel_ball', 1: 'robot', 2: 'hub_opening'}
```

### Step 5: Get a TBA API Key

The app uses [The Blue Alliance](https://www.thebluealliance.com/) API for event data.

1. Go to https://www.thebluealliance.com/account
2. Create an account or sign in
3. Generate a **Read API Key**
4. Create a `.env` file in the project root:

```bash
echo "TBA_API_KEY=your_key_here" > .env
```

### Step 6: Verify Everything

```bash
# Activate the environment
source .venv/bin/activate

# Quick smoke test — should start without errors
cd frc-vision-scorer
python web/app.py &
sleep 3

# Check it's running
curl -s http://localhost:8004/api/health | python -m json.tool

# Stop the test
kill %1
```

---

## Running the Application

### Start

From the project root directory:

```bash
source .venv/bin/activate
python web/app.py
```

Or use the convenience script:

```bash
./web/start.sh
```

The server starts at **http://localhost:8004**. Open this URL in your browser.

### Stop

```bash
# If running in foreground: Ctrl+C

# If running in background via start.sh:
./web/stop.sh

# Or manually:
kill $(lsof -t -i :8004)
```

---

## Usage Guide

This is the complete workflow from loading an event to saving scored results.

### Step 1: Load an Event

When you first open the app, you'll see the **Event Setup** screen.

1. Paste a Blue Alliance event URL or event key into the input field
   - Full URL: `https://www.thebluealliance.com/event/2026tuis`
   - Or just the key: `2026tuis`
2. Click **Load Event**
3. The app fetches from TBA: all teams, match schedule, alliance assignments, YouTube video links, and score breakdowns

Once loaded, you'll see the **Match Setup** screen with an event banner showing the event name, team count, and match count.

### Step 2: Select a Match

1. Choose a match from the dropdown (organized by Qualifications, Semifinals, Finals)
2. Each option shows: match label, blue/red alliance teams, and TBA scores (if available)
3. After selecting, you'll see match details:
   - Blue and red alliance team numbers
   - TBA scores (ground truth from The Blue Alliance)
   - A video input area — **YouTube URL** (auto-filled from TBA if available) or **file upload**
4. Confirm the YouTube URL or upload a local video file
5. Click **Run Analysis**

### Step 3: Calibrate Scoreboard (Per-Match)

After clicking Run Analysis, the app grabs a frame from ~60 seconds into the match video and shows the **Scoreboard Calibration** screen. This step happens every match because scoreboard layouts can vary.

You need to mark four things on the frame:

1. **Click RED Hub** — Click directly on the red alliance's hub on the field image
2. **Click BLUE Hub** — Click directly on the blue alliance's hub
3. **Draw RED Score** — Click and drag a rectangle around the red alliance's score number on the scoreboard
4. **Draw BLUE Score** — Click and drag a rectangle around the blue alliance's score number

Each button highlights when active. Drawn regions appear as colored rectangles with labels. Hub clicks appear as crosshairs.

When all four are marked:
- Click **Save & Continue** to save the regions and proceed to inference
- Or click **Skip (use defaults)** to use previously saved regions (if any exist for this event)

> **Tips for accurate calibration:**
> - The frame shows only the top 2/3 of the video (the overhead field view)
> - Draw score regions tightly around just the numbers, not team names or labels
> - Hub positions should be the center of the hub structure on the field
> - If the frame is from before the match starts (no scores visible), the scoreboard numbers should still be visible as "0"

### Step 4: Run Inference

After calibration, you'll see the **Inference View** with the video canvas and live results panels.

**Controls:**

| Button | Action |
|--------|--------|
| **Start Inference** (green) | Begin processing the video |
| **Pause** (orange) | Freeze processing at the current frame — status bar shows "Paused at frame N" |
| **Resume** (green) | Continue processing from where you paused |
| **Stop** (red) | End processing early and show results collected so far |

During inference you'll see:
- **Live annotated video** — bounding boxes around robots (blue/red), fuel balls (yellow), and hub openings (green)
- **HUD overlay** — frame count, blue/red scores, OCR readings
- **Alliance cards** — per-robot goal counts and alliance totals
- **Scoring timeline** — each detected goal event with timestamp, robot ID, and confidence

The progress bar shows how far through the video you are.

### Step 5: Review Results

When inference completes (or you click Stop), the **Results** section appears below the video.

- **Robot-to-Team Assignment:** Each detected robot is listed with a dropdown to assign it to a team number. Team numbers are pre-filled from the TBA schedule. Verify or correct the assignments.
- **Predicted Fuel Counts:** The model's per-robot goal predictions (read-only — never modified).
- **Corrected Fuel Counts:** Editable fields if you need to override the model's predictions based on what you observed.
- **TBA Ground Truth:** The official alliance-level scores from TBA for comparison.
- **Notes:** Free-text field for observations about the match.

Click **Save Results** to persist everything to the local SQLite database.

### Step 6: View Match History

Click the **Match History** tab at the top to browse all analyzed matches:

- Each match shows TBA scores vs predicted scores
- Status badges: **Pending** (not yet analyzed), **Predicted** (model results saved), or **Corrected** (human-reviewed)
- Click **View** on any match to see/edit team assignments, corrected counts, and notes
- Use the **Back to Match Setup** button to analyze another match

### Recalibrating

If you need to redo the scoreboard calibration for the current event without selecting a new match:
- On the Match Setup screen, click the yellow **Recalibrate Scoreboard** button in the event banner

---

## Configuration

### Port

Default: **8004**. Change in the last line of `web/app.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8004)
```

### Detection Parameters

Key tuning parameters in `web/app.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_SKIP` | 2 | Process every Nth frame (higher = faster, less accurate) |
| `OVERHEAD_CROP_RATIO` | 0.67 | Fraction of frame height used (top = overhead view) |
| `LOST_TRACK_BUFFER` | 150 | Frames before a lost robot track is dropped |
| `ROBOT_MERGE_THRESHOLD` | 120 | Pixel distance to merge duplicate robot track IDs |
| `MAX_ROBOTS_PER_ALLIANCE` | 3 | Maximum robots tracked per alliance |
| `OCR_FRAME_INTERVAL` | 5 | Run scoreboard OCR every N processed frames |
| `MIN_SPEED` | 3.0 | Minimum ball speed (px/frame) to count as a trajectory |
| `MIN_SCORING_DISPLACEMENT` | 8.0 | Minimum ball movement (px) to count as a trajectory |
| `SCORE_LOCATION_COOLDOWN` | 10 | Frames before same location can score again |
| `MAX_SCORES_PER_FRAME_PER_ALLIANCE` | 2 | Cap on goals detected in a single frame per alliance |

### OCR Regions

Scoreboard OCR regions are configured interactively per-match via the calibration UI. Regions are saved to the SQLite database per event and can be recalibrated at any time.

Default fallback regions (if no calibration is done) are defined in `OCR_REGIONS_1080P` in `web/app.py`, scaled proportionally to the actual video resolution.

---

## Project Structure

```
frc-vision-scorer/
├── web/                         # Web application
│   ├── app.py                   # FastAPI server, inference pipeline, scoring logic
│   ├── database.py              # SQLite schema and async CRUD operations
│   ├── tba.py                   # The Blue Alliance API client
│   ├── start.sh                 # Start server in background
│   ├── stop.sh                  # Stop background server
│   ├── static/
│   │   ├── index.html           # Frontend HTML
│   │   ├── app.js               # Frontend logic (WebSocket, canvas, controls)
│   │   └── style.css            # Dark theme styles
│   ├── templates/               # (unused, static frontend)
│   └── uploads/                 # Temporary video uploads (gitignored)
│
├── runs/
│   └── train/
│       └── hub_zone_yolo26s/
│           └── weights/
│               └── best.pt      # Trained YOLO model (19 MB, via Git LFS)
│
├── scripts/                     # Training, data prep, and utility scripts
│   ├── train_hub_zone_model.py  # YOLO training script
│   ├── scoreboard_ocr.py        # Standalone OCR testing
│   ├── extract_frames.py        # Frame extraction from videos
│   └── ...                      # Various data prep scripts
│
├── data/                        # Training data (not in repo, gitignored)
├── docs/                        # Design docs and research notes
├── notebooks/                   # Jupyter notebooks (exploration)
├── src/                         # Legacy inference modules
│
├── .env                         # API keys (not in repo)
├── .gitattributes               # Git LFS tracking for model weights
├── .gitignore
├── .python-version              # Python version pin (3.12)
├── pyproject.toml
├── frc_scorer.db                # SQLite database (created on first run)
└── README.md
```

---

## Troubleshooting

### "Model not found" error

```bash
ls -lh runs/train/hub_zone_yolo26s/weights/best.pt
# Should be ~19 MB. If tiny (< 1 KB), run:
git lfs pull
```

### Tesseract not found

```bash
which tesseract
tesseract --version

# If installed via Homebrew but not found:
export PATH="/opt/homebrew/bin:$PATH"

# Or set explicitly in Python (add to top of web/app.py):
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
```

### Port 8004 already in use

```bash
lsof -i :8004
kill $(lsof -t -i :8004)
```

### YouTube video won't download

```bash
# Update yt-dlp (YouTube frequently changes their API)
pip install --upgrade yt-dlp

# Some videos are region-restricted or private — try a different match
```

### Scoreboard calibration frame won't load

- The app extracts a frame at 60 seconds into the video via ffmpeg
- If the video is shorter than 60 seconds, this may fail — try a longer video
- Check that ffmpeg is installed: `ffmpeg -version`
- Check server logs: `tail -50 /tmp/frc-server.log`

### Slow processing on CPU

- Increase `FRAME_SKIP` to 4 or 6 in `web/app.py`
- Use shorter video clips for testing
- Consider running on a machine with a GPU

### WebSocket connection drops

- Check browser console (F12) for errors
- Try `http://127.0.0.1:8004` instead of `localhost`
- Ensure no proxy or firewall is blocking WebSocket connections

### OCR returns no scores

- Verify calibration regions are drawn around the actual score numbers (not team names)
- Check server logs for "OCR authoritative" lines: `grep "OCR" /tmp/frc-server.log`
- Some scoreboard fonts/layouts may not be readable by Tesseract — try adjusting region placement

### Database errors

The SQLite database (`frc_scorer.db`) is created automatically on first run. If corrupted:

```bash
# Back up and recreate
mv frc_scorer.db frc_scorer.db.bak
python web/app.py
# The schema is recreated on startup
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | Deep learning framework (YOLO backend) |
| `ultralytics` | YOLO model loading and inference |
| `opencv-python-headless` | Video I/O and image processing |
| `fastapi` | Web framework (REST API + WebSocket) |
| `uvicorn` | ASGI server |
| `python-multipart` | File upload handling |
| `pytesseract` | Tesseract OCR Python wrapper |
| `yt-dlp` | YouTube video downloading |
| `supervision` | ByteTrack object tracking |
| `numpy` | Numerical operations |
| `Pillow` | Image handling |
| `aiosqlite` | Async SQLite database |
| `httpx` | Async HTTP client (TBA API) |
| `python-dotenv` | Load `.env` config |

**System dependencies:** Python 3.12+, Tesseract OCR 4+, ffmpeg, Git LFS

**API keys:** TBA Read API Key (free — [thebluealliance.com/account](https://www.thebluealliance.com/account))

---

## YOLO Model Details

The production model (`runs/train/hub_zone_yolo26s/weights/best.pt`) uses Ultralytics YOLO26s architecture — a small variant with ProgLoss/STAL features designed for improved small object detection.

- **Architecture:** YOLO26s (260 layers, 9.95M parameters, 22.5 GFLOPs)
- **Input size:** 640x640
- **Classes:** `fuel_ball`, `robot`, `hub_opening`
- **Device support:** Apple Silicon (MPS), NVIDIA (CUDA), and CPU

### Current Production Model (updated 2026-03-07)

Fine-tuned from the prior hub-zone model using newly labeled Istanbul overhead frames.

- **Training set:** 103 labeled images total
  - 51 alliance-side hub-zone images (Week 0)
  - 52 Istanbul Finals overhead-only images (new labels)
- **Split:** 82 train / 21 val
- **Validation (combined val set):**
  - **mAP50:** 0.533
  - **mAP50-95:** 0.245
  - **Precision:** 0.848
  - **Recall:** 0.424

For reference, the previous production model on the same combined validation set scored:

- mAP50: 0.274
- mAP50-95: 0.089

So the current model substantially improves detection quality on mixed Week 0 + Istanbul camera angles.

> Note: Label Studio YOLO exports class order as `fuel_ball, hub_opening, robot`. For this project, labels must be remapped to `fuel_ball=0, robot=1, hub_opening=2` before training.

---

## License

Internal use — Team 2102 scouting tool.

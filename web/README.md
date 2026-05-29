# FRC Vision Scorer - Web Application

Real-time YOLO inference testing web app for FRC match videos with per-robot score attribution.

## Features

- **FastAPI Backend**: RESTful API + WebSocket for real-time frame streaming
- **YOLO v4 Model**: Pre-trained model (mAP50: 0.943) detecting fuel balls, robots (red/blue), and hub
- **ByteTrack Integration**: Multi-object tracking across frames
- **Per-Robot Score Attribution**: Scores attributed to nearest robot when ball enters hub zone
- **Real-Time Streaming**: WebSocket streaming of annotated frames at 10-15 FPS
- **Clean Dark UI**: Modern interface with live scoring dashboard

## Architecture

### Backend (`app.py`)
- **POST /api/infer**: Create inference job (upload video or YouTube URL)
- **WebSocket /ws/infer/{job_id}**: Stream annotated frames + scoring data
- **GET /api/jobs/{job_id}/status**: Check job status
- **GET /api/models**: List available models (v2, v4)

### Frontend
- **index.html**: Main UI with video canvas and controls
- **style.css**: Dark-themed styling
- **app.js**: WebSocket client, UI updates, scoring display

## Installation

The web app uses the existing project virtual environment:

```bash
cd ~/development/projects/frc-vision-scorer
.venv/bin/python -m pip install fastapi uvicorn python-multipart websockets
```

(Already installed)

## Usage

### Start the Server

```bash
cd ~/development/projects/frc-vision-scorer/web
../venv/bin/python app.py
```

Or use the start script:

```bash
./start.sh
```

Server runs on `http://localhost:8004`

### Stop the Server

```bash
./stop.sh
```

Or manually:

```bash
kill $(cat /tmp/frc-vision-dev.pid)
```

## Testing

Test with one of the Week 0 videos:

1. Open `http://localhost:8004` in your browser
2. Select "Upload Video File"
3. Choose `~/development/projects/frc-vision-scorer/data/videos/2026/week0_f1m1.mp4`
4. Select model "YOLO v4"
5. Click "Start Inference"

The app will:
- Process video frames at ~10-15 FPS
- Display annotated frames with bounding boxes:
  - Yellow: Fuel balls
  - Red: Red alliance robots
  - Blue: Blue alliance robots
  - Green: Hub
- Show per-robot scores in real-time
- Display scoring events timeline

## Scoring Logic

Per-robot score attribution works as follows:

1. **Track all objects** with ByteTrack (maintains consistent IDs)
2. **Detect hub position** from YOLO detections (class 3)
3. **Monitor balls near hub**:
   - When a ball enters hub zone (< 100px from hub center)
   - Or disappears within scoring distance
4. **Attribute score**:
   - Find nearest robot (of appropriate alliance) within last 15 frames
   - Must be within 200px attribution distance
   - If no robot nearby, attribute by field position (left=red, right=blue)
5. **Update displays**:
   - Alliance scores (Red/Blue totals)
   - Per-robot scoring table
   - Scoring events timeline

## Configuration

Adjust scoring parameters in `app.py`:

```python
HUB_ZONE_RADIUS = 100  # pixels from hub center
BALL_DISAPPEAR_THRESHOLD = 5  # frames without detection
SCORING_ATTRIBUTION_DISTANCE = 200  # max distance for robot attribution  
SCORING_ATTRIBUTION_FRAMES = 15  # look back N frames for nearest robot
```

## Video Sources

The app supports:

- **File Upload**: MP4, AVI, MOV (up to 720p recommended for CPU inference)
- **YouTube URLs**: Automatically downloads with yt-dlp (max 720p)
- **Test Videos**: Week 0 matches at `data/videos/2026/week0_f1m{1,2,3}.mp4`

## Performance Notes

- **CPU-only inference** (Mac Mini M4, no GPU)
- **Frame sampling**: Processes every 2nd frame for real-time display (~15 FPS)
- **JPEG encoding**: Frames compressed to 85% quality for WebSocket transmission
- **Typical throughput**: 5-10 FPS on M4 CPU (varies by video resolution)

## Models

Two trained models available:

- **v2** (`runs/train/frc_v2/weights/best.pt`): Earlier version
- **v4** (`runs/train/frc_v4/weights/best.pt`): Current best (mAP50: 0.943, 3375 images)

Classes:
- 0: fuel_ball
- 1: robot_red
- 2: robot_blue
- 3: hub

## Development

### File Structure

```
web/
├── app.py              # FastAPI backend
├── static/
│   ├── index.html      # Main UI
│   ├── style.css       # Styling
│   └── app.js          # Frontend logic
├── uploads/            # Temporary video uploads
├── start.sh            # Start script
├── stop.sh             # Stop script
└── README.md           # This file
```

### Logs

Application logs: `/tmp/frc-vision-dev.log`

Process PID: `/tmp/frc-vision-dev.pid`

## Future Enhancements

- [ ] Scoreboard OCR integration
- [ ] Frame seeking/scrubbing
- [ ] Download annotated video
- [ ] Multi-camera support
- [ ] Export scoring data (JSON/CSV)
- [ ] GPU acceleration (when available)
- [ ] Model comparison mode

## License

MIT (consistent with parent project)

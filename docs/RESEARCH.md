# FRC Vision Scorer - Research Document

## Project Goal
Build a vision model that watches live-streamed FRC competition video and predicts how many yellow fuel balls each individual robot successfully scores into the hub per match.

## Game Context

### 2026 REBUILT
- **Scoring element**: FUEL - 5.91" (15cm) diameter, high-density yellow foam ball
- **Scoring target**: HUB - large structure at center of field with openings at top
- **Match structure**: 2:40 total (20s auto, 10s transition, 4x25s shifts, 30s endgame)
- **Hub activation**: Hubs alternate active/inactive during shifts. Scoring into inactive hub = 0 points
- **Ranking points**: Energized (100 fuel, 1 RP), Supercharged (360 fuel, 1 RP)
- **Volume**: Robots shoot 1-6+ balls/second in bursts. 100-360+ balls per alliance per match
- **Field**: 3v3 alliances, 6 robots total
- **Robot ID**: Team numbers on bumpers, alliance colors (red/blue)

### 2017 Steamworks (Training Data Source)
- **Very similar game**: Robots shoot fuel (5" wiffle balls, yellow) into a Boiler
- **Boiler**: Has high goal and low goal. High = 1 kPa, Low = 1/3 kPa per ball
- **Match videos**: Extensive archive on YouTube and The Blue Alliance
- **Scoring data**: Available via The Blue Alliance API (alliance-level, not per-robot)

### Key Differences 2017 vs 2026
| Feature | 2017 Steamworks | 2026 REBUILT |
|---------|----------------|--------------|
| Ball size | 5" wiffle ball | 5.91" foam ball |
| Scoring target | Boiler (side of field) | Hub (center of field) |
| Ball color | Yellow/green | Yellow |
| Active/Inactive | No | Yes (hub shifts) |
| Volume | Moderate | Very high (360+ for RP) |

Despite differences, the core CV challenge is identical: detect yellow balls, track them, determine if they enter the scoring structure, attribute to specific robots.

## Architecture Plan

### Pipeline Overview
```
Live Video Stream
    │
    ▼
[Frame Extraction] (ffmpeg / OpenCV)
    │
    ▼
[Object Detection] (YOLOv8/v11)
    ├── Detect: balls, robots, hub/boiler openings
    ├── Detect: bumper numbers/colors (team ID)
    │
    ▼
[Object Tracking] (ByteTrack / BoT-SORT)
    ├── Track balls across frames (trajectory)
    ├── Track robots across frames (identity persistence)
    │
    ▼
[Scoring Event Detection]
    ├── Ball enters hub region → "score event"
    ├── Backtrack trajectory → attribute to launching robot
    │
    ▼
[Aggregation & Output]
    └── Per-robot scoring tallies, timestamped events
```

### Component Details

#### 1. Object Detection (YOLO)
- **Model**: YOLOv8x or YOLOv11 (Ultralytics)
- **Classes**: `fuel_ball`, `robot`, `hub_opening`, `bumper_number`
- **Training data**: Annotated frames from 2017 Steamworks matches
- **Fine-tuning**: 2026 REBUILT footage when available
- **Input resolution**: 1280x720 or 1920x1080

#### 2. Object Tracking
- **ByteTrack** or **BoT-SORT** for multi-object tracking
- Track ball IDs across frames to build trajectories
- Track robot IDs across frames for persistent identification
- Handle occlusion (balls behind robots, multiple balls in flight)

#### 3. Scoring Event Detection
Two approaches to evaluate:
1. **Region-based**: Define hub/boiler as a polygon zone. Ball trajectory enters zone → score
2. **Trajectory-based**: Ball appears, follows arc, disappears into hub region → score
3. **Delta counting**: Count balls visible before/after scoring region per time window

#### 4. Robot Attribution
- **Bumper OCR**: Read team numbers from bumper text (challenging at broadcast resolution)
- **Color + position tracking**: Red/blue alliance identification + persistent robot tracking
- **Proximity + trajectory**: When a ball begins moving, the nearest robot is the shooter
- **Launcher detection**: Detect the shooting action (ball leaves robot frame)

#### 5. Alternative: Video Understanding Model
- **Gemini 2.5 Pro** or **GPT-4V** for video understanding
- Feed 30-60s clips, ask structured questions: "How many yellow balls did each robot score?"
- Pros: No training needed, handles context well
- Cons: Latency, cost, may not be accurate enough for rapid ball counting
- Worth evaluating as a baseline or verification layer

## Data Strategy

### Training Data Sources

#### 2017 Steamworks Match Videos
- YouTube playlists (FIRST official + regional streams)
- The Blue Alliance: Match videos + scoring breakdowns
- **Key playlist**: https://www.youtube.com/playlist?list=PLFjf3FNm6d6UzfHUqOEYf7_JOPzdnKgqM

#### wg-provided reference videos
- Game animation: https://www.youtube.com/watch?v=_fybREErgyM
- 2017 robot reference: https://www.youtube.com/watch?v=59zGltwCBnU

#### Existing Models
- Little-Apple-Cyborgs-5968/2026-limelight-ball-detection-models (on-robot ball detection, Limelight-optimized)
- Football/tennis analysis YOLO projects (architecture reference)

### Annotation Plan
1. Extract frames from 2017 match videos (1-5 fps sampling)
2. Annotate using Roboflow or CVAT:
   - Bounding boxes: balls, robots, boiler/hub openings
   - Labels: alliance color, team number (where readable)
3. Scoring event labels: timestamp when ball enters boiler
4. Target: 5,000-10,000 annotated frames for initial model

### Ground Truth
- The Blue Alliance API provides alliance-level scoring per match
- Per-robot scoring requires manual annotation or cross-referencing with scouting data
- Statbotics.io may have OPR (Offensive Power Rating) data for estimating per-robot contribution

## Infrastructure

### Training
- **Primary**: Local DL machine with 2x 24GB GPUs (to be configured)
- **Backup**: Modal.com (cloud GPU, serverless)
- **Development**: Mac Mini M4 (data prep, small experiments)

### Inference (Production)
- **Target**: Cloud-based (Modal.com or similar)
- **Requirements**: Real-time or near-real-time processing of live video
- **Input**: YouTube/Twitch livestream URL
- **Output**: JSON with per-robot scoring events + tallies

## Technology Stack

### Core
- Python 3.11+ (for ML ecosystem compatibility)
- PyTorch 2.x
- Ultralytics (YOLOv8/v11)
- OpenCV for video processing
- Supervision (Roboflow) for visualization and tracking

### Tracking
- ByteTrack or BoT-SORT (integrated with Ultralytics)

### Data
- The Blue Alliance API (match data, video links)
- Roboflow or CVAT (annotation)
- yt-dlp (video download)

### Deployment
- Modal.com (serverless GPU inference)
- FastAPI (API layer)
- WebSocket for real-time score updates

## Research Questions

1. **Can YOLO reliably detect small yellow balls at broadcast resolution?** (5" balls on a 54'x27' field, 720p/1080p camera)
2. **How accurate is trajectory-based scoring vs. region-based counting?**
3. **Can we read bumper numbers from broadcast cameras?** (likely challenging)
4. **Is a multimodal LLM (Gemini/GPT-4V) viable for ball counting at 1-6 balls/second?**
5. **What's the minimum FPS needed for reliable ball tracking?** (balls travel fast)
6. **Can we use The Blue Alliance scoring data as weak supervision?** (alliance totals as constraints)

## Phase Plan

### Phase 1: Data Pipeline (Week 1)
- [ ] Download 2017 Steamworks match videos
- [ ] Set up The Blue Alliance API access
- [ ] Build frame extraction pipeline
- [ ] Begin annotation (Roboflow)
- [ ] Evaluate existing 2026 ball detection model

### Phase 2: Detection Model (Week 2-3)
- [ ] Train YOLOv8 on annotated 2017 frames
- [ ] Evaluate detection accuracy for balls, robots
- [ ] Experiment with tracking (ByteTrack)
- [ ] Build scoring zone detection

### Phase 3: Scoring Pipeline (Week 3-4)
- [ ] Implement scoring event detection
- [ ] Build robot attribution logic
- [ ] Validate against known match scores (TBA data)
- [ ] Evaluate multimodal LLM baseline

### Phase 4: 2026 Fine-tuning (When data available)
- [ ] Collect 2026 match footage
- [ ] Fine-tune detection model
- [ ] Adjust for hub geometry (center vs side)
- [ ] Validate on live matches

### Phase 5: Production (Before target competitions)
- [ ] Deploy inference pipeline on Modal.com
- [ ] Build livestream ingestion
- [ ] Real-time scoring dashboard
- [ ] API for external consumption

---
Last updated: 2026-02-09

# FRC Vision Scorer

Automated vision scouting for FRC (FIRST Robotics Competition) from ordinary
broadcast match video. Point it at an event's match videos from The Blue
Alliance and it tracks every robot in real field coordinates, identifies each
one by reading bumper numbers with a vision-language model, and renders a
reviewable match replay with a live top-down field map. The whole pipeline
is designed to process a match in under ~15 minutes on a single 24 GB GPU.

Built for the 2026 REBUILT season by Team 2102 (Team Paradox). **Very much a
work in progress** — robot detection, tracking, and identity work today;
fuel attribution doesn't exist yet. See [Limitations](#limitations).

## Table of Contents

- [Why](#why)
- [Goals](#goals)
- [How It Works](#how-it-works)
  - [Calibration](#calibration)
  - [Robot Detection](#robot-detection)
  - [Robot Tracking](#robot-tracking)
  - [Field Coordinates and the 2D Map](#field-coordinates-and-the-2d-map)
  - [Robot Identity](#robot-identity)
  - [Fitting the Time Budget](#fitting-the-time-budget)
- [Limitations](#limitations)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Running](#running)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Models](#models)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Why

Good scouting data takes an army of well-trained, motivated scouters, and
most teams don't have one. This project asks: how much of that can a vision
pipeline do on its own, using nothing but the public event stream?

Robot identity came first, before any scoring analysis, deliberately —
tracking game pieces does you no good if you can't attribute them to the
right robot.

## Goals

- **No human scouters** identifying robot images
- **No special equipment at events** — works from the broadcast stream's
  YouTube video, no cameras in the stands
- **Universal to any event** — no per-event training, datasets, or robot
  photo sessions
- **Adaptable to future seasons** — detection/tracking/identity carry over
  even as the game changes
- **Not realtime, but bounded** — under ~15 minutes per match, so it can
  catch up during field resets and lunch and be ready for match strategy and
  the picklist meeting
- Develops on consumer hardware (one RTX 3090); can rent a beefier GPU
  (e.g. an H100) for event weekends

## How It Works

### Calibration

Cameras at an event are consistent, but their viewing angles are not — and
they get bumped or re-zoomed mid-event. Instead of requiring our own
cameras, each broadcast camera is calibrated against the field itself: you
click AprilTags in the frame and their locations on a 2D field map
(`/calibrate`), and a single-view pinhole resection solves the camera's
position, rotation, and lens distortion from CAD-derived field geometry.
(Most tags are too small in the broadcast to decode automatically, hence the
click-to-map UI.)

The composite broadcast feed carries three views — a main overhead camera
and two low corner cameras — and all three are calibrated the same way, so
the pipeline knows it's looking at one field from three viewpoints. Because
cameras move mid-event, every job runs an automatic drift check first: if a
camera has moved more than ~half a meter's worth, the job refuses to run
rather than produce garbage positions.

Per match there's also a quick scoreboard setup: draw boxes around the red
score, blue score, and clock (kept manual so it adapts when the stream
overlay changes), and segment out auto/teleop from the clock so frames
outside the match aren't processed at all.

### Robot Detection

Detection is three YOLO models working together:

- a **bumper detector** trained specifically on red and blue bumpers, with
  hard-negative training for the things that kept getting detected as robots
  — scoring tablets, FMS radios on poles, and similar field-side equipment;
- a **field model** that detects fuel, robots in general, and the hub
  openings;
- a **person model**, because refs and scorers walk in front of the field
  view constantly.

The bumper model contributes alliance color; if the alliance can't be
determined yet, the robot is still tracked until it can be. (An earlier
HSV color approach was rejected — it hated dark blue bumpers and kept
reading painted robot parts as bumper color.)

On top of the models sits a stack of runtime sanity checks, added one
painful false-positive at a time: detections overlapping a person are
rejected, boxes implausibly wide for their distance from the camera are
rejected, and masks that are tall and top-heavy (the silhouette of a radio
on a stand) are rejected — carefully, so a climbing robot isn't.

### Robot Tracking

YOLO boxes are used as prompts to a segmentation tracker (SAM) that masks
out the entire robot. Segmentation handles overlapping and partially
occluded robots better than per-frame boxes, and crucially it's *temporally
aware*: if a robot disappears behind another for a few seconds, the tracker
carries the same identity through the occlusion instead of "discovering" a
brand-new robot on the other side. YOLO re-prompts the tracker every ~2
seconds so anything missed initially gets picked up quickly.

The default backend is **SAM 3.1**. SAM 2.1 (still available via
`TRACKING_BACKEND=sam2`) wants to hold the whole video's state in VRAM,
which doesn't fit a full match on 24 GB; the SAM3 backend processes the
match in chunks of a few hundred frames and re-seeds every live robot
across each chunk boundary so identities survive. It confirms more robots
at roughly half the wall-clock time.

After the match finishes, a **backward-confirmation pass** re-resolves
identity over the whole timeline with hindsight — a robot that wasn't
positively identified until 30 seconds in still gets correctly attributed
for the time before that.

### Field Coordinates and the 2D Map

Image-space tracks are projected through the camera calibration into field
coordinates and fused across the three views with a Kalman filter. The
top-down map is drawn from the same CAD-derived field geometry as the
calibration overlay — one source of truth, so the map and the camera
overlays can never disagree about where the hubs, bumps, towers, and
trenches are.

Making that map trustworthy took explicit distrust:

- **Corner cameras are for identity, not location.** They sit low and have
  terrible depth resolution, so corner-only measurements are heavily
  downweighted — they nudge a position rather than yank it.
- **Occluded robots ease to a stop** instead of sailing off the field: a
  coasting track's velocity decays each missed frame.
- **Physically impossible jumps are distrusted** rather than snapped to.
- **Exclusion zones** drawn on the field map (FMS radio stands, the hubs)
  prevent the tracker from ever *seeding* a "robot" on a static structure,
  and the known field bounds reject robots in the queue line and oddities
  in the audience.

On the replay HUD's radar panel, each robot carries its team number, a
fading trail, and styling that says how much to trust it: a solid dot is a
confident main-camera fix, a hollow dot is coasting on the motion model,
and a dashed ring means corner-cameras-only.

### Robot Identity

Bumper crops are periodically read by a small VLM (Qwen3-VL-4B, served
locally — in-process or behind a vLLM server for better batching). The
reader knows the six teams in the match and the robot's alliance color, and
its reads feed a voting system. Two robots can never be positively
identified as the same team at once; an assignment step resolves conflicts
toward the read it's more confident about.

The locking logic is deliberately reversible: a team **locks after 3
consecutive agreeing reads, but 2 consecutive contradicting reads flip it**.
That sounds backwards until you've watched a confidently *wrong* lock freeze
for a whole match — an early misread is far worse than an unconfident one.
Even confirmed robots are periodically re-audited with fresh reads.

Two supporting systems:

- **Appearance gallery.** Every positively-identified robot deposits a
  DINOv2 visual embedding into a per-event gallery, so later in the event a
  robot whose bumper is obscured can still get a vote from what it *looks
  like*. Appearance is a supplementary vote, never an override. Match-time
  voting uses only human-verified exemplars, and human corrections outvote
  automatic deposits — a few minutes of spot-checking per event improves
  every later match.
- **Not wasting reads.** VLM reads are among the most expensive steps, so
  reads are skipped when a bumper is occluded in every camera and when a
  robot hasn't moved since its last read — substantially fewer reads with
  no identity loss.

### Fitting the Time Budget

The tracker and the bumper reader both want the entire GPU, and a resident
vLLM server (~11 GB) on top of SAM3's propagation peak (~11 GB) OOMs a
24 GB card. The fix is that they never actually need the GPU at the same
time: the reader is **phase-exclusive** — shut down during tracking, spun
up only for the bumper-reading phase, torn down after the job (with an
in-process fallback if the server won't start).

On the tracking side, mask export resolution turns out not to be a speed
lever at all — SAM resizes every frame to a fixed internal square, so cost
scales with frames-per-view, making per-view frame rate
(`SAM3_TRACK_FPS`) the real knob. Combined with the read-skipping above, a
match lands under the 15-minute budget on a 3090.

## Limitations

Identity is where most of the iteration has gone, and it's still the
fragile part:

- **Roughly a fifth of bumper fonts are simply unreadable** at broadcast
  resolution — too skinny, kerning too tight. Bumpers sit low, so robots
  and game pieces obscure them constantly. Higher camera angles give better
  positions but make bumpers nearly impossible to read.
- **Visual embeddings can't tell teammates apart.** DINOv2 reliably
  separates red robots from blue robots and is nearly useless at separating
  two robots on the *same* alliance — from a wide camera, boxes with red
  bumpers all look alike. The bumper number is the only signal that truly
  discriminates individuals; everything else can only support it.
- **Results are noisy run-to-run.** A clean run on the main test match
  confirms 5 of 6 team numbers; a rougher pass might lock 3. One team's
  bumpers never resolve no matter how many reads are thrown at them.
- **Fuel attribution doesn't exist yet.** Fuel *detection* is in (the field
  model sees fuel every frame), but per-robot scoring attribution is future
  work.
- The broadcast's event-name banner eats a chunk of the corner cameras'
  view, and corner calibration is manual and fiddly
  (see `docs/corner-camera-calibration.md`).

## Hardware Requirements

The pipeline is **designed for a single NVIDIA GPU with 24 GB of VRAM**
(developed and tuned on an RTX 3090). That number is a real design
constraint, not a suggestion: SAM3's tracking propagation peaks around
~11 GB and the vLLM bumper reader reserves another ~11 GB, and the
pipeline only fits because the two run in separate phases on the same card
(see [Fitting the Time Budget](#fitting-the-time-budget)). Cards with less
VRAM will OOM mid-match; more VRAM (or a rented H100 at an event) just
makes it faster.

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA, 24 GB VRAM minimum (RTX 3090 / 4090 class) |
| RAM | 32 GB recommended |
| Storage | ~20 GB (environments, model weights, downloaded match video) |
| OS | Linux (developed on Ubuntu-based Pop!_OS) |

## Installation

### 1. System prerequisites

```bash
sudo apt install python3.12 python3.12-venv ffmpeg tesseract-ocr
# Git LFS is required for the model weights
sudo apt install git-lfs && git lfs install
```

### 2. Clone

Run `git lfs install` **before** cloning (step 1) — the trained model
weights are stored in Git LFS, and without it you get tiny pointer files
instead of models.

```bash
git clone https://github.com/Paradox2102/frc-vision-scorer.git
cd frc-vision-scorer

# Verify LFS pulled real weights (~50 MB each, not 1 KB pointers)
ls -lh runs/train/*/weights/best.pt

# If they're tiny, LFS wasn't set up at clone time — fix with:
git lfs install && git lfs pull
```

### 3. Main environment (`.venv`)

The app's ML stack is installed directly into the venv (the `pyproject.toml`
intentionally under-declares it):

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
# PyTorch with CUDA (cu126 used in development)
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Core app + SAM2 (pulled from GitHub via pyproject)
.venv/bin/pip install -e .
# The rest of the stack
.venv/bin/pip install ultralytics supervision fastapi "uvicorn[standard]" \
    aiosqlite opencv-python pillow pytesseract yt-dlp transformers \
    accelerate qwen-vl-utils python-dotenv websockets
```

### 4. Tracking backend

**SAM 3.1 (default).** The checkpoint repo is gated — request access to
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) on Hugging Face,
put a read token in `.env` (`HF_TOKEN`), then:

```bash
.venv/bin/pip install -e '.[sam3]'
```

See `docs/SAM3.md` for details and quirks.

**SAM 2.1 (alternative, `TRACKING_BACKEND=sam2`).** Download a checkpoint:

```bash
mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..
```

(Other sizes — tiny/small/base_plus — are selectable via `SAM2_MODEL_SIZE`.)

### 5. vLLM environment (`.venv-vllm`)

The team-number reader runs Qwen3-VL-4B as a separate vLLM server. vLLM pins
its own torch build and **must not** share a venv with the app:

```bash
python3.12 -m venv .venv-vllm
.venv-vllm/bin/pip install vllm
```

### 6. API keys

```bash
cp .env.example .env   # then fill in your keys (TBA_API_KEY is required)
```

## Running

Always start via the scripts (they run from the project root — model and
checkpoint paths resolve relative to it):

```bash
# Start (loads your API keys, then launches the app in the background)
set -a; source .env; set +a
web/start_all.sh

# It's up when this responds (UI at http://localhost:8004)
curl http://localhost:8004/api/status

# Stop the app and any running vLLM reader
web/stop_all.sh
```

You do **not** start the vLLM bumper reader yourself: it's job-managed and
phase-exclusive. Each match job spins it up only for the bumper-reading
phase (first start takes ~90 s to load the model) and tears it down when
the job finishes, so idle time between matches costs zero VRAM.
`web/start_vllm.sh` / `web/stop_vllm.sh` exist for manual debugging only.

Where to look when something's wrong:

| What | Where |
|------|-------|
| App logs | `/tmp/frc-vision-dev.log` |
| vLLM reader logs | `/tmp/frc-vllm.log` |
| App PID | `/tmp/frc-vision-dev.pid` |
| Database | `frc_scorer.db` in the project root (SQLite) |

## Usage

1. **Load an event** — enter a TBA event key; schedules, rosters, and match
   video links come from The Blue Alliance.
2. **Calibrate the cameras** (once per event, redo if a camera moves) —
   open `/calibrate`, click AprilTags in each camera view and their
   positions on the field map, and solve. Draw exclusion zones over static
   field structures while you're there. Corner cameras:
   `docs/corner-camera-calibration.md`.
3. **Set up the match** — pick a match; if it's the first one for this
   stream layout, draw the score and clock boxes. The clock segments
   auto/teleop so dead video isn't processed.
4. **Run the job** — confirm the team roster and submit. Pipeline knobs can
   be overridden per job.
5. **Review** — watch the replay HUD (camera views + radar map), check
   per-robot identity confidence, and verify or correct bumper reads in the
   gallery UI. Verified exemplars improve every later match at the event.

## Configuration

Almost every pipeline knob is an environment variable with a per-job
override (`config_json`). The authoritative list with defaults and
documentation lives in `web/job_config.py`. Frequently used:

| Variable | Purpose |
|----------|---------|
| `TRACKING_BACKEND` | `sam3` (default) or `sam2` |
| `SAM3_TRACK_FPS` | tracking speed-vs-quality tradeoff (legacy `SAM2_*` names of renamed knobs/env vars are still accepted) |
| `SAM2_MODEL_SIZE` | SAM2-backend encoder size (`TRACKING_BACKEND=sam2` only) |
| `TEAM_READER` | `vllm` (what `start_all.sh` sets), `local` (in-process), or `clef` (Cloudflare Workers AI; needs `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN`, model via `CLEF_MODEL`) |
| `VLM_MODEL_NAME` | reader model (default `Qwen/Qwen3-VL-4B-Instruct`) |
| `GALLERY_ENABLED` | appearance-gallery identity assists |

## Project Structure

```
web/                  FastAPI app, job runner, UI
  app.py              API + pipeline orchestration
  field/              the vision pipeline
    calibration.py    tag-anchored pinhole resection
    mask_tracker.py   shared mask-tracker logic; builds SAM 2.1 for TRACKING_BACKEND=sam2
    sam3_tracker.py   SAM3 tracking backend (default)
    tracker.py        field-space Kalman tracking
    fusion.py         cross-view fusion
    identity.py       team-number voting / locking
    appearance_gallery.py  DINOv2 exemplar gallery
    radar.py          2D field-map rendering
  static/             frontend
scripts/              training, dataset prep, evaluation, benchmarks
deploy/runpod/        optional cloud-GPU (RunPod / Vast.ai) job runner
runs/train/           shipped model weights (Git LFS)
data/                 field geometry, calibration data, datasets
  field/              field CAD + AprilTag layouts (2019–2026)
  bumper_hardneg_equipment/  committed bumper training imageset
  eval/               ground truth + reader benchmarks used by the tests
docs/                 design notes
tests/                pytest suite (run: .venv/bin/python -m pytest)
```

## Models

Committed weights (Git LFS):

- **Bumper detector** — `runs/train/frc_bumpers_equipment_hardneg/weights/best.pt`
  (preferred; includes equipment hard negatives) with
  `runs/train/frc_bumpers/weights/best.pt` as fallback. Override with
  `FRC_BUMPER_MODEL`. Training data is committed under `data/` and
  reproducible via `scripts/download_bumper_datasets.py`; retrain with
  `scripts/train_bumper_model.py`.
- **Field model** — `runs/train/hub_zone_yolo26s/weights/best.pt` (fuel,
  robots, hub openings).

Downloaded at install/first use:

- **Team-number reader** — `Qwen/Qwen3-VL-4B-Instruct` (Hugging Face).
- **Trackers** — SAM 3.1 (gated HF repo) or SAM 2.1 checkpoints.
- **Appearance embeddings** — DINOv2 (auto-downloaded).

## Acknowledgements

Heavily inspired by
[Roboflow's basketball player detection/tracking/identification](https://www.youtube.com/watch?v=yGQb9KkvQ1Q)
and [Team 1148's Project Vector](https://www.chiefdelphi.com/t/1148-project-vector-ai-scouting-for-rebuilt/519559).

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — free to use, study, modify, and
share for any noncommercial purpose (education, research, personal projects,
FRC teams). Commercial use is not permitted.

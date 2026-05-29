# Re-ID Acceptance Test

## Purpose

Measure three things on a real match clip before and after tuning:

| # | Metric | Target |
|---|--------|--------|
| 1 | **Canonical-ID switches per physical robot** — how many times a tracked robot's assigned team_number changed | ≤ 1 flip per robot across a full match |
| 2 | **Frames with duplicate team numbers** — frames where two live tracks share the same non-null team_number | **0** (guaranteed by Hungarian assignment; any violation is a bug) |
| 3 | **Team-number accuracy at ground-truth frames** — fraction of labelled boxes whose predicted team_number is correct | ≥ 90 % |

Use this harness to compare the HSV-only baseline (no DINOv2) against the
DINOv2-based re-ID pipeline, and to validate parameter changes before
deploying to a live event.

---

## Where to place a clip

Place the video at:

```
data/videos/<event_key>/<match_key>.mp4
```

For example:

```
data/videos/2025txcha/qm42.mp4
```

No video is currently present in the repository.  Download a match from
YouTube (the scorer's `yt_dlp` integration or any direct download works).
A 150-second clip of an active match is sufficient to exercise all metrics.

---

## Ground-truth JSON schema

The `--ground-truth` file maps **sampled frame indices** to the robot boxes
visible in that frame.  You do not need to label every frame — 10–15
well-chosen frames covering different robot positions are enough.

### Schema

```json
{
  "<frame_index>": {
    "<x1>,<y1>,<x2>,<y2>": "<team_number>",
    "<x1>,<y1>,<x2>,<y2>": "<team_number>"
  },
  ...
}
```

- `<frame_index>` — integer, the **raw** video frame number (0-based,
  before any `--sample-rate` thinning).
- `<x1>,<y1>,<x2>,<y2>` — bounding box in the **overhead crop** coordinate
  space (pixels, floats OK).  The overhead crop is the top 67 % of the frame.
- `<team_number>` — team number as a string, e.g. `"1234"`.

### Small example

```json
{
  "300": {
    "120.0,80.0,220.0,180.0": "1234",
    "450.0,90.0,550.0,190.0": "5678",
    "800.0,100.0,900.0,200.0": "9012"
  },
  "600": {
    "130.0,75.0,230.0,175.0": "1234",
    "460.0,85.0,560.0,185.0": "5678"
  }
}
```

### How to build a ground-truth file

1. Open the clip in any video player that shows frame numbers (VLC: View →
   Status Bar shows the current frame, or use `ffprobe`).
2. Choose ~10 frames spread across the match (every ~15 seconds works well).
3. For each frame, take a screenshot, locate each robot in the **overhead
   crop** (top 67 % of the image), and note its bounding box corners.
   A simple way: open the screenshot in GIMP or Photoshop, hover the cursor
   over each robot corner and read the pixel coordinates.
4. Identify each robot's team number from its bumper text in the screenshot.
5. Write the JSON with the frame index, box, and team_number.

Box coordinates only need to be approximately correct — the harness uses
IoU ≥ 0.30 to match predictions to GT boxes, so a few-pixel error in the
annotation is fine.

---

## Commands

### Compile check (no model/video required)

```bash
cd /path/to/frc-vision-scorer
./.venv/bin/python -m py_compile scripts/reid_acceptance.py && echo "compile OK"
./.venv/bin/python scripts/reid_acceptance.py --help
```

### Basic run (no ground truth)

```bash
./.venv/bin/python scripts/reid_acceptance.py \
    --video data/videos/2025txcha/qm42.mp4 \
    --sample-rate 3
```

### Full run with ground truth

```bash
./.venv/bin/python scripts/reid_acceptance.py \
    --video data/videos/2025txcha/qm42.mp4 \
    --ground-truth data/videos/2025txcha/qm42_gt.json \
    --sample-rate 3
```

`--sample-rate 1` processes every frame (slowest, most accurate).
`--sample-rate 5` is faster for a quick sanity check.

---

## Results table

Fill in after running:

| Metric | Value |
|--------|-------|
| Total robots tracked | |
| Total canonical-ID flips (sum across all robots) | |
| Worst single-robot flip count | |
| Frames with duplicate team numbers | |
| Ground-truth accuracy (metric 3) | |
| Processed frames | |
| Sample rate used | |

### Constants in use at time of run

Record these from the script's printed header so results are reproducible:

| Constant | Value |
|----------|-------|
| `EMBED_MODEL_NAME` | |
| `APP_FLOOR` | |
| `ASSIGN_MAX_COST` | |
| `POS_GATE_PX` | |
| `IDENTITY_INTERVAL` | |

---

## Tuning

### Same-alliance robots are under-discriminated

If two robots on the same alliance are frequently swapped (high flip count,
metric 1 > 1 per robot), the DINOv2-small embeddings may not distinguish
robots that share a similar colour scheme or are far from the camera.

**Fix:** switch to a SigLIP checkpoint, which produces richer semantic
embeddings and is better at fine-grained visual similarity.  Only
`web/embedder.py` needs to change — update `EMBED_MODEL_NAME` (the constant
in `web/app.py`) and the default in `web/embedder.py`:

```python
# web/app.py
EMBED_MODEL_NAME = "google/siglip-base-patch16-224"   # was "facebook/dinov2-small"
```

No other caller changes are needed because `RobotEmbedder` is model-agnostic.
Re-run this harness after the change to confirm the flip count drops.

### Hungarian cost threshold

If metric 2 fires (duplicates), or if metric 3 accuracy is low despite
embeddings looking good, try tightening `ASSIGN_MAX_COST` (lower value =
stricter matching).  The default is `0.45` (≈ cosine similarity ≥ 0.55).

### Position gate

If robots are being merged across the field, increase `POS_GATE_PX` (default
200 px).  If continuity breaks when robots stop, decrease it.

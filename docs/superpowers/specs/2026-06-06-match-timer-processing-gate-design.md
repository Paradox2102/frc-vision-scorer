# Match-Timer Processing Gate — Design

**Date:** 2026-06-06
**Status:** Draft for review

## Summary

Skip processing the dead air outside live match play. Read the broadcast **match
clock** via OCR, find the frame windows where the clock is actually *running*
(autonomous + teleop, plus a 3-second tail after the clock hits `0:00`), and restrict
all expensive work — YOLO detection, SAM2 export/tracking, scoring — to those windows.
Everything before the match, the auto→teleop break, and everything after the last fuel
settles is skipped.

If the clock can't be read, fall back to processing the whole video (current behavior)
with a warning — never silently drop a match.

## Motivation

A 215-second broadcast clip contains maybe ~160 s of actual play (≈20 s auto + ~2:20
teleop). The pre-match lineup, the auto→teleop break, and the post-match celebration
are all processed today at full cost (YOLO every frame + SAM2 export/propagation over
the whole range). Gating to live play removes a large fraction of that compute with no
loss of scored events.

## Period model (the windows we keep)

Two "running" windows, each ending with a 3-second tail to catch fuel in flight when
the clock reaches zero:

- **Autonomous:** clock counts `0:20 → 0:00`, then **+3 s**.
- **Teleop:** clock counts `2:20 → 0:00`, then **+3 s**.

Skipped: everything before auto starts, the auto→teleop break (clock paused/absent),
and everything after the teleop tail.

> "Running" = the OCR'd clock value is **decreasing** over time. A static, absent, or
> unparseable clock is "not running."
>
> Open choice for review: the 3 s tail is applied to **both** windows here (fuel can be
> in flight at the end of either period). If you only want it after teleop, that's a
> one-line change — flagged in the plan.

## Approach: a cheap pre-scan pass (decided)

A first pass reads **only** the clock region across the video — no YOLO, no SAM2 — and
produces the active frame ranges. Every downstream phase then iterates only those
frames. This is chosen over an inline per-frame gate because SAM2 is a **batch** phase
(it exports a frame range, then propagates); computing the windows once up front lets
the export, detection, and scoring phases all consume the same ranges without
duplicating the decision, and lets us know the teleop-end (+3 s) frame before exporting.

### Pre-scan algorithm
1. **Sample the clock** at a coarse interval (e.g. every ~0.5 s of video) by decoding
   that frame and OCRing the `match_clock` ROI. Parse `M:SS` → seconds; unparseable →
   `None`. (Tesseract on a ~120×56 ROI is negligible; the cost is the sparse decode.)
2. **Build the timeline** of `(frame_index, seconds|None)` samples.
3. **Find running segments:** maximal runs where `seconds` is non-increasing within a
   tolerance (allow OCR jitter: ignore a single off-sample, require an overall decrease
   across the run). Each run is a candidate live window.
4. **Classify by start value:** a run starting near `0:20` is autonomous; a run starting
   near `2:20` is teleop. (Used for labeling/validation and the tail; gating itself only
   needs "is a running window".)
5. **Build active frame ranges:** for each running segment, `[start_frame − small_pad,
   end_frame + 3 s · fps]`, where `end_frame` is where the run reaches `0:00` (or the
   last decreasing sample). A small lead pad (e.g. 0.5 s) avoids clipping the first
   action.
6. **Sanity gate / fallback:** if no `match_clock` region is configured, or the scan
   finds no sane decreasing run that reaches near `0:00`, **fall back to the full video**
   and log a clear warning. Coverage is never silently reduced by a bad read.
7. **Merge** overlapping/adjacent ranges into a final sorted list of `(start, end)`
   frame intervals.

The pre-scan reuses the **same coarse decode pass** the SAM2 frame-export already does
where possible; at minimum it is a single sequential read of sparse frames.

## Reading the clock

- New `match_clock` OCR region, in the same per-event region system as the scores
  (`OCR_REGIONS_1080P`, `scale_ocr_regions`, `/api/set-ocr-regions`).
- **Default region (verified against a real 1920×1080 broadcast frame):**
  `match_clock: (905, 68, 118, 56)` in the 1920×723 overhead-crop reference — the white
  clock box between the blue and red score boxes. Scales per-video via
  `scale_ocr_regions` exactly like the score regions. Works with no calibration for the
  standard layout; overridable per event.
- **Reader:** a `read_match_clock(overhead_frame) -> int | None` method mirroring
  `ocr_score`: crop ROI → 4× upscale → grayscale → threshold → `pytesseract` with
  `--psm 7 -c tessedit_char_whitelist=0123456789:`. Parse `M:SS` (and bare `SS`) to total
  seconds; reject implausible values (> ~3:00 or non-matching) as `None`.
  - Verified: this region + config reads `1:48, 1:27, 1:06, 0:44, 0:23` across a sample
    match, identical under normal and inverted threshold.

## Calibration UI (overridable default)

Add a **"Draw Match Clock"** button to the existing OCR setup flow in `index.html`
(beside "Draw RED/BLUE Score"), drawing a box that saves under the `match_clock` key.
The save/load path needs no backend change — `/api/set-ocr-regions` already persists
arbitrary region keys per event. With the verified default in place, this is an override,
not a requirement.

## Components / files

- `web/app.py`
  - `OCR_REGIONS_1080P`: add `"match_clock": (905, 68, 118, 56)`.
  - `read_match_clock()` method (mirrors `ocr_score`, `app.py:1601`).
  - `MatchTimeline` / `compute_active_windows(...)` — the pre-scan: sample clock, build
    running segments, emit merged active frame ranges (+ fallback). Pure-ish logic split
    into a testable helper (see Testing) that takes a list of `(frame, seconds|None)`
    samples and returns active `(start, end)` ranges, separate from the I/O that produces
    the samples.
  - Processing loop + SAM2 export: consult the active ranges; skip frames outside them.
  - A startup log line: windows found (auto/teleop frame ranges) or the fallback warning.
- `web/static/index.html` (+ its OCR-setup JS): "Draw Match Clock" button + wiring into
  the existing region-draw/save flow.

## Out of scope

- Using the clock to label scoring events as auto vs teleop (this is a *processing* gate
  only; scoring logic is unchanged beyond running on fewer frames).
- Reading the secondary sub-timer (`:07` shot clock) — not needed for gating.
- Auto-detecting the clock location — the verified default + manual override cover it.

## Testing

**Unit (`.venv/bin/python -m pytest`, per project memory):**
- `read_match_clock` parsing: `"1:27"→87`, `"0:23"→23`, `"2:20"→140`, `""/garbage→None`,
  implausible (`"9:99"`, `> 3:00`) → `None`. (Can test the parse helper directly on
  strings without a frame.)
- `compute_active_windows` on synthetic sample timelines:
  - auto run `20→0` then gap then teleop `140→0` → two ranges, each extended by 3 s·fps;
    the break between is excluded.
  - single-sample OCR dropout mid-run is tolerated (run not split).
  - all-`None` / no decreasing run → fallback (full-video range) flagged.
  - lead pad and 3 s tail land on the expected frame indices for a given fps.

**Integration:** run one real match (a `web/uploads/*.mp4`); confirm the logged auto +
teleop windows match the visible clock, the pre-match/break/post-match frames are
skipped, and scored events are unchanged versus a full-video run.

## Verified artifacts
- Sample frame + grid: clock located at `(905,68,118,56)` on 1920×1080 broadcast.
- OCR readout across the match confirmed monotonic countdown through teleop.

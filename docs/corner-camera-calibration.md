# Calibrating the corner cameras

The composite broadcast feed carries three cameras: the **main** overhead view (top)
and two low, wide-angle **corner** cameras (bottom-left / bottom-right, split at the
seam — see `data/field/composite_layout.json`). The corner cameras exist to (a) read
bumper numbers more clearly than the distant main view and (b) see field regions the
central hub occludes from main.

All three are calibrated in the `/calibrate` page with the same tag-anchored pinhole
resection. The corner cameras are the fiddly ones — this note records what works.

## Procedure (per corner camera)

1. Load the event in the app first (the page auto-fills the event key so the
   calibration saves where inference will look it up).
2. In `/calibrate`, pick the camera (`corner_left` / `corner_right`) from the
   dropdown. It grabs that camera's crop; the **central hub tags auto-detect** as
   green quads.
3. The other tags you can see are **not machine-decodable** at this resolution — the
   detector can't read their payload even zoomed. You identify them by their position
   on the field map. Click the tag on the frame, then its marker on the map. ROI
   auto-refine will *try* to snap corners but will usually fall back to a single
   center point for these — that's expected and fine.
4. Add as many **widely-spread** tags as you can — push for ones near the frame
   **edges** and at **different depths** (far wall vs. near rail). Spread is what
   conditions the solve; a tight central cluster alone is degenerate.
5. Solve. Judge the result by **(a) RMS** and **(b) the overlay** — see below.

## How to judge a corner calibration

- **RMS (px):** aim for under ~10 px. *Suspiciously* low RMS (< 1 px) with only a
  few clustered tags means overfitting, not a good fit — add spread tags.
- **Overlay coherence:** the reprojected green field grid should look like a
  sensible perspective grid, not exploding diagonal streaks. A wild overlay means
  one or more bad correspondences are dragging the pose.
- **Camera center sanity:** corner cams should land near their physical spot
  (corner_left near the blue end x≈0–1 m, corner_right near the red end x≈15–16 m,
  both low, ~0.7 m up). The solve log / overlay reflects this.
- **One bad click wrecks it.** The solver has no RANSAC, so a single mis-clicked tag
  (mapped to the wrong marker, or a sloppy center) shows up as a large per-tag
  residual and skews the whole pose. If the overlay is wild, delete tags one at a
  time (✕) — the culprit usually has the worst reprojection.

### Why the held-out **meters** number is unreliable for corner cams

The held-out validation projects a held tag's image point onto a field plane and
compares to its known position. For the **main** overhead camera this is meaningful.
For the **low corner** cameras it is **not**: their rays graze the ground plane at
shallow angles, so back-projecting image → field is numerically unstable (tiny
denominators → huge/∞), even when the forward reprojection is perfectly fine. So for
corner cameras, **trust RMS + overlay coherence, not the held-out meters.**

(The held-out projection now uses each tag's true height rather than the carpet, so
the metric is at least geometrically correct — but the grazing-angle instability
remains inherent to low oblique views.)

## What's intentionally NOT automated

The off-hub tags can't be auto-detected (see
`memory/corner-tag-calibration.md`), so corner calibration **requires a human** to
locate them by field position. There is no headless corner-calibration path.

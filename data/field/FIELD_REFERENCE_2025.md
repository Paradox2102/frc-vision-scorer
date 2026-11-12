# 2025 REEFSCAPE: Field and Broadcast Reference

This is the 2025 counterpart to `FIELD_REFERENCE.md` (2026 REBUILT). It is meant to
keep the pipeline from depending on one season's field. Everything below is in the
same WPILib field frame as 2026: origin at the Blue-wall corner on the
scoring-table side, +x toward Red, +y across the field, +z up, units in metres.

## Files

| File | What | Source |
|------|------|--------|
| `2025-reefscape-welded.json` | AprilTag layout, 22 tags (36h11, 6.5 in) | WPILib `allwpilib@v2026.2.2` (same tag as the 2026 file) |
| `2025-reefscape-andymark.json` | AndyMark-field variant (width 8.042 m; tags shift by mm–cm) | same |
| `2025_landmarks.json` | corners, midline, reef centres, processors, coral stations | derived from the welded tag layout |
| `advantagescope/2025_field2d.{jpg,json}` | top-down diagram, blue on the left, carpet rectangle in px | AdvantageScope `Field2d_2025FRCFieldWeldedV3` |
| `advantagescope/2025_andymark_field2d.{jpg,json}` | AndyMark variant | `Field2d_2025FRCFieldAndyMarkV3` |
| `wireframes/2025-reefscape-*.json` | 3D edge segments by structure group (reef, barge, …) | extracted from AdvantageScope `Field3d_2025FRCField*V3` GLB by `scripts/extract_field_wireframe.py` |
| `reference_frames/2025cur_qm11_*.jpg` | full 1920×1080 broadcast frames, teleop and endgame layouts | `data/prior_years/2025cur/2025cur_qm11.mp4` |

`FieldModel.load("2025-reefscape-welded")` loads the tag file as it is, since it has the
same schema as the 2026 file. The GT annotator's `/api/field2d?season=2025-…`
resolves to `2025_field2d.*` by year prefix.

**Frame check.** I loaded the AdvantageScope 3D field model, applied its config's
rotations (x 90°, z 180°), and shifted it by (L/2, W/2). After that, all 22 WPILib tag
centres were within 0.2 mm of the mesh surface. I also projected the tags onto the 2D
diagram, and each one landed on its feature.

**Broadcast check.** I solved a pinhole calibration for the 2025cur qm11 main view from
7 tags (2, 3, 6, 11, 13, 17, 22) at 0.52 px RMS, then projected the GLB wireframe through
it. The reef bases, coral stations, processor, barge and perimeter all sit on the real
structures (`reference_frames/2025cur_qm11_main_wireframe_check.jpg`).

## Field

- **Welded:** 17.548 × 8.052 m (690.875 × 317 in). **AndyMark:** 17.548 × 8.042 m.
  The 2025 field is about 1 m longer than 2026 (16.541 m).
- **Midline / barge:** x = 8.774 m. Blue half is x < 8.774.

| Element | Per alliance | Tags (Blue / Red) | Position |
|---------|-------------|-------------------|----------|
| **Reef**: hexagon with coral branches L2–L4 and the L1 trough; algae sits on it | 1 | 17–22 / 6–11 (z 0.308 m, one per face) | centres (4.489, 4.026) / (13.059, 4.026); tag faces 0.83 m from the centre |
| **Barge**: runs the full field width along the midline. It holds the **net** (algae) and the **cages**. | 3 cages each | 14, 15 / 4, 5 (z 1.868 m, pitched 30°) | x 8.272 (blue-facing) / 9.276 (red-facing), y 1.915 and 6.138 |
| **Coral station**: human-player chute in the corners of the alliance wall | 2 | 12, 13 / 1, 2 (z 1.486 m) | blue x 0.851, red x 16.697; y 0.655 (scoring-table side) and 7.396 (far side) |
| **Processor**: algae scoring slot in the side wall | 1 | 16 / 3 (z 1.302 m) | blue (5.988, 0.0) on the **scoring-table** wall; red (11.561, 8.056) on the **far** wall |

The field has 180° rotational symmetry, not mirror symmetry. That is why the blue
processor is on the near wall and the red one is on the far wall.

**Game pieces:** *coral* is a white PVC tube about 12 in long and 4 in in diameter. It
looks like a pale blob in footage and sticks out of reef branches. *Algae* is a teal
ball, 16 in across. Both appear in large numbers on and around the reefs. Expect
coral and algae to fool the detector, the same way 2026 fuel does.

**Match timing:** 15 s auto, then 2:15 teleop. The endgame (barge climb) is the last
**20 s**. 2026 auto is 20 s, so any code that assumes a 20 s auto clock is
2026-specific (see below).

**Endgame scoring:** park, shallow cage, or deep cage. Robots hang off the barge in
the middle of the field. A climbing robot's bumpers are **lifted off the carpet**,
which breaks the box-bottom-on-the-ground position assumption during the last 20 s.

## Broadcast layout (2025 Championship, Curie, qm11)

The composite is 1920×1080 at 29.97 fps. The main overhead camera is on the
**y ≈ 0 (scoring-table) side** looking across +y, with Blue on the left. That is the
same vantage as 2026. Tags detected in the main view: far-wall 2, 3, 13; reef faces
toward the camera 6, 11, 17, 22.

The colour-gradient seam between the main view and the bottom row is at
**y ≈ 724–731**. That matches the 2026 main crop of 0–723. The **bottom row changes
layout mid-match:**

**Teleop and auto layout (2 panels).** One divider at x ≈ 956–962.
- `aux_left`, about [0, 732, 956, 348]. It looks at the **Blue reef** from near
  midfield, toward the Blue wall. It sees reef tag 21, the face pointing at midfield,
  with coral station 12 behind it.
- `aux_right`, about [963, 732, 957, 348]. It is the mirror image: the **Red reef**,
  looking toward the Red wall. It sees coral station 1 and the reef.

These are not the 2026 corner cameras. Each one is a reef close-up shot from the
midfield side, so they are good for reading scoring and bumpers.

**Endgame layout (3 panels).** The switch happens around **0:20 on the match clock**
(144 s into this video), when the endgame starts. Dividers are at x ≈ 564–570 and
1350–1356.
- `aux_left` ≈ [0, 732, 564, 348]: the same Blue-reef camera, cropped narrower
  (tag 21 is still visible).
- `aux_centre` ≈ [571, 732, 779, 348]: a **new barge/cage close-up** for the climb.
- `aux_right` ≈ [1357, 732, 563, 348]: the same Red-reef camera, cropped narrower
  (tag 10 visible).

You described the endgame as switching to *two* other views. This Champs feed shows
three panels instead: two narrowed reef cameras plus one barge view. Regional
production probably differs, so detect the layout per event and per match phase
rather than hard-coding it.

**Overlay:** a division banner at about x 585–1335, y 990–1065 sits on top of the
bottom-row panels. The score bar across the top of the main view has the same
structure as 2026.

## What is still 2026-locked in the pipeline (as of main deb248f4)

**Update (season registry, `web/field/season.py`):** items 1, 2 and 4 are now driven by
the event's year. `season_for_event("2025hop")` picks `2025-reefscape-welded` (an
override in `web/uploads/field_season_<event>.json` can pick the AndyMark layout or
another year). Auto/manual calibration, the tag map, the wireframe (GLB groups for
non-2026 fields), the structure prisms (2025: two reef hexagons as no-robot zones,
occluders and radar shapes; the barge is not a zone), the match-clock search (15 / 135)
and the GT auto window follow it. 2026 is unchanged. Items 3 and 5 remain, as do the
2026-tuned corner-zone constants in `web/field/detect_bridge.py`.

1. **Season literal:** `"2026-rebuilt-welded"` is the default in about 10 Python files
   (`web/app.py`, `web/hud_render.py`, `web/field/auto_calibrate.py`, `corner_per_match.py`,
   `manta.py`, several `scripts/`). It should come from the event's year.
2. **Structure geometry:** `web/field/overlay_geometry.py` hand-codes the hub, tower,
   trench and bump. `zones.py`, `occlusion*.py`, `coverage.py`, `radar.py` and `tracker.py`
   reference 2026 elements. The GLB-derived wireframe groups are a per-season
   replacement source.
3. **Camera layout:** `composite_layout.json` assumes `main + corner_left + corner_right`
   for the whole match, and auto-calibration *drops cameras that move mid-match*. The
   2025 endgame switch would look like a camera move, and the barge view would never
   be used. Layouts need to be per phase.
4. **Match clock:** `web/app.py:640` calls `_find_clock_period(samples, 20, …)`, which
   assumes a 20 s auto. 2025 auto is 15 s. `gt_scoring.AUTO_SECONDS = 20.0` has the
   same problem.
5. **Detector and reader training data** are 2026-only: equipment hard negatives (hub,
   outpost, tablets) and the hub-zone datasets. 2025 needs its own coral and algae
   hard negatives.

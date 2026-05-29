# 2026 REBUILT — Field Reference

Reference data for the field-coordinate rework (see
`docs/superpowers/specs/2026-05-28-field-coordinate-foundation-design.md`).

## Canonical source & coordinate frame

- **AprilTag layout (canonical):** `data/field/2026-rebuilt-welded.json` — WPILib
  `AprilTagFieldLayout`, downloaded from
  `wpilibsuite/allwpilib@v2026.2.2`. 32 tags, full 3D pose (translation + quaternion).
- **Units:** WPILib JSON is **meters**; the CAD engineering drawings are **inches**.
  **Conversion verified exact:** inches × 0.0254 = meters. They share the **same
  origin and axes** — confirmed against tag 1 (CAD `467.64, 292.31, 35.00` →
  `11.8780, 7.4247, 0.889 m` = JSON `11.8779798, 7.4247756, 0.889`).
- **Frame (WPILib):** origin at the corner by the **Blue** alliance area
  (Blue Outpost/Tower tags 29–32 sit at x≈0.30 in / ≈0.008 m), **+x** runs down the
  long axis toward the **Red** alliance area (Red Outpost/Tower tags 13–16 at
  x≈650.9 in), **+y** across the field, **+z** up. Carpet at z=0.
- **Field size:** length **16.541 m** (651.22 in), width **8.069 m** (317.69 in).
- **Alliance dividing plane:** field midline at x = 16.541/2 = **8.2705 m**
  (325.61 in). x < midline → Blue half; x > midline → Red half. (Cross-check against
  per-element geometry during implementation.)

## Tag size

Not present in the WPILib JSON. The official FRC 36h11 tag **edge length** must be
supplied as a constant before computing tag-corner 3D coordinates from pose.
**TODO (confirm from 2026 game manual / FIRST AprilTag spec):** likely 6.5 in outer
(0.1651 m); verify the value used for pose estimation.

## Main-camera vantage

Primary feed is shot **from behind and above the scoring table**, looking across the
field — used for **x/y** localization. The scoring table runs along one long side
(the "SCORING TABLE" edge in the drawings). Player-station / Section A-A views give
**height** info. (Composited single video; corner cameras handled in SP3.)

## AprilTag table (CAD, inches) — verified mirror of the JSON

Mirror of the engineering "Welded Perimeter" tag key. Coordinates in inches;
Z-Rotation is heading about +z. Use the JSON for precise corner geometry (full
quaternion); this table is the human-readable cross-check and manual-fallback source.

| ID | Element | X (in) | Y (in) | Z (in) | Z-Rot |
|----|---------|--------|--------|--------|-------|
| 1  | Trench, Red   | 467.64 | 292.31 | 35.00 | 180° |
| 2  | Hub, Red      | 469.11 | 182.60 | 44.25 | 90°  |
| 3  | Hub, Red      | 445.35 | 172.84 | 44.25 | 180° |
| 4  | Hub, Red      | 445.35 | 158.84 | 44.25 | 180° |
| 5  | Hub, Red      | 469.11 | 135.09 | 44.25 | 270° |
| 6  | Trench, Red   | 467.64 | 25.37  | 35.00 | 180° |
| 7  | Trench, Red   | 470.59 | 25.37  | 35.00 | 0°   |
| 8  | Hub, Red      | 483.11 | 135.09 | 44.25 | 270° |
| 9  | Hub, Red      | 492.88 | 144.84 | 44.25 | 0°   |
| 10 | Hub, Red      | 492.88 | 158.84 | 44.25 | 0°   |
| 11 | Hub, Red      | 483.11 | 182.60 | 44.25 | 90°  |
| 12 | Trench, Red   | 470.59 | 292.31 | 35.00 | 0°   |
| 13 | Outpost, Red  | 650.92 | 291.47 | 21.75 | 180° |
| 14 | Outpost, Red  | 650.92 | 274.47 | 21.75 | 180° |
| 15 | Tower, Red    | 650.90 | 170.22 | 21.75 | 180° |
| 16 | Tower, Red    | 650.90 | 153.22 | 21.75 | 180° |
| 17 | Trench, Blue  | 183.59 | 25.37  | 35.00 | 0°   |
| 18 | Hub, Blue     | 182.11 | 135.09 | 44.25 | 270° |
| 19 | Hub, Blue     | 205.87 | 144.84 | 44.25 | 0°   |
| 20 | Hub, Blue     | 205.87 | 158.84 | 44.25 | 0°   |
| 21 | Hub, Blue     | 182.11 | 182.60 | 44.25 | 90°  |
| 22 | Trench, Blue  | 183.59 | 292.31 | 35.00 | 0°   |
| 23 | Trench, Blue  | 180.64 | 292.31 | 35.00 | 180° |
| 24 | Hub, Blue     | 168.11 | 182.60 | 44.25 | 90°  |
| 25 | Hub, Blue     | 158.34 | 172.84 | 44.25 | 180° |
| 26 | Hub, Blue     | 158.34 | 158.84 | 44.25 | 180° |
| 27 | Hub, Blue     | 168.11 | 135.09 | 44.25 | 270° |
| 28 | Trench, Blue  | 180.64 | 25.37  | 35.00 | 180° |
| 29 | Outpost, Blue | 0.30   | 26.22  | 21.75 | 0°   |
| 30 | Outpost, Blue | 0.30   | 43.22  | 21.75 | 0°   |
| 31 | Tower, Blue   | 0.32   | 147.47 | 21.75 | 0°   |
| 32 | Tower, Blue   | 0.32   | 164.47 | 21.75 | 0°   |

## Key field dimensions (CAD, inches) — for curated landmarks

From the overall top-view drawing (for `landmarks_3d`; transcribe more precisely
from the source CAD during implementation):

- Overall to diamond plate: **651.22** long; **317.69** wide.
- Half-length to centerline: **325.61** each side.
- Scoring-table-edge offsets: **182.11** to a reference upright each side.
- Hub uprights spacing: **143.50 + 143.50** about center; **167.00**, **156.61**
  reference spans.
- Hub structure heading marks: **120.00**, **121.30**, **121.25** verticals;
  **90.95 + 90.95** about midline; **35.95 + 35.95** central span.
- Hub element offsets: **44.40**, **47.00**, **115.05** (to uprights), **87.38**.
- Fuel/ball staging block centered on field midline (the dotted array).

Heights (Section A-A, inches): hub funnel top **120.36**; inside funnel **41.73**;
mid platform **72.00 / 44.25**; tag bands at **44.25** (hub), **35.00** (trench),
**21.75** (outpost/tower); scoring-table-side rail **22.25** / **35.00**.

> These are convenience dimensions; the controlled/CAD assembly drawings are
> authoritative. Pull exact landmark coordinates from the 3D model when building
> `data/field/2026_landmarks.json`.

# AdvantageScope 2D field drawings (2019–2026)

`<year>_field2d.jpg` + `<year>_field2d.json` are top-down field images from
AdvantageScope's asset archive (github.com/Mechanical-Advantage/AdvantageScopeAssets,
release `archive-v1`). Each one is originally a Chief Delphi render; the json's
`original_source` has the link.

| File | AdvantageScope asset | Tag layout |
|------|----------------------|------------|
| 2026_field2d | Field2d_2026FRCFieldV2 | 2026-rebuilt-welded.json |
| 2025_field2d | Field2d_2025FRCFieldWeldedV3 | 2025-reefscape-welded.json |
| 2025_andymark_field2d | Field2d_2025FRCFieldAndyMarkV3 | 2025-reefscape-andymark.json |
| 2024_field2d | Field2d_2024FRCFieldV3 | 2024-crescendo.json |
| 2023_field2d | Field2d_2023FRCFieldV4 | 2023-chargedup.json |
| 2022_field2d | Field2d_2022FRCFieldV3 | (no tags that year) |
| 2020_field2d | Field2d_2020FieldV1 | (no tags) |
| 2019_field2d | Field2d_2019FieldV1 | (no tags) |

(2021 had no new field; it reused 2020.)

All images are normalised to the **WPILib field frame**: Blue alliance wall on the
left, +x to the right, +y up, and about 1950 px wide. The AdvantageScope 2020–2026
assets are stored red-on-left, so they were rotated 180°. The 2019 asset was already
blue-on-left and is not rotated; each json's `transform` field records which.
`field_top_left_px` / `field_bottom_right_px` give the carpet rectangle in image pixels.

How orientation was checked:
- **2023–2026:** every WPILib AprilTag was projected through the json mapping and
  landed on its feature, with the blue-coloured structures on the left.
- **2019–2022:** by alliance colouring. 2019 has the blue HAB on the left. 2020 has
  the blue power-port triangle on the left wall. 2022 has the blue tarmacs on the
  left of the hub.

Used by the ground-truth annotator's top-down map (`/api/field2d`, which picks the
file by the season's year prefix).

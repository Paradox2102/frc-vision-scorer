# AdvantageScope 2D field drawing (2026)

`2026_field2d.jpg` is AdvantageScope's `Field2d_2026FRCFieldV2` asset
(github.com/Mechanical-Advantage/AdvantageScopeAssets, release `archive-v1`),
itself from the Chief Delphi high-resolution REBUILT field image
(https://www.chiefdelphi.com/t/2026-rebuilt-high-resolution-field-image/510611).

Changes from the original: rotated 180° so the blue alliance wall is on the left
(WPILib field frame, x right / y up, matching our field model and pipeline), and
scaled to 0.5×. `2026_field2d.json` gives the carpet rectangle in image pixels;
field size 16.541 × 8.069 m matches `data/field/2026-rebuilt-welded.json`.
The orientation was verified by projecting every AprilTag in our field model onto
the image: hub, tower, outpost and trench tags all land on their features.

Used by the ground-truth annotator's top-down map (`/api/field2d`).

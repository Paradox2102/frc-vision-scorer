# Vision scouting blog post (Nov 2026)

- `vision-scouting-post.md` is the post, in Chief Delphi / Discourse markdown with `[color=...]` headers.
- `vision-scouting.md` is the shareable GitHub-rendering copy (no color tags, captions under images).
  It's generated: edit the post, then run `python3 discourse.py github`.
- Images are referenced as `img/...` with Discourse size hints (`|690xH`). To post with all images:
  `python3 discourse.py prep`, drag everything in `upload/` into a new Chief Delphi composer, Ctrl+A,
  Ctrl+C, `python3 discourse.py merge`, then Ctrl+A, Ctrl+V in the composer. If some uploads fail
  (Chief Delphi rate-limits them), merge remembers the links it got and refills `upload/` with just
  the missing images; upload those and merge again. See `discourse.py`.
- `diagrams/*.html` are the sources for the diagram PNGs. Re-render one at 2x with headless Chrome
  (1000 px wide CSS, full page) after editing it.
- `img/candidates/` holds alternate replay frames and raw screenshots that didn't make the cut.

## Placeholders to fill

1. YouTube link to the demo reel (top of the post, and the "[demo reel](PLACEHOLDER)" link in "A Whole Event").
   Reel: `data/demo_reel/curie_reel.mp4` (19:32), built by `scripts/demo_reel.py data/demo_reel/curie_reel_spec.json --jobs 3`.

## Where the numbers come from

- GT chart: `scripts/score_ground_truth.py` on the latest replays as of Nov 15 2026 (current main,
  "visible" headline): 2026cur_qm8 d400c2a2, 2026casnd_qm14 e8274619, 2026hop_qm45 de207b14,
  2025hop_qm2 b1fc98da. Mean 86% tracked / 75% right / 1.1% wrong / 8.9 s mis-ID, consistent with the
  Nov 14 vast.ai R1 baseline (85.6 / 74.4 / 1.13 / 9.1).
- Speed: R1 ~5.2 min on vast RTX 5090 + Ultra 9 285K ($0.77/h), ~6.0 min on RunPod RTX PRO 6000 ($2.69/h).
- Swarm cost: 27 h of event x pods x $0.75-1.00/h; Jev ~5 cents per match (live reads + lineup stage).
- Curie full event (Nov 16 2026): 138 scheduled, 137 processed (qm37 robot-POV refused), qm32/qm72 wrong
  video, qm76 broadcast cut. 16.7 h total on the 3090, median 7.5 min, Jev $8.18. Tracked / named = mean
  over the 136 replays with real footage of slot-frames with a position / of those, with a team.
- Event-wide wrong-label estimate: `scripts/audit_replay.py` (sample + jev + score) on every replay,
  pooled mis-ID seconds / labelled seconds = 1.1% (4,413 confident judge items, judge coverage ~24%).
- Ours vs Manta / Match13 charts: `cmp/charts.py` (after `cmp/build.py`). In the post: `cmp_summary.png`
  (one row per metric, Manta + Match13 bars), paired Manta | Match13 scatters `cmp_auto/cycles/defense/speed.png`,
  and Manta-only `cmp_bump_manta.png`. `python charts.py pairs summary` re-renders those; `scatter` / `grid`
  make the older single-source versions (not in the post). Match13 = `GET https://actions.match13.com/v1/events/2026cur/matches`,
  `Authorization: Bearer $MATCH13_API_KEY` (repo .env, never commit); each team's xpPost / xAutoPost / xTelePost
  after its last qual, cached as `cmp/data/match13_2026cur_matches.json`.
- Blind-spot maps: `python blindspots.py 2026cur|2025hop` (`--refresh` re-reads replays) -> `img/blindspots_<event>.png`.
  Radar `state`/`corner_only` per robot per draw frame (a slot missing from radar = dropped: unseen at its last position), 0.25 m cells, Gaussian blur (1 cell cur, 2 cells hop). Green = main box
  >= 50% of the time, yellow = else side cams only >= 15%, red = neither. 2026cur: latest complete replay per qual in the
  Nov 15-16 batch (120 matches, qm32/37/72/76 skipped). 2025hop: qm2 b1fc98da (GT chart replay), qm3 4ca89189, qm4 568d7ea8;
  its "corner" insets are barge-mounted cams at midfield. Sparse cells (hop) take the nearest measured tier inside main's frame (lighter). Cone lines = outline of ground points inside each camera's image frame
  (`web.field.coverage._in_frustum` on the event's latest calibration), dashed back to the camera. Caches in `cmp/data/blindspots_<event>.json`.
- Scouting screenshots: `img/scouting_*_cur.png` (3128 auto paths + heatmap), `scouting_heatmap_defender.png`,
  `scouting_rankings_*.png` from :8004 /scouting (Playwright, 2x); alternates in `img/candidates/cur_*`.

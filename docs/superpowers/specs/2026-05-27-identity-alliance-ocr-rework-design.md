# Identity, Alliance & OCR Rework — Design

**Date:** 2026-05-27
**Status:** Approved (pending spec review)
**Component:** `web/app.py` identity/alliance pipeline + VLM read path
**Follows:** [`2026-05-27-reid-collision-recovery-design.md`](./2026-05-27-reid-collision-recovery-design.md)

## Problem

After the collision-recovery work (continuity now yields a stable 6 canonical
tracks), a live run on Qualification 37 surfaced three identity-layer failures:

1. **Wrong alliance color.** A robot's displayed alliance disagrees with its team
   number, and flip-flops frame to frame.
2. **HUD shows the wrong robot.** A robot whose number is plainly visible is
   labeled as a different team, on the wrong alliance.
3. **Bumper reads are low-yield.** The VLM fires often but rarely returns a valid
   roster number, and once a (possibly wrong) identity is confirmed it is never
   re-read, so mistakes are permanent.

### Evidence (log `20260527_164156`)

End-of-run state:
```
Robot 4 [blue] team=4322 confirmed=False   ← 4322 is a RED team, shown blue
Robot 3 [blue] team=9452 confirmed=True    ← correct
Robot 0 [red]  team=294  confirmed=True    ← correct
```
Alliance thrashing every interval, with no color evidence:
```
16:42:48  Alliance rebalance: Robot 1 blue -> red (weakest color evidence; B0/R0)
16:43:00  Alliance rebalance: Robot 1 blue -> red (B0/R0)
16:43:22  Alliance rebalance: Robot 1 red -> blue (B0/R0)   ← flips back
```
VLM yield: **90 Qwen passes, 16 valid roster reads.** The rest were `null` or
non-roster noise (`"200"`, `"236"`, `"20"`, `"35"`).

### Root cause

The collision-recovery change wired two previously-dead functions into the loop:

- **`_assign_alliances_by_embedding()`** re-clusters alliances by appearance every
  interval and overrides `r.alliance` for any robot that is not *confirmed* —
  including robots that already carry a tentative, roster-derived alliance from
  `assign_teams_from_votes` (`app.py:676`). That is how Robot 4 ends up `team=4322`
  (red) but `alliance=blue`. Its `_enforce_alliance_caps_by_color()` then reshuffles
  robots arbitrarily when color votes are zero (`B0/R0`), causing the flip-flop.
- **`assign_identities_now()`** assigns identities from reference embeddings — a
  signal that cannot reliably separate two same-alliance robots (the original
  design's stated central risk), producing cross-frame identity drift (the 9452
  robot's reads landed on both Robot 1 and Robot 3 at different times).

Identity (#2) is worsened by sparse, noisy VLM reads (#3): weak vote evidence lets
the global Hungarian bind a team to the wrong track, and the confirmed-robot skip
makes any such error permanent.

## Goals

- A robot's alliance is always consistent with its identity and stops flickering.
- Each robot gets the correct team number, with a deterministic fallback for the
  last unread robot on an alliance.
- The VLM spends its calls on legible bumpers, and a wrong identity can
  self-correct.

## Non-goals

- Touching continuity (Algorithm 1 / `choose_canonical_match` / `claimed`). It
  works and is unchanged.
- A live TBA API call during inference. The roster (`match_blue_teams` /
  `match_red_teams`) is already loaded from TBA via WebSocket before inference.
- Reworking scoring, calibration UX, or the detector.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Embedding identity/alliance functions | **Remove** `_assign_alliances_by_embedding` and `assign_identities_now` from the loop. Embeddings remain for continuity only. |
| Alliance authority | Roster is authoritative when a team is identified; bumper color when it is not. |
| Last-robot identity | Process-of-elimination auto-**confirms**, but remains overturnable by re-reads. |
| OCR strategy | Smart frame selection (legibility-scored crops) + re-read confirmed robots so wrong locks self-correct. |

## Design

### Component A — Alliance determination (single source of truth)

A robot's persistent `alliance` is derived, in priority order:

1. **Team identified (tentative or confirmed) → roster.**
   `alliance = "red" if team in match_red_teams else "blue"`. Nothing may override
   this.
2. **No team → bumper color.** Use the robot's accumulated `red_votes`/`blue_votes`
   (gathered each frame via `detect_bumper_color`, `app.py:~1630-1637`) majority;
   `"unknown"` if no votes. (Note: the separate `detect_bumper_color` call at
   `app.py:1582` only computes the per-frame provisional `det_alliance` for
   continuity — it is not the vote accumulator.)

Consequences:

- **Remove** the `_assign_alliances_by_embedding()` and `assign_identities_now()`
  calls from the loop (`app.py:~1668`). `_enforce_alliance_caps_by_color()`'s
  arbitrary `B0/R0` reshuffle is no longer reachable; caps are enforced naturally
  by alliance-constrained assignment (Component B).
- The per-frame provisional `alliance` computed in the tracking loop (used only as
  `det_alliance` for continuity preference and to drive color votes) is unchanged.

### Component B — Team identity: alliance-constrained votes + elimination

Rework `assign_teams_from_votes()` (`app.py:644`):

1. Determine each unconfirmed robot's alliance per Component A (roster if it has a
   tentative team, else color votes).
2. **Constrain candidates by alliance:** a robot may only be assigned teams on its
   own alliance. An `"unknown"`-color robot is unconstrained.
3. Solve the vote-maximizing assignment as a **single Hungarian with
   alliance-masked costs**: a cross-alliance (robot, team) cell is forbidden
   (infinite cost), an unknown-color robot's cells are all open, and same-alliance
   cells carry the negative vote total. This is one solve, not three separate ones,
   so the one-to-one uniqueness guarantee holds across the whole field. A robot
   with no votes for any allowed team stays unlabeled.
4. **Process-of-elimination:** after assignment, for each alliance, if exactly one
   roster team is still unassigned **and** exactly one still-unidentified robot has
   that alliance's color, assign that robot that team and **confirm** it.
5. Alliance is set from the assigned team (roster); never overwritten elsewhere.

Confirmation threshold for read-driven labels is unchanged (`v >= 3.0 and
v/total >= 0.6`).

### Component C — Smart OCR frame selection + self-correcting re-reads

Rework `vision_identify_robots()` (`app.py:456`) and the call gate (`app.py:~1651`):

1. **Legibility score** per candidate crop — a small pure helper over the bumper
   ROI, reusing the HSV machinery in `detect_bumper_color`:
   - **color strength:** max(red_ratio, blue_ratio) — is a colored bumper present?
   - **text presence:** ratio of bright/near-white, high-contrast pixels *within*
     the colored band — are there digits?
   - **sharpness:** variance of Laplacian on the band — reject motion blur.
   Combine into one score; expose thresholds as named constants.
2. **Selection:** rank all candidate robots' crops by score each pass; send the
   top-K above a floor to Qwen; skip crops below the floor (don't spend a call on a
   blurry/back-facing robot).
3. **Re-reads:** candidates include **confirmed** robots at a reduced cadence (not
   only unconfirmed). Reads keep accumulating votes for everyone.
4. **Audit / overturn:** a periodic check unlocks and relabels a confirmed robot
   when a *different* roster team's accumulated read votes exceed the current
   label's by a margin (named constant). This is how a wrong lock self-corrects.
   The audit operates purely on **read-vote margins**, not embeddings — it is
   independent of continuity/embedding quality and does not reintroduce the
   same-alliance discrimination weakness.

### Component D — Removals / walk-back

- Delete the `_assign_alliances_by_embedding()` and `assign_identities_now()` calls
  added to the loop in the collision-recovery work. The function bodies may remain
  (dead) or be removed; the plan will decide based on references.
- Re-introduce the alliance↔team consistency test that was dropped as "advisory."

## Data structures

- `RobotTrack`: no new fields required. Uses existing `team_number`,
  `team_number_confirmed`, `team_ocr_votes`, `red_votes`, `blue_votes`,
  `appearance_embedding`.
- New tunable constants (named, in the constants block): legibility weights/floor,
  re-read cadence, audit overturn margin.

## Error handling / edge cases

| Situation | Behavior |
|---|---|
| Robot has a tentative wrong team, color says other alliance | Roster (team) wins for alliance display; the read-vote audit (C4) is what corrects the *team*, which then flips alliance correctly. |
| All crops below legibility floor this pass | Send nothing; try again next pass. No Qwen call wasted. |
| Bumper color unknown for an unread robot | Robot is unconstrained in assignment; no alliance shown until a color vote or team arrives. |
| Two unidentified robots of the same color, one team left | Elimination does **not** fire (needs exactly one of each); waits for a read. |
| Re-read audit oscillation | Overturn requires a margin (not a single read) to prevent flapping; confirmed labels are sticky below the margin. |

## Testing

1. **Alliance-constrained assignment (unit, pure):** factor the assignment +
   elimination into a pure function (votes, alliances, roster, locked) → labels.
   Tests: a red robot is never assigned a blue team; elimination fills the last
   team on an alliance; two-unknowns case does not eliminate; no team assigned
   twice.
2. **Alliance↔team consistency (unit):** a robot with a team always reports the
   roster alliance for that team (the regression guard).
3. **Legibility score (unit, pure):** on saved crops — a sharp white-on-color
   bumper scores above a blurry/back-facing crop; expose via saved fixtures.
4. **Audit/overturn (unit, pure):** a confirmed label flips when another team
   exceeds it by the margin, and is sticky below it.
5. **Acceptance (`scripts/reid_acceptance.py`):** re-run on `qual37.mp4`; continuity
   metrics must not regress (≤6 tracks, 0 collisions). Identity accuracy still needs
   ground truth to score, but alliance↔team consistency can be asserted.

## Open questions / risks

- **Legibility heuristic tuning** needs real crops; the weights are starting points
  validated on saved frames.
- **Elimination depends on the other five being right.** Auto-confirm is justified
  by the re-read audit being able to overturn it; without reliable reads on the
  rest, a single early mis-confirm could cascade — the audit margin mitigates this.

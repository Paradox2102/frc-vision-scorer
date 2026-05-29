# Embedding-Based Robot Re-Identification — Design

**Date:** 2026-05-27
**Status:** Approved (pending spec review)
**Component:** `web/app.py` inference pipeline, new `web/embedder.py`

## Problem

The live inference pipeline cannot keep robot identities consistent across a
match. Three failures dominate, all observed by the user:

1. **ID swaps** — when robots cross or cluster near the hub, canonical IDs and
   team numbers stick to the wrong robot afterward.
2. **Wrong team-number assignment** — the team number bound to a track is wrong
   or flips.
3. **Duplicate team numbers** — two physically distinct robots are labeled with
   the same team number, which is physically impossible.

### Root cause

Two code-level facts produce all three failures:

- **Continuity is decided by pixel position alone.** `get_canonical_robot_id()`
  merges a new ByteTrack ID into the *nearest* existing canonical track by pixel
  distance (`ROBOT_MERGE_THRESHOLD = 120`). When ByteTrack drops or re-issues an
  ID during occlusion, the nearest-position guess scrambles identities.
- **The appearance descriptor cannot separate same-alliance robots.**
  `_compute_appearance_fingerprint()` is an HSV color histogram. All three
  robots on an alliance wear the same red (or blue) bumpers, so the histogram
  distinguishes alliance but carries almost no signal for telling the three
  robots of one alliance apart.

Identity is also computed greedily per robot (`appearance_match_robots()` votes
each robot independently, then patches collisions with
`_clear_team_from_other_robots()`). Greedy per-robot voting cannot express the
invariant "these N labels are a permutation of these N robots," so it reaches
impossible states and cleans them up after the fact.

## Goals

- Keep each physical robot on one stable canonical track.
- Assign the correct team number to each track.
- Make duplicate team numbers structurally impossible.

## Non-goals (deferred)

- Replacing ByteTrack with BoT-SORT or SAM2. The game camera is static, so
  camera-motion compensation is unnecessary; ByteTrack stays.
- Field homography / metric coordinates / top-down radar view.
- Changing the detector (YOLO26s) or the calibration UX.

## Constraints

- **Fully local / offline.** After a one-time model-weights download, no network
  or cloud API. (Matches the existing local Qwen2.5-VL-7B pattern.)
- **Offline, accuracy-first.** Processing recorded match video; throughput is not
  a hard constraint, so a strong embedding run generously is acceptable.
- **Roster from TBA.** The six team numbers and their alliances are known per
  match; a mandatory per-match calibration step labels each robot.
- **Hardware:** RTX 3090 (24 GB). DINOv2-small (~22M params) is negligible
  beside the existing Qwen-7B.

## Architecture

One new self-contained module plus surgical edits at four seams in `app.py`.
YOLO, ByteTrack, OCR/Qwen, and the calibration UX are unchanged.

```
                detection box + frame
                        │
                  ┌─────▼──────┐
                  │ embedder.py │  DINOv2, FRC-agnostic
                  │  embed()    │  → L2-normalized vector
                  └─────┬──────┘
                        │ embedding
        ┌───────────────┼────────────────────┐
        ▼                                     ▼
  Algorithm 1                           Algorithm 2
  embedding-aware continuity            joint identity assignment
  (get_canonical_robot_id)              (replaces greedy HSV voting)
  → cleaner tracks                      → correct + unique team numbers
                                         anchored by OCR/Qwen locks,
                                         constrained by alliance color
```

Two IDs are deliberately separated:

- **canonical `robot_id`** — the track (short-term continuity; used for
  trajectory→shooter scoring).
- **`team_number`** — the identity label on the track.

Algorithm 1 keeps tracks clean. Algorithm 2 re-derives labels from appearance
every interval with a uniqueness guarantee, so it self-corrects even when
continuity briefly slips.

## Components

### New: `web/embedder.py`

A small class with no knowledge of FRC, robots, or app state. Images in, vectors
out — making it unit-testable and swappable (DINOv2 ↔ SigLIP is one constant).

```python
class RobotEmbedder:
    def __init__(self, model_name="facebook/dinov2-small"):  # lazy-loaded, cached
        ...
    def embed(self, frame, box) -> np.ndarray | None:   # L2-normalized vector
        ...
    def embed_batch(self, frame, boxes) -> list[np.ndarray | None]:  # batched
        ...
```

- Crops the detection box with a small inset to shed background, runs DINOv2,
  returns an L2-normalized embedding (or `None` for an invalid/tiny crop).
- Comparison is cosine similarity (dot product of normalized vectors), replacing
  `cv2.compareHist`.
- `embed_batch()` exists because both real call sites embed multiple crops from
  one frame: the calibration detection path loops over every detected robot, and
  inference re-embeds all live robots each interval. It is a single GPU forward
  over stacked crops, not a per-crop loop. If profiling shows it unnecessary, it
  can collapse to `embed()` in a loop without touching callers.

### Edits in `app.py`

| Seam | Today | After |
|---|---|---|
| `RobotTrack` dataclass | `appearance_fingerprint` (HSV) | `appearance_embedding` (DINOv2, EMA) |
| `_reference_fingerprints` (`:233`, seeded `:513`) | team→HSV | `_reference_embeddings` team→DINOv2 |
| `get_canonical_robot_id()` (`:1087`) | nearest pixel position | position **+ embedding** (Algorithm 1) |
| `appearance_match_robots()` (`:519`) | greedy per-robot HSV voting | joint Hungarian assignment (Algorithm 2) |

**Retired:** `_compute_appearance_fingerprint()` (`:1870`) and
`_compare_fingerprints()` (`:1935`). There are **three** live call sites to
account for, not two:

1. `appearance_match_robots()` (`:519`) — replaced by Algorithm 2.
2. Reference seeding (`:513`) — replaced by DINOv2 reference-embedding seeding.
3. `_detect_robots_in_frame()` (`:2405`), the calibration detection path —
   computes a fingerprint and returns it as a `"fingerprint"` field in the
   calibration API response (`:2417`). That response field is **dead output**:
   the frontend (`web/static/`) never reads it and it does not feed
   `_reference_*`. Removing the function therefore requires deleting this call
   and dropping the dead `"fingerprint"` field from the response payload.

**Reference embeddings have a single source of truth: the `:513` seeding seam.**
The calibration detection path (`:2405`) does *not* seed references and must not;
it only returned the now-dead field. Do not introduce embedding seeding at
`:2405`.

## Algorithm 1 — Embedding-aware continuity

The new detection's embedding is computed once and passed into
`get_canonical_robot_id()`. Already-mapped ByteTrack IDs still return their
cached canonical ID (ByteTrack continuity within an unbroken track is trusted).
For a **new** ByteTrack ID:

```
for each existing canonical track r (same-alliance preferred):
    pos_score = max(0, 1 - dist(det, r) / POS_GATE_PX)
    app_sim   = clamp01(cosine(det.embed, r.embed))
    combined  = W_POS * pos_score + W_APP * app_sim

best = argmax_r(combined)
if app_sim(best) >= APP_FLOOR and dist(best) <= POS_GATE_PX:
    merge to best
elif caps allow (≤3 per alliance, ≤6 total):
    create new canonical
else:                          # at cap — must be an existing robot
    assign to best EMBEDDING match (not nearest position)
```

`APP_FLOOR` is what prevents a swap: a detection is never absorbed into a
visually different track merely because it is the closest pixel-wise. At the
3-per-alliance cap, the fallback uses *appearance*, the opposite of today's code,
which grabs the nearest robot regardless of how it looks.

A track with no embedding yet (just created) is treated with a neutral
`app_sim` so position decides until it accrues one.

## Algorithm 2 — Joint identity assignment

Every identity interval, solve a one-to-one matching of live tracks to
calibration references:

```
locked = tracks whose team is OCR/Qwen-confirmed
         → removed from the problem; their team removed from available refs
rows = remaining live tracks ; cols = remaining reference teams
C[i,j] = 1 - cosine(track_i.embed, ref_team_j.embed)
C[i,j] = +inf  if alliance(track_i) != alliance(team_j)   # hard constraint
row_idx, col_idx = scipy.optimize.linear_sum_assignment(C)
for each matched (i,j) with C[i,j] <= ASSIGN_MAX_COST:
    vote team_j onto track_i           # accumulate across intervals
    confirm once agreement is strong enough
```

Because each interval's assignment is one-to-one by construction, "two robots →
same team number" is structurally impossible. The
`_clear_team_from_other_robots()` post-hoc cleanup is no longer needed.

Alliance is a **hard constraint** (forbidden pairs get infinite cost) because
bumper color is reliable; this halves the matrix. A track whose alliance is
unknown forbids no column.

### How OCR/Qwen fold in

The existing OCR/Qwen vote-lock is the high-confidence anchor. A confidently read
bumper number locks that track+team and removes both from the matrix. The
embedding assignment then resolves only the unread robots, still with the
uniqueness guarantee. The three signals compose: **OCR locks what it can read,
embeddings + Hungarian resolve the rest, alliance color constrains the search.**

## Reference embeddings

- **Seeding (calibration, `:513` seam):** for each clicked-and-confirmed robot,
  compute its DINOv2 embedding, optionally averaged over a few frames near the
  calibration frame for robustness. Store `team_number → embedding`.
- **Mid-match refresh (nice-to-have):** when a team is OCR-locked during
  inference, refresh that team's reference embedding from the track's live EMA —
  a confirmed example in this match's exact lighting — improving matches for the
  remaining robots.

## Data structures

- `RobotTrack.appearance_fingerprint` → `appearance_embedding: Optional[np.ndarray]`,
  updated by EMA: `e = normalize(0.7*e_old + 0.3*e_new)` when the crop is valid.
- `InferenceJob._reference_fingerprints` → `_reference_embeddings: Dict[str, np.ndarray]`.
- `appearance_votes` semantics unchanged (now driven by assignment results).

## Constants (tunable; starting points)

| Name | Value | Meaning |
|---|---|---|
| `EMBED_MODEL_NAME` | `"facebook/dinov2-small"` | embedding backend |
| `APP_FLOOR` | 0.55 | min cosine to merge a new detection into a track |
| `POS_GATE_PX` | 200 | max pixel distance to consider a merge |
| `W_POS` / `W_APP` | 0.4 / 0.6 | continuity score weights |
| `ASSIGN_MAX_COST` | 0.45 | max Hungarian cost (≈ min cosine 0.55) to accept a label |
| `IDENTITY_INTERVAL` | TBD frames | how often Algorithm 2 runs |

These need tuning on real footage; they are explicit constants, not magic
numbers buried in logic.

## Error handling / edge cases

| Situation | Behavior |
|---|---|
| Embedding model fails to load (OOM, no weights) | Log warning; fall back to position-only continuity + OCR/Qwen identity (today's behavior). No crash. |
| Empty/tiny crop | `embed()` returns `None`; keep last embedding, skip EMA update |
| Track has no embedding yet | Neutral `app_sim`; excluded from that interval's matrix |
| Partial calibration / <6 references | Rectangular matrix; `linear_sum_assignment` matches `min(rows, cols)`; unmatched tracks get no label that interval |
| Track alliance unknown | No hard constraint for that row; embedding decides |
| Calibration skipped | References empty → assignment no-ops; continuity still benefits (tracks accrue embeddings → fewer swaps) |

**One-time network:** DINOv2-small weights download from Hugging Face on first
run, then cache locally. Fully offline thereafter.

## Testing

Decision logic is factored into pure functions so the hard parts test without
video, YOLO, or ByteTrack.

1. **`embedder.py` (unit):** `embed()` returns an L2-normalized vector of the
   expected dim; `cosine(crop, crop) ≈ 1.0`; distinct-robot crops score
   measurably lower than same-robot. Uses saved crops from existing data.
2. **Continuity decision (unit):** pure `choose_canonical(pos, embed, tracks, caps)`.
   Ambiguous position picks the embedding match, not the nearest; below
   `APP_FLOOR` refuses the merge and creates new; at cap assigns by embedding and
   never exceeds the cap.
3. **Joint assignment (unit, property-based):** pure
   `assign_identities(tracks, refs, alliance_map, locked)`. Never assigns a team
   twice; honors the alliance hard-constraint; excludes locked tracks/teams;
   handles rectangular input.
4. **Acceptance on real footage (the real proof):** a clip from
   `data/videos/2026`. Metrics — (a) canonical-ID switches per physical robot,
   (b) frames with duplicate team numbers (must be 0 by construction),
   (c) team-number accuracy at sampled frames. HSV baseline vs DINOv2.

## Open questions / risks

- **Embedding discrimination on robots.** DINOv2 is expected to separate
  same-alliance robots well, but this is the central assumption. The acceptance
  test validates it; SigLIP is the proven fallback (one-constant swap).
- **`IDENTITY_INTERVAL` and threshold tuning** require a labeled clip; values
  above are starting points.
- **Reference quality from a single calibration crop.** Averaging a few frames
  mitigates this; the mid-match refresh further improves it.

## Future work (explicitly out of scope here)

- BoT-SORT with ReID (if a non-static camera is ever introduced).
- Field homography → metric coordinates + top-down radar view, which would make
  `POS_GATE_PX` physically meaningful and enable a motion model.

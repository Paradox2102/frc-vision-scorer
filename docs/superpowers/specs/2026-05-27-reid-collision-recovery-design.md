# Re-ID Collision Recovery & Algorithm 2 Wiring — Design

**Date:** 2026-05-27
**Status:** Approved (pending spec review)
**Component:** `web/app.py` inference loop, `web/reid.py`
**Follows:** [`2026-05-27-embedding-reid-robot-tracking-design.md`](./2026-05-27-embedding-reid-robot-tracking-design.md)

## Problem

During live analysis, the VLM (Qwen) reads bumper numbers correctly, but the HUD
labels robots wrong — e.g. a blue robot whose crop reads `2485` is drawn as
`2102?`. The user reported this as a VLM↔HUD mismatch. It is not: the label layer
is faithfully displaying a corrupted track.

### Evidence (session `20260527_153951`)

- Only **3 distinct canonical tracks** ever received VLM reads, for 6 robots:
  `Robot 0` (3 reads), `Robot 4` (16 reads), `Robot 5` (3 reads).
- **Canonical track 4 absorbed five teams** across both alliances — `4322`,
  `2102`, `1538`, `2485`, `9452`.
- Two physically distinct robots merged into track 4 **in a single frame**:
  ```
  15:42:11.529  Qwen raw output: {"team": "9452", "color": "blue"}
  15:42:11.529  Qwen raw output: {"team": "2485", "color": "blue"}
  15:42:11.529  Vision read: Robot 4 += 9452
  15:42:11.529  Vision read: Robot 4 += 2485
  ```
  Two simultaneous detections both resolved to canonical ID 4 — impossible for
  one robot.
- Track 4's accumulated votes `{2102: 3.5, 2485: 3.5, 9452: 2.0, 4322: 2.0,
  1538: 1.0}` feed `assign_teams_from_votes()` (`app.py:644`), which picks the
  earliest-accumulated max (`2102`). The HUD label is just `robot.team_number`,
  so the box reads `2102?` while the live crop reads `2485`.

The VLM was never wrong. Its correct reads are poured into a track that is
physically two-plus robots fused together.

### Root cause

The embedding re-ID (Algorithm 1) is wired into `get_canonical_robot_id`
(`app.py:1207`) and the embedder loads successfully (no failure logged), yet the
merge still happens. Three structural gaps:

1. **No per-frame uniqueness.** Each detection independently resolves its
   canonical ID. Nothing forbids two detections in the same frame from both
   resolving to canonical 4.
2. **ByteTrack-ID trust bypass** (`app.py:1215`). Once a `bytetrack_id →
   canonical` mapping exists it is returned with no appearance check. When
   ByteTrack reuses or swaps an ID in the crowded hub area, a different physical
   robot flows into the same canonical track ungated.
3. **EMA embedding contamination** (`app.py:1567`). A track's embedding is an EMA
   over *every* detection that maps to it. Once two robots merge, the centroid
   becomes a muddy average that clears `APP_FLOOR` against many robots — so the
   veto that should *stop* merges starts *permitting* them. Self-reinforcing, and
   worst for same-alliance robots (2485 and 9452 are both blue).

Compounding: **Algorithm 2 is dead code.** `assign_identities_now()`
(`app.py:682`) and `_assign_alliances_by_embedding()` (`app.py:579`) are fully
implemented but never called. The joint embedding assignment with the uniqueness
guarantee — designed to re-derive labels every interval and self-correct
continuity slips — never runs. `appearance_votes` is therefore always empty.

## Goals

- One canonical track per physical robot (`≤6` total, `≤3` per alliance).
- No two live detections share a canonical track in one frame.
- A track whose continuity slips self-heals rather than staying corrupted.
- The designed embedding-identity assignment (Algorithm 2) actually runs.

## Non-goals

- Replacing the greedy per-detection canonical resolution with a joint per-frame
  Hungarian (Option B, considered and rejected below).
- Changing the label-rendering code. Once tracks are clean, labels are correct.
- Re-tuning thresholds beyond what the acceptance clip requires.

## Decisions

Three decisions were made during brainstorming:

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Full design-intended wiring | Address all three gaps plus wire Algorithm 2. |
| Collision recovery | **Split & re-resolve (self-healing)** | Evict the mismatched detection; let Algorithm 2 re-label. Matches the original design's "self-correct even when continuity slips." A robot may flash `R#` briefly. |
| Signal composition | **VLM-primary, embeddings fill gaps** | A corroborated Qwen read (existing `v≥3 & 60%` rule) locks the team; the embedding Hungarian only labels tracks Qwen never read. Matches the original design intent. |

### Approach fork (resolved)

**Option A — Greedy "claimed-set" (chosen).** Keep per-detection resolution; pass
a `claimed: set[int]` of canonical IDs already taken this frame into
`choose_canonical_match`, which excludes claimed tracks as merge/assign
candidates. Process detections best-match-first so the strongest claimant keeps
the track. Minimal, well-isolated change to an already-tested pure function.

**Option B — Joint per-frame Hungarian (rejected).** One optimal detection→track
assignment per frame. Cleaner in theory but a sizable rewrite of the loop;
ByteTrack already supplies most within-track continuity, so the marginal gain is
small for the cost and risk.

## Design

### Component 1 — Self-healing continuity

`reid.choose_canonical_match` gains a `claimed` parameter (set of canonical IDs
already assigned this frame). Claimed tracks are removed from both the
same-alliance and other-alliance candidate pools before scoring, and from the
cap-full `assign` fallback. The caps still count all tracks; only candidacy for
*this* detection is restricted.

**Return-contract change.** Today `choose_canonical_match` returns exactly
`("merge", id)`, `("create", None)`, or `("assign", id)`, and
`get_canonical_robot_id` handles them with `if action in ("merge","assign")` else
create. Component 1 adds a fourth outcome: when caps are full and **every** track
is already claimed this frame, no canonical can be issued — return
`("drop", None)`. `get_canonical_robot_id` then returns a sentinel (e.g. `None`)
and the loop **skips that detection**: it is not appended to
`tracked_robot_boxes`, gets no EMA update, and is omitted from
`detection_to_tracked` (so the draw loop falls through to the untracked-robot
branch). A 7th simultaneous detection cannot be a 7th robot.

**Ordering key (pins "best-match-first").** Resolution is a two-pass greedy, not
raw YOLO confidence: Pass 1 computes each detection's best candidate canonical and
its combined position+appearance score (the same `W_POS·pos + W_APP·app` score
`choose_canonical_match` uses). Detections are then resolved in descending
best-score order, so the strongest overall claimant binds a contested track
first; weaker detections fall to unclaimed tracks, new IDs, or drop. Using the
match score (not detection confidence) is what makes the Option A correctness
argument hold, since the ordering key is the same quantity that decides the
contest.

`get_canonical_robot_id` (`app.py`) changes:

- Accepts and maintains the per-frame `claimed` set; marks the returned canonical
  ID claimed before returning.
- **Cached-ID re-validation:** for a `bytetrack_id` already in
  `bytetrack_to_canonical`, compute `cosine(det_embed, track.embedding)`. If it is
  below `APP_FLOOR` (the cached canonical no longer looks like this detection) OR
  the cached canonical is already claimed this frame, drop the stale mapping and
  re-resolve through `choose_canonical_match`. Otherwise return the cached ID as
  today (fast path, ByteTrack trusted).

The inference loop resolves detections **best-match-first**: compute a cheap
match score (or rely on detection confidence) so the most confident claimant
binds a contested track first and weaker detections fall to unclaimed tracks or
new IDs. Loser detections that cannot match any unclaimed track within gate/floor
create a new canonical (caps permitting) — the "split."

### Component 2 — Stop EMA contamination

In the loop's EMA update (`app.py:1567`), fold a detection into a track's
`appearance_embedding` only when it matches the existing embedding
(`cosine ≥ APP_FLOOR`), or when the track has no embedding yet. A transitional or
mismatched crop is skipped, keeping the centroid clean and the veto sharp.

### Component 3 — Wire in Algorithm 2

Every `IDENTITY_INTERVAL` frames, after `assign_teams_from_votes()`:

1. `_assign_alliances_by_embedding()` — robust alliance assignment by embedding
   clustering (confirmed tracks anchor; unconfirmed cluster).
2. `assign_identities_now()` — embedding Hungarian over unlocked tracks/teams,
   accumulating `appearance_votes`, confirming at `APPEARANCE_CONFIRM_VOTES`.

**VLM-primary precedence:** `assign_teams_from_votes()` confirms corroborated
Qwen reads first; `assign_identities_now()` already excludes
`team_number_confirmed` tracks and their teams, so it composes by construction —
it fills only what Qwen has not locked. Alliance color remains a hard constraint
inside the Hungarian.

### Component 4 — Label layer

No change. With tracks one-per-robot, `assign_teams_from_votes` no longer sees a
track holding five teams' votes, so stale `2102?`-style labels stop appearing.

## Data flow

```
detection + frame
   │
   ├─ embedder.embed(box) ─────────────► det_embed
   │
   ├─ get_canonical_robot_id(bt_id, pos, alliance, det_embed, claimed)
   │     • cached bt_id + appearance OK + unclaimed → return cached (fast path)
   │     • else choose_canonical_match(..., claimed) → merge/assign/create
   │     • mark canonical claimed
   │
   ├─ EMA update IFF cosine(det_embed, track.embed) ≥ APP_FLOOR
   │
   └─ every IDENTITY_INTERVAL:
         assign_teams_from_votes()          # VLM locks (primary)
         _assign_alliances_by_embedding()   # robust alliance
         assign_identities_now()            # embeddings fill unlocked (uniqueness)
```

## Error handling / edge cases

| Situation | Behavior |
|---|---|
| `det_embed` is `None` (tiny/invalid crop) | Skip cached-ID re-validation (no veto without an embedding); fall back to position-only candidacy, as today. No EMA update. |
| All 6 canonicals claimed, loser detection unmatched | Cap-full `assign` to best **unclaimed** embedding match; if none unclaimed, `choose_canonical_match` returns `("drop", None)` and the loop skips the detection (see Component 1 return-contract change). A 7th simultaneous detection cannot be a 7th robot. |
| Track briefly evicted (split) | Shows `R#` until Algorithm 2 / VLM re-labels; expected and acceptable per the recovery decision. |
| Algorithm 2 with no references (calibration skipped) | `assign_identities_now` no-ops (guards on `_reference_embeddings`); continuity still benefits. |

## Testing

1. **`reid.choose_canonical_match` (unit, extend `tests/`):**
   - A claimed canonical is never returned as merge/assign, even when it is the
     best position+appearance match.
   - With all good candidates claimed, a dissimilar detection creates a new track
     rather than force-merging.
2. **Cached-ID re-validation (unit):** a cached `bytetrack_id` whose detection
   embedding diverges below `APP_FLOOR` triggers re-resolution.
3. **EMA gating (unit):** a sub-`APP_FLOOR` detection does not alter the track
   embedding.
4. **Algorithm 2 composition (unit):** after a tentative (unconfirmed) vote-team
   is set, `_assign_alliances_by_embedding()` must not churn that robot's alliance
   in a way that flips a still-valid assignment — assert that re-clustering leaves
   confirmed robots' alliances fixed and does not destabilize a track that
   `assign_teams_from_votes` just labeled mid-interval.
5. **Acceptance (`scripts/reid_acceptance.py`):** re-run on the logged clip.
   Metrics — (a) canonical count converges to ≤6, (b) **zero** frames with a
   duplicate canonical across simultaneous detections, (c) **zero** duplicate
   team numbers, (d) team-number accuracy at sampled frames vs. the current
   baseline.

## Open questions / risks

- **Greedy ordering.** Best-match-first mitigates but does not eliminate
  ordering artifacts; the acceptance metric (b) is the guard. If artifacts
  appear, Option B (joint Hungarian) is the documented fallback.
- **APP_FLOOR reuse for cached-ID re-validation.** The same floor that gates
  merges gates re-validation; same-alliance discrimination is the central risk
  carried over from the original design. Tunable; validated on the clip.

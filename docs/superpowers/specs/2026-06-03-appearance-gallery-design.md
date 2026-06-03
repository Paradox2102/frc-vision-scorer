# Per-Event Appearance Gallery for Obscured-Number Recovery

**Date:** 2026-06-03
**Status:** Approved design (pending spec review)

## Problem

The bumper-number OCR pipeline reads a team number off each robot crop and matches
it against the match roster, producing a confidence (e.g. `B2 1787 > 1787, 95%`).
When the number is obscured, motion-blurred, or angled away, the read fails outright
(`B3 37 > no match, 0%`) even when the robot is clearly visible and was confidently
identified moments — or matches — earlier.

We want to recover the identity of an obscured robot by matching its overall
appearance to a previously confident identification at the same competition event.

## Key observations

- A team runs the **same physical robot for the whole event**, so its overall look is
  highly distinctive and stable across matches.
- Bumpers themselves are **low-information**: a solid red or blue band with white
  digits in the standard FRC font, near-identical across teams. The discriminating
  signal lives in the **superstructure**, captured by the whole-robot image.
- Bumpers only ever take **two forms** per event: red (on red alliance) and blue
  (on blue alliance). The same robot therefore produces two distinct appearance
  clusters.
- A robot looks **substantially different from each side** (front/back/left/right),
  so a single fingerprint per `(team, color)` only matches one viewpoint.
- The pipeline already computes a per-track **DINOv2 embedding** (`web/embedder.py`,
  384-dim, L2-normalized, inset whole-robot crop) for within-match re-ID. This is the
  right fingerprint — no new feature extractor is needed.

## Concept

A persistent, **event-scoped** appearance gallery mapping
`(team, alliance_color) → diverse set of appearance fingerprints`.

- Confident OCR reads **deposit** fingerprints into the gallery.
- Obscured / unreadable tracks **query** the gallery to recover their team.
- The gallery accumulates across every match at the event, so recovery gets stronger
  as the event progresses ("gets better over time").

## Authority model

A gallery match **can confirm a team on its own**, but:

- It **never overrides** an existing confirmed lock.
- A genuine high-confidence OCR read always outweighs it.
- "Confirms on its own" is routed through the existing 3-consecutive-agreement lock
  (`ConsecutiveValueTracker`), so confirmation requires **sustained** appearance
  agreement across passes — a single frame can never mislabel a look-alike robot.

## Architecture

### New component — `web/field/appearance_gallery.py`

A self-contained `AppearanceGallery` class with no knowledge of tracks, votes, or the
database (those live at the integration layer). Unit-testable with synthetic vectors.

Interface:

- `add(team: str, color: str, embedding: np.ndarray, legibility: float) -> None`
  Deposit an exemplar. Maintains a **diverse, capped set** per `(team, color)`:
  - A new exemplar is stored only if its max cosine similarity to existing exemplars
    for that key is below `GALLERY_DEDUP_SIM` (≈0.92). This captures distinct
    viewpoints instead of many near-identical frames — i.e. the diverse set *is* the
    viewpoint-coverage mechanism (front/back/side fill in as the robot rotates).
  - Capped at `GALLERY_MAX_EXEMPLARS` (≈12, sized for ~4 sides × a couple variations).
    When full, evict the lowest-legibility exemplar.
- `query(embedding: np.ndarray, color: Optional[str]) -> list[tuple[str, float]]`
  Cosine similarity against exemplars, filtered to the detected `color` (searches both
  colors if `color` is unknown/None). Returns the best similarity per team, sorted
  descending. "Second-best team" in the matching rules below means the next *distinct
  team* in this ranked list (not the second exemplar of the same team).
- `serialize() -> list[dict]` / `AppearanceGallery.load(rows) -> AppearanceGallery`
  for database round-trip.

### Persistence — new `appearance_gallery` table

```
appearance_gallery(
  id            INTEGER PRIMARY KEY,
  event_key     TEXT NOT NULL,
  team          TEXT NOT NULL,
  alliance_color TEXT NOT NULL,        -- 'red' | 'blue'
  embedding     BLOB NOT NULL,         -- float32 × 384
  legibility    REAL,
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_gallery_event ON appearance_gallery(event_key, alliance_color);
```

One row per exemplar. Added via the same lightweight `ALTER TABLE`/try-except
migration pattern already used for the replays table, so existing installs upgrade
without manual migration.

### Integration into the inference loop (`web/app.py`)

- **Load:** at job start, `self.gallery = AppearanceGallery.load(rows for self.event_key)`.
  If `event_key` is absent, the gallery is disabled (in-memory empty, no persistence).
- **Deposit:** deposit a track's `appearance_embedding` under `(team, detected red/blue
  color)` whenever its identity is confirmed by an **externally-grounded label** — i.e.
  an OCR read crossing the lock threshold, or an OCR re-affirmation of an already-locked
  team on a re-read pass (subject to dedup) so coverage improves. A future manual
  team-set / calibration confirm endpoint would also be a valid deposit site (no such
  endpoint exists today). Do **not** deposit from:
  - a **gallery-driven** confirmation — appearance matches must not reinforce themselves;
  - the **process-of-elimination** path (`reid.eliminate_last_team`, ~`app.py:1249-1258`),
    which assigns a team by inference rather than by reading the bumper or a human label —
    seeding the gallery from a guess would corrupt it.
- **Query:** each read pass, for tracks that are **not** confirmed, query with the
  track's embedding + detected color:
  - **Strong match** — `best_sim ≥ GALLERY_CONFIRM_SIM` (≈0.88) **and**
    `best_sim − second_best_team_sim ≥ GALLERY_CONFIRM_GAP` (≈0.06): cast a
    high-weight vote for that team into the existing `team_ocr_votes` tally, routed
    through `ConsecutiveValueTracker`. Sustained strong matches lock the identity.
  - **Moderate match** — `≥ GALLERY_SUPPLEMENT_SIM` (≈0.80), or gap too small: low-weight
    supplemental vote only. Can break ties / seed an identity, cannot lock alone.
  - **Weak / no match** — below supplement threshold: no vote.
- **Safety:** never overrides an existing lock; existing high-confidence contradiction
  handler is reused unchanged.

### UI

Extend the `ocr_crops` WebSocket payload with `source: "ocr" | "gallery"` and
`gallery_sim`. The card renders an appearance-recovered identity distinctly, e.g.
`B3 ~ 1787 · appearance 86%` with an "appearance" badge, so a reviewer can tell a
recovered identity from a directly-read one.

## Tunables (environment variables, matching existing pipeline style)

| Var | Default | Meaning |
|-----|---------|---------|
| `GALLERY_ENABLED` | on | Master toggle |
| `GALLERY_CONFIRM_SIM` | 0.88 | Min similarity for a confirming vote |
| `GALLERY_CONFIRM_GAP` | 0.06 | Min gap to 2nd-best team for a confirming vote |
| `GALLERY_SUPPLEMENT_SIM` | 0.80 | Min similarity for a supplemental vote |
| `GALLERY_MAX_EXEMPLARS` | 12 | Max exemplars per (team, color) |
| `GALLERY_DEDUP_SIM` | 0.92 | New exemplar rejected if more similar than this to an existing one |
| `GALLERY_STRONG_VOTE_WEIGHT` | 1.0 | Vote weight for a strong match |
| `GALLERY_WEAK_VOTE_WEIGHT` | 0.4 | Vote weight for a supplemental match |

## Testing

- **`AppearanceGallery` units** (synthetic embeddings): dedup behavior, cap +
  lowest-legibility eviction, color filtering, ranking, gap logic.
- **DB round-trip** (real sqlite, matching existing test style): deposit → save →
  reload → query yields identical results.
- **Integration**: a track whose embedding matches one gallery team locks; a track
  roughly equidistant from two teams stays unconfirmed (gap guard).

## Edge cases

- **Cold start:** first match of an event has an empty gallery → query is a no-op;
  the gallery populates as confident reads land.
- **Unknown alliance color at query time:** search both color sets.
- **Look-alike robots / non-distinctive viewpoint:** gap check declines to confirm —
  fails safe rather than guessing.
- **Embedding consistency:** deposit and query both use the same `RobotEmbedder`
  instance, so embeddings are directly comparable.
- **Missing `event_key`:** gallery disabled; pipeline behaves exactly as today.

## Non-goals (YAGNI)

- No explicit pose/side classifier — the diverse exemplar set covers viewpoints
  implicitly.
- No cross-event gallery — fingerprints are scoped to a single event.
- No number-region-specific fingerprint — the whole-robot embedding is the signal.

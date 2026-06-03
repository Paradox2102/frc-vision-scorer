# View & Correct Gallery Identities

**Date:** 2026-06-03
**Status:** Approved design (pending spec review)
**Builds on:** `2026-06-03-appearance-gallery-design.md` (the per-event appearance gallery)

## Problem

The appearance gallery recovers obscured bumper numbers by matching a robot's
DINOv2 appearance to confidently-identified robots. In practice OCR produces many
**false identifications**, and once the gallery is live those wrong labels can be
deposited and propagated. There is currently **no way for a human to see the bumper
images a robot/exemplar was tagged with and correct the team number**. We need that —
both to fix a match's results and to curate the gallery so future matches improve.

Two prerequisites the codebase doesn't yet meet:

- **Bumper crop images are never persisted** after a match (they live only in the
  ephemeral `/tmp` diagnostic JSONL and the live WebSocket). Viewing images requires
  capturing representative crops per robot at match end.
- The **gallery stores only embeddings** — no images, no trust tier.

## Decisions (from brainstorming)

- A correction affects **both** the current match's identification/attribution **and**
  the gallery (the corrected crop becomes a human-verified exemplar).
- Trust model: **verified outweighs, auto still confirms** — auto-deposited exemplars
  keep working, but human-verified exemplars carry much higher vote weight and win ties.
- **Both surfaces**, built together in one effort: a post-match review/correct panel on
  the replay, and a gallery-manager page.
- Keep the **top ~3** highest-legibility crops per robot so a human can judge from the
  clearest view; a correction can seed several verified exemplars at once.

## Architecture overview

```
 During run ──► capture top-3 crops+embeddings per slot
      │
 Finalize  ──► match_robot_identity (1/slot) + match_robot_crop (≤3/slot)   [new tables]
      │
 Replay review UI ──► GET identities ──► user fixes team ──► POST correct
      │                                                          │
      │                              ┌───────────────────────────┤
      ▼                              ▼                           ▼
 update replay/match teams    mint VERIFIED gallery        (gallery manager page
                              exemplars (emb+image)         curates exemplars:
                                                            relabel / delete / verify)
```

Embeddings come from `RobotTrack.appearance_embedding`, now populated on the
field-tracking path (fixed separately). The gallery gains image + verified columns and
verified-weighted matching.

## Shared foundation

### F1. Persist representative crops per robot, per match

New tables in `web/database.py`:

```
match_robot_identity(
    job_id TEXT, slot INTEGER, slot_label TEXT, alliance TEXT,
    team_number TEXT, confirmed INTEGER, source TEXT, confidence REAL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (job_id, slot)
)
match_robot_crop(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT, slot INTEGER, legibility REAL,
    embedding BLOB, image BLOB /* ~128px JPEG thumbnail */
)
-- INDEX on match_robot_crop(job_id, slot)
```

Capture: during `read_fused_bumper_numbers`, maintain per slot the **top-3** crops by
legibility (`self._slot_crops[slot]` — a small sorted list of
`{legibility, crop_b64, embedding}`, where `embedding` is the slot's
`appearance_embedding` snapshot at that read). At finalize, downscale each kept crop to
a ~128px-tall JPEG thumbnail and write `match_robot_identity` (assigned team/alliance/
confidence/source) + its `match_robot_crop` rows. `source` is "ocr" or "gallery".

### F2. Gallery gains images + a verified tier

Add to `appearance_gallery`: `image BLOB` (crop thumbnail) and
`verified INTEGER DEFAULT 0`. Auto-deposits (`_deposit_to_gallery`) now also store the
crop thumbnail (the crop is in scope in `read_fused_bumper_numbers`) with `verified=0`.
Human corrections store `verified=1`. Added via `CREATE TABLE IF NOT EXISTS` for the new
columns through the lightweight `ALTER TABLE` try/except pattern already used for
replays, so existing installs upgrade cleanly.

### F3. Verified-outweighs matching

`AppearanceGallery.query(embedding, color)` returns
`list[(team, best_sim, best_is_verified)]`, sorted by `best_sim` desc with **verified
first on ties**. In `_apply_gallery_vote`, the vote weight becomes
`GALLERY_VERIFIED_VOTE_WEIGHT` (default 2.0) when the winning match is verified,
otherwise the existing strong/weak weights. `classify_match` is unchanged (it still
operates on `(team, sim)` projection); only the weight and tie-break change. New tunable
`GALLERY_VERIFIED_VOTE_WEIGHT`.

## Phase 1 — Post-match review & correction (on the replay)

### Backend (`web/app.py`, `web/database.py`)
- `GET /api/replays/{job_id}/identities` → per-slot records: `slot_label`, `alliance`,
  `team_number`, `confirmed`, `source`, `confidence`, and up to 3 `crops` as
  `data:image/jpeg;base64,...` URLs. **Never** returns raw embeddings.
- `POST /api/replays/{job_id}/correct` body `{slot, team_number}` →
  1. Update `match_robot_identity.team_number` for `(job_id, slot)`.
  2. Update the replay's stored team lists (`replays.blue_teams`/`red_teams`) so the
     replay shows corrected teams.
  3. Mint **verified** gallery exemplars: for each of that slot's `match_robot_crop`
     rows, insert an `appearance_gallery` row (embedding + image, `verified=1`) for
     `(event_key, team_number, alliance_color)`. Subject to the gallery's dedup/cap on
     load; here we insert directly to the DB.
  4. Best-effort: if the replay maps to a saved scored match, update the corresponding
     `match_robots.team_number` via the existing `update_robot_scores`.
  - If the run had no `event_key`, steps 1–2 still run; the gallery deposit (step 3) is
    skipped.

### Frontend (`web/static/index.html`, `app.js`, `style.css`)
A "Review identities" panel inside the existing replay player modal: 6 robot cards, each
showing the stored bumper thumbnails (≤3), slot label, an editable team (roster dropdown
+ free entry), and a confidence/source badge. "Save corrections" posts each changed slot
to the correct endpoint and shows a ✓ verified marker on success.

## Phase 2 — Gallery manager page

### Backend
- `GET /api/gallery?event_key=` → exemplars grouped by `(team, alliance_color)`, each
  with `id`, image data-URL, `verified`, `legibility`.
- `POST /api/gallery/{id}` body `{team?, alliance_color?, verified?}` → relabel/move or
  toggle verified.
- `DELETE /api/gallery/{id}` → remove an exemplar.

### Frontend
A new "Gallery" view in the SPA nav (alongside Scorer / Replays / History): pick an
event, see thumbnails grouped by team and alliance color, each with verified state;
click a thumbnail to relabel (move to another team/color), delete, or mark verified.
This is the surface for cleaning exemplars that false IDs polluted.

## Components & boundaries

- `web/field/appearance_gallery.py` — `query` returns verified flag; otherwise unchanged
  pure module.
- `web/database.py` — new tables + CRUD: `insert/get match_robot_identity`,
  `insert/list match_robot_crop`, gallery `image`/`verified` columns, gallery list/
  update/delete by id, `get_gallery_exemplars` returns `verified`.
- `web/app.py` — capture top-3 crops; finalize writes identity+crop rows; deposit stores
  image; verified weighting in `_apply_gallery_vote`; new endpoints (identities, correct,
  gallery CRUD).
- Frontend — replay review panel (Phase 1) and Gallery view (Phase 2), following the
  existing vanilla-JS SPA patterns.

## Testing

- **Pure units** (`AppearanceGallery`): `query` returns/sorts the verified flag, verified
  wins ties; `_apply_gallery_vote` uses the verified weight.
- **DB round-trip**: new tables; gallery `image`/`verified` columns; gallery
  list/update/delete by id; migration adds columns to a pre-existing gallery table.
- **Endpoint tests** (real sqlite, matching `test_replay_api`/`test_gallery_db`):
  identities returns per-slot crops as data-URLs and no embeddings; correct updates
  identity + replay teams + mints a verified exemplar (and skips deposit when no
  event_key); gallery CRUD relabel/delete/verify.
- **Thumbnail**: encode produces a small (<~15KB) JPEG of bounded height.

## Edge cases

- No `event_key`: correction fixes match/replay, skips gallery deposit.
- Slot with no captured crop: card shows "no image", team still editable; correction
  with no crops updates labels but mints no exemplar.
- Correcting to a team already held by another slot: release it from the other slot
  (mirror the OCR lock path's `_clear_team_from_other_robots` intent at the data layer).
- Deleting the last exemplar of a team: allowed; that team simply has no gallery entry.
- Re-correcting: idempotent — re-running correct updates the label and adds/dedups
  verified exemplars.

## Non-goals (YAGNI)

- No per-crop manual selection of which crops to deposit (correction deposits all of the
  slot's captured crops for the chosen team; dedup/cap bound the set).
- No cross-event gallery management; the manager is per-event.
- No editing of historical scoring math — correction changes team attribution and the
  gallery, not alliance score totals.

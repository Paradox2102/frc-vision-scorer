# Track-Identity Persistence & Confidence-Driven VLM — Design

**Date:** 2026-06-06
**Status:** Draft for review

## Summary

Keep each track bound to the same physical robot for the whole match so the VLM
becomes a *verification* step, not a continuous *identification* step. Three changes,
phased:

1. **Persistent identity** decoupled from the 6 ephemeral tracker slots, so long
   occlusion no longer resets a robot's team/votes.
2. **Identity confidence + uncertainty-triggered VLM** — a per-robot confidence score
   drives VLM cadence: read to seed, then read *only* when confidence drops, plus a
   mandatory sparse audit floor.
3. **Joint same-alliance occlusion tiebreak** — detect the genuinely ambiguous
   re-emergence (two+ same-color robots occluded together) and resolve it with a
   targeted VLM read instead of guessing.

This builds on machinery that already exists (fused-track re-link by appearance, sticky
slot binding, roster-exclusive team assignment, gallery re-ID, the periodic re-read
audit). It does not replace the tracker; it adds an identity layer above it and changes
*when* the VLM fires.

## Background: how identity flows today (and where it breaks)

Chain: SAM2 `obj_id` (per-camera, resets each re-prompt) → **fused_id**
(`fused_track_id.py`, cross-camera, re-links to inactive tracks by appearance ≥
`APP_FLOOR=0.55`) → one of **6 fixed slots** (`tracker.py`, sticky-bound via `_id_to_slot`,
roster-exclusive) → **RobotTrack** (`app.py`, holds votes / EMA embedding / confirmed
team). VLM cadence already ramps to an 8 s audit once confirmed, with a forced re-read
every `REREAD_EVERY_N_PASSES` to break wrong locks.

Two gaps:
- **Slot expiry resets identity.** After `max_occ=90` occluded frames a slot expires
  (`tracker.py:138`); the re-emerged robot seeds a fresh slot at **zero votes** — team
  and lock are orphaned. Identity is coupled to the transient slot, not the robot.
- **Joint same-alliance occlusion is unguarded.** A swap needs appearance, color, and
  Kalman position to all fail together — which is exactly what two red robots occluded
  together produce (appearance similar, color identical, positions ambiguous). The only
  current corrective is a *later* lucky audit read.

## Invariant (non-negotiable, from prior incident)

**A maximally-confident track is still audited periodically.** Per the
`reread-audit-deadlock` lesson, a wrong-but-confident lock must always remain
challengeable. Confidence may *lengthen* the audit interval; it must **never** drive it
to "never." There is a hard floor (e.g. one audit read per confirmed robot every
`AUDIT_FLOOR_SECONDS`, default ~30 s) independent of confidence.

## Phase 1 — Persistent identity (decouple from slots)

**Goal:** A robot's team, locked state, anchor appearance, and last trajectory survive
slot expiry and re-attach on re-emergence, instead of resetting.

- Introduce a **`RobotIdentity`** that owns the match-durable state currently scattered
  on the slot-keyed `RobotTrack`: `team_number`, `team_number_confirmed`, vote tallies,
  anchor embedding(s), `alliance`, last Kalman state + timestamp, and an identity id.
  Identities are created on first appearance and **persist for the match** (active or
  inactive); they are NOT destroyed when a slot expires.
- The 6 slots become a *current-visibility binding*: each active slot points at exactly
  one `RobotIdentity`. Slot expiry detaches the binding but leaves the identity intact
  (now "inactive / occluded").
- **Re-attach on re-emergence:** when a fused detection cannot bind to an active slot,
  match it against **inactive identities** using a combined score: appearance (anchor +
  gallery), trajectory plausibility (could the robot have reached here given elapsed
  time?), and **roster exclusivity** (an inactive identity's team is still reserved).
  On a confident match, re-bind the identity (votes/lock restored); otherwise create a
  new identity.
- **Roster cap stays hard:** at most 3 active+inactive identities per alliance. A
  would-be 4th same-alliance identity is a signal of a tracking error, not a new robot —
  prefer re-attaching to an existing inactive identity.

This subsumes much of `FusedTrackRegistry`'s inactive-track re-link (which already does
appearance re-link at the fused layer); Phase 1 lifts that persistence up to the
team/vote level so re-link also restores identity, not just a track id.

## Phase 2 — Identity confidence + uncertainty-triggered VLM

**Goal:** Replace mostly-time-based VLM scheduling with a per-identity **confidence**
that fires the VLM where tracking is weak and rests it where tracking is strong.

- **`identity_confidence` ∈ [0,1]** per identity, updated each frame from:
  - **Continuity:** frames of uninterrupted tracking since the last confident
    observation (rises slowly), reset/decayed on occlusion or missed frames.
  - **Appearance agreement:** cosine of the current crop to the identity's anchor /
    gallery exemplars (high agreement raises; divergence lowers).
  - **Assignment margin:** how decisively this frame's assignment beat the runner-up
    (Kalman cost gap + appearance gap). A close call lowers confidence.
  - **Roster exclusivity:** team uniquely accounted for among active identities raises it.
  - **VLM agreement:** a recent VLM read matching the locked team raises it; a
    contradiction drops it sharply.
- **VLM cadence as a function of confidence:**
  - *Unconfirmed / low confidence* → read aggressively to seed (current fast/med/slow
    intervals based on count).
  - *Confirmed + high confidence* → cadence stretches toward `AUDIT_FLOOR_SECONDS`
    (the invariant floor — never longer).
  - *Confidence drop event* (post-occlusion re-emergence, low assignment margin, swap
    suspicion, appearance divergence) → **trigger an immediate targeted read** of just
    that identity, ahead of schedule.
- Net effect: far fewer reads on calm, well-separated play; reads concentrated at
  re-emergence and ambiguity — exactly where identity is at risk.

## Phase 3 — Joint same-alliance occlusion tiebreak

**Goal:** Correctly re-assign 2+ same-alliance robots that were occluded together.

- **Detect the co-occlusion event:** when ≥2 identities of the same alliance are
  simultaneously occluded (from the existing per-frame occlusion state) and then
  re-emerge within a short window.
- **Attempt tracking resolution first** (Phase 1/2 machinery): trajectory + appearance
  assign the re-emerged detections to the occluded identities.
- **Ambiguity test:** if the best assignment's margin over the swapped alternative is
  below `IDENTITY_AMBIGUITY_MARGIN` (appearance + trajectory combined), the re-emergence
  is genuinely ambiguous (the hard case) → **escalate to a targeted VLM read on just the
  ambiguous identities** and assign from the result. If still unreadable (bumpers
  occluded), hold both identities in a low-confidence "ambiguous" state (do not
  hard-commit a guess) until a later read resolves them; scoring during this window uses
  alliance (still certain) but defers per-robot attribution if needed.
- This is the targeted-tiebreak realization of Phase 2's uncertainty trigger, specialized
  to the case the user called out.

## Components / files (high level — detailed in per-phase plans)

- `web/field/tracker.py` — slot↔identity binding; stop destroying identity on slot expiry.
- `web/field/fused_track_id.py` — feed re-link results into identity re-attach (Phase 1).
- New `web/field/identity.py` (proposed) — `RobotIdentity` + the persistent registry and
  the combined re-attach score (appearance + trajectory + roster). Pure-testable.
- New `web/field/identity_confidence.py` (proposed) — confidence update (pure function of
  per-frame signals), unit-testable in isolation.
- `web/app.py` — move durable state from slot-keyed `RobotTrack` onto `RobotIdentity`;
  rewrite VLM scheduling to consult confidence + the audit floor; wire the co-occlusion
  tiebreak; preserve the existing overturn/contradiction unlock paths.
- `web/field/appearance_gallery.py` — reused as the cross-occlusion appearance anchor
  (no change beyond the diversity-cap work already in flight).

## Risks & mitigations

- **Wrong-but-confident lock deadlock** → the audit-floor invariant (above) + keep the
  existing overturn-on-contradiction path.
- **Over-trusting appearance for same-alliance robots** → appearance alone never commits
  an ambiguous same-alliance re-assignment; trajectory + (if ambiguous) VLM decide.
- **Persistent identity accumulating error** → an identity whose appearance has drifted
  far from its anchor loses confidence and gets re-audited; a confident VLM contradiction
  can split/reset an identity.
- **Scope/regression** → phased; each phase is independently shippable and testable, and
  each preserves current behavior when its new signals are inconclusive (fall back to
  today's logic).

## Phasing / build order

1. **Phase 1** (persistent identity + re-attach) — foundation; removes the long-occlusion
   reset. Shippable alone (already a correctness win).
2. **Phase 2** (confidence + uncertainty-triggered VLM) — depends on Phase 1's identity
   object to hang confidence + durable cadence state on.
3. **Phase 3** (joint-occlusion tiebreak) — specializes Phase 2's trigger; depends on both.

Each phase gets its own implementation plan (writing-plans) after this spec is approved.

## Dependencies / interactions

- **Bumper model (in flight):** better class-derived color improves alliance certainty
  (a confidence input) but does **not** help separate same-alliance robots — the Phase 3
  tiebreak is still needed. No hard dependency; either can land first.
- **Appearance gallery diversity cap (done, on branch):** the diverse 6-exemplar set is
  the appearance anchor Phase 1/3 lean on — complementary.

## Testing (per-phase, `.venv/bin/python -m pytest`)

- **Phase 1:** identity survives a simulated >90-frame occlusion (slot expires, identity
  persists, re-attach restores team/votes); roster cap blocks a phantom 4th same-alliance
  identity; re-attach prefers an inactive identity over a new one when appearance+roster
  agree. Pure tests on `identity.py` with synthetic detections.
- **Phase 2:** confidence rises under clean tracking, drops on occlusion/low-margin;
  cadence stretches with confidence but never exceeds `AUDIT_FLOOR_SECONDS` (the
  invariant — explicit test); a confidence-drop event forces an immediate read. Pure
  tests on `identity_confidence.py`.
- **Phase 3:** a synthetic co-occlusion of two same-alliance identities with (a)
  separable appearance → resolved by tracking, no VLM; (b) inseparable appearance →
  flagged ambiguous, VLM tiebreak invoked, correct assignment; (c) inseparable +
  unreadable → held ambiguous, no hard mis-commit.
- **Integration:** a real match with a known occlusion event — confirm the same physical
  robot keeps its slot/team across it, and VLM read count drops materially vs today.

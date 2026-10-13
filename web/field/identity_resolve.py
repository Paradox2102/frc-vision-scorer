"""Exact global slot->team assignment solver for the 6-robot field.

The pipeline tracks 6 slots (0-2 blue alliance, 3-5 red alliance) and, over
the course of a match, accumulates per-slot evidence (VLM bumper reads,
appearance-gallery matches) for which team number occupies that slot. A
greedy/Hungarian pass over the raw vote counters is what runs today, but
greedy per-slot argmax is provably wrong whenever two slots both prefer the
same team: it locks the globally-best team onto whichever slot it visits
first and starves the other slot, even when swapping the assignment would
raise the *total* score (see ``test_global_beats_greedy_lock``).

This module replaces that step with an exact search. Per alliance there are
at most 3 roster teams and 3 slots, so the space of partial injective
assignments (a slot may be left unassigned; a team may be assigned to at
most one slot) is tiny: for roster size r <= 3,

    sum_{k=0..min(3,r)} C(3, k) * P(r, k)

which is exactly 34 when r == 3 (1 + 9 + 18 + 6). That is cheap enough to
enumerate exhaustively via ``itertools.combinations`` (which 3 slots get a
team) x ``itertools.permutations`` (which teams, in which slot order) and
take the true maximum, rather than reach for a general-purpose optimizer for
a search space this size. The two alliances are independent (no roster
overlap, no cross-alliance interactions in the objective), so the max total
score decomposes into the sum of two independent per-alliance maxima and can
be solved separately.

v2 design note: if this ever generalizes to whole-match *segment* assignment
(each track occupies several time segments, each of which could plausibly
carry a different team, and evidence must be pooled/split across segments
rather than collapsed into one static per-slot evidence blob), the candidate
space stops being small enough to enumerate and this should become an ILP
solved with ``scipy.optimize.milp`` (binary assignment variables x[segment,
team] with one-team-per-segment and one-segment-window-per-team-at-a-time
constraints). That is out of scope here; this module only handles the
static, whole-match, per-slot case. The time-varying case that did land
(``identity_swap_resolve``, web/field/identity_swap.py) avoids the ILP: per
alliance only the 3! slot->team permutations are states, a Viterbi over
time finds where slots swapped robots, and THIS module's
``resolve_slot_labels`` then labels the resulting virtual tracks.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

__all__ = [
    "SlotEvidence",
    "combine_scores",
    "assign_teams_global",
    "merge_global_assignment",
    "attribute_reads",
    "build_slot_evidence",
    "vlm_read_rows_from_jsonl",
    "resolve_slot_labels",
]

# Absolute tolerance for score comparisons: vote weights are sums of products
# of small decimals (0.5, 0.55, 1.25 ...), so "exactly min_score" and "exact
# tie" must survive float noise.
SCORE_EPS = 1e-9

# Fixed slot-id -> alliance convention used throughout the pipeline.
BLUE_SLOTS = (0, 1, 2)
RED_SLOTS = (3, 4, 5)
ALL_SLOTS = BLUE_SLOTS + RED_SLOTS


@dataclass
class SlotEvidence:
    """Accumulated per-slot evidence for one tracked robot slot.

    ``read_scores`` and ``gallery_scores`` map team number (string) to an
    accumulated score from that evidence source; presence in either dict is
    enough to be considered a candidate for the slot.
    """

    slot_id: int
    alliance: str
    read_scores: dict[str, float] = field(default_factory=dict)
    gallery_scores: dict[str, float] = field(default_factory=dict)
    frames_seen: int = 0


def combine_scores(
    ev: SlotEvidence, w_reads: float = 1.0, w_gallery: float = 0.4
) -> dict[str, float]:
    """Linear blend of read and gallery scores for every team either mentions."""
    teams = set(ev.read_scores) | set(ev.gallery_scores)
    return {
        team: w_reads * ev.read_scores.get(team, 0.0)
        + w_gallery * ev.gallery_scores.get(team, 0.0)
        for team in teams
    }


def _expected_alliance(slot_id: int) -> str | None:
    if slot_id in BLUE_SLOTS:
        return "blue"
    if slot_id in RED_SLOTS:
        return "red"
    return None


def _validate_evidence(evidence: dict[int, SlotEvidence]) -> None:
    for slot_id, ev in evidence.items():
        expected = _expected_alliance(slot_id)
        if expected is None:
            raise ValueError(
                f"unknown slot_id {slot_id!r}: not in the fixed 0-5 slot universe"
            )
        if ev.alliance != expected:
            raise ValueError(
                f"slot {slot_id} evidence.alliance={ev.alliance!r} does not match "
                f"slot-id convention (expected {expected!r})"
            )


def _slot_team_scores(
    slots: tuple[int, ...],
    roster: list[str],
    evidence: dict[int, SlotEvidence],
    w_reads: float,
    w_gallery: float,
) -> dict[int, dict[str, float]]:
    """combined score for every (slot, roster-team) pair; missing evidence -> 0.0.

    Cross-alliance evidence is ignored by construction: only ``roster``
    teams are ever looked up, so evidence for teams outside this alliance's
    roster is never consulted.
    """
    out: dict[int, dict[str, float]] = {}
    for slot_id in slots:
        ev = evidence.get(slot_id)
        combined = combine_scores(ev, w_reads, w_gallery) if ev is not None else {}
        out[slot_id] = {team: combined.get(team, 0.0) for team in roster}
    return out


def _best_alliance_assignment(
    slots: tuple[int, ...],
    roster: list[str],
    slot_scores: dict[int, dict[str, float]],
    min_score: float,
) -> tuple[dict[int, str], set[int]]:
    """Exhaustively search partial assignments of ``roster`` to ``slots``.

    Returns ``(assignment, contested)``: ``assignment`` holds only the
    assigned slot_id -> team pairs (unassigned slots are absent). Enumerates
    every subset of slots of size 0..min(3, len(roster)) paired with every
    permutation of roster teams of that size; a candidate is valid only if
    every pair in it clears ``min_score`` (a sub-threshold pair may not be
    assigned at all, per spec — its "cleaned" version is already covered by
    a smaller candidate enumerated separately, so simply discarding invalid
    candidates is exact, not an approximation).

    Optimum: maximize total score, then the number of assigned slots. Ties
    are handled conservatively instead of arbitrarily: when several
    candidates share that optimum (totals within SCORE_EPS, same count),
    only the pairs ALL of them agree on are returned, and every slot they
    disagree on is reported in ``contested`` and left unassigned. A
    per-slot tie that the rest of the alliance breaks (qm11 R3: 4256/6800
    tied at 3.5 but R1 can only take 6800) is not a tie of the joint
    optimum, so it is resolved, not contested.
    """
    max_k = min(len(slots), len(roster))
    best_total: float | None = None
    best_count = -1
    optima: list[dict[int, str]] = []

    for k in range(0, max_k + 1):
        for slot_subset in itertools.combinations(slots, k):
            for team_perm in itertools.permutations(roster, k):
                candidate = dict(zip(slot_subset, team_perm))
                pair_scores = [slot_scores[slot_id][team]
                               for slot_id, team in candidate.items()]
                if any(score < min_score - SCORE_EPS for score in pair_scores):
                    continue  # a sub-threshold pair may not be assigned
                total_score = sum(pair_scores)
                assigned_count = len(candidate)
                if (best_total is None
                        or total_score > best_total + SCORE_EPS
                        or (abs(total_score - best_total) <= SCORE_EPS
                            and assigned_count > best_count)):
                    best_total, best_count = total_score, assigned_count
                    optima = [candidate]
                elif (abs(total_score - best_total) <= SCORE_EPS
                        and assigned_count == best_count):
                    optima.append(candidate)

    if not optima:
        return {}, set()
    agreed = {slot_id: team for slot_id, team in optima[0].items()
              if all(o.get(slot_id) == team for o in optima[1:])}
    touched = {slot_id for o in optima for slot_id in o}
    return agreed, touched - set(agreed)


def assign_teams_global(
    evidence: dict[int, SlotEvidence],
    blue_roster: list[str],
    red_roster: list[str],
    w_reads: float = 1.0,
    w_gallery: float = 0.4,
    min_score: float = 0.5,
    contested_out: set | None = None,
) -> dict[int, str | None]:
    """Exact max-total-score slot->team assignment, solved per alliance.

    Returns all six slot ids (0-5); a slot with no qualifying assignment
    (no evidence, every candidate pair below ``min_score``, or contested by
    a tie of the optimum) maps to ``None``.

    ``min_score`` is inclusive: a (slot, team) pair is eligible iff its
    combined score >= min_score (within SCORE_EPS). With the default 0.5 a
    single half-weight read (vote_weight 0.5: confidence < 0.85) is enough
    to make a team eligible for a slot; it is then assigned only if the
    whole-alliance optimum puts it there. ``contested_out``, when given, is
    filled with the slots left unassigned by a tie (see
    ``_best_alliance_assignment``). Confirmation margins are not decided
    here — see ``merge_global_assignment``'s ``confirm_margin``.
    """
    _validate_evidence(evidence)

    result: dict[int, str | None] = {slot_id: None for slot_id in ALL_SLOTS}

    for slots, roster in ((BLUE_SLOTS, blue_roster), (RED_SLOTS, red_roster)):
        if not roster:
            continue
        slot_scores = _slot_team_scores(slots, roster, evidence, w_reads, w_gallery)
        assignment, contested = _best_alliance_assignment(
            slots, roster, slot_scores, min_score)
        for slot_id, team in assignment.items():
            result[slot_id] = team
        if contested_out is not None:
            contested_out.update(contested)

    return result


def merge_global_assignment(
    slot_meta: dict,
    assignment: dict[int, str | None],
    scores: dict[int, dict[str, float]],
    confirm_margin: float = 2.0,
    strip_slots: set | None = None,
) -> tuple[dict, list[dict]]:
    """Merge a solved assignment into existing per-slot metadata.

    Never mutates ``slot_meta``; returns a new dict, copying only the slot
    entries that actually change. Rules:

    - assigned team == existing team_number -> entry left untouched
      (including its existing ``confirmed`` flag).
    - assigned team differs and is not ``None`` -> ``team_number`` is
      overridden; ``confirmed`` is set True iff (winner score - best
      other-team score for that slot) >= ``confirm_margin``, else False.
    - assignment is ``None`` (or the slot never entered the solve) -> the
      existing entry is left untouched UNLESS its team was assigned to a
      different slot by this pass, in which case the stale label is stripped
      (team ``None``, unconfirmed) and recorded as a change. Keeping it would
      duplicate the team on the overlay (qm11 2026-07-20: R2 assigned 1477
      while R3 kept its greedy OCR 1477), voiding the assignment's
      collision-free guarantee.
    - ``strip_slots`` (the solver's contested slots): an existing label on
      a slot the solve left ``None`` because of a tie is stripped (team
      ``None``, unconfirmed) — a tie leaves the slot unassigned rather than
      keeping whichever label the last identity happened to carry. Recorded
      as a change with ``"reason": "tie"``.

    The confirm margin is the winner's combined score minus the best OTHER
    team's score on the same slot (including teams the solve gave to other
    slots), so an assignment the alliance optimum forced onto a slot whose
    own evidence prefers another team gets a negative margin and stays
    unconfirmed (qm11 R1: 6800 at 0.5 vs 1477 at 1.8).

    Returns ``(new_slot_meta, changes)`` where ``changes`` lists only the
    actual overrides, each as
    ``{"slot_id", "old_team", "new_team", "score", "margin"}``.
    """
    new_meta = dict(slot_meta)
    changes: list[dict] = []

    for slot_id, new_team in assignment.items():
        if new_team is None:
            continue  # leave existing entry untouched

        existing = slot_meta.get(slot_id, {})
        old_team = existing.get("team_number")
        if new_team == old_team:
            continue  # already correct -> untouched, including confirmed

        slot_scores = scores.get(slot_id, {})
        winner_score = slot_scores.get(new_team, 0.0)
        other_scores = [s for t, s in slot_scores.items() if t != new_team]
        best_other = max(other_scores) if other_scores else 0.0
        margin = winner_score - best_other
        confirmed = margin >= confirm_margin

        updated = dict(existing)
        updated["team_number"] = new_team
        updated["confirmed"] = confirmed
        new_meta[slot_id] = updated

        changes.append(
            {
                "slot_id": slot_id,
                "old_team": old_team,
                "new_team": new_team,
                "score": winner_score,
                "margin": margin,
            }
        )

    # Contested (tied) slots: strip whatever label the slot still carries.
    for slot_id in sorted(strip_slots or ()):
        if assignment.get(slot_id) is not None:
            continue
        current = new_meta.get(slot_id)
        if not current or current.get("team_number") is None:
            continue
        updated = dict(current)
        updated["team_number"] = None
        updated["confirmed"] = False
        new_meta[slot_id] = updated
        changes.append(
            {
                "slot_id": slot_id,
                "old_team": current.get("team_number"),
                "new_team": None,
                "score": 0.0,
                "margin": 0.0,
                "reason": "tie",
            }
        )

    # Collision sweep: a team this pass assigned to one slot must not survive
    # as a kept label on another (None-assigned or absent slots skipped the
    # loop above and were never part of the solve's constraint set).
    assigned = {t for t in assignment.values() if t is not None}
    for slot_id, existing in slot_meta.items():
        team = (existing or {}).get("team_number")
        if team is None or team not in assigned or assignment.get(slot_id) == team:
            continue
        current = new_meta.get(slot_id, existing)
        if current.get("team_number") != team:
            continue  # already overridden to a non-colliding team above
        updated = dict(current)
        updated["team_number"] = None
        updated["confirmed"] = False
        new_meta[slot_id] = updated
        changes.append(
            {
                "slot_id": slot_id,
                "old_team": team,
                "new_team": None,
                "score": 0.0,
                "margin": 0.0,
            }
        )

    return new_meta, changes


def attribute_reads(
    vlm_read_rows: list[dict] | None, identity_log: list[dict]
) -> tuple[list[tuple[int, int, str, float]], int]:
    """Credit each team-naming read to the slot its identity held at the
    read's OWN frame (the misattribution guard described in
    :func:`build_slot_evidence`, which is built on this helper).

    Returns ``(attributed, dropped)``: ``attributed`` lists
    ``(frame, slot_id, matched_team, vote_weight)`` in input order (rows
    without a ``matched_team`` are skipped silently); ``dropped`` counts
    team-naming rows whose ``(frame, robot_id)`` has no binding.
    """
    frame_identity_to_slot: dict[tuple[int, int], int] = {
        (row["frame_num"], row["identity_id"]): row["slot_id"] for row in identity_log
    }
    attributed: list[tuple[int, int, str, float]] = []
    dropped = 0
    for row in vlm_read_rows or ():
        matched_team = row.get("matched_team")
        if matched_team is None:
            continue  # no team vote carried by this row
        slot_id = frame_identity_to_slot.get((row["frame"], row["robot_id"]))
        if slot_id is None:
            dropped += 1
            continue
        attributed.append(
            (row["frame"], slot_id, matched_team, row.get("vote_weight", 0.0)))
    return attributed, dropped


def build_slot_evidence(
    vlm_read_rows: list[dict] | None,
    identity_log: list[dict],
    team_ocr_votes_by_slot: dict[int, dict[str, float]],
    gallery_scores_by_slot: dict[int, dict[str, float]],
    slot_alliance,
) -> tuple[dict[int, SlotEvidence], dict]:
    """Aggregate raw per-frame VLM reads (or a fallback vote source) into
    per-slot :class:`SlotEvidence`, plus gallery scores.

    Pure function: no file I/O, no mutation of any input. The app-side
    caller (Task 3) is responsible for loading ``vlm_read_rows`` from the
    diagnostics JSONL and for computing ``gallery_scores_by_slot``.

    Read path (used when ``vlm_read_rows`` is not ``None`` and at least one
    row is successfully attributed to a slot): each row's ``robot_id`` (an
    identity id) is bound to the slot it occupied *at that exact frame* via
    ``identity_log``. This exact-frame binding is a deliberate misattribution
    guard -- crediting a read to whichever slot an identity happens to
    occupy *now* (rather than at the read's own frame) would silently move
    votes across a re-seed or identity swap. A read whose (frame, robot_id)
    has no binding is dropped and counted in
    ``stats["reads_dropped_misattributed"]``; its ``vote_weight`` (already
    fully baked with confidence/surface/prob multipliers upstream) is
    accumulated as-is into ``read_scores[matched_team]`` on a hit.

    Fallback path (``vlm_read_rows`` is ``None``, empty, or every row failed
    to attribute -- e.g. all misattributed or lacking a ``matched_team``):
    ``read_scores`` is populated directly from ``team_ocr_votes_by_slot``
    instead, and ``stats["fallback_used"]`` is ``True``.

    Gallery scores are always attached from ``gallery_scores_by_slot`` for
    every slot that ends up in the returned evidence dict, independent of
    which read path ran. The slot universe is the union of: slots with a
    successful read attribution (read path only), slots in
    ``team_ocr_votes_by_slot`` (fallback path only), and slots in
    ``gallery_scores_by_slot`` (always) -- so a slot with gallery evidence
    alone is still a valid assignment candidate.

    Returns ``(evidence, stats)`` where
    ``stats = {"reads_used": int, "reads_dropped_misattributed": int,
    "fallback_used": bool}``.
    """
    read_scores_by_slot: dict[int, dict[str, float]] = {}
    reads_used = 0
    reads_dropped_misattributed = 0

    if vlm_read_rows:
        attributed, reads_dropped_misattributed = attribute_reads(
            vlm_read_rows, identity_log)
        for _frame, slot_id, matched_team, weight in attributed:
            slot_scores = read_scores_by_slot.setdefault(slot_id, {})
            slot_scores[matched_team] = slot_scores.get(matched_team, 0.0) + weight
            reads_used += 1

    fallback_used = vlm_read_rows is None or len(vlm_read_rows) == 0 or reads_used == 0
    if fallback_used:
        read_scores_by_slot = {
            slot_id: dict(votes) for slot_id, votes in team_ocr_votes_by_slot.items()
        }
        reads_used = 0

    frames_seen_by_slot: dict[int, int] = {}
    for row in identity_log:
        slot_id = row["slot_id"]
        frames_seen_by_slot[slot_id] = frames_seen_by_slot.get(slot_id, 0) + 1

    slot_ids = set(read_scores_by_slot) | set(gallery_scores_by_slot)

    evidence: dict[int, SlotEvidence] = {
        slot_id: SlotEvidence(
            slot_id=slot_id,
            alliance=slot_alliance(slot_id),
            read_scores=dict(read_scores_by_slot.get(slot_id, {})),
            gallery_scores=dict(gallery_scores_by_slot.get(slot_id, {})),
            frames_seen=frames_seen_by_slot.get(slot_id, 0),
        )
        for slot_id in slot_ids
    }

    stats = {
        "reads_used": reads_used,
        "reads_dropped_misattributed": reads_dropped_misattributed,
        "fallback_used": fallback_used,
    }
    return evidence, stats


def vlm_read_rows_from_jsonl(lines) -> list[dict]:
    """``vlm_read`` rows from diagnostic JSONL lines (an iterable of str).

    Skips unparseable lines and rows missing ``frame``/``robot_id`` (the
    evidence builder direct-indexes those); drops the bulky ``crop_b64``.
    Shared by the live finalize path and scripts/identity_offline_check.py.
    """
    rows: list[dict] = []
    for line in lines:
        try:
            r = json.loads(line)
        except (ValueError, TypeError):
            continue
        if (isinstance(r, dict) and r.get("type") == "vlm_read"
                and "frame" in r and "robot_id" in r):
            r.pop("crop_b64", None)
            rows.append(r)
    return rows


def resolve_slot_labels(
    vlm_read_rows: list[dict] | None,
    identity_log: list[dict],
    votes_by_slot: dict[int, dict[str, float]],
    gallery_by_slot: dict[int, dict[str, float]],
    slot_alliance,
    blue_roster: list[str],
    red_roster: list[str],
    slot_meta: dict,
    min_score: float = 0.5,
    confirm_margin: float = 2.0,
) -> dict:
    """Whole-match per-slot final labels: evidence -> exact assignment ->
    merge into ``slot_meta``. The single code path behind the
    ``global_identity_resolve`` finalize step and the offline checker.

    Reads are credited to the slot their identity held at the read's own
    frame (``build_slot_evidence``), so every identity that ever sat on a
    slot contributes — the slot's final identity may have no votes at all
    (qm11 80a1cd4c: 5 of 6 final identities were born in the last ~10 s).

    Returns ``{"slot_meta", "changes", "assignment", "contested", "scores",
    "evidence", "stats"}``; ``slot_meta`` is a new dict (input unmutated).
    """
    evidence, stats = build_slot_evidence(
        vlm_read_rows, identity_log, votes_by_slot, gallery_by_slot, slot_alliance)
    contested: set[int] = set()
    assignment = assign_teams_global(
        evidence, list(blue_roster), list(red_roster), min_score=min_score,
        contested_out=contested)
    scores = {sid: combine_scores(ev) for sid, ev in evidence.items()}
    norm_meta = {sid: {**m, "team_number": (str(m["team_number"])
                 if m.get("team_number") is not None else None)}
                 for sid, m in slot_meta.items()}
    new_meta, changes = merge_global_assignment(
        norm_meta, assignment, scores, confirm_margin=confirm_margin,
        strip_slots=contested)
    stats = {**stats, "contested": sorted(contested)}
    return {
        "slot_meta": new_meta,
        "changes": changes,
        "assignment": assignment,
        "contested": contested,
        "scores": scores,
        "evidence": evidence,
        "stats": stats,
    }

"""Final labels and tag strings for the post-finalize HUD render
(docs/superpowers/specs/2026-11-10-hud-post-render-design.md section 2).

Pure: no I/O. A slot's label at replay frame i is
``meta.frames[i].robots[slot].team_number`` (AMBIGUOUS / empty = unlabelled),
then the slot's recorded correction (``replay_slot_corrections``) applies:
O = old of the slot's first correction, N = new of its latest; N == O is no
correction; O a team -> frames labelled O become N; O None -> unlabelled frames
become N; frames naming any other team are left alone; replaced frames render
confirmed.
"""
from __future__ import annotations

from typing import Iterable, Optional


def norm_team(value) -> Optional[str]:
    """Team as a string; None for None, empty and "AMBIGUOUS"."""
    if value is None:
        return None
    s = str(value).strip()
    return None if s in ("", "AMBIGUOUS") else s


def effective_corrections(rows: Iterable[dict]) -> dict:
    """{slot (int): (O, N)} from correction rows ({slot, old_team, new_team})
    given oldest first. A slot whose N equals O has no entry."""
    by_slot: dict = {}
    for r in rows:
        by_slot.setdefault(int(r["slot"]), []).append(r)
    out = {}
    for slot, rs in by_slot.items():
        o, n = norm_team(rs[0].get("old_team")), norm_team(rs[-1].get("new_team"))
        if n != o:
            out[slot] = (o, n)
    return out


def final_label(team, confirmed, correction) -> tuple:
    """(team, confirmed, relabelled) for one slot at one frame."""
    team = norm_team(team)
    if correction is not None:
        o, n = correction
        if team == o:                      # O a team: equal frames; O None: unlabelled frames
            return n, True, True
    return team, bool(confirmed) and team is not None, False


def final_tag(team: Optional[str], confirmed: bool, score: Optional[int]) -> str:
    """The robot tag without the slot prefix (draw_robot adds it):
    ``"<team> S<n>"`` confirmed, ``"<team>? S<n>"`` named, ``"no # S<n>"``
    unlabelled; the `` S<n>`` suffix is omitted when the score is unknown."""
    suffix = f" S{int(score)}" if score is not None else ""
    if team is None:
        return f"no #{suffix}"
    return f"{team}{suffix}" if confirmed else f"{team}?{suffix}"


def radar_label(team: Optional[str], slot: str) -> str:
    """Field-map label: the team when labelled, else the slot (no "?")."""
    return team if team is not None else slot


def entry_text(entry: dict, rows_by_slot: dict, corr_by_label: dict) -> tuple:
    """(kind, text) for one recorded robot (a draw-list ``main`` or ``corners``
    entry) at one frame:

    - ``"live"``: a corner entry with no slot; text = its recorded live string;
    - ``"phantom"``: the slot has no meta row at this frame (finalize rejected
      it); text = the slot;
    - ``"ghost"``: text = ``"<slot> <tag>"`` (draw_ghost appends " coast");
    - ``"robot"``: text = the tag (draw_robot prefixes the slot).
    """
    slot = entry.get("slot")
    if slot is None:
        return "live", entry.get("live_text") or ""
    row = rows_by_slot.get(slot)
    if row is None:
        return "phantom", slot
    team, confirmed, _ = final_label(row.get("team_number"), row.get("confirmed"),
                                     corr_by_label.get(slot))
    tag = final_tag(team, confirmed, entry.get("score"))
    if entry.get("ghost"):
        return "ghost", f"{slot} {tag}"
    return "robot", tag


def radar_text(entry: dict, rows_by_slot: dict, corr_by_label: dict) -> tuple:
    """(has_row, label) for a recorded field-map dot: the final team, else
    the slot; a slot with no meta row is labelled with the slot."""
    slot = entry.get("slot")
    row = rows_by_slot.get(slot)
    if row is None:
        return False, slot
    team, _c, _r = final_label(row.get("team_number"), row.get("confirmed"),
                               corr_by_label.get(slot))
    return True, radar_label(team, slot)


def relabel_counts(frames: list, corrections: dict, slot_label) -> dict:
    """{slot label: frames relabelled} for every corrected slot, over all of
    ``meta.frames`` (``slot_label`` maps a slot id to "B1".."R3")."""
    out = {}
    for slot, corr in corrections.items():
        lab = slot_label(slot)
        n = 0
        for f in frames:
            for r in f.get("robots") or []:
                if r.get("slot") == lab and final_label(r.get("team_number"),
                                                        r.get("confirmed"), corr)[2]:
                    n += 1
        out[lab] = n
    return out


def label_before_correction(summary: Optional[dict], slot: int, identity_team) -> Optional[str]:
    """``old`` for the slot's FIRST recorded correction: the finalize label in
    ``meta.summary.slot_labels`` (AMBIGUOUS -> None) when the replay has one,
    else the ``match_robot_identity`` row's team_number."""
    for e in (summary or {}).get("slot_labels") or []:
        if e.get("slot") == slot:
            return norm_team(e.get("team_number"))
    return norm_team(identity_team)

"""Exact per-(frame, slot) discontinuity records (knob ``slot_boundary_log``;
also collected whenever ``identity_segment_resolve`` is on).

A slot's physical robot can only change at a tracker / identity discontinuity
(qm8/qm11 diagnosis, 2026-10-13: consecutive same-slot reads with no
discontinuity between them agree 16/16 and 30/30). Until now those had to be
reconstructed offline from replay presence gaps, position steps and
``identity_reject`` rows; this module records them exactly, live, with the
geometry behind each, so the finalize segment resolve
(``web.field.identity_segments``) and offline analysis
(``scripts/identity_offline_check.py --segments``) can cut on them.

One row per (frame, slot) that had any of:

- ``seed``: a free slot was (re)seeded (``FieldTracker.slot_events``) --
  ``seed_dist_m`` from where the slot last expired (12-13 m teleports come
  from ``free_slots.pop(0)`` handing the lowest free slot to whatever robot
  appears next);
- ``expire``: the slot ran out of coast budget (it is absent from this frame);
- ``reacquire``: a COASTING slot matched a detection again -- ``missed``
  updates of coast, ``jump_m`` from its last measurement, ``pred_jump_m``
  from its coasted prediction, ``speed_mps`` implied over the coast, the
  widened ``gate_m`` (``gate*min(4,1+0.5*missed)``, up to 10 m) and ``via``
  (sticky track id / distance / portal);
- ``feed_change``: the fused track id feeding the slot changed in place
  (``prev_track_id`` -> ``track_id``);
- ``birth``: the identity registry rejected the slot's observation against
  its nearest candidate and created an identity (``birth_distance_m``,
  ``birth_disrupted`` = the slot had a tracker disruption, ``birth_reason``);
- ``corner_only_on`` / ``corner_only_off``: the slot's position source
  switched to / from corner cameras only (poor-range projections).

Recording is observation-only: it reads tracker / registry state after the
frame's decisions and never feeds back into tracking or labelling. Which
kinds actually CUT a slot into segments is decided at finalize
(``identity_segments.is_cut``), so the policy can be re-tuned offline on the
persisted rows (diagnostic JSONL ``"type": "slot_boundary"``).
"""
from __future__ import annotations

__all__ = ["frame_boundary_rows", "BOUNDARY_KINDS"]

BOUNDARY_KINDS = ("seed", "expire", "reacquire", "feed_change", "birth",
                  "corner_only_on", "corner_only_off")


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def frame_boundary_rows(frame: int, slot_events: list[dict] | None,
                        feed_changes: dict | None, births: list[dict] | None,
                        corner_only: dict | None, prev_corner_only: dict | None,
                        positions: dict | None = None) -> list[dict]:
    """Boundary rows for one processed frame, one per slot with any event.

    ``slot_events``: ``FieldTracker.slot_events`` of this frame's update;
    ``feed_changes``: ``{slot: (prev_track_id, track_id)}``; ``births``: the
    registry's ``frame_rejections`` (outcome born / cap_rejected);
    ``corner_only`` / ``prev_corner_only``: ``{slot: bool}`` for this / the
    previous frame's emitted tracks; ``positions``: ``{slot: (x, y)}``.
    """
    rows: dict[int, dict] = {}

    def row(slot) -> dict:
        slot = int(slot)
        if slot not in rows:
            pos = (positions or {}).get(slot)
            rows[slot] = {"frame": int(frame), "slot_id": slot, "kinds": [],
                          "pos": [_r(pos[0]), _r(pos[1])] if pos is not None else None}
        return rows[slot]

    for ev in slot_events or ():
        kind = ev.get("event")
        if kind not in ("seed", "expire", "reacquire") or ev.get("slot_id") is None:
            continue
        r = row(ev["slot_id"])
        r["kinds"].append(kind)
        if kind == "seed":
            r["seed_dist_m"] = _r(ev.get("seed_dist_m"))
            r["track_id"] = ev.get("track_id")
        elif kind == "expire":
            r["missed"] = ev.get("missed")
        else:
            r["missed"] = ev.get("missed")
            r["jump_m"] = _r(ev.get("jump_m"))
            r["pred_jump_m"] = _r(ev.get("pred_jump_m"))
            r["speed_mps"] = _r(ev.get("speed_mps"), 2)
            r["gate_m"] = _r(ev.get("gate_m"))
            r["via"] = ev.get("via")
            r["track_id"] = ev.get("track_id")
    for slot, (prev_tid, tid) in sorted((feed_changes or {}).items()):
        r = row(slot)
        r["kinds"].append("feed_change")
        r["prev_track_id"] = prev_tid
        r["track_id"] = tid
    for b in births or ():
        if b.get("slot_id") is None:
            continue
        r = row(b["slot_id"])
        if "birth" in r["kinds"]:
            continue
        r["kinds"].append("birth")
        r["birth_distance_m"] = _r(b.get("distance_m"))
        r["birth_disrupted"] = bool(b.get("slot_disrupted"))
        r["birth_reason"] = b.get("reason")
        r["birth_outcome"] = b.get("outcome")
    for slot, now in sorted((corner_only or {}).items()):
        before = (prev_corner_only or {}).get(slot)
        if before is None or bool(before) == bool(now):
            continue
        row(slot)["kinds"].append("corner_only_on" if now else "corner_only_off")
    return [rows[s] for s in sorted(rows)]

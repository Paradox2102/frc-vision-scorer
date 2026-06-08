# Backward Track Confirmation — Slice 3b (Slot-Keyed Re-Resolution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Fix the slice-3 catastrophic drop by keying re-resolution on the stable `slot_id` (not the churning `identity_id`): ~full frame coverage, ref/static *slots* excluded, retroactive team per slot. This is the foundation the fragment-stitcher (next milestone) refines.

**Why:** The `43d5f2e2` run created 562 identity_ids for 6 robots; keying `slot_team_map` by the final canonical identity_id matched ~none of the per-frame snapshots → 94% of rows dropped (mean 0.29 rows/frame vs 4.97). Every snapshot carries a stable `slot_id` (0–5); keying by it restores coverage.

**Spec:** `docs/superpowers/specs/2026-06-07-backward-track-confirmation-design.md`. **Builds on:** slice-3 commits + guard (`5e237ebb`).

## Design
- Aggregate the identity_log per **slot_id** → `TrackEvidence` per slot → `classify_track` → `{slot_id: verdict}`. A slot that is a ref the whole match (high overlap, no confirm) classifies human → excluded; a robot slot → kept.
- Build `slot_meta = {slot_id: {team_number, confirmed, alliance}}` from the final canonical assignment (`_canonical_identity_slots()` → identity → `robot_tracks` team), keyed by **slot_id** (stable).
- Re-emit: for each snapshot, if its slot's verdict is robot, emit a row at that slot with `slot_meta`'s team/alliance and the snapshot's position. Retroactive team applies per slot across all frames.
- Slot-churn correction (a robot swapping slots) is NOT addressed here — that's the stitcher. This restores coverage + ref-exclusion + retro-team and produces the churn data to design the stitcher.

## File structure
- **Modify** `web/field/track_evidence.py` — refactor a shared `_evidence_keyed(log, key_field)`; add `slot_evidence` (and keep `identity_evidence` delegating to it); add `reresolve_frames_by_slot`. Pure.
- **Modify** `tests/field/test_track_evidence.py` — tests.
- **Modify** `web/app.py` — rework `_reresolve_replay_frames` to slot-keyed (keep the catastrophic-drop guard).
- **Modify** `scripts/rerun_reresolve.py` — dump `job._identity_log` to a JSON sidecar for churn analysis.

---

## Task 1: `slot_evidence` + `reresolve_frames_by_slot` (pure)

**Files:** Modify `web/field/track_evidence.py`, `tests/field/test_track_evidence.py`.

- [ ] **Step 1: Failing tests** (append)
```python
from web.field.track_evidence import slot_evidence, reresolve_frames_by_slot, classify_track


def test_slot_evidence_aggregates_by_slot():
    log = []
    for f in range(10):  # slot 0: robot (moves), two different identity_ids over time
        log.append({"frame_num": f, "identity_id": 100 + f, "slot_id": 0, "alliance": "blue",
                    "pos": [2.0 + 0.5 * f, 4.0], "person_overlap": False, "confirmed": False})
    for f in range(10):  # slot 3: ref (overlap, no motion/confirm) -> human
        log.append({"frame_num": f, "identity_id": 200 + f, "slot_id": 3, "alliance": "red",
                    "pos": [8.0, 7.0], "person_overlap": True, "confirmed": False})
    sev = slot_evidence(log)
    assert set(sev) == {0, 3}
    assert classify_track(sev[0]) == "robot"   # aggregated motion across churning ids
    assert classify_track(sev[3]) == "human"


def test_reresolve_by_slot_full_coverage_and_retro_team():
    # slot 0 robot present all 3 frames under DIFFERENT identity_ids (churn); slot 3 ref
    log = [
        {"frame_num": 0, "identity_id": 11, "slot_id": 0, "alliance": "blue", "pos": [1.0, 4.0]},
        {"frame_num": 1, "identity_id": 12, "slot_id": 0, "alliance": "blue", "pos": [1.1, 4.0]},
        {"frame_num": 2, "identity_id": 13, "slot_id": 0, "alliance": "blue", "pos": [1.2, 4.0]},
        {"frame_num": 1, "identity_id": 99, "slot_id": 3, "alliance": "red", "pos": [8.0, 7.0]},
    ]
    slot_meta = {0: {"team_number": "254", "confirmed": True, "alliance": "blue"}}
    slot_verdicts = {0: "robot", 3: "human"}
    frames = reresolve_frames_by_slot(log, slot_meta, slot_verdicts, frame_nums=[0, 1, 2])
    # slot 0 robot emitted in ALL 3 frames despite 3 different identity_ids; slot 3 excluded
    assert [len(f["robots"]) for f in frames] == [1, 1, 1]
    for f in frames:
        r = f["robots"][0]
        assert r["slot"] == "B1" and r["team_number"] == "254" and r["confirmed"] is True
    assert frames[0]["robots"][0]["field_pos"] == [1.0, 4.0]
```

- [ ] **Step 2:** Run `-k "slot_evidence or reresolve_by_slot"` — expect FAIL.

- [ ] **Step 3: Implement.** Refactor the shared aggregator and add the two functions:
```python
def _evidence_keyed(identity_log, key_field):
    out: dict = {}
    for s in identity_log:
        k = s[key_field]
        te = out.get(k)
        if te is None:
            te = TrackEvidence(track_id=k)
            out[k] = te
        te.observe_frame(s["frame_num"], bool(s.get("person_overlap")))
        pos = s.get("pos")
        if pos is not None:
            te.add_position(s["frame_num"], pos[0], pos[1])
        if s.get("confirmed"):
            te.confirm(s["frame_num"], team_read=True)
    return out


def identity_evidence(identity_log) -> dict:
    """{identity_id: TrackEvidence}."""
    return _evidence_keyed(identity_log, "identity_id")


def slot_evidence(identity_log) -> dict:
    """{slot_id: TrackEvidence} — aggregates every snapshot that occupied each slot,
    across identity_id churn. slot_id is the stable whole-match unit."""
    return _evidence_keyed(identity_log, "slot_id")


def reresolve_frames_by_slot(identity_log, slot_meta, slot_verdicts, frame_nums):
    """Rebuild rows keyed by stable slot_id. Emit each snapshot at its own slot when
    that slot's whole-match verdict is 'robot'; team/alliance from slot_meta (final
    canonical, keyed by slot_id), position from the snapshot. Non-robot slots excluded."""
    by_frame: dict = {fn: [] for fn in frame_nums}
    for s in identity_log:
        sid = s["slot_id"]
        if slot_verdicts.get(sid) != "robot":
            continue
        fn = s["frame_num"]
        if fn not in by_frame:
            continue
        meta = slot_meta.get(sid, {})
        by_frame[fn].append({
            "slot": _ID_TO_SLOT_LABEL.get(sid),
            "alliance": meta.get("alliance") or s.get("alliance"),
            "team_number": meta.get("team_number"),
            "confirmed": bool(meta.get("confirmed")),
            "field_pos": s.get("pos"),
        })
    return [{"frame_num": fn, "robots": by_frame[fn]} for fn in frame_nums]
```
(Replace the existing `identity_evidence` body with the delegation; keep the existing identity_evidence tests green.)

- [ ] **Step 4:** Run full `tests/field/test_track_evidence.py` — expect PASS (existing + 2 new).
- [ ] **Step 5: Commit:** `feat(evidence): slot-keyed evidence + re-emit (stable whole-match unit)`

---

## Task 2: Rework `_reresolve_replay_frames` to slot-keyed + persist identity_log in rerun

**Files:** Modify `web/app.py` (edit + report, NO commit — controller stages); Modify `scripts/rerun_reresolve.py`.

- [ ] **Step 1:** Rework `_reresolve_replay_frames` (keep the guard, swap identity→slot keying):
```python
    def _reresolve_replay_frames(self) -> None:
        """Slice 3b: rebuild replay frames keyed by stable slot_id (identity_id churns
        — hundreds per match). Per-slot verdict excludes ref/static slots; retroactive
        team per slot. Slot-churn correction is the stitcher's job. Guarded fallback."""
        from web.field.track_evidence import slot_evidence, classify_track, reresolve_frames_by_slot
        rec = getattr(self, "_replay", None)
        if rec is None or not getattr(self, "_identity_log", None):
            return self._apply_track_verdicts()
        sev = slot_evidence(self._identity_log)
        slot_verdicts = {sid: classify_track(te) for sid, te in sev.items()}
        assignments = self._canonical_identity_slots()        # {slot_id: identity_id}
        slot_meta = {}
        for slot_id, iid in assignments.items():
            robot = self.robot_tracks.get(iid)
            slot_meta[slot_id] = {
                "team_number": self._get_display_team_number(robot) if robot is not None else None,
                "confirmed": bool(robot and robot.team_number_confirmed),
                "alliance": self._slot_alliance(slot_id),
            }
        frame_nums = [f["frame_num"] for f in rec.frames]
        orig_rows = sum(len(f.get("robots", [])) for f in rec.frames)
        rebuilt = reresolve_frames_by_slot(self._identity_log, slot_meta, slot_verdicts, frame_nums)
        rebuilt_rows = sum(len(f["robots"]) for f in rebuilt)
        if rebuilt_rows >= max(1, orig_rows) * RERESOLVE_MIN_KEEP_FRAC:
            orig = {f["frame_num"]: f for f in rec.frames}
            rec.frames = [{**orig.get(f["frame_num"], {}), **f} for f in rebuilt]
        else:
            print(f"[backward-confirm] re-resolution kept {rebuilt_rows}/{orig_rows} rows "
                  f"(< {RERESOLVE_MIN_KEEP_FRAC:.0%}); falling back to slice-2 prune")
            self._apply_track_verdicts()
        try:
            self.identity_diagnostics.gauge("reresolve_slot_verdicts",
                {k: list(slot_verdicts.values()).count(k) for k in ("robot", "human", "static")})
        except Exception:
            pass
```

- [ ] **Step 2:** In `scripts/rerun_reresolve.py`, after `process_video`, dump the log for churn analysis:
```python
    import json as _json
    log = getattr(job, "_identity_log", [])
    out = ROOT / "web/replays" / JOB_ID / "identity_log.json"
    out.write_text(_json.dumps(log))
    print(f"identity_log dumped: {len(log)} entries -> {out}")
```

- [ ] **Step 3:** Update the integration test if needed — the existing `test_reresolve_rebuilds_frames_excluding_human` and `test_reresolve_falls_back_on_catastrophic_drop` must still pass against the slot-keyed method. They use `_canonical_identity_slots` + per-slot data; verify slot 0 robot / slot 3 human still hold (the human at slot 3 is now excluded by slot_verdict, identity 9's slot 3 has overlap=True/no-confirm → human). Adjust the tests' setup ONLY if the slot-keyed semantics require it (e.g. the catastrophic-drop test's single-slot rebuild). Run them.

- [ ] **Step 4:** Verify: `.venv/bin/python -m pytest tests/field/test_track_evidence.py tests/field/test_reresolve_integration.py -q` green; `.venv/bin/python -c "import web.app"` clean; `tests/field -q` no new failures. Report edit list (controller stages app.py surgically; commit scripts/rerun_reresolve.py with Task-1 or report).

---

## After build: validate on `43d5f2e2`
- Re-run `.venv/bin/python scripts/rerun_reresolve.py` (~29 min) → expect mean rows/frame back near ~5 (matching the original 4.97), ref/static slots excluded.
- Analyze the dumped `identity_log.json`: measure slot-churn (how often a slot's occupant swaps robots / position discontinuities) → evidence for the stitcher design.

## Done criteria (slice 3b)
- `slot_evidence`/`reresolve_frames_by_slot` pure + tested; `identity_evidence` refactored to share the aggregator, existing tests green.
- `_reresolve_replay_frames` keyed by slot_id; coverage restored on real data (no catastrophic drop); guard intact.
- identity_log persisted by the rerun script for churn analysis.
- Summary unchanged (pass still runs after the snapshot). Slot-churn correction explicitly deferred to the stitcher.

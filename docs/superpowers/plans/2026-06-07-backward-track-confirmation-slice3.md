# Backward Track Confirmation — Slice 3 (Global Identity Re-Resolution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-emit the recorded replay frames from whole-match robot IDENTITIES rather than the live forward binding — giving each identity one consistent slot across the match, its confirmed team number applied retroactively (incl. the parked-in-auto period), and human/static identities excluded.

**Architecture:** Identity-anchored (reuse the live `IdentityRegistry` clustering). Add per-frame identity logging (instrumentation), synthesize a per-identity `TrackEvidence` and reuse `classify_track` for the verdict, build a stable identity→slot/team map from the final canonical assignment, and rebuild every replay frame's rows. Supersedes slice 2's row pruning when enabled.

**Tech stack:** Python 3.12, pytest. Reuses slice-1/2 `TrackEvidence`/`classify_track` and `box_ios`.

**Spec:** `docs/superpowers/specs/2026-06-07-backward-track-confirmation-design.md`
**Depends on:** slices 1 (`99209697`,`a8d1c664`,`3a2cf35a`,`88fdfc52`) and 2 (`b0c76691`,`3b10b579`,`80905a8c`).

---

## Design (READ FIRST)

### Why new logging is required
`RobotIdentity` keeps only `last_seen_pos` (no trajectory); `active_slot` is reset to `None` each frame and re-set by Hungarian matching in `IdentityRegistry.update_frame`; there is no retained `fused_track_id→identity_id` map. So whole-match per-identity trajectories cannot be reconstructed from final state — we must log them as they happen.

### The pass (runs at finalize, identity-anchored)
1. **Instrument (Task 1):** each processed frame, after `_sync_identity_registry_frame`, append to `self._identity_log` one snapshot per identity that is currently slot-bound:
   `{frame_num, identity_id, slot_id, alliance, pos:[x,y]|None, person_overlap:bool, confirmed:bool}`.
   `pos` from `ident.last_seen_pos`; `person_overlap` via `box_ios` of the backing fused observation's box vs. that frame's main person boxes (`slot_to_fused` is in scope inside `_sync`); `confirmed` from `ident.is_confirmed or ident.team_number is not None`.
2. **Per-identity verdict (Task 2):** group the log by `identity_id`; synthesize a `TrackEvidence` per identity (positions → `add_position`; overlap frames → `observe_frame`; `confirmed` → `ever_team_read=True`); reuse `classify_track`. Verdict robot/human/static at the identity level.
3. **Final slot/team map (Task 3):** at finalize, the canonical assignment `{slot_id: identity_id}` and each slot's display team / confirmed flag are known. Build `identity_id → (slot_id, team_number, confirmed)` for identities that (a) classify **robot** AND (b) hold a slot in the final canonical assignment.
4. **Re-emit (Task 3):** for each logged frame, for each robot identity present that frame (snapshot with a `pos`), emit a row `{slot: label(final_slot), alliance, team_number: final_team, confirmed: final_confirmed, field_pos: pos}`. Frames get rows for exactly the identities that were live that frame — but each carries its FINAL slot (consistent across match) and FINAL team (retroactive). Human/static identities are never emitted.
5. **Wire (Task 4):** in the finalize block, when `BACKWARD_CONFIRM_ENABLED`, call `_reresolve_replay_frames()` INSTEAD of slice-2's `_apply_track_verdicts()` (re-emit subsumes pruning). Placed after the summary snapshot is built, before `rec.finalize`, fully wrapped. If re-resolution raises or yields zero rows, fall back to leaving `rec.frames` as-is (slice-2 pruned) — never produce an empty replay.

### Decisions (explicit)
- **Unit = durable identity**, reusing live clustering (chosen: identity-anchored).
- **One slot per identity for the whole match** = its FINAL canonical slot. This is what makes it "re-resolved/consistent" (live `active_slot` churns).
- **Retroactive team** = the identity's final display team, applied to all its frames.
- **Per-identity verdict** reuses `classify_track` on aggregated evidence; `confirmed` maps to the strong (read/appearance) signal.
- **Out of scope / limitations (flagged):**
  - An identity present early but NOT in the final canonical assignment (e.g., a robot that left before match end, or a tombstoned duplicate) gets no slot → its frames are dropped. Acceptable for the fixed 6-robot model; note in code.
  - A real robot that was tracked but NEVER slot-bound during its parked period (so never logged) is not back-filled by this slice (would need the slice-1 track-level evidence fused to the identity). Deferred.
- **Safety:** gated + wrapped; non-empty-result guard; summary built before the pass (unchanged), identical to slice 2.

### File structure
- **Modify** `web/field/track_evidence.py` — pure functions `identity_evidence(log)` (→ {identity_id: TrackEvidence}) and `reresolve_frames(identity_log, slot_team_map, verdicts, frame_nums)` (→ rebuilt frames). Pure, tested.
- **Modify** `tests/field/test_track_evidence.py` — unit tests.
- **Modify** `web/app.py` — `self._identity_log` init; log inside `_sync_identity_registry_frame` (pass main person boxes); `_reresolve_replay_frames()` method; swap the finalize call.
- **Create** `tests/field/test_reresolve_integration.py`.

DRY/YAGNI: reuse `TrackEvidence`/`classify_track`/`box_ios`/`slot_label_to_id`. Do not implement slice-4 attribution re-run.

---

## Task 1: Per-frame identity logging (instrumentation, no output change)

**Files:** Modify `web/app.py`; Test via Task-4 integration (this task adds no pure unit surface).

> Implementer: `web/app.py` has unrelated WIP — make edits, run import + tests, but DO NOT commit; report the edit list for surgical staging.

- [ ] **Step 1:** In `InferenceJob.__init__` (near `self._binding_log`), add:
```python
        self._identity_log: list[dict] = []   # per-frame identity snapshots for re-resolution
```

- [ ] **Step 2:** Log from INSIDE `_sync_identity_registry_frame` (it already builds the local `slot_to_fused` map at ~app.py:3123). Add TWO parameters: `main_person_boxes=None` and `log_frame_num=None`. **CRITICAL — use the passed `log_frame_num`, NOT `self._current_frame`.** `self._current_frame` is assigned at the END of the loop body (app.py:4864), so inside this method (called at 4481) it still holds the PREVIOUS processed frame (and `0` on the first iteration). If the snapshot frame numbers don't match `rec.frames`' frame numbers (recorded with the loop-local `frame_num` at 4792), `reresolve_frames` filters every snapshot out and silently falls back to the slice-2 prune on every run. At the END of the method (after `update_frame`), append the snapshot, wrapped so it can never break the sync:
```python
        if log_frame_num is not None:
            try:
                for ident in self.identity_registry.identities.values():
                    slot = ident.active_slot
                    if slot is None:
                        continue
                    fobs = slot_to_fused.get(slot)
                    box = fobs.detection.box if (fobs is not None and fobs.detection is not None) else None
                    overlap = bool(box is not None and main_person_boxes and any(
                        box_ios(tuple(box), tuple(pb)) >= PERSON_OVERLAP_IOS
                        for pb in main_person_boxes))
                    self._identity_log.append({
                        "frame_num": log_frame_num,
                        "identity_id": ident.identity_id,
                        "slot_id": slot,
                        "alliance": ident.alliance,
                        "pos": list(ident.last_seen_pos) if ident.last_seen_pos is not None else None,
                        "person_overlap": overlap,
                        "confirmed": bool(ident.is_confirmed or ident.team_number is not None),
                    })
            except Exception:
                pass
```

- [ ] **Step 3:** Pass the frame number + person boxes at BOTH call sites (verified line numbers):
  - **Main loop, app.py:4481** — `frame_num` and `person_boxes_by_cam` are in scope:
```python
                await self._sync_identity_registry_frame(
                    fused, ev_list, track_outs,
                    main_person_boxes=person_boxes_by_cam.get("main", []),
                    log_frame_num=frame_num)
```
  - **Early/auto pre-pass, app.py:4275** — pass its frame (`early_ocr_source_frame` is the early frame in scope ~4270); person boxes default to None (overlap=False). This contributes parked-auto identity evidence:
```python
                await self._sync_identity_registry_frame(
                    early_fused, [], early_outs, log_frame_num=early_ocr_source_frame)
```
  (Early frames usually aren't in `rec.frames`, so their snapshots inform the per-identity verdict but don't emit rows — harmless and useful.)

- [ ] **Step 4:** Verify: `.venv/bin/python -c "import web.app"` clean; `.venv/bin/python -m pytest tests/field -q` no new failures. Report edit list; do NOT commit.

---

## Task 2: `identity_evidence` + per-identity verdicts (pure)

**Files:** Modify `web/field/track_evidence.py`; Test `tests/field/test_track_evidence.py`.

- [ ] **Step 1: Failing test** (append)
```python
from web.field.track_evidence import identity_evidence, classify_track


def test_identity_evidence_aggregates_and_classifies():
    log = []
    # identity 1: confirmed robot, parked (no motion) -> robot via confirmed
    for f in range(6):
        log.append({"frame_num": f, "identity_id": 1, "slot_id": 0, "alliance": "blue",
                    "pos": [2.0, 4.0], "person_overlap": False, "confirmed": (f >= 4)})
    # identity 2: moving ref (overlap, no confirm) -> human
    for f in range(10):
        log.append({"frame_num": f, "identity_id": 2, "slot_id": 3, "alliance": "red",
                    "pos": [2.0 + 0.5 * f, 7.0], "person_overlap": True, "confirmed": False})
    ev = identity_evidence(log)
    assert set(ev) == {1, 2}
    assert ev[1].ever_team_read is True          # confirmed anywhere -> strong evidence
    assert classify_track(ev[1]) == "robot"      # stationary but confirmed
    assert classify_track(ev[2]) == "human"      # moving but person-overlap
```

- [ ] **Step 2:** Run `-k identity_evidence` — expect FAIL.

- [ ] **Step 3: Implement** in `web/field/track_evidence.py`:
```python
def identity_evidence(identity_log) -> dict:
    """Synthesize a TrackEvidence per identity_id from per-frame identity snapshots,
    so the slice-1/2 classify_track can be reused at identity granularity. A snapshot
    with confirmed=True maps to the strong read/appearance signal."""
    out: dict = {}
    for s in identity_log:
        iid = s["identity_id"]
        te = out.get(iid)
        if te is None:
            te = TrackEvidence(track_id=iid)
            out[iid] = te
        te.observe_frame(s["frame_num"], bool(s.get("person_overlap")))
        pos = s.get("pos")
        if pos is not None:
            te.add_position(s["frame_num"], pos[0], pos[1])
        if s.get("confirmed"):
            te.confirm(s["frame_num"], team_read=True)
    return out
```

- [ ] **Step 4:** Run full `tests/field/test_track_evidence.py` — expect PASS.

- [ ] **Step 5: Commit** (track_evidence.py + test; Co-Authored-By trailer):
`feat(evidence): identity_evidence — per-identity TrackEvidence for re-resolution`

---

## Task 3: `reresolve_frames` (pure)

**Files:** Modify `web/field/track_evidence.py`; Test `tests/field/test_track_evidence.py`.

- [ ] **Step 1: Failing test** (append)
```python
from web.field.track_evidence import reresolve_frames


def test_reresolve_emits_consistent_slot_and_retro_team():
    # identity 1 robot in slot 0; present frames 0..2; final team "254" confirmed
    identity_log = [
        {"frame_num": 0, "identity_id": 1, "slot_id": 0, "alliance": "blue", "pos": [1.0, 4.0]},
        {"frame_num": 1, "identity_id": 1, "slot_id": 0, "alliance": "blue", "pos": [1.1, 4.0]},
        {"frame_num": 2, "identity_id": 1, "slot_id": 0, "alliance": "blue", "pos": [1.2, 4.0]},
        # identity 9 human, present frame 1 — must NOT be emitted
        {"frame_num": 1, "identity_id": 9, "slot_id": 3, "alliance": "red", "pos": [8.0, 7.0]},
    ]
    slot_team_map = {1: {"slot_id": 0, "team_number": "254", "confirmed": True, "alliance": "blue"}}
    verdicts = {1: "robot", 9: "human"}
    frames = reresolve_frames(identity_log, slot_team_map, verdicts, frame_nums=[0, 1, 2])
    # one row per frame for identity 1, slot B1, retro team on every frame incl. frame 0
    assert [f["frame_num"] for f in frames] == [0, 1, 2]
    for f in frames:
        assert len(f["robots"]) == 1
        r = f["robots"][0]
        assert r["slot"] == "B1" and r["team_number"] == "254" and r["confirmed"] is True
    assert frames[0]["robots"][0]["field_pos"] == [1.0, 4.0]
```

- [ ] **Step 2:** Run `-k reresolve` — expect FAIL.

- [ ] **Step 3: Implement** (uses `slot_label_to_id`'s inverse — add a small `_slot_id_to_label` or reuse `_SLOT_LABELS`):
```python
_ID_TO_SLOT_LABEL = {v: k for k, v in _SLOT_LABELS.items()}


def reresolve_frames(identity_log, slot_team_map, verdicts, frame_nums):
    """Rebuild per-frame replay rows from identity snapshots. Only robot-verdict
    identities that hold a final slot (in slot_team_map) are emitted; each carries
    its FINAL slot label and FINAL team (retroactive) with the snapshot's position."""
    by_frame: dict = {fn: [] for fn in frame_nums}
    for s in identity_log:
        iid = s["identity_id"]
        if verdicts.get(iid) != "robot":
            continue
        final = slot_team_map.get(iid)
        if final is None:
            continue
        fn = s["frame_num"]
        if fn not in by_frame:
            continue
        by_frame[fn].append({
            "slot": _ID_TO_SLOT_LABEL.get(final["slot_id"]),
            "alliance": final.get("alliance") or s.get("alliance"),
            "team_number": final.get("team_number"),
            "confirmed": bool(final.get("confirmed")),
            "field_pos": s.get("pos"),
        })
    return [{"frame_num": fn, "robots": by_frame[fn]} for fn in frame_nums]
```

- [ ] **Step 4:** Run full module tests — expect PASS.

- [ ] **Step 5: Commit:** `feat(evidence): reresolve_frames — rebuild replay rows from identities`

---

## Task 4: Wire `_reresolve_replay_frames` into finalize (replaces prune when enabled)

**Files:** Modify `web/app.py`; Create `tests/field/test_reresolve_integration.py`.

> Implementer: make edits, run tests, DO NOT commit; report edit list.

- [ ] **Step 1:** Add method to `InferenceJob`:
```python
    def _reresolve_replay_frames(self) -> None:
        """Slice 3: rebuild replay frames from whole-match identities (consistent
        slot per identity, retroactive team, parked back-fill, human/static excluded).
        Supersedes the slice-2 row prune. No-ops/falls back rather than emptying."""
        from web.field.track_evidence import identity_evidence, classify_track, reresolve_frames
        rec = getattr(self, "_replay", None)
        if rec is None or not getattr(self, "_identity_log", None):
            return self._apply_track_verdicts()   # fall back to slice-2 prune
        ev = identity_evidence(self._identity_log)
        verdicts = {iid: classify_track(te) for iid, te in ev.items()}
        assignments = self._canonical_identity_slots()        # {slot_id: identity_id}
        slot_team_map = {}
        for slot_id, iid in assignments.items():
            robot = self.robot_tracks.get(iid)
            slot_team_map[iid] = {
                "slot_id": slot_id,
                "team_number": self._get_display_team_number(robot) if robot is not None else None,
                "confirmed": bool(robot and robot.team_number_confirmed),
                "alliance": self._slot_alliance(slot_id),
            }
        frame_nums = [f["frame_num"] for f in rec.frames]
        rebuilt = reresolve_frames(self._identity_log, slot_team_map, verdicts, frame_nums)
        # carry over per-frame scores/extra keys from the originals; never empty-out
        if any(f["robots"] for f in rebuilt):
            orig = {f["frame_num"]: f for f in rec.frames}
            rec.frames = [{**orig.get(f["frame_num"], {}), **f} for f in rebuilt]
        else:
            self._apply_track_verdicts()
        try:
            self.identity_diagnostics.gauge("reresolve_verdicts",
                {k: list(verdicts.values()).count(k) for k in ("robot", "human", "static")})
        except Exception:
            pass
```

- [ ] **Step 2:** In the finalize block, REPLACE the slice-2 call:
```python
                    if BACKWARD_CONFIRM_ENABLED:
                        try:
                            self._reresolve_replay_frames()
                        except Exception as e:
                            print(f"[backward-confirm] re-resolution skipped: {e}")
                    artifacts = rec.finalize(
```
This REPLACES the slice-2 `self._apply_track_verdicts()` call committed at app.py:4972 (inside `if BACKWARD_CONFIRM_ENABLED:` at 4970), which sits after the summary is built at app.py:4951 (`"robots": [r for f in rec.frames[:1] ...]`) and before `artifacts = rec.finalize(` at 4975. (The `{**orig, **f}` merge preserves `scores`/`i`/`t` keys; `reresolve_frames` overwrites `robots`. The summary at 4951 is still built BEFORE this call — unchanged.)

- [ ] **Step 3: Integration test** `tests/field/test_reresolve_integration.py`:
```python
from types import SimpleNamespace
from web.app import InferenceJob


def _job():
    job = InferenceJob.__new__(InferenceJob)
    job._identity_log = []
    job._replay = SimpleNamespace(frames=[])
    job.robot_tracks = {}
    job.identity_diagnostics = SimpleNamespace(gauge=lambda *a, **k: None)
    return job


def test_reresolve_rebuilds_frames_excluding_human(monkeypatch):
    job = _job()
    # identity 1 robot slot 0 team 254 (frames 0-2); identity 9 human slot 3 (frame 1)
    for f in range(3):
        job._identity_log.append({"frame_num": f, "identity_id": 1, "slot_id": 0,
                                  "alliance": "blue", "pos": [1.0, 4.0],
                                  "person_overlap": False, "confirmed": f >= 2})
    job._identity_log.append({"frame_num": 1, "identity_id": 9, "slot_id": 3,
                              "alliance": "red", "pos": [8.0, 7.0],
                              "person_overlap": True, "confirmed": False})
    job._replay.frames = [{"frame_num": f, "scores": {}, "robots": []} for f in range(3)]
    job.robot_tracks = {1: SimpleNamespace(team_number="254", team_number_confirmed=True)}
    job._canonical_identity_slots = lambda: {0: 1}
    job._get_display_team_number = lambda r: r.team_number if r else None
    job._slot_alliance = lambda sid: "blue" if sid < 3 else "red"
    job._reresolve_replay_frames()
    # identity 1 emitted in all 3 frames as B1/254; identity 9 (human) excluded everywhere
    assert all(len(f["robots"]) == 1 and f["robots"][0]["slot"] == "B1"
               and f["robots"][0]["team_number"] == "254" for f in job._replay.frames)
    assert job._replay.frames[0]["scores"] == {}   # original keys preserved
```

- [ ] **Step 4: Verify:**
  - `.venv/bin/python -m pytest tests/field/test_track_evidence.py tests/field/test_reresolve_integration.py -q` → green.
  - `.venv/bin/python -c "import web.app"` clean.
  - `.venv/bin/python -m pytest tests/field -q` → no new failures.
  - Report edit list; do NOT commit.

---

## Done criteria (slice 3)
- Per-frame identity snapshots logged (`_identity_log`), no output change from logging alone.
- `identity_evidence` + `reresolve_frames` pure and tested; reuse `classify_track`.
- `_reresolve_replay_frames` rebuilds `rec.frames` at finalize when enabled, replacing the slice-2 prune; falls back to the prune (or leaves frames) rather than emptying the replay.
- A confirmed robot shows ONE consistent slot across the match with its team retroactively on every frame incl. parked auto; a moving ref / static structure identity is excluded.
- Summary/DB roster row unchanged (pass runs after the snapshot is built).
- Knobs unchanged (`BACKWARD_CONFIRM_ENABLED`, `MOTION_CONFIRM_M`, `PERSON_OVERLAP_FRAC`, `PERSON_OVERLAP_IOS`).

## Risks / open questions
- **Identities present early but absent from the final canonical assignment are dropped** (no slot). Flagged; acceptable for the fixed roster, revisit if real robots vanish mid-match.
- **Never-slot-bound parked robots aren't back-filled** (need track→identity fusion) — deferred.
- **`_canonical_identity_slots` is end-state** — if two identities ping-pong one slot, the final winner takes all frames; the loser (if robot-verdict but unassigned) is dropped. Revisit with real footage.
- Re-resolution **replaces** the live per-frame binding in the replay; if it regresses on some match, set `BACKWARD_CONFIRM_ENABLED=0` to fall back entirely.

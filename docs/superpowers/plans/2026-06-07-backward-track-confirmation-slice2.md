# Backward Track Confirmation — Slice 2 (Classifier + Retroactive Cleanup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify each whole-match track as robot / human / static from slice-1 evidence, then retroactively prune the human/static tracks out of the recorded replay frames — so refs and parked field structures stop appearing as robots.

**Architecture:** A pure classifier (`classify_track`) and pure pruning helpers in `web/field/track_evidence.py`, plus a post-loop `_apply_track_verdicts()` on `InferenceJob` that runs after the analysis loop and rewrites `self._replay.frames` before finalize. This is the FIRST slice that changes output (pruned rows); it is gated by an env flag and fully wrapped so it can never break finalize.

**Tech stack:** Python 3.12, pytest (`.venv/bin/python -m pytest`). Consumes slice-1's `self._match_evidence` (MatchEvidence) and `self._binding_log` ([{frame_num, bindings:{track_id: slot_id}}]).

**Spec:** `docs/superpowers/specs/2026-06-07-backward-track-confirmation-design.md`
**Depends on:** slice 1 (committed: `99209697`, `a8d1c664`, `3a2cf35a`, `88fdfc52`).

---

## Classification rules (with precedence — READ FIRST)

For a track's `TrackEvidence`, in THIS ORDER:

1. `ever_team_read or ever_appearance_match` → **robot** (strong identity wins even under occlusion).
2. else `person_overlap_frac() >= PERSON_OVERLAP_FRAC` → **human** (a high-overlap mover with no read is a ref; this is checked BEFORE motion so a *moving ref* is not mislabeled robot — motion cannot exclude refs).
3. else `net_displacement_m() > MOTION_CONFIRM_M` → **robot** (moved, not a person, not read → real robot, incl. parked-in-auto-then-drove).
4. else → **static** (never moved, no read, not a person → prune).

> Confirmed product rule (user): a never-moving, no-evidence track is pruned. A track **read once but never moved** is currently KEPT (rule 1). This is the one rule to re-confirm with the user; the plan implements KEEP.

## File Structure

- **Modify** `web/field/track_evidence.py` — add `import os`, env defaults `MOTION_CONFIRM_M`/`PERSON_OVERLAP_FRAC`, pure functions `classify_track`, `track_verdicts`, `slot_label_to_id`, `prune_replay_frames`. Pure (no torch/app imports).
- **Modify** `tests/field/test_track_evidence.py` — unit tests for the above.
- **Modify** `web/app.py` — add `_apply_track_verdicts()` method + a gated, wrapped call in the finalize path; `BACKWARD_CONFIRM_ENABLED` env flag.
- **Create** `tests/field/test_track_verdicts_integration.py` — drives `_apply_track_verdicts` on a synthetic job.

DRY/YAGNI: reuse slice-1 `TrackEvidence`/`MatchEvidence`. Do NOT implement back-fill of stationary spans or attribution re-run (slices 3–4). Pruning only removes rows; it does not re-bind slots.

---

## Task 1: `classify_track` (pure, with precedence)

**Files:** Modify `web/field/track_evidence.py`; Test `tests/field/test_track_evidence.py`.

- [ ] **Step 1: Write the failing test** (append to test file)

```python
from web.field.track_evidence import classify_track


def _te(**kw):
    e = TrackEvidence(track_id=kw.pop("track_id", 1))
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def test_classify_read_track_is_robot_even_if_stationary():
    e = _te(ever_team_read=True)  # no motion, no positions
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "robot"


def test_classify_moving_ref_is_human_not_robot():
    # high person-overlap AND moved a lot: precedence must call it human
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=True)        # 100% overlap
    for f in range(10):
        e.add_position(f, 2.0 + 0.5 * f, 4.0)             # ~4.5 m of motion
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "human"


def test_classify_moving_robot_is_robot():
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=False)
    for f in range(10):
        e.add_position(f, 2.0 + 0.5 * f, 4.0)
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "robot"


def test_classify_never_moving_no_evidence_is_static():
    e = TrackEvidence(track_id=1)
    for f in range(10):
        e.observe_frame(f, overlapped_person=False)
        e.add_position(f, 5.0, 5.0)                        # parked
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "static"


def test_classify_read_beats_person_overlap():
    e = TrackEvidence(track_id=1, ever_appearance_match=True)
    for f in range(10):
        e.observe_frame(f, overlapped_person=True)
    assert classify_track(e, motion_confirm_m=0.75, person_overlap_frac=0.5) == "robot"
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/field/test_track_evidence.py -k classify -q` — expect FAIL (classify_track undefined).

- [ ] **Step 3: Implement** (in `web/field/track_evidence.py`; add `import os` at top, env defaults below the imports):

```python
import os

MOTION_CONFIRM_M = float(os.environ.get("MOTION_CONFIRM_M", "0.75"))
PERSON_OVERLAP_FRAC = float(os.environ.get("PERSON_OVERLAP_FRAC", "0.5"))


def classify_track(evidence, motion_confirm_m: float = MOTION_CONFIRM_M,
                   person_overlap_frac: float = PERSON_OVERLAP_FRAC) -> str:
    """Whole-match verdict: 'robot' | 'human' | 'static'. Precedence matters:
    person-overlap is checked BEFORE motion so a *moving ref* is classified human
    (motion cannot exclude refs); a read/appearance-confirmed track is robot even
    under occlusion."""
    if evidence.ever_team_read or evidence.ever_appearance_match:
        return "robot"
    if evidence.person_overlap_frac() >= person_overlap_frac:
        return "human"
    if evidence.net_displacement_m() > motion_confirm_m:
        return "robot"
    return "static"
```

- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/field/test_track_evidence.py -q` — expect PASS (17 total).

- [ ] **Step 5: Commit**
```
git add web/field/track_evidence.py tests/field/test_track_evidence.py
git commit  # "feat(evidence): classify_track (robot/human/static) with person-before-motion precedence" + blank line + Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: `track_verdicts`, `slot_label_to_id`, `prune_replay_frames` (pure)

**Files:** Modify `web/field/track_evidence.py`; Test `tests/field/test_track_evidence.py`.

- [ ] **Step 1: Write the failing test** (append)

```python
from web.field.track_evidence import (
    track_verdicts, slot_label_to_id, prune_replay_frames, MatchEvidence)


def test_slot_label_to_id_roundtrip():
    assert [slot_label_to_id(s) for s in ("B1", "B2", "B3", "R1", "R2", "R3")] == [0, 1, 2, 3, 4, 5]
    assert slot_label_to_id("??") is None


def test_track_verdicts_maps_each_track():
    m = MatchEvidence()
    m.confirm(track_id=1, frame=0, team_read=True)          # robot
    for f in range(10):                                     # human: overlap, no read/motion
        m.observe(track_id=2, frame=f, field_xy=(5.0, 5.0), overlapped_person=True)
    for f in range(10):                                     # static: parked, no evidence
        m.observe(track_id=3, frame=f, field_xy=(8.0, 8.0), overlapped_person=False)
    v = track_verdicts(m, motion_confirm_m=0.75, person_overlap_frac=0.5)
    assert v == {1: "robot", 2: "human", 3: "static"}


def test_prune_replay_frames_drops_non_robot_rows():
    # frame 7: slot R1 (id 3) held by track 9 (human) -> dropped; slot B1 (id 0) by track 1 (robot) -> kept
    frames = [{"frame_num": 7, "robots": [
        {"slot": "B1", "alliance": "blue"},
        {"slot": "R1", "alliance": "red"},
    ]}]
    binding_log = [{"frame_num": 7, "bindings": {1: 0, 9: 3}}]
    verdicts = {1: "robot", 9: "human"}
    out = prune_replay_frames(frames, binding_log, verdicts)
    assert [r["slot"] for r in out[0]["robots"]] == ["B1"]


def test_prune_keeps_rows_with_unknown_binding():
    # conservative: a row whose slot has no known track/verdict is KEPT, not dropped
    frames = [{"frame_num": 1, "robots": [{"slot": "B2", "alliance": "blue"}]}]
    out = prune_replay_frames(frames, [{"frame_num": 1, "bindings": {}}], {})
    assert len(out[0]["robots"]) == 1
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/field/test_track_evidence.py -k "verdicts or slot_label or prune" -q` — expect FAIL.

- [ ] **Step 3: Implement** (in `web/field/track_evidence.py`)

```python
def track_verdicts(match_evidence, motion_confirm_m: float = MOTION_CONFIRM_M,
                   person_overlap_frac: float = PERSON_OVERLAP_FRAC) -> dict:
    """{track_id: 'robot'|'human'|'static'} for every track in the registry."""
    return {tid: classify_track(te, motion_confirm_m, person_overlap_frac)
            for tid, te in match_evidence.by_track.items()}


_SLOT_LABELS = {"B1": 0, "B2": 1, "B3": 2, "R1": 3, "R2": 4, "R3": 5}


def slot_label_to_id(label):
    """'B1'..'R3' -> 0..5, or None if unrecognized."""
    return _SLOT_LABELS.get(label)


def prune_replay_frames(frames, binding_log, verdicts):
    """Return new frame records with human/static-track rows removed.

    A replay robot row is keyed by slot label only; `binding_log` (per frame,
    {track_id: slot_id}) recovers which track held that slot, and `verdicts`
    gives that track's whole-match verdict. A row is KEPT iff its track's verdict
    is 'robot'. Rows whose slot has no known binding/verdict are KEPT (conservative
    — never hide a row we cannot classify). Input is not mutated."""
    by_frame = {}
    for entry in binding_log:
        slot_to_track = {sid: tid for tid, sid in entry.get("bindings", {}).items()}
        by_frame[entry["frame_num"]] = slot_to_track
    out = []
    for fr in frames:
        slot_to_track = by_frame.get(fr.get("frame_num"), {})
        kept = []
        for row in fr.get("robots", []):
            sid = slot_label_to_id(row.get("slot"))
            tid = slot_to_track.get(sid) if sid is not None else None
            verdict = verdicts.get(tid) if tid is not None else None
            if verdict in (None, "robot"):   # keep robots and unclassifiable rows
                kept.append(row)
        out.append({**fr, "robots": kept})
    return out
```

- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/field/test_track_evidence.py -q` — expect PASS (21 total).

- [ ] **Step 5: Commit**
```
git add web/field/track_evidence.py tests/field/test_track_evidence.py
git commit  # "feat(evidence): track verdicts + replay-frame pruning helpers" + Co-Authored-By trailer
```

---

## Task 3: Wire `_apply_track_verdicts` into the finalize path (behavior change)

**Files:** Modify `web/app.py`; Create `tests/field/test_track_verdicts_integration.py`.

> Implementer note: locate by anchors, not absolute line. `web/app.py` has unrelated uncommitted WIP — make edits and run tests but DO NOT `git add`/commit; report the exact edit list so the controller can stage only these hunks.

- [ ] **Step 1: Add the env flag** near the other module-level env constants (by `PERSON_OVERLAP_IOS`):
```python
BACKWARD_CONFIRM_ENABLED = os.environ.get("BACKWARD_CONFIRM_ENABLED", "1") == "1"
```

- [ ] **Step 2: Add the method** to `InferenceJob`:
```python
    def _apply_track_verdicts(self) -> None:
        """Slice 2: after the whole match is processed, classify each track from the
        accumulated evidence and retroactively drop human/static-track rows from the
        recorded replay frames (refs / parked structures stop appearing as robots).
        Pruning only; does not re-bind slots (that is slice 3)."""
        from web.field.track_evidence import track_verdicts, prune_replay_frames
        rec = getattr(self, "_replay", None)
        if rec is None or not getattr(self, "_match_evidence", None):
            return
        verdicts = track_verdicts(self._match_evidence)
        rec.frames = prune_replay_frames(rec.frames, self._binding_log, verdicts)
        try:
            self.identity_diagnostics.gauge(
                "track_verdicts",
                {str(k): list(verdicts.values()).count(k)
                 for k in ("robot", "human", "static")})
        except Exception:
            pass
```

- [ ] **Step 3: Call it inside the `finally:` finalize block, AFTER the summary is built and BEFORE `rec.finalize(...)`.**

The finalize block lives in a `finally:` clause (so it runs on success AND partial/error runs), inside `if rec is not None and rec.is_open:` (~app.py:4920). The `summary` dict is built at ~4926-4938 — crucially its `"robots": [r for f in rec.frames[:1] for r in f["robots"]]` (line 4930) snapshots the FIRST frame's rows. We must place the verdict pass AFTER the summary literal is constructed and immediately BEFORE `artifacts = rec.finalize(...)` (~4949):

```python
                    # ... summary = { ... } built above (line ~4938) ...
                    # (gauge calls ~4939-4948) ...
                    if BACKWARD_CONFIRM_ENABLED:
                        try:
                            self._apply_track_verdicts()
                        except Exception as e:
                            print(f"[backward-confirm] verdict pass skipped: {e}")
                    artifacts = rec.finalize(
                        status=status, summary=summary, ...
                    )
```

Why here (resolves two issues the review caught):
1. **In `finally`, not `try`** — pruning applies on partial/errored runs too, and `rec.finalize` (which serializes `rec.frames`) always sees the pruned frames.
2. **After the summary is built** — the summary's `rec.frames[:1]` snapshot is evaluated BEFORE pruning, so the persisted summary / DB roster row is byte-identical to today. `prune_replay_frames` does not mutate row dicts (it builds new frame dicts with filtered lists), so the already-evaluated summary list keeps its original rows. Slice 2 therefore changes ONLY the per-frame `frames` timeline, never the summary.

- [ ] **Step 4: Write the integration test** `tests/field/test_track_verdicts_integration.py`:
```python
from types import SimpleNamespace
from web.field.track_evidence import MatchEvidence


def _job():
    from web.app import InferenceJob
    job = InferenceJob.__new__(InferenceJob)
    job._match_evidence = MatchEvidence()
    job._binding_log = []
    job._replay = SimpleNamespace(frames=[])
    job.identity_diagnostics = SimpleNamespace(gauge=lambda *a, **k: None)
    return job


def test_apply_track_verdicts_prunes_human_and_static_rows():
    job = _job()
    # track 1 robot (read), track 2 human (overlap), track 3 static (parked)
    job._match_evidence.confirm(1, 0, team_read=True)
    for f in range(8):
        job._match_evidence.observe(2, f, (5.0, 5.0), overlapped_person=True)
        job._match_evidence.observe(3, f, (9.0, 9.0), overlapped_person=False)
    job._binding_log = [{"frame_num": 0, "bindings": {1: 0, 2: 3, 3: 1}}]
    job._replay.frames = [{"frame_num": 0, "robots": [
        {"slot": "B1"}, {"slot": "R1"}, {"slot": "B2"},
    ]}]
    job._apply_track_verdicts()
    assert [r["slot"] for r in job._replay.frames[0]["robots"]] == ["B1"]  # only the robot


def test_apply_track_verdicts_noop_without_evidence():
    job = _job()
    job._replay.frames = [{"frame_num": 0, "robots": [{"slot": "B1"}]}]
    job._match_evidence = MatchEvidence()  # empty
    job._binding_log = []
    job._apply_track_verdicts()
    assert len(job._replay.frames[0]["robots"]) == 1  # nothing pruned
```

- [ ] **Step 5: Verify**
  - `.venv/bin/python -m pytest tests/field/test_track_evidence.py tests/field/test_track_verdicts_integration.py -q` → green (21 + 2).
  - `.venv/bin/python -c "import web.app"` → clean.
  - `.venv/bin/python -m pytest tests/field -q` → no NEW failures vs. baseline.
  - Report the exact edit list (controller stages only these hunks; do NOT commit).

---

## Done criteria (slice 2)

- `classify_track` implements the 4-step precedence; `track_verdicts`/`prune_replay_frames`/`slot_label_to_id` pure and tested (21 module tests).
- `_apply_track_verdicts` runs in finalize, gated by `BACKWARD_CONFIRM_ENABLED`, fully wrapped; prunes human/static rows from `self._replay.frames` before `rec.finalize`.
- A moving ref (person-overlap, no read) is classified **human** and pruned; a parked structure (no motion/evidence) is **static** and pruned; a stationary read robot and a moving robot are **kept**.
- **Summary/DB roster row is unchanged** — the verdict pass runs after the summary's `rec.frames[:1]` snapshot is built; only the persisted per-frame `frames` timeline changes.
- A replay `frame_num` with no matching `_binding_log` entry (e.g. frames before the replay opened lazily) falls through to keep-all — conservative, no crash.
- Knobs: `MOTION_CONFIRM_M` (0.75), `PERSON_OVERLAP_FRAC` (0.5), `BACKWARD_CONFIRM_ENABLED` (1).
- Slice 3 (backward identity propagation / stationary back-fill) and slice 4 (attribution re-run) remain out of scope.

## Risks / open questions

- **Pruning vs. back-fill ordering.** Slice 2 only removes rows; it does not re-assign a freed slot to a real robot that was mis-bound. If a ref and a real robot shared a slot over time, slice 2 removes the ref's frames and leaves the real robot's frames; full re-resolution is slice 3.
- **Threshold tuning.** `MOTION_CONFIRM_M`/`PERSON_OVERLAP_FRAC` may need per-event tuning; defaults are conservative (keep-biased).
- **Read-but-never-moved = robot** (kept) — flagged for user confirmation.
- **Binding/verdict gaps keep rows** (conservative) — a real robot is never hidden by a missing binding, at the cost of occasionally retaining a clutter row that lacked a binding.

# Diagnostic Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Diagnostic top-level page (Pipeline + Reads tabs) that exposes the identity/re-ID/OCR pipeline's internal decisions live and on replay, backed by a per-match `diagnostic.jsonl` and a sibling `corrections.jsonl` for user-applied ground-truth labels. Strip the everyday view of scoring chrome and rename it Tracker.

**Architecture:** A new `DiagnosticRecorder` module owns the schema and writes JSONL while inference runs. The recorder is the only place app.py touches diagnostic plumbing — callers pass typed args, never dicts. Frontend gets a vanilla-JS page that subscribes to the existing WebSocket (multiplexed by message `type`) for live mode and to a streaming HTTP endpoint for replay.

**Tech Stack:** Python (FastAPI, asyncio), vanilla JS, existing WebSocket + HTTP infrastructure. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-27-diagnostic-mode-design.md`

---

## Prerequisites

This plan starts from a tree where the in-flight identity/alliance/OCR rework (`docs/superpowers/plans/2026-05-27-identity-alliance-ocr-rework.md`) has been **merged to the working branch** so that the structured outputs the diagnostic instruments (alliance-masked Hungarian results, elimination dict, audit-overturn returns, legibility-floored crop selection) already exist.

If the rework is not merged yet, Tasks 1, 8, 9, 10, 11, 12 (recorder module, HTTP endpoints, file-rename, frontend) can still be done — they have no source dependency on the rework. Tasks 2–7 (call-site instrumentation) require the post-rework code shapes and should wait.

**Worktree:** Create a fresh worktree on a new branch (e.g. `feature/diagnostic-mode`) so this work stays isolated from any other in-progress feature. See `@superpowers:using-git-worktrees`.

**Test command:** `.venv/bin/python -m pytest tests/ -m "not model" -v` (see the `test-env-uses-venv` memory — `pytest` lives in `.venv`, not the base conda python).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `web/diagnostic.py` | Pure recorder + replay reader. Owns the diagnostic schema. | **New** |
| `web/app.py` | Inference orchestrator | Modify — instantiate recorder, sprinkle `record_*` calls at decision points, add HTTP endpoints |
| `web/static/index.html` | Tracker view shell | Modify — rename nav, remove scoring elements, add Diagnostic page section |
| `web/static/app.js` | Tracker page logic | Modify — drop scoring renders, add roster panel, route messages by `type` |
| `web/static/style.css` | All styles | Extend — diagnostic-page styles |
| `web/static/diagnostic.js` | Diagnostic page logic | **New** — subscribes to WS or replay HTTP, renders Pipeline + Reads tabs |
| `tests/test_diagnostic.py` | Recorder unit tests | **New** |
| `tests/test_diagnostic_corrections.py` | Corrections endpoint tests | **New** |

---

## Task 1: `web/diagnostic.py` — recorder module + replay iter

**Files:**
- Create: `web/diagnostic.py`
- Create: `tests/test_diagnostic.py`

Pure module. TDD. No I/O during inference except the JSONL append.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diagnostic.py`:

```python
"""Tests for web.diagnostic — the inference recorder and replay reader."""
import json
from pathlib import Path

import pytest

from web.diagnostic import DiagnosticRecorder, NullRecorder, replay_iter


def _snapshot_robots():
    return [
        {"robot_id": 3, "alliance": "red", "team": "1538", "confirmed": True,
         "votes": {"1538": 4.0}, "red_votes": 12, "blue_votes": 0,
         "last_seen_frame": 11,
         "embedding": {"age_frames": 0, "last_match_similarity": 0.82,
                       "last_match_gap": 0.11}},
    ]


def test_recorder_writes_one_of_each_event_type(tmp_path):
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    rec.record_vlm_read(frame=10, robot_id=3, crop_b64="abc", legibility=0.5,
                        raw_read="1538", matched_team="1538", alliance_read="red",
                        confidence=0.9, vote_weight=1.0)
    rec.record_vlm_filtered(frame=10, robot_id=1, crop_b64="def", legibility=0.001,
                            reason="below_legibility_floor")
    rec.record_assignment(frame=12, robot_id=3, team="1538", alliance="red",
                          confirmed=True, votes_for_assigned=4.0, votes_total=4.5,
                          top_team_share=0.89)
    rec.record_elimination(frame=12, robot_id=5, team="4322", alliance="red")
    rec.record_overturn(frame=20, robot_id=3, old_team="1538",
                        challenger_team="294", current_votes=3.0, challenger_votes=7.5)
    rec.record_reid_match(frame=15, track_id=7, canonical_id=3,
                          similarity=0.82, second_best=0.71)
    rec.record_reid_drop(frame=15, track_id=9, reason="claimed_by_lower_index",
                         would_have_matched=3, similarity=0.79)
    rec.record_snapshot(frame=12,
                        robots_state=_snapshot_robots(),
                        roster={"red": ["1538"], "blue": ["2102"]},
                        identified={"red": ["1538"], "blue": []})
    rec.close()

    lines = out.read_text().splitlines()
    assert len(lines) == 8
    parsed = [json.loads(l) for l in lines]
    assert [p["type"] for p in parsed] == [
        "vlm_read", "vlm_filtered", "assignment", "elimination",
        "overturn", "reid_match", "reid_drop", "snapshot",
    ]
    for p in parsed:
        assert "frame" in p and "t_ms" in p and "type" in p
    assert parsed[5]["gap"] == pytest.approx(0.11)
    assert parsed[7]["robots"][0]["embedding"]["age_frames"] == 0


def test_recorder_survives_io_error(tmp_path, monkeypatch):
    """If the JSONL fails to open, the recorder must degrade to null mode and
    subsequent calls must not raise."""
    real_open = open

    def fail_open(*args, **kwargs):
        path = str(args[0]) if args else ""
        if path.endswith("diag.jsonl") and "a" in (args[1] if len(args) > 1 else kwargs.get("mode", "")):
            raise OSError("simulated: no space left")
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_open)
    rec = DiagnosticRecorder(tmp_path / "diag.jsonl", ws_send=None)
    assert rec.is_null
    rec.record_elimination(frame=1, robot_id=0, team="1538", alliance="red")  # must not raise
    rec.flush()
    rec.close()


def test_recorder_drops_input_with_missing_required_fields(tmp_path):
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    rec.record_vlm_read(frame=None, robot_id=3, crop_b64="abc", legibility=0.5,
                        raw_read="1538", matched_team="1538", alliance_read="red",
                        confidence=0.9, vote_weight=1.0)
    rec.close()
    assert out.read_text() == ""


def test_replay_iter_skips_malformed_lines(tmp_path):
    p = tmp_path / "diag.jsonl"
    p.write_text(
        '{"frame":1,"t_ms":0.0,"type":"elimination","robot_id":0,"team":"1538","alliance":"red"}\n'
        '{"frame":2,"t_ms":1.0,"type":"trunc\n'
        '{"frame":3,"t_ms":2.0,"type":"elimination","robot_id":1,"team":"4322","alliance":"red"}\n'
    )
    records = list(replay_iter(p))
    assert len(records) == 2
    assert records[0]["robot_id"] == 0
    assert records[1]["robot_id"] == 1


def test_replay_iter_handles_missing_file(tmp_path):
    assert list(replay_iter(tmp_path / "nope.jsonl")) == []


def test_snapshot_round_trips_embedding_age(tmp_path):
    out = tmp_path / "diag.jsonl"
    rec = DiagnosticRecorder(out, ws_send=None)
    rec.record_snapshot(
        frame=50,
        robots_state=[
            {"robot_id": 3, "alliance": "red", "team": "1538", "confirmed": True,
             "votes": {"1538": 4.0}, "red_votes": 12, "blue_votes": 0,
             "last_seen_frame": 49,
             "embedding": {"age_frames": 42, "last_match_similarity": 0.55,
                           "last_match_gap": 0.04}},
        ],
        roster={"red": ["1538"], "blue": []},
        identified={"red": ["1538"], "blue": []},
    )
    rec.close()
    parsed = json.loads(out.read_text().strip())
    assert parsed["robots"][0]["embedding"]["age_frames"] == 42


def test_null_recorder_swallows_known_methods():
    rec = NullRecorder()
    # All record_* methods and flush/close must be callable on the null recorder.
    rec.record_vlm_read(frame=1, robot_id=2, crop_b64="x", legibility=0.1,
                        raw_read="x", matched_team=None, alliance_read="red",
                        confidence=0.5, vote_weight=0.5)
    rec.record_snapshot(frame=1, robots_state=[], roster={}, identified={})
    rec.flush()
    rec.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv/bin/python -m pytest tests/test_diagnostic.py -v
```

Expected: `ImportError: cannot import name 'DiagnosticRecorder'`.

- [ ] **Step 3: Implement `web/diagnostic.py`**

Create the file with this content:

```python
"""Diagnostic event/snapshot recorder for inference jobs.

Captures structured records of the identity pipeline's decisions and writes them
as JSONL (one record per line) for both live consumption via WebSocket and
post-match replay.

The recorder is the only place that knows the diagnostic schema — callers pass
typed arguments, never raw dicts. Failures are isolated: any I/O error degrades
the recorder to a no-op state so inference is never blocked.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

logger = logging.getLogger(__name__)

# Records are flushed to disk every N writes. Small enough to bound data loss
# on crash; large enough that we don't hammer fsync per event.
FLUSH_EVERY_N_WRITES = 32
DEFAULT_RING_SIZE = 2000


class NullRecorder:
    """Drop-in replacement that swallows all `record_*`, `flush`, `close` calls.

    Used when the recorder cannot open its output file. Inference keeps running;
    diagnostic data is silently dropped.
    """

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("record_") or name in ("flush", "close"):
            def noop(*_a: Any, **_kw: Any) -> None:
                return None
            return noop
        raise AttributeError(name)

    @property
    def is_null(self) -> bool:
        return True


class DiagnosticRecorder:
    """Writes diagnostic events/snapshots to JSONL and optionally pushes them
    via a WebSocket send callable.

    Args:
        out_path: JSONL file to append to (opened in 'a' mode).
        ws_send: Optional async function called with each record dict for live
                 streaming. Errors are caught and logged; never raised.
        ring_size: Number of most-recent records to keep in memory for late-
                   joining WS clients.
    """

    def __init__(
        self,
        out_path: Path,
        ws_send: Callable[[dict], Awaitable[None]] | None = None,
        ring_size: int = DEFAULT_RING_SIZE,
    ) -> None:
        self._out_path = Path(out_path)
        self._ws_send = ws_send
        self._ring_size = ring_size
        self._ring: list[dict] = []
        self._t0 = time.monotonic()
        self._writes_since_flush = 0
        try:
            self._fp = open(self._out_path, "a", buffering=1)
        except OSError as e:
            logger.error("DiagnosticRecorder: failed to open %s: %s", self._out_path, e)
            self._fp = None

    @property
    def is_null(self) -> bool:
        return self._fp is None

    def _t_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    def _emit(self, record: dict) -> None:
        if self._fp is None:
            return
        try:
            line = json.dumps(record, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            logger.warning("DiagnosticRecorder: dropping malformed record (%s): %r", e, record)
            return
        try:
            self._fp.write(line + "\n")
            self._writes_since_flush += 1
            if self._writes_since_flush >= FLUSH_EVERY_N_WRITES:
                self._fp.flush()
                self._writes_since_flush = 0
        except OSError as e:
            logger.error("DiagnosticRecorder: write failed, degrading to null: %s", e)
            self._close_fp()
            return

        self._ring.append(record)
        if len(self._ring) > self._ring_size:
            del self._ring[: len(self._ring) - self._ring_size]

        if self._ws_send is not None:
            try:
                result = self._ws_send(record)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                logger.warning("DiagnosticRecorder: ws_send failed: %s", e)

    # ---- typed record_* methods ----

    def record_vlm_read(self, frame, robot_id, crop_b64, legibility,
                       raw_read, matched_team, alliance_read, confidence, vote_weight):
        if frame is None or robot_id is None or crop_b64 is None:
            logger.warning("record_vlm_read: missing required field; dropped")
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "vlm_read",
            "robot_id": int(robot_id), "crop_b64": crop_b64,
            "legibility": float(legibility), "raw_read": raw_read,
            "matched_team": matched_team, "alliance_read": alliance_read,
            "confidence": float(confidence), "vote_weight": float(vote_weight),
        })

    def record_vlm_filtered(self, frame, robot_id, crop_b64, legibility, reason):
        if frame is None or robot_id is None:
            logger.warning("record_vlm_filtered: missing required field; dropped")
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "vlm_filtered",
            "robot_id": int(robot_id), "crop_b64": crop_b64,
            "legibility": float(legibility), "reason": reason,
        })

    def record_assignment(self, frame, robot_id, team, alliance, confirmed,
                         votes_for_assigned, votes_total, top_team_share):
        if frame is None or robot_id is None or team is None:
            logger.warning("record_assignment: missing required field; dropped")
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "assignment",
            "robot_id": int(robot_id), "team": str(team), "alliance": alliance,
            "confirmed": bool(confirmed),
            "votes_for_assigned": float(votes_for_assigned),
            "votes_total": float(votes_total),
            "top_team_share": float(top_team_share),
        })

    def record_elimination(self, frame, robot_id, team, alliance):
        if frame is None or robot_id is None or team is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "elimination",
            "robot_id": int(robot_id), "team": str(team), "alliance": alliance,
        })

    def record_overturn(self, frame, robot_id, old_team, challenger_team,
                       current_votes, challenger_votes):
        if frame is None or robot_id is None or old_team is None or challenger_team is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "overturn",
            "robot_id": int(robot_id), "old_team": str(old_team),
            "challenger_team": str(challenger_team),
            "current_votes": float(current_votes),
            "challenger_votes": float(challenger_votes),
        })

    def record_reid_match(self, frame, track_id, canonical_id, similarity, second_best):
        if frame is None or track_id is None or canonical_id is None:
            return
        gap = (float(similarity) - float(second_best)) if second_best is not None else None
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "reid_match",
            "track_id": int(track_id), "canonical_id": int(canonical_id),
            "similarity": float(similarity),
            "second_best": float(second_best) if second_best is not None else None,
            "gap": gap,
        })

    def record_reid_drop(self, frame, track_id, reason, would_have_matched, similarity):
        if frame is None or track_id is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "reid_drop",
            "track_id": int(track_id), "reason": reason,
            "would_have_matched": int(would_have_matched) if would_have_matched is not None else None,
            "similarity": float(similarity) if similarity is not None else None,
        })

    def record_snapshot(self, frame, robots_state, roster, identified):
        """`robots_state` is a list of dicts pre-shaped per the snapshot schema
        (including the `embedding` sub-dict). The caller in app.py is responsible
        for building it from self.robot_tracks + the embedder's last-match scores.
        The recorder validates field presence on emit but does not introspect
        RobotTrack or the embedder."""
        if frame is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "snapshot",
            "robots": list(robots_state or []),
            "roster": roster or {},
            "identified": identified or {},
        })

    # ---- lifecycle ----

    def flush(self) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
            except OSError as e:
                logger.warning("DiagnosticRecorder.flush: %s", e)

    def close(self) -> None:
        self.flush()
        self._close_fp()

    def _close_fp(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None


def replay_iter(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file. Skip malformed lines and log the count
    of skipped lines on completion."""
    p = Path(path)
    if not p.exists():
        return
    skipped = 0
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
    if skipped:
        logger.info("replay_iter(%s): skipped %d malformed lines", path, skipped)
```

- [ ] **Step 4: Run the tests to verify they pass**

```
.venv/bin/python -m pytest tests/test_diagnostic.py -v
```

Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add web/diagnostic.py tests/test_diagnostic.py
git commit -m "$(cat <<'EOF'
feat(diagnostic): recorder module + replay reader

Adds DiagnosticRecorder + NullRecorder + replay_iter. Owns the diagnostic
schema; callers pass typed args. Degrades to a null state on I/O failure
so inference is never blocked by diagnostic I/O.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire recorder into `InferenceJob` lifecycle

**Files:**
- Modify: `web/app.py` (`InferenceJob.__init__`, completion/stop paths)

No call sites yet — just the lifecycle plumbing so subsequent tasks can call `self.diagnostic.record_*(...)`.

- [ ] **Step 1: Add the import + instantiation in `__init__`**

In `web/app.py`, near the existing `from web.embedder import ...` (line ~36), add:

```python
from web.diagnostic import DiagnosticRecorder, NullRecorder
```

In `InferenceJob.__init__` (around line 246), at the end of the existing setup (after `self._ocr_crops_dirty = False`, ~line 287), append:

```python
        # Diagnostic recorder — writes JSONL during inference and pushes
        # live records over the existing websocket (multiplex by type).
        # Constructor already swallows OSError internally and falls back to
        # a null state (is_null=True), so no outer try/except is needed.
        self.diagnostic = DiagnosticRecorder(
            UPLOAD_DIR / f"diagnostic_{self.job_id}.jsonl",
            ws_send=None,  # wired in Task 7
        )
```

- [ ] **Step 2: Flush + close on completion and stop**

In `web/app.py`, find the inference-loop end and the stop path. After `self.status = "completed"` (around line 1871 currently), add:

```python
            try:
                self.diagnostic.close()
            except Exception:
                pass
```

Do the same at the stop branch (where `self._stopped` causes the loop to break out and status is set to "stopped" or similar — search for `self.status = "stopped"` or follow the existing pattern for cleanup). If there is no clean "stopped" branch, add a `try/finally` around the main loop:

```python
        try:
            # ... existing inference loop ...
        finally:
            try:
                self.diagnostic.close()
            except Exception:
                pass
```

- [ ] **Step 3: Smoke-test import + existing suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green (no behavioral change).

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "$(cat <<'EOF'
feat(app): instantiate DiagnosticRecorder per InferenceJob

Adds lifecycle plumbing only — no call sites yet. JSONL file is keyed by
job_id under the existing UPLOAD_DIR flat layout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Instrument `vision_identify_robots` — vlm_read + vlm_filtered

**Files:**
- Modify: `web/app.py` (`vision_identify_robots`, ~lines 456–556 pre-rework, shifted post-rework)

- [ ] **Step 1: Locate the two emission points**

Open `web/app.py`. In `vision_identify_robots` (after the rework), find:
1. The candidates loop — there's a `continue` immediately after `if score < LEGIBILITY_FLOOR:`. That's where `record_vlm_filtered` goes.
2. The `for score, canonical_id, box, crop_b64 in chosen:` loop — after the Qwen call returns `res`, that's where `record_vlm_read` goes.

Read those two regions (a few lines of context each) before editing.

- [ ] **Step 2: Add the `record_vlm_filtered` call**

Replace this section in `vision_identify_robots`:

```python
            score = legibility_score(crop)
            if score < LEGIBILITY_FLOOR:
                continue  # blurry / back-facing / no bumper visible — don't waste a call
```

with:

```python
            score = legibility_score(crop)
            if score < LEGIBILITY_FLOOR:
                # Record the rejected crop so the diagnostic view can show why
                # this robot went unread.
                try:
                    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    rejected_b64 = base64.b64encode(buf).decode('utf-8')
                except Exception:
                    rejected_b64 = ""
                self.diagnostic.record_vlm_filtered(
                    frame=self._current_frame, robot_id=canonical_id,
                    crop_b64=rejected_b64, legibility=score,
                    reason="below_legibility_floor",
                )
                continue
```

- [ ] **Step 3: Add the `record_vlm_read` call**

Inside the `for score, canonical_id, box, crop_b64 in chosen:` loop, after the existing `results.append((canonical_id, box, res))` and the `ocr_crops.append({...})` block, add (one record per crop sent — emit regardless of whether the read matched a team):

```python
            self.diagnostic.record_vlm_read(
                frame=self._current_frame, robot_id=canonical_id,
                crop_b64=crop_b64,
                legibility=float(score),
                raw_read=res.get("raw_read"),
                matched_team=res.get("team_number"),
                alliance_read=res.get("alliance", "unknown"),
                confidence=float(res.get("confidence", 0.0)),
                vote_weight=1.0 if float(res.get("confidence", 0.0)) >= 0.85 else 0.5,
            )
```

- [ ] **Step 4: Verify import + existing suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green.

- [ ] **Step 5: Commit**

```bash
git add web/app.py
git commit -m "$(cat <<'EOF'
feat(app): instrument vision_identify_robots — vlm_read + vlm_filtered

Records every crop sent to Qwen and every crop the legibility floor rejected
(previously invisible). Per-record fields per the diagnostic spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Instrument `assign_teams_from_votes` — assignment + elimination

**Files:**
- Modify: `web/app.py` (`assign_teams_from_votes`, post-rework ~lines 644–711)

- [ ] **Step 1: Add `record_assignment` after each Hungarian apply**

In the `for rid, team in assignment.items():` block (post-rework), where the function already sets `r.team_number`, `r.alliance`, and `r.team_number_confirmed`, append a recorder call inside the `if team is None: continue` branch:

```python
        for rid, team in assignment.items():
            r = self.robot_tracks[rid]
            if team is None:
                r.team_number = None
                continue
            v = (r.team_ocr_votes or {}).get(team, 0.0)
            total = sum((r.team_ocr_votes or {}).values()) or 1.0
            r.team_number = team
            r.alliance = "red" if team in self.match_red_teams else "blue"
            r.team_number_confirmed = bool(v >= 3.0 and v / total >= 0.6)
            self.diagnostic.record_assignment(
                frame=self._current_frame, robot_id=rid, team=team,
                alliance=r.alliance, confirmed=r.team_number_confirmed,
                votes_for_assigned=float(v), votes_total=float(total),
                top_team_share=float(v / total) if total else 0.0,
            )
            for event in self.scoring_events:
                if event.robot_id == rid:
                    event.team_number = team
```

- [ ] **Step 2: Replace the elimination print with `record_elimination`**

In the `for rid, team in eliminated.items():` block, replace the existing `print(f"  Elimination: Robot {rid} = Team {team} (only candidate left on {r.alliance})")` with:

```python
            self.diagnostic.record_elimination(
                frame=self._current_frame, robot_id=rid, team=team, alliance=r.alliance,
            )
```

(Drop the print entirely — the diagnostic log replaces it. If you want to keep stdout traces for now, add a logger.info call instead of print.)

- [ ] **Step 3: Verify import + suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green (the alliance-consistency tests must still pass).

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "$(cat <<'EOF'
feat(app): instrument assign_teams_from_votes — assignment + elimination

Records each Hungarian-produced (robot, team) and each
process-of-elimination assignment. Elimination print becomes a structured
event.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Instrument `audit_overturn_confirmations` + per-tick snapshot

**Files:**
- Modify: `web/app.py` (`audit_overturn_confirmations` method body, and the identity-interval block in the inference loop)

- [ ] **Step 1: Replace the overturn print with `record_overturn`**

In `audit_overturn_confirmations`, inside the `if new_team is None: continue` block, replace the existing `print(f"  Overturn: Robot {r.robot_id} {r.team_number} -> unlocked ...")` with:

```python
            self.diagnostic.record_overturn(
                frame=self._current_frame, robot_id=r.robot_id,
                old_team=r.team_number, challenger_team=new_team,
                current_votes=float((r.team_ocr_votes or {}).get(r.team_number, 0.0)),
                challenger_votes=float((r.team_ocr_votes or {}).get(new_team, 0.0)),
            )
            r.team_number_confirmed = False
```

- [ ] **Step 2: Add a snapshot emit at the identity-interval tick**

In the inference loop, the post-rework identity tick looks like:

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()
                    self.audit_overturn_confirmations()
```

Add a snapshot record right after those two calls:

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()
                    self.audit_overturn_confirmations()
                    self._emit_diagnostic_snapshot(processed_frame_num)
```

Then add the helper near the bottom of `InferenceJob` (next to `assign_teams_from_votes`):

```python
    def _emit_diagnostic_snapshot(self, frame: int) -> None:
        """Build a snapshot dict from current state and hand to the recorder."""
        robots_state = []
        for r in self.robot_tracks.values():
            embedding_info = {
                "age_frames": max(0, frame - getattr(r, "last_embedding_frame", frame)),
                "last_match_similarity": getattr(r, "last_reid_similarity", None),
                "last_match_gap": getattr(r, "last_reid_gap", None),
            }
            robots_state.append({
                "robot_id": r.robot_id,
                "alliance": r.alliance,
                "team": r.team_number,
                "confirmed": bool(r.team_number_confirmed),
                "votes": dict(r.team_ocr_votes or {}),
                "red_votes": int(r.red_votes),
                "blue_votes": int(r.blue_votes),
                "last_seen_frame": int(getattr(r, "last_seen_frame", frame)),
                "embedding": embedding_info,
            })
        identified = {
            "red": [r.team_number for r in self.robot_tracks.values()
                    if r.team_number and r.alliance == "red"],
            "blue": [r.team_number for r in self.robot_tracks.values()
                     if r.team_number and r.alliance == "blue"],
        }
        roster = {"red": list(self.match_red_teams), "blue": list(self.match_blue_teams)}
        self.diagnostic.record_snapshot(
            frame=frame, robots_state=robots_state,
            roster=roster, identified=identified,
        )
```

`last_embedding_frame`, `last_reid_similarity`, and `last_reid_gap` are added to `RobotTrack` in Task 6. Until then, `getattr` returns the defaults (current frame, None, None) — snapshot still works.

- [ ] **Step 3: Verify import + suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "$(cat <<'EOF'
feat(app): instrument audit_overturn_confirmations + per-tick snapshot

Records each audit-overturn unlock and a full identity-state snapshot
once per IDENTITY_INTERVAL. Snapshot reads embedding freshness fields
defensively (getattr) — they are populated in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Instrument re-ID — `reid_match` + `reid_drop`

**Files:**
- Modify: `web/app.py` (`RobotTrack` dataclass and the `reid.choose_canonical_match` call site in `get_canonical_robot_id`)

- [ ] **Step 1: Add embedding-freshness fields to `RobotTrack`**

In `web/app.py`, extend `RobotTrack` (around lines 200–216) with three optional fields:

```python
@dataclass
class RobotTrack:
    # ... existing fields ...
    appearance_votes: Optional[Dict[str, float]] = None
    # Diagnostic / embedding-freshness state. Set when a re-ID match resolves
    # to this canonical; read by _emit_diagnostic_snapshot.
    last_embedding_frame: int = 0
    last_reid_similarity: Optional[float] = None
    last_reid_gap: Optional[float] = None
    last_seen_frame: int = 0
```

(If any of these already exist in the post-rework code, keep the existing definition and don't duplicate.)

- [ ] **Step 2: Wrap the `choose_canonical_match` call site**

In `get_canonical_robot_id` (around lines 372–448), find the `reid.choose_canonical_match(...)` call. The function returns either a canonical_id (with similarity info) or a drop instruction — check the post-rework signature in `web/reid.py` to confirm the exact return shape.

For a successful match (the existing assignment branch), add right after the canonical is set:

```python
                self.robot_tracks[canonical_id].last_embedding_frame = self._current_frame
                self.robot_tracks[canonical_id].last_seen_frame = self._current_frame
                self.robot_tracks[canonical_id].last_reid_similarity = float(top_similarity)
                self.robot_tracks[canonical_id].last_reid_gap = (
                    float(top_similarity - second_similarity)
                    if second_similarity is not None else None
                )
                self.diagnostic.record_reid_match(
                    frame=self._current_frame, track_id=tracking_id,
                    canonical_id=canonical_id,
                    similarity=float(top_similarity),
                    second_best=float(second_similarity) if second_similarity is not None else None,
                )
```

`top_similarity` and `second_similarity` come from the existing scoring logic in `get_canonical_robot_id` — surface the second-best score by capturing it next to the top score (today only the top is kept).

**Concrete second-best recipe** (use when `choose_canonical_match` doesn't return it): in `get_canonical_robot_id`, the function already computes a list of candidate similarities (search for a `for r in candidates` or `scores =` block). After picking the top:

```python
# After: top_similarity, canonical_id = max(candidate_scores, key=...)
sorted_scores = sorted([s for s in candidate_scores.values()], reverse=True)
top_similarity = sorted_scores[0] if sorted_scores else 0.0
second_similarity = sorted_scores[1] if len(sorted_scores) >= 2 else None
```

Use whatever variable name the existing code uses for the score map; the recipe is "sort descending; index [0] is top, [1] is second-best or None."

For a drop (the branch that does not assign a canonical), add:

```python
                self.diagnostic.record_reid_drop(
                    frame=self._current_frame, track_id=tracking_id,
                    reason=drop_reason,
                    would_have_matched=would_match_id,  # None if no candidate
                    similarity=float(would_similarity) if would_similarity is not None else None,
                )
```

`drop_reason`, `would_match_id`, and `would_similarity` come from the drop branch's local state (the candidate that was rejected and why — e.g., `"claimed_by_lower_index"`, `"below_threshold"`, `"alliance_mismatch"`). Use the existing variable names if present.

- [ ] **Step 3: Verify import + suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green (including `tests/test_reid_identity.py` and the acceptance harness's continuity metric).

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "$(cat <<'EOF'
feat(app): instrument re-ID match/drop + add embedding-freshness fields

Records each track→canonical resolution with similarity gap, plus each
dropped track with the would-have-matched candidate. Adds RobotTrack
fields the snapshot consumes for embedding-age display.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: WebSocket multiplex — push diagnostic records live

**Files:**
- Modify: `web/app.py` (websocket attach path + recorder's `ws_send`)

- [ ] **Step 1: Define the WS push callable on the InferenceJob**

In `InferenceJob`, find where `self.websocket` is set (search for `self.websocket = `). After that, add a method:

```python
    async def _diagnostic_ws_send(self, record: dict) -> None:
        """Push one diagnostic record over the existing websocket. Best-effort:
        any failure is swallowed (the JSONL file is the canonical record)."""
        if not self.websocket:
            return
        try:
            msg_type = ("diagnostic_snapshot"
                       if record.get("type") == "snapshot"
                       else "diagnostic_event")
            await self.websocket.send_json({"type": msg_type, "record": record})
        except Exception:
            pass
```

- [ ] **Step 2: Add `set_ws_send` to `DiagnosticRecorder` and wire it at WS-attach time**

First, extend `web/diagnostic.py` to expose the wiring cleanly (avoids reaching into a `_private` attribute from app.py):

```python
    def set_ws_send(self, ws_send):
        """Attach (or replace) the WebSocket push callable after construction.
        Called by InferenceJob once the websocket connects. Pass None to detach."""
        self._ws_send = ws_send
```

Add the matching null-method stanza to `NullRecorder.__getattr__` by extending the existing predicate — `set_ws_send` becomes another method that's silently no-op'd on the null path:

```python
        if name.startswith("record_") or name in ("flush", "close", "set_ws_send"):
```

Then in the websocket-attach path (the FastAPI WS endpoint in `web/app.py`), after `job.websocket = websocket` is set, call:

```python
    job.websocket = websocket
    job.diagnostic.set_ws_send(job._diagnostic_ws_send)
```

No `hasattr` / `is_null` guard needed — the null recorder swallows it.

- [ ] **Step 3: Verify import + suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "$(cat <<'EOF'
feat(app): multiplex diagnostic records onto the existing WebSocket

Diagnostic events ride the same connection as frame data, tagged
'diagnostic_event' or 'diagnostic_snapshot'. Tracker frontend ignores
them; Diagnostic page (next tasks) handles them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Replay endpoint — `GET /api/diagnostic/<key>`

**Files:**
- Modify: `web/app.py` (FastAPI route)

- [ ] **Step 1: Add the streaming endpoint**

Find where other `@app.get(...)` routes are defined (search for `@app.get`). At the top of `web/app.py`, **extend the existing fastapi import** (currently `from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect`) by adding `HTTPException` to the list — without this, every route below that uses `raise HTTPException(...)` will throw `NameError` on first request:

```python
from fastapi import (FastAPI, File, Form, HTTPException, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
```

Then add:

```python
from fastapi.responses import StreamingResponse


@app.get("/api/diagnostic/{key}")
async def get_diagnostic(key: str):
    """Stream the diagnostic JSONL for a live job or a saved match.
    Resolution order: diagnostic_<key>.jsonl (live) then diagnostic_match_<key>.jsonl (saved).
    """
    # Path-traversal guard: only allow safe key characters.
    if not key.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="invalid key")

    candidates = [
        UPLOAD_DIR / f"diagnostic_{key}.jsonl",
        UPLOAD_DIR / f"diagnostic_match_{key}.jsonl",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        raise HTTPException(status_code=404, detail="no diagnostic data")

    def iter_file():
        with target.open("rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(iter_file(), media_type="application/x-ndjson")
```

- [ ] **Step 2: Smoke test the route manually**

```
# In one terminal:
.venv/bin/python -m web.app  # or whatever the existing launch command is
# In another:
echo '{"frame":1,"t_ms":0,"type":"elimination","robot_id":0,"team":"1538","alliance":"red"}' \
  > web/uploads/diagnostic_smoketest.jsonl
curl -s http://localhost:8000/api/diagnostic/smoketest
```

Expected: the JSONL line is returned verbatim.

- [ ] **Step 3: Verify import + suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "$(cat <<'EOF'
feat(app): GET /api/diagnostic/{key} streams JSONL for live or saved

Resolves <key> against diagnostic_<key>.jsonl first (live job) and
diagnostic_match_<key>.jsonl second (saved match). Streams as
application/x-ndjson without server-side buffering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Corrections endpoints + match-save file rename

**Files:**
- Modify: `web/app.py` (add 2 routes + hook the match-save flow)
- Create: `tests/test_diagnostic_corrections.py`

- [ ] **Step 1: Write the failing endpoint tests**

Create `tests/test_diagnostic_corrections.py`:

```python
"""Tests for the corrections HTTP endpoints (GET/POST)."""
import json

import pytest
from fastapi.testclient import TestClient

from web.app import app, UPLOAD_DIR


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Redirect UPLOAD_DIR to a temp path so the test doesn't litter web/uploads."""
    monkeypatch.setattr("web.app.UPLOAD_DIR", tmp_path)
    return TestClient(app)


def _seed_diag(tmp_path, key, reads):
    """Write a minimal diagnostic.jsonl so corrections referencing those reads validate."""
    p = tmp_path / f"diagnostic_{key}.jsonl"
    with p.open("w") as f:
        for frame, robot_id in reads:
            f.write(json.dumps({
                "frame": frame, "t_ms": 0.0, "type": "vlm_read",
                "robot_id": robot_id, "crop_b64": "",
                "legibility": 0.5, "raw_read": "1538",
                "matched_team": "1538", "alliance_read": "red",
                "confidence": 0.9, "vote_weight": 1.0,
            }) + "\n")


def test_post_correction_appends_and_get_returns_it(tmp_path, client):
    _seed_diag(tmp_path, "j1", [(10, 3)])
    r = client.post("/api/diagnostic/j1/corrections", json={
        "vlm_read": {"frame": 10, "robot_id": 3},
        "verdict": "misidentified",
        "actual_team": "4322",
        "note": "shows 4322",
    })
    assert r.status_code == 200
    g = client.get("/api/diagnostic/j1/corrections")
    assert g.status_code == 200
    payload = g.json()
    assert len(payload) == 1
    assert payload[0]["verdict"] == "misidentified"
    assert payload[0]["actual_team"] == "4322"


def test_post_correction_overrides_prior(tmp_path, client):
    _seed_diag(tmp_path, "j2", [(10, 3)])
    client.post("/api/diagnostic/j2/corrections", json={
        "vlm_read": {"frame": 10, "robot_id": 3},
        "verdict": "misidentified", "actual_team": "4322",
    })
    client.post("/api/diagnostic/j2/corrections", json={
        "vlm_read": {"frame": 10, "robot_id": 3},
        "verdict": "correct", "actual_team": None,
    })
    payload = client.get("/api/diagnostic/j2/corrections").json()
    # GET returns the latest correction per (frame, robot_id).
    assert len(payload) == 1
    assert payload[0]["verdict"] == "correct"


def test_post_correction_rejects_unknown_read(tmp_path, client):
    _seed_diag(tmp_path, "j3", [(10, 3)])
    r = client.post("/api/diagnostic/j3/corrections", json={
        "vlm_read": {"frame": 99, "robot_id": 9},  # not in diagnostic
        "verdict": "correct",
    })
    assert r.status_code == 400


def test_get_corrections_for_match_with_no_corrections(tmp_path, client):
    _seed_diag(tmp_path, "j4", [(10, 3)])
    r = client.get("/api/diagnostic/j4/corrections")
    assert r.status_code == 200
    assert r.json() == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```
.venv/bin/python -m pytest tests/test_diagnostic_corrections.py -v
```

Expected: 4 failures (404 routes don't exist).

- [ ] **Step 3: Implement the endpoints**

Add to `web/app.py` near the diagnostic endpoint from Task 8:

```python
from datetime import datetime, timezone


def _resolve_diagnostic_path(key: str) -> Path | None:
    for prefix in ("diagnostic_", "diagnostic_match_"):
        p = UPLOAD_DIR / f"{prefix}{key}.jsonl"
        if p.exists():
            return p
    return None


def _resolve_corrections_path(key: str) -> Path:
    # Match the diagnostic file's prefix family. New files use the live form.
    for prefix in ("corrections_match_", "corrections_"):
        p = UPLOAD_DIR / f"{prefix}{key}.jsonl"
        if p.exists():
            return p
    return UPLOAD_DIR / f"corrections_{key}.jsonl"


def _vlm_read_exists(diag_path: Path, frame: int, robot_id: int) -> bool:
    with diag_path.open("r") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("type") == "vlm_read"
                    and r.get("frame") == frame
                    and r.get("robot_id") == robot_id):
                return True
    return False


@app.get("/api/diagnostic/{key}/corrections")
async def get_corrections(key: str):
    if not key.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="invalid key")
    p = _resolve_corrections_path(key)
    if not p.exists():
        return []
    # Last-write-wins per (frame, robot_id).
    by_key: dict[tuple, dict] = {}
    with p.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vr = rec.get("vlm_read") or {}
            by_key[(vr.get("frame"), vr.get("robot_id"))] = rec
    return list(by_key.values())


import re

_TEAM_RE = re.compile(r"^\d{1,4}$")
NOTE_MAX_LEN = 500


@app.post("/api/diagnostic/{key}/corrections")
async def post_correction(key: str, body: dict):
    if not key.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="invalid key")
    vr = body.get("vlm_read") or {}
    frame = vr.get("frame")
    robot_id = vr.get("robot_id")
    if not isinstance(frame, int) or not isinstance(robot_id, int):
        raise HTTPException(status_code=400, detail="vlm_read.frame and .robot_id must be ints")
    verdict = body.get("verdict")
    if verdict not in ("correct", "misidentified", "unreadable", "off_roster"):
        raise HTTPException(status_code=400, detail="invalid verdict")

    actual_team = body.get("actual_team")
    if verdict in ("misidentified", "off_roster"):
        if not isinstance(actual_team, str) or not _TEAM_RE.match(actual_team):
            raise HTTPException(status_code=400,
                                detail="actual_team must be a 1–4 digit string for this verdict")
    else:
        actual_team = None  # ignore any value for "correct"/"unreadable"

    note = body.get("note") or ""
    if not isinstance(note, str):
        raise HTTPException(status_code=400, detail="note must be a string")
    note = note[:NOTE_MAX_LEN]

    diag = _resolve_diagnostic_path(key)
    if diag is None or not _vlm_read_exists(diag, frame, robot_id):
        raise HTTPException(status_code=400,
                            detail=f"no vlm_read at frame={frame}, robot_id={robot_id}")

    record = {
        "t": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "author": "local",
        "vlm_read": {"frame": frame, "robot_id": robot_id},
        "verdict": verdict,
        "actual_team": actual_team,
        "note": note,
    }
    p = _resolve_corrections_path(key)
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
```

- [ ] **Step 4: Run the tests to verify they pass**

```
.venv/bin/python -m pytest tests/test_diagnostic_corrections.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Hook the match-save flow to rename `diagnostic_<job_id>.jsonl` → `diagnostic_match_<match_id>.jsonl`**

The existing `save_match_results` endpoint at `web/app.py:2855` (`POST /api/match/save`) identifies the match by `match_key` and currently has no awareness of `job_id`. Wire it like this:

**5a. Extend the request body to carry `job_id`.** In `save_match_results`, right after `match_id = match['id']` is resolved (~line 2870), add:

```python
    # If the client tells us which inference job produced this match, rename
    # the diagnostic + corrections JSONLs from job-keyed to match-keyed so they
    # become a stable artifact of the saved match.
    job_id = data.get("job_id")
    if job_id and str(job_id).replace("_", "").replace("-", "").isalnum():
        for prefix in ("diagnostic_", "corrections_"):
            src = UPLOAD_DIR / f"{prefix}{job_id}.jsonl"
            if src.exists():
                dst = UPLOAD_DIR / f"{prefix}match_{match_id}.jsonl"
                try:
                    src.rename(dst)
                except OSError as e:
                    print(f"warning: failed to rename {src} -> {dst}: {e}")
```

**5b. Update the frontend save call.** The existing save flow (Match History edit modal in `web/static/app.js` — search for `/api/match/save`) sends a body without `job_id`. Add it:

```javascript
const body = {
    match_key: matchKey,
    job_id: window.currentJobId || null,   // set in app.js when WS opens
    // ... existing fields ...
};
```

`window.currentJobId` is the value assigned in Task 11 Step 2 when the WS attaches. If no job is active (editing an old match), `job_id` is null and the rename block above no-ops — old matches are unaffected.

Until a match is saved, diagnostic files live as `diagnostic_<job_id>.jsonl` and are reachable via the source dropdown's "Live" entry. After save, they're stably keyed by match.

- [ ] **Step 6: Verify import + full suite**

```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green.

- [ ] **Step 7: Commit**

```bash
git add web/app.py tests/test_diagnostic_corrections.py
git commit -m "$(cat <<'EOF'
feat(app): corrections GET/POST + rename JSONLs on match save

Corrections persist to corrections_<job_id>.jsonl alongside the
diagnostic file. POST validates that the (frame, robot_id) refers
to a real vlm_read. GET returns last-write-wins per key. On match
save, both files are renamed to the match_<id> form.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Strip the Tracker view + rename nav

**Files:**
- Modify: `web/static/index.html`
- Modify: `web/static/app.js`
- Modify: `web/static/style.css`

No automated tests — verify by loading the app in a browser.

- [ ] **Step 1: Update the nav**

In `web/static/index.html`, replace the `<nav id="mainNav">` block with:

```html
<nav id="mainNav">
    <button class="nav-btn active" data-view="tracker">Tracker</button>
    <button class="nav-btn" data-view="diagnostic">Diagnostic</button>
    <button class="nav-btn" data-view="history">Match History</button>
</nav>
```

Also change the `<h1>` from `FRC Vision Scorer` to `FRC Vision Tracker`.

Rename `<div id="scorerView">` to `<div id="trackerView">`. Update the corresponding selectors in `app.js`.

- [ ] **Step 2: Remove scoring chrome from the HUD**

In the HUD overlay block (`<div class="hud-overlay">`), drop the `.hud-scores` element entirely. Leave only the `.hud-left` block (Frame + FPS).

- [ ] **Step 3: Replace the alliance-column block with a Roster panel**

Replace the entire `<div class="alliance-column">` block (the two `.alliance-card` tables + the `.timeline-card`) with:

```html
<div class="roster-column">
    <div class="roster-card">
        <h3>Roster</h3>
        <div class="roster-alliance blue">
            <div class="roster-alliance-label">BLUE</div>
            <ul id="blueRosterList" class="roster-team-list"></ul>
        </div>
        <div class="roster-alliance red">
            <div class="roster-alliance-label">RED</div>
            <ul id="redRosterList" class="roster-team-list"></ul>
        </div>
    </div>
</div>
```

- [ ] **Step 4: Remove the bumper-crops panel**

Delete `<div class="ocr-crops-panel">...</div>` from the Tracker view. (The same data is rendered on the Diagnostic page in later tasks.)

- [ ] **Step 5: Remove the post-inference Results section**

Delete the entire `<section class="results-section" id="resultsSection">...</section>` block.

- [ ] **Step 6: Update `web/static/app.js`**

In `app.js`, find the WebSocket message handler (`ws.onmessage` or similar). Add an early-return for diagnostic message types (Tracker view doesn't render them):

```javascript
if (data.type === "diagnostic_event" || data.type === "diagnostic_snapshot") {
    return;  // handled by the Diagnostic page only
}
```

Replace the existing `updateRobotsTable(...)` / scoring-table rendering with a new `renderRoster(snapshot)` that reads `snapshot.roster` and `snapshot.identified` from the snapshot stream and renders the team chips. Driven off `diagnostic_snapshot` is wrong here since Tracker ignores those — instead, derive the roster from the existing `match_blue_teams` / `match_red_teams` (loaded at match select) and the `robots` array in the `frame` message.

Roster rendering function (vanilla):

```javascript
function renderRoster(blueTeams, redTeams, robots) {
    function statusFor(team) {
        const r = robots.find(x => x.team_number === team);
        if (!r) return "unread";
        return r.team_number_confirmed ? "confirmed" : "tentative";
    }
    function render(listEl, teams) {
        listEl.innerHTML = teams.map(t => {
            const s = statusFor(t);
            const icon = { confirmed: "●", tentative: "◐", unread: "○" }[s];
            return `<li class="roster-team ${s}"><span class="ico">${icon}</span>${t}</li>`;
        }).join("");
    }
    render(document.getElementById("blueRosterList"), blueTeams);
    render(document.getElementById("redRosterList"), redTeams);
}
```

Call `renderRoster(matchBlueTeams, matchRedTeams, data.robots)` from the `frame` message handler. `robots` already carries `team_number` and `team_number_confirmed` per the existing payload (check the schema; if `team_number_confirmed` isn't streamed yet, extend the frame payload to include it).

Drop any code that updates `scoreBlue`, `scoreRed`, `blueAllianceScore`, `redAllianceScore`, the scoring-events timeline, or the results-edit tables. Leave functions that handle event/match setup, OCR setup, and the upload flow alone.

- [ ] **Step 7: Add roster styles to `web/static/style.css`**

Append:

```css
.roster-card { background: var(--card-bg, #1a1a1f); padding: 16px; border-radius: 10px; }
.roster-card h3 { margin: 0 0 12px; font-size: 14px; letter-spacing: .08em; opacity: .8; }
.roster-alliance { margin-bottom: 16px; }
.roster-alliance-label { font-size: 11px; font-weight: 600; letter-spacing: .12em; margin-bottom: 6px; }
.roster-alliance.blue .roster-alliance-label { color: #4a8df0; }
.roster-alliance.red  .roster-alliance-label { color: #f06a6a; }
.roster-team-list { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 6px; }
.roster-team { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px;
               border-radius: 16px; background: rgba(255,255,255,0.04); font-size: 13px;
               border: 1px solid rgba(255,255,255,0.06); }
.roster-team.confirmed { background: rgba(74,141,240,0.15); border-color: rgba(74,141,240,0.4); }
.roster-alliance.red .roster-team.confirmed { background: rgba(240,106,106,0.15); border-color: rgba(240,106,106,0.4); }
.roster-team.tentative { opacity: 0.85; }
.roster-team.unread { opacity: 0.5; }
```

- [ ] **Step 8: Manual verification**

Start the app and open the Tracker view. Confirm:
- Nav reads Tracker / Diagnostic / Match History.
- No score chips in the HUD.
- No bumper-crops panel under the video.
- No scoring-events timeline.
- A Roster panel appears on the right with team chips for both alliances.
- Loading an event and running a short inference still works end-to-end; scoring still happens in the backend (visible in logs) but is not surfaced anywhere on the page.

- [ ] **Step 9: Commit**

```bash
git add web/static/index.html web/static/app.js web/static/style.css
git commit -m "$(cat <<'EOF'
feat(ui): strip Tracker view of scoring chrome + add Roster panel

Renames Scorer→Tracker, removes score chips, the bumper-crops panel, the
scoring-events timeline, and the post-match results section. Adds a
compact Roster panel showing identification status per team.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Diagnostic page shell + Pipeline tab

**Files:**
- Modify: `web/static/index.html` (add the Diagnostic view section)
- Create: `web/static/diagnostic.js`
- Modify: `web/static/style.css` (extend with diagnostic styles)
- Modify: `web/static/app.js` (route `data-view="diagnostic"` clicks)

- [ ] **Step 1: Add the Diagnostic view markup**

In `index.html`, after the `</div>` that closes `trackerView` and before `historyView`, insert:

```html
<div id="diagnosticView" style="display:none;">
    <section class="diag-section">
        <div class="diag-toolbar">
            <div class="diag-tabs">
                <button class="diag-tab active" data-tab="pipeline">Pipeline</button>
                <button class="diag-tab" data-tab="reads">Reads</button>
            </div>
            <div class="diag-source">
                <label>Source:</label>
                <select id="diagSourceSelect">
                    <option value="live">Live (current job)</option>
                </select>
            </div>
            <div class="diag-filters" id="diagFilters">
                <label><input type="checkbox" value="vlm_read" checked>vlm_read</label>
                <label><input type="checkbox" value="vlm_filtered" checked>vlm_filtered</label>
                <label><input type="checkbox" value="assignment" checked>assignment</label>
                <label><input type="checkbox" value="elimination" checked>elimination</label>
                <label><input type="checkbox" value="overturn" checked>overturn</label>
                <label><input type="checkbox" value="reid_match" checked>reid_match</label>
                <label><input type="checkbox" value="reid_drop" checked>reid_drop</label>
            </div>
        </div>

        <div id="diagPipelinePane" class="diag-pane active">
            <div class="diag-pipeline-grid">
                <div class="diag-state-column">
                    <div class="diag-video-card">
                        <canvas id="diagVideoCanvas"></canvas>
                        <div class="diag-video-frame" id="diagVideoFrame">Frame —</div>
                    </div>
                    <div class="diag-cards" id="diagIdentityCards"></div>
                    <div class="diag-ocr-strip" id="diagOcrStrip"></div>
                    <div class="diag-reid-panel" id="diagReidPanel"></div>
                </div>
                <div class="diag-log" id="diagLog"></div>
            </div>
        </div>

        <div id="diagReadsPane" class="diag-pane">
            <!-- Populated in Task 12 -->
        </div>
    </section>
</div>
```

- [ ] **Step 2: Wire nav routing in `app.js`**

In the existing nav handler that swaps views by `data-view`, add a branch for `"diagnostic"`:

```javascript
document.getElementById("trackerView").style.display = view === "tracker" ? "" : "none";
document.getElementById("diagnosticView").style.display = view === "diagnostic" ? "" : "none";
document.getElementById("historyView").style.display = view === "history" ? "" : "none";

if (view === "diagnostic" && !window.__diagInited) {
    window.__diagInited = true;
    window.diagInit && window.diagInit();
}
```

Add the script tag in `index.html` before `app.js`:

```html
<script src="/static/diagnostic.js?v=20260527a"></script>
<script src="/static/app.js?v=20260527a"></script>
```

- [ ] **Step 3: Implement `diagnostic.js` — state + WS handlers + Pipeline rendering**

Create `web/static/diagnostic.js`:

```javascript
(function () {
    "use strict";

    const state = {
        events: [],            // newest-first; capped at 5000
        latestSnapshot: null,
        filterTypes: new Set([
            "vlm_read", "vlm_filtered", "assignment", "elimination",
            "overturn", "reid_match", "reid_drop",
        ]),
        currentTab: "pipeline",
        replayMode: false,
        replayKey: null,
    };

    const MAX_LOG = 5000;

    // HTML-escape any value that flows into innerHTML or a quoted attribute.
    // The diagnostic JSONL contains VLM raw_read text and user-entered
    // correction notes — both untrusted. Use esc() on EVERY interpolation
    // that isn't a hard-coded literal or a number we cast ourselves.
    function esc(s) {
        if (s === null || s === undefined) return "";
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function fmtFrame(n) { return "f" + Number(n); }

    function appendEvent(rec) {
        state.events.unshift(rec);
        if (state.events.length > MAX_LOG) state.events.length = MAX_LOG;
        renderLog();
        renderOcrStrip();
        renderReidPanel();
    }

    function setSnapshot(rec) {
        state.latestSnapshot = rec;
        renderIdentityCards();
    }

    // -------- Render functions --------

    function renderLog() {
        const el = document.getElementById("diagLog");
        if (!el) return;
        const rows = state.events
            .filter(e => state.filterTypes.has(e.type))
            .slice(0, 500)
            .map(rowForEvent)
            .join("");
        el.innerHTML = rows || '<div class="diag-empty">No events yet</div>';
    }

    function rowForEvent(e) {
        // Build the human-readable body. esc() every field that could carry
        // untrusted text (raw_read, reason, team strings — JSONL on disk
        // could be tampered with). Numbers we format ourselves are safe.
        let body = "";
        switch (e.type) {
            case "vlm_read":
                body = `R${Number(e.robot_id)} read "${esc(e.raw_read ?? "?")}" → ${esc(e.matched_team ?? "—")} ` +
                       `(leg ${(e.legibility ?? 0).toFixed(2)}, w ${Number(e.vote_weight ?? 0)})`;
                break;
            case "vlm_filtered":
                body = `R${Number(e.robot_id)} filtered (${esc(e.reason ?? "?")}, leg ${(e.legibility ?? 0).toFixed(3)})`;
                break;
            case "assignment":
                body = `R${Number(e.robot_id)} → ${esc(e.team)} ${e.confirmed ? "✓" : "tentative"} ` +
                       `(${Number(e.votes_for_assigned)}/${Number(e.votes_total)}, ` +
                       `${(Number(e.top_team_share) * 100).toFixed(0)}%)`;
                break;
            case "elimination":
                body = `R${Number(e.robot_id)} ← ${esc(e.team)} (last on ${esc(e.alliance)})`;
                break;
            case "overturn":
                body = `R${Number(e.robot_id)} unlocked: ${esc(e.old_team)} (${Number(e.current_votes)}) ` +
                       `vs ${esc(e.challenger_team)} (${Number(e.challenger_votes)})`;
                break;
            case "reid_match":
                body = `T${Number(e.track_id)} → C${Number(e.canonical_id)} ` +
                       `sim ${Number(e.similarity).toFixed(2)} gap ${(e.gap ?? 0).toFixed(2)}`;
                break;
            case "reid_drop":
                body = `T${Number(e.track_id)} dropped (${esc(e.reason)}) ` +
                       `would-match C${e.would_have_matched ?? "—"}`;
                break;
            default:
                body = esc(JSON.stringify(e));
        }
        return `<div class="diag-log-row diag-type-${esc(e.type)}">
                  <span class="diag-log-frame">${fmtFrame(e.frame)}</span>
                  <span class="diag-log-type">${esc(e.type)}</span>
                  <span class="diag-log-body">${body}</span>
                </div>`;
    }

    function renderIdentityCards() {
        const el = document.getElementById("diagIdentityCards");
        if (!el || !state.latestSnapshot) {
            if (el) el.innerHTML = '<div class="diag-empty">No snapshot yet</div>';
            return;
        }
        const robots = state.latestSnapshot.robots || [];
        el.innerHTML = robots.map(cardForRobot).join("");
    }

    function cardForRobot(r) {
        const stale = (r.embedding?.age_frames ?? 0) > 60;
        const voteEntries = Object.entries(r.votes || {})
            .sort((a, b) => b[1] - a[1]).slice(0, 3);
        const maxV = voteEntries[0]?.[1] || 1;
        const bars = voteEntries.map(([t, v]) =>
            `<div class="diag-vote-row">
               <span class="diag-vote-team">${esc(t)}</span>
               <span class="diag-vote-bar" style="--w:${(Number(v)/maxV*100).toFixed(0)}%"></span>
               <span class="diag-vote-val">${Number(v).toFixed(1)}</span>
             </div>`
        ).join("");
        // alliance is constrained to "red"/"blue"/"unknown" by the backend, but
        // the JSONL could be tampered with — esc() it before using as a class.
        const allianceClass = esc(r.alliance || "unknown");
        return `<div class="diag-card ${allianceClass} ${stale ? "stale" : ""}">
                  <div class="diag-card-head">
                    <span class="diag-card-id">R${Number(r.robot_id)}</span>
                    <span class="diag-card-alliance ${allianceClass}">${esc((r.alliance||"?")).toUpperCase()}</span>
                    <span class="diag-card-team">${esc(r.team ?? "?")}</span>
                    <span class="diag-card-state">${r.confirmed ? "✓" : "?"}</span>
                  </div>
                  <div class="diag-card-votes">${bars || '<em>no reads</em>'}</div>
                  <div class="diag-card-foot">
                    embed age ${Number(r.embedding?.age_frames ?? 0)}
                    · bumper R${Number(r.red_votes ?? 0)}/B${Number(r.blue_votes ?? 0)}
                  </div>
                </div>`;
    }

    function renderOcrStrip() {
        const el = document.getElementById("diagOcrStrip");
        if (!el) return;
        const crops = state.events
            .filter(e => e.type === "vlm_read" || e.type === "vlm_filtered")
            .slice(0, 12);
        el.innerHTML = crops.map(cropTile).join("") ||
            '<div class="diag-empty">No reads yet</div>';
    }

    // Base64 must be a strict alphabet — if it isn't, drop it. Prevents the
    // data: URL from absorbing attribute-breakout characters.
    function safeB64(b) {
        if (typeof b !== "string") return null;
        return /^[A-Za-z0-9+/=\s]+$/.test(b) ? b : null;
    }

    function cropTile(e) {
        const labelRaw = e.type === "vlm_filtered"
            ? `filt: ${e.reason ?? ""}`
            : (e.matched_team ?? "—");
        const matchClass = e.type === "vlm_filtered" ? "filtered"
                         : (e.matched_team ? "matched" : "unmatched");
        const b64 = safeB64(e.crop_b64);
        const img = b64
            ? `<img src="data:image/jpeg;base64,${b64}" alt="">`
            : '<div class="diag-crop-placeholder">no image</div>';
        return `<div class="diag-crop ${matchClass}">
                  ${img}
                  <div class="diag-crop-meta">
                    R${Number(e.robot_id)} · ${esc(labelRaw)}
                    <span class="diag-crop-leg">leg ${(e.legibility ?? 0).toFixed(2)}</span>
                  </div>
                </div>`;
    }

    function renderReidPanel() {
        const el = document.getElementById("diagReidPanel");
        if (!el) return;
        const recent = state.events
            .filter(e => e.type === "reid_match" || e.type === "reid_drop")
            .slice(0, 8);
        el.innerHTML = recent.map(e => {
            if (e.type === "reid_match") {
                return `<div class="diag-reid-row match">
                          T${Number(e.track_id)} → C${Number(e.canonical_id)}
                          sim ${Number(e.similarity).toFixed(2)} gap ${(e.gap ?? 0).toFixed(2)}
                        </div>`;
            }
            return `<div class="diag-reid-row drop">
                      T${Number(e.track_id)} dropped (${esc(e.reason)})
                      would-match C${e.would_have_matched ?? "—"}
                      sim ${(e.similarity ?? 0).toFixed(2)}
                    </div>`;
        }).join("") || '<div class="diag-empty">No re-ID activity yet</div>';
    }

    // -------- WS hook (live mode shares the existing connection) --------

    function onDiagMessage(data) {
        if (data.type === "diagnostic_event") appendEvent(data.record);
        else if (data.type === "diagnostic_snapshot") setSnapshot(data.record);
    }

    // -------- Tab switching + filters + init --------

    function bindTabs() {
        document.querySelectorAll(".diag-tab").forEach(b =>
            b.addEventListener("click", () => {
                document.querySelectorAll(".diag-tab").forEach(x => x.classList.remove("active"));
                b.classList.add("active");
                state.currentTab = b.dataset.tab;
                document.getElementById("diagPipelinePane").classList.toggle("active", state.currentTab === "pipeline");
                document.getElementById("diagReadsPane").classList.toggle("active", state.currentTab === "reads");
                if (state.currentTab === "reads") window.diagRenderReads && window.diagRenderReads();
            })
        );
        document.querySelectorAll("#diagFilters input").forEach(cb =>
            cb.addEventListener("change", () => {
                if (cb.checked) state.filterTypes.add(cb.value);
                else state.filterTypes.delete(cb.value);
                renderLog();
            })
        );
    }

    window.diagInit = function () {
        bindTabs();
        renderLog();
        renderIdentityCards();
        renderOcrStrip();
        renderReidPanel();
    };
    window.diagOnWsMessage = onDiagMessage;
    window.diagState = state;
})();
```

In `app.js`'s existing WebSocket message handler, after the early return for diagnostic messages, instead of returning, dispatch to the diagnostic module:

```javascript
if (data.type === "diagnostic_event" || data.type === "diagnostic_snapshot") {
    if (window.diagOnWsMessage) window.diagOnWsMessage(data);
    return;
}
```

- [ ] **Step 4: Add diagnostic styles**

Append to `web/static/style.css`:

```css
/* Diagnostic page */
.diag-section { padding: 16px; }
.diag-toolbar { display: flex; gap: 16px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
.diag-tabs { display: flex; gap: 4px; }
.diag-tab { padding: 6px 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 6px; color: inherit; cursor: pointer; }
.diag-tab.active { background: rgba(74,141,240,0.2); border-color: rgba(74,141,240,0.5); }
.diag-source { display: flex; align-items: center; gap: 6px; }
.diag-filters { display: flex; gap: 8px; flex-wrap: wrap; font-size: 12px; opacity: 0.85; }
.diag-filters label { display: inline-flex; gap: 4px; align-items: center; }

.diag-pane { display: none; }
.diag-pane.active { display: block; }

.diag-pipeline-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; }

.diag-video-card { position: relative; }
.diag-video-card canvas { width: 100%; height: auto; background: #000; border-radius: 6px; }
.diag-video-frame { position: absolute; top: 8px; left: 8px; padding: 2px 8px;
                    background: rgba(0,0,0,0.6); border-radius: 4px; font-size: 12px; }

.diag-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
              gap: 8px; margin-top: 12px; }
.diag-card { padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.04);
             border: 1px solid rgba(255,255,255,0.08); font-size: 12px; }
.diag-card.red { border-color: rgba(240,106,106,0.4); }
.diag-card.blue { border-color: rgba(74,141,240,0.4); }
.diag-card.stale { border-color: rgba(255,180,0,0.6); box-shadow: 0 0 0 1px rgba(255,180,0,0.3) inset; }
.diag-card-head { display: flex; justify-content: space-between; align-items: center;
                  gap: 8px; margin-bottom: 6px; }
.diag-card-team { font-weight: 700; font-size: 14px; }
.diag-card-alliance.red  { color: #f06a6a; font-size: 10px; letter-spacing: .1em; }
.diag-card-alliance.blue { color: #4a8df0; font-size: 10px; letter-spacing: .1em; }
.diag-card-votes { display: flex; flex-direction: column; gap: 2px; margin: 4px 0; }
.diag-vote-row { display: grid; grid-template-columns: 48px 1fr 36px; gap: 6px; align-items: center; }
.diag-vote-bar { display: block; height: 6px; border-radius: 3px;
                 background: linear-gradient(90deg, currentColor var(--w, 0%), transparent var(--w, 0%));
                 color: rgba(74,141,240,0.6); }
.diag-card.red .diag-vote-bar { color: rgba(240,106,106,0.6); }
.diag-card-foot { opacity: 0.7; margin-top: 4px; font-size: 11px; }

.diag-ocr-strip { display: flex; gap: 8px; overflow-x: auto; padding: 8px 0; margin-top: 12px; }
.diag-crop { flex: 0 0 auto; width: 140px; background: rgba(255,255,255,0.04);
             border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 6px; }
.diag-crop.matched   { border-color: rgba(140, 220, 140, 0.4); }
.diag-crop.unmatched { border-color: rgba(255, 180, 0, 0.4); }
.diag-crop.filtered  { opacity: 0.6; }
.diag-crop img { width: 100%; height: 60px; object-fit: cover; border-radius: 4px; background: #000; }
.diag-crop-placeholder { height: 60px; display: flex; align-items: center; justify-content: center;
                         background: rgba(255,255,255,0.06); border-radius: 4px; font-size: 10px;
                         opacity: 0.6; }
.diag-crop-meta { font-size: 11px; margin-top: 4px; display: flex; justify-content: space-between; }
.diag-crop-leg { opacity: 0.7; }

.diag-reid-panel { display: flex; flex-direction: column; gap: 4px; margin-top: 12px; font-size: 12px; }
.diag-reid-row { padding: 4px 8px; border-radius: 4px; background: rgba(255,255,255,0.04); }
.diag-reid-row.drop { color: #f0a06a; }

.diag-log { max-height: 80vh; overflow-y: auto; background: rgba(0,0,0,0.3); border-radius: 6px; padding: 6px; }
.diag-log-row { display: grid; grid-template-columns: 50px 90px 1fr; gap: 8px;
                padding: 3px 6px; font-family: ui-monospace, monospace; font-size: 12px; }
.diag-log-row + .diag-log-row { border-top: 1px solid rgba(255,255,255,0.04); }
.diag-log-frame { opacity: 0.6; }
.diag-log-type { font-weight: 600; }
.diag-type-overturn   .diag-log-type { color: #f0a06a; }
.diag-type-elimination .diag-log-type { color: #f6d36a; }
.diag-type-assignment .diag-log-type { color: #8cc8ff; }
.diag-type-vlm_read   .diag-log-type { color: #8cd99a; }
.diag-type-vlm_filtered .diag-log-type { color: #c8c8c8; opacity: 0.7; }
.diag-type-reid_drop  .diag-log-type { color: #f06a6a; }

.diag-empty { padding: 10px; opacity: 0.6; font-style: italic; font-size: 12px; }
```

- [ ] **Step 5: Manual verification**

Start the app. Load an event, pick a match, run inference. Switch to the Diagnostic tab. Confirm:
- Pipeline pane is active by default.
- Decision log fills with events as inference runs.
- Identity cards appear after the first snapshot tick.
- OCR strip fills as Qwen reads happen.
- Re-ID panel shows recent matches/drops.

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html web/static/diagnostic.js web/static/style.css web/static/app.js
git commit -m "$(cat <<'EOF'
feat(ui): diagnostic page shell + Pipeline tab

Adds the Diagnostic top-level page with the Pipeline tab populated:
decision log, robot identity cards, OCR strip, re-ID continuity panel.
Live mode rides the existing WS multiplex.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Reads tab + correction editor + replay mode

**Files:**
- Modify: `web/static/diagnostic.js` (add Reads tab + correction editor + replay loader)
- Modify: `web/static/style.css` (Reads-tab styles)
- Modify: `web/static/index.html` (Reads pane skeleton)

- [ ] **Step 1: Add the Reads pane skeleton**

In `index.html`, replace the placeholder `<!-- Populated in Task 12 -->` inside `<div id="diagReadsPane">` with:

```html
<div class="diag-reads-summary" id="diagReadsSummary"></div>
<div class="diag-reads-list" id="diagReadsList"></div>
<div class="diag-editor" id="diagEditor" style="display:none;"></div>
```

- [ ] **Step 2: Implement the Reads rendering**

Append to `diagnostic.js` (inside the IIFE, before the closing `})();`):

```javascript
    // -------- Reads tab --------

    let _corrections = {};        // key: "frame:robot_id" -> correction record
    let _readsKey = null;         // current source key (job_id or match_id)

    function loadCorrections(key) {
        return fetch(`/api/diagnostic/${encodeURIComponent(key)}/corrections`)
            .then(r => r.ok ? r.json() : [])
            .then(list => {
                _corrections = {};
                for (const c of list) {
                    const v = c.vlm_read || {};
                    _corrections[`${v.frame}:${v.robot_id}`] = c;
                }
            })
            .catch(() => { _corrections = {}; });
    }

    function rosterTeams() {
        const s = state.latestSnapshot;
        if (!s || !s.roster) return { red: [], blue: [] };
        return { red: s.roster.red || [], blue: s.roster.blue || [] };
    }

    function readsForTeam(team) {
        return state.events
            .filter(e => e.type === "vlm_read" && e.matched_team === team)
            .reverse();
    }

    function unmatchedReads() {
        return state.events
            .filter(e =>
                e.type === "vlm_filtered" ||
                (e.type === "vlm_read" && !e.matched_team)
            )
            .reverse();
    }

    function correctionFor(e) {
        return _corrections[`${e.frame}:${e.robot_id}`];
    }

    window.diagRenderReads = function () {
        const summary = document.getElementById("diagReadsSummary");
        const list = document.getElementById("diagReadsList");
        if (!summary || !list) return;

        const { red, blue } = rosterTeams();
        const totals = [...red, ...blue].reduce((acc, t) => {
            const reads = readsForTeam(t);
            const fixed = reads.filter(e => {
                const c = correctionFor(e);
                return c && c.verdict === "misidentified";
            }).length;
            acc[t] = { total: reads.length, fixed };
            return acc;
        }, {});
        const all = Object.values(totals);
        const totalReads = all.reduce((a, b) => a + b.total, 0);
        const totalFixed = all.reduce((a, b) => a + b.fixed, 0);
        summary.innerHTML =
            `<strong>${totalReads}</strong> reads across roster · ` +
            `<strong>${totalFixed}</strong> labeled misidentified · ` +
            `<strong>${unmatchedReads().length}</strong> unmatched/filtered`;

        // name is always a hard-coded "red"/"blue" — safe. team strings come
        // from the snapshot roster (originally from TBA) — escape on render.
        const renderAlliance = (name, teams) => `
            <h3 class="diag-reads-alliance ${name}">${name.toUpperCase()}</h3>
            ${teams.map(t => renderTeamRow(t)).join("")}
        `;
        const renderTeamRow = team => {
            const reads = readsForTeam(team);
            if (reads.length === 0) {
                return `<div class="diag-reads-team empty">
                          <div class="diag-reads-team-head">Team ${esc(team)}</div>
                          <div class="diag-empty">no reads</div>
                        </div>`;
            }
            const tiles = reads.map(e => readTile(e)).join("");
            return `<div class="diag-reads-team">
                      <div class="diag-reads-team-head">
                        Team ${esc(team)} <span class="diag-reads-count">${reads.length}</span>
                      </div>
                      <div class="diag-reads-tiles">${tiles}</div>
                    </div>`;
        };

        list.innerHTML = renderAlliance("red", red)
            + renderAlliance("blue", blue)
            + `<h3 class="diag-reads-alliance">UNMATCHED / FILTERED</h3>
               <div class="diag-reads-team">
                 <div class="diag-reads-tiles">
                   ${unmatchedReads().map(readTile).join("") || '<div class="diag-empty">none</div>'}
                 </div>
               </div>`;

        list.querySelectorAll("[data-key]").forEach(el =>
            el.addEventListener("click", () => openEditor(el.dataset.key))
        );
    };

    function readTile(e) {
        const c = correctionFor(e);
        let badge = "";
        if (c) {
            if (c.verdict === "correct") badge = `<span class="diag-tile-badge ok">✓</span>`;
            else if (c.verdict === "misidentified")
                badge = `<span class="diag-tile-badge fix">→${esc(c.actual_team ?? "?")}</span>`;
            else if (c.verdict === "unreadable")
                badge = `<span class="diag-tile-badge filt">filt</span>`;
            else if (c.verdict === "off_roster")
                badge = `<span class="diag-tile-badge off">off</span>`;
        }
        // key is built from numeric fields — Number() casts make this safe to
        // embed without escaping. The lookup side parses these back as Numbers.
        const key = `${Number(e.frame)}:${Number(e.robot_id)}`;
        const b64 = safeB64(e.crop_b64);
        const img = b64
            ? `<img src="data:image/jpeg;base64,${b64}" alt="">`
            : '<div class="diag-crop-placeholder">no image</div>';
        return `<div class="diag-tile" data-key="${key}">
                  ${img}${badge}
                  <div class="diag-tile-meta">f${Number(e.frame)} · R${Number(e.robot_id)}</div>
                </div>`;
    }

    function openEditor(key) {
        const [frameStr, robotStr] = key.split(":");
        const frame = Number(frameStr), robot_id = Number(robotStr);
        const e = state.events.find(x =>
            x.frame === frame && x.robot_id === robot_id &&
            (x.type === "vlm_read" || x.type === "vlm_filtered")
        );
        if (!e) return;
        const c = _corrections[key] || {};
        const ed = document.getElementById("diagEditor");
        const { red, blue } = rosterTeams();
        const rosterOpts = [...red, ...blue].map(t =>
            `<option value="${esc(t)}" ${c.actual_team === t ? "selected" : ""}>${esc(t)}</option>`
        ).join("");
        const b64 = safeB64(e.crop_b64);
        // Render the static shell with esc()'d data only. The <input> values
        // for note/off come from prior user input — esc() prevents an attribute
        // breakout via a quote in the saved value.
        ed.innerHTML = `
            <div class="diag-editor-head">
                <strong>Read @ frame ${Number(frame)}, robot R${Number(robot_id)}</strong>
                <button class="diag-editor-close" id="diagEditorClose">×</button>
            </div>
            <div class="diag-editor-body">
                ${b64 ? `<img src="data:image/jpeg;base64,${b64}" alt="">` : ''}
                <div class="diag-editor-info">
                    Raw read: <code>${esc(e.raw_read ?? "?")}</code><br>
                    Matched team: <code>${esc(e.matched_team ?? "—")}</code><br>
                    Legibility: <code>${(e.legibility ?? 0).toFixed(3)}</code>
                </div>
                <div class="diag-editor-form">
                    <label><input type="radio" name="verdict" value="correct"
                                  ${c.verdict === "correct" ? "checked" : ""}> Correct</label>
                    <label><input type="radio" name="verdict" value="misidentified"
                                  ${c.verdict === "misidentified" ? "checked" : ""}>
                           Misidentified, actually:
                           <select id="diagEditorActual">${rosterOpts}</select></label>
                    <label><input type="radio" name="verdict" value="unreadable"
                                  ${c.verdict === "unreadable" ? "checked" : ""}> Unreadable</label>
                    <label><input type="radio" name="verdict" value="off_roster"
                                  ${c.verdict === "off_roster" ? "checked" : ""}>
                           Off-roster — number: <input type="text" id="diagEditorOff"
                                  pattern="\\d{1,4}" value="${esc(c.verdict === 'off_roster' ? (c.actual_team ?? '') : '')}"></label>
                    <label>Note: <input type="text" id="diagEditorNote" value="${esc(c.note ?? '')}"></label>
                    <div class="diag-editor-actions">
                        <button id="diagEditorSave" class="btn-primary">Save</button>
                        <button id="diagEditorCancel" class="btn-secondary">Cancel</button>
                    </div>
                </div>
            </div>`;
        ed.style.display = "";

        document.getElementById("diagEditorClose").onclick =
        document.getElementById("diagEditorCancel").onclick = () => { ed.style.display = "none"; };

        document.getElementById("diagEditorSave").onclick = async () => {
            const verdict = ed.querySelector('input[name="verdict"]:checked')?.value;
            if (!verdict) return;
            let actual_team = null;
            if (verdict === "misidentified") actual_team = document.getElementById("diagEditorActual").value;
            else if (verdict === "off_roster") {
                const v = document.getElementById("diagEditorOff").value.trim();
                if (!/^\d{1,4}$/.test(v)) { alert("Off-roster number must be 1–4 digits"); return; }
                actual_team = v;
            }
            const note = document.getElementById("diagEditorNote").value.trim().slice(0, 500);
            const body = { vlm_read: { frame, robot_id }, verdict, actual_team, note };
            const r = await fetch(`/api/diagnostic/${encodeURIComponent(_readsKey)}/corrections`,
                { method: "POST", headers: { "content-type": "application/json" },
                  body: JSON.stringify(body) });
            if (!r.ok) { alert("save failed: " + r.status); return; }
            await loadCorrections(_readsKey);
            ed.style.display = "none";
            window.diagRenderReads();
        };
    }

    // -------- Replay mode --------

    async function startReplay(key) {
        state.replayMode = true;
        state.replayKey = key;
        _readsKey = key;
        state.events = [];
        state.latestSnapshot = null;
        await loadCorrections(key);
        const r = await fetch(`/api/diagnostic/${encodeURIComponent(key)}`);
        if (!r.ok) return;
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let nl;
            while ((nl = buf.indexOf("\n")) !== -1) {
                const line = buf.slice(0, nl).trim();
                buf = buf.slice(nl + 1);
                if (!line) continue;
                try {
                    const rec = JSON.parse(line);
                    if (rec.type === "snapshot") setSnapshot(rec);
                    else appendEvent(rec);
                } catch { /* skip malformed */ }
            }
        }
        if (state.currentTab === "reads") window.diagRenderReads();
    }

    window.diagStartReplay = startReplay;
```

Then extend the source dropdown (`#diagSourceSelect`) to be populated dynamically by querying `/api/matches` (existing endpoint). Add at the bottom of `diagInit`:

```javascript
        // Populate source dropdown. The existing /api/history/all returns
        // events with a `matches` array on each event; flatten to one list
        // and use match.id as the replay key (matches diagnostic_match_<id>.jsonl).
        const sel = document.getElementById("diagSourceSelect");
        fetch("/api/history/all")
            .then(r => r.ok ? r.json() : [])
            .then(events => {
                const flat = [];
                for (const ev of (events || [])) {
                    for (const m of (ev.matches || [])) {
                        flat.push({ id: m.id, label: m.match_key || `Match ${m.id}` });
                    }
                }
                // Newest first if the API returns time-ordered; otherwise leave as-is.
                for (const m of flat) {
                    const opt = document.createElement("option");
                    opt.value = `match_${m.id}`;
                    opt.textContent = esc(m.label);
                    sel.appendChild(opt);
                }
            });
        sel.addEventListener("change", () => {
            const v = sel.value;
            if (v === "live") {
                state.replayMode = false;
                _readsKey = window.currentJobId || null;
            } else if (v.startsWith("match_")) {
                window.diagStartReplay(v.replace("match_", ""));
            }
        });
        // Live mode: set _readsKey when the Tracker view creates a job.
        window.diagSetLiveKey = (jobId) => { _readsKey = jobId; };
```

In `app.js`, after a job is created (when the WS connection opens), call `window.diagSetLiveKey && window.diagSetLiveKey(job_id);`.

If `/api/matches` doesn't exist with that exact name, find the existing list-matches endpoint and adjust the URL above. The match list returned by Match History page already has this data — reuse that fetch.

- [ ] **Step 3: Add Reads-tab styles**

Append to `style.css`:

```css
.diag-reads-summary { padding: 8px 12px; margin-bottom: 12px; background: rgba(255,255,255,0.04);
                      border-radius: 6px; font-size: 13px; }
.diag-reads-alliance { margin: 16px 0 8px; font-size: 12px; letter-spacing: .12em; }
.diag-reads-alliance.red { color: #f06a6a; }
.diag-reads-alliance.blue { color: #4a8df0; }
.diag-reads-team { background: rgba(255,255,255,0.03); border-radius: 8px;
                   padding: 12px; margin-bottom: 10px; }
.diag-reads-team-head { font-weight: 600; margin-bottom: 8px; display: flex; gap: 8px; align-items: center; }
.diag-reads-count { font-size: 11px; opacity: 0.7; }
.diag-reads-tiles { display: flex; flex-wrap: wrap; gap: 8px; }
.diag-tile { position: relative; width: 120px; cursor: pointer;
             background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
             border-radius: 6px; padding: 4px; transition: border-color .15s; }
.diag-tile:hover { border-color: rgba(74,141,240,0.5); }
.diag-tile img { width: 100%; height: 60px; object-fit: cover; border-radius: 4px; background: #000; }
.diag-tile-badge { position: absolute; top: 6px; right: 6px; padding: 2px 6px; border-radius: 4px;
                   font-size: 10px; font-weight: 700; background: rgba(0,0,0,0.7); }
.diag-tile-badge.ok  { color: #8cd99a; }
.diag-tile-badge.fix { color: #f0a06a; }
.diag-tile-badge.filt { color: #c8c8c8; }
.diag-tile-badge.off { color: #f6d36a; }
.diag-tile-meta { font-size: 11px; opacity: 0.7; margin-top: 4px; }

.diag-editor { position: fixed; top: 10%; left: 50%; transform: translateX(-50%);
               width: 480px; max-width: 90vw; max-height: 80vh; overflow: auto;
               background: #1a1a1f; border-radius: 10px; padding: 16px; z-index: 1000;
               border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
.diag-editor-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.diag-editor-close { background: none; border: none; color: inherit; font-size: 24px; cursor: pointer; }
.diag-editor-body img { width: 100%; max-height: 180px; object-fit: contain;
                        background: #000; border-radius: 4px; margin-bottom: 8px; }
.diag-editor-info { font-size: 12px; opacity: 0.85; margin-bottom: 12px; }
.diag-editor-form { display: flex; flex-direction: column; gap: 6px; font-size: 13px; }
.diag-editor-form input[type="text"], .diag-editor-form select {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    color: inherit; padding: 4px 6px; border-radius: 4px; margin-left: 6px; }
.diag-editor-actions { display: flex; gap: 8px; margin-top: 8px; }
```

- [ ] **Step 4: Manual verification**

Start the app, run a short match, then:
- Switch to Diagnostic → Reads tab. Confirm rows appear per team with read tiles. Unmatched bucket fills with filtered/unmatched reads.
- Click a tile → editor opens. Pick "Misidentified, actually 4322", Save.
- Switch tabs, switch back. Confirm the `→4322` badge persists on that tile.
- Save the match. Open the Diagnostic page in replay mode by picking the saved match from the source dropdown. Confirm the same events and corrections appear.

- [ ] **Step 5: Commit**

```bash
git add web/static/index.html web/static/diagnostic.js web/static/style.css web/static/app.js
git commit -m "$(cat <<'EOF'
feat(ui): Reads tab + correction editor + replay mode

Reads tab groups VLM crops by matched team plus an Unmatched/Filtered
bucket. Click a tile to open an inline editor and label as correct,
misidentified, unreadable, or off-roster. Replay mode streams a saved
match's diagnostic JSONL and renders the same UI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] Full non-model suite green: `.venv/bin/python -m pytest tests/ -m "not model" -v`
- [ ] Acceptance harness still passes on `qual37.mp4` (`Frames with a collision: 0`, `Total ID flips: 0`, `Total robots tracked: 6`).
- [ ] Manual run on `data/videos/qual37.mp4`:
  - Tracker view shows no scoring chrome; Roster panel updates as identifications happen.
  - Diagnostic → Pipeline shows decision log, identity cards updating, OCR strip filling, re-ID activity.
  - Diagnostic → Reads shows team rows + Unmatched bucket; click-to-label persists across tab switches.
  - After save, picking the match from the source dropdown replays the same data.
- [ ] @superpowers:verification-before-completion before claiming done — do not assert "diagnostic mode shipped" until the manual run above passes end-to-end. Unit tests prove the recorder works, not that the UI is usable.

## v1 scope cut: replay frame scrubber

The spec describes a frame scrubber under the video in Replay mode plus log-row clicks that jump the scrubber. **This is deferred to a follow-up.** Replay v1 loads the entire JSONL and renders the final state of all panels; the decision log is scrollable but does not advance by frame. The Reads tab is not affected by this (it's frame-agnostic by design). Adding the scrubber later requires:

1. A `state.cursor` (frame number or null) that all `visibleEvents()` / `visibleSnapshot()` helpers respect.
2. A `<input type="range">` under the video, populated to `max=last_frame` when the JSONL finishes streaming.
3. Log-row `onclick` that sets `state.cursor` and triggers a re-render.

Documenting the omission so a strict spec-vs-implementation diff is unambiguous.

## Open follow-ups (not in this plan)

- Replay frame scrubber (see v1 scope cut above).
- Cross-match misidentification summary view (would live on Match History as a new column or its own subpage).
- "Re-run inference on this match with corrections forced as ground truth" — useful for sanity-checking the rest of the pipeline given perfect OCR.
- Export-as-tarball button on the Reads tab — produces `crops/<verdict>/<actual_team>/<frame>_<robot>.jpg` from corrections + diagnostic file. Trivial to add once the labeling is in use.

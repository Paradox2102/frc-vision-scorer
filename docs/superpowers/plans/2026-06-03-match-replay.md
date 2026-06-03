# Match Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every analyzed match as a replayable artifact (recorded HUD video + per-frame metadata) and add a web library + player to replay it with scrubbing, variable speed, frame-stepping, and jump-to-event.

**Architecture:** Always-on recording. During analysis, each composited HUD frame (output of `_compose_hud_frame`) is written to an H.264 `.mp4` via a new `ReplayRecorder`, alongside a compact `meta.json` sidecar (per-frame scores/robots/field-positions + scoring events). A standalone `replays` SQLite table indexes the artifacts. New REST endpoints list/serve/delete replays (the video endpoint supports HTTP Range for seeking). The frontend gains a library view and a player view that swaps the live WebSocket canvas for a native `<video>` and drives the robot tables, clickable event timeline, and a client-side field map from the metadata synced to `video.currentTime`.

**Tech Stack:** Python 3.12, FastAPI + Starlette, aiosqlite, OpenCV (`cv2.VideoWriter`/`VideoCapture`), vanilla JS frontend, pytest (`.venv/bin/python -m pytest`).

**Spec:** `docs/superpowers/specs/2026-06-03-match-replay-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `web/replay_recorder.py` | `ReplayRecorder` — owns the `.mp4` writer + metadata buffering + finalize (writes `meta.json`, `thumb.jpg`); knows nothing of FastAPI/DB | Create |
| `web/database.py` | `replays` table in `SCHEMA` + CRUD (`insert_replay`, `list_replays`, `get_replay`, `delete_replay`) | Modify |
| `web/app.py` | `REPLAYS_DIR` const; recording hook in `InferenceJob.process_video`; finalize + DB upsert; REST endpoints | Modify |
| `web/static/app.js` | Library view + `ReplayPlayer`; refactor `updateTimeline` to full-list + clickable | Modify |
| `web/static/index.html`, `web/static/style.css` | Replays nav, library grid, player containers, results "Replay" button | Modify |
| `tests/test_replay_recorder.py` | Recorder unit tests | Create |
| `tests/test_replay_db.py` | `replays` table CRUD tests | Create |
| `tests/test_replay_api.py` | Endpoint tests incl. Range/206 | Create |

**Conventions to follow:**
- Tests run with `.venv/bin/python -m pytest` (base conda python lacks pytest).
- DB functions follow the existing `async with aiosqlite.connect(DB_PATH)` per-call pattern in `web/database.py`; tests monkeypatch `database.DB_PATH` to a temp file.
- Endpoints follow the existing `@app.get(...)`/`StreamingResponse` patterns in `web/app.py` (a `StreamingResponse` chunked-file reader already exists at the diagnostic endpoint ~`web/app.py:4840`).
- Imports inside `web/` modules are bare (`from database import ...`, `import database`), matching the existing app (it runs with `web/` on `sys.path`).

---

## Task 1: `replays` table + DB CRUD

**Files:**
- Modify: `web/database.py` (add to `SCHEMA` string ~line 72; add functions after `get_scored_matches`)
- Test: `tests/test_replay_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay_db.py
import asyncio
import importlib
import json
import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import database
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _sample_row(replay_id="job-1"):
    return dict(
        id=replay_id, created_at="2026-06-03T00:00:00", label="Qual 10",
        event_key=None, match_key=None, comp_level=None,
        blue_score=70, red_score=67,
        blue_teams=json.dumps([581, 9785, 573]), red_teams=json.dumps([1732, 4698, 3297]),
        video_path=f"web/replays/{replay_id}/hud.mp4",
        meta_path=f"web/replays/{replay_id}/meta.json",
        source_video_path="web/uploads/x.mp4",
        thumb_path=f"web/replays/{replay_id}/thumb.jpg",
        fps=8.0, frame_count=1583, duration_sec=197.9,
        width=1280, height=1020, codec="avc1", status="complete",
    )


def test_insert_list_get_delete_replay(db):
    asyncio.run(db.insert_replay(_sample_row("job-1")))
    asyncio.run(db.insert_replay(_sample_row("job-2")))

    rows = asyncio.run(db.list_replays())
    assert [r["id"] for r in rows] == ["job-2", "job-1"]  # newest first by created_at desc, id tiebreak

    one = asyncio.run(db.get_replay("job-1"))
    assert one["label"] == "Qual 10"
    assert one["blue_score"] == 70

    asyncio.run(db.delete_replay("job-1"))
    assert asyncio.run(db.get_replay("job-1")) is None
    assert [r["id"] for r in asyncio.run(db.list_replays())] == ["job-2"]


def test_insert_replay_is_idempotent_upsert(db):
    asyncio.run(db.insert_replay(_sample_row("job-1")))
    row = _sample_row("job-1")
    row["status"] = "partial"
    asyncio.run(db.insert_replay(row))  # re-insert same id
    got = asyncio.run(db.get_replay("job-1"))
    assert got["status"] == "partial"
    assert len(asyncio.run(db.list_replays())) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_db.py -v`
Expected: FAIL with `AttributeError: module 'database' has no attribute 'insert_replay'`

- [ ] **Step 3: Add the table to `SCHEMA`**

In `web/database.py`, append inside the `SCHEMA` triple-quoted string (after the `scoring_events` table, before the closing `"""`):

```sql

CREATE TABLE IF NOT EXISTS replays (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    label TEXT,
    event_key TEXT,
    match_key TEXT,
    comp_level TEXT,
    blue_score INTEGER DEFAULT 0,
    red_score INTEGER DEFAULT 0,
    blue_teams TEXT,
    red_teams TEXT,
    video_path TEXT,
    meta_path TEXT,
    source_video_path TEXT,
    thumb_path TEXT,
    fps REAL,
    frame_count INTEGER,
    duration_sec REAL,
    width INTEGER,
    height INTEGER,
    codec TEXT,
    status TEXT
);
```

- [ ] **Step 4: Add CRUD functions**

Append to `web/database.py`:

```python
# --- Replays ---

_REPLAY_COLUMNS = [
    "id", "created_at", "label", "event_key", "match_key", "comp_level",
    "blue_score", "red_score", "blue_teams", "red_teams",
    "video_path", "meta_path", "source_video_path", "thumb_path",
    "fps", "frame_count", "duration_sec", "width", "height", "codec", "status",
]


async def insert_replay(row: Dict[str, Any]) -> None:
    """Insert or replace a replay row (keyed by id)."""
    cols = ", ".join(_REPLAY_COLUMNS)
    placeholders = ", ".join("?" for _ in _REPLAY_COLUMNS)
    values = [row.get(c) for c in _REPLAY_COLUMNS]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"INSERT OR REPLACE INTO replays ({cols}) VALUES ({placeholders})", values)
        await db.commit()


async def list_replays() -> List[Dict]:
    """All replays, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM replays ORDER BY created_at DESC, id DESC")
        return [dict(r) for r in await cursor.fetchall()]


async def get_replay(replay_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM replays WHERE id = ?", (replay_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def delete_replay(replay_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM replays WHERE id = ?", (replay_id,))
        await db.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_replay_db.py -v`
Expected: PASS (both tests)

- [ ] **Step 6: Commit**

```bash
git add web/database.py tests/test_replay_db.py
git commit -m "feat(replay): add replays table and CRUD"
```

---

## Task 2: `ReplayRecorder` — video + metadata core

**Files:**
- Create: `web/replay_recorder.py`
- Test: `tests/test_replay_recorder.py`

**Design notes:**
- `open_lazy(first_frame, fps, out_dir)` opens the writer using `first_frame.shape`. Try fourcc `avc1`; if the writer fails to open (`isOpened()` is False), fall back to `mp4v`. Store the chosen fourcc string and `(width, height)`.
- `add_frame(frame, record)`: if not yet open, raise (caller must `open_lazy` first); if `frame.shape[:2] != (height, width)` resize to the locked size; write; append `record` (a plain dict) to `self.frames` after stamping `i` (the recorded index) and `t = i / fps`.
- `finalize(status, summary)`: release writer; pick a mid frame as `thumb.jpg` (re-read from the written mp4 via `cv2.VideoCapture` at `frame_count // 2`, or cache a mid frame during `add_frame`); stamp event `t` (Task 3); write `meta.json`; return a dict of the artifact fields needed for the DB row (`video_path`, `meta_path`, `thumb_path`, `fps`, `frame_count`, `duration_sec`, `width`, `height`, `codec`, `status`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay_recorder.py
import json
import numpy as np
import cv2
import pytest

from replay_recorder import ReplayRecorder


def _frame(w=320, h=240, val=50):
    return np.full((h, w, 3), val, np.uint8)


def test_records_playable_video_and_meta(tmp_path):
    rec = ReplayRecorder()
    rec.open_lazy(_frame(), fps=8.0, out_dir=tmp_path)
    for i in range(10):
        rec.add_frame(_frame(val=10 + i), {"scores": {"blue": i, "red": 0}, "robots": []})
    artifacts = rec.finalize(status="complete", summary={"match_label": "Test"})

    # mp4 is playable and has the right frame count
    cap = cv2.VideoCapture(str(tmp_path / "hud.mp4"))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()

    # meta.json is valid with per-frame records and the sync key
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["summary"]["frame_count"] == 10
    assert meta["summary"]["fps"] == 8.0
    assert len(meta["frames"]) == 10
    assert meta["frames"][0]["i"] == 0 and meta["frames"][0]["t"] == 0.0
    assert meta["frames"][8]["t"] == pytest.approx(1.0)  # 8 / 8fps

    assert (tmp_path / "thumb.jpg").exists()
    assert artifacts["frame_count"] == 10
    assert artifacts["duration_sec"] == pytest.approx(10 / 8.0)
    assert artifacts["codec"] in ("avc1", "mp4v")


def test_add_frame_before_open_raises(tmp_path):
    rec = ReplayRecorder()
    with pytest.raises(RuntimeError):
        rec.add_frame(_frame(), {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replay_recorder'`

(Note: tests import bare `replay_recorder`; ensure `tests/conftest.py` adds `web/` to `sys.path`. If no such conftest exists, create `tests/conftest.py` with:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
```
Check first — other tests import `web.field...`; confirm how they resolve and match that. If existing tests use `from web.field...`, instead import `from web.replay_recorder import ReplayRecorder` in the tests and skip conftest.)

- [ ] **Step 3: Implement `ReplayRecorder` (core, no events yet)**

```python
# web/replay_recorder.py
"""Records the composited HUD frame stream + per-frame metadata for match replay.

Knows nothing about FastAPI, jobs, or the database: it consumes composited frames
and plain-dict metadata records and produces hud.mp4 + meta.json + thumb.jpg in a
given output directory.  The caller owns DB insertion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


class ReplayRecorder:
    def __init__(self) -> None:
        self._writer: Optional[cv2.VideoWriter] = None
        self._out_dir: Optional[Path] = None
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.codec: str = ""
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self._mid_frame: Optional[np.ndarray] = None

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    def open_lazy(self, first_frame: np.ndarray, fps: float, out_dir) -> None:
        if self._writer is not None:
            return
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 8.0
        self.height, self.width = first_frame.shape[:2]
        path = str(self._out_dir / "hud.mp4")
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*codec), self.fps, (self.width, self.height))
            if writer.isOpened():
                self._writer = writer
                self.codec = codec
                return
            writer.release()
        raise RuntimeError("Could not open a VideoWriter with avc1 or mp4v")

    def add_frame(self, frame: np.ndarray, record: dict) -> None:
        if self._writer is None:
            raise RuntimeError("add_frame called before open_lazy")
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        self._writer.write(frame)
        i = len(self.frames)
        rec = dict(record)
        rec["i"] = i
        rec["t"] = round(i / self.fps, 4)
        self.frames.append(rec)
        if self._mid_frame is None or i == len(self.frames) // 2:
            self._mid_frame = frame.copy()

    def add_event(self, event: dict) -> None:
        self.events.append(dict(event))

    def finalize(self, status: str, summary: dict) -> dict:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        frame_count = len(self.frames)
        duration = round(frame_count / self.fps, 4) if self.fps else 0.0

        # thumbnail: cached mid frame (fallback: black)
        thumb = self._mid_frame
        if thumb is None:
            thumb = np.zeros((max(1, self.height), max(1, self.width), 3), np.uint8)
        cv2.imwrite(str(self._out_dir / "thumb.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])

        self._stamp_event_times()  # Task 3 fills this in; no-op stub for now

        full_summary = dict(summary)
        full_summary.update({
            "fps": self.fps, "frame_count": frame_count, "duration_sec": duration,
        })
        meta = {"summary": full_summary, "frames": self.frames, "events": self.events}
        (self._out_dir / "meta.json").write_text(json.dumps(meta))

        return {
            "video_path": str(self._out_dir / "hud.mp4"),
            "meta_path": str(self._out_dir / "meta.json"),
            "thumb_path": str(self._out_dir / "thumb.jpg"),
            "fps": self.fps, "frame_count": frame_count, "duration_sec": duration,
            "width": self.width, "height": self.height,
            "codec": self.codec, "status": status,
        }

    def _stamp_event_times(self) -> None:
        """Stamp each event's `t` from its frame number. Implemented in Task 3."""
        return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_replay_recorder.py -v`
Expected: PASS (both tests). If `avc1` is unavailable in this OpenCV build the codec will be `mp4v` — the assertion allows both.

- [ ] **Step 5: Commit**

```bash
git add web/replay_recorder.py tests/test_replay_recorder.py tests/conftest.py
git commit -m "feat(replay): ReplayRecorder video + metadata core"
```

---

## Task 3: Event-time mapping + partial finalize

**Files:**
- Modify: `web/replay_recorder.py` (`add_event`, `_stamp_event_times`)
- Test: `tests/test_replay_recorder.py` (add tests)

**Design note — the off-by-frame_skip trap (from the spec):** scoring events carry a source `frame_num`. The video plays at the *recorded* rate, so an event's replay time must be `recorded_index(frame_num) / fps`, NOT `event.timestamp` (which is source video-time `frame_num / video_fps`). The recorder knows the mapping: as it writes frames, the caller passes the source `frame_num` in each record, so `_stamp_event_times` maps each event's `frame_num` to the recorded index of the nearest frame at or before it.

- [ ] **Step 1: Write the failing test**

```python
def test_event_times_use_recorded_index_not_video_timestamp(tmp_path):
    rec = ReplayRecorder()
    rec.open_lazy(_frame(), fps=8.0, out_dir=tmp_path)
    # Source video is 60fps, frame_skip 6 -> recorded frame i corresponds to frame_num=180+6*i
    for i in range(20):
        rec.add_frame(_frame(), {"frame_num": 180 + 6 * i, "scores": {"blue": 0, "red": 0}, "robots": []})
    # An event at source frame_num 240 -> recorded index (240-180)/6 = 10 -> t = 10/8 = 1.25s
    rec.add_event({"frame_num": 240, "alliance": "blue", "slot": "B2", "robot_label": "B2 9785", "confidence": "high"})
    rec.finalize(status="complete", summary={})

    meta = json.loads((tmp_path / "meta.json").read_text())
    ev = meta["events"][0]
    assert ev["t"] == pytest.approx(1.25)  # recorded-index based, NOT 240/60=4.0s


def test_partial_finalize_produces_valid_artifacts(tmp_path):
    rec = ReplayRecorder()
    rec.open_lazy(_frame(), fps=8.0, out_dir=tmp_path)
    rec.add_frame(_frame(), {"frame_num": 180, "scores": {"blue": 0, "red": 0}, "robots": []})
    artifacts = rec.finalize(status="partial", summary={})
    assert artifacts["status"] == "partial"
    cap = cv2.VideoCapture(str(tmp_path / "hud.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
    cap.release()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_recorder.py::test_event_times_use_recorded_index_not_video_timestamp -v`
Expected: FAIL (event has no `t`, or `t` is wrong/`None`)

- [ ] **Step 3: Implement `_stamp_event_times`**

Replace the stub:

```python
    def _stamp_event_times(self) -> None:
        """Map each event's source frame_num to the recorded-frame index, then t=i/fps.

        Uses the largest recorded frame whose frame_num <= event frame_num (the frame
        on screen when the event fired). Falls back to index 0 if none precede it.
        """
        # (recorded_index, frame_num) pairs, ascending by frame_num
        indexed = [(f["i"], f.get("frame_num")) for f in self.frames if f.get("frame_num") is not None]
        for ev in self.events:
            fn = ev.get("frame_num")
            idx = 0
            if fn is not None and indexed:
                idx = 0
                for i, frame_num in indexed:
                    if frame_num <= fn:
                        idx = i
                    else:
                        break
            ev["t"] = round(idx / self.fps, 4) if self.fps else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_replay_recorder.py -v`
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add web/replay_recorder.py tests/test_replay_recorder.py
git commit -m "feat(replay): recorded-index event timing + partial finalize"
```

---

## Task 4: Wire the recorder into analysis

**Files:**
- Modify: `web/app.py` — add `REPLAYS_DIR` const near `UPLOAD_DIR` (~line 115); construct/use a `ReplayRecorder` in `InferenceJob.process_video`; finalize + DB upsert.

**No new unit test** (this is integration glue over the full pipeline; covered by manual verification + the recorder's own tests). Keep the diff minimal and mechanical.

- [ ] **Step 1: Add the constant and import**

Near `UPLOAD_DIR` (~`web/app.py:115`):
```python
REPLAYS_DIR = Path(__file__).parent / "replays"
REPLAYS_DIR.mkdir(exist_ok=True)
```
At the top of the module with the other imports:
```python
from replay_recorder import ReplayRecorder
import database
```
(Check whether `database` is already imported at module scope — `init_db` is imported locally in `startup()`. If `database` isn't module-imported, add it.)

- [ ] **Step 2: Create the recorder when SAM2/analysis starts**

In `process_video`, where the job initializes per-run state (near `self.sam2_raw = {}`), add:
```python
self._replay = ReplayRecorder()
```

- [ ] **Step 3: Record each composited frame**

Immediately after `annotated_frame = self._compose_hud_frame(...)` (~`web/app.py:3090`) and after `robots_data`/`events_data` are built (move the recording call to *after* line 3133 so the record can reuse them), add:
```python
                # --- Replay recording: capture the exact composited HUD frame + metadata
                try:
                    eff_fps = (self.video_fps / self.frame_skip) if (self.video_fps and self.frame_skip) else 8.0
                    if not self._replay.is_open:
                        self._replay.open_lazy(annotated_frame, eff_fps, REPLAYS_DIR / self.job_id)
                    record = {
                        "frame_num": frame_num,
                        "scores": dict(self.alliance_scores),
                        "robots": [
                            {
                                "slot": self._slot_label(t.slot_id),
                                "alliance": t.alliance,
                                "team_number": (self.robot_tracks.get(t.slot_id).team_number
                                                if self.robot_tracks.get(t.slot_id) else None),
                                "confirmed": (self.robot_tracks.get(t.slot_id).team_number_confirmed
                                              if self.robot_tracks.get(t.slot_id) else False),
                                "field_pos": [float(t.pos[0]), float(t.pos[1])] if t.pos is not None else None,
                            }
                            for t in track_outs if t.state != "empty"
                        ],
                    }
                    self._replay.add_frame(annotated_frame, record)
                except Exception as e:
                    print(f"[replay] frame record error: {e}")
```
And where scoring events are appended to `self.scoring_events` — there are **three append sites at `web/app.py:2220, 2233, 2261`, inside the scoring-attribution method (NOT in `process_video`'s frame loop)** — mirror each new event into the recorder at each site. `self._replay` lives on the instance, so it is reachable from that method; `add_event` only buffers and `finalize` maps `frame_num → i`, so ordering across methods is fine:
```python
                    try:
                        self._replay.add_event({
                            "frame_num": ev.frame_num, "alliance": ev.alliance,
                            "slot": self._slot_label(ev.robot_id),
                            "team_number": ev.team_number, "confidence": ev.confidence,
                            "robot_label": ev.team_number or self._slot_label(ev.robot_id),
                        })
                    except Exception:
                        pass
```
(Adapt field names to the actual `ScoringEvent` attributes: `frame_num`, `timestamp`, `alliance`, `confidence`, `team_number`, `robot_id`.)

- [ ] **Step 4: Finalize in the `finally` block + upsert DB row**

In `process_video`'s `finally` (~`web/app.py:3213`), after closing the log/cap, add:
```python
            try:
                if getattr(self, "_replay", None) is not None and self._replay.is_open:
                    status = "complete" if self.status == "completed" else "partial"
                    blue_teams = [self.robot_tracks[s].team_number for s in sorted(self.robot_tracks)
                                  if self.robot_tracks[s].alliance == "blue"]
                    red_teams = [self.robot_tracks[s].team_number for s in sorted(self.robot_tracks)
                                 if self.robot_tracks[s].alliance == "red"]
                    summary = {
                        "match_label": Path(str(self.video_path)).stem,
                        "blue_score": self.alliance_scores.get("blue", 0),
                        "red_score": self.alliance_scores.get("red", 0),
                        "robots": [r for f in self._replay.frames[:1] for r in f["robots"]],
                    }
                    artifacts = self._replay.finalize(status=status, summary=summary)
                    import json as _json
                    from datetime import datetime as _dt
                    await database.insert_replay({
                        "id": self.job_id,
                        "created_at": _dt.now().isoformat(timespec="seconds"),
                        "label": Path(str(self.video_path)).stem,
                        "event_key": None, "match_key": None, "comp_level": None,
                        "blue_score": self.alliance_scores.get("blue", 0),
                        "red_score": self.alliance_scores.get("red", 0),
                        "blue_teams": _json.dumps(blue_teams),
                        "red_teams": _json.dumps(red_teams),
                        "source_video_path": str(self.video_path),
                        **artifacts,
                    })
            except Exception as e:
                print(f"[replay] finalize error: {e}")
```
(`artifacts` already includes `video_path`, `meta_path`, `thumb_path`, `fps`, `frame_count`, `duration_sec`, `width`, `height`, `codec`, `status` — matching `_REPLAY_COLUMNS`.)

- [ ] **Step 5: Manual verification**

Run a short analysis (the test footage). Confirm:
```bash
ls web/replays/<job_id>/   # hud.mp4, meta.json, thumb.jpg
.venv/bin/python -c "import cv2; c=cv2.VideoCapture('web/replays/<job_id>/hud.mp4'); print(c.isOpened(), c.get(cv2.CAP_PROP_FRAME_COUNT))"
.venv/bin/python -c "import json; m=json.load(open('web/replays/<job_id>/meta.json')); print(m['summary'], len(m['frames']), len(m['events']))"
```
Expected: a playable mp4 with frame_count ≈ processed frames; meta summary with scores and a non-empty frames list. Verify the existing suite still passes: `.venv/bin/python -m pytest tests/field/ tests/test_tracker.py -q`.

- [ ] **Step 6: Commit**

```bash
git add web/app.py
git commit -m "feat(replay): record HUD frames + metadata during analysis"
```

---

## Task 5: REST endpoints — list / get / thumb / delete

**Files:**
- Modify: `web/app.py` (add endpoints near the other `@app.get("/api/...")` handlers)
- Test: `tests/test_replay_api.py`

**Design note:** Use `fastapi.testclient.TestClient`. The endpoints read from the DB and filesystem, so tests monkeypatch `database.DB_PATH` to a temp file, `init_db`, insert a replay row whose `meta_path`/`video_path`/`thumb_path` point at temp files the test creates, then assert responses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay_api.py
import asyncio, json, importlib
import numpy as np, cv2
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import database
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    import app as appmod
    importlib.reload(appmod)            # pick up patched DB_PATH if cached
    monkeypatch.setattr(appmod, "database", database, raising=False)
    return TestClient(appmod.app), database, tmp_path


def _make_replay_files(tmp_path, rid="job-1", frames=5, fps=8.0):
    d = tmp_path / rid
    d.mkdir()
    vw = cv2.VideoWriter(str(d / "hud.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for _ in range(frames):
        vw.write(np.zeros((120, 160, 3), np.uint8))
    vw.release()
    (d / "meta.json").write_text(json.dumps({"summary": {"frame_count": frames, "fps": fps},
                                             "frames": [{"i": i, "t": i / fps} for i in range(frames)],
                                             "events": []}))
    cv2.imwrite(str(d / "thumb.jpg"), np.zeros((120, 160, 3), np.uint8))
    return d


def test_list_get_delete(client, monkeypatch):
    c, database, tmp_path = client
    d = _make_replay_files(tmp_path)
    asyncio.run(database.insert_replay({
        "id": "job-1", "created_at": "2026-06-03T00:00:00", "label": "Qual 10",
        "blue_score": 70, "red_score": 67, "blue_teams": "[581]", "red_teams": "[1732]",
        "video_path": str(d / "hud.mp4"), "meta_path": str(d / "meta.json"),
        "thumb_path": str(d / "thumb.jpg"), "source_video_path": "x.mp4",
        "fps": 8.0, "frame_count": 5, "duration_sec": 0.625,
        "width": 160, "height": 120, "codec": "mp4v", "status": "complete",
    }))

    r = c.get("/api/replays")
    assert r.status_code == 200
    assert any(x["id"] == "job-1" and x["label"] == "Qual 10" for x in r.json())

    r = c.get("/api/replays/job-1")
    assert r.status_code == 200 and r.json()["summary"]["frame_count"] == 5

    r = c.get("/api/replays/job-1/thumb")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")

    r = c.delete("/api/replays/job-1")
    assert r.status_code == 200
    assert c.get("/api/replays/job-1").status_code == 404
    assert not d.exists()  # artifacts removed


def test_get_missing_returns_404(client):
    c, database, tmp_path = client
    assert c.get("/api/replays/nope").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_api.py::test_list_get_delete -v`
Expected: FAIL (404 / route missing)

- [ ] **Step 3: Implement the endpoints**

Add to `web/app.py` (near other `/api` GET handlers). Confirm `import database`, `import shutil`, `from fastapi.responses import JSONResponse, FileResponse` are present:

```python
@app.get("/api/replays")
async def list_replays_endpoint():
    rows = await database.list_replays()
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "label": r["label"], "created_at": r["created_at"],
            "blue_score": r["blue_score"], "red_score": r["red_score"],
            "blue_teams": json.loads(r["blue_teams"] or "[]"),
            "red_teams": json.loads(r["red_teams"] or "[]"),
            "duration_sec": r["duration_sec"], "status": r["status"],
            "thumb_url": f"/api/replays/{r['id']}/thumb",
            "video_url": f"/api/replays/{r['id']}/video",
        })
    return out


@app.get("/api/replays/{replay_id}")
async def get_replay_endpoint(replay_id: str):
    row = await database.get_replay(replay_id)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not found"})
    meta_path = Path(row["meta_path"])
    if not meta_path.exists():
        return JSONResponse(status_code=404, content={"error": "metadata missing"})
    return JSONResponse(content=json.loads(meta_path.read_text()))


@app.get("/api/replays/{replay_id}/thumb")
async def get_replay_thumb(replay_id: str):
    row = await database.get_replay(replay_id)
    if not row or not Path(row["thumb_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(row["thumb_path"], media_type="image/jpeg")


@app.delete("/api/replays/{replay_id}")
async def delete_replay_endpoint(replay_id: str):
    row = await database.get_replay(replay_id)
    if not row:
        return JSONResponse(status_code=404, content={"error": "not found"})
    replay_dir = REPLAYS_DIR / replay_id
    if replay_dir.exists():
        shutil.rmtree(replay_dir, ignore_errors=True)
    await database.delete_replay(replay_id)
    return {"status": "deleted", "id": replay_id}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_replay_api.py -v`
Expected: PASS (both tests). If `importlib.reload(app)` is problematic (heavy module import), instead patch `appmod.database.DB_PATH` without reload — adjust the fixture so the app module's `database` reference uses the temp DB. Keep the fixture minimal.

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_replay_api.py
git commit -m "feat(replay): list/get/thumb/delete endpoints"
```

---

## Task 6: Video endpoint with HTTP Range

**Files:**
- Modify: `web/app.py` (add `/api/replays/{id}/video`)
- Test: `tests/test_replay_api.py` (add Range tests)

**Design note:** Browsers seeking in `<video>` send `Range: bytes=START-END` and require a `206 Partial Content` response with `Content-Range`, `Accept-Ranges: bytes`, and the exact byte slice. A plain `FileResponse` does not reliably satisfy seeking across browsers, so implement Range manually.

- [ ] **Step 1: Write the failing test**

```python
def test_video_range_returns_206(client):
    c, database, tmp_path = client
    d = _make_replay_files(tmp_path, rid="job-2")
    size = (d / "hud.mp4").stat().st_size
    asyncio.run(database.insert_replay({
        "id": "job-2", "created_at": "2026-06-03T00:00:00", "label": "x",
        "blue_score": 0, "red_score": 0, "blue_teams": "[]", "red_teams": "[]",
        "video_path": str(d / "hud.mp4"), "meta_path": str(d / "meta.json"),
        "thumb_path": str(d / "thumb.jpg"), "source_video_path": "x.mp4",
        "fps": 8.0, "frame_count": 5, "duration_sec": 0.625,
        "width": 160, "height": 120, "codec": "mp4v", "status": "complete",
    }))

    # full GET -> 200
    r = c.get("/api/replays/job-2/video")
    assert r.status_code == 200 and len(r.content) == size

    # ranged GET -> 206 with correct slice
    r = c.get("/api/replays/job-2/video", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 0-99/{size}"
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_api.py::test_video_range_returns_206 -v`
Expected: FAIL (route missing / not 206)

- [ ] **Step 3: Implement the Range-aware endpoint**

```python
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

@app.get("/api/replays/{replay_id}/video")
async def get_replay_video(replay_id: str, request: Request):
    row = await database.get_replay(replay_id)
    if not row or not Path(row["video_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    path = Path(row["video_path"])
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header is None:
        return FileResponse(str(path), media_type="video/mp4")

    # Parse "bytes=START-END"
    units, _, rng = range_header.partition("=")
    start_s, _, end_s = rng.partition("-")
    try:
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
    length = end - start + 1

    def iter_slice(chunk=64 * 1024):
        remaining = length
        with path.open("rb") as f:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iter_slice(), status_code=206,
                             media_type="video/mp4", headers=headers)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_replay_api.py -v`
Expected: PASS (all). Note: `TestClient` does not auto-add a `Range` header, so the full-GET case returns 200 via `FileResponse` and the ranged case returns 206 — both asserted.

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_replay_api.py
git commit -m "feat(replay): Range-aware video streaming endpoint"
```

---

## Task 7: Frontend — library + player

**Files:**
- Modify: `web/static/index.html` (Replays nav entry; `#replayLibrary` grid container; `#replayPlayer` view with `<video id="replayVideo" controls>`, reused robot tables / `#timelineContainer` / a `<canvas id="replayFieldMap">`; a "Replay" button on the results screen)
- Modify: `web/static/style.css` (library card grid, player layout)
- Modify: `web/static/app.js` (library fetch/render/delete; `ReplayPlayer`; refactor `updateTimeline` to accept full event list + be clickable)

**No automated test — verified manually.** Implement in small commits. Reuse existing helpers: `updateRobotTables(robots)` (already arg-driven), and refactor `updateTimeline(events, {onSeek})`.

**Design notes:**
- **Sync:** `ReplayPlayer` loads `GET /api/replays/{id}` (meta) once, holds `meta.frames` (sorted by `i`) and `meta.summary.fps`. On the `<video>`'s `timeupdate` event: `const idx = Math.round(video.currentTime * fps); const rec = frames[Math.min(idx, frames.length-1)];` then `updateRobotTables(rec.robots)`, repaint the field-map canvas from `rec.robots[].field_pos`, and highlight the current timeline event (largest `event.t <= currentTime`).
- **Field map (client-side):** draw a simple top-down field on `#replayFieldMap` (a rectangle split into blue/red halves with a center line), then plot each robot as a dot at its `field_pos`. Map field metres → canvas pixels using the field dimensions exposed by the existing radar/field config. Fetch field dims from an existing status/config endpoint if available; otherwise hardcode the 2026 field extent from `FieldModel` (the `field_length`/`field_width` passed into `Radar.__init__`; `web/field/radar.py` itself does not define the extent) and cite it. Keep it deliberately simple — this is a lightweight scrub indicator, not a re-render of the baked field map.
- **Timeline click-to-seek:** `updateTimeline(events, {onSeek})` renders each event as a clickable element; clicking calls `onSeek(event.t)`, which sets `replayVideo.currentTime = event.t`.
- **Library:** `GET /api/replays` → render cards (`thumb_url`, label, `blue_teams` vs `red_teams`, `blue_score`-`red_score`, date, status badge if `partial`). Each card: "Replay" (opens player with `video_url` + meta) and "Delete" (confirm → `DELETE /api/replays/{id}` → refresh).
- **Just-finished hook:** in the existing completion handler (`app.js` ~`handleComplete`/`showResultsEditor`, ~line 963/995), add a "Replay" button that opens the player for the current `job_id`.
- **Nav:** add a "Replays" entry that shows the library view and hides the live/analysis view (follow whatever view-switching pattern `app.js` already uses; if none, toggle `display` on the top-level sections).

- [ ] **Step 1: Refactor `updateTimeline` to take a full event list + optional `onSeek`; verify the live view still renders (manual).**
- [ ] **Step 2: Add the library view (HTML/CSS + `loadReplayLibrary()` fetch/render + delete).** Manual: load page, see cards for past matches, delete removes a card.
- [ ] **Step 3: Add `ReplayPlayer` (HTML/CSS + class).** Manual: click Replay → video plays; scrub/seek/speed work; robot tables + field map update with playback; clicking a timeline event seeks.
- [ ] **Step 4: Add the "Replay" button to the results screen.** Manual: finish an analysis → Replay button opens the player for that match.
- [ ] **Step 5: Commit (one commit per step or a single frontend commit).**

```bash
git add web/static/app.js web/static/index.html web/static/style.css
git commit -m "feat(replay): library + player frontend"
```

---

## Final verification

- [ ] `.venv/bin/python -m pytest tests/test_replay_db.py tests/test_replay_recorder.py tests/test_replay_api.py -v` — all pass.
- [ ] `.venv/bin/python -m pytest tests/field/ tests/test_tracker.py -q` — existing suite still green (95 passing).
- [ ] End-to-end manual: analyze the test footage → it appears in the Replays library → open it → playback syncs tables/timeline/field map, click-to-seek and delete work.
- [ ] Stop an analysis mid-run → a `partial` replay is still listed and playable.
```

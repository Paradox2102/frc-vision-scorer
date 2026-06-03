# Per-Event Appearance Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover obscured/unreadable bumper numbers by matching a robot's whole-robot appearance against a persistent, event-scoped gallery of confidently-identified robots.

**Architecture:** A new pure-numpy `AppearanceGallery` keeps a diverse, capped set of DINOv2 embeddings per `(team, alliance_color)`. Confident OCR/manual identifications deposit embeddings; unconfirmed tracks query the gallery and cast votes through the existing `team_ocr_votes` + `ConsecutiveValueTracker` lock machinery, so a sustained strong appearance match can confirm an identity on its own without ever overriding a real OCR lock. The gallery persists to a new SQLite table keyed by `event_key` and reloads at the start of every match.

**Tech Stack:** Python, numpy, aiosqlite (existing `web/database.py`), the existing `web/embedder.py` DINOv2 embedder, vanilla JS frontend (`web/static/app.js`/`style.css`). Tests use pytest (`tests/`), run via `.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-03-appearance-gallery-design.md`

---

## Key facts for the implementer (read before starting)

- **Tests live in `tests/`** (not `web/tests/`) and import via `web.*` (e.g. `import web.database`). Run with `.venv/bin/python -m pytest` — the base conda python has no pytest.
- **The embedding** is a 384-dim L2-normalized float32 vector from `web.embedder.RobotEmbedder.embed()`. `web.embedder.cosine_similarity(a, b)` already exists and returns 0.0 for `None` inputs.
- **Per-track state** lives in `RobotTrack` (`web/app.py:329`): `appearance_embedding` (np.ndarray|None), `team_number` (str|None), `team_number_confirmed` (bool), `alliance` ("blue"|"red"|"unknown"), `team_ocr_votes` (dict|None), `consecutive_tracker` (`ConsecutiveValueTracker`|None).
- **The lock**: `ConsecutiveValueTracker(n_consecutive=3)` (`web/field/number_lock.py`). `.update(value)` returns `(locked_value, confirmed)`; 3 consecutive equal non-None values flips `confirmed`.
- **Main integration function**: `read_fused_bumper_numbers` (`web/app.py:823-976`). OCR votes accumulate at line 950-952; the lock-confirm site is 956-967; the per-crop UI payload (`ocr_crops`) is built at 926-937.
- **`event_key`** is on the job: `self.event_key` (`web/app.py:415`), may be `None`.
- **DB pattern**: SCHEMA string + `init_db()` runs `executescript` (`CREATE TABLE IF NOT EXISTS` is safe for fresh and existing DBs). CRUD funcs are `async def`, open `aiosqlite.connect(DB_PATH)`. Tests monkeypatch `database.DB_PATH` and call `asyncio.run(...)` (see `tests/test_replay_db.py`).
- **Frontend**: `web/static/app.js:901-925` (`updateVlmCrops`) renders each crop card. `web/static/style.css:676` has `.confidence-badge` styles to mirror.

---

## Task 1: `AppearanceGallery` core (pure numpy)

A self-contained class with no DB or track knowledge, fully unit-testable with synthetic vectors. Also includes the pure match-classification function the integration layer will call.

**Files:**
- Create: `web/field/appearance_gallery.py`
- Test: `tests/test_appearance_gallery.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_appearance_gallery.py
import numpy as np
import pytest

from web.field.appearance_gallery import AppearanceGallery, classify_match


def _unit(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_add_then_query_returns_team():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("1787", "blue", e, legibility=0.9)
    ranked = g.query(e, "blue")
    assert ranked[0][0] == "1787"
    assert ranked[0][1] == pytest.approx(1.0, abs=1e-5)


def test_query_filters_by_color():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("1787", "blue", e, legibility=0.9)
    assert g.query(e, "red") == []          # no red exemplars
    assert g.query(e, None)[0][0] == "1787"  # unknown color searches both


def test_dedup_rejects_near_duplicate_keeps_diverse():
    g = AppearanceGallery(dedup_sim=0.92, max_exemplars=8)
    base = _unit(1, 0, 0)
    near = _unit(1, 0.05, 0)        # cosine > 0.92 to base
    diverse = _unit(0, 1, 0)        # orthogonal -> kept
    g.add("254", "red", base, 0.9)
    g.add("254", "red", near, 0.9)      # rejected (too similar)
    g.add("254", "red", diverse, 0.9)   # kept (covers another viewpoint)
    assert g.exemplar_count("254", "red") == 2


def test_cap_evicts_lowest_legibility():
    g = AppearanceGallery(dedup_sim=0.999, max_exemplars=2)
    # all sufficiently distinct so dedup never triggers
    g.add("9", "red", _unit(1, 0, 0), legibility=0.3)
    g.add("9", "red", _unit(0, 1, 0), legibility=0.9)
    g.add("9", "red", _unit(0, 0, 1), legibility=0.8)  # full -> evict the 0.3
    assert g.exemplar_count("9", "red") == 2
    legs = sorted(g.legibilities("9", "red"))
    assert legs == [0.8, 0.9]


def test_query_returns_best_per_team_sorted():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    g.add("254", "blue", _unit(0, 1, 0), 0.9)
    q = _unit(0.9, 0.2, 0)  # closest to 1787
    ranked = g.query(q, "blue")
    assert [t for t, _ in ranked] == ["1787", "254"]
    assert ranked[0][1] > ranked[1][1]


def test_classify_match_confirm_supplement_none():
    # ranked = list[(team, sim)] sorted desc; second-best means next DISTINCT team
    assert classify_match([("1787", 0.90), ("254", 0.70)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80) == ("confirm", "1787", 0.90)
    # high sim but 2nd team too close -> supplement, not confirm
    assert classify_match([("1787", 0.90), ("254", 0.87)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80)[0] == "supplement"
    # moderate sim -> supplement
    assert classify_match([("1787", 0.82)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80)[0] == "supplement"
    # weak -> none
    assert classify_match([("1787", 0.50)],
                          confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80) == ("none", None, 0.50)
    assert classify_match([], confirm_sim=0.88, confirm_gap=0.06, supplement_sim=0.80) == ("none", None, 0.0)


def test_serialize_roundtrip():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    rows = g.serialize()
    assert rows and set(rows[0]) >= {"team", "alliance_color", "embedding", "legibility"}
    g2 = AppearanceGallery.from_rows(rows)
    assert g2.query(_unit(1, 0, 0), "blue")[0][0] == "1787"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_appearance_gallery.py -v`
Expected: FAIL with `ModuleNotFoundError: web.field.appearance_gallery`.

- [ ] **Step 3: Implement `web/field/appearance_gallery.py`**

```python
"""Per-event appearance gallery for recovering obscured bumper numbers.

FRC-agnostic and storage-agnostic: holds DINOv2 embeddings keyed by
(team, alliance_color), supports diverse-set deposits and nearest-team queries.
The integration layer (web/app.py) owns persistence and vote casting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from web.embedder import cosine_similarity


class _Exemplar:
    __slots__ = ("embedding", "legibility")

    def __init__(self, embedding: np.ndarray, legibility: float):
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.legibility = float(legibility)


class AppearanceGallery:
    """Diverse, capped sets of appearance embeddings per (team, alliance_color)."""

    def __init__(self, dedup_sim: float = 0.92, max_exemplars: int = 12):
        self.dedup_sim = dedup_sim
        self.max_exemplars = max_exemplars
        # {(team, color): [_Exemplar, ...]}
        self._sets: dict[tuple[str, str], list[_Exemplar]] = {}

    def add(self, team: str, color: str, embedding, legibility: float) -> bool:
        """Deposit an exemplar. Returns True if stored, False if rejected as a
        near-duplicate. Keeps the set diverse (dedup) and capped (evict lowest
        legibility)."""
        if embedding is None or color not in ("red", "blue") or not team:
            return False
        emb = np.asarray(embedding, dtype=np.float32)
        bucket = self._sets.setdefault((team, color), [])
        for ex in bucket:
            if cosine_similarity(emb, ex.embedding) >= self.dedup_sim:
                return False  # too similar to an existing viewpoint
        bucket.append(_Exemplar(emb, legibility))
        if len(bucket) > self.max_exemplars:
            # evict the lowest-legibility exemplar
            worst = min(range(len(bucket)), key=lambda i: bucket[i].legibility)
            bucket.pop(worst)
        return True

    def query(self, embedding, color: Optional[str]) -> list[tuple[str, float]]:
        """Best cosine similarity per team, filtered to `color` (both if None),
        sorted descending."""
        if embedding is None:
            return []
        emb = np.asarray(embedding, dtype=np.float32)
        best: dict[str, float] = {}
        for (team, c), bucket in self._sets.items():
            if color in ("red", "blue") and c != color:
                continue
            for ex in bucket:
                sim = cosine_similarity(emb, ex.embedding)
                if sim > best.get(team, -1.0):
                    best[team] = sim
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    # --- introspection (used by tests) ---
    def exemplar_count(self, team: str, color: str) -> int:
        return len(self._sets.get((team, color), []))

    def legibilities(self, team: str, color: str) -> list[float]:
        return [ex.legibility for ex in self._sets.get((team, color), [])]

    # --- persistence bridge ---
    def serialize(self) -> list[dict]:
        rows = []
        for (team, color), bucket in self._sets.items():
            for ex in bucket:
                rows.append({
                    "team": team,
                    "alliance_color": color,
                    "embedding": ex.embedding.astype(np.float32).tobytes(),
                    "legibility": ex.legibility,
                })
        return rows

    @classmethod
    def from_rows(cls, rows, dedup_sim: float = 0.92, max_exemplars: int = 12):
        g = cls(dedup_sim=dedup_sim, max_exemplars=max_exemplars)
        for r in rows:
            emb = r["embedding"]
            if isinstance(emb, (bytes, bytearray)):
                emb = np.frombuffer(emb, dtype=np.float32)
            # load directly (bypass dedup/cap; DB is the source of truth)
            g._sets.setdefault((r["team"], r["alliance_color"]), []).append(
                _Exemplar(emb, r.get("legibility") or 0.0))
        return g


def classify_match(ranked: list[tuple[str, float]], *, confirm_sim: float,
                   confirm_gap: float, supplement_sim: float):
    """Decide what a gallery query result warrants.

    Returns (verdict, team, sim) where verdict is 'confirm' | 'supplement' | 'none'.
    'second-best team' is the next DISTINCT team in the ranked list.
    """
    if not ranked:
        return ("none", None, 0.0)
    team, sim = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if sim >= confirm_sim and (sim - second) >= confirm_gap:
        return ("confirm", team, sim)
    if sim >= supplement_sim:
        return ("supplement", team, sim)
    return ("none", None, sim)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_appearance_gallery.py -v`
Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add web/field/appearance_gallery.py tests/test_appearance_gallery.py
git commit -m "feat(gallery): AppearanceGallery core + match classification (TDD)"
```

---

## Task 2: Gallery persistence (table + CRUD)

**Files:**
- Modify: `web/database.py` (add table to `SCHEMA` ~line 98; add CRUD funcs after `delete_replay` ~line 501)
- Test: `tests/test_gallery_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gallery_db.py
import asyncio
import numpy as np
import pytest

import web.database as database


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _emb_bytes(*vals):
    return np.array(vals, dtype=np.float32).tobytes()


def test_insert_and_fetch_gallery_exemplars_by_event(db):
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "1787", "alliance_color": "blue",
        "embedding": _emb_bytes(1, 0, 0), "legibility": 0.9,
    }))
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026wamon", "team": "254", "alliance_color": "red",
        "embedding": _emb_bytes(0, 1, 0), "legibility": 0.8,
    }))
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026other", "team": "9", "alliance_color": "red",
        "embedding": _emb_bytes(0, 0, 1), "legibility": 0.7,
    }))

    rows = asyncio.run(db.get_gallery_exemplars("2026wamon"))
    assert len(rows) == 2
    teams = sorted(r["team"] for r in rows)
    assert teams == ["1787", "254"]
    # embedding survives round-trip as bytes -> float32
    blue = next(r for r in rows if r["team"] == "1787")
    arr = np.frombuffer(blue["embedding"], dtype=np.float32)
    assert np.allclose(arr, [1, 0, 0])


def test_get_gallery_exemplars_empty_for_unknown_event(db):
    assert asyncio.run(db.get_gallery_exemplars("nope")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gallery_db.py -v`
Expected: FAIL with `AttributeError: module 'web.database' has no attribute 'insert_gallery_exemplar'`.

- [ ] **Step 3: Implement schema + CRUD**

In `web/database.py`, append this table to the `SCHEMA` string (before the closing `"""`, after the `replays` table):

```sql
CREATE TABLE IF NOT EXISTS appearance_gallery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL,
    team TEXT NOT NULL,
    alliance_color TEXT NOT NULL,
    embedding BLOB NOT NULL,
    legibility REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_gallery_event ON appearance_gallery(event_key, alliance_color);
```

(No `ALTER` migration needed — `executescript` runs `CREATE TABLE IF NOT EXISTS`, which adds the table to existing DBs automatically.)

Add CRUD functions after `delete_replay` (~line 501):

```python
# --- Appearance gallery ---

async def insert_gallery_exemplar(row: Dict[str, Any]) -> None:
    """Insert one appearance-gallery exemplar."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO appearance_gallery "
            "(event_key, team, alliance_color, embedding, legibility) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["event_key"], row["team"], row["alliance_color"],
             row["embedding"], row.get("legibility")))
        await db.commit()


async def get_gallery_exemplars(event_key: str) -> List[Dict]:
    """All gallery exemplars for an event."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT team, alliance_color, embedding, legibility "
            "FROM appearance_gallery WHERE event_key = ?", (event_key,))
        return [dict(r) for r in await cursor.fetchall()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gallery_db.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add web/database.py tests/test_gallery_db.py
git commit -m "feat(gallery): appearance_gallery table + CRUD (TDD)"
```

---

## Task 3: Tunables + load gallery + deposit on confirm

Wire the gallery into `web/app.py`: env tunables, load at job start, and deposit exemplars whenever an identity is confirmed by a high-trust source (OCR lock or manual/calibration confirm). Deposits go to both the in-memory gallery and the DB.

**Files:**
- Modify: `web/app.py` — constants block (near `VLM_TOP_K`); `InferenceJob.__init__`; gallery-load site (early in `process_video`, near where `event_key` calibration loads, ~line 2361); `read_fused_bumper_numbers` confirm site (956-967); manual confirm site (~1254).
- Test: `tests/test_gallery_db.py` (extend with a deposit-helper-level check is hard without InferenceJob; instead verify wiring via the verification step below).

- [ ] **Step 1: Add constants**

Near the other VLM constants (search for `VLM_TOP_K =`), add:

```python
GALLERY_ENABLED = os.environ.get("GALLERY_ENABLED", "1") not in ("0", "false", "False")
GALLERY_CONFIRM_SIM = float(os.environ.get("GALLERY_CONFIRM_SIM", "0.88"))
GALLERY_CONFIRM_GAP = float(os.environ.get("GALLERY_CONFIRM_GAP", "0.06"))
GALLERY_SUPPLEMENT_SIM = float(os.environ.get("GALLERY_SUPPLEMENT_SIM", "0.80"))
GALLERY_MAX_EXEMPLARS = int(os.environ.get("GALLERY_MAX_EXEMPLARS", "12"))
GALLERY_DEDUP_SIM = float(os.environ.get("GALLERY_DEDUP_SIM", "0.92"))
GALLERY_STRONG_VOTE_WEIGHT = float(os.environ.get("GALLERY_STRONG_VOTE_WEIGHT", "1.0"))
GALLERY_WEAK_VOTE_WEIGHT = float(os.environ.get("GALLERY_WEAK_VOTE_WEIGHT", "0.4"))
```

Add the import near the top with the other `web.field` imports:

```python
from web.field.appearance_gallery import AppearanceGallery, classify_match
```

While here, correct the now-inaccurate type hint on `RobotTrack.team_ocr_votes` (web/app.py:342) — existing code and gallery votes both store floats:

```python
    team_ocr_votes: Optional[Dict[str, float]] = None  # team_number -> vote weight
```

- [ ] **Step 2: Initialize the gallery fields in `__init__`**

In `InferenceJob.__init__`, near the other tracking state (after `self.robot_tracks = {}`), add:

```python
self.gallery: Optional[AppearanceGallery] = None  # loaded in process_video if enabled
self._pending_gallery_rows: list = []  # buffered exemplar rows, flushed (awaited) at finalize
```

> **Async context (critical):** `process_video` is `async def` (web/app.py:2419) and runs inside a live event loop; `read_fused_bumper_numbers` is a **sync** method called on the loop thread at web/app.py:2986. Therefore: **never call `asyncio.run(...)`** on these paths (it raises `RuntimeError: asyncio.run() cannot be called from a running event loop`). Gallery *loading* happens in async `process_video` via `await`. Gallery *deposits* happen from the sync read path, so they only mutate the in-memory gallery and **buffer** their DB rows; the buffer is flushed with `await` at the async finalize region. This keeps within-match recovery live and persists across matches at job end (satisfying "gets better over time").

- [ ] **Step 3: Load the gallery at job start**

In `process_video`, immediately after the `_init_field_tracking()` gate passes — i.e. at the top of the `try:` block at web/app.py:2435 (event_key/calibration is known by then) — add an `await`ed load that tolerates a missing event_key or DB error:

```python
        if GALLERY_ENABLED and self.event_key:
            try:
                rows = await database.get_gallery_exemplars(self.event_key)
                self.gallery = AppearanceGallery.from_rows(
                    rows, dedup_sim=GALLERY_DEDUP_SIM, max_exemplars=GALLERY_MAX_EXEMPLARS)
                print(f"  Appearance gallery loaded: {len(rows)} exemplars for {self.event_key}")
            except Exception as e:
                print(f"  Appearance gallery load failed ({e}); appearance recovery disabled")
                self.gallery = None
```

(`database` is already imported at module level — it is used at web/app.py:3320. Use it directly; do not add a local import.)

- [ ] **Step 4: Add the deposit helper (in-memory + buffer, NO DB call)**

Add a method on `InferenceJob` (near `read_fused_bumper_numbers`). It is sync-safe: it updates the in-memory gallery immediately and appends the persistable row to `self._pending_gallery_rows`. It never touches the DB.

```python
def _deposit_to_gallery(self, robot, legibility: float = 0.0) -> None:
    """Deposit a confirmed robot's appearance into the in-memory event gallery and
    buffer it for persistence at finalize. Only externally-grounded labels (an OCR
    lock or OCR re-affirmation) should call this — never a gallery-driven
    confirmation, to avoid self-reinforcement (spec non-goal)."""
    if self.gallery is None or robot is None:
        return
    color = robot.alliance if robot.alliance in ("red", "blue") else None
    if color is None or not robot.team_number or robot.appearance_embedding is None:
        return
    stored = self.gallery.add(robot.team_number, color, robot.appearance_embedding, legibility)
    if not stored or not self.event_key:
        return  # rejected as near-duplicate, or no event to persist under
    self._pending_gallery_rows.append({
        "event_key": self.event_key, "team": robot.team_number,
        "alliance_color": color,
        "embedding": np.asarray(robot.appearance_embedding, dtype=np.float32).tobytes(),
        "legibility": legibility,
    })
```

- [ ] **Step 5: Call the deposit helper at the (externally-grounded) confirm sites**

In `read_fused_bumper_numbers`, inside the `if confirmed and locked_team:` block (after line 967, after the print), add:

```python
                self._deposit_to_gallery(robot, legibility=conf)
```

Also deposit on confirmed re-read passes when OCR strongly re-affirms the locked team. Just before the `if not team: continue` at line 948, after computing `team`/`conf`, add:

```python
            if (robot.team_number_confirmed and team == robot.team_number
                    and conf >= 0.85):
                self._deposit_to_gallery(robot, legibility=conf)
```

> **Do NOT deposit at the `eliminate_last_team` site (~web/app.py:1249-1258).** That path confirms an identity by *process of elimination* (an inference), not by reading the bumper or a human label. Seeding the gallery from it would violate the spec's external-grounding / no-self-reinforcement rule. There is currently no manual team-set endpoint; if one is added later, it is the correct additional deposit site (with `legibility=1.0`).

- [ ] **Step 6: Flush buffered exemplars at finalize (awaited)**

In `process_video`, just after the replay-finalize `try/except` block (after web/app.py:3336), add an awaited flush in its own guarded block:

```python
            try:
                if self._pending_gallery_rows:
                    for row in self._pending_gallery_rows:
                        await database.insert_gallery_exemplar(row)
                    print(f"  Appearance gallery: persisted {len(self._pending_gallery_rows)} "
                          f"new exemplars for {self.event_key}")
                    self._pending_gallery_rows = []
            except Exception as e:
                print(f"  Appearance gallery flush failed ({e})")
```

- [ ] **Step 7: Verify nothing is broken**

Run: `.venv/bin/python -m pytest tests/ -v -m "not model"`
Expected: PASS (existing suite still green; no new failures). The deposit/flush wiring has no isolated unit test — it is exercised by the integration verification in the Final Verification section (run two matches at one event_key, confirm exemplars persist) and by the existing suite continuing to pass.

- [ ] **Step 8: Commit**

```bash
git add web/app.py
git commit -m "feat(gallery): load per-event gallery, buffer deposits, flush at finalize"
```

---

## Task 4: Query the gallery & cast votes

Add a method that, each read pass, queries the gallery for **unconfirmed** tracks and casts votes through the existing vote/lock machinery. A `confirm` verdict casts a strong-weight vote (sustained agreement still required to lock via `ConsecutiveValueTracker`); a `supplement` verdict casts a weak vote. Never touches already-confirmed tracks. Also patches matching `ocr_crops` entries so the UI can show the recovery.

**Files:**
- Modify: `web/app.py` — new method `apply_gallery_matches`; call site at the end of `read_fused_bumper_numbers` (after `ocr_crops` is built, ~line 971).
- Test: `tests/test_gallery_match_logic.py` (tests the pure decision via `classify_match` + a vote-application helper that operates on a minimal fake track).

- [ ] **Step 1: Write the failing test**

The vote application is small enough to test against a lightweight stand-in track (duck-typed), keeping it independent of `InferenceJob`.

```python
# tests/test_gallery_match_logic.py
import numpy as np

from web.field.appearance_gallery import AppearanceGallery
from web.field.number_lock import ConsecutiveValueTracker
from web.app import _apply_gallery_vote, GALLERY_CONFIRM_SIM


class FakeTrack:
    def __init__(self, color, emb):
        self.alliance = color
        self.appearance_embedding = emb
        self.team_number = None
        self.team_number_confirmed = False
        self.team_ocr_votes = None
        self.consecutive_tracker = None


def _unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_strong_match_three_times_locks_identity():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("1787", "blue", e, 0.9)
    t = FakeTrack("blue", _unit(1, 0.01, 0))  # ~identical -> strong, no rival
    verdict = None
    for _ in range(3):
        verdict, team, sim = _apply_gallery_vote(t, g)
    assert verdict == "confirm"
    assert t.team_number_confirmed is True
    assert t.team_number == "1787"


def test_ambiguous_match_does_not_confirm():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    g.add("254", "blue", _unit(1, 0.02, 0), 0.9)  # two near-identical teams
    t = FakeTrack("blue", _unit(1, 0.01, 0))
    for _ in range(5):
        _apply_gallery_vote(t, g)
    assert t.team_number_confirmed is False  # gap guard prevents a confirm vote


def test_no_match_when_below_supplement():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9)
    t = FakeTrack("blue", _unit(0, 1, 0))  # orthogonal -> weak
    verdict, team, sim = _apply_gallery_vote(t, g)
    assert verdict == "none"
    assert (t.team_ocr_votes or {}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gallery_match_logic.py -v`
Expected: FAIL with `ImportError: cannot import name '_apply_gallery_vote'`.

- [ ] **Step 3: Implement the module-level vote helper + the method**

Add a module-level function in `web/app.py` (so it is unit-testable without an `InferenceJob`):

```python
def _apply_gallery_vote(robot, gallery,
                        confirm_sim=GALLERY_CONFIRM_SIM, confirm_gap=GALLERY_CONFIRM_GAP,
                        supplement_sim=GALLERY_SUPPLEMENT_SIM,
                        strong_w=GALLERY_STRONG_VOTE_WEIGHT, weak_w=GALLERY_WEAK_VOTE_WEIGHT):
    """Cast a gallery-derived vote into a track's existing vote/lock state.

    Returns (verdict, team, sim) where verdict is 'confirm'|'supplement'|'none'.
    Skips confirmed tracks (never overrides a lock)."""
    if gallery is None or robot.team_number_confirmed or robot.appearance_embedding is None:
        return ("none", None, 0.0)
    color = robot.alliance if robot.alliance in ("red", "blue") else None
    ranked = gallery.query(robot.appearance_embedding, color)
    verdict, team, sim = classify_match(
        ranked, confirm_sim=confirm_sim, confirm_gap=confirm_gap, supplement_sim=supplement_sim)
    if verdict == "none" or not team:
        return ("none", None, sim)
    weight = strong_w if verdict == "confirm" else weak_w
    if robot.team_ocr_votes is None:
        robot.team_ocr_votes = {}
    robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + weight
    if verdict == "confirm":
        if robot.consecutive_tracker is None:
            robot.consecutive_tracker = ConsecutiveValueTracker(n_consecutive=3)
        locked_team, confirmed = robot.consecutive_tracker.update(team)
        if confirmed and locked_team:
            robot.team_number = locked_team
            robot.team_number_confirmed = True
    return (verdict, team, sim)
```

> Note: `_apply_gallery_vote` does NOT call `_deposit_to_gallery` — gallery-driven confirmations must not seed the gallery (self-reinforcement guard from the spec).

Add the `InferenceJob` method that applies it across unconfirmed tracks and annotates the UI payload:

```python
def apply_gallery_matches(self, ocr_crops: list) -> None:
    """Query the gallery for unconfirmed tracks, cast votes, and annotate any
    matching ocr_crops cards so the UI distinguishes appearance recovery."""
    if self.gallery is None:
        return
    by_slot = {c["robot_id"]: c for c in ocr_crops}
    for slot_id, robot in self.robot_tracks.items():
        if robot.team_number_confirmed:
            continue
        verdict, team, sim = _apply_gallery_vote(robot, self.gallery)
        if verdict == "none" or not team:
            continue
        card = by_slot.get(slot_id)
        if card is not None and not card.get("matched"):
            card["matched"] = team
            card["via"] = "gallery"
            card["gallery_sim"] = round(sim, 3)
        if robot.team_number_confirmed:
            for event in self.scoring_events:
                if event.robot_id == slot_id:
                    event.team_number = robot.team_number
            print(f"  Gallery-confirmed: Robot {slot_id} = Team {robot.team_number} "
                  f"(appearance {sim:.2f})")
```

> `_apply_gallery_vote` returns the `(verdict, team, sim)` it already computed, so the method casts the vote and annotates the card from a single query — no double `classify_match`.

Tag OCR-sourced cards with `via: "ocr"`. In `read_fused_bumper_numbers`, in the `ocr_crops.append({...})` dict (line 926-937), add `"via": "ocr"`.

Call `apply_gallery_matches` at the end of `read_fused_bumper_numbers`, right after `self.last_ocr_crops = ocr_crops` (line 971), but pass `ocr_crops` before assigning so annotations land:

```python
        self.apply_gallery_matches(ocr_crops)
        self.last_ocr_crops = ocr_crops
        self._ocr_crops_dirty = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gallery_match_logic.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -m "not model"`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add web/app.py tests/test_gallery_match_logic.py
git commit -m "feat(gallery): query gallery & vote for unconfirmed tracks (TDD)"
```

---

## Task 5: UI — distinguish appearance-recovered identities

Render an "appearance" badge and the gallery similarity on cards recovered via the gallery, so a human reviewer can tell them from directly-read numbers.

**Files:**
- Modify: `web/static/app.js:901-925` (`updateVlmCrops`)
- Modify: `web/static/style.css` (near `.confidence-badge`, ~line 676)

- [ ] **Step 1: Update the card renderer**

In `web/static/app.js`, inside the `entries.map(crop => { ... })` block, after the `confidence` line, add a via-badge and adjust the subtext:

```javascript
            const viaGallery = crop.via === 'gallery';
            const galleryBadge = viaGallery
                ? `<span class="appearance-badge">appearance ${Number.isFinite(crop.gallery_sim) ? Math.round(crop.gallery_sim * 100) + '%' : ''}</span>`
                : '';
            const subtext = viaGallery
                ? `recovered by appearance`
                : `${confidence} confidence`;
```

Then in the returned template, replace the result/subtext block:

```javascript
                        <div class="vlm-result ${resultClass}">
                            ${this.escapeHtml(crop.slot)} ${this.escapeHtml(raw)} &gt; ${this.escapeHtml(matched)} ${galleryBadge}
                        </div>
                        <div class="vlm-subtext">${subtext}</div>
```

- [ ] **Step 2: Add the badge style**

In `web/static/style.css`, near `.confidence-badge` (~line 676), add:

```css
.appearance-badge {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 6px;
    font-size: 0.7rem;
    background: rgba(139, 92, 246, 0.2);
    color: #c4b5fd;
}
```

- [ ] **Step 3: Bump the cache-bust version**

The frontend uses a cache-bust query param (search `app.js?v=` in `web/static/index.html`). Increment it so browsers reload the changed JS/CSS.

- [ ] **Step 4: Verify rendering manually**

Use the @run skill (or the project's run pattern) to launch the app and process a short clip on an event where the gallery has at least one confirmed team. Confirm:
- A directly-read card still shows `… > 1787` and `95% confidence` (no badge).
- A track whose number is unreadable but appearance-matched shows `… > 1787 [appearance 86%]` and `recovered by appearance`.

Expected: appearance-recovered cards are visually distinct; OCR cards unchanged.

- [ ] **Step 5: Commit**

```bash
git add web/static/app.js web/static/style.css web/static/index.html
git commit -m "feat(gallery): UI badge distinguishing appearance-recovered identities"
```

---

## Final verification

- [ ] Run the whole suite: `.venv/bin/python -m pytest tests/ -m "not model"` → green.
- [ ] Run model-dependent tests if a GPU/sample data is present: `.venv/bin/python -m pytest tests/ -m model` (optional).
- [ ] Spot-check a real match end-to-end (Task 5 Step 4) and confirm gallery exemplars persist across two matches at the same event (run match A, confirm a team; run match B for the same `event_key` and verify the log prints a non-zero "Appearance gallery loaded: N exemplars").

## Notes on scope / non-goals

- No pose/side classifier — the diverse exemplar set (dedup rule) covers viewpoints implicitly.
- No cross-event gallery — everything is keyed by `event_key`.
- Gallery never overrides an existing lock; a real high-confidence OCR read always wins; gallery confirmation requires sustained agreement (3 passes) via the existing lock.
- Gallery-driven confirmations never deposit (no self-reinforcement).

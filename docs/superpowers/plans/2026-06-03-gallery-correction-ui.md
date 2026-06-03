# View & Correct Gallery Identities — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users view the bumper crops a robot/exemplar was tagged with and correct the team number — fixing the match/replay and seeding human-verified gallery exemplars — plus a per-event gallery-manager page to curate exemplars.

**Architecture:** A shared foundation (capture top-3 crops+embeddings per robot per match; gallery gains `image`+`verified` columns and verified-weighted matching; replay rows finally store `event_key`) underpins two surfaces: a post-match review/correct panel in the replay player (Phase 1) and a gallery-manager page (Phase 2).

**Tech Stack:** Python, numpy, PIL (Pillow, already used), aiosqlite (`web/database.py`), the existing DINOv2 embedder, vanilla-JS SPA (`web/static/index.html` + `app.js` + `style.css`). Tests: pytest in `tests/`, run with `.venv/bin/python -m pytest` (base conda python has no pytest).

**Spec:** `docs/superpowers/specs/2026-06-03-gallery-correction-ui-design.md`

---

## Key facts for the implementer

- Tests live in `tests/` (NOT `web/tests/`), import via `web.*`. DB tests monkeypatch `database.DB_PATH` to a tmp file and call async fns with `asyncio.run(...)` (see `tests/test_gallery_db.py`, `tests/test_replay_db.py`).
- `web/field/appearance_gallery.py`: `_Exemplar` (`__slots__` at line ~16), `add` (~33), `query` (~51), `serialize` (~84), `from_rows` (~90), `classify_match` (~110). Reuses `web.embedder.cosine_similarity`.
- `web/database.py`: `SCHEMA` string ends ~line 110 (gallery table 100-109); `init_db` ALTER-migration loop ~106-122; gallery CRUD `insert_gallery_exemplar`/`get_gallery_exemplars` ~516-535; replay CRUD ~469-501.
- `web/app.py`: `GALLERY_*` constants ~218-226; `_apply_gallery_vote` (module fn) ~397; `apply_gallery_matches` ~886; `_deposit_to_gallery` ~866; `read_fused_bumper_numbers` ~913 (per-result loop 1001-1064; deposit sites 1041 & 1062; `crop_b64` and `conf` in scope there; `robot.appearance_embedding` now populated); finalize `insert_replay` 3422-3455 (the `"event_key": None` bug at **3445**); gallery flush 3458-3464; `/api/replays/{id}` endpoints ~5218+.
- `web/static/app.js`: replay open/render — `openReplay`-style loader fetches `/api/replays/{id}` (~1600-1651), `renderReplayFrame` (~1674), nav switch pattern `switchToReplaysView` (~1653). `index.html` has the replay player modal (~312-401) and nav buttons (`.nav-btn[data-view=...]`).
- PIL is available: `from PIL import Image`. `crop_b64` is **raw** base64 JPEG (no `data:` prefix); the UI prepends `data:image/jpeg;base64,`.

---

# FOUNDATION

## Task 1: Gallery verified tier (pure module)

**Files:**
- Modify: `web/field/appearance_gallery.py`
- Test: `tests/test_appearance_gallery.py` (add cases; keep existing)

- [ ] **Step 1: Write failing tests** (append to `tests/test_appearance_gallery.py`):

```python
def test_query_returns_verified_flag_and_sorts_verified_first_on_tie():
    g = AppearanceGallery()
    e = _unit(1, 0, 0)
    g.add("254", "red", e, 0.9, verified=False)
    g.add("1787", "red", e, 0.9, verified=True)  # identical sim -> verified wins tie
    ranked = g.query(e, "red")
    assert ranked[0][0] == "1787"      # verified team first
    assert ranked[0][2] is True        # 3-tuple: (team, sim, verified)
    assert len(ranked[0]) == 3


def test_verified_roundtrips_through_serialize_from_rows():
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9, verified=True)
    rows = g.serialize()
    assert rows[0]["verified"] is True
    g2 = AppearanceGallery.from_rows(rows)
    assert g2.query(_unit(1, 0, 0), "blue")[0][2] is True
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_appearance_gallery.py -q`
Expected: FAIL (`add()` got unexpected kwarg `verified`, and `query` returns 2-tuples).

- [ ] **Step 3: Implement** in `web/field/appearance_gallery.py`:

`_Exemplar`: add `verified` to `__slots__` and `__init__`:
```python
class _Exemplar:
    __slots__ = ("embedding", "legibility", "verified")

    def __init__(self, embedding, legibility, verified=False):
        self.embedding = np.array(embedding, dtype=np.float32, copy=True)
        self.legibility = float(legibility)
        self.verified = bool(verified)
```

`add`: accept `verified=False` and pass it through:
```python
    def add(self, team, color, embedding, legibility, verified=False) -> bool:
        ...
        bucket.append(_Exemplar(emb, legibility, verified))
        ...
```
(Dedup/cap logic unchanged. Eviction still by lowest legibility.)

`query`: track best-per-team sim AND verified, sort by sim desc with verified-first tiebreak, return 3-tuples:
```python
    def query(self, embedding, color):
        if embedding is None:
            return []
        emb = np.asarray(embedding, dtype=np.float32)
        best = {}  # team -> (sim, verified)
        for (team, c), bucket in self._sets.items():
            if color in ("red", "blue") and c != color:
                continue
            for ex in bucket:
                sim = cosine_similarity(emb, ex.embedding)
                cur = best.get(team)
                # take higher sim; on equal sim prefer a verified exemplar
                if cur is None or sim > cur[0] or (sim == cur[0] and ex.verified and not cur[1]):
                    best[team] = (sim, ex.verified)
        ranked = [(team, s, v) for team, (s, v) in best.items()]
        ranked.sort(key=lambda t: (t[1], t[2]), reverse=True)  # sim desc, verified-first on tie
        return ranked
```

`serialize`: emit `verified`:
```python
                rows.append({
                    "team": team, "alliance_color": color,
                    "embedding": ex.embedding.astype(np.float32).tobytes(),
                    "legibility": ex.legibility, "verified": ex.verified,
                })
```

`from_rows`: read `verified` (default False); also tolerate extra `id`/`image` keys (ignore them here):
```python
            g._sets.setdefault((r["team"], r["alliance_color"]), []).append(
                _Exemplar(emb, r.get("legibility") or 0.0, bool(r.get("verified"))))
```

`classify_match`: read positionally so it works on 3-tuples (logic unchanged):
```python
    if not ranked:
        return ("none", None, 0.0)
    team, sim = ranked[0][0], ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
```

- [ ] **Step 4: Run all gallery-module tests** (existing + new)

Run: `.venv/bin/python -m pytest tests/test_appearance_gallery.py -q`
Expected: PASS (existing positional asserts and serialize `>=` keyset still hold).

- [ ] **Step 5: Commit**

```bash
git add web/field/appearance_gallery.py tests/test_appearance_gallery.py
git commit -m "feat(gallery): verified exemplar tier (query 3-tuple, verified-first tiebreak)"
```

---

## Task 2: DB layer — new tables, gallery columns, CRUD

**Files:**
- Modify: `web/database.py`
- Test: `tests/test_gallery_correction_db.py` (new)

- [ ] **Step 1: Write failing tests** — create `tests/test_gallery_correction_db.py`:

```python
import asyncio
import numpy as np
import pytest
import web.database as database


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    asyncio.run(database.init_db())
    return database


def _b(*v): return np.array(v, dtype=np.float32).tobytes()


def test_match_robot_identity_and_crops_roundtrip(db):
    asyncio.run(db.upsert_match_robot_identity({
        "job_id": "job1", "slot": 3, "slot_label": "R1", "alliance": "red",
        "team_number": "2486", "confirmed": 1, "source": "ocr", "confidence": 0.95}))
    asyncio.run(db.insert_match_robot_crop({
        "job_id": "job1", "slot": 3, "legibility": 0.9,
        "embedding": _b(1, 0, 0), "image": b"\xff\xd8jpeg"}))
    ids = asyncio.run(db.get_match_identities("job1"))
    assert ids[0]["team_number"] == "2486" and ids[0]["slot"] == 3
    crops = asyncio.run(db.get_match_crops("job1", 3))
    assert len(crops) == 1 and crops[0]["image"] == b"\xff\xd8jpeg"


def test_identity_upsert_replaces(db):
    base = {"job_id": "j", "slot": 0, "slot_label": "B1", "alliance": "blue",
            "team_number": "111", "confirmed": 0, "source": "ocr", "confidence": 0.5}
    asyncio.run(db.upsert_match_robot_identity(base))
    asyncio.run(db.upsert_match_robot_identity({**base, "team_number": "222"}))
    ids = asyncio.run(db.get_match_identities("j"))
    assert len(ids) == 1 and ids[0]["team_number"] == "222"


def test_gallery_has_image_and_verified_columns_and_crud(db):
    asyncio.run(db.insert_gallery_exemplar({
        "event_key": "2026x", "team": "1787", "alliance_color": "blue",
        "embedding": _b(1, 0, 0), "legibility": 0.9, "image": b"img", "verified": 1}))
    rows = asyncio.run(db.get_gallery_exemplars("2026x"))
    assert rows[0]["verified"] == 1 and rows[0]["image"] == b"img" and "id" in rows[0]
    gid = rows[0]["id"]
    # relabel + verify toggle
    asyncio.run(db.update_gallery_exemplar(gid, team="254", verified=0))
    rows = asyncio.run(db.get_gallery_exemplars("2026x"))
    assert rows[0]["team"] == "254" and rows[0]["verified"] == 0
    # delete
    asyncio.run(db.delete_gallery_exemplar(gid))
    assert asyncio.run(db.get_gallery_exemplars("2026x")) == []


def test_gallery_migration_adds_columns_to_old_table(tmp_path, monkeypatch):
    import sqlite3
    dbfile = tmp_path / "old.db"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE appearance_gallery (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "event_key TEXT, team TEXT, alliance_color TEXT, embedding BLOB, legibility REAL, "
                "created_at TEXT)")
    con.commit(); con.close()
    monkeypatch.setattr(database, "DB_PATH", dbfile)
    asyncio.run(database.init_db())  # must ALTER-add image + verified without error
    asyncio.run(database.insert_gallery_exemplar({
        "event_key": "e", "team": "9", "alliance_color": "red",
        "embedding": _b(0, 1, 0), "legibility": 0.5, "image": b"i", "verified": 1}))
    assert asyncio.run(database.get_gallery_exemplars("e"))[0]["verified"] == 1
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_gallery_correction_db.py -q`
Expected: FAIL (missing functions / columns).

- [ ] **Step 3: Implement** in `web/database.py`:

(a) Append two tables to the `SCHEMA` string (before the closing `"""`):
```sql
CREATE TABLE IF NOT EXISTS match_robot_identity (
    job_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    slot_label TEXT,
    alliance TEXT,
    team_number TEXT,
    confirmed INTEGER DEFAULT 0,
    source TEXT,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (job_id, slot)
);
CREATE TABLE IF NOT EXISTS match_robot_crop (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    legibility REAL,
    embedding BLOB,
    image BLOB
);
CREATE INDEX IF NOT EXISTS idx_match_crop ON match_robot_crop(job_id, slot);
```

(b) In `init_db`, extend the ALTER-migration loop to add gallery columns. Find the existing `for col_def in (...)` loop and add a gallery one (mirroring it):
```python
        for col_def in ("image BLOB", "verified INTEGER DEFAULT 0"):
            try:
                await db.execute(f"ALTER TABLE appearance_gallery ADD COLUMN {col_def}")
            except Exception:
                pass
```

(c) Update the gallery `SCHEMA` `CREATE TABLE` for fresh installs to include `image BLOB` and `verified INTEGER DEFAULT 0` (so new DBs have them without relying on ALTER).

(d) Update `insert_gallery_exemplar` to write `image` + `verified`:
```python
async def insert_gallery_exemplar(row):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO appearance_gallery "
            "(event_key, team, alliance_color, embedding, legibility, image, verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (row["event_key"], row["team"], row["alliance_color"], row["embedding"],
             row.get("legibility"), row.get("image"), int(row.get("verified", 0))))
        await db.commit()
```

(e) Extend `get_gallery_exemplars` SELECT to include `id, image, verified`:
```python
        cursor = await db.execute(
            "SELECT id, team, alliance_color, embedding, legibility, image, verified "
            "FROM appearance_gallery WHERE event_key = ? ORDER BY id", (event_key,))
```

(f) Add new CRUD functions (after the gallery section):
```python
async def update_gallery_exemplar(gid, team=None, alliance_color=None, verified=None):
    sets, vals = [], []
    if team is not None: sets.append("team = ?"); vals.append(team)
    if alliance_color is not None: sets.append("alliance_color = ?"); vals.append(alliance_color)
    if verified is not None: sets.append("verified = ?"); vals.append(int(verified))
    if not sets: return
    vals.append(gid)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE appearance_gallery SET {', '.join(sets)} WHERE id = ?", vals)
        await db.commit()


async def delete_gallery_exemplar(gid):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM appearance_gallery WHERE id = ?", (gid,))
        await db.commit()


async def upsert_match_robot_identity(row):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO match_robot_identity "
            "(job_id, slot, slot_label, alliance, team_number, confirmed, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row["job_id"], row["slot"], row.get("slot_label"), row.get("alliance"),
             row.get("team_number"), int(row.get("confirmed", 0)), row.get("source"),
             row.get("confidence")))
        await db.commit()


async def insert_match_robot_crop(row):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO match_robot_crop (job_id, slot, legibility, embedding, image) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["job_id"], row["slot"], row.get("legibility"), row.get("embedding"),
             row.get("image")))
        await db.commit()


async def get_match_identities(job_id):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM match_robot_identity WHERE job_id = ? ORDER BY slot", (job_id,))
        return [dict(r) for r in await cur.fetchall()]


async def get_match_crops(job_id, slot):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM match_robot_crop WHERE job_id = ? AND slot = ? ORDER BY legibility DESC",
            (job_id, slot))
        return [dict(r) for r in await cur.fetchall()]
```

> NOTE: confirm `get_gallery_exemplars`'s extra columns don't break Task-1 `from_rows` — it uses `.get()` for `verified` and ignores `id`/`image`, so it's safe.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_gallery_correction_db.py tests/test_gallery_db.py -q`
Expected: PASS (new + existing gallery DB tests).

- [ ] **Step 5: Commit**

```bash
git add web/database.py tests/test_gallery_correction_db.py
git commit -m "feat(gallery): match-identity/crop tables + gallery image/verified columns + CRUD"
```

---

## Task 3: Verified-weighted voting in app.py

**Files:**
- Modify: `web/app.py` (constants ~218-226; `_apply_gallery_vote` ~397; `from_rows` load passes verified already via Task 1)
- Test: `tests/test_gallery_match_logic.py` (add a case)

- [ ] **Step 1: Add failing test** (append to `tests/test_gallery_match_logic.py`):

```python
def test_verified_match_uses_higher_weight():
    from web.app import _apply_gallery_vote, GALLERY_VERIFIED_VOTE_WEIGHT
    g = AppearanceGallery()
    g.add("1787", "blue", _unit(1, 0, 0), 0.9, verified=True)
    t = FakeTrack("blue", _unit(1, 0.01, 0))
    verdict, team, sim = _apply_gallery_vote(t, g)
    assert verdict == "confirm" and team == "1787"
    assert t.team_ocr_votes["1787"] == GALLERY_VERIFIED_VOTE_WEIGHT  # verified weight, not strong_w
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_gallery_match_logic.py::test_verified_match_uses_higher_weight -q`
Expected: FAIL (`GALLERY_VERIFIED_VOTE_WEIGHT` undefined / weight differs).

- [ ] **Step 3: Implement**

(a) Add the constant near the other `GALLERY_*` (web/app.py ~226):
```python
GALLERY_VERIFIED_VOTE_WEIGHT = float(os.environ.get("GALLERY_VERIFIED_VOTE_WEIGHT", "2.0"))
```

(b) In `_apply_gallery_vote` (web/app.py ~397), read the winning team's verified flag and pick weight. The `ranked` list now holds 3-tuples; `classify_match` returns `(verdict, team, sim)`:
```python
    ranked = gallery.query(robot.appearance_embedding, color)
    verdict, team, sim = classify_match(
        ranked, confirm_sim=confirm_sim, confirm_gap=confirm_gap, supplement_sim=supplement_sim)
    if verdict == "none" or not team:
        return ("none", None, sim)
    best_verified = bool(ranked[0][2]) if ranked else False
    if best_verified:
        weight = GALLERY_VERIFIED_VOTE_WEIGHT
    else:
        weight = strong_w if verdict == "confirm" else weak_w
```
(Add `GALLERY_VERIFIED_VOTE_WEIGHT` to the function's default kwargs list is unnecessary — reference the module global directly inside the body.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_gallery_match_logic.py -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_gallery_match_logic.py
git commit -m "feat(gallery): verified exemplars cast a higher vote weight"
```

---

## Task 4: Thumbnails + store crop image in deposits

**Files:**
- Modify: `web/app.py` (`_deposit_to_gallery` ~866, deposit sites 1041 & 1062, add a thumbnail helper)
- Test: `tests/test_thumbnail.py` (new)

- [ ] **Step 1: Write failing test** — create `tests/test_thumbnail.py`:

```python
import base64, io
from PIL import Image
from web.app import _thumbnail_bytes


def _jpeg_b64(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_thumbnail_bounds_height_and_returns_jpeg_bytes():
    out = _thumbnail_bytes(_jpeg_b64(400, 300), max_h=128)
    assert isinstance(out, (bytes, bytearray)) and len(out) > 0
    img = Image.open(io.BytesIO(out))
    assert img.height <= 128
    assert img.format == "JPEG"


def test_thumbnail_handles_bad_input():
    assert _thumbnail_bytes(None) is None
    assert _thumbnail_bytes("not-base64!!") is None
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_thumbnail.py -q`
Expected: FAIL (`_thumbnail_bytes` undefined).

- [ ] **Step 3: Implement**

(a) Add a module-level helper in `web/app.py` (near other helpers; `base64`, `io`, `Image` are imported — confirm `from PIL import Image` exists, else add):
```python
def _thumbnail_bytes(crop_b64, max_h=128):
    """Decode a raw-base64 JPEG bumper crop and return a small JPEG thumbnail (bytes),
    bounded to max_h pixels tall. Returns None on bad/empty input."""
    if not crop_b64:
        return None
    try:
        raw = base64.b64decode(crop_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.height > max_h:
            w = max(1, int(img.width * max_h / img.height))
            img = img.resize((w, max_h))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=80)
        return out.getvalue()
    except Exception:
        return None
```

(b) Extend `_deposit_to_gallery` to accept and persist an image (web/app.py ~866):
```python
    def _deposit_to_gallery(self, robot, legibility=0.0, image_bytes=None):
        ...
        self._pending_gallery_rows.append({
            "event_key": self.event_key, "team": robot.team_number,
            "alliance_color": color,
            "embedding": np.asarray(robot.appearance_embedding, dtype=np.float32).tobytes(),
            "legibility": legibility,
            "image": image_bytes,
            "verified": 0,
        })
```
(Auto-deposits are `verified=0`.)

(c) Pass the thumbnail at the two deposit call sites in `read_fused_bumper_numbers`. At ~1041 and ~1062, `crop_b64` is in scope:
```python
                self._deposit_to_gallery(robot, legibility=conf,
                                         image_bytes=_thumbnail_bytes(crop_b64))
```

- [ ] **Step 4: Run to verify pass + import check**

Run: `.venv/bin/python -c "import web.app" && .venv/bin/python -m pytest tests/test_thumbnail.py -q`
Expected: import OK; tests PASS.

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_thumbnail.py
git commit -m "feat(gallery): thumbnail helper + store crop image on auto-deposit"
```

---

## Task 5: Capture top-3 crops per slot + finalize identity/crop rows + event_key fix

**Files:**
- Modify: `web/app.py` (`__init__` state; `read_fused_bumper_numbers` capture; finalize 3422-3464; the `"event_key": None` at 3445)

This task is integration wiring; verified by import + existing suite + the end-to-end re-run in Final Verification (no isolated unit test — the capture/finalize touch the heavy pipeline). Be precise with anchors; read the cited regions before editing.

- [ ] **Step 1: Add per-slot capture state in `__init__`**

After `self._pending_gallery_rows = []` add:
```python
        self._slot_crops: dict = {}   # slot_id -> list of {quality, crop_b64, embedding} (top-3)
        self._slot_source: dict = {}  # slot_id -> "ocr" | "gallery" (last confirm source)
```

- [ ] **Step 2: Add a capture helper** (method on `InferenceJob`, near `_deposit_to_gallery`):

```python
    def _capture_slot_crop(self, slot_id, crop_b64, quality, embedding):
        """Keep the top-3 (by quality) bumper crops per slot for post-match review.
        Stores a COPY of the embedding (field-path embeddings are EMA-updated in place)."""
        if not crop_b64:
            return
        emb = None if embedding is None else np.array(embedding, dtype=np.float32, copy=True)
        bucket = self._slot_crops.setdefault(slot_id, [])
        bucket.append({"quality": float(quality), "crop_b64": crop_b64, "embedding": emb})
        bucket.sort(key=lambda c: c["quality"], reverse=True)
        del bucket[3:]
```

- [ ] **Step 3: Call capture in the read loop**

In `read_fused_bumper_numbers`, in the per-result loop right after `conf = float(res.get("confidence", 0.0))` (web/app.py ~1006):
```python
            self._capture_slot_crop(slot_id, crop_b64, conf, robot.appearance_embedding)
```
Also record the confirm source: at the OCR lock site (~1056, inside `if confirmed and locked_team:`) add `self._slot_source[slot_id] = "ocr"`, and in `apply_gallery_matches` where it logs "Gallery-confirmed" add `self._slot_source[slot_id] = "gallery"`.

- [ ] **Step 4: Fix the event_key bug at finalize**

At web/app.py:3445 change:
```python
                        "event_key": None, "match_key": None, "comp_level": None,
```
to:
```python
                        "event_key": self.event_key, "match_key": None, "comp_level": None,
```

- [ ] **Step 5: Write identity/crop rows at finalize**

In the `finally:` block, after the gallery-flush try/except (web/app.py ~3464), add a new awaited block:
```python
            try:
                for slot_id, robot in self.robot_tracks.items():
                    crops = self._slot_crops.get(slot_id, [])
                    await database.upsert_match_robot_identity({
                        "job_id": self.job_id, "slot": slot_id,
                        "slot_label": self._slot_label(slot_id),
                        "alliance": robot.alliance, "team_number": robot.team_number,
                        "confirmed": int(bool(robot.team_number_confirmed)),
                        "source": self._slot_source.get(slot_id, "ocr"),
                        "confidence": (crops[0]["quality"] if crops else None),
                    })
                    for c in crops:
                        await database.insert_match_robot_crop({
                            "job_id": self.job_id, "slot": slot_id,
                            "legibility": c["quality"],
                            "embedding": (None if c["embedding"] is None
                                          else np.asarray(c["embedding"], dtype=np.float32).tobytes()),
                            "image": _thumbnail_bytes(c["crop_b64"]),
                        })
            except Exception as e:
                print(f"  Match-identity capture failed ({e})")
```

- [ ] **Step 6: Verify**

Run: `.venv/bin/python -c "import web.app" && .venv/bin/python -m pytest tests/ -q -m "not model"`
Expected: import OK; full suite green (no regressions).

- [ ] **Step 7: Commit**

```bash
git add web/app.py
git commit -m "feat(gallery): capture top-3 crops/slot, persist match identities, store event_key on replay"
```

---

> **Frontend security (applies to Tasks 7 & 9):** all interpolated text that originates
> from the DB or user input — team numbers, slot labels, alliance — MUST go through
> `this.escapeHtml(...)` before being placed in an `innerHTML` template (the existing
> `updateVlmCrops` does this). Image `src` values are server-generated `data:image/jpeg`
> URLs (base64 of stored bytes), not user text, so they're safe. Do not interpolate raw
> team/label strings into `innerHTML` without escaping — that's an XSS vector.

# PHASE 1 — Post-match review & correction

## Task 6: Backend endpoints — identities + correct

**Files:**
- Modify: `web/app.py` (new endpoints near `/api/replays/{id}` ~5218+; small helpers in `web/database.py` if needed)
- Test: `tests/test_correction_api.py` (new)

- [ ] **Step 1: Write failing tests** — create `tests/test_correction_api.py` using FastAPI's TestClient (match the style of `tests/test_replay_api.py`; read it first for the client fixture + DB monkeypatch pattern):

```python
# Mirror tests/test_replay_api.py setup (TestClient, monkeypatched database.DB_PATH).
# Seed: a replays row (event_key="2026x", blue_teams/red_teams JSON), a match_robot_identity
# row for slot 0 (alliance blue, team "111"), and one match_robot_crop for slot 0
# (embedding bytes + image bytes).
#
# test_get_identities_returns_crops_as_data_urls_no_embeddings:
#   GET /api/replays/{job}/identities -> 200; slot 0 present with team "111";
#   crops are "data:image/jpeg;base64,..." strings; response contains NO 'embedding' key.
#
# test_correct_updates_identity_and_mints_verified_exemplar:
#   POST /api/replays/{job}/correct {slot:0, team_number:"999"} -> 200;
#   get_match_identities shows team "999"; get_gallery_exemplars("2026x") contains a
#   verified=1 row for team "999" (one per crop); replays row blue_teams now includes "999".
#
# test_correct_skips_gallery_when_no_event_key:
#   replay with event_key=None: POST correct -> 200; identity updated; gallery still empty.
```
(Write these out concretely against the real helpers; assert on DB state via `database.get_match_identities`, `database.get_gallery_exemplars`, `database.get_replay`.)

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_correction_api.py -q`
Expected: FAIL (404 / endpoints missing).

- [ ] **Step 3: Implement endpoints** in `web/app.py` (near the other `/api/replays/{replay_id}` routes):

```python
@app.get("/api/replays/{job_id}/identities")
async def get_replay_identities(job_id: str):
    ids = await database.get_match_identities(job_id)
    out = []
    for r in ids:
        crops = await database.get_match_crops(job_id, r["slot"])
        out.append({
            "slot": r["slot"], "slot_label": r["slot_label"], "alliance": r["alliance"],
            "team_number": r["team_number"], "confirmed": bool(r["confirmed"]),
            "source": r["source"], "confidence": r["confidence"],
            "crops": [
                "data:image/jpeg;base64," + base64.b64encode(c["image"]).decode()
                for c in crops if c["image"] is not None
            ],
        })
    return {"job_id": job_id, "robots": out}


@app.post("/api/replays/{job_id}/correct")
async def correct_replay_identity(job_id: str, request: Request):
    body = await request.json()
    slot = int(body["slot"]); team = str(body["team_number"]).strip()
    ids = {r["slot"]: r for r in await database.get_match_identities(job_id)}
    rec = ids.get(slot)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "slot not found"})
    alliance = rec["alliance"]
    # 1. update identity
    await database.upsert_match_robot_identity({**rec, "job_id": job_id, "slot": slot,
                                                "team_number": team})
    # 2. update the replay's team lists for display
    replay = await database.get_replay(job_id)
    minted = 0
    if replay is not None:
        import json as _json
        key = "blue_teams" if alliance == "blue" else "red_teams"
        try:
            teams = _json.loads(replay.get(key) or "[]")
        except Exception:
            teams = []
        # replace this slot's old team with the corrected one (best-effort positional)
        old = rec["team_number"]
        teams = [team if str(t) == str(old) else t for t in teams]
        if team not in [str(t) for t in teams]:
            teams.append(team)
        await database.update_replay_teams(job_id, key, _json.dumps(teams))
        # 3. mint verified exemplars from this slot's crops (needs event_key + color)
        event_key = replay.get("event_key")
        color = alliance if alliance in ("red", "blue") else None
        if event_key and color:
            for c in await database.get_match_crops(job_id, slot):
                if c["embedding"] is None:
                    continue
                await database.insert_gallery_exemplar({
                    "event_key": event_key, "team": team, "alliance_color": color,
                    "embedding": c["embedding"], "legibility": c["legibility"],
                    "image": c["image"], "verified": 1})
                minted += 1
    return {"status": "ok", "slot": slot, "team_number": team, "verified_exemplars": minted}
```

Add a tiny DB helper `update_replay_teams(job_id, key, value)` in `web/database.py` (key is validated to be `blue_teams`/`red_teams` by the caller):
```python
async def update_replay_teams(replay_id, key, value):
    assert key in ("blue_teams", "red_teams")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE replays SET {key} = ? WHERE id = ?", (value, replay_id))
        await db.commit()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_correction_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/app.py web/database.py tests/test_correction_api.py
git commit -m "feat(correction): /identities + /correct endpoints (mint verified exemplars)"
```

---

## Task 7: Frontend — replay review/correct panel

**Files:**
- Modify: `web/static/index.html` (add a "Review identities" panel inside the replay player modal ~312-401), `web/static/app.js` (fetch identities in the replay loader ~1600-1651; render + save), `web/static/style.css`.

No automated tests (frontend). Verify with `node --check web/static/app.js`, the diff, and the manual end-to-end in Final Verification.

- [ ] **Step 1: Add the panel container to `index.html`**

Inside the replay player modal (after the robot tables / timeline; read ~312-401 to place it), add:
```html
<div id="replayReviewCard" class="card" style="display:none">
  <h3>Review &amp; correct identities</h3>
  <div id="replayReviewGrid" class="review-grid"></div>
</div>
```

- [ ] **Step 2: Fetch + render in `app.js`**

In the replay loader (where `/api/replays/${id}` is fetched, ~1606), after priming frame 0, add:
```javascript
            this.loadReplayIdentities(id);
```
Add the methods (near `renderReplayFrame`):
```javascript
    async loadReplayIdentities(id) {
        this._replayId = id;
        const card = document.getElementById('replayReviewCard');
        const grid = document.getElementById('replayReviewGrid');
        if (!card || !grid) return;
        try {
            const resp = await fetch(`/api/replays/${id}/identities`);
            if (!resp.ok) { card.style.display = 'none'; return; }
            const data = await resp.json();
            const robots = data.robots || [];
            if (robots.length === 0) { card.style.display = 'none'; return; }
            grid.innerHTML = robots.map(r => {
                const imgs = (r.crops || []).map(src =>
                    `<img class="review-crop" src="${src}" alt="${r.slot_label} crop">`).join('')
                    || '<span class="empty-state">no image</span>';
                const badge = r.source === 'gallery' ? '<span class="appearance-badge">appearance</span>' : '';
                return `
                <div class="review-robot" data-slot="${r.slot}">
                    <div class="review-imgs">${imgs}</div>
                    <div class="review-meta">
                        <strong>${this.escapeHtml(r.slot_label || ('slot ' + r.slot))}</strong> ${badge}
                        <input class="review-team" type="text" value="${this.escapeHtml(r.team_number || '')}"
                               placeholder="team #">
                        <button class="review-save" data-slot="${r.slot}">Save</button>
                        <span class="review-status"></span>
                    </div>
                </div>`;
            }).join('');
            grid.querySelectorAll('.review-save').forEach(btn => {
                btn.addEventListener('click', () => this.saveIdentityCorrection(btn));
            });
            card.style.display = 'block';
        } catch (e) {
            card.style.display = 'none';
        }
    }

    async saveIdentityCorrection(btn) {
        const wrap = btn.closest('.review-robot');
        const slot = Number(btn.dataset.slot);
        const team = wrap.querySelector('.review-team').value.trim();
        const status = wrap.querySelector('.review-status');
        status.textContent = '…';
        try {
            const resp = await fetch(`/api/replays/${this._replayId}/correct`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ slot, team_number: team }),
            });
            const data = await resp.json();
            status.textContent = resp.ok
                ? `✓ verified (${data.verified_exemplars || 0})` : `✗ ${data.error || 'error'}`;
        } catch (e) {
            status.textContent = '✗ ' + e.message;
        }
    }
```

- [ ] **Step 3: Style in `style.css`** (reuse `.appearance-badge` from the earlier gallery work):
```css
.review-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
.review-robot { border: 1px solid var(--border, #243); border-radius: 8px; padding: 8px; }
.review-imgs { display: flex; gap: 4px; flex-wrap: wrap; min-height: 40px; }
.review-crop { height: 56px; border-radius: 4px; }
.review-meta { display: flex; align-items: center; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
.review-team { width: 80px; }
.review-status { font-size: 0.8rem; color: var(--text-secondary, #9bb); }
```

- [ ] **Step 4: Bump cache-bust version** in `index.html` (the `?v=` on app.js/style.css).

- [ ] **Step 5: Verify**

Run: `node --check web/static/app.js` → SYNTAX OK. Grep that `replayReviewCard` exists in both index.html and app.js, and the version bumped.

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html web/static/app.js web/static/style.css
git commit -m "feat(correction): replay review panel to view crops and correct team numbers"
```

---

# PHASE 2 — Gallery manager page

## Task 8: Backend — gallery manager endpoints

**Files:**
- Modify: `web/app.py` (new `/api/gallery` routes), `web/database.py` (a list-by-event-with-id helper if `get_gallery_exemplars` isn't enough — it now returns `id`/`image`/`verified`, so reuse it)
- Test: `tests/test_gallery_manager_api.py` (new)

- [ ] **Step 1: Write failing tests** (TestClient style):

```python
# Seed two gallery rows for event "2026x" (teams "1787" blue verified=0, "254" red verified=1).
#
# test_list_gallery_groups_and_images:
#   GET /api/gallery?event_key=2026x -> 200; returns exemplars with id, team,
#   alliance_color, verified, and image as data-url; grouped or flat list with those fields.
#
# test_update_gallery_relabel_and_verify:
#   POST /api/gallery/{id} {team:"1799", verified:true} -> 200; get_gallery_exemplars reflects it.
#
# test_delete_gallery_exemplar:
#   DELETE /api/gallery/{id} -> 200; row gone.
```

- [ ] **Step 2: Run to verify fail** → 404s.

- [ ] **Step 3: Implement** in `web/app.py`:
```python
@app.get("/api/gallery")
async def list_gallery(event_key: str):
    rows = await database.get_gallery_exemplars(event_key)
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "team": r["team"], "alliance_color": r["alliance_color"],
            "verified": bool(r["verified"]), "legibility": r["legibility"],
            "image": ("data:image/jpeg;base64," + base64.b64encode(r["image"]).decode())
                      if r.get("image") else None,
        })
    return {"event_key": event_key, "exemplars": out}


@app.post("/api/gallery/{gid}")
async def update_gallery(gid: int, request: Request):
    body = await request.json()
    await database.update_gallery_exemplar(
        gid, team=body.get("team"), alliance_color=body.get("alliance_color"),
        verified=body.get("verified"))
    return {"status": "ok", "id": gid}


@app.delete("/api/gallery/{gid}")
async def remove_gallery(gid: int):
    await database.delete_gallery_exemplar(gid)
    return {"status": "deleted", "id": gid}
```

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit**
```bash
git add web/app.py tests/test_gallery_manager_api.py
git commit -m "feat(gallery): manager endpoints (list/relabel/verify/delete exemplars)"
```

---

## Task 9: Frontend — Gallery manager view

**Files:**
- Modify: `web/static/index.html` (new nav button `data-view="gallery"` + a `galleryView` div), `web/static/app.js` (nav switch + render), `web/static/style.css`.

- [ ] **Step 1: Add nav button + view container in `index.html`** (mirror the existing `data-view="replays"` button and `replaysView` div ~76-91 / view divs):
```html
<button class="nav-btn" data-view="gallery">Gallery</button>
...
<div id="galleryView" style="display:none">
  <div class="card">
    <h2>Appearance gallery</h2>
    <input id="galleryEventKey" type="text" placeholder="event key (e.g. 2026x)">
    <button id="galleryLoadBtn">Load</button>
  </div>
  <div id="galleryGrid" class="review-grid"></div>
</div>
```

- [ ] **Step 2: Wire nav + render in `app.js`** (mirror `switchToReplaysView` ~1653 and the nav handler ~76-119):
```javascript
    switchToGalleryView() {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelector('.nav-btn[data-view="gallery"]')?.classList.add('active');
        ['scorerView','replaysView','historyView'].forEach(v => {
            const el = document.getElementById(v); if (el) el.style.display = 'none';
        });
        document.getElementById('galleryView').style.display = 'block';
    }

    async loadGallery() {
        const key = document.getElementById('galleryEventKey').value.trim();
        const grid = document.getElementById('galleryGrid');
        if (!key) { grid.innerHTML = '<p class="empty-state">Enter an event key</p>'; return; }
        const resp = await fetch(`/api/gallery?event_key=${encodeURIComponent(key)}`);
        const data = await resp.json();
        const ex = data.exemplars || [];
        if (ex.length === 0) { grid.innerHTML = '<p class="empty-state">No exemplars</p>'; return; }
        grid.innerHTML = ex.map(e => `
            <div class="review-robot" data-id="${e.id}">
                <div class="review-imgs">${e.image ? `<img class="review-crop" src="${e.image}">` : '<span class="empty-state">no image</span>'}</div>
                <div class="review-meta">
                    <input class="g-team" type="text" value="${this.escapeHtml(e.team || '')}" style="width:70px">
                    <select class="g-color"><option value="blue"${e.alliance_color==='blue'?' selected':''}>blue</option><option value="red"${e.alliance_color==='red'?' selected':''}>red</option></select>
                    <label><input type="checkbox" class="g-verified"${e.verified?' checked':''}> verified</label>
                    <button class="g-save" data-id="${e.id}">Save</button>
                    <button class="g-del" data-id="${e.id}">Delete</button>
                </div>
            </div>`).join('');
        grid.querySelectorAll('.g-save').forEach(b => b.addEventListener('click', () => this.saveGalleryExemplar(b)));
        grid.querySelectorAll('.g-del').forEach(b => b.addEventListener('click', () => this.deleteGalleryExemplar(b)));
    }

    async saveGalleryExemplar(btn) {
        const wrap = btn.closest('.review-robot');
        await fetch(`/api/gallery/${btn.dataset.id}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                team: wrap.querySelector('.g-team').value.trim(),
                alliance_color: wrap.querySelector('.g-color').value,
                verified: wrap.querySelector('.g-verified').checked,
            }),
        });
        this.loadGallery();
    }

    async deleteGalleryExemplar(btn) {
        await fetch(`/api/gallery/${btn.dataset.id}`, { method: 'DELETE' });
        this.loadGallery();
    }
```
Wire the nav button to `switchToGalleryView` (find where `data-view` clicks are dispatched ~76-119 and add a `gallery` case), and `galleryLoadBtn` click → `this.loadGallery()` in the init/bind section.

- [ ] **Step 3: Bump cache-bust version** in `index.html`.

- [ ] **Step 4: Verify**: `node --check web/static/app.js` → SYNTAX OK; grep `galleryView`/`switchToGalleryView` present.

- [ ] **Step 5: Commit**
```bash
git add web/static/index.html web/static/app.js web/static/style.css
git commit -m "feat(gallery): per-event gallery manager view (relabel/verify/delete)"
```

---

## Final verification

- [ ] `.venv/bin/python -m pytest tests/ -q -m "not model"` → green.
- [ ] `node --check web/static/app.js` → SYNTAX OK.
- [ ] **End-to-end (requires server restart — no `--reload`):** restart `python web/app.py`; run a match on a calibrated event. Confirm: match log shows `Appearance gallery loaded: N`; after finishing, `match_robot_identity`/`match_robot_crop` rows exist for the job; open the replay → the Review panel shows bumper crops; correct a wrong team → `appearance_gallery` gains a `verified=1` row and the replay's team list updates; open the Gallery view for the event → thumbnails appear and relabel/verify/delete work.

## Notes / non-goals
- Corrections do NOT propagate to the separately-saved scored-match (`match_robots`/`scoring_events`) flow — no replay→match linkage exists; users edit those in `/api/match/save`.
- No DB-side pruning of gallery images (bounded only on load by the per-(team,color) cap); a few MB/event is acceptable.

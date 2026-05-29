# Embedding-Based Robot Re-Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the HSV color-histogram appearance descriptor with DINOv2 embeddings that feed embedding-aware track continuity and a Hungarian one-to-one identity assignment, eliminating ID swaps, wrong team-number assignment, and duplicate team numbers.

**Architecture:** A new FRC-agnostic `web/embedder.py` (DINOv2 wrapper) produces L2-normalized embeddings. Two pure decision functions in a new `web/reid.py` consume them: `choose_canonical_match` (continuity, with a similarity floor) and `assign_identities` (Hungarian matching of tracks→calibration references, anchored by OCR locks, constrained by alliance color). `web/app.py` is edited at four seams to use these; the HSV functions are retired.

**Tech Stack:** Python 3.12, PyTorch + transformers (DINOv2, already installed), scipy (Hungarian, installed), OpenCV, pytest (added in Task 0).

**Spec:** `docs/superpowers/specs/2026-05-27-embedding-reid-robot-tracking-design.md`

**Conventions for this plan:**
- Virtualenv is `./.venv`. Run Python as `./.venv/bin/python` and pytest as `./.venv/bin/pytest`.
- Embedding dim is **384** (`facebook/dinov2-small`, CLS token).
- Tests that load the DINOv2 model are marked `@pytest.mark.model` and download weights once on first run (the only network step; fully offline after). Fast tests never load a model.

---

### Task 0: Test scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

- [ ] **Step 1: Install pytest into the venv**

Run: `./.venv/bin/pip install pytest`
Expected: installs pytest; `./.venv/bin/pytest --version` prints a version.

- [ ] **Step 2: Add pytest config and the `model` marker to `pyproject.toml`**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "model: tests that load the DINOv2 model (downloads weights once, slower)",
]
```

- [ ] **Step 3: Create empty `tests/__init__.py` and a conftest with shared fixtures**

`tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
"""Shared test fixtures for the re-ID test suite."""
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGES = sorted((PROJECT_ROOT / "data" / "roboflow_frc" / "valid" / "images").glob("*.jpg"))


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A real FRC frame (BGR) from the roboflow dataset."""
    if not SAMPLE_IMAGES:
        pytest.skip("no sample images available")
    img = cv2.imread(str(SAMPLE_IMAGES[0]))
    assert img is not None
    return img


@pytest.fixture
def second_frame() -> np.ndarray:
    """A different real FRC frame (BGR)."""
    if len(SAMPLE_IMAGES) < 2:
        pytest.skip("need at least two sample images")
    img = cv2.imread(str(SAMPLE_IMAGES[1]))
    assert img is not None
    return img
```

- [ ] **Step 4: Verify the suite collects with zero tests**

Run: `./.venv/bin/pytest -q`
Expected: "no tests ran" (exit code 5) — confirms config is valid.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/conftest.py
git commit -m "test: add pytest scaffolding and shared fixtures"
```

---

### Task 1: `RobotEmbedder` (DINOv2 wrapper)

**Files:**
- Create: `web/embedder.py`
- Test: `tests/test_embedder.py`

The pure `cosine_similarity` helper is tested without the model; `embed`/`embed_batch` are model-marked.

- [ ] **Step 1: Write the failing tests**

`tests/test_embedder.py`:

```python
import numpy as np
import pytest

from web.embedder import RobotEmbedder, cosine_similarity

EMBED_DIM = 384


def test_cosine_identical_vectors_is_one():
    v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    v = v / np.linalg.norm(v)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_handles_none():
    assert cosine_similarity(None, np.array([1.0])) == 0.0
    assert cosine_similarity(np.array([1.0]), None) == 0.0


def test_embed_returns_none_for_tiny_box(sample_frame):
    emb = RobotEmbedder()
    assert emb.embed(sample_frame, (0, 0, 3, 3)) is None  # < min crop size, no model load


@pytest.mark.model
def test_embed_returns_normalized_vector(sample_frame):
    emb = RobotEmbedder()
    h, w = sample_frame.shape[:2]
    v = emb.embed(sample_frame, (w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    assert v is not None
    assert v.shape == (EMBED_DIM,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.model
def test_same_crop_more_similar_than_different(sample_frame, second_frame):
    emb = RobotEmbedder()
    h, w = sample_frame.shape[:2]
    box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    v1 = emb.embed(sample_frame, box)
    v1b = emb.embed(sample_frame, box)
    v2 = emb.embed(second_frame, box)
    assert cosine_similarity(v1, v1b) == pytest.approx(1.0, abs=1e-3)
    assert cosine_similarity(v1, v2) < cosine_similarity(v1, v1b)


@pytest.mark.model
def test_embed_batch_matches_embed(sample_frame):
    emb = RobotEmbedder()
    h, w = sample_frame.shape[:2]
    boxes = [(w // 4, h // 4, 3 * w // 4, 3 * h // 4), (0, 0, w // 2, h // 2)]
    batched = emb.embed_batch(sample_frame, boxes)
    assert len(batched) == 2
    single = emb.embed(sample_frame, boxes[0])
    assert cosine_similarity(batched[0], single) == pytest.approx(1.0, abs=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_embedder.py -q -m "not model"`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.embedder'`.

- [ ] **Step 3: Implement `web/embedder.py`**

```python
"""DINOv2 appearance embeddings for robot re-identification.

FRC-agnostic: images in, L2-normalized vectors out. Swap the embedding backend
by changing DEFAULT_MODEL (e.g. to a SigLIP checkpoint) — no caller changes.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

DEFAULT_MODEL = "facebook/dinov2-small"  # 384-dim CLS embedding
MIN_CROP_PX = 10  # boxes smaller than this on either side yield no embedding
BOX_INSET = 0.08  # shrink box by this fraction per side to shed background


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 if either is None.

    Assumes inputs are L2-normalized (embeddings are), but normalizes defensively.
    """
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _inset_box(box, w: int, h: int):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_CROP_PX or bh < MIN_CROP_PX:
        return None
    dx, dy = int(bw * BOX_INSET), int(bh * BOX_INSET)
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


class RobotEmbedder:
    """Lazy-loaded DINOv2 embedder. The model loads on first embed() call."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_model(self) -> bool:
        """Load model/processor once. Returns False on failure (graceful degradation)."""
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.train(False)  # inference mode: disable dropout etc.
            return True
        except Exception as e:  # pragma: no cover - exercised only on env failure
            print(f"  RobotEmbedder: failed to load {self.model_name} ({e}); "
                  f"embeddings disabled, falling back to position-only re-ID.")
            self._model = None
            return False

    def embed(self, frame: np.ndarray, box) -> Optional[np.ndarray]:
        """Return an L2-normalized embedding for the crop, or None if invalid/unavailable."""
        results = self.embed_batch(frame, [box])
        return results[0] if results else None

    def embed_batch(self, frame: np.ndarray, boxes) -> list[Optional[np.ndarray]]:
        """Embed multiple crops from one frame in a single forward pass.

        Returns a list aligned with `boxes`; entries are None for invalid crops.
        """
        h, w = frame.shape[:2]
        insets = [_inset_box(b, w, h) for b in boxes]
        valid_idx = [i for i, b in enumerate(insets) if b is not None]
        out: list[Optional[np.ndarray]] = [None] * len(boxes)
        if not valid_idx:
            return out
        if not self._ensure_model():
            return out  # all None — caller falls back to position-only

        import torch
        crops = []
        for i in valid_idx:
            x1, y1, x2, y2 = insets[i]
            crop_bgr = frame[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        cls = outputs.last_hidden_state[:, 0]            # CLS token, [N, 384]
        cls = torch.nn.functional.normalize(cls, dim=1)  # L2-normalize
        vecs = cls.cpu().numpy().astype(np.float32)
        for slot, i in enumerate(valid_idx):
            out[i] = vecs[slot]
        return out
```

- [ ] **Step 4: Run fast tests, then model tests**

Run: `./.venv/bin/pytest tests/test_embedder.py -q -m "not model"`
Expected: PASS (cosine + tiny-box tests).

Run: `./.venv/bin/pytest tests/test_embedder.py -q -m model`
Expected: PASS (downloads DINOv2-small once; may take a minute on first run).

- [ ] **Step 5: Commit**

```bash
git add web/embedder.py tests/test_embedder.py
git commit -m "feat: add DINOv2 RobotEmbedder with cosine similarity"
```

---

### Task 2: Pure continuity decision (`choose_canonical_match`)

**Files:**
- Create: `web/reid.py`
- Test: `tests/test_reid_continuity.py`

The function decides what to do with a *new* ByteTrack detection. It is pure: no app state, no model. Inputs use a small typed structure so app.py can adapt its `RobotTrack` objects to it.

- [ ] **Step 1: Write the failing tests**

`tests/test_reid_continuity.py`:

```python
import numpy as np
import pytest

from web.reid import TrackView, choose_canonical_match

# Constants mirror app.py defaults; passed explicitly to keep the function pure.
APP_FLOOR = 0.55
POS_GATE = 200.0
W_POS, W_APP = 0.4, 0.6
CAP_PER_ALLIANCE = 3
CAP_TOTAL = 6


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _decide(det_pos, det_embed, det_alliance, tracks):
    return choose_canonical_match(
        det_pos=det_pos, det_embed=det_embed, det_alliance=det_alliance,
        tracks=tracks, app_floor=APP_FLOOR, pos_gate=POS_GATE,
        w_pos=W_POS, w_app=W_APP,
        cap_per_alliance=CAP_PER_ALLIANCE, cap_total=CAP_TOTAL,
    )


def test_ambiguous_position_picks_embedding_match():
    # Two blue tracks both ~near the detection; embedding matches track 2.
    a = _e(1, 0, 0)
    b = _e(0, 1, 0)
    tracks = [
        TrackView(robot_id=1, alliance="blue", position=(100, 100), embedding=a),
        TrackView(robot_id=2, alliance="blue", position=(110, 100), embedding=b),
    ]
    action, rid = _decide((105, 100), b, "blue", tracks)
    assert (action, rid) == ("merge", 2)


def test_below_floor_refuses_merge_and_creates():
    a = _e(1, 0, 0)
    far = _e(0, 0, 1)  # cosine ~0 with the only track, below floor
    tracks = [TrackView(robot_id=1, alliance="blue", position=(100, 100), embedding=a)]
    action, rid = _decide((101, 100), far, "blue", tracks)  # close in position
    assert action == "create"  # appearance veto beats proximity


def test_at_cap_assigns_best_embedding_not_nearest():
    a = _e(1, 0, 0)
    b = _e(0, 1, 0)
    c = _e(0, 0, 1)
    tracks = [
        TrackView(1, "blue", (100, 100), a),
        TrackView(2, "blue", (105, 100), b),
        TrackView(3, "blue", (110, 100), c),  # alliance full (3/3)
    ]
    # Detection is positionally nearest track 1 but visually identical to track 3.
    # The appearance veto rejects tracks 1 and 2, so it resolves to track 3 by
    # embedding rather than nearest position. Whether that is reported as "merge"
    # or "assign" is internal — the mapping to robot 3 is what matters.
    action, rid = _decide((101, 100), c, "blue", tracks)
    assert rid == 3
    assert action in ("merge", "assign")


def test_track_without_embedding_uses_position():
    tracks = [TrackView(1, "blue", (100, 100), None)]
    action, rid = _decide((105, 100), _e(1, 0, 0), "blue", tracks)
    assert (action, rid) == ("merge", 1)  # neutral app_sim, position close enough


def test_empty_tracks_creates():
    action, rid = _decide((100, 100), _e(1, 0, 0), "red", [])
    assert action == "create"
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/pytest tests/test_reid_continuity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.reid'`.

- [ ] **Step 3: Implement `choose_canonical_match` in `web/reid.py`**

```python
"""Pure re-identification decision logic (no app state, no model, no I/O).

Two functions:
- choose_canonical_match: continuity — which canonical track is a new detection?
- assign_identities: identity — one-to-one matching of tracks to references.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from web.embedder import cosine_similarity


@dataclass
class TrackView:
    """A read-only view of a canonical track for the continuity decision."""
    robot_id: int
    alliance: str
    position: tuple
    embedding: Optional[np.ndarray]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _dist(p, q) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def choose_canonical_match(
    det_pos, det_embed, det_alliance, tracks,
    app_floor: float, pos_gate: float, w_pos: float, w_app: float,
    cap_per_alliance: int, cap_total: int,
):
    """Decide what a new detection is. Returns one of:
        ("merge", robot_id)   merge into an existing track (within gate + floor)
        ("create", None)      genuinely new robot (caps permitting)
        ("assign", robot_id)  caps full — must be an existing robot; best embedding

    Same-alliance tracks are preferred for merge; the cap-full fallback considers
    all tracks (alliance/color can be misread).
    """
    same = [t for t in tracks if t.alliance == det_alliance]
    other = [t for t in tracks if t.alliance != det_alliance]

    def best_merge(candidates):
        best, best_score = None, -1.0
        for t in candidates:
            d = _dist(det_pos, t.position)
            if d > pos_gate:
                continue
            pos_score = max(0.0, 1.0 - d / pos_gate)
            # Neutral similarity (0.0) when a track has no embedding yet.
            app_sim = _clamp01(cosine_similarity(det_embed, t.embedding)) if t.embedding is not None else 0.0
            # Appearance veto: never absorb a visually different detection.
            if t.embedding is not None and det_embed is not None and app_sim < app_floor:
                continue
            score = w_pos * pos_score + w_app * app_sim
            if score > best_score:
                best, best_score = t, score
        return best

    merged = best_merge(same) or best_merge(other)
    if merged is not None:
        return ("merge", merged.robot_id)

    alliance_count = len(same)
    if alliance_count < cap_per_alliance and len(tracks) < cap_total:
        return ("create", None)

    if tracks:  # caps full — assign to the best EMBEDDING match, not nearest position
        if det_embed is not None and any(t.embedding is not None for t in tracks):
            best = max(tracks, key=lambda t: cosine_similarity(det_embed, t.embedding))
        else:
            best = min(tracks, key=lambda t: _dist(det_pos, t.position))
        return ("assign", best.robot_id)

    return ("create", None)  # absolute first-detection fallback
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/bin/pytest tests/test_reid_continuity.py -q`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add web/reid.py tests/test_reid_continuity.py
git commit -m "feat: add pure embedding-aware continuity decision"
```

---

### Task 3: Pure identity assignment (`assign_identities`)

**Files:**
- Modify: `web/reid.py`
- Test: `tests/test_reid_identity.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_reid_identity.py`:

```python
import numpy as np
import pytest

from web.reid import assign_identities

MAX_COST = 0.45  # accept a label only if (1 - cosine) <= 0.45


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_never_assigns_same_team_twice():
    # Two tracks both look most like team "A"; only one may get it.
    a = _e(1, 0, 0)
    tracks = {1: a, 2: _e(0.9, 0.1, 0)}
    refs = {"A": a, "B": _e(0, 1, 0)}
    alliance_track = {1: "blue", 2: "blue"}
    alliance_team = {"A": "blue", "B": "blue"}
    result = assign_identities(tracks, refs, alliance_track, alliance_team,
                               locked_tracks=set(), locked_teams=set(), max_cost=MAX_COST)
    assigned_teams = [t for t in result.values() if t is not None]
    assert len(assigned_teams) == len(set(assigned_teams))  # unique


def test_alliance_hard_constraint_forbids_cross():
    # Track 1 is red but visually identical to a blue reference; must NOT match it.
    v = _e(1, 0, 0)
    tracks = {1: v}
    refs = {"BLUE_TEAM": v}
    result = assign_identities(tracks, refs, {1: "red"}, {"BLUE_TEAM": "blue"},
                               set(), set(), MAX_COST)
    assert result.get(1) is None


def test_locked_tracks_and_teams_excluded():
    a, b = _e(1, 0, 0), _e(0, 1, 0)
    tracks = {1: a, 2: b}
    refs = {"A": a, "B": b}
    # Track 1 already locked to "A"; only track 2 / team "B" remain.
    result = assign_identities(tracks, refs, {1: "blue", 2: "blue"},
                               {"A": "blue", "B": "blue"},
                               locked_tracks={1}, locked_teams={"A"}, max_cost=MAX_COST)
    assert 1 not in result        # locked track not reassigned
    assert result.get(2) == "B"


def test_cost_above_threshold_unassigned():
    tracks = {1: _e(1, 0, 0)}
    refs = {"FAR": _e(0, 1, 0)}  # cosine ~0 -> cost ~1 > MAX_COST
    result = assign_identities(tracks, refs, {1: "blue"}, {"FAR": "blue"},
                               set(), set(), MAX_COST)
    assert result.get(1) is None


def test_rectangular_more_tracks_than_refs():
    a = _e(1, 0, 0)
    tracks = {1: a, 2: _e(0.8, 0.2, 0), 3: _e(0.6, 0.4, 0)}
    refs = {"A": a}
    result = assign_identities(tracks, refs,
                               {1: "blue", 2: "blue", 3: "blue"}, {"A": "blue"},
                               set(), set(), MAX_COST)
    assert sum(1 for v in result.values() if v == "A") == 1  # only one gets it
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/pytest tests/test_reid_identity.py -q`
Expected: FAIL with `ImportError: cannot import name 'assign_identities'`.

- [ ] **Step 3: Implement `assign_identities` in `web/reid.py`**

Append to `web/reid.py`:

```python
FORBIDDEN = 1e6  # effectively-infinite cost for alliance-forbidden pairs


def assign_identities(
    track_embeds: dict, ref_embeds: dict,
    alliance_of_track: dict, alliance_of_team: dict,
    locked_tracks: set, locked_teams: set, max_cost: float,
) -> dict:
    """One-to-one assignment of tracks to reference teams (Hungarian).

    Args:
        track_embeds: {robot_id: embedding}
        ref_embeds:   {team_number: embedding}
        alliance_of_track: {robot_id: "blue"|"red"|None}
        alliance_of_team:  {team_number: "blue"|"red"}
        locked_tracks: robot_ids already confirmed (excluded from the problem)
        locked_teams:  team_numbers already taken (excluded)
        max_cost: accept a match only if (1 - cosine) <= max_cost

    Returns {robot_id: team_number or None} for every UNLOCKED track with an
    embedding. Uniqueness is guaranteed by construction.
    """
    rows = [rid for rid, e in track_embeds.items()
            if rid not in locked_tracks and e is not None]
    cols = [team for team in ref_embeds
            if team not in locked_teams and ref_embeds[team] is not None]
    result = {rid: None for rid in rows}
    if not rows or not cols:
        return result

    cost = np.full((len(rows), len(cols)), FORBIDDEN, dtype=np.float64)
    for i, rid in enumerate(rows):
        ta = alliance_of_track.get(rid)
        for j, team in enumerate(cols):
            if ta is not None and ta != alliance_of_team.get(team):
                continue  # alliance hard constraint
            cost[i, j] = 1.0 - cosine_similarity(track_embeds[rid], ref_embeds[team])

    row_idx, col_idx = linear_sum_assignment(cost)
    for i, j in zip(row_idx, col_idx):
        if cost[i, j] <= max_cost:  # rejects FORBIDDEN and weak matches
            result[rows[i]] = cols[j]
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/bin/pytest tests/test_reid_identity.py -q`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add web/reid.py tests/test_reid_identity.py
git commit -m "feat: add Hungarian one-to-one identity assignment"
```

---

### Task 4: Add constants and wire embedder + reference embeddings into `InferenceJob`

**Files:**
- Modify: `web/app.py` (constants block ~`:55-100`; `RobotTrack` `:128-144`; `InferenceJob.__init__` `:185-233`; calibration reference seeding `:511-514`)

This task adds plumbing only; behavior changes land in Tasks 5–6. Verify with the existing suite plus a manual import smoke test (full pipeline tests need video).

- [ ] **Step 1: Add constants near the existing tracking constants**

In the constants block (alongside `ROBOT_MERGE_THRESHOLD`, `APPEARANCE_FRAME_INTERVAL`, etc.), add:

```python
# Embedding re-ID (DINOv2)
EMBED_MODEL_NAME = "facebook/dinov2-small"
APP_FLOOR = 0.55          # min cosine to merge a new detection into a track
POS_GATE_PX = 200.0       # max pixel distance to consider a merge
W_POS, W_APP = 0.4, 0.6   # continuity score weights
ASSIGN_MAX_COST = 0.45    # max Hungarian cost (≈ min cosine 0.55) to accept a label
IDENTITY_INTERVAL = 15    # run joint identity assignment every Nth processed frame
EMBED_EMA_ALPHA = 0.3     # weight of the new embedding in the running EMA
```

- [ ] **Step 2: Add imports**

Near the top imports of `web/app.py`:

```python
from web.embedder import RobotEmbedder, cosine_similarity
from web import reid
```

- [ ] **Step 3: Rename the appearance field on `RobotTrack`**

In `RobotTrack` (`:142-143`), replace:

```python
    # Appearance fingerprint for re-ID (bottom-weighted HSV histogram)
    appearance_fingerprint: Optional[np.ndarray] = None
```

with:

```python
    # Appearance embedding for re-ID (DINOv2, EMA-updated)
    appearance_embedding: Optional[np.ndarray] = None
```

- [ ] **Step 4: Instantiate the embedder and rename the reference store in `InferenceJob.__init__`**

Add after the model/tracker setup:

```python
        self.embedder = RobotEmbedder(EMBED_MODEL_NAME)
```

Replace (`:233`):

```python
        self._reference_fingerprints: Dict[str, np.ndarray] = {}  # team_number -> fingerprint
```

with:

```python
        self._reference_embeddings: Dict[str, np.ndarray] = {}  # team_number -> DINOv2 embedding
```

- [ ] **Step 5: Seed reference embeddings at calibration**

At the reference-seeding seam (`:511-514`), replace the HSV fingerprint computation/storage so it stores a DINOv2 embedding instead. This live seeding lives inside `vision_identify_robots` (whose `frame` param is the overhead crop). **Edit only this `:511-514` instance** — there is a near-identical HSV block at `:548-562` inside the dead `appearance_match_robots`, which disappears in Task 6; do not touch it here. The change: where it currently does `fp = _compute_appearance_fingerprint(...)` / `self._reference_fingerprints[team] = fp`, compute `emb = self.embedder.embed(frame, box)` and store `self._reference_embeddings[team] = emb` (guard `if emb is not None`).

- [ ] **Step 6: Smoke-test imports and construction**

Run:
```bash
./.venv/bin/python -c "import web.app; print('app imports OK')"
./.venv/bin/pytest -q -m "not model"
```
Expected: import prints OK; existing fast tests still PASS. (Do not run inference — no video.)

- [ ] **Step 7: Commit**

```bash
git add web/app.py
git commit -m "feat: wire DINOv2 embedder and reference embeddings into InferenceJob"
```

---

### Task 5: Make `get_canonical_robot_id` embedding-aware

**Files:**
- Modify: `web/app.py` — `get_canonical_robot_id` (`:1087-1164`) and its **two** call sites; EMA update where robots are seen each frame.

- [ ] **Step 1: Read both call sites**

`get_canonical_robot_id(` has **two** call sites (grep to confirm). Both must be updated to pass the new `det_embed` argument, each computing the embedding from the frame that site's boxes belong to:

| Line | Path | Frame variable | Notes |
|---|---|---|---|
| `:1351` | early-OCR setup | `early_overhead` | bytetrack id is `int(i + 9000)`, position `(cx, cy)` |
| `:1436` | main tracking loop | `overhead_frame` | position `(center_x, center_y)`, box is `box`; `processed_frame_num` available |

**Do not use `frame`** — YOLO runs on the `overhead_frame` / `early_overhead` crop, so boxes are in that crop's coordinates. Read both sites before editing.

- [ ] **Step 2: Change the signature to accept the detection embedding**

New signature:

```python
def get_canonical_robot_id(self, bytetrack_id: int, position: Tuple[float, float],
                           alliance: str, det_embed: Optional[np.ndarray]) -> int:
```

Keep the early return for an already-mapped `bytetrack_id`. Replace the position-only nearest-neighbor body with a call into the pure function:

```python
        tracks = [
            reid.TrackView(robot_id=r.robot_id, alliance=r.alliance,
                           position=r.last_position, embedding=r.appearance_embedding)
            for r in self.robot_tracks.values()
        ]
        action, rid = reid.choose_canonical_match(
            det_pos=position, det_embed=det_embed, det_alliance=alliance,
            tracks=tracks, app_floor=APP_FLOOR, pos_gate=POS_GATE_PX,
            w_pos=W_POS, w_app=W_APP,
            cap_per_alliance=MAX_ROBOTS_PER_ALLIANCE, cap_total=MAX_ROBOTS_TOTAL,
        )
        if action in ("merge", "assign"):
            self.bytetrack_to_canonical[bytetrack_id] = rid
            return rid
        # action == "create"
        canonical_id = self.next_canonical_id
        self.next_canonical_id += 1
        self.robot_tracks[canonical_id] = RobotTrack(
            robot_id=canonical_id, alliance=alliance, team_number=None,
            last_position=position, positions_history=deque(maxlen=30),
            appearance_embedding=det_embed,
        )
        self.bytetrack_to_canonical[bytetrack_id] = canonical_id
        return canonical_id
```

- [ ] **Step 3: Compute the detection embedding at BOTH call sites and update the EMA**

At each call site, before calling, compute the embedding from that site's frame variable:
- main loop (`:1436`): `det_embed = self.embedder.embed(overhead_frame, box)`
- early-OCR (`:1351`): `det_embed = self.embedder.embed(early_overhead, <box at that site>)`

Pass `det_embed` into `get_canonical_robot_id(...)`. In the **main loop only** (where the robot is updated each frame), after obtaining the canonical id and updating `last_position`, update the running embedding (the block already does `robot = self.robot_tracks[canonical_id]` at `:1439`):

```python
        if det_embed is not None:
            if robot.appearance_embedding is None:
                robot.appearance_embedding = det_embed
            else:
                blended = (1 - EMBED_EMA_ALPHA) * robot.appearance_embedding + EMBED_EMA_ALPHA * det_embed
                n = np.linalg.norm(blended)
                robot.appearance_embedding = blended / n if n > 0 else blended
```

- [ ] **Step 4: Smoke-test**

Run:
```bash
./.venv/bin/python -c "import web.app; print('OK')"
./.venv/bin/pytest -q -m "not model"
```
Expected: OK; fast tests PASS. The continuity logic itself is covered by `tests/test_reid_continuity.py`.

- [ ] **Step 5: Commit**

```bash
git add web/app.py
git commit -m "feat: embedding-aware canonical track continuity"
```

---

### Task 6: Add joint identity assignment, remove dead `appearance_match_robots`

**Files:**
- Modify: `web/app.py` — delete `appearance_match_robots` (`:519-614`), add `assign_identities_now`, and **add a new call site** in the main frame loop.

> **Important:** `appearance_match_robots` is **dead code** — it is defined but never called anywhere (verified by grep). The HSV appearance re-ID never actually ran. So this task does **not** replace an existing call; it adds a brand-new periodic call into the loop and deletes the dead method.

- [ ] **Step 1: Read the OCR confirm path and locate the new call site**

Confirm how `team_number_confirmed` is set (in `ocr_robot_bumpers` `:374` and `vision_identify_robots` `:503`, both of which call `_clear_team_from_other_robots` — that helper stays, it is still live). The new call site goes in the main loop right after `tracked_robot_boxes` is built and near the existing `self.vision_identify_robots(overhead_frame, tracked_robot_boxes)` call (`:1491`); `processed_frame_num` is in scope there.

- [ ] **Step 2: Rewrite the method to call `reid.assign_identities`**

```python
    def assign_identities_now(self) -> None:
        """Periodic one-to-one assignment of unlocked tracks to reference teams.

        OCR/Qwen-confirmed tracks (and their teams) are locked out. The Hungarian
        match fills the rest with a uniqueness guarantee; results accumulate as
        votes and confirm once stable.
        """
        if not self._reference_embeddings:
            return
        track_embeds = {r.robot_id: r.appearance_embedding for r in self.robot_tracks.values()}
        alliance_of_track = {r.robot_id: r.alliance for r in self.robot_tracks.values()}
        alliance_of_team = {t: "blue" for t in self.match_blue_teams}
        alliance_of_team.update({t: "red" for t in self.match_red_teams})
        locked_tracks = {r.robot_id for r in self.robot_tracks.values() if r.team_number_confirmed}
        locked_teams = {r.team_number for r in self.robot_tracks.values()
                        if r.team_number_confirmed and r.team_number}

        assignment = reid.assign_identities(
            track_embeds, self._reference_embeddings,
            alliance_of_track, alliance_of_team,
            locked_tracks, locked_teams, ASSIGN_MAX_COST,
        )

        for robot_id, team in assignment.items():
            if team is None:
                continue
            robot = self.robot_tracks.get(robot_id)
            if robot is None or robot.team_number_confirmed:
                continue
            if robot.appearance_votes is None:
                robot.appearance_votes = {}
            robot.appearance_votes[team] = robot.appearance_votes.get(team, 0) + 1
            top = max(robot.appearance_votes, key=robot.appearance_votes.get)
            if robot.appearance_votes[top] >= APPEARANCE_CONFIRM_VOTES:
                robot.team_number = top
                robot.team_number_confirmed = True
                if top in self.match_blue_teams:
                    robot.alliance = "blue"
                elif top in self.match_red_teams:
                    robot.alliance = "red"
                for event in self.scoring_events:
                    if event.robot_id == robot_id:
                        event.team_number = top
```

Then **add** the new call in the main loop (near `:1491`, after `tracked_robot_boxes` is built):

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_identities_now()
```

Delete the dead `appearance_match_robots` method entirely. Keep `_clear_team_from_other_robots` — it is still called by the OCR/Qwen confirm paths (`:374`, `:503`).

- [ ] **Step 3: Smoke-test AND verify the call is wired**

Run:
```bash
./.venv/bin/python -c "import web.app; print('OK')"
grep -n "self.assign_identities_now()" web/app.py
./.venv/bin/pytest -q -m "not model"
```
Expected: OK; the grep prints the new loop call site (import alone cannot detect an unwired periodic call); fast tests PASS. Uniqueness is covered by `tests/test_reid_identity.py`.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat: joint Hungarian identity assignment replaces greedy HSV voting"
```

---

### Task 7: Retire HSV functions and clean the calibration path

**Files:**
- Modify: `web/app.py` — remove `_compute_appearance_fingerprint` (`:1870`), `_compare_fingerprints` (`:1935`), the `:2405` call + dead `"fingerprint"` field (`:2417`), and the now-dead constants `APPEARANCE_FRAME_INTERVAL` (`:93`) and `APPEARANCE_MIN_SIMILARITY` (`:95`).

> **Keep `_clear_team_from_other_robots`** — after Task 6 it is still called by the OCR/Qwen confirm paths (`:374`, `:503`). Do not delete it.

- [ ] **Step 1: Remove the calibration-path HSV call and its dead field**

In `_detect_robots_in_frame` (`:2404-2417`), delete:
```python
        # Compute appearance fingerprint for re-ID during inference
        fp = _compute_appearance_fingerprint(frame, box)
        fp_list = fp.tolist() if fp is not None else None
```
and remove the `"fingerprint": fp_list,` entry from the appended dict (the frontend never reads it).

- [ ] **Step 2: Delete the two HSV functions and the dead constants**

Remove `_compute_appearance_fingerprint` and `_compare_fingerprints` entirely. Also remove the `APPEARANCE_FRAME_INTERVAL` and `APPEARANCE_MIN_SIMILARITY` constants (only the dead `appearance_match_robots` used them).

- [ ] **Step 3: Verify nothing still references the retired names**

Run:
```bash
grep -n "_compute_appearance_fingerprint\|_compare_fingerprints\|appearance_fingerprint\|_reference_fingerprints\|appearance_match_robots\|APPEARANCE_FRAME_INTERVAL\|APPEARANCE_MIN_SIMILARITY\|\"fingerprint\"" web/app.py web/static/*.js
```
Expected: **no output**. (`_clear_team_from_other_robots` is intentionally retained and is not in this list.)

- [ ] **Step 4: Full import + fast suite + a quick byte-compile**

Run:
```bash
./.venv/bin/python -c "import web.app; print('OK')"
./.venv/bin/python -m py_compile web/app.py web/embedder.py web/reid.py && echo "compile OK"
./.venv/bin/pytest -q -m "not model"
```
Expected: OK; compile OK; fast tests PASS.

- [ ] **Step 5: Commit**

```bash
git add web/app.py
git commit -m "refactor: retire HSV fingerprint functions and dead calibration field"
```

---

### Task 8: Acceptance harness (manual)

**Files:**
- Create: `scripts/reid_acceptance.py`
- Create: `docs/acceptance/reid.md` (how to run + record results)

No match video is present in `data/videos/` right now, so this is a **documented manual acceptance step**, not an automated test. The user runs it against a clip when available.

- [ ] **Step 1: Write `scripts/reid_acceptance.py`**

A CLI that runs the pipeline on a clip and reports the three spec metrics:
- canonical-ID switches per physical robot (lower is better),
- count of frames exhibiting duplicate team numbers among live tracks (**must be 0** by construction),
- team-number accuracy at frames listed in an optional ground-truth JSON (`{frame_idx: {robot_box: team}}`).

Reuse `InferenceJob` to process the video; after each processed frame, snapshot `{robot_id: (team_number, alliance, position)}` and accumulate the metrics. Print a summary table.

- [ ] **Step 2: Write `docs/acceptance/reid.md`**

Document: where to place a clip, how to build the small ground-truth JSON (label team numbers at ~10 sampled frames), the exact command, and a results table to fill in (HSV baseline vs DINOv2). Record `IDENTITY_INTERVAL` / threshold values used.

- [ ] **Step 3: Byte-compile the script**

Run: `./.venv/bin/python -m py_compile scripts/reid_acceptance.py && echo "OK"`
Expected: OK. (A full run is deferred until a clip is available.)

- [ ] **Step 4: Commit**

```bash
git add scripts/reid_acceptance.py docs/acceptance/reid.md
git commit -m "feat: re-ID acceptance harness and procedure"
```

---

## Tuning note

After a real clip is available, tune `APP_FLOOR`, `ASSIGN_MAX_COST`, `POS_GATE_PX`, and `IDENTITY_INTERVAL` using `scripts/reid_acceptance.py`. If DINOv2 under-discriminates same-alliance robots, switch `EMBED_MODEL_NAME` to a SigLIP checkpoint — `web/embedder.py` is the only file that changes.

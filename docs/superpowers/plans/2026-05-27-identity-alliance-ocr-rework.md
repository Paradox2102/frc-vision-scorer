# Identity, Alliance & OCR Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make alliance roster-authoritative (kills the live alliance flip-flop), constrain VLM-vote team assignment by alliance with process-of-elimination for the last unread robot, and let the VLM spend its calls on legible bumpers while allowing wrong identities to self-correct.

**Architecture:** Walk back the Algorithm 2 wiring (Component D) to stop the active alliance bug first, then rebuild the identity layer as pure functions in `web/reid.py` (alliance-masked Hungarian + elimination + audit), a small new `web/legibility.py` for crop scoring, and a reworked `assign_teams_from_votes` / `vision_identify_robots` in `web/app.py` that uses them. Embeddings keep ownership of continuity only.

**Tech Stack:** Python, NumPy, SciPy (`linear_sum_assignment`), OpenCV (HSV masks + Laplacian for legibility), pytest. Existing dependencies — no new ones.

**Spec:** `docs/superpowers/specs/2026-05-27-identity-alliance-ocr-rework-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `web/reid.py` | Pure identity/continuity decisions | Add 3 pure functions: `assign_teams_alliance_masked`, `eliminate_last_team`, `audit_overturn` |
| `web/legibility.py` | Pure crop scoring (no app state) | **New file** — `legibility_score(crop_bgr) -> float` and a few helpers |
| `web/app.py` | Live inference orchestration | Remove Algorithm 2 calls from loop; rework `assign_teams_from_votes` to use the pure functions; rework `vision_identify_robots` for legibility-scored top-K + reduced-cadence re-reads of confirmed robots; add `audit_overturn_confirmations` method and call it in the identity interval; add new constants |
| `tests/test_alliance_consistency.py` | Regression guard | **New** — restore the test dropped as "advisory" |
| `tests/test_reid_assignment.py` | Unit tests for the three new pure functions | **New** |
| `tests/test_legibility.py` | Unit tests on synthetic crops | **New** |

**Test command:** `.venv/bin/python -m pytest tests/ -m "not model" -v` (see the `test-env-uses-venv` memory — `pytest` is in `.venv`, not in the base conda python).

---

## Task 1: Walk back Algorithm 2 wiring + alliance consistency guard

This is the smallest change that stops the live alliance flip-flop. Do it first so subsequent runs are usable while the rest of the rework lands.

**Files:**
- Modify: `web/app.py:1668-1670` (identity-interval block)
- Create: `tests/test_alliance_consistency.py`

- [ ] **Step 1: Write the failing regression test**

Create `tests/test_alliance_consistency.py`:

```python
"""Regression guard for the alliance flip-flop bug.

A robot with a tentative team number must report the roster alliance for that
team. No code path may override that with embedding clustering or anything else.
"""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from web.app import InferenceJob, RobotTrack


def _make_job():
    """A lightweight stand-in exposing only the attributes
    `assign_teams_from_votes` actually reads. Lets us call the method as an
    unbound function (`InferenceJob.assign_teams_from_votes(job)`) without
    paying the cost of `InferenceJob.__init__` (which loads YOLO weights)."""
    return SimpleNamespace(
        match_blue_teams=["2102", "9452", "2485"],
        match_red_teams=["1538", "294", "4322"],
        match_teams=["2102", "9452", "2485", "1538", "294", "4322"],
        robot_tracks={},
        scoring_events=[],
    )


def test_tentative_team_alliance_matches_roster():
    """A robot with strong votes for a red team must end up red, not blue."""
    job = _make_job()
    job.robot_tracks = {
        0: RobotTrack(robot_id=0, alliance="blue", team_number=None,
                      last_position=(0, 0), positions_history=deque(maxlen=30)),
    }
    # Strong VLM votes for a RED team
    job.robot_tracks[0].team_ocr_votes = {"4322": 5.0}

    InferenceJob.assign_teams_from_votes(job)

    r = job.robot_tracks[0]
    assert r.team_number == "4322"
    assert r.alliance == "red", (
        f"Robot has team 4322 (red) but alliance={r.alliance!r}; "
        "regression: alliance must follow the roster when a team is assigned."
    )


def test_no_embedding_alliance_calls_in_loop():
    """Structural guard: the loop must not call _assign_alliances_by_embedding
    or assign_identities_now (Component D removal). They caused the flip-flop."""
    src = Path("web/app.py").read_text()
    # Function bodies may exist, but no `self.<func>()` calls should remain in the loop.
    assert "self._assign_alliances_by_embedding()" not in src, (
        "_assign_alliances_by_embedding() must not be called from the loop."
    )
    assert "self.assign_identities_now()" not in src, (
        "assign_identities_now() must not be called from the loop."
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_alliance_consistency.py -v`

Expected: `test_no_embedding_alliance_calls_in_loop` FAILS (calls still present at `web/app.py:1669-1670`). The first test may pass or fail depending on whether `_assign_alliances_by_embedding` clobbers under this minimal fixture — running it confirms the regression direction.

- [ ] **Step 3: Remove the embedding alliance/identity calls from the loop**

In `web/app.py`, replace lines 1668-1670:

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()       # VLM locks (primary)
                    self._assign_alliances_by_embedding()  # robust alliance via embedding clustering
                    self.assign_identities_now()         # embeddings fill unlocked tracks (uniqueness)
```

with:

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()       # VLM-vote Hungarian (rework in Task 4)
                    self.audit_overturn_confirmations()  # let wrong locks self-correct (added in Task 6)
```

If `audit_overturn_confirmations` does not exist yet (it is added in Task 6), make this single line for now and add the second call in Task 6 — comment it out:

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()       # VLM-vote Hungarian (rework in Task 4)
                    # self.audit_overturn_confirmations()  # added in Task 6
```

- [ ] **Step 4: Run the tests + full suite to verify**

Run:
```
.venv/bin/python -m pytest tests/test_alliance_consistency.py -v
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: both consistency tests PASS; all 21 pre-existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_alliance_consistency.py
git commit -m "fix(app): remove embedding alliance/identity from loop + consistency guard

Walks back the Algorithm 2 wiring that overrode roster-derived alliance for
unconfirmed robots and caused the B0/R0 alliance flip-flop. Adds regression
test that was wrongly skipped as 'advisory' in the prior plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `reid.py` — alliance-masked Hungarian + elimination + audit

Pure functions, TDD. No app state, no I/O.

**Files:**
- Modify: `web/reid.py`
- Create: `tests/test_reid_assignment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reid_assignment.py`:

```python
import numpy as np
import pytest

from web.reid import (
    assign_teams_alliance_masked,
    eliminate_last_team,
    audit_overturn,
)

BLUE = ["2102", "9452", "2485"]
RED  = ["1538", "294", "4322"]


def _alliance_of(spec: dict[int, str]) -> dict[int, str]:
    return dict(spec)


# --- assign_teams_alliance_masked --------------------------------------------

def test_assignment_never_crosses_alliance():
    """A red robot cannot be assigned a blue team, even if its votes prefer one."""
    votes = {0: {"2102": 10.0}}              # robot 0 has only blue-team votes
    alliance = _alliance_of({0: "red"})       # but bumper says red
    out = assign_teams_alliance_masked(
        votes, alliance, BLUE, RED, locked_teams=set(), locked_robots=set(),
    )
    assert out[0] is None, "cross-alliance assignment must be forbidden"


def test_unknown_alliance_is_unconstrained():
    votes = {0: {"2102": 5.0}}
    alliance = _alliance_of({0: "unknown"})
    out = assign_teams_alliance_masked(votes, alliance, BLUE, RED, set(), set())
    assert out[0] == "2102"


def test_no_team_assigned_twice():
    """Two robots both vote highest for the same team; only one gets it."""
    votes = {0: {"9452": 5.0}, 1: {"9452": 3.0, "2485": 1.0}}
    alliance = _alliance_of({0: "blue", 1: "blue"})
    out = assign_teams_alliance_masked(votes, alliance, BLUE, RED, set(), set())
    assigned = [t for t in out.values() if t is not None]
    assert len(assigned) == len(set(assigned)), f"duplicate team in {out}"
    assert out[0] == "9452"  # strongest vote keeps the team


def test_no_votes_means_no_assignment():
    votes = {0: {}}
    alliance = _alliance_of({0: "blue"})
    out = assign_teams_alliance_masked(votes, alliance, BLUE, RED, set(), set())
    assert out[0] is None


def test_locked_robots_and_teams_excluded():
    """Confirmed robots and their teams must not appear in the new assignment."""
    votes = {0: {"9452": 5.0}, 1: {"9452": 4.0, "2485": 1.0}}
    alliance = _alliance_of({0: "blue", 1: "blue"})
    out = assign_teams_alliance_masked(
        votes, alliance, BLUE, RED,
        locked_teams={"9452"}, locked_robots={0},
    )
    assert 0 not in out                        # locked robot excluded
    assert out.get(1) == "2485"                # 9452 locked → robot 1 takes 2485


# --- eliminate_last_team -----------------------------------------------------

def test_elimination_fires_when_one_team_and_one_robot_remain():
    """Two of three red teams already assigned; one red-bumper robot still unlabeled."""
    alliance = _alliance_of({0: "red", 1: "red", 2: "red"})
    team_of = {0: "1538", 1: "294", 2: None}
    out = eliminate_last_team(alliance, team_of, BLUE, RED)
    assert out == {2: "4322"}


def test_elimination_does_not_fire_with_two_unread_robots():
    alliance = _alliance_of({0: "red", 1: "red", 2: "red"})
    team_of = {0: "1538", 1: None, 2: None}  # two unread on red
    out = eliminate_last_team(alliance, team_of, BLUE, RED)
    assert out == {}


def test_elimination_does_not_cross_alliance():
    """A blue-bumper robot must never be eliminated into a red team."""
    alliance = _alliance_of({0: "red", 1: "red", 2: "blue"})
    team_of = {0: "1538", 1: "294", 2: None}  # robot 2 is blue, last red team open
    out = eliminate_last_team(alliance, team_of, BLUE, RED)
    assert out == {}


# --- audit_overturn ----------------------------------------------------------

def test_audit_overturns_when_other_team_dominates():
    out = audit_overturn(current_team="9452",
                         votes={"9452": 3.0, "2485": 7.0},
                         margin=3.0)
    assert out == "2485"


def test_audit_sticky_below_margin():
    out = audit_overturn(current_team="9452",
                         votes={"9452": 3.0, "2485": 5.0},  # only +2 over current
                         margin=3.0)
    assert out is None


def test_audit_no_other_candidates():
    out = audit_overturn("9452", {"9452": 5.0}, margin=3.0)
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reid_assignment.py -v`

Expected: ImportError (`cannot import name 'assign_teams_alliance_masked'`).

- [ ] **Step 3: Implement the three pure functions in `web/reid.py`**

Add at the end of `web/reid.py` (after `assign_identities`):

```python
def assign_teams_alliance_masked(
    votes: dict, alliance_of_robot: dict,
    blue_teams: list, red_teams: list,
    locked_teams: set, locked_robots: set,
) -> dict:
    """Alliance-masked one-to-one assignment of robots to teams by accumulated
    VLM read votes. A single Hungarian solve over the whole field with
    cross-alliance (robot, team) cells forbidden — unknown-alliance robots are
    open to all teams. Returns {robot_id: team_or_None} for every unlocked robot.
    Robots with no positive votes for any allowed team get None.
    """
    rows = [rid for rid in votes if rid not in locked_robots]
    cols = [t for t in (blue_teams + red_teams) if t not in locked_teams]
    result = {rid: None for rid in rows}
    if not rows or not cols:
        return result

    cost = np.full((len(rows), len(cols)), FORBIDDEN, dtype=np.float64)
    for i, rid in enumerate(rows):
        ra = alliance_of_robot.get(rid, "unknown")
        rvotes = votes.get(rid) or {}
        for j, team in enumerate(cols):
            ta = "red" if team in red_teams else "blue"
            if ra != "unknown" and ra != ta:
                continue  # cross-alliance forbidden
            cost[i, j] = -float(rvotes.get(team, 0.0))

    row_idx, col_idx = linear_sum_assignment(cost)
    for i, j in zip(row_idx, col_idx):
        # Skip cells with no positive evidence (cost == 0) and forbidden cells.
        if cost[i, j] >= 0:
            continue
        result[rows[i]] = cols[j]
    return result


def eliminate_last_team(
    alliance_of_robot: dict, team_of_robot: dict,
    blue_teams: list, red_teams: list,
) -> dict:
    """Process-of-elimination: for each alliance, if exactly one roster team is
    unassigned AND exactly one robot has that alliance's color with no team,
    assign that robot that team. Returns {robot_id: team} for newly-eliminated
    pairs (empty dict if no alliance qualifies).
    """
    result: dict = {}
    for alliance, roster in (("red", red_teams), ("blue", blue_teams)):
        assigned = {team_of_robot.get(rid) for rid in team_of_robot}
        unassigned = [t for t in roster if t not in assigned]
        unidentified = [
            rid for rid, a in alliance_of_robot.items()
            if a == alliance and team_of_robot.get(rid) is None
        ]
        if len(unassigned) == 1 and len(unidentified) == 1:
            result[unidentified[0]] = unassigned[0]
    return result


def audit_overturn(current_team: str, votes: dict, margin: float):
    """Return a different team if its accumulated votes exceed `current_team`'s
    by `margin`; else None. Sticky-by-default so a confirmed label flips only
    when a competitor genuinely dominates.
    """
    current = float((votes or {}).get(current_team, 0.0))
    best_t, best_v = None, -1.0
    for t, v in (votes or {}).items():
        if t == current_team:
            continue
        if v > best_v:
            best_t, best_v = t, float(v)
    if best_t is not None and best_v > current + margin:
        return best_t
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reid_assignment.py -v`

Expected: all 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add web/reid.py tests/test_reid_assignment.py
git commit -m "feat(reid): alliance-masked Hungarian + elimination + audit helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `web/legibility.py` — pure crop scoring

**Files:**
- Create: `web/legibility.py`
- Create: `tests/test_legibility.py`

- [ ] **Step 1: Write the failing tests with synthetic crops**

Create `tests/test_legibility.py`:

```python
import cv2
import numpy as np

from web.legibility import legibility_score


def _good_red_bumper_crop():
    """Sharp red bumper with white digit-like blobs — high legibility."""
    img = np.full((80, 200, 3), 30, np.uint8)             # dark background
    img[50:75, 20:180] = (40, 40, 220)                    # red bumper band (BGR)
    # White "digits"
    cv2.rectangle(img, (40, 55), (60, 70), (255, 255, 255), -1)
    cv2.rectangle(img, (90, 55), (110, 70), (255, 255, 255), -1)
    cv2.rectangle(img, (140, 55), (160, 70), (255, 255, 255), -1)
    return img


def _blurry_red_bumper_crop():
    sharp = _good_red_bumper_crop()
    return cv2.GaussianBlur(sharp, (15, 15), 0)


def _back_facing_crop():
    """Uniform mid-gray — no bumper color, no text."""
    return np.full((80, 200, 3), 128, np.uint8)


def test_invalid_crop_returns_zero():
    assert legibility_score(None) == 0.0
    assert legibility_score(np.zeros((0, 0, 3), np.uint8)) == 0.0
    assert legibility_score(np.zeros((4, 4, 3), np.uint8)) == 0.0   # too small


def test_no_bumper_color_scores_zero():
    assert legibility_score(_back_facing_crop()) == 0.0


def test_good_scores_above_blurry_scores_above_bad():
    good = legibility_score(_good_red_bumper_crop())
    blurry = legibility_score(_blurry_red_bumper_crop())
    bad = legibility_score(_back_facing_crop())
    assert good > blurry > bad >= 0.0
    assert good > 0.0
```

- [ ] **Step 2: Run tests to verify they fail (ImportError)**

Run: `.venv/bin/python -m pytest tests/test_legibility.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `web/legibility.py`**

```python
"""Pure crop scoring for OCR frame selection.

`legibility_score(crop_bgr)` returns a non-negative float estimating how
likely a Qwen-VL pass on this bumper crop will yield a valid roster read.
Higher is better. Combines three signals:

  1. **color strength** — fraction of pixels matching red or blue bumper HSV
     ranges (reuses the band tuned for `detect_bumper_color`).
  2. **text contrast** — fraction of near-white pixels inside / next to the
     colored band (the digits sit on the bumper).
  3. **sharpness** — variance of the Laplacian; rejects motion blur.

No app state, no I/O — easy to unit-test with synthetic crops.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_CROP_PX = 8
WHITE_THRESHOLD = 200          # gray-value threshold for "near white" pixels
SHARPNESS_NORMALIZER = 200.0   # ~typical Laplacian variance on a sharp bumper
DILATE_PX = 5                  # how far to dilate the color mask to capture nearby text


def _color_masks(hsv: np.ndarray):
    red1 = cv2.inRange(hsv, np.array([0, 45, 40]), np.array([16, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 45, 40]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([88, 45, 40]), np.array([140, 255, 255]))
    return red1 | red2, blue


def legibility_score(crop_bgr) -> float:
    if crop_bgr is None:
        return 0.0
    if not isinstance(crop_bgr, np.ndarray) or crop_bgr.ndim != 3 or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX:
        return 0.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    red_mask, blue_mask = _color_masks(hsv)
    total = float(h * w)
    color_strength = max(cv2.countNonZero(red_mask), cv2.countNonZero(blue_mask)) / total
    if color_strength == 0.0:
        return 0.0

    # Dilate the dominant-color mask to include adjacent white text pixels.
    color_mask = red_mask if cv2.countNonZero(red_mask) >= cv2.countNonZero(blue_mask) else blue_mask
    band = cv2.dilate(color_mask, np.ones((DILATE_PX, DILATE_PX), np.uint8), iterations=1)
    band_pixels = max(1, cv2.countNonZero(band))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    white_in_band = cv2.countNonZero(((gray > WHITE_THRESHOLD).astype(np.uint8)) & (band > 0).astype(np.uint8))
    text_contrast = white_in_band / band_pixels

    sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / SHARPNESS_NORMALIZER, 1.0)

    return float(color_strength * text_contrast * sharpness)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_legibility.py -v`
Expected: all 3 PASS, with `good > blurry > bad`.

- [ ] **Step 5: Commit**

```bash
git add web/legibility.py tests/test_legibility.py
git commit -m "feat(legibility): pure crop scoring for OCR frame selection

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Rework `assign_teams_from_votes` to use the alliance-masked Hungarian + elimination

**Files:**
- Modify: `web/app.py:644-680` (the function body)

The regression test in Task 1 already checks alliance consistency. Add no new tests here; existing tests must still pass and the consistency test must keep passing.

- [ ] **Step 1: Replace the function body**

In `web/app.py`, replace the entire body of `assign_teams_from_votes` (lines ~644-680) with:

```python
    def assign_teams_from_votes(self) -> None:
        """Assign teams from accumulated VLM read votes using an alliance-masked
        Hungarian, plus process-of-elimination for the last unread robot on an
        alliance. Alliance is roster-authoritative when a team is set and
        bumper-color-derived otherwise. Re-runs each interval; early misreads get
        outvoted over time.
        """
        if not self.match_teams:
            return

        locked_teams = {r.team_number for r in self.robot_tracks.values()
                        if r.team_number_confirmed and r.team_number}
        locked_robots = {r.robot_id for r in self.robot_tracks.values()
                         if r.team_number_confirmed}

        # Per-robot alliance for the constraint:
        #  - if it has a team (tentative or confirmed) → roster (authoritative)
        #  - else → red/blue color-vote majority, or "unknown" if equal/zero
        alliance_of_robot = {}
        for r in self.robot_tracks.values():
            if r.team_number and r.team_number in self.match_red_teams:
                alliance_of_robot[r.robot_id] = "red"
            elif r.team_number and r.team_number in self.match_blue_teams:
                alliance_of_robot[r.robot_id] = "blue"
            elif r.red_votes > r.blue_votes:
                alliance_of_robot[r.robot_id] = "red"
            elif r.blue_votes > r.red_votes:
                alliance_of_robot[r.robot_id] = "blue"
            else:
                alliance_of_robot[r.robot_id] = "unknown"

        votes = {r.robot_id: dict(r.team_ocr_votes or {})
                 for r in self.robot_tracks.values()}

        assignment = reid.assign_teams_alliance_masked(
            votes, alliance_of_robot,
            self.match_blue_teams, self.match_red_teams,
            locked_teams, locked_robots,
        )

        # Apply Hungarian assignments.
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
            for event in self.scoring_events:
                if event.robot_id == rid:
                    event.team_number = team

        # Process-of-elimination: last roster team + last unidentified bumper-colored
        # robot → assign and auto-confirm. Subject to overturn by the audit (Task 6).
        team_of_robot = {rid: self.robot_tracks[rid].team_number
                         for rid in self.robot_tracks}
        eliminated = reid.eliminate_last_team(
            alliance_of_robot, team_of_robot,
            self.match_blue_teams, self.match_red_teams,
        )
        for rid, team in eliminated.items():
            r = self.robot_tracks[rid]
            r.team_number = team
            r.alliance = "red" if team in self.match_red_teams else "blue"
            r.team_number_confirmed = True
            print(f"  Elimination: Robot {rid} = Team {team} (only candidate left on {r.alliance})")
            for event in self.scoring_events:
                if event.robot_id == rid:
                    event.team_number = team
```

- [ ] **Step 2: Verify import + full suite**

Run:
```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; full suite green (the alliance consistency test now exercises the rewritten function).

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(app): alliance-masked Hungarian + elimination in assign_teams_from_votes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Rework `vision_identify_robots` — legibility-scored top-K + re-read confirmed

**Files:**
- Modify: `web/app.py:456-556` (the `vision_identify_robots` method)
- Modify: `web/app.py` constants block (~108-155) — add legibility/re-read constants

- [ ] **Step 1: Add new constants near the existing block**

In `web/app.py`, near the other tunables (around line 150-155, after `EMBED_EMA_ALPHA`):

```python
LEGIBILITY_FLOOR = 0.005       # crops below this are not worth a Qwen call
VLM_TOP_K = 6                  # max crops to send to Qwen per pass (whole field)
REREAD_EVERY_N_PASSES = 4      # include confirmed robots once every Nth pass (self-correction)
OVERTURN_VOTE_MARGIN = 3.0     # competitor team's votes must exceed current by this to overturn
```

Also add to the same area: import `from web.legibility import legibility_score` — placed alongside the existing `from web.embedder import RobotEmbedder, cosine_similarity` at line 36.

- [ ] **Step 2: Open the upstream VLM-call gate for re-read passes**

The existing gate at `web/app.py:1659` short-circuits the call when `unconfirmed_count == 0`, so once every robot is confirmed the VLM never runs again — that would make the re-read pass below (and Task 6's audit) dead code in exactly the scenario they exist to fix.

Replace:

```python
                if (unconfirmed_count > 0 and
                        tracked_robot_boxes and
                        processed_frame_num >= 10 and
                        current_time - self._claude_vision_last_time >= vision_interval):
                    self._claude_vision_calls += 1
                    self._claude_vision_last_time = current_time
                    self.vision_identify_robots(overhead_frame, tracked_robot_boxes)
```

with:

```python
                # Let the call through even when everything is confirmed if the
                # next pass is a re-read pass — that's how a wrong lock self-corrects.
                is_reread_due = ((self._claude_vision_calls + 1) % REREAD_EVERY_N_PASSES == 0)
                if ((unconfirmed_count > 0 or is_reread_due) and
                        tracked_robot_boxes and
                        processed_frame_num >= 10 and
                        current_time - self._claude_vision_last_time >= vision_interval):
                    self._claude_vision_calls += 1
                    self._claude_vision_last_time = current_time
                    self.vision_identify_robots(overhead_frame, tracked_robot_boxes)
```

- [ ] **Step 3: Replace the candidate-selection portion of `vision_identify_robots`**

Replace the body of `vision_identify_robots` (lines ~456-556) with this. The two changes vs. today: (1) confirmed robots are included every `REREAD_EVERY_N_PASSES`th call, and (2) candidates are ranked by `legibility_score` and only the top `VLM_TOP_K` (above floor) are sent to Qwen.

```python
    def vision_identify_robots(self, frame: np.ndarray, tracked_robots) -> None:
        """Use Qwen-VL to read bumper numbers on a legibility-ranked subset of
        the field. Confirmed robots are included at a reduced cadence so a wrong
        identity can be overturned by the audit (assign_teams_from_votes +
        audit_overturn_confirmations). tracked_robots: list of (canonical_id, box)."""
        if not self.match_teams:
            return

        # Roster minus already-confirmed teams (the OCR matcher still sees the
        # full per-alliance lists; confirmed teams are pre-removed to bias reads).
        confirmed_teams = {r.team_number for r in self.robot_tracks.values()
                           if r.team_number_confirmed and r.team_number}
        remaining_teams = [t for t in self.match_teams if t not in confirmed_teams]
        if not remaining_teams and self._claude_vision_calls % REREAD_EVERY_N_PASSES != 0:
            return  # everything confirmed and not a re-read pass
        remaining_blue = [t for t in self.match_blue_teams if t not in confirmed_teams]
        remaining_red = [t for t in self.match_red_teams if t not in confirmed_teams]

        # Build candidate crops. Confirmed robots only on re-read passes.
        is_reread_pass = (self._claude_vision_calls % REREAD_EVERY_N_PASSES == 0)
        candidates = []  # list of (score, canonical_id, box, crop_b64)
        for canonical_id, box in tracked_robots:
            robot = self.robot_tracks.get(canonical_id)
            if not robot:
                continue
            if robot.team_number_confirmed and not is_reread_pass:
                continue  # skip confirmed unless this is a re-read pass

            x1, y1, x2, y2 = map(int, box)
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Bumper band (lower part of the robot box).
            box_h = y2 - y1
            bumper_y1 = y1 + int(box_h * 0.35)
            crop = frame[bumper_y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            score = legibility_score(crop)
            if score < LEGIBILITY_FLOOR:
                continue  # blurry / back-facing / no bumper visible — don't waste a call

            # Upscale so the bumper crop is at least ~384px tall (existing behavior).
            if crop.shape[0] < 384:
                scale = max(2, int(np.ceil(384 / max(1, crop.shape[0]))))
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buf).decode('utf-8')
            candidates.append((score, canonical_id, box, crop_b64))

        if not candidates:
            return

        # Send the top-K legible crops to Qwen.
        candidates.sort(key=lambda c: c[0], reverse=True)
        chosen = candidates[:VLM_TOP_K]

        results = []
        ocr_crops = []
        for score, canonical_id, box, crop_b64 in chosen:
            res = _vision_read_bumpers([crop_b64], remaining_teams, remaining_blue, remaining_red)[0]
            results.append((canonical_id, box, res))
            ocr_crops.append({
                "robot_id": canonical_id,
                "alliance": res.get("alliance", "unknown"),
                "crop": crop_b64,
                "raw_read": res.get("raw_read"),
                "matched": res.get("team_number"),
                "legibility": round(score, 4),
            })
        self.last_ocr_crops = ocr_crops
        self._ocr_crops_dirty = True

        for canonical_id, box, result in results:
            team = result.get("team_number")
            if not team:
                continue
            robot = self.robot_tracks.get(canonical_id)
            if not robot:
                continue
            conf = result.get("confidence", 0.0)
            weight = 1.0 if conf >= 0.85 else 0.5
            if robot.team_ocr_votes is None:
                robot.team_ocr_votes = {}
            robot.team_ocr_votes[team] = robot.team_ocr_votes.get(team, 0.0) + weight
            tag = "RE-READ" if robot.team_number_confirmed else "read"
            print(f"  Vision {tag}: Robot {canonical_id} += {team} "
                  f"(w={weight}, total={robot.team_ocr_votes[team]:.1f})")
```

Notes for the implementer:

- The signature change (`tracked_robots` is still the same list of `(canonical_id, box)`) — call sites do not change.
- `_claude_vision_calls` is incremented in the caller (`web/app.py:~1663`) *before* this method is invoked; using `% REREAD_EVERY_N_PASSES == 0` cleanly cycles re-read passes.
- The upstream gate change (Step 2) and this `is_reread_pass` check use the same `% REREAD_EVERY_N_PASSES` cycle, so they stay aligned: a pass that opens the gate is exactly the pass that includes confirmed robots.

- [ ] **Step 4: Verify import + tests**

Run:
```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; all tests green.

- [ ] **Step 5: Commit**

```bash
git add web/app.py
git commit -m "feat(app): legibility-scored top-K crop selection + re-read confirmed robots

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Audit / overturn step

**Files:**
- Modify: `web/app.py` — add new method `audit_overturn_confirmations` and wire it into the loop

- [ ] **Step 1: Add the method (near `assign_teams_from_votes`)**

Insert after `assign_teams_from_votes` in `web/app.py`:

```python
    def audit_overturn_confirmations(self) -> None:
        """Let a wrong locked identity self-correct. For each confirmed robot,
        if a different roster team's accumulated VLM votes exceed the current
        label's by `OVERTURN_VOTE_MARGIN`, unlock the robot — the next
        `assign_teams_from_votes` will re-assign it from the now-broader pool."""
        for r in self.robot_tracks.values():
            if not r.team_number_confirmed or not r.team_number:
                continue
            new_team = reid.audit_overturn(
                current_team=r.team_number,
                votes=r.team_ocr_votes or {},
                margin=OVERTURN_VOTE_MARGIN,
            )
            if new_team is None:
                continue
            print(f"  Overturn: Robot {r.robot_id} {r.team_number} -> "
                  f"unlocked (challenger {new_team} dominates by margin)")
            r.team_number_confirmed = False  # next assign_teams_from_votes picks it up
```

- [ ] **Step 2: Wire it into the loop**

In `web/app.py`, replace the identity-interval block (currently from Task 1):

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()       # VLM-vote Hungarian (rework in Task 4)
                    # self.audit_overturn_confirmations()  # added in Task 6
```

with:

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()
                    self.audit_overturn_confirmations()
```

- [ ] **Step 3: Verify**

Run:
```
.venv/bin/python -c "import web.app; print('IMPORT OK')"
.venv/bin/python -m pytest tests/ -m "not model" -q
```

Expected: import clean; suite green.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(app): audit step lets a wrong locked identity self-correct

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Acceptance harness re-run

**Files:**
- Run only — no edits.

The harness exercises only continuity (no Qwen, no calibration references); this task verifies the rework didn't regress the collision-recovery work.

- [ ] **Step 1: Run the harness on the downloaded clip**

```bash
export PATH="$HOME/.deno/bin:$PATH"
.venv/bin/python scripts/reid_acceptance.py --video data/videos/qual37.mp4 --sample-rate 5 2>&1 \
  | grep -vE "Missing Sequence|pixel format|hardware accelerated" | tail -25
```

Expected:
- `Frames with a collision : 0`
- `Total ID flips : 0`
- `Total robots tracked : 6`

- [ ] **Step 2: Manual app run (out-of-band, surface to user)**

The full identity/alliance behavior requires a live app run on the match with Qwen and (optionally) calibration. The acceptance harness does not exercise the new code paths in `assign_teams_from_votes` or `vision_identify_robots`. After this plan lands, the user should run the live app on `data/videos/qual37.mp4` and confirm:

- No `Alliance rebalance ... (B0/R0)` lines in the new log.
- No robot ends with `team=<red-team> alliance=blue` (or vice versa) in the run-end state print.
- Process-of-elimination prints `Elimination: Robot N = Team T` if any robot is unread by the end.
- The HUD label flips correctly via `Overturn: Robot N ... -> unlocked` if a wrong lock occurred.

---

## Final verification

- [ ] Full non-model suite green: `.venv/bin/python -m pytest tests/ -m "not model" -v`
- [ ] Acceptance harness collision metric == 0 on `qual37.mp4`.
- [ ] Skill `superpowers:verification-before-completion` before claiming done. Do **not** assert "alliance bug fixed" until the live app run (Task 7 Step 2) shows the expected behavior — unit tests are necessary but not sufficient (we already learned this lesson this branch).

# Re-ID Collision Recovery & Algorithm 2 Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop multiple physical robots from collapsing onto one canonical track (the cause of the "VLM reads 2485 but HUD shows 2102?" bug), and run the embedding-identity assignment that the design intended but left unwired.

**Architecture:** Enforce one-canonical-track-per-robot per frame via a `claimed` set threaded into the pure continuity decision; re-validate cached ByteTrack→canonical mappings by appearance so ByteTrack ID-swaps self-heal; gate the EMA embedding update so a track centroid never gets contaminated; and call the already-implemented `_assign_alliances_by_embedding()` + `assign_identities_now()` each identity interval, after the VLM lock pass (VLM-primary, embeddings fill gaps).

**Tech Stack:** Python, NumPy, SciPy (`linear_sum_assignment`), pytest, OpenCV, Ultralytics YOLO + ByteTrack (`supervision`), DINOv2 via `transformers`.

**Spec:** `docs/superpowers/specs/2026-05-27-reid-collision-recovery-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `web/reid.py` | Pure continuity/identity decisions (no app state) | Add `claimed` param + `("drop", None)` to `choose_canonical_match`; add `_merge_score` helper (DRY) and `best_candidate_score` ordering helper |
| `web/app.py` | Live inference orchestration | `get_canonical_robot_id`: claimed + cached-ID appearance re-validation + drop handling. Main loop: batch embeddings, best-match-first ordering, per-frame `claimed`, EMA gating, drop-skip. Wire Algorithm 2 calls. |
| `scripts/reid_acceptance.py` | Offline acceptance harness | Thread the same continuity fixes; add a duplicate-canonical-per-frame metric and assert it is 0 |
| `tests/test_reid_continuity.py` | Unit tests for `choose_canonical_match` | Add claimed/drop tests |
| `tests/test_reid_ordering.py` | Unit tests for `best_candidate_score` | New file |

Pure-function changes (`web/reid.py`) are TDD'd directly. The `app.py` integration is verified by the acceptance harness (Task 6), since `get_canonical_robot_id` and the loop are coupled to app state.

**Run tests with:** `python -m pytest tests/ -v` (config in `pyproject.toml`; `testpaths=["tests"]`). Model-loading tests are marked `model`; deselect with `-m "not model"` when no GPU/weights.

---

## Task 1: `reid.choose_canonical_match` — claimed-set + drop action

**Files:**
- Modify: `web/reid.py` (the `choose_canonical_match` function, ~`35-86`)
- Test: `tests/test_reid_continuity.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_reid_continuity.py`. Update the `_decide` helper to accept an optional `claimed`:

```python
def _decide(det_pos, det_embed, det_alliance, tracks, claimed=None):
    return choose_canonical_match(
        det_pos=det_pos, det_embed=det_embed, det_alliance=det_alliance,
        tracks=tracks, app_floor=APP_FLOOR, pos_gate=POS_GATE,
        w_pos=W_POS, w_app=W_APP,
        cap_per_alliance=CAP_PER_ALLIANCE, cap_total=CAP_TOTAL,
        claimed=claimed,
    )


def test_claimed_track_is_never_returned():
    # Detection is visually identical + positionally on track 1, but track 1 is
    # already claimed this frame. It must NOT be returned.
    a = _e(1, 0, 0)
    tracks = [
        TrackView(1, "blue", (100, 100), a),
        TrackView(2, "blue", (400, 400), _e(0, 1, 0)),
    ]
    action, rid = _decide((100, 100), a, "blue", tracks, claimed={1})
    assert rid != 1  # cannot merge into the claimed track


def test_claimed_best_match_falls_to_create_when_caps_allow():
    # Only one (claimed) track exists and caps allow more → split into a new track.
    a = _e(1, 0, 0)
    tracks = [TrackView(1, "blue", (100, 100), a)]
    action, rid = _decide((100, 100), a, "blue", tracks, claimed={1})
    assert action == "create"


def test_caps_full_and_all_claimed_drops():
    # 6 tracks (3/alliance) all claimed this frame → a 7th detection cannot get a
    # canonical and must be dropped.
    tracks = [TrackView(i, "blue" if i < 3 else "red", (i * 10, 0), _e(1, 0, 0))
              for i in range(6)]
    action, rid = _decide((5, 0), _e(1, 0, 0), "blue", tracks, claimed=set(range(6)))
    assert action == "drop"
    assert rid is None


def test_caps_full_assigns_to_unclaimed_only():
    # Alliance full (3 blue); two claimed. The forced assign must pick the one
    # UNCLAIMED blue track, never a claimed one.
    tracks = [
        TrackView(1, "blue", (100, 100), _e(1, 0, 0)),
        TrackView(2, "blue", (105, 100), _e(0, 1, 0)),
        TrackView(3, "blue", (110, 100), _e(0, 0, 1)),
    ]
    action, rid = _decide((101, 100), _e(1, 0, 0), "blue", tracks, claimed={1, 2})
    assert rid == 3
    assert action in ("merge", "assign")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reid_continuity.py -v`
Expected: the four new tests FAIL with `TypeError: choose_canonical_match() got an unexpected keyword argument 'claimed'`.

- [ ] **Step 3: Implement the change in `web/reid.py`**

Add a shared scoring helper above `choose_canonical_match` (after `_dist`):

```python
def _merge_score(det_pos, det_embed, track, pos_gate, w_pos, w_app, app_floor):
    """Combined position+appearance score for merging det into a track.

    Returns None when the track is out of the position gate or vetoed by the
    appearance floor; otherwise w_pos*pos_score + w_app*app_sim.
    """
    d = _dist(det_pos, track.position)
    if d > pos_gate:
        return None
    pos_score = max(0.0, 1.0 - d / pos_gate)
    app_sim = _clamp01(cosine_similarity(det_embed, track.embedding)) if track.embedding is not None else 0.0
    # Appearance veto: never absorb a visually different detection.
    if track.embedding is not None and det_embed is not None and app_sim < app_floor:
        return None
    return w_pos * pos_score + w_app * app_sim
```

Replace the body of `choose_canonical_match` (keep the docstring; add `claimed` to the signature):

```python
def choose_canonical_match(
    det_pos, det_embed, det_alliance, tracks,
    app_floor: float, pos_gate: float, w_pos: float, w_app: float,
    cap_per_alliance: int, cap_total: int, claimed=None,
):
    claimed = claimed or set()
    avail = [t for t in tracks if t.robot_id not in claimed]
    same = [t for t in avail if t.alliance == det_alliance]
    other = [t for t in avail if t.alliance != det_alliance]

    def best_merge(candidates):
        best, best_score = None, -1.0
        for t in candidates:
            s = _merge_score(det_pos, det_embed, t, pos_gate, w_pos, w_app, app_floor)
            if s is not None and s > best_score:
                best, best_score = t, s
        return best

    # Primary path: merge by combined position + appearance score (with veto).
    merged = best_merge(same) or best_merge(other)
    if merged is not None:
        return ("merge", merged.robot_id)

    # Caps count ALL tracks (claimed or not) — they reflect physical robots.
    alliance_count = sum(1 for t in tracks if t.alliance == det_alliance)
    if alliance_count < cap_per_alliance and len(tracks) < cap_total:
        return ("create", None)

    if avail:  # caps full — assign to the best UNCLAIMED match (embedding first)
        if det_embed is not None and any(t.embedding is not None for t in avail):
            best = max(avail, key=lambda t: cosine_similarity(det_embed, t.embedding))
        else:
            best = min(avail, key=lambda t: _dist(det_pos, t.position))
        return ("assign", best.robot_id)

    if tracks:  # caps full AND every track already claimed this frame
        return ("drop", None)

    return ("create", None)  # absolute first-detection fallback
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reid_continuity.py -v`
Expected: all tests PASS (the 5 pre-existing tests still pass — `claimed` defaults to empty, preserving behavior).

- [ ] **Step 5: Commit**

```bash
git add web/reid.py tests/test_reid_continuity.py
git commit -m "feat(reid): per-frame claimed-set + drop action in choose_canonical_match

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `reid.best_candidate_score` — ordering key

**Files:**
- Modify: `web/reid.py` (add public helper after `choose_canonical_match`)
- Test: `tests/test_reid_ordering.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reid_ordering.py`:

```python
import numpy as np

from web.reid import TrackView, best_candidate_score

APP_FLOOR, POS_GATE, W_POS, W_APP = 0.55, 200.0, 0.4, 0.6


def _e(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _score(det_pos, det_embed, tracks):
    return best_candidate_score(det_pos, det_embed, tracks, POS_GATE, W_POS, W_APP, APP_FLOOR)


def test_no_candidate_returns_negative():
    assert _score((0, 0), _e(1, 0, 0), []) == -1.0


def test_closer_and_more_similar_scores_higher():
    a = _e(1, 0, 0)
    tracks = [TrackView(1, "blue", (100, 100), a)]
    near = _score((100, 100), a, tracks)   # identical + on top
    far = _score((100, 290), a, tracks)    # same look, 190px away (< gate)
    assert near > far > 0.0


def test_vetoed_match_does_not_count():
    a = _e(1, 0, 0)
    tracks = [TrackView(1, "blue", (100, 100), a)]
    # Visually orthogonal detection right on the track → vetoed → no candidate.
    assert _score((100, 100), _e(0, 1, 0), tracks) == -1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reid_ordering.py -v`
Expected: FAIL with `ImportError: cannot import name 'best_candidate_score'`.

- [ ] **Step 3: Implement in `web/reid.py`**

Add after `choose_canonical_match`:

```python
def best_candidate_score(det_pos, det_embed, tracks, pos_gate, w_pos, w_app, app_floor) -> float:
    """Best merge score of this detection against any track (claimed-agnostic).

    Used to order detections best-match-first so the strongest claimant binds a
    contested canonical track before weaker ones. Returns -1.0 if no track
    qualifies (out of gate or vetoed).
    """
    best = -1.0
    for t in tracks:
        s = _merge_score(det_pos, det_embed, t, pos_gate, w_pos, w_app, app_floor)
        if s is not None and s > best:
            best = s
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reid_ordering.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add web/reid.py tests/test_reid_ordering.py
git commit -m "feat(reid): best_candidate_score ordering helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `get_canonical_robot_id` — claimed param + cached-ID re-validation

**Files:**
- Modify: `web/app.py` `get_canonical_robot_id` (`1207-1241`)

No new unit test (method depends on app state); verified by the acceptance harness in Task 6. `cosine_similarity` and `reid` are already imported (`web/app.py:36-37`).

- [ ] **Step 1: Replace the method body**

Replace `get_canonical_robot_id` (`1207-1241`) with:

```python
    def get_canonical_robot_id(self, bytetrack_id: int, position: Tuple[float, float],
                               alliance: str, det_embed: Optional[np.ndarray],
                               claimed: set) -> Optional[int]:
        """Get or create the canonical robot ID for a ByteTrack ID.

        Enforces one canonical track per physical robot PER FRAME: a canonical
        already in `claimed` cannot be (re)used this frame. Cached ByteTrack
        mappings are re-validated by appearance so a ByteTrack ID-swap doesn't
        silently feed a different robot into the same track. Returns None when no
        canonical can be issued (caps full and all tracks claimed) — caller skips
        the detection.
        """
        cached = self.bytetrack_to_canonical.get(bytetrack_id)
        if cached is not None and cached in self.robot_tracks:
            track = self.robot_tracks[cached]
            appearance_ok = (
                det_embed is None or track.appearance_embedding is None
                or cosine_similarity(det_embed, track.appearance_embedding) >= APP_FLOOR
            )
            if appearance_ok and cached not in claimed:
                claimed.add(cached)
                return cached
            # Stale (appearance diverged) or contested (already claimed): drop the
            # mapping and re-resolve from scratch below.
            del self.bytetrack_to_canonical[bytetrack_id]

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
            claimed=claimed,
        )
        if action == "drop":
            return None
        if action in ("merge", "assign"):
            self.bytetrack_to_canonical[bytetrack_id] = rid
            claimed.add(rid)
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
        claimed.add(canonical_id)
        return canonical_id
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "import web.app"`
Expected: no error. (Does not exercise the new path yet — that's Task 4's loop wiring + Task 6's run.)

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(app): claimed-set + cached-ID appearance re-validation in get_canonical_robot_id

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Main loop — batch embeddings, best-match-first ordering, claimed, EMA gating

**Files:**
- Modify: `web/app.py` early-OCR pass (`1470-1481`) and main robot loop (`1523-1576`)

- [ ] **Step 1: Update the early-OCR pass call (`1470-1481`)**

The early pass loops detected robots and calls `get_canonical_robot_id`. Add a per-pass `claimed` set and pass it; skip dropped detections. Change the loop at `1470`:

```python
                    early_claimed = set()
                    for i in range(len(robot_dets)):
                        box = robot_dets.xyxy[i]
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        # Before frame 1600: left = blue, right = red
                        alliance = "red" if cx < midpoint else "blue"
                        det_embed = self.embedder.embed(early_overhead, box)
                        cid = self.get_canonical_robot_id(int(i + 9000), (cx, cy), alliance, det_embed, early_claimed)
                        if cid is None:
                            continue
                        robot = self.robot_tracks.get(cid)
                        if robot:
                            robot.last_position = (cx, cy)
                        early_robots.append((cid, box))
```

- [ ] **Step 2: Restructure the main robot loop (`1523-1576`)**

Replace the `if len(robot_detections) > 0:` block header and the start of the per-robot loop. Insert the batch-embed + ordering + claimed setup immediately after `tracked = self.tracker.update_with_detections(robot_detections)` (`1524`), and change the loop header from `for i in range(len(tracked)):` to iterate in best-match-first order:

```python
                if len(robot_detections) > 0:
                    tracked = self.tracker.update_with_detections(robot_detections)

                    # Embed every tracked crop once (single batched GPU forward).
                    n_tracked = len(tracked)
                    det_embeds = self.embedder.embed_batch(
                        overhead_frame, [tracked.xyxy[i] for i in range(n_tracked)]
                    )
                    # Order detections best-match-first so the strongest claimant
                    # binds a contested canonical before weaker detections.
                    tracks_view = [
                        reid.TrackView(r.robot_id, r.alliance, r.last_position, r.appearance_embedding)
                        for r in self.robot_tracks.values()
                    ]
                    def _order_key(i):
                        b = tracked.xyxy[i]
                        return reid.best_candidate_score(
                            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), det_embeds[i],
                            tracks_view, POS_GATE_PX, W_POS, W_APP, APP_FLOOR,
                        )
                    order = sorted(range(n_tracked), key=_order_key, reverse=True)
                    claimed = set()

                    # Process tracked robots
                    for i in order:
                        bytetrack_id = tracked.tracker_id[i]
                        box = tracked.xyxy[i]
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
```

Leave the alliance-determination block (`1533-1559`) unchanged.

Then change the canonical-resolution + EMA region (`1561-1576`). Replace:

```python
                        # Get canonical robot ID
                        det_embed = self.embedder.embed(overhead_frame, box)
                        canonical_id = self.get_canonical_robot_id(bytetrack_id, (center_x, center_y), alliance, det_embed)

                        # Update position (with frame number for trajectory matching)
                        robot = self.robot_tracks[canonical_id]
                        if det_embed is not None:
                            if robot.appearance_embedding is None:
                                robot.appearance_embedding = det_embed
                            else:
                                blended = (1 - EMBED_EMA_ALPHA) * robot.appearance_embedding + EMBED_EMA_ALPHA * det_embed
                                n = np.linalg.norm(blended)
                                robot.appearance_embedding = blended / n if n > 0 else blended
                        robot.last_position = (center_x, center_y)
```

with:

```python
                        # Get canonical robot ID (claimed-aware; may return None = drop)
                        det_embed = det_embeds[i]
                        canonical_id = self.get_canonical_robot_id(
                            bytetrack_id, (center_x, center_y), alliance, det_embed, claimed)
                        if canonical_id is None:
                            continue  # surplus detection this frame — cannot be a 7th robot

                        # Update position (with frame number for trajectory matching)
                        robot = self.robot_tracks[canonical_id]
                        # EMA only when the crop actually matches the track, so a
                        # transitional/mismatched detection never poisons the centroid.
                        if det_embed is not None:
                            if robot.appearance_embedding is None:
                                robot.appearance_embedding = det_embed
                            elif cosine_similarity(det_embed, robot.appearance_embedding) >= APP_FLOOR:
                                blended = (1 - EMBED_EMA_ALPHA) * robot.appearance_embedding + EMBED_EMA_ALPHA * det_embed
                                n = np.linalg.norm(blended)
                                robot.appearance_embedding = blended / n if n > 0 else blended
                        robot.last_position = (center_x, center_y)
```

The rest of the loop body (`positions_history`, `tracked_robot_boxes.append`, bumper-color votes, `detection_to_tracked` mapping at `1591-1600`) is unchanged — it already runs per processed detection and is correctly skipped for dropped detections by the `continue`.

- [ ] **Step 3: Verify import + smoke compile**

Run: `python -c "import web.app"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add web/app.py
git commit -m "feat(app): batch embeddings, best-match-first ordering, per-frame claimed, EMA gating

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Wire in Algorithm 2 (currently dead code)

**Files:**
- Modify: `web/app.py` loop scheduling (`1619-1620`)

- [ ] **Step 1: Add the two calls after the VLM lock pass**

Replace (`1619-1620`):

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()
```

with (VLM-primary precedence: locks first, then robust alliance, then embeddings fill the unlocked tracks — `assign_identities_now` already excludes confirmed tracks/teams):

```python
                if processed_frame_num % IDENTITY_INTERVAL == 0:
                    self.assign_teams_from_votes()       # VLM locks (primary)
                    self._assign_alliances_by_embedding()  # robust alliance via embedding clustering
                    self.assign_identities_now()         # embeddings fill unlocked tracks (uniqueness)
```

- [ ] **Step 2: Verify import**

Run: `python -c "import web.app"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat(app): wire _assign_alliances_by_embedding + assign_identities_now into loop

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Acceptance harness — thread fixes + duplicate-canonical metric

**Files:**
- Modify: `scripts/reid_acceptance.py` resolution loop (`262-328`), metrics section, and `choose_canonical_match` call (`294`)

This makes the offline harness faithful to production and adds the metric that directly measures the bug.

- [ ] **Step 1: Add per-frame claimed + ordering + re-validation + EMA gating + a collision counter**

In the resolution loop (around `266-328`), introduce a frame-level `claimed` set and a module/run-level `duplicate_canonical_frames` counter. Replace the per-detection loop so it:
1. Batch-embeds all tracked crops for the frame.
2. Orders detection indices by `reid.best_candidate_score(...)` descending.
3. For each, applies the cached-ID fast path with appearance re-validation (`cosine(det_embed, track["embedding"]) >= APP_FLOOR`) and the `claimed` check; otherwise calls `reid.choose_canonical_match(..., claimed=claimed)` and handles `("drop", None)` by skipping.
4. Marks the resolved `cid` in `claimed`; if a `cid` is about to be added that is already in `claimed`, increment `duplicate_canonical_frames` once for that frame and skip.
5. Gates the EMA update on `cosine >= APP_FLOOR`.

Mirror the exact logic of `web/app.py:get_canonical_robot_id` (Task 3) and the loop (Task 4), adapted to the harness's `robot_tracks[cid]` dict shape (`{"alliance","position","embedding","team_number"}`).

- [ ] **Step 2: Add the metric to the summary**

In the metrics/printout section (after the existing Metric 1 block, ~`385`), add:

```python
    # --- Metric 2 (collision): frames with a duplicate canonical -------------
    print()
    print(f"Metric 2 — frames where two detections claimed one canonical: {duplicate_canonical_frames}")
    assert duplicate_canonical_frames == 0, (
        f"FAIL: {duplicate_canonical_frames} frame(s) had two detections collapse "
        f"onto one canonical track — the collision bug is not fixed."
    )
```

- [ ] **Step 3: Run the acceptance harness on a clip**

Run (substitute an actual clip under `data/videos/2026/`):

```bash
python scripts/reid_acceptance.py --video data/videos/2026/<clip>.mp4 2>&1 | tail -40
```

Expected:
- "Metric 2 — frames where two detections claimed one canonical: 0"
- Metric 1: distinct team labels per canonical trends toward 1 (no track holding 5 teams like the logged `Robot 4`).
- No duplicate-team-number assertion failure.

If the harness has a `--help`-only path or needs a model/clip not present, run `python scripts/reid_acceptance.py --help` to confirm it imports, and note the clip needed for a human to run.

- [ ] **Step 4: Commit**

```bash
git add scripts/reid_acceptance.py
git commit -m "test(acceptance): thread collision fixes + duplicate-canonical metric

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7 (advisory, lowest priority): Algorithm 2 composition test

**Files:**
- Test: `tests/test_identity_composition.py` (new)

Spec test #4 / reviewer recommendation. Guards that re-clustering alliances after the vote pass does not flip a confirmed robot's alliance or destabilize a freshly tentative one. `_assign_alliances_by_embedding` and `assign_teams_from_votes` are `InferenceJob` methods, so the test constructs a minimal `InferenceJob` and populates `robot_tracks` directly.

- [ ] **Step 1: Write the test**

```python
# Construct an InferenceJob with stubbed robot_tracks (3 blue + 3 red, embeddings
# set), confirm two robots, run _assign_alliances_by_embedding(), and assert:
#   - confirmed robots keep their team-derived alliance
#   - no alliance ends up with >3 robots
```

(Fill in using `web.app.InferenceJob` and `RobotTrack`; set `match_blue_teams`/`match_red_teams`/`match_teams`. Keep it focused on the invariant, not full pipeline behavior.)

- [ ] **Step 2: Run + iterate to green, then commit**

```bash
python -m pytest tests/test_identity_composition.py -v
git add tests/test_identity_composition.py
git commit -m "test: Algorithm 2 alliance-clustering composition invariant

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the full unit suite: `python -m pytest tests/ -m "not model" -v` → all green.
- [ ] Re-run the acceptance harness (Task 6 Step 3) and confirm Metric 2 == 0.
- [ ] Manual: re-run a live analysis on the screenshot's match and confirm a blue robot whose crop reads `2485` is no longer labeled `2102?`, and that ≤6 canonical tracks exist. Use `superpowers:verification-before-completion` before claiming done.

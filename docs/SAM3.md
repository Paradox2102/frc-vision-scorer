# SAM 3.1 Tracking Backend

SAM 3.1 is available as an experimental alternative to the default SAM 2.1
tracker. The rest of the field-tracking pipeline is shared between backends.

## Setup

1. Request access to `facebook/sam3.1` on Hugging Face.
2. Authenticate on the inference host with `hf auth login`.
3. Install the optional dependency set:

   ```bash
   pip install -e '.[sam3]'
   ```

The SAM3 package requires NumPy `<2`, so this install may downgrade an existing
NumPy 2.x environment. A separate deployment environment is recommended while
the backend remains experimental.

## Run

Select **SAM 3.1** in the web upload form, or set it as the process default:

```bash
TRACKING_BACKEND=sam3 python web/app.py
```

The checkpoint downloads from Hugging Face on first use. To use a pre-downloaded
checkpoint instead:

```bash
SAM3_CHECKPOINT=/models/sam3_base.pt TRACKING_BACKEND=sam3 python web/app.py
```

`SAM3_CHECKPOINT` alone only works with a **base-architecture** checkpoint
(the one `build_sam3_video_predictor` loads by default, opt-in `sam3_multiplex`
OFF). It does NOT work with a multiplex checkpoint (e.g. `sam3.1_multiplex.pt`)
-- the base builder's strict state-dict load can't consume the multiplex
checkpoint's keys. To run the multiplex checkpoint, enable the `sam3_multiplex`
knob below instead; it selects the multiplex builder, which resolves its own
checkpoint (either `SAM3_CHECKPOINT` if set, or an auto-download of
`facebook/sam3.1`'s `sam3.1_multiplex.pt`).

Optional H100 optimizations:

```bash
SAM3_COMPILE=1 SAM3_USE_FA3=1 TRACKING_BACKEND=sam3 python web/app.py
```

`SAM3_USE_FA3=1` requires the optional Flash Attention 3 installation described
by Meta. Leave it disabled until that dependency is installed on the target host.
Note: `SAM3_USE_FA3` is currently accepted for call-site compatibility but not
wired to either builder -- the base builder (`sam3_multiplex` OFF) doesn't take
a `use_fa3` argument at all, and the multiplex builder (below) always forces
`use_fa3=False` regardless of this env var.

## SAM 3.1 Object Multiplex (`sam3_multiplex`, default OFF)

`SAM3_MULTIPLEX=1` (or the `sam3_multiplex` per-job knob) swaps the base SAM3
video predictor for Meta's recommended SAM 3.1 entry point,
`build_sam3_predictor(version="sam3.1", ...)`, which builds a joint
multi-object "multiplex" predictor: per Meta, roughly 2x propagation
throughput and about half the VRAM per tracked object versus the base
builder. It dispatches through the same request protocol
(`start_session`/`add_prompt`/`propagate_in_video`/`remove_object`), so the
rest of the pipeline (re-prompting, culling, fusion, identity) is unaffected.

This is **not a pure performance toggle**. The multiplex builder bakes in
different tracking heuristics than the base builder:

| heuristic                     | base | multiplex |
|--------------------------------|------|-----------|
| `new_det_thresh`               | 0.7  | 0.65      |
| `score_threshold_detection`    | 0.5  | 0.4       |
| `suppress_det_close_to_boundary` | off | on      |
| `masklet_confirmation_enable`  | off  | on        |

Lower detection thresholds plus boundary suppression and masklet confirmation
shift mask, re-seed, and occlusion behavior versus the base path -- this is
why the knob defaults OFF pending a qm8-qm12 A/B, not because of any known
regression.

`use_fa3` is always forced `False` when building the multiplex predictor:
its builder default is `use_fa3=True` (FlashAttention-3), which is NOT
installed on this host. Bucket capacity (`max_num_objects` /
`multiplex_count`, builder default 16) is left untouched -- plenty for our
~6-robot rosters, and an untested larger capacity is a bigger risk than the
default.

## Selective mask propagation (`selective_mask_prop`, default OFF)

Re-implementation of arXiv 2606.13033 (Holmberg 2026). The reference project
[holma91/selective-mask-propagation](https://github.com/holma91/selective-mask-propagation)
is MIT in its own code but vendors third-party components under other licenses
including GPL-3.0, so rather than vendor it we re-implemented its ~4-file
algorithm surface clean-room over our own predictor protocol, cost units, and
evidence schema.

### The inversion

Every other tracking path in this doc runs SAM propagation on **every**
frame for **every** tracked object: full-frame mask decode, encode, and
memory-bank update, whether or not the object's identity is actually in
doubt. Profiling attributes about 77% of a job's wall clock to that
propagation stage (see the memory notes on job perf), so it is the only
lever left with real headroom.

`selective_mask_prop` inverts the architecture: a cheap per-frame
tracking-by-detection (TBD) pass runs on *every* frame for *every* object
(Hungarian assignment on expanded-IoU cost, no GPU), and the expensive SAM3
predictor is invoked **only** inside short windows where the TBD assignment
is actually ambiguous -- a close second-best competitor, or a tracked object
reappearing after a real gap. Outside those windows the TBD box **is** the
tracked output; no mask, no propagation cost. This targets the plan's P2
(propagation is the dominant cost), P3 (most frames are unambiguous -- SAM's
precision is wasted on them), and P4 (paying for precision only where it
resolves an actual ambiguity) findings from the July 2026 tech-upgrade
review.

`selective_mask_prop` is a per-job/env knob (`SELECTIVE_MASK_PROP`) and
defaults OFF pending a qm8-qm12 A/B, same posture as `sam3_multiplex` above.

### Knobs

| knob | env var | default | meaning |
|------|---------|---------|---------|
| `selective_mask_prop` | `SELECTIVE_MASK_PROP` | `False` | master switch for the whole selective path |
| `selective_margin_entry` | `SELECTIVE_MARGIN_ENTRY` | `0.05` | TBD assignment margin below which a frame opens an ambiguity window |
| `selective_margin_exit` | `SELECTIVE_MARGIN_EXIT` | `0.10` | margin an assignment must clear to count as a "clean" seed/settle frame |
| `selective_gap_trigger` | `SELECTIVE_GAP_TRIGGER` | `7` | consecutive missed frames before a reappearing track opens a window |
| `selective_min_streak` | `SELECTIVE_MIN_STREAK` | `5` | consecutive clean frames required for both the backward seed walk and the exit machine's settle |
| `selective_window_max_frames` | `SELECTIVE_WINDOW_MAX_FRAMES` | `120` | hard cap on a window's SAM-propagated span before it force-closes healthy (`end`) |
| `selective_sam_budget_frac` | `SELECTIVE_SAM_BUDGET_FRAC` | `1.0` | cross-episode SAM-frame budget, as a fraction of the view's frame count |
| `selective_app_weight` | `SELECTIVE_APP_WEIGHT` | `0.0` | appearance cost weight in the TBD assignment cost (same-color pairs only); `0.0` = off |
| `selective_app_ema` | `SELECTIVE_APP_EMA` | `0.9` | track-gallery EMA retention on confident matches |

Only these nine are job_config knobs / env vars. The other exit/seed
thresholds referenced in the algorithm below (`area_degradation`,
`mask_overlap_exit`, `ioma_exit`, `seed_clean_iou`, `border_margin_px`,
`iou_expand`, `max_missed`) are **internal `SelectiveConfig` constants** in
`web/field/selective_prop.py`, not exposed through job_config or env -- change
them in source if an A/B warrants.

**Caveat:** the margin knobs are in **1 - expanded-IoU cost units** (this
repo's Deep-EIoU-style cost, `web/field/selective_prop.py`), **not** the
units used in the paper's Eq. 1. The defaults above are the paper's
start-points carried over verbatim; they have not yet been re-tuned to this
codebase's cost scale. Treat them as an A/B starting point, not a validated
value -- see the `selective` / `selective-sweep` profiles in
`scripts/ab_qm8.py` below.

### Algorithm, in brief

1. **Per-frame TBD** (`TbdTracker.step`, image-space, no GPU): each frame,
   Hungarian-match live track predictions (constant-velocity) against raw
   per-frame detections on expanded-IoU cost. Each assignment's *margin* is
   the cheapest competing (finite-cost) track's cost minus the winner's cost
   for that detection -- large margin means unambiguous, small margin means
   another track could plausibly have matched instead (paper Eq. 1).
2. **Window construction** (`find_windows`, offline over the completed TBD
   history): a window opens when an assignment's margin drops below
   `selective_margin_entry`, or when a track reappears after
   `selective_gap_trigger`+ missed frames. Any other object whose box
   overlaps a trigger object's box at the entry frame is pulled in too as a
   witness (SAM needs the neighbor prompted or it will bleed onto it).
   Each candidate then runs a **backward seed walk**: walk back from the
   entry frame to find the nearest run of `selective_min_streak` consecutive
   *clean* frames (margin >= `selective_margin_exit`, box not overlapping
   any other object) -- that run's earliest frame is where SAM gets seeded.
   Candidates with no clean seed run are dropped.
3. **SAM3 forward through the window** (`Sam2FieldTracker.track_selective`,
   Task 4/5 orchestration): windows sharing an *episode* (the interval-merged
   cluster described below) are grouped into one predictor session, seeded
   with *only* that group's objects (`solo_ids`) -- other tracked objects
   never enter the SAM session for a window that doesn't need them.
   Propagation runs frame-by-frame through
   `min(selective_window_max_frames, remaining video)`.
4. **Exit state machine** (`ExitMachine`): watches the propagated masks from
   the entry frame onward and closes each window as soon as it has an
   answer. Healthy closes (`clean`, `swap`, `end`) contribute their masks to
   the merged output; everything else is an abort and the TBD-only output
   stands unchanged for that window's frames. States: `stale` (seeded mask
   never covers the TBD box at entry -- SAM never latched on), `degraded`
   (mask goes dark, or its area collapses below `area_degradation` x seed
   area, for `selective_min_streak` frames), `edge` (mask goes dark right
   after sitting near the frame border -- likely exited-frame, not lost),
   `killed` (two windows' masks converge onto the same object, pixel-IoU >=
   `mask_overlap_exit`, for `selective_min_streak` frames -- both die),
   `clean` (mask settles back onto its own TBD box for `selective_min_streak`
   frames -- no identity change, matches the paper's "accept only on
   confident contradiction": SAM ran, agreed with TBD, contributes masks but
   no rename), `swap` (mask instead settles onto a *different* object's box
   -- SAM found an identity swap TBD couldn't see; this is the paper's
   confident-contradiction case and the only one that renames), `end`
   (window hits `selective_window_max_frames` or end-of-video still
   unresolved -- closes healthy since nothing contradicted TBD).
5. **Merge** (`merge_selective_results`): the base stream is pure TBD
   everywhere (`mask=None`, evidence `"tbd"`). `swap` outcomes apply pairwise
   relabeling transpositions (composed chronologically, raw-label side) to
   the whole downstream stream from their entry frame. Then every healthy
   window's SAM masks overlay their oid's entries across `[entry, close]`
   with the real mask, `mask_to_box(mask)`, and evidence `"selective_sam"`
   carrying the frame's `tracker_prob`.

### Episode merging and the SAM budget

Two windows opened independently by Phase B can still need the same stretch
of video -- e.g. a witness pulled into one window's entry frame is also the
trigger object for a second window a few frames later. `cluster_windows`
merges any windows whose padded `[seed_frame - min_streak, entry_frame +
min_streak]` intervals overlap (standard sort-and-sweep interval merge) into
one *episode*, and `track_selective` runs exactly one SAM predictor session
per episode rather than one per window. This is not just a cost win: an
episode's single `ExitMachine` sees every concurrent mask in the group each
step, so the `killed` (mutual mask convergence) and border/edge checks can
actually compare windows against each other -- two windows split across
separate sessions would never observe one another's masks at all.

`selective_sam_budget_frac` puts a hard ceiling on that cost: total SAM
frames fed to any `ExitMachine`, summed across a view's episodes, are capped
at `sam_budget_frac * view_frame_count`. The default, `1.0`, means selective
mode can never cost more SAM propagation than the uniform `track()` path
would have for the same view (the budget bounds propagation frames only;
per-episode `init_state`/prompt cost is separate, and episode merging
strictly reduces it versus the old one-session-per-seed-frame grouping)
-- the motivating case was a qm11 main view
with 17 overlapping windows that, before episode merging existed, drove
`sam_frames` to 213% of uniform (`web/logs/20260719_222247_ec06450f*.log`).
Episodes run in chronological order; one that cannot fit the remaining
budget (a conservative estimate of its minimum cost) is skipped whole
(`dropped_windows` in the stats/log line) so a smaller, later episode still
gets a chance, and those frames' TBD-only output stands unchanged. If the
budget instead runs out in the middle of an episode, propagation stops and
every window still open in that episode force-closes healthy as `end` --
the masks collected up to that point are kept, nothing is discarded.

Note that `selective_window_max_frames` (`120`) is a per-window cap, not a
per-view one -- a single auto-only qm11 view is 181 frames, so one window
can already span most of it. The per-view SAM budget is what actually bounds
aggregate cost; the per-window cap only bounds a single window's worst case.

### Evidence types and the `mask=None` contract

Two evidence `type`s appear in `sam2_raw` under selective mode:

- `"tbd"` -- `mask=None`, `tracker_prob=None`. The box is the TBD tracker's
  matched/predicted detection; no SAM ever touched this frame/object.
- `"selective_sam"` -- a real mask, carrying the propagated `tracker_prob`.
  Only present inside a healthy window's `[entry_frame, close_frame]` span.

Every downstream consumer of `info["mask"]` was audited (Task 6) to treat
`mask=None` as a first-class case, not a bug:

- `ground_uv` (`web/app.py`) falls back to the box bottom-center when
  `mask` is None (or `mask_footprint_pos` is off, or the mask is empty).
- `RobotEmbedder`/`_apply_mask` (`web/embedder.py`) return the crop
  unmodified (unmasked) when the mask argument is None -- the DINOv2/DINOv3
  embedding is computed on the plain box crop.
- `_restore_sam2_mask_to_view` (`web/app.py`) returns `None` (not an
  all-False array) when its input mask is `None`, so callers can tell "no
  mask" from "empty mask".
- The HUD mask overlay (`web/app.py`, `HUD_DRAW_ROBOT_MASK`) only draws when
  `mask is not None`; TBD-only frames simply show no silhouette.

### Fallback behavior

`Sam2FieldTracker._run_view_tracking` (`web/app.py`) calls `track_selective`
only when `selective_mask_prop` is on. **Any** exception it raises (other
than a user-initiated stop) is caught, logged as
`[selective] FALLBACK view=<name>: <error>`, and the view falls through to
the identical uniform `track()` call used when the knob is off -- a bad
selective run can never lose a view's tracks outright.

### Interaction with other knobs

- `mask_footprint_pos` degrades to box-bottom-center for every frame
  **outside** a selective window, because there is no mask there to derive a
  footprint from (see the `ground_uv` contract above). Inside a healthy
  window it works exactly as it does under uniform tracking.
- `late_seed_backfill` and `propagate_backward` are **no-ops** in selective
  mode. Both are parameters/logic that live entirely inside `track()`'s
  forward-propagation loop; `track_selective` is a separate method that
  never calls into that code path. This is not a gap needing a fix: selective
  mode's per-frame TBD detection already covers every frame from the start,
  so there is no "late-seeded" history left to backfill and no bootstrap
  frame whose earlier history needs a backward pass.
- DEFAULT OFF pending a qm8-qm12 A/B (`scripts/ab_qm8.py` profiles
  `selective` and `selective-sweep`), same posture as `sam3_multiplex`: no
  known regression, just not yet validated on the identity-heavy qm11/qm12
  matches.

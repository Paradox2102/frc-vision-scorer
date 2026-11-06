# SAM 3.1 multiplex: end-to-end integration and live-speed verdict (2026-11-06)

**Question:** can `sam3_multiplex` cut a full match below 12 min, ideally below 8, without losing accuracy? The Nov 6 speed A/B showed SAM at roughly 420–590 ms/frame/view regardless of fps.

**Verdict: no. Keep `sam3_multiplex` OFF.** It now runs end to end without crashing on auto-only matches. It is **slower** than base SAM3, though: +26% job wall time on qm11 auto-only, and main view 727 vs 456 ms/frame. It also still OOMs on full matches. The design cannot be the speed lever: instance-prompted objects are never batched.

Branch `claude/loving-saha-ae51d1` (on top of `live-speed` 3bd77ddd), not merged.

## What was broken: three upstream layers, all fixed in `web/field/sam3_tracker.py`

| # | Symptom | Root cause (sam3 package) | Fix |
|---|---|---|---|
| 1 | `KeyError: <int>` on every prompted view (the Jul-20 bug) | Not an obj-id mismatch. The 3.1 builder defaults `max_num_objects=16`, while base SAM3 has no limit. The pipeline mints a new obj_id on every (re)seed, up to 36 per view, so adds were silently refused (`outputs: None`). A window with no accepted add then parses as `propagation_fetch`, and multiplex `fetch_and_process_single_frame_results` indexes `cached_frame_outputs[f]` without a guard (base uses `.get(f, {})`). The KeyError value is the **frame index** where the cap filled, which is why it tracked prompt count. | Build with `max_num_objects=max(64, max_objects)`. Guarded fetch shim. Refused adds are counted and logged (`refused_prompts=K`). 9a8b4bc2 |
| 2 | Crash gone, but the qm11 auto job exported 3 slots × **2 samples** | `_build_sam2_output` starts with `if frame_idx not in cached_frame_outputs: return {}`, which **discards the freshly propagated masks** on every frame not already cached. Meta's demo always runs a text-prompt `propagation_full` first, which caches every frame; pure instance prompting never does. Masks survived only on prompt frames, tracks died one frame later, and re-seeding burned the 36-id lifetime budget. | Shim that mirrors base semantics (cached-or-empty, plus refined overlay). The test runs the real upstream method and fails if upstream fixes it. a783947f |
| 3 | Every view hit CUDA OOM about 20 frames in | Each instance-prompted object gets a **singleton** SAM2 state whose memory is laid out in mux space: about 19 MB of maskmem per object per frame, roughly 30× SAM2. Every non-cond and cond frame is kept, because the tracker's `trim_past_non_cond_mem_for_eval` path is never reached by the interactive step. Offline: +0.27 GB/frame with 3 objects. | After each frame, strip heavy tensors outside the memory-attention window (non-cond > r·num_maskmem away; cond outside first + 6 closest). Entries are kept, not deleted. Offline VRAM is now flat (peak 9.7 GB vs 15.6 GB and climbing) with **byte-identical** masks. 90ad665f |

Also fixed: harness scoring (bfd69865). `load_our_auto_paths` keeps the newest replay per **slot**, so a job missing slots silently inherited them from older runs; the `expect_replay` guard only checked that one slot was the job's. `score()` now loads only the job's replay. Rescoring the B4 baselines in isolation reproduces 0.795 / 0.751 exactly. Propagation errors now print a full traceback (`web/app.py`).

**Trap seen twice:** a job whose views all OOM still "completes" via the no-SAM3 detector fallback and scores well. One intermediate mux run scored qm11 auto 0.505. Always grep `DEGRADED] sam3` / `Error during propagation`.

## Measurements

### qm11 auto-only, same code, back-to-back, server restarted (fps 4, corner stride 3, stream on, jev)

| arm | job | wall | SAM tracking | main | corner_left | corner_right | agg meanNN | slots |
|---|---|---|---|---|---|---|---|---|
| B4 base SAM3 | c716ee47 | **143 s** | 111 s | 456 ms/f | 285 | 407 | 0.939 | 6 |
| B4 + multiplex | 4f449595 | **180 s** (+26%) | 148 s | 727 ms/f | 332 | 470 | 0.988 | 6 |

Coverage, mux vs base: blue 4786 80 vs 94, 3181 100 vs 79, 2992 61 vs 99; red 4256 84 vs 99, 1477 99 vs 35, 6800 28 vs 38. Mislabels 3 vs 2. Accuracy is roughly comparable on one 15 s window (single run each, so noise-level), but it is not faster.

### Offline adapter A/B (40 qm11 frames, 3 identical box prompts, stepped 3-frame propagation)
- ms per yielded frame: base 248, mux 264 (after all three fixes).
- VRAM: base +0.02 GB/frame; mux +0.27 GB/frame before the memory bound, flat after it.

### Full match (qm8 / qm11): not measurable
With multiplex, main and corner_right OOM on the full match even with the memory bound:
- default 200-frame chunks (job e543c307);
- `sam3_chunk_frames=100` (job 0d546ed0), where the second main chunk `[103,203)` OOMs with 10–12 live objects.

Some further per-object/per-frame accumulation remains (not root-caused). Because of that there are no full-match wall-time, meanNN or `identity_offline_check --swap` numbers for multiplex.

B4 references, for the record:
- qm8 bab02709: 754 s, agg 0.795, main/cl/cr 545/533/488 ms/f, identity 5/6 (auto: gir 5/6, swap 3/6).
- qm11 33c38824: 704 s, agg 0.751, 550/391/437 ms/f, identity 3/6 (auto: gir 3/6, swap 3/6).

## Why multiplex cannot be the lever
1. **No batching for our objects.** In `add_sam2_new_points`, every new instance-prompted object gets its own singleton state (`_init_new_sam2_state`), and `_propogate_tracker_one_frame_local_gpu` loops over those states serially. Only detector-born objects (from `propagation_full`) share multiplex buckets. Each of our objects pays a full 16-slot mux-space memory, plus a whole singleton `output_dict` rebuild (with clones) on every refine, and the adapter refines every live object every window. That is why it is slower than base.
2. **Per-object cost is small anyway.** Base SAM3 costs about 400–440 ms/frame on a view tracking **zero** objects (qm11 corner_right in B4: 437 ms/f, empty roster) versus about 456–550 with 3 objects. Perfect batching could remove only about 15–30 ms/frame per object; the per-frame fixed cost dominates.

**Where the time actually goes:** the per-frame fixed cost, presumably the image backbone at 1008 px plus feature caching. That is the lever toward 8 minutes and is worth a profiling pass of its own.

## Known limitations of the shims (from independent review)
- A stale object refined more than 7 frames back (allowed up to `SAM3_RESEED_MAX_AGE`=24) gets no memory conditioning, because those frames were trimmed. This is a narrow quality change, not a crash.
- Latent: upstream `remove_objects._slice_state` indexes `maskmem_features` without a guard. It would KeyError on a trimmed entry if `remove_object` were ever called on a multi-object state. The adapter never calls it, and all states are singletons.
- The `speed-mux` / `speed-mux-c100` profiles in `scripts/ab_qm8.py` reproduce these runs.

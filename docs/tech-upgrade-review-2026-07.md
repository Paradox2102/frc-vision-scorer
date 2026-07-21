# Technology Upgrade Review — July 2026

Survey of technologies worth adding or upgrading for better **tracking**, **identifying**,
and **2D movement-track plotting**, grounded in the current pipeline internals and a
web-research pass over the mid-2026 state of the art (detection/tracking, identity/OCR/re-ID,
and field registration/multi-view/smoothing). Companion to
`frc-vision-geometry-todo.md`, which covers leverage on geometry we *already have*;
this doc covers *new tech*.

> Claims about post-January-2026 releases come from the web-research pass (URLs inline).
> Verify version/license details before adopting anything.

---

## Current stack snapshot (confirmed in-repo)

- **Detection**: 3 fine-tuned YOLOs — bumper detector (hard-negative equipment set),
  `hub_zone_yolo26s` field model, person model. ultralytics **8.4.21** (latest is 8.4.101).
- **Tracking**: SAM 3.1 chunked mask propagation (`sam3_chunk_frames=200`, re-prompt 2 s,
  `late_seed_backfill=True`), repo pin = upstream commit 2026-05-23. **77% of wall-clock.**
- **Re-ID**: DINOv3 ViT-S CLS 384-d gallery (timm), relink floor 0.55.
- **Identity**: Qwen3-VL-4B via vLLM, guided JSON, 3-lock/2-flip voting, backward
  re-resolution; EasyOCR corroboration at 0.3 vote weight; `legibility.py` is a
  hand heuristic (HSV × contrast × Laplacian), not a classifier.
- **2D tracks**: full pinhole/fisheye resection → ray ∩ ground plane at **box bottom-center**
  (`mask_footprint_pos` exists, default OFF) → greedy cross-view clustering + ray-gate →
  hand-rolled **causal** constant-velocity Kalman (6 fixed slots, coast decay 0.85).
  **No backward smoother.** Drift check aborts at match start; mid-match monitor warns only;
  pose autofix exists but is staged for human approval.
- Crops fed to the VLM get only `cv2.resize` — no learned super-resolution anywhere.

## Pain-point map (what each upgrade attacks)

| # | Pain point | Evidence |
|---|-----------|----------|
| P1 | Identity binding is the bottleneck (fonts unreadable, same-alliance re-ID fails, misattributed votes, dup-tid flips, run-to-run nondeterminism) | README limitations; memory notes |
| P2 | SAM3 propagation = 77% of job time | perf profile |
| P3 | Equipment hijacks, detector-corroborated | qm10/qm12; hard-negative retrain is the accepted fix |
| P4 | Fast robots missed by seeding | qm11 6800; backfill now default-ON helps |
| P5 | Occlusion gaps coast-and-decay; causal filter only | tracker.py internals |
| P6 | Manual calibration; abort-on-drift | calibration.py / drift_monitor.py |
| P7 | Corner cams: identity-only, terrible depth | fusion internals |

---

## Tier 1 — cheap, high leverage (days each; no new models needed for 1–2)

### 1. RTS smoother (+ optional IMM) over the 2D tracks — attacks P5, directly moves the Manta `agg` metric
> **Status 2026-07-20:** built (`rts_smooth`, merge 460b3688), A/B'd on qm11+qm8 —
> measured samples improve on every robot, but long-gap insertions regress the
> compare_manta gate (qm11 0.816→0.890). Default stays OFF pending a max-gap cap
> on insertion. Full data: docs/default-on-ab-2026-07-20.md.

The pipeline is **offline** but the exported `field_pos` is the raw causal KF estimate.
A Rauch–Tung–Striebel backward pass is a few dozen lines over the existing hand-rolled
KF (store per-frame `x, P, F, Q`; run the standard backward recursion) and turns every
occlusion gap from "coast with velocity decay" into a two-sided interpolation. Literature
gains for forward-vs-smoothed on comparable problems: tens of percent of track error,
concentrated exactly in gaps and maneuvers. An IMM bank (CV/CA/coordinated-turn) helps
agile stop/sprint/pivot targets (11–69% RMSE reductions reported), but do RTS first —
it's the 80/20.
- Libraries if not hand-rolling: **Stone Soup** (dstl, v1.7 Oct 2025, MIT — IMM +
  smoothers + association) <https://github.com/dstl/Stone-Soup>; **dynamax** (JAX,
  parallel smoothers, EM to fit Q/R per event) <https://github.com/probml/dynamax>.
  filterpy is unmaintained; pykalman was revived Nov 2025.
- Keep the causal estimate for the live HUD; export the smoothed track for radar
  trails, autonomous paths, and Manta comparison.
- Stretch (probably overkill at N=6): a GTSAM 4.x factor graph holding raw per-view
  image observations + motion prior + per-segment camera nodes — folds naturally into
  auto-recalibration (Tier 3 #9). <https://gtsam.org/>

### 2. Global ILP track↔identity assignment — attacks P1 (flips, dup-tid, nondeterminism)
> **Status 2026-07-20:** the exact-assignment final arbiter shipped as
> `global_identity_resolve` (merge 0df6cfed). qm11+qm8 A/B: geometry-neutral;
> debut exposed a merge collision (kept greedy label duplicating an assigned
> team — fixed with the collision sweep + tests); qm8 produced a confirmed true
> positive greedy missed. Default OFF pending the full-match identity protocol
> (docs/default-on-ab-2026-07-20.md).

Replace greedy lock/flip as the *final arbiter* (keep voting as the per-read
aggregator): build `C[tracklet, team] = −log(reader posterior) + λ·gallery affinity +
zone/seed priors`, solve one whole-match assignment with hard constraints —
time-overlapping same-alliance tracklets take distinct teams; each team covers ≤1
tracklet per instant. With 6 identities × dozens of tracklets this is a
millisecond-scale CP-SAT/PuLP problem, zero VRAM.
- This is what the SoccerNet Game State Reconstruction winners converged on:
  2025 winner splits tracklets on identity inconsistency then merges under
  spatio-temporal + identity constraints (<https://arxiv.org/html/2508.19182v1>);
  2024 winner used hierarchical merging with hard vetoes (temporal overlap,
  conflicting jersey/team) (<https://arxiv.org/abs/2504.06357>).
- Directly addresses the qm11 backfill A/B's Hungarian-pairing-flip nondeterminism:
  a jointly-optimal, collision-free whole-match assignment is stable under small
  perturbations in a way greedy voting isn't.

### 3. Specialist digit recognizer + learned legibility gate + candidate-constrained scoring — attacks P1 ("1/5 fonts unreadable")
The strongest identity finding: small fine-tuned recognizers beat small VLMs on
jersey-style numbers. Roboflow (Sept 2025): fine-tuned ResNet-32 **93%** vs fine-tuned
SmolVLM2 86% on NBA jerseys (<https://blog.roboflow.com/identify-basketball-players/>).
Koshkina & Elder (CVPRW 2024): legibility classifier → pose-guided crop → fine-tuned
**PARSeq** → tracklet aggregation, 87–91% (<https://arxiv.org/abs/2405.13896>,
code <https://github.com/mkoshkina/jersey-number-pipeline>, CC BY-NC).
- Recognizer: **PARSeq-tiny** (<https://github.com/baudm/parseq>, Apache) or
  **SVTRv2 via OpenOCR** (ICCV 2025 — CTC beats encoder-decoder on accuracy *and*
  speed; <https://github.com/Topdu/OpenOCR>, Apache). <0.5 GB, ~ms/crop batched —
  runs beside vLLM, no phase-exclusivity pressure.
- **Candidate-constrained scoring is the key trick**: don't decode open-vocabulary —
  score each of the 6 known team numbers under the recognizer logits and emit
  argmax + margin. Converts "unreadable font" into 6-way discrimination with
  calibrated confidence for the voting layer. (Minimal variant: per-match 7-way
  classifier head — 6 teams + illegible.)
- Training data: **SynthTIGER** (<https://github.com/clovaai/synthtiger>) or
  **UnionST** (CVPR 2026, <https://github.com/YesianRohn/UnionST>) renders of 4-digit
  numbers in many fonts on red/blue wrinkled fabric + broadcast degradation
  (downscale to 20–60 px, JPEG, motion blur) — plus the free self-labeled corpus the
  pipeline already produces (vote-locked, human-verified crops).
- Replace the `legibility.py` heuristic with a small learned legibility classifier
  (Koshkina-style ResNet34) trained on crops the reader historically succeeded/failed on —
  better read-skipping = fewer wasted VLM calls *and* cleaner votes.
- Wire it as a second voting arm next to the VLM (like EasyOCR today but with real weight).

### 4. Zero/near-zero-cost version moves
- **SAM 3.1 "Object Multiplex"** (Meta, 2026-03-27 per research pass): joint propagation
  of up to 16 objects in one forward pass — ~2× throughput at medium object counts and
  roughly half the propagation VRAM, drop-in checkpoint, no API break. The repo pin
  (2026-05-23) already contains 3.1-era code — **verify the multiplex checkpoint is the
  one being loaded and that all ~6 robots propagate jointly rather than per-object.**
  Attacks P2 for ~free. <https://github.com/facebookresearch/sam3>
  > **Status 2026-07-20:** built (`sam3_multiplex`, merge 2ffa1124) but "no API
  > break" was false — upstream start_session forwards an offload kwarg the
  > multiplex model's init_state rejects; every session crashed to the no-SAM3
  > fallback until the sam3_tracker shim (docs/SAM3.md, docs/default-on-ab-2026-07-20.md).
  > Default stays OFF pending post-shim quality parity on qm8–qm12.
- **ultralytics 8.4.21 → 8.4.101** (ReID-crop robustness fixes, big seg-memory reduction).
- Re-test `sam3_compile` after bumping the SAM3 pin: upstream added a
  production-faithful compile benchmark (`scripts/bench_compile.py`, 2026-07-15) with a
  warmup pattern that avoids serve-time recompiles — the earlier "no win" A/B may have
  been measuring recompile stalls.
- Try the existing **`mask_footprint_pos=True`** knob in an A/B — mask contact point vs
  box bottom-center is exactly the ground-contact bias the multi-view literature says
  moves association accuracy by points (AI City 2025 3rd place: +7% AssA from
  foot-point selection alone).

---

## Tier 2 — medium effort, big payoffs

### 5. Selective mask propagation (architecture inversion) — attacks P2 + P3 + P4 together
The standout tracking find: **run cheap tracking-by-detection every frame; dispatch SAM3
only into ambiguous windows** (assignment margin < threshold), seeding backward from a
confident frame and propagating through the ambiguity; accept SAM3's answer only on
confident contradiction. Reported: **87.2 HOTA SportsMOT (#1)**, +6–8 HOTA on DanceTrack
across three base trackers, **1.5–7.5× faster than uniform SAM propagation, 5.7 GB vs
11.3 GB peak VRAM**, training-free, both components black-box.
(arXiv 2606.13033, June 2026; MIT code
<https://github.com/holma91/selective-mask-propagation>;
writeup <https://www.holma.io/blog/posts/selective-mask-propagation>.)
- This pipeline already owns every ingredient (YOLO det, DINOv3 embeddings, SAM3,
  Kalman) — it's a re-plumbing, not new models. It's also the *strategic* answer to the
  propagation-cost ceiling: stop paying SAM3 on the ~90% of frames where 6 well-separated
  robots are trivially associated.
- Supporting evidence TBD+re-ID rules sports in 2025–26: SoccerNet GSR 2025 winner =
  YOLO-X + Deep-EIoU + OSNet; SoccerTrack 2025 winner = Deep-EIoU + global tracklet
  association. For quick A/B plumbing of modern TBD trackers: **BoxMOT**
  (<https://github.com/mikel-brostrom/boxmot>, active Jul 2026).
- Treat as a branch-level experiment measured on qm8–qm12 before any default change.

### 6. GTA-style global tracklet split + reconnect — attacks P1 + P3
Plug-and-play post-pass, tracker-agnostic: (1) split tracklets whose embedding stream
shows multiple identities (the hijack signature), (2) globally reconnect same-identity
fragments. SoccerNet HOTA 79.41→83.11; won again in 2026 (GTATrack, SoccerTrack 2025).
Reuses the DINOv3 crop embeddings already computed.
(<https://arxiv.org/abs/2411.08216>, code <https://github.com/sjc042/gta-link>.)
- Natural companion to #2: GTA fixes the *tracklet* structure, the ILP then binds
  identities to the cleaned tracklets.

### 7. Reader ladder — attacks P1 (read accuracy)
1. **Qwen3-VL-8B-Instruct** — drop-in family upgrade from 4B (same template/guided
   JSON, Apache-2.0, mature vLLM); BF16 ≈ 17–18 GB fits the phase-exclusive reader
   budget. Qwen3-VL tops OCRBench v2 among open families.
2. **GLM-4.6V-Flash 9B** (Dec 2025, MIT) — top ≤9B on the 2026.06 OCRBench v2 refresh;
   caveat: thinking model — cap/disable reasoning tokens or hundreds of reads/match
   pay latency. <https://huggingface.co/zai-org/GLM-4.6V-Flash>
3. **HunyuanOCR-1.5 (1B)** (Jul 2026) — OCR-expert VLM that beats much larger models on
   scene/video text spotting at ~3–4 GB; candidate *second cheap reader arm* for
   ensemble/adjudication on disagreements. <https://arxiv.org/html/2607.04884v1>
4. **LoRA the winner on bumper crops** (Unsloth documents Qwen3-VL 4B/8B fine-tunes on
   24 GB) using the verified-crop corpus + synthetic renders from #3.
- Note the GSR-2025 lesson: the winning add-on was constraint-driven split/merge
  (#2/#6), *not* a bigger reader. Upgrade the reader, but expect the assignment-layer
  work to matter more.

### 8. Same-alliance re-ID program — attacks P1 (the DINOv3 gap)
"Two robots of one alliance" ≈ "same species, different individual" (wildlife re-ID) ≈
"same-color vehicle" (vehicle re-ID). Global CLS embeddings saturate on alliance color;
discrimination lives in parts.
- **Per-event self-supervised fine-tune** (the most credible fix): overnight
  LoRA/adapter on the embedder with InfoNCE — positives = temporally-distant frames of
  the same *vote-locked* tracklet; **hard negatives are free: tracks co-occurring in
  the same frame are guaranteed different robots**. Only use vote-locked segments
  (hijack-polluted pseudo-labels are the qm10/qm12 failure mode). Cluster-contrast
  lineage; a few hundred lines with pytorch-metric-learning.
- **Part/patch pooling**: pool DINOv3 patch tokens over the SAM mask (or concat part
  crops: drivetrain / superstructure / bumper corners) instead of CLS-only — the CLS
  vs avg-pool eval already proved pooling choice moves relink AUC.
- **Benchmark alternatives** on the existing crop re-ID eval: **MiewID-msv3**
  (<https://huggingface.co/conservationxlabs/miewid-msv3>, beats MegaDescriptor by
  +19% top-1 on unseen categories — i.e., transfers better to non-animals) and
  **SigLIP2** as a language-aligned complement.
- **LightGlue local-feature matching** as a training-free same-alliance tiebreaker:
  robots are rigid with repeatable structure; match-count between a query crop and each
  same-alliance robot's verified gallery is cheap and won AnimalCLEF-style open-set
  re-ID when fused with deep descriptors.

---

## Tier 3 — strategic bets (1–2+ weeks each)

### 9. Automatic field registration + drift auto-refresh — attacks P6 (and enables per-segment recalibration)
The production pattern across sports analytics 2024–26: train a keypoint/line detector
for the known field model, resect per frame/segment.
- **PnLCalib** (keypoints init + line refinement + lens distortion; CVIU 2026;
  0.42 m median on a 105 m pitch, 164–439 ms/frame on a 2080 Ti; GPL-2.0)
  <https://github.com/mguti97/PnLCalib>; predecessor **No Bells Just Whistles**
  (CVPRW 2024) <https://github.com/mguti97/No-Bells-Just-Whistles>. An FRC field
  (16.5×8.1 m, denser landmarks, closer cameras) should land ~5–15 cm — comparable to
  careful manual clicking, refreshable continuously.
- **Synthetic training from the CAD you already ship** (`data/field/`): render a few
  thousand domain-randomized views per season, fine-tune the keypoint/line head. Proven
  at scale by SoccerNet SynLoc 2026 (synthetic-trained winner, 97.67 mAP-LocSim,
  <https://github.com/Spiideo/sskit>).
- **BroadTrack** (WACV 2025) tripod model for the drift case specifically: solve camera
  *center* once per event, continuously refresh rotation/zoom — halves reprojection
  error vs per-frame calibration. <https://arxiv.org/abs/2412.01721>
- Integration path already exists: the staged pose autofix
  (`/api/calibration/autofix/approve`) becomes "auto-refresh with human notify" instead
  of "abort and wait." Minimal-sophistication variant: Roboflow-style YOLO-pose on ~32
  field keypoints + RANSAC resection
  (<https://blog.roboflow.com/camera-calibration-sports-computer-vision/>).

### 10. YOLOE-26 negative-class veto + visual-prompted detection — attacks P3, P4-at-unseen-events
YOLOE integrated with YOLO26 (Feb 2026), in the ultralytics package already installed:
text prompts, visual-exemplar prompts, prompt-free mode; re-parameterizes to
zero-overhead closed-set. Use as (a) a **negative-class veto** — prompt "tablet",
"radio", "tripod", "laptop" and veto YOLO robot detections that match; (b)
visual-prompt robots from the first seed frame at unseen events.
<https://docs.ultralytics.com/models/yoloe>
- Temper expectations: SAM3's concept head already failed on FRC robots and the 4B-VLM
  census failed in clutter — open-vocab "robot" recall will likely disappoint; the
  *negative veto* is the realistic win, complementing (not replacing) the accepted
  hard-negative retrain.

### 11. Detector A/B at the next hard-negative retrain
When retraining for equipment hard-negatives anyway, A/B against `yolo26s` on the same
data: **DEIMv2-S/M** (DINOv3 backbone — synergy with the embedder; Apache;
<https://github.com/Intellindust-AI-Lab/DEIMv2>) and **RF-DETR-M** (ICLR 2026, Apache
for N–L sizes, "designed for fine-tuning"; <https://github.com/rf-detr>). Modern
Apache DETRs beat YOLO26 by ~1–3 AP at equal TensorRT latency and ViT-dense backbones
are better on small objects. Expect incremental, not transformative — the retrain data
matters more than the architecture.

### 12. Learned ground-contact keypoint — attacks P7 + projection bias
SoccerNet SynLoc pattern: regress the object's *ground-projected center* (not a visible
pixel) per detection, trained on synthetic renders composited onto real field frames —
robust to bumper occlusion and airborne-over-bump moments. Shares the render pipeline
with #9. Try `mask_footprint_pos=True` first (free), then this if corner-cam geometry
is still the limiter. (Winner recipe: YOLO26-L + adapted RTMPose-X two-stage;
<https://arxiv.org/pdf/2607.07320>.)

### 13. TAPNext++ point-track verification — cheap hijack alarm
CVPR 2026, Apache, in google-deepmind/tapnet (pushed 2026-07-15): 256 points at
~50–150 FPS expected on a 3090 — near-free for 2.5-min matches. Seed points on mask
centroids; a mask whose point cloud walks off it is being hijacked; points also carry
identity through occlusions as an independent channel. <https://tap-next-plus-plus.github.io/>

### 14. Rerun instrumentation — debugging force multiplier
Rerun 0.32 (May 2026; MIT/Apache): synchronized multi-camera image views + top-down map
+ time-series on one scrubbable timeline, blueprint API from Python, H.264 frames.
Log per-view detections, projected rays, fused states, association decisions, and
calibration residuals per frame — replaces bespoke replay-log archaeology for fusion
debugging. 1–2 days. <https://rerun.io/> (Foxglove's open-source Studio was
discontinued at 2.0 — Rerun is the right horse.)

---

## Researched and rejected (don't spend time here)

| Technology | Why not |
|---|---|
| Learned BEV multi-view trackers (EarlyBird, TrackTacular) | Stale research code (last release Mar 2024), needs scene-specific BEV training data; calibrated-camera competitions in 2025–26 are still won by geometric fusion + global offline association at this scale (3 views, 6 objects) |
| Monocular metric depth as corner-cam range (UniDepthV2, Metric3Dv2, DA3-metric) | Best zero-shot ≈14–15% abs-rel ⇒ ±2–3.5 m at 15–25 m vs the ~0.3 m needed; only ordinal/relative use is plausible, low priority |
| Learned gap imputation (DiffMOT, SportMamba) | ~1 HOTA over smoothed interpolation on *dense* player datasets; RTS smoothing captures nearly all of it at N=6 |
| Grounding DINO 1.5/1.6, DINO-X, T-Rex2 | API-only, no local weights ("Grounding DINO 2.0" doesn't exist); local alternatives: MM-Grounding-DINO, LLMDet if ever needed |
| Bigger/thinking VLM as the *primary* fix | GSR 2025 evidence: the win came from constraint-driven split/merge, not reader size; also Ampere has no FP8, 12B+ needs quantization |
| Diffusion text SR as a default pre-read step | Hallucination risk — a text-prior SR can render a plausible *wrong* digit that 3-agree voting then locks; only as a gated assist on low-confidence crops (textbsr → OSEDiff one-step ~100 ms), SR reads must never outvote raw reads |
| SAMURAI, SAM2Long, Cutie, EfficientTAM | Stale/superseded, or quality cliff on fast/distractor scenes; DAM4SAM's distractor-aware memory is worth *porting as an idea* to SAM3 but is SAM2.1-only code with no license |
| CoTracker3 | CC-BY-NC + ~5× slower than TAPNext++ |
| MegaDescriptor | CC BY-NC and beaten by MiewID-msv3 on unseen categories |
| Moondream 3, document-OCR minis (GLM-OCR, PaddleOCR-VL, DeepSeek-OCR 2) | License/serving friction or document-parsing-oriented — wrong tool for 20–60 px scene text |
| Full continuous-time GP/spline trajectory estimation | Elegant, but discrete RTS at 30 fps captures the value at a fraction of the complexity |

## Suggested sequencing

1. **This week, no new models**: RTS smoother (#1) · ILP assignment (#2) ·
   SAM3.1-multiplex verify + ultralytics bump + `mask_footprint_pos` A/B (#4).
   All three are measurable on the existing qm8–qm12 + Manta harness.
2. **Next**: digit recognizer + learned legibility (#3) and GTA post-pass (#6) —
   both feed the ILP with better inputs. Reader ladder A/B (#7) in parallel
   (it's config-level).
3. **One render-pipeline investment, three payoffs**: CAD-rendered synthetic data
   feeds auto-calibration (#9), the digit recognizer's font robustness (#3), and the
   ground-contact keypoint (#12).
4. **Strategic branch**: selective mask propagation (#5) — the propagation-cost and
   hijack ceiling-breaker; evaluate like the backfill A/B was run.
5. **Ongoing**: same-alliance re-ID program (#8), TAPNext++ alarm (#13), Rerun (#14).

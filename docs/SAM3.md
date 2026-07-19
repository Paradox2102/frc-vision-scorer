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

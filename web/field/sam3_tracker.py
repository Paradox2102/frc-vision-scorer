"""SAM 3 adapter for the existing offline field-tracker pipeline.

Targets Meta's full SAM3 video model (detector + tracker,
``Sam3VideoInferenceWithInstanceInteractivity``) through its session/request
API, while keeping the lower-level SAM2 video-predictor protocol the
application is built around, so re-prompting, culling, fusion, and identity
logic do not need separate SAM3 implementations.

Protocol mapping (validated against sam3 @ pinned commit, June 2026):

- Box prompts ride the *instance* path, not the semantic path. The semantic
  ``add_prompt(boxes_xywh=...)`` resets all tracking state on every call,
  which is incompatible with one-call-per-robot prompting and mid-stream
  re-prompts. Instead, each box is encoded as two points with labels 2
  (top-left) and 3 (bottom-right) — the same encoding SAM 2 was trained on —
  and sent with a stable caller ``obj_id``. That routes to
  ``add_tracker_new_points`` -> the SAM3 tracker's ``add_new_points_or_box``,
  which adds new objects or refines existing ones incrementally.
- Point coordinates are sent normalized (``rel_coordinates=True``); the model
  scales them by its square input size internally.

Chunked sessions (VRAM)
-----------------------

The SAM3 video model copies the whole session's frames to the GPU at 1008²
(``offload_video_to_cpu`` is ignored by its input-batch construction), so a
full match cannot be one session on a 24GB card. The adapter therefore opens
one session per window of ``SAM3_CHUNK_FRAMES`` frames (measured: ~14 MB
VRAM/frame + ~4 GB model; 360-frame chunk peaks at ~8.8 GB) and carries
objects across boundaries by re-seeding each live object's last observed
mask bbox under its SAME obj_id at the start of the next chunk.

What a boundary loses: the tracker's appearance memory, so an object whose
mask is empty at the boundary (occluded) is not re-seeded and will not be
re-found by SAM3 itself. The pipeline's existing detector re-prompts and
identity machinery (VLM votes + gallery) already handle exactly that case
for SAM2 re-seeds.

Windowed propagation (action history)
-------------------------------------

Unlike SAM2, a SAM3 session interprets each ``propagate_in_video`` request
through its action history (``parse_action_history_for_propagation``): with
no new prompts since the previous propagate it returns cached outputs only
(``propagation_fetch``), and after new prompts it tracks ONLY the newly
prompted obj_ids (``propagation_partial``). The pipeline propagates in
``reprompt_every`` windows, so without intervention every object emits masks
for exactly one window and then goes dark. The adapter therefore re-sends
each live object's latest observed box (fresh within SAM3_RESEED_MAX_AGE)
before every windowed propagate, keeping the full roster on the partial
path; per-object tracker states persist within the session, so this is a
re-anchor, not a cold re-seed.

Late-seed backward backfill (solo_ids)
---------------------------------------

``mask_tracker.MaskFieldTracker.track`` (shared by this adapter) can, after
its forward loop finishes, re-prompt a late-seeded object at its own seed
frame and propagate BACKWARD to recover history YOLO missed. Doing that
naively would hit both roster-carry loops above (``_open_chunk_for`` and
``_stream_chunk``) and re-seed/re-prompt the WHOLE live roster across every
boundary the reverse pass crosses — wasted compute, and a regression of every
other object's ``last_box`` to the backfill's own reverse timeline. Setting
``state["solo_ids"]`` to the backfill's target obj_ids confines both loops to
just those ids for the duration of the reverse pass; the per-frame
``last_box`` update in ``_stream_chunk`` is NOT gated by ``solo_ids``, since
it's what lets a target keep propagating across a reverse chunk boundary.
``None`` (the default) leaves both loops covering the full roster, unchanged.
"""
from __future__ import annotations

import concurrent.futures
import os
import shutil
import tempfile
import threading
import time
from typing import Any

import numpy as np

from web.field.mask_tracker import MaskFieldTracker

# Frames per SAM3 session window. ~14 MB GPU/frame: 200 frames ≈ 2.8 GB video
# residency, leaving headroom for the model and a co-resident vLLM reader.
# Lowered from 400 after a 24GB-card whole-card OOM with co-resident state.
SAM3_CHUNK_FRAMES = int(os.environ.get("SAM3_CHUNK_FRAMES", "200"))
# Only re-seed an object across a boundary if its mask was non-empty within
# this many recent frames (stale boxes hijack background clutter).
SAM3_RESEED_MAX_AGE = int(os.environ.get("SAM3_RESEED_MAX_AGE", "24"))
# SAM 3.1 Object Multiplex builder opt-in (see job_config.sam3_multiplex for
# the full rationale/risk note). Module env default stays off for direct/
# library callers; the pipeline passes the Knob value into the constructor.
SAM3_MULTIPLEX = os.environ.get("SAM3_MULTIPLEX", "0") not in ("0", "false", "False")
# Per-session object capacity for the SAM 3.1 multiplex model (the builder
# default is 16). Every (re)seed mints a new obj_id and a chunk session never
# removes objects, so a session's lifetime id count — not the ~6-robot roster —
# is what hits the cap; at 16 a chunk filled within ~16 frames and every later
# add_prompt was refused.
SAM3_MULTIPLEX_MAX_OBJECTS = 64


def _shim_init_state_drop_state_offload(model: Any) -> None:
    """Sam3BasePredictor.start_session unconditionally forwards
    offload_state_to_cpu into model.init_state(**kwargs), but the 3.1
    multiplex model's init_state doesn't accept it — every session raised
    TypeError and all views fell to the no-SAM3 fallback (qm11 2026-07-20
    A/B). Shim the built model's init_state to accept-and-drop the kwarg,
    signature-guarded so a fixed upstream disables it."""
    import functools
    import inspect

    init_state = getattr(model, "init_state", None)
    if init_state is None:
        return
    try:
        params = inspect.signature(init_state).parameters
    except (TypeError, ValueError):
        params = {}
    accepts = "offload_state_to_cpu" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts:
        return

    @functools.wraps(init_state)
    def _drop_state_offload(*a, offload_state_to_cpu=False,
                            _orig=init_state, **kw):
        return _orig(*a, **kw)

    model.init_state = _drop_state_offload


def _shim_fetch_tolerate_uncached_frame(model: Any) -> None:
    """When no add/refine was accepted since the previous propagate, the
    multiplex model routes the request to ``propagation_fetch``, whose
    ``fetch_and_process_single_frame_results`` indexes
    ``cached_frame_outputs[frame_idx]`` unguarded — any frame never
    propagated raises a bare ``KeyError: <frame_idx>``. Base SAM3 does
    ``cached_frame_outputs.get(frame_idx, {})`` and yields an empty frame;
    mirror that by inserting an empty placeholder for the duration of the
    call only (no cached state is left behind)."""
    import functools

    fetch = getattr(model, "fetch_and_process_single_frame_results", None)
    if fetch is None or getattr(fetch, "_frc_uncached_frame_shim", False):
        return

    @functools.wraps(fetch)
    def _fetch_tolerating_uncached(inference_state, frame_idx, *a,
                                   _orig=fetch, **kw):
        cache = inference_state.get("cached_frame_outputs") \
            if isinstance(inference_state, dict) else None
        if not isinstance(cache, dict) or frame_idx in cache:
            return _orig(inference_state, frame_idx, *a, **kw)
        placeholder: dict = {}
        cache[frame_idx] = placeholder
        try:
            return _orig(inference_state, frame_idx, *a, **kw)
        finally:
            if cache.get(frame_idx) is placeholder:
                del cache[frame_idx]

    _fetch_tolerating_uncached._frc_uncached_frame_shim = True
    model.fetch_and_process_single_frame_results = _fetch_tolerating_uncached


def _shim_build_output_keep_refined(model: Any) -> None:
    """The multiplex ``_build_sam2_output(state, frame_idx, refined)`` opens
    with ``if frame_idx not in cached_frame_outputs: return {}`` — it DROPS
    the freshly propagated (refined) masks on any frame without a cached
    entry. Meta's demo flow always runs a text-prompt ``propagation_full``
    first, which caches every frame; pure instance-prompt tracking never
    does, so every propagated frame came back empty and masks survived only
    on prompt frames (2026-11-06 offline A/B: SAM2 produced masks for all
    objects, output was ``{}``). Base SAM3 starts from
    ``cached.get(frame_idx, {})`` and overlays the refined masks; mirror that
    with a call-scoped empty placeholder so the upstream overlay (and its
    non-None assert) still runs and no cached state is left behind."""
    import functools

    build = getattr(model, "_build_sam2_output", None)
    if build is None or getattr(build, "_frc_keep_refined_shim", False):
        return

    @functools.wraps(build)
    def _build_keeping_refined(inference_state, frame_idx, *a, _orig=build, **kw):
        cache = inference_state.get("cached_frame_outputs") \
            if isinstance(inference_state, dict) else None
        if not isinstance(cache, dict) or frame_idx in cache:
            return _orig(inference_state, frame_idx, *a, **kw)
        placeholder: dict = {}
        cache[frame_idx] = placeholder
        try:
            return _orig(inference_state, frame_idx, *a, **kw)
        finally:
            if cache.get(frame_idx) is placeholder:
                del cache[frame_idx]

    _build_keeping_refined._frc_keep_refined_shim = True
    model._build_sam2_output = _build_keeping_refined


# Conditioning frames (prompt/refine frames) whose heavy memory tensors the
# multiplex memory bound keeps per object state: the first (initial prompt)
# plus this many temporally closest to the frame being tracked.
SAM3_MULTIPLEX_COND_MEM_FRAMES = 6
# Heavy per-frame tensors dropped from out-of-window frame outputs; the small
# ones (pred_masks, object_score_logits, obj_ptr, ...) stay so every upstream
# lookup still finds its frame entry.
_MUX_HEAVY_KEYS = ("maskmem_features", "maskmem_pos_enc", "image_features",
                   "image_pos_enc", "pred_masks_high_res")


def _trim_cond_memory(output_dict: dict, frame_idx: int, keep: int) -> int:
    """Drop heavy tensors from conditioning-frame outputs outside {first} ∪
    {``keep`` temporally closest to ``frame_idx``}. Entries are replaced by
    trimmed copies, never deleted (preflight asserts consolidated cond frame
    indices exist in output_dict; the memory gather skips an entry whose
    ``maskmem_features`` is None). Returns how many entries were trimmed."""
    cond = output_dict.get("cond_frame_outputs") if isinstance(output_dict, dict) else None
    if not isinstance(cond, dict) or len(cond) <= keep + 1:
        return 0
    keys = sorted(cond)
    keepers = {keys[0]} | set(sorted(keys, key=lambda t: abs(t - frame_idx))[:keep])
    n = 0
    for t in keys:
        out = cond[t]
        if t in keepers or not isinstance(out, dict):
            continue
        if any(out.get(k) is not None for k in _MUX_HEAVY_KEYS):
            cond[t] = {k: v for k, v in out.items() if k not in _MUX_HEAVY_KEYS}
            n += 1
    return n


def _trim_non_cond_memory(output_dict: dict, frame_idx: int, window: int) -> int:
    """Drop heavy tensors from non-conditioning frame outputs more than
    ``window`` frames from ``frame_idx`` (entries kept, trimmed in place of
    deletion, same contract as _trim_cond_memory). Returns entries trimmed."""
    nc = output_dict.get("non_cond_frame_outputs") if isinstance(output_dict, dict) else None
    if not isinstance(nc, dict):
        return 0
    n = 0
    for t in list(nc):
        out = nc[t]
        if abs(t - frame_idx) <= window or not isinstance(out, dict):
            continue
        if any(out.get(k) is not None for k in _MUX_HEAVY_KEYS):
            nc[t] = {k: v for k, v in out.items() if k not in _MUX_HEAVY_KEYS}
            n += 1
    return n


def _shim_bound_tracker_memory(model: Any) -> None:
    """Instance-prompted objects each get a singleton SAM2 state whose
    per-frame memory is laid out in multiplex ("mux") space -- ~19 MB of
    maskmem per object per frame, ~30x SAM2's -- and the multiplex tracker
    keeps EVERY propagated (non-cond) and prompted (cond) frame's memory (its
    ``trim_past_non_cond_mem_for_eval`` path is never reached by the
    interactive per-frame step). Offline, VRAM grew ~0.27 GB/frame with 3
    objects and the qm11 auto-only job OOM'd every view ~20 frames in
    (2026-11-06). After each propagated frame, trim heavy tensors outside
    what memory attention can read:
    - non-cond: frames > r * num_maskmem away (attention reads the previous
      num_maskmem - 1 frames when use_memory_selection is off);
    - cond: all but the first and the ``SAM3_MULTIPLEX_COND_MEM_FRAMES``
      closest (attention reads max_cond_frames_in_attn=4 closest, plus any
      unselected cond frame inside the num_maskmem window)."""
    import functools

    step = getattr(model, "_propogate_tracker_one_frame_local_gpu", None)
    if step is None or getattr(step, "_frc_bound_memory_shim", False):
        return
    tracker = getattr(model, "tracker", None)
    if getattr(tracker, "use_memory_selection", False):
        return  # memory selection may read far frames; don't trim under it
    window = int(getattr(tracker, "memory_temporal_stride_for_eval", 1) or 1) * \
        int(getattr(tracker, "num_maskmem", 7) or 7)

    @functools.wraps(step)
    def _step_bounding_memory(inference_states, frame_idx, *a, _orig=step, **kw):
        result = _orig(inference_states, frame_idx, *a, **kw)
        for s in inference_states or ():
            if isinstance(s, dict):
                od = s.get("output_dict")
                _trim_non_cond_memory(od, int(frame_idx), window)
                _trim_cond_memory(od, int(frame_idx),
                                  SAM3_MULTIPLEX_COND_MEM_FRAMES)
        return result

    _step_bounding_memory._frc_bound_memory_shim = True
    model._propogate_tracker_one_frame_local_gpu = _step_bounding_memory


def _install_multiplex_shims(model: Any) -> None:
    """Compat shims for the SAM 3.1 multiplex model (no-op without a model)."""
    if model is None:
        return
    _shim_init_state_drop_state_offload(model)
    _shim_fetch_tolerate_uncached_frame(model)
    _shim_build_output_keep_refined(model)
    _shim_bound_tracker_memory(model)


def _is_multiplex_model(model: Any) -> bool:
    """True for SAM 3.1 multiplex model classes (their
    ``_prepare_backbone_feats`` has the same signature but different state
    layout, so the base-SAM3 skip-detector replacement must never land there)."""
    return any("multiplex" in cls.__name__.lower()
               for cls in type(model).__mro__)


def install_skip_detector_heads(model: Any) -> bool:
    """Replace base SAM3's ``_prepare_backbone_feats`` with a backbone-only
    version (the ``sam3_skip_detector_heads`` knob).

    On the instance-prompt (``propagation_partial``) path, and for every
    instance add/refine, upstream ``_prepare_backbone_feats`` calls
    ``run_backbone_and_detection`` and DISCARDS the detections: only
    ``feature_cache[frame_idx]`` (tracker backbone features) is consumed. The
    DETR encoder/decoder/seg heads/NMS cost ~33-38 ms/frame. This computes just
    ``detector.backbone.forward_image`` + the tracker neck convs, matching the
    upstream cache entry exactly (bf16 cast of the 3 FPN levels as upstream's
    all-gather does, ``vision_pos_enc`` passed through as the list, previous
    frame popped). Offline 2026-11-06: byte-identical masks, -36 ms/frame.

    ``propagation_full`` (text/VG propagation) calls
    ``run_backbone_and_detection`` directly and is untouched.

    Guarded like the multiplex shims: returns False (and logs) without
    touching the model unless it has the expected shape; idempotent.
    """
    import functools
    import inspect

    if model is None:
        return False
    orig = getattr(model, "_prepare_backbone_feats", None)
    if orig is None:
        print("  [SAM3] skip_detector_heads: model has no _prepare_backbone_feats "
              "— not installed")
        return False
    if getattr(orig, "_frc_skip_detector_shim", False):
        return True
    if _is_multiplex_model(model):
        print("  [SAM3] skip_detector_heads: multiplex model — not installed")
        return False
    detector = getattr(model, "detector", None)
    backbone = getattr(detector, "backbone", None)
    decoder = getattr(getattr(model, "tracker", None), "sam_mask_decoder", None)
    if not callable(getattr(backbone, "forward_image", None)) or decoder is None \
            or not callable(getattr(decoder, "conv_s0", None)) \
            or not callable(getattr(decoder, "conv_s1", None)):
        print("  [SAM3] skip_detector_heads: missing detector.backbone.forward_image "
              "or tracker.sam_mask_decoder.conv_s0/conv_s1 — not installed")
        return False
    try:
        params = list(inspect.signature(orig).parameters)
    except (TypeError, ValueError):
        params = []
    if params != ["inference_state", "frame_idx", "reverse"]:
        print(f"  [SAM3] skip_detector_heads: unexpected _prepare_backbone_feats "
              f"signature {params} — not installed")
        return False

    import torch

    fallback_logged = [False]

    @functools.wraps(orig)
    def _prepare_backbone_feats(inference_state, frame_idx, reverse, _orig=orig):
        input_batch = inference_state["input_batch"]
        feature_cache = inference_state["feature_cache"]
        if frame_idx in feature_cache:
            return
        img = input_batch.img_batch[frame_idx]
        image = img.unsqueeze(0).to(dtype=torch.float32, device=model.device)
        bb = model.detector.backbone.forward_image(image)
        sam2_out = bb.get("sam2_backbone_out") if isinstance(bb, dict) else None
        if not isinstance(sam2_out, dict) or "backbone_fpn" not in sam2_out \
                or "vision_pos_enc" not in sam2_out:
            # Unknown backbone layout: never break tracking, run upstream.
            if not fallback_logged[0]:
                fallback_logged[0] = True
                keys = list(bb) if isinstance(bb, dict) else type(bb).__name__
                print(f"  [SAM3] skip_detector_heads: unexpected backbone output "
                      f"{keys} — falling back to the full detector path")
            return _orig(inference_state, frame_idx, reverse)
        # Upstream casts these to bf16 for its all-gather; keep parity.
        fpn = [x.to(torch.bfloat16) for x in sam2_out["backbone_fpn"]]
        sam_mask_decoder = model.tracker.sam_mask_decoder
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(fpn[0]),
            sam_mask_decoder.conv_s1(fpn[1]),
            fpn[2],  # fpn_2 doesn't need conv (upstream)
        ]
        feature_cache[frame_idx] = (
            input_batch.img_batch[frame_idx],
            {"tracker_backbone_out": {
                "vision_features": tracker_backbone_fpn[-1],
                "vision_pos_enc": sam2_out["vision_pos_enc"],
                "backbone_fpn": tracker_backbone_fpn,
            }},
        )
        # remove old features to save GPU memory (upstream)
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)

    _prepare_backbone_feats._frc_skip_detector_shim = True
    model._prepare_backbone_feats = _prepare_backbone_feats
    return True


# ── sam3_frame_load_workers: threaded chunk-session frame loader ──────────
#
# Every chunk session open (start_session -> io_utils.load_resource_as_video_
# frames -> load_video_frames -> load_video_frames_from_image_folder, all
# resolved through io_utils module globals at call time) decodes + resizes the
# chunk's JPEGs one at a time with PIL. PIL decode/resize release the GIL, so
# fanning upstream's own per-frame ``_load_img_as_tensor`` over a thread pool
# is byte-identical and much faster on many-core hosts (Nov 12 RunPod
# microbench: session open 8.3 s -> 1.2 s at 16 threads).

_FRAME_LOAD_POOL: Any = None
_FRAME_LOAD_POOL_WORKERS = 0
_FRAME_LOAD_POOL_LOCK = threading.Lock()
_ORIG_LOAD_FROM_IMAGE_FOLDER: Any = None


def _frame_load_pool(workers: int) -> Any:
    """Module-shared executor, created lazily and reused across sessions;
    recreated only when a larger worker count is requested."""
    global _FRAME_LOAD_POOL, _FRAME_LOAD_POOL_WORKERS
    with _FRAME_LOAD_POOL_LOCK:
        if _FRAME_LOAD_POOL is None or workers > _FRAME_LOAD_POOL_WORKERS:
            old = _FRAME_LOAD_POOL
            _FRAME_LOAD_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="sam3-frame-load")
            _FRAME_LOAD_POOL_WORKERS = workers
            if old is not None:
                old.shutdown(wait=False)
        return _FRAME_LOAD_POOL


def _make_threaded_loader(orig: Any, workers: int) -> Any:
    from sam3.model import io_utils

    def load_video_frames_from_image_folder(
        image_folder,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        async_loading_frames,
    ):
        # Lazy (async) loader: upstream path unchanged.
        if async_loading_frames:
            return orig(
                image_folder=image_folder,
                image_size=image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                img_mean=img_mean,
                img_std=img_std,
                async_loading_frames=async_loading_frames,
            )
        import torch

        # Below mirrors upstream exactly except the per-frame map.
        frame_names = [
            p
            for p in os.listdir(image_folder)
            if os.path.splitext(p)[-1].lower() in io_utils.IMAGE_EXTS
        ]
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            io_utils.logger.warning(
                f'frame names are not in "<frame_index>.<img_ext>" format: '
                f"{frame_names[:5]=}, falling back to lexicographic sort."
            )
            frame_names.sort()
        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {image_folder}")
        img_paths = [os.path.join(image_folder, n) for n in frame_names]
        img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

        images = torch.zeros(num_frames, 3, image_size, image_size,
                             dtype=torch.float16)
        video_height, video_width = None, None
        pool = _frame_load_pool(workers)
        load_one = io_utils._load_img_as_tensor
        for n, (img, video_height, video_width) in enumerate(
            pool.map(lambda p: load_one(p, image_size), img_paths)
        ):
            images[n] = img
        if not offload_video_to_cpu:
            images = images.cuda()
            img_mean = img_mean.cuda()
            img_std = img_std.cuda()
        images -= img_mean
        images /= img_std
        return images, video_height, video_width

    load_video_frames_from_image_folder._frc_threaded_loader = True
    load_video_frames_from_image_folder._frc_workers = workers
    return load_video_frames_from_image_folder


def install_threaded_frame_loader(workers: int) -> bool:
    """Monkeypatch ``sam3.model.io_utils.load_video_frames_from_image_folder``
    with a byte-identical thread-pooled version (the
    ``sam3_frame_load_workers`` knob). Idempotent; a later call with more
    workers re-installs with the larger pool. ``workers <= 1`` = no install.
    Returns True when the threaded loader is in place."""
    global _ORIG_LOAD_FROM_IMAGE_FOLDER
    workers = int(workers or 0)
    if workers <= 1:
        return False
    try:
        from sam3.model import io_utils
    except Exception as exc:
        print(f"  [SAM3] threaded frame loader: sam3 not importable ({exc!r}) "
              "— not installed")
        return False
    current = io_utils.load_video_frames_from_image_folder
    if getattr(current, "_frc_threaded_loader", False):
        if int(getattr(current, "_frc_workers", 0)) >= workers:
            return True
    else:
        _ORIG_LOAD_FROM_IMAGE_FOLDER = current
    io_utils.load_video_frames_from_image_folder = _make_threaded_loader(
        _ORIG_LOAD_FROM_IMAGE_FOLDER, workers)
    return True


def _uninstall_threaded_frame_loader() -> None:
    """Test helper: restore upstream's loader and drop the shared pool."""
    global _ORIG_LOAD_FROM_IMAGE_FOLDER, _FRAME_LOAD_POOL, _FRAME_LOAD_POOL_WORKERS
    try:
        from sam3.model import io_utils
    except Exception:
        return
    if _ORIG_LOAD_FROM_IMAGE_FOLDER is not None:
        io_utils.load_video_frames_from_image_folder = _ORIG_LOAD_FROM_IMAGE_FOLDER
        _ORIG_LOAD_FROM_IMAGE_FOLDER = None
    with _FRAME_LOAD_POOL_LOCK:
        if _FRAME_LOAD_POOL is not None:
            _FRAME_LOAD_POOL.shutdown(wait=True)
        _FRAME_LOAD_POOL = None
        _FRAME_LOAD_POOL_WORKERS = 0


def _list_frame_paths(frames_dir: str) -> list[str]:
    names = [n for n in sorted(os.listdir(frames_dir))
             if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not names:
        raise RuntimeError(f"no frames in {frames_dir}")
    return [os.path.join(frames_dir, n) for n in names]


def _probe_image_size(path: str) -> tuple[int, int]:
    import cv2

    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"unreadable frame {path}")
    h, w = frame.shape[:2]
    return w, h


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """xyxy bbox of a binary mask, or None when empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


class Sam3PredictorAdapter:
    """Expose a SAM 3 session predictor through the SAM2 predictor protocol,
    windowing the video into VRAM-sized chunk sessions internally."""

    def __init__(self, predictor: Any, chunk_frames: int = SAM3_CHUNK_FRAMES) -> None:
        self._predictor = predictor
        self._chunk_frames = max(int(chunk_frames), 16)
        # remove_object: send the session-level ``remove_object`` request
        # (True) or do adapter bookkeeping only (False). Sam3FieldTracker sets
        # False for the SAM 3.1 multiplex predictor, whose
        # remove_objects._slice_state can KeyError on the memory the
        # _shim_bound_tracker_memory trim leaves behind.
        self.session_remove = True
        # state["resume_refine"] (selective_retire_closed / selective_lazy_seed
        # segment resumes): allowed to keep already-tracked ids on
        # propagation_partial with a bare ``refine`` action instead of a new
        # box prompt. Sam3FieldTracker sets False for multiplex (its objects
        # share bucketed states; the bare-refine path is unverified there).
        self.refine_resume = True
        # Session-open wall time (start_session = frame load + init_state),
        # accumulated across chunks; shown as ``open=`` on chunk-closed lines.
        self.session_open_s_total = 0.0
        self.session_open_count = 0

    # ── SAM2-protocol surface ────────────────────────────────────────

    def init_state(
        self,
        *,
        video_path: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ) -> dict:
        if not os.path.isdir(video_path):
            raise RuntimeError(
                "Sam3PredictorAdapter requires a frames directory (the "
                "pipeline's export dir); got a file: " + video_path
            )
        frame_paths = _list_frame_paths(video_path)
        width, height = _probe_image_size(frame_paths[0])
        return {
            "frame_paths": frame_paths,
            "video_size": (width, height),
            "offload": (offload_video_to_cpu, offload_state_to_cpu),
            # live chunk session: {"session_id", "lo", "hi", "dir"} or None
            "chunk": None,
            # obj_id -> (global_frame_idx, xyxy box) most recent evidence;
            # prompts and observed masks both update it
            "last_box": {},
            # prompts not yet sent to any session:
            # [(global_idx, obj_id, box, neg_points)]
            "pending": [],
            "closed": False,
            # Late-seed backward backfill scoping (mask_tracker.py): when set
            # to a set of obj_ids, the roster-carry loops below re-prompt
            # ONLY those ids instead of the whole live roster. None (default)
            # is the normal full-roster behavior.
            "solo_ids": None,
        }

    def add_new_points_or_box(
        self,
        state: dict,
        *,
        frame_idx: int,
        obj_id: int,
        box,
        points=None,
        labels=None,
    ) -> None:
        box = tuple(float(v) for v in box)
        # Only label-0 (negative) extras are meaningful here; positive points
        # would fight the box corners the adapter already sends as labels 2,3.
        # points/labels arrive as ndarrays from _prompt_with_negatives (the
        # real SAM2 predictor takes them natively); `x or []` on a
        # multi-element ndarray raises "truth value ... is ambiguous", which
        # killed a whole view's tracking once neg_point_prompts defaulted on —
        # so gate on None explicitly.
        pts = [] if points is None else points
        lbls = [] if labels is None else labels
        neg = [tuple(float(v) for v in p)
               for p, l in zip(pts, lbls) if int(l) == 0]
        state["last_box"][int(obj_id)] = (int(frame_idx), box)
        chunk = state["chunk"]
        if chunk is not None and chunk["lo"] <= frame_idx < chunk["hi"]:
            self._send_prompt(state, frame_idx, obj_id, box, neg=neg)
        else:
            state["pending"].append((int(frame_idx), int(obj_id), box, neg))

    def propagate_in_video(
        self,
        state: dict,
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        num_frames = len(state["frame_paths"])
        start = int(start_frame_idx) if start_frame_idx is not None else (
            num_frames - 1 if reverse else 0
        )
        span = int(max_frame_num_to_track) if max_frame_num_to_track is not None \
            else num_frames
        if reverse:
            lo, hi = max(0, start - span), start + 1
        else:
            lo, hi = start, min(num_frames, start + span + 1)

        frames = range(hi - 1, lo - 1, -1) if reverse else range(lo, hi)
        cursor = None
        for global_idx in frames:
            chunk = state["chunk"]
            if chunk is None or not (chunk["lo"] <= global_idx < chunk["hi"]):
                self._open_chunk_for(state, global_idx, reverse=reverse)
                cursor = None
            if cursor is None or cursor != global_idx:
                # (re)start streaming within the current chunk from global_idx
                yield from self._stream_chunk(
                    state, global_idx, lo, hi, reverse=reverse
                )
                return  # _stream_chunk recurses across remaining chunks
        # zero-length range: nothing to do

    def remove_object(self, state: dict, obj_id: int) -> None:
        """Retire ``obj_id`` (a culled/evicted track) from SAM3 entirely.

        Without this a culled hijack keeps being re-anchored every window and
        re-seeded at every chunk boundary (its mask stays non-empty), so SAM3
        tracks it forever at ~26-42 ms/frame while the tracker just filters
        its outputs. Bookkeeping: drop it from ``last_box`` (no more
        re-anchor/re-seed), ``pending``, the live chunk's ``prompted`` set and
        ``solo_ids``. Session: send upstream ``remove_object`` when the id is
        live in the current chunk session (survivors' masks are unchanged —
        offline 2026-11-06, byte-identical).

        Upstream records the removal as a ``remove`` action, which
        ``parse_action_history_for_propagation`` treats as a no-op: the next
        propagate still runs ``propagation_partial`` over whatever was
        added/refined since the last propagate (the per-window re-anchor of
        survivors), or ``propagation_fetch`` when nothing was — the same
        outcome as today's all-stale window. A failed removal degrades to
        bookkeeping-only; it must never kill the view.
        """
        oid = int(obj_id)
        state["last_box"].pop(oid, None)
        state["pending"] = [p for p in state["pending"] if int(p[1]) != oid]
        solo = state.get("solo_ids")
        if isinstance(solo, set):
            solo.discard(oid)
        chunk = state.get("chunk")
        if not chunk:
            return
        chunk["prompted"].discard(oid)
        if not self.session_remove:
            return
        sid = chunk["session_id"]
        in_session = self._session_has_object(sid, oid)
        if in_session is False:
            return
        try:
            self._predictor.handle_request(
                {"type": "remove_object", "session_id": sid, "obj_id": oid}
            )
            chunk["retired"] = chunk.get("retired", 0) + 1
        except Exception as exc:
            print(f"  [SAM3] remove_object failed for obj {oid} in chunk "
                  f"[{chunk['lo']},{chunk['hi']}) — bookkeeping-only retire: "
                  f"{exc!r}")

    def _session_has_object(self, session_id: str, obj_id: int) -> bool | None:
        """Whether the chunk session tracks ``obj_id`` (upstream
        ``remove_object`` asserts it does). None when the predictor exposes
        no session introspection (stubs) — the caller then just tries."""
        ids = self._session_object_ids(session_id)
        if ids is None:
            return None
        return int(obj_id) in ids

    def _session_object_ids(self, session_id: str) -> set | None:
        """All obj_ids the chunk session currently tracks (empty before the
        first add). None when the predictor exposes no session introspection
        (stubs) or the lookup fails."""
        get_session = getattr(self._predictor, "_get_session", None)
        if get_session is None:
            return None
        try:
            sess_state = get_session(session_id)["state"]
            per_gpu = (sess_state.get("tracker_metadata") or {}).get(
                "obj_ids_per_gpu")
            if per_gpu is None:
                return set()  # metadata uninitialized: nothing added yet
            return {int(x) for ids in per_gpu
                    for x in np.asarray(ids).reshape(-1)}
        except Exception:
            return None

    def _resume_refiner(self, chunk: dict):
        """For a ``resume_refine`` propagate: ``(session_ids, carry, refine)``
        or None when the bare-refine path is unavailable (multiplex, or a
        predictor/model without ``_get_session`` / ``add_action_history``) —
        the caller then re-anchors with real prompts as usual.

        ``carry`` is the obj_id list of the session's last propagation action
        (what the interrupted request was propagating); ``refine(oid, local)``
        records a bare upstream ``refine`` action, which
        ``parse_action_history_for_propagation`` folds into the next
        ``propagation_partial`` WITHOUT ``add_tracker_new_points`` — no new
        conditioning frame, no score reset, so the object's masks continue
        exactly as in an uninterrupted propagate."""
        if not self.refine_resume:
            return None
        get_session = getattr(self._predictor, "_get_session", None)
        model = getattr(self._predictor, "model", None)
        add_action = getattr(model, "add_action_history", None)
        if get_session is None or add_action is None:
            return None
        try:
            sess_state = get_session(chunk["session_id"])["state"]
        except Exception:
            return None
        ids = self._session_object_ids(chunk["session_id"])
        if ids is None:
            return None
        carry: list = []
        for action in reversed(sess_state.get("action_history") or []):
            if "propagation" in str(action.get("type", "")):
                carry = [int(o) for o in (action.get("obj_ids") or [])]
                break

        def refine(obj_id: int, local_idx: int) -> None:
            add_action(sess_state, action_type="refine",
                       frame_idx=int(local_idx), obj_ids=[int(obj_id)])

        return ids, carry, refine

    def reset_state(self, state: dict) -> None:
        if state.get("closed"):
            return
        try:
            self._close_chunk(state)
        finally:
            state["closed"] = True

    # ── chunk-session internals ──────────────────────────────────────

    def _send_prompt(self, state: dict, frame_idx: int, obj_id: int, box,
                      neg=None) -> None:
        chunk = state["chunk"]
        width, height = state["video_size"]
        x1, y1, x2, y2 = box
        # Box -> two corner points with SAM2's box labels (2: TL, 3: BR), in
        # normalized coordinates: the instance path keeps stable obj_ids and
        # adds/refines incrementally instead of resetting state.
        pts = [[x1 / width, y1 / height], [x2 / width, y2 / height]]
        lbl = [2, 3]
        for nx, ny in (neg or []):
            pts.append([nx / width, ny / height])
            lbl.append(0)
        response = self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": chunk["session_id"],
                "frame_index": int(frame_idx) - chunk["lo"],
                "obj_id": int(obj_id),
                "points": pts,
                "point_labels": lbl,
                "rel_coordinates": True,
            }
        )
        # An explicit ``"outputs": None`` is the multiplex model refusing a
        # NEW object at its capacity (add_sam2_new_points); the add never
        # enters the session's action history, so don't count it as
        # prompted. A None response / missing key (base predictor, stubs)
        # means accepted.
        if isinstance(response, dict) and "outputs" in response \
                and response["outputs"] is None:
            chunk["refused"] = chunk.get("refused", 0) + 1
            if chunk["refused"] == 1:
                print(f"  [SAM3] add_prompt refused for obj {int(obj_id)} in "
                      f"chunk [{chunk['lo']},{chunk['hi']}) (tracker object "
                      f"capacity reached) — further refusals in this chunk "
                      f"suppressed")
            return
        chunk["prompted"].add(int(obj_id))

    def _open_chunk_for(self, state: dict, global_idx: int, *, reverse: bool) -> None:
        self._close_chunk(state)
        num_frames = len(state["frame_paths"])
        if reverse:
            hi = min(num_frames, global_idx + 1)
            lo = max(0, hi - self._chunk_frames)
        else:
            lo = global_idx
            hi = min(num_frames, lo + self._chunk_frames)

        # SAM3 sessions open a resource path; hand it a directory of symlinks
        # covering just this window.
        tmp_dir = tempfile.mkdtemp(prefix="sam3_chunk_")
        for i, src in enumerate(state["frame_paths"][lo:hi]):
            os.symlink(os.path.abspath(src), os.path.join(tmp_dir, f"{i:08d}.jpg"))

        offload_video, offload_state = state["offload"]
        t_open = time.perf_counter()
        response = self._predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": tmp_dir,
                "offload_video_to_cpu": offload_video,
                "offload_state_to_cpu": offload_state,
            }
        )
        open_s = time.perf_counter() - t_open
        self.session_open_s_total += open_s
        self.session_open_count += 1
        state["chunk"] = {
            "session_id": response["session_id"],
            "lo": lo,
            "hi": hi,
            "dir": tmp_dir,
            "open_s": open_s,
            # obj_ids prompted since the last propagate request in this
            # session; consumed (cleared) by each _stream_chunk call.
            "prompted": set(),
            # add_prompts the session refused (object capacity reached)
            "refused": 0,
        }

        # Seed: pending prompts that fall in this window, then carried
        # objects (fresh mask evidence only) that have no pending prompt here.
        seed_frame = (hi - 1) if reverse else lo
        pending_here = [(f, o, b, n) for (f, o, b, n) in state["pending"]
                        if lo <= f < hi]
        state["pending"] = [(f, o, b, n) for (f, o, b, n) in state["pending"]
                            if not (lo <= f < hi)]
        prompted_ids = set()
        for f, o, b, n in pending_here:
            self._send_prompt(state, f, o, b, neg=n)
            prompted_ids.add(o)
        solo = state.get("solo_ids")
        for obj_id, (seen_idx, box) in sorted(state["last_box"].items()):
            if obj_id in prompted_ids:
                continue
            if solo is not None and obj_id not in solo:
                # Late-seed backfill (mask_tracker.py): during a reverse
                # backfill pass this loop would otherwise re-seed the WHOLE
                # live roster across the boundary — wasted compute, and it
                # would drag every other object's `last_box` backward too.
                # `solo_ids` confines the chunk to just the backfill targets.
                continue
            if abs(seed_frame - seen_idx) > SAM3_RESEED_MAX_AGE:
                continue  # stale: let detector re-prompts recover it
            self._send_prompt(state, seed_frame, obj_id, box)

    def _close_chunk(self, state: dict) -> None:
        chunk = state.get("chunk")
        if not chunk:
            return
        try:
            self._predictor.handle_request(
                {"type": "close_session", "session_id": chunk["session_id"]}
            )
        finally:
            shutil.rmtree(chunk["dir"], ignore_errors=True)
            state["chunk"] = None
            # VRAM forensics for the recurring whole-card OOM: if allocated
            # memory climbs across chunk closes, session state is not being
            # released (close_session leak / lingering references); if it is
            # flat, the growth is intra-chunk (prompt accumulation).
            refused = int(chunk.get("refused", 0) or 0)
            refused_note = f" refused_prompts={refused}" if refused else ""
            retired = int(chunk.get("retired", 0) or 0)
            if retired:
                refused_note += f" retired_objects={retired}"
            open_s = chunk.get("open_s")
            open_note = f" open={open_s:.2f}s" if open_s is not None else ""
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 2**30
                    reserved = torch.cuda.memory_reserved() / 2**30
                    print(f"  [SAM3] chunk closed [{chunk['lo']},{chunk['hi']}) "
                          f"VRAM allocated={alloc:.2f}GiB reserved={reserved:.2f}GiB"
                          f"{open_note}{refused_note}")
                elif refused_note:
                    print(f"  [SAM3] chunk closed [{chunk['lo']},{chunk['hi']})"
                          f"{refused_note}")
            except Exception:
                pass

    def _stream_chunk(self, state: dict, start_global: int, lo: int, hi: int,
                      *, reverse: bool):
        """Stream one chunk from start_global, then recurse into the next
        chunk until the caller's [lo, hi) range is exhausted."""
        import torch

        chunk = state["chunk"]

        # One-shot segment resume (track_selective's selective_retire_closed /
        # selective_lazy_seed): the caller broke a propagate mid-stream to
        # change the object set and is resuming at the next frame. Ids the
        # session already tracks get a bare ``refine`` action instead of a box
        # prompt (which would add a conditioning frame and change their
        # masks); see _resume_refiner. Unset (default) = unchanged behaviour.
        refiner = (self._resume_refiner(chunk)
                   if state.pop("resume_refine", False) else None)
        refined: set = set()
        solo = state.get("solo_ids")
        local_start = start_global - chunk["lo"]
        if refiner is not None:
            session_ids, carry, refine = refiner
            # Everything the interrupted request was propagating (minus
            # retired ids, which upstream remove_object dropped from the
            # session) continues — stale or not, as it would have uninterrupted.
            for obj_id in sorted(set(carry)):
                if obj_id in chunk["prompted"] or obj_id not in session_ids:
                    continue
                if solo is not None and obj_id not in solo:
                    continue
                refine(obj_id, local_start)
                refined.add(obj_id)

        # SAM3 sessions route each propagate request through their action
        # history: objects WITHOUT an add/refine since the previous propagate
        # are not tracked at all (fetch/partial-only semantics, see
        # parse_action_history_for_propagation). Refresh every live object's
        # latest observed box before each window so the whole roster rides
        # ``propagation_partial``; stale objects stay with the detector
        # re-prompt recovery path, same rule as the chunk-boundary re-seed.
        for obj_id, (seen_idx, box) in sorted(state["last_box"].items()):
            if obj_id in chunk["prompted"] or obj_id in refined:
                continue
            if solo is not None and obj_id not in solo:
                continue  # backfill scoping — see _open_chunk_for
            if not (chunk["lo"] <= seen_idx < chunk["hi"]):
                continue
            if abs(start_global - seen_idx) > SAM3_RESEED_MAX_AGE:
                continue
            if refiner is not None and obj_id in refiner[0]:
                refiner[2](obj_id, local_start)
                continue
            self._send_prompt(state, seen_idx, obj_id, box)
        chunk["prompted"] = set()

        # frames of this chunk that fall inside the caller's range
        chunk_lo = max(lo, chunk["lo"])
        chunk_hi = min(hi, chunk["hi"])
        span = (start_global - chunk_lo) if reverse else (chunk_hi - 1 - start_global)

        request = {
            "type": "propagate_in_video",
            "session_id": chunk["session_id"],
            "propagation_direction": "backward" if reverse else "forward",
            "start_frame_index": int(local_start),
            "max_frame_num_to_track": int(max(span, 0)),
        }
        last_global = start_global
        for response in self._predictor.handle_stream_request(request):
            outputs = response.get("outputs") or {}
            raw_obj_ids = outputs.get("out_obj_ids", [])
            raw_masks = outputs.get("out_binary_masks", [])
            raw_probs = outputs.get("out_tracker_probs")
            if hasattr(raw_obj_ids, "detach"):
                raw_obj_ids = raw_obj_ids.detach().cpu().numpy()
            if hasattr(raw_masks, "detach"):
                raw_masks = raw_masks.detach().cpu().numpy()
            obj_ids = np.asarray(raw_obj_ids, dtype=np.int64)
            masks = np.asarray(raw_masks)
            if masks.ndim == 2:
                masks = masks[None, ...]

            # Per-object tracker confidence, aligned with obj_ids. Absent on
            # older models / stubs; flatten (N,1)->(N,); drop it defensively
            # if it doesn't line up rather than mis-attaching scores.
            probs = None
            if raw_probs is not None:
                if hasattr(raw_probs, "detach"):
                    raw_probs = raw_probs.detach().cpu().numpy()
                probs = np.asarray(raw_probs, dtype=np.float64).reshape(-1)
                if probs.shape[0] != obj_ids.shape[0]:
                    probs = None
            global_idx = int(response["frame_index"]) + chunk["lo"]
            last_global = global_idx

            # record per-object boxes so the next chunk can re-seed
            for i, oid in enumerate(obj_ids.tolist()):
                bbox = _mask_bbox(masks[i, 0] if masks.ndim == 4 else masks[i])
                if bbox is not None:
                    state["last_box"][int(oid)] = (global_idx, bbox)

            yield (
                global_idx,
                obj_ids,
                torch.as_tensor(masks, dtype=torch.float32),
                probs,
            )

        # continue into the next chunk if the caller's range extends past it
        if reverse and last_global > lo:
            next_global = last_global - 1
            self._open_chunk_for(state, next_global, reverse=True)
            yield from self._stream_chunk(state, next_global, lo, hi, reverse=True)
        elif not reverse and last_global < hi - 1:
            next_global = last_global + 1
            self._open_chunk_for(state, next_global, reverse=False)
            yield from self._stream_chunk(state, next_global, lo, hi, reverse=False)


class Sam3FieldTracker(MaskFieldTracker):
    """Run the field tracker with Meta's SAM3 video model (instance prompts,
    chunked sessions).

    ``track(..., should_stop=...)`` (cooperative user-STOP abort) is inherited
    from ``MaskFieldTracker`` and works here too: the shared loop polls the
    predicate once per propagated frame of the adapter's stream and raises
    ``TrackStopped``; the ``finally`` cleanup calls this adapter's
    ``reset_state``, which closes the live chunk session and removes its
    temp dir.

    ``multiplex=True`` (the ``sam3_multiplex`` knob) swaps the base SAM3
    video predictor for Meta's SAM 3.1 Object Multiplex builder
    (``build_sam3_predictor(version="sam3.1", ...)``). Both predictors
    dispatch through the same ``Sam3BasePredictor.handle_request`` /
    ``handle_stream_request`` protocol, so ``Sam3PredictorAdapter`` below is
    unchanged either way -- only the predictor construction differs. See
    job_config.sam3_multiplex for the tracking-behavior caveat.
    """

    LOG_TAG = "SAM3"
    DIAG_PREFIX = "sam3"

    def __init__(
        self,
        predictor: Any | None = None,
        *,
        checkpoint: str | None = None,
        device: str = "cuda",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        compile_model: bool = False,
        max_objects: int = 20,
        use_fa3: bool = True,  # accepted for call-site compat; multiplex-only
        chunk_frames: int | None = None,
        multiplex: bool = SAM3_MULTIPLEX,
        skip_detector_heads: bool = False,
        retire_culled: bool = False,
        frame_load_workers: int = 0,
    ) -> None:
        if device != "cuda":
            raise RuntimeError("SAM 3 video model currently requires CUDA")

        if predictor is None:
            checkpoint = checkpoint or os.environ.get("SAM3_CHECKPOINT") or None

            if multiplex:
                # SAM 3.1 Object Multiplex: joint multi-object propagation
                # through the SAME handle_request/handle_stream_request
                # protocol (Sam3BasePredictor), so Sam3PredictorAdapter needs
                # no changes. use_fa3 is forced False here regardless of the
                # use_fa3 param above: the builder's own default is True, and
                # FlashAttention-3 is NOT installed on this host (see
                # docs/SAM3.md) -- passing the builder's default would break
                # at model-build time. Object capacity (max_num_objects,
                # builder default 16) is NOT the roster size: each (re)seed
                # mints a new obj_id and chunk sessions never remove objects,
                # so 16 filled within ~16 frames and later adds were refused
                # (see SAM3_MULTIPLEX_MAX_OBJECTS). multiplex_count (objects
                # per bucket) stays at its default.
                from sam3.model_builder import build_sam3_predictor

                predictor = build_sam3_predictor(
                    version="sam3.1",
                    checkpoint_path=checkpoint,
                    compile=compile_model,
                    use_fa3=False,
                    max_num_objects=max(SAM3_MULTIPLEX_MAX_OBJECTS,
                                        int(max_objects)),
                )
                # init_state offload kwarg + uncached-frame fetch KeyError —
                # see the module-level shim helpers.
                _install_multiplex_shims(getattr(predictor, "model", None))
            else:
                from sam3.model_builder import build_sam3_video_predictor

                predictor = build_sam3_video_predictor(
                    checkpoint_path=checkpoint,
                    compile=compile_model,
                )
                # Instance-tracking capacity is a model attribute, not a
                # builder arg; raise it so re-prompting can exceed the demo
                # default.
                model = getattr(predictor, "model", None)
                if model is not None and hasattr(model, "max_num_objects"):
                    model.max_num_objects = max(
                        int(getattr(model, "max_num_objects", 0) or 0),
                        int(max_objects),
                    )

        # sam3_skip_detector_heads: base SAM3 only (never the multiplex model;
        # install_skip_detector_heads also refuses multiplex classes itself).
        self.skip_detector_heads_active = False
        if skip_detector_heads and not multiplex:
            self.skip_detector_heads_active = install_skip_detector_heads(
                getattr(predictor, "model", None))
            if self.skip_detector_heads_active:
                print("  [SAM3] skip_detector_heads: partial-path detector heads "
                      "disabled (backbone + tracker neck only)")

        # sam3_frame_load_workers: thread-pooled chunk frame loader
        # (byte-identical to upstream; patches sam3.model.io_utils globally).
        self.frame_load_workers_active = 0
        if int(frame_load_workers or 0) > 1:
            if install_threaded_frame_loader(int(frame_load_workers)):
                self.frame_load_workers_active = int(frame_load_workers)
                print(f"  [SAM3] threaded frame loader: "
                      f"{self.frame_load_workers_active} workers")

        adapter = Sam3PredictorAdapter(
            predictor,
            chunk_frames=chunk_frames if chunk_frames else SAM3_CHUNK_FRAMES,
        )
        # Multiplex remove_objects can KeyError on trimmed memory: retire by
        # adapter bookkeeping only there.
        adapter.session_remove = not multiplex
        adapter.refine_resume = not multiplex
        self.retire_culled = bool(retire_culled)

        super().__init__(
            predictor=adapter,
            device=device,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        # The adapter returns real torch tensors and benefits from inference mode.
        self._real_torch = True

    def _retire_object(self, state: Any, oid: int) -> None:
        """sam3_retire_culled: drop a culled/evicted id from the SAM3 session
        so it stops costing propagation (base-class hook, no-op unless the
        knob is on)."""
        if not self.retire_culled:
            return
        try:
            self._predictor.remove_object(state, oid)
        except Exception as exc:  # never let a retire kill the view
            print(f"  [SAM3] retire of obj {int(oid)} failed: {exc!r}")

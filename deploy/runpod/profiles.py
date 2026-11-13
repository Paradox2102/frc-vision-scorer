"""VRAM tiers and the quality ladder for the RunPod deployment.

Stdlib only: imported by the local CLI (runpod_cli.py) and run directly on the
pod by bootstrap.sh (system python3, before the app venv exists) to write the
starting ``profile.env``.

A "profile" is a set of environment variables exported before web/app.py
starts. Every knob in web/job_config.py resolves per-job override > runtime
setting > env > default, so the env is the server-wide default and the tuner
can still try a rung per job via ``config_json`` without restarting.

Base env (every tier)
---------------------
* The three ``SAM3_*`` lines mirror web/start_all.sh (the 2026-06-11 known-good
  OOM-containment profile) so the pod runs exactly what :8004 runs locally.
* ``TEAM_READER=jev`` mirrors start_all.sh too (hosted reader, no vLLM).
* ``VLLM_PHASE_EXCLUSIVE=0``: there is no vLLM on the pod. With the knob on, every
  job probes :8006 and then ``asyncio.sleep(5)``s before tracking "to let CUDA
  reclaim the pool" (web/app.py, tracking setup) — 5 s of pure waste per job.
  Output-identical: the knob only gates vLLM stop/start calls.

What VRAM does and does not buy (read from the code, Nov 12 2026)
-----------------------------------------------------------------
* ``sam3_chunk_frames`` is the only VRAM-bound knob. SAM3 copies a session's
  whole frame window to the GPU at 1008^2 (~14 MB/frame; offload_video_to_cpu
  is ignored by its input-batch construction — web/field/sam3_tracker.py
  header). The default 200 exists because 400 OOM'd a 24 GB card. Bigger
  chunks mean fewer chunk boundaries, and a boundary drops SAM3's appearance
  memory (an occluded object is not re-seeded) — job_config: "Bigger = fewer
  identity-risky chunk boundaries; cap is VRAM, not speed". So it CHANGES
  output (hopefully for the better) and is not a speed lever: it is a ladder
  rung, and the tier only sets how big that rung may go.
* Tracker state stays on the GPU automatically when >= 16 GB is free
  (web/app.py VRAM probe) — every pod tier already gets that.
* ``sam3_max_gpu_frames`` (VRAM-fit mode) can only RAISE frame_skip (fewer
  tracked frames) and disables the stream_analysis GPU prefetch: never useful
  on a big card, left at 0.
* ``sam3_export_scale`` is left at 0.5: "export resolution is NOT a lever — the
  model resizes everything to a fixed square" (job_config corner_track_stride
  comment), and the Nov 11 tracking-coverage study found export resolution is
  not a factor in missed boxes. Costs export time + mask RAM for no evidence of
  gain, so it is not on the ladder.
* ``sam3_multiplex`` / ``sam3_compile``: A/B'd, slower or no win — stay OFF.
* ``selective_mask_prop`` and the three output-identical SAM3 speed levers
  (skip-detector-heads, retire-culled, retire-closed) stay at their ON defaults.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Knob <-> env mapping for the knobs this module touches (must match
# web/job_config.py KNOBS; tests/test_runpod_deploy.py cross-checks it).
# ---------------------------------------------------------------------------
KNOB_ENV = {
    "corner_track_stride": "CORNER_TRACK_STRIDE",
    "sam3_track_fps": "SAM3_TRACK_FPS",
    "sam3_chunk_frames": "SAM3_CHUNK_FRAMES",
}

# Current repo defaults for the ladder knobs (web/job_config.py, Nov 12 2026).
REPO_DEFAULTS = {
    "corner_track_stride": 3,
    "sam3_track_fps": 5.0,
    "sam3_chunk_frames": 200,
}

BASE_ENV = {
    # web/start_all.sh parity
    "TEAM_READER": "jev",
    "SAM3_DARK_REANCHOR_ENABLED": "0",
    "SAM3_EVICT_DARK_FRAMES": "99999",
    "SAM3_EVICT_MIN_ABSENCES": "99",
    # no vLLM on the pod (see module docstring)
    "VLLM_PHASE_EXCLUSIVE": "0",
}


@dataclass(frozen=True)
class Tier:
    name: str
    min_gib: float          # inclusive lower bound of total VRAM (GiB)
    max_gib: float          # exclusive upper bound
    chunk_cap: int          # sam3_chunk_frames used by the chunk rung
    note: str = ""


# chunk_cap budget: ~14 MB/frame video + ~4 GB model + per-object tracker state
# (also roughly linear in frames). A full match is ~160 s active, i.e. ~800
# main-view frames at 5 fps and ~1300 at 8 fps; the caps keep the video window
# at <= ~30% of the card so object state has room.
TIERS = (
    Tier("t24", 0, 40, 200, "below the target range; repo default chunk (24 GB-safe)"),
    Tier("t48", 40, 64, 600, "A40 / L40S / RTX A6000 / RTX 6000 Ada (48 GB)"),
    Tier("t80", 64, 110, 1000, "A100 80GB / H100 / H100 NVL / RTX PRO 6000 (80-96 GB)"),
    Tier("t141", 110, float("inf"), 2000, "H200 (141 GB): one session per view for a full match"),
)


def tier_for_vram(total_mib: float) -> Tier:
    """Tier for a card with ``total_mib`` MiB of VRAM (nvidia-smi memory.total).
    Cards report a little under their marketing size (an 80 GB H100 shows
    81559 MiB = 79.6 GiB, a 48 GB A40 46068 MiB = 45 GiB)."""
    gib = float(total_mib) / 1024.0
    for t in TIERS:
        if t.min_gib <= gib < t.max_gib:
            return t
    return TIERS[0]


# ---------------------------------------------------------------------------
# Quality ladder: cumulative rungs, cheapest expected-value-per-minute first.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rung:
    name: str
    knobs: dict             # FULL ladder-knob values for this rung (explicit)
    delta: dict             # what this rung adds over the previous one
    rationale: str = ""

    def config(self) -> dict:
        """Per-job config_json for this rung (every ladder knob explicit, so
        a profile.env already applied on the server cannot leak into it)."""
        return dict(self.knobs)

    def env(self) -> dict:
        return {KNOB_ENV[k]: _env_str(v) for k, v in self.knobs.items()}


# (delta, rationale) — chunk value is filled from the tier.
_STEPS = (
    ("R1", {"corner_track_stride": 2},
     "Undo the Nov 6 'B3' interim corner decimation. Stride 2 was the Oct 31 "
     "validated default (qm8+qm11 full-match A/B: identical per-slot identity vs "
     "stride 1, a corner frame every 0.27 s at 8 fps). Stride 3 'costs some auto "
     "coverage' (job_config) and the Nov 6 held-out A/B lost auto coverage on "
     "4/4 matches vs pre-B3. Cost: corner propagation +~50% (~+25-35% job wall "
     "on the 3090)."),
    ("R2", {"sam3_track_fps": 8.0},
     "Back to the pre-B3 tracking rate (8 fps). Held-out Nov 6: B3 (5 fps + "
     "stride 3) agg meanNN 0.47 -> 0.73 m vs pre-B3, and selective_mask_prop's "
     "frame-denominated knobs (gap_trigger 7, min_streak 5, window_max 120) were "
     "tuned at 8 fps and are NOT rescaled for fps (ab_qm8.py 'speed' note), so "
     "this also puts selective back on its tuned time scale. Cost: ~+60% SAM "
     "frames on every view."),
    ("R3", {"sam3_chunk_frames": None, "sam3_track_fps": 10.0},
     "BIG JUMP (Nov 12, Nick: big GPUs sit mostly idle at R1/R2 -- GPU util "
     "0-77% spiky, CPU idle, peak 12 GB of 96). Tier-cap SAM3 chunks (fewer "
     "chunk boundaries, each drops SAM3 appearance memory; VRAM-bound, roughly "
     "time-neutral; UNVALIDATED beyond 400 frames) + 10 fps (denser evidence "
     "through crossings; no A/B above 8 fps). Cost ~+25% SAM frames over R2."),
    ("R4", {"corner_track_stride": 1},
     "Full-rate corners. Oct 31 A/B found no per-slot identity gain of stride 1 "
     "over stride 2 at 8 fps, for ~2x corner propagation -- measured again "
     "here at 10 fps because corners are the cheap views (~70-90 ms/frame)."),
    ("R5", {"sam3_track_fps": 15.0},
     "15 fps: the ceiling probe. Only reachable inside 8 min if views run "
     "concurrently or the card is much faster per frame than the 3090."),
)


def ladder(tier: Tier) -> list[Rung]:
    """Cumulative rungs R0..R5 for ``tier``. R0 = repo defaults; R3+ jump harder."""
    rungs = [Rung("R0", dict(REPO_DEFAULTS), {},
                  "Repo defaults (Nov 12 2026: 5 fps, corner stride 3, 200-frame "
                  "SAM3 chunks) — what :8004 runs on the 3090.")]
    cur = dict(REPO_DEFAULTS)
    for name, delta, why in _STEPS:
        d = dict(delta)
        if "sam3_chunk_frames" in d:
            d["sam3_chunk_frames"] = max(REPO_DEFAULTS["sam3_chunk_frames"], tier.chunk_cap)
        cur = {**cur, **d}
        rungs.append(Rung(name, dict(cur), d, why))
    return rungs


# ---------------------------------------------------------------------------
# Experimental arms: measured by `tune` AFTER the ladder, each on top of the
# recommended rung (R1 if none), named <rung><tag> (e.g. R1M). Never
# recommended or applied automatically -- the table shows them next to their
# base rung so the A/B is read by eye. Skip with `tune --no-arms`.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    tag: str
    knobs: dict
    rationale: str = ""


ARMS = (
    Arm("M", {"sam3_multiplex": True},
        "SAM 3.1 Object Multiplex (sam3.1_multiplex.pt): objects propagated "
        "jointly in shared buckets instead of one tracker pass each. Nov 6 on "
        "the 24 GB 3090: +26% slower and OOM on a full match -> knob OFF. The "
        "pod profile (Nov 12) is per-object launch/sync bound (~5.5 ms per "
        "object-frame, ~640 kernels + ~20 .item() each, GPU ~60% busy) with "
        "VRAM to spare, which is exactly what multiplex batching targets -- "
        "re-measure on a big card before writing it off."),
    Arm("P", {"sam3_parallel_views": True},
        "Thread-parallel SAM3 views (one predictor + CUDA stream per view, "
        "web/field/parallel_views.py). Nov 12 pod R1P gained nothing (6.6 -> "
        "6.5 min): the GIL serialised the Python-heavy per-frame work. A py-spy "
        "--gil profile put the scalar hub-occlusion test at ~33% of GIL-held "
        "time; it is now vectorised (~40x) on main, and chunk frame loading is "
        "thread-pooled (sam3_frame_load_workers) -- re-measure."),
    # The resident Qwen3.5-4B reader: the scripts/ab_qm8.py "qwen35-escape" arm
    # (team_reader=vllm + vlm_model=qwen3.5-4b; read_census / read_keep_evicted
    # are already the defaults). The escape prompt and thinking-off come from
    # the web/vlm_models.py registry entry, not knobs. vllm_phase_exclusive is
    # explicit (BASE_ENV sets it too) so the reader stays resident beside SAM3
    # instead of being stopped/cold-started (~106 s on the pod) every job. Its
    # VRAM budget needs no pod value: on a >= 30 GB card serve_util makes it
    # absolute, gpu_util 0.5 x 24 = 12 GiB (util 0.126 on the 96 GB card) --
    # exactly what the pod bench ran (12.5 GB up, 13.4 GB after reads).
    Arm("Q", {"team_reader": "vllm", "vlm_model": "qwen3.5-4b",
              "vllm_phase_exclusive": False},
        "Resident Qwen3.5-4B team reader (vLLM, escape prompt, no-think) instead "
        "of Jev for live reads. 200-crop reader bench on the pod: 71/0 wrong of "
        "90 legible, 2% FP, 13 GB vs Jev 59/0, 2%; the live full-match A/B has "
        "never been run. The big card has the VRAM to keep it resident beside "
        "SAM3 (R1 peak ~12 GB). Needs .venv-vllm on the pod; start vLLM before "
        "this arm so its wall time excludes the cold start."),
)


def arm_rung(base: Rung, arm: Arm) -> Rung:
    """``base`` + ``arm.knobs`` as a measurable rung named ``base.name + tag``."""
    return Rung(base.name + arm.tag, {**base.knobs, **arm.knobs}, dict(arm.knobs),
                f"{base.name} + arm {arm.tag}: {arm.rationale}")


def rung_by_name(tier: Tier, name: str) -> Rung:
    name = name.upper()
    for r in ladder(tier):
        if r.name == name:
            return r
    raise KeyError(f"unknown rung {name!r} (have {', '.join(r.name for r in ladder(tier))})")


# ---------------------------------------------------------------------------
# profile.env rendering
# ---------------------------------------------------------------------------
def _env_str(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and v.is_integer():
        return f"{v:.1f}"
    return str(v)


def profile_env(tier: Tier, rung: Rung | None = None, extra: dict | None = None) -> dict:
    env = dict(BASE_ENV)
    env["FRC_GPU_TIER"] = tier.name
    env["FRC_PROFILE_RUNG"] = rung.name if rung else "R0"
    if rung is not None:
        env.update(rung.env())
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def render_profile_env(env: dict, header: str = "") -> str:
    lines = ["# Generated by deploy/runpod/profiles.py — sourced (set -a) by",
             "# deploy/runpod/serve.sh before web/app.py starts. Re-apply with",
             "#   runpod_cli.py tune --apply R<n>   (or profile --apply R<n>)"]
    if header:
        lines += [f"# {h}" for h in header.splitlines()]
    for k in sorted(env):
        v = str(env[k])
        if any(c in v for c in " \t\"'$`\\#"):
            raise ValueError(f"profile value for {k} needs quoting: {v!r}")
        lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


def parse_env_file(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.startswith("export "):
            k = k[len("export "):]
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def detect_gpu() -> tuple[str, float]:
    """(gpu name, total VRAM MiB) of GPU 0 via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found")
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    name, mib = [s.strip() for s in out.rsplit(",", 1)]
    return name, float(mib)


def describe_ladder(tier: Tier) -> str:
    rows = [f"tier {tier.name}: {tier.note} (chunk cap {tier.chunk_cap})"]
    for r in ladder(tier):
        k = r.knobs
        rows.append(f"  {r.name}: fps={k['sam3_track_fps']:g} corner_stride={k['corner_track_stride']} "
                    f"chunk={k['sam3_chunk_frames']}  (+{r.delta or 'defaults'})")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write/show the RunPod knob profile")
    ap.add_argument("--write-profile", metavar="PATH",
                    help="write profile.env for the detected GPU (skipped if it exists, unless --force)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--vram-mib", type=float, help="override detection (testing)")
    ap.add_argument("--rung", default=None, help="write this rung instead of R0 (env-only start)")
    ap.add_argument("--show", action="store_true", help="print tier + ladder")
    a = ap.parse_args(argv)
    if a.vram_mib is not None:
        name, mib = "override", a.vram_mib
    else:
        name, mib = detect_gpu()
    tier = tier_for_vram(mib)
    print(f"GPU: {name}, {mib / 1024:.1f} GiB -> {tier.name} ({tier.note})")
    if a.show or not a.write_profile:
        print(describe_ladder(tier))
    if a.write_profile:
        p = Path(a.write_profile)
        if p.exists() and not a.force:
            print(f"{p} exists; keeping it (use --force to regenerate)")
            return 0
        rung = rung_by_name(tier, a.rung) if a.rung else None
        env = profile_env(tier, rung)
        p.write_text(render_profile_env(env, f"GPU {name}, {mib:.0f} MiB"))
        print(f"wrote {p} (tier {tier.name}, rung {env['FRC_PROFILE_RUNG']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

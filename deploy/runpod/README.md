# RunPod weekend runbook

One rented GPU for a whole competition weekend, processing each full match as
its video appears. Goal: 6-8 min wall time per full match at the best quality
that fits. Everything here is driven by `deploy/runpod/runpod_cli.py` (stdlib
Python; run it from the main checkout):

```bash
alias rp='.venv/bin/python deploy/runpod/runpod_cli.py'
```

The CLI reads `RUNPOD_API` from the main checkout's `.env`. That key never
leaves your machine.


> **API keys are never in the repo or the bundle.** The pod's keys come from
> whoever runs the CLI: typed at the `up`/`secrets` prompt, or — only with the
> explicit `--secrets-from-env` flag — read from *your own* local `.env`.
> `--yes` skips the rent prompt but never reads `.env` by itself.

## 1. Pick a GPU

```bash
rp gpus            # live prices, VRAM, stock, and the tier each card lands in
```

Prices are on-demand $/GPU-hour (Nov 12 2026, `rp gpus` for current):

| GPU (gpuTypeId) | VRAM | Secure $/h | ~60 h weekend | Tier |
|---|---|---|---|---|
| NVIDIA A40 / RTX A6000 | 48 GB | 0.59 / 0.69 | $35-41 | t48 |
| NVIDIA L40S | 48 GB | 1.09 | $65 | t48 |
| NVIDIA A100 80GB PCIe | 80 GB | 2.09 | $125 | t80 |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 96 GB | 2.69 | $161 | t80 |
| NVIDIA H100 PCIe | 80 GB | 2.89 | $173 | t80 |
| NVIDIA H100 80GB HBM3 (SXM) | 80 GB | 4.39 | $263 | t80 |
| NVIDIA H200 | 141 GB | 5.29 | $317 | t141 |

Add pod-volume storage (150 GB default; billed while the pod exists, even
when stopped). The pipeline is mostly SAM3 propagation (GPU) plus per-frame
CPU work (decode, tesseract OCR), so pick a card with plenty of vCPUs (`rp gpus`
shows the minimum vCPU/RAM RunPod currently offers for it).

Default `up` priority when `--gpu` is not given: H100 SXM, RTX PRO 6000
Server, H100 PCIe, A100 SXM 80GB, A100 PCIe 80GB (`gpuTypePriority=custom`,
so the first one in stock wins). None of these has been timed on this
pipeline yet; `tune` is what measures it.

## 2. Bring it up

```bash
rp up --dry-run --secrets-from-env        # exact API body + bundle manifest; creates NOTHING
rp up --gpu "NVIDIA H100 80GB HBM3" --gpu "NVIDIA H100 PCIe"    # interactive
rp up --gpu "NVIDIA H100 80GB HBM3" --yes  # non-interactive: secrets from local .env, no prompts
```

`up` will:

1. **Create the pod** (`POST /v1/pods`, after you type `rent`): Secure cloud,
   1 GPU, image `runpod/pytorch:1.5.1-cu1281-torch291-ubuntu2404` (Ubuntu
   24.04, CUDA 12.8.1, sshd), 150 GB volume at `/workspace`, 40 GB container
   disk, CUDA >= 12.8 hosts only, **only `22/tcp` exposed**. The only `env`
   sent is your ssh `PUBLIC_KEY`. If `.state.json` already has a pod, `up`
   reuses it (and offers to start it if it is stopped).
2. **Wait for SSH** on the public IP/port.
3. **Ship the bundle** (tar over ssh into `/workspace/frc`): `git archive` of
   `web/ src/ data/field/ pyproject.toml deploy/runpod/` from `--ref` (default
   `main`, committed content only; it warns when the tree is dirty), minus
   uploads, replays, logs, the 35 MB field CAD file and docs screenshots. Also
   ships the weights below, the event state from `web/uploads` (calibrations,
   camera layouts, zones, season overrides, clock-scan cache), and a
   consistent snapshot of `frc_scorer.db`. `--no-db` skips the DB and the event
   state for a fresh start. Re-running `up` against a pod that already has a
   DB keeps the pod's DB and event state (`--overwrite-db` to replace them),
   and `up` refuses to run while a job is processing.
4. **Install secrets**: asks for each var (`[local .env: ****abcd]`,
   Enter=keep, paste=replace, `-`=skip) and writes them over ssh to
   `/workspace/frc/.env` (chmod 600). Required: `JEV_TYPESAFE_API_TOKEN`,
   `HF_TOKEN`, `TBA_API_KEY`. Optional: `ANTHROPIC_API_KEY`, `DEEPSEEK_API`
   (non-default reader paths only). Never sent: `RUNPOD_API`, `CLOUDFLARE_*`,
   `ROBOFLOW_API`, `MANTA_API` (only local scripts read it).
5. **Bootstrap** (`bootstrap.sh`, idempotent): apt ffmpeg/tesseract/libGL, uv
   0.10.0 + deno 2.8.1 on the volume, `/workspace/frc/.venv` (Python 3.12,
   torch 2.10.0+cu128) from `requirements-runpod*.txt`, prefetch of SAM3
   (`facebook/sam3`, ~3.5 GB, gated) / DINOv3 / EasyOCR into
   `/workspace/cache`, an import smoke test (CUDA kernels for this GPU arch,
   `import web.app`), `profile.env` from the VRAM tier, then the server
   (`serve.sh`, log `/workspace/logs/server.log`). It runs detached on the
   pod and `up` streams `/workspace/logs/bootstrap.log`, so a dropped ssh
   session does not kill a half-finished install. Re-running it is safe.

First boot takes about 10-15 min (venv about 5 GB, SAM3 download). A re-run
after a stop/start only reinstalls the apt packages.

Weights manifest (from the main checkout, never LFS pointers):

| file | size | read by |
|---|---|---|
| runs/train/hub_zone_yolo26s_2026cur_v2/weights/best.pt | 20 MB | app.py MODEL_PATH |
| runs/train/frc_bumpers_equipment_hardneg/weights/best.pt | 52 MB | app.py BUMPER_MODEL_PATH |
| runs/train/frc_bumpers_2026cur_v3/weights/best.pt | 52 MB | window-label student detector |
| yolo26n.pt | 6 MB | person detector |
| data/teacher/models/s1_digits.pt | 48 MB | window_labels_reader=student_only |
| data/teacher/models/s4_embedder.pt | 44 MB | robot_embedder=s4 |
| checkpoints/sam2.1_hiera_small.pt | 184 MB | tracking_backend=sam2 fallback only (`--no-sam2-fallback`) |

SAM3 itself comes from Hugging Face on the pod.

## 3. Use it

```bash
rp tunnel --reconnect      # http://localhost:8014 -> pod :8004 (your local :8004 stays yours)
```

Browse to http://localhost:8014, load the event, and process matches exactly
as you do locally (YouTube URL or upload). The app has **no auth**, so it is
never exposed on RunPod's public HTTP proxy. The tunnel is the only way in.

```bash
rp status                  # pod state, $/h, uptime cost, GPU util/VRAM, server, active job
rp logs -f                 # server log
rp restart                 # refuses while a job runs (--force to override)
rp push                    # re-ship main (code only) + restart; --bootstrap if deps changed
rp secrets                 # re-enter secrets (+ --cookies FILE, --restart)
rp ssh                     # shell; `rp ssh -- nvidia-smi` for one command
```

## 4. Tune: pick the best profile that fits the budget

```bash
rp tune                    # R0 upwards, stops at the first rung over 8.0 min
rp tune --budget-min 7 --start R2
rp tune --show             # table only
rp tune --apply R3         # write that rung into profile.env + restart
rp tune --arms-only        # only the experimental arms, on the measured recommendation
```

After the climb, `tune` also measures the **experimental arms** in
`profiles.ARMS` on top of the recommended rung (R1 if none), as `<rung><tag>`
rows in the same table — currently `M` = `sam3_multiplex` (SAM 3.1 Object
Multiplex; lost on the 24 GB 3090, re-measured on big cards), `P` =
`sam3_parallel_views` (thread-parallel views, re-measured after the GIL fixes)
and `Q` = the resident Qwen3.5-4B vLLM team reader instead of Jev
(`team_reader=vllm`, `vlm_model=qwen3.5-4b`). Bootstrap builds
`/workspace/frc/.venv-vllm` (vLLM 0.25.1, ~9 GB) and prefetches every arm's
reader weights via `setup_vllm.sh` — a soft step (`--no-vllm` skips; log in
`/workspace/logs/vllm-setup.log`). Before a reader arm, `tune` starts vLLM
through the app's own lifecycle (`vllm_ctl.py start <model>`: same served
model, GPU budget and serve args as a job) so the arm's wall excludes the
~106 s cold start and vLLM profiles memory on an idle card; it stops vLLM after
the arm. A failed pre-start marks only that arm as an error. The Q row's
peak VRAM is whole-card, so it includes the reader's ~13 GB. Arms are never
recommended or applied; `--no-arms` skips them. The 3.1 checkpoint
(`facebook/sam3.1`, gated separately) is prefetched by bootstrap with a warning,
not a failure, if the HF token lacks access.

Each rung is one **full-match** run of 2026cur qm8 (the cached local video,
uploaded once to `/workspace/tune`; no YouTube), submitted with the rung's
knobs as per-job `config_json`, so no restart is needed between rungs. The
driver (`pod_job.py`) runs detached **on the pod**: a dropped ssh session or
a closed laptop does not kill the run, and re-running `tune` re-attaches. Per
rung it records:

* `job`: submit to websocket "complete".
* `final`: submit to the replay carrying the window-label relabel. This is
  when the labelled result exists, and it is what the budget checks by
  default (`--budget-on job` to budget the core job only).
* Peak VRAM (whole card, `nvidia-smi` every 2 s).
* Local scores: the replay's `meta.json` is pulled to
  `data/runpod_pull/<podId>/tune/` and scored with the existing scorers
  (`scripts/ab_qm8.score` Manta mean-NN, `scripts/score_ground_truth.score_replay`
  hand-GT coverage / error / labels right / wrong / misID, `--score-shift`
  0.475). The local DB and `web/replays` are never written.

Results: `deploy/runpod/tune_results/<podId>.json`. Measured rungs are
skipped on re-run (`--redo` to repeat). Each rung costs about one match of GPU
time. The recommendation is the highest rung that finished within budget.
Read the scores before applying: qm8 label metrics swing a few points between
identical runs.

### Host fingerprint and per-rung profiles

Before the first rung, `tune` runs `deploy/runpod/host_bench.py` on the pod
(app venv, idle server, about 10 s, CPU only) and prints one line (numbers
illustrative):

```
host: AMD EPYC ... (max 3800 MHz) | 16/128 cpus, quota 16 | 1007.6 GB RAM (cgroup 180.0 GB) | NVIDIA RTX PRO 6000 ... drv 570.x | 1T: py 14 Mops/s, SAM3 frame load 54 ms (PIL 20 ms); 16T load 9.5x
```

It reports the CPU model and max clock, the sched affinity vs. logical CPU
count, the cgroup CPU quota (v2 `cpu.max` or v1 `cfs_quota_us`), RAM (host and
cgroup limit), and the GPU name and driver. It also runs three single-thread
benchmarks, because per-thread speed matters more to this pipeline than core
count:

* `py_mops`: a pure-Python integer loop.
* `load_ms_per_frame`: the SAM3 frame-loader op, exactly as
  `sam3 io_utils._load_img_as_tensor` runs it (PIL decode of a deterministic
  960x540 JPEG, RGB, `TF.resize` to 1008x1008, `TF.to_tensor`).
  `pil_ms_per_frame` is the decode and resize alone.
* `thread_speedup`: the same op on `min(16, affinity)` threads.

The Nov 12 EPYC pod ran the loader op at about 54 ms per frame. A Ryzen 7600X
runs it at about 20 ms, with `py_mops` about 24. The result is stored under
`host` in the results JSON and cached per pod; `--redo` re-measures it. The
raw line is also kept at `/workspace/tune/host.json`.

Every job, ladder rungs and arms alike, is **profiled by default**
(`--no-profile` turns it off). `pod_job.py`, the detached driver on the pod,
starts two processes right before it submits the job and stops them when the
job's final result exists (or when it fails):

* `py-spy record --gil --rate 20 --format raw`, attached to the server PID
  (the process LISTENing on :8004, found through `/proc`, with serve.sh's
  `server.pid` as a fallback). It is stopped with SIGINT, which makes py-spy
  flush the file. If attaching fails (ptrace not allowed), the driver retries
  with `--nonblocking`. If that fails too, the error is recorded and the rung
  still runs. `--duration 7200` caps an orphaned py-spy, and a new job
  SIGINTs any stale profiler for the same files.
* `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used -l 1`.

Because the profilers belong to the pod-side driver, they follow the job, not
your ssh session: re-running `tune` re-attaches to the same driver and its
profilers. The summary is written into the rung's row as `profile`, before
the final status:

| key | meaning |
|---|---|
| `gpu_util_mean`, `gpu_util_p50` | GPU utilization over the job (1 Hz) |
| `gpu_busy_frac` | fraction of samples at or above 50% |
| `gpu_mem_peak_mib` | peak memory.used |
| `pyspy_samples`, `gil_held_frac` | GIL-held samples (`--gil` drops the rest); their share of wall × rate |
| `server_cpu_cores_mean` | server CPU time / wall (avg cores busy, incl. reaped children) |
| `gil_top` | top 15 project/sam3 functions by **inclusive** share of GIL samples. Each function counts at most once per stack, line numbers are stripped, and pure pass-through wrappers are hidden. `self_share` is shown alongside. |
| `self_top`, `project_self_top` | top 10 leaf frames (any code), and the innermost project frame of each stack |
| `pyspy_error`, `gpu_error` | why a half of the profile is missing |

The table shows `gpu%` (util mean) per rung. For one rung's full breakdown:

```bash
rp tune --profile-report R1                 # from tune_results/<podId>.json
rp tune --profile-report R1 --pod-id 1wjnhb11v7bgle   # an old (terminated) pod
rp tune --profile-report data/runpod_pull/<podId>/tune/jobs/R1.pyspy.txt   # any raw file
```

Raw files on the pod: `/workspace/tune/jobs/<rung>.pyspy.txt` (collapsed
stacks, about 2-3 MB), `<rung>.pyspy.log`, `<rung>.gpu.csv`, next to
`<rung>.json` and `<rung>.log`. `rp pull` copies `tune/jobs/` and
`tune/host.json` to `data/runpod_pull/<podId>/tune/` (videos excluded). A
re-summarised report also works offline from those files.

**Overhead.** In its default blocking mode, py-spy pauses the server for
each sample while it walks every thread's stack. Measured locally (py-spy
0.4.2, CPU-bound loop, 31 threads, 40-frame stack), the slowdown was:

| mode | slowdown |
|---|---|
| 20 Hz | about 1.5-2% |
| 100 Hz | about 8-10% |
| `--nonblocking` | under 1% |

Expect about 2% on a rung at 20 Hz, which is within run-to-run noise. Use
`--no-profile` for a clean wall-time A/B. nvidia-smi at 1 Hz costs nothing
measurable. py-spy needs ptrace on the server process. On Nov 12 it worked as
root in the RunPod container, but `py-spy dump` once returned "Permission
denied". If `pyspy_error` shows that, the GPU half of the profile is still
recorded.

### The ladder (cumulative; `profiles.py` has the full rationale)

| rung | change | why |
|---|---|---|
| R0 | repo defaults: 5 fps, corner stride 3, 200-frame SAM3 chunks | what :8004 runs on the 3090 (qm8 about 8 min there) |
| R1 | corner_track_stride 2 | Oct 31 validated default. The Nov 6 "B3" stride 3 lost auto coverage on 4/4 held-out matches. |
| R2 | sam3_track_fps 8 | Pre-B3 rate (held-out agg meanNN 0.47 vs 0.73 m). Also the rate selective_mask_prop was tuned at, since its frame-denominated knobs are not fps-rescaled. |
| R3 | sam3_chunk_frames = tier cap | Fewer SAM3 chunk boundaries. Each boundary drops appearance memory and does not re-seed occluded objects. VRAM-bound; untested above 400. |
| R4 | sam3_track_fps 10 | Denser evidence through crossings. No A/B evidence above 8 fps. |
| R5 | corner_track_stride 1 | Last: the Oct 31 A/B found no identity gain over stride 2 at about 2x the corner cost. |

Not on the ladder: `sam3_export_scale` (the code says export resolution is not
a lever because SAM resizes to a fixed square, and the Nov 11 coverage study
agrees), `sam3_max_gpu_frames` (can only lower the tracked frame rate),
`sam3_multiplex` and `sam3_compile` (A/B'd slower or no win).
`selective_mask_prop` and the output-identical SAM3 speed levers stay ON.

### VRAM tiers (`profiles.py`, from `nvidia-smi memory.total`)

| tier | cards | R3 chunk cap |
|---|---|---|
| t48 | A40, A6000, L40S, RTX 6000 Ada (48 GB) | 600 |
| t80 | A100 80GB, H100, H100 NVL, RTX PRO 6000 (80-96 GB) | 1000 |
| t141 | H200 | 2000 (a whole full-match view in one session) |

At about 14 MB per frame of 1008² video plus about 4 GB of model, the caps
keep the video window under ~30% of the card so per-object tracker state has
room. Every tier also gets `BASE_ENV`: the `web/start_all.sh` exports plus
`VLLM_PHASE_EXCLUSIVE=0`. There is no vLLM on the pod, and that knob alone
costs a 5 s sleep per job. That setting does not change output.

## 5. During the event

* Process each match as its video appears (UI through the tunnel). Keep one
  job at a time: the GPU is serial.
* **YouTube from a datacenter IP** often hits "Sign in to confirm you're not
  a bot". Fallback: export a logged-in browser's cookies to `cookies.txt`
  (Netscape format, for example with a "Get cookies.txt" extension) and run
  `rp secrets --cookies cookies.txt --restart`. That sets
  `YTDLP_COOKIES_FILE`, and every yt-dlp call in the app then passes it as
  `cookiefile`. Last resort: download the video locally and upload the file
  in the UI.
* `rp pull` regularly (and **always before terminate**). It copies a
  consistent DB snapshot, new replays (`--with-video` for the HUD mp4s), and
  the event-state files (calibrations made at the event), plus the tune job
  results and raw profiles (`tune/jobs/`, `tune/host.json`), into
  `data/runpod_pull/<podId>/`. It never touches your local `frc_scorer.db` or
  `web/replays`. Scouting metrics are computed from replays, so pulling
  replays and the DB is enough.

## Vast.ai (single host)

For benchmarking on a high-clock desktop CPU (RTX 5090 / 4090 hosts on Ryzen
/ Core CPUs), `up` can rent ONE vast.ai instance instead of a RunPod pod. The
CLI reads `VASTAI_API` from the main checkout's `.env` (it never goes to the
instance). Code: `deploy/runpod/vast.py`.

```bash
rp vast-offers                                  # read-only: 1x RTX 5090/4090, verified, CUDA>=12.8,
                                                # direct ssh port, >=120 GB disk, reliability>=0.97,
                                                # >=8 effective cores; sorted by cpu_ghz desc, then $/h
rp vast-offers --min-ghz 5 --max-dph 0.8 --gpu "RTX 5090"   # --include-unverified, --min-cores, --json
rp up --provider vast --offer 1234567 --dry-run --secrets-from-env   # exact create body; rents NOTHING
rp up --provider vast --offer 1234567           # type "rent"; --disk-gb (default 120)
```

`up --provider vast` checks that `~/.ssh/id_ed25519.pub` (or
`$RUNPOD_SSH_KEY.pub`) is registered on the vast account (vast injects the
ACCOUNT's ssh keys into the container; there is no per-instance key in the
create body — register it under console > Keys first, `--skip-ssh-key-check`
to override), shows the offer and your credit, asks for `rent`, then
`PUT /api/v0/asks/<offer>/` with the image below, `runtype "ssh_direc
ssh_proxy"` and an empty `env`. The default image is vast's own
`vastai/base-image:@vastai-automatic-tag`: the RunPod image has no ssh host keys,
so under vast's launcher sshd exits ("no hostkeys available") and port 22
refuses (seen Nov 14 2026; an onstart `ssh-keygen -A` is also set as a backstop).
After that it is the same flow as RunPod:
wait for ssh (direct IP:port, falling back to vast's ssh proxy), ship the
bundle, install secrets over ssh, bootstrap. `.state.json` gets
`"provider": "vast"`, and `tune`, `pull`, `status`, `ssh`, `logs`, `tunnel`,
`push`, `stop` and `terminate` (= destroy) then work exactly as usual.

Differences from RunPod:

* Measured Nov 14 2026 (RTX 5090 + Core Ultra 9 285K, $0.77/h): ssh in 43 s,
  bootstrap ~5 min, qm8 R0 5.1 / R1 5.4 / R2 7.1 min vs 5.8 / 6.0 / 7.4 on the
  RunPod EPYC 9554 pod (single-thread Python +43%). A whole night (tune +
  27 GT-scored sweep jobs) cost ~$4.30.
* Container RAM is capped by the host's cgroup (127.6 GiB on that box), not by
  the GPU. Sweep config Q2 (`sam3_chunk_frames` 1000 + `selective_mask_prop`
  off) was OOM-killed by it on a 2:40 match (`oom_kill 1` in
  `/sys/fs/cgroup/memory.events`; the server just disappears, no traceback).
  Check `cpu_ram` on the offer, or keep chunks smaller, for full-tracking arms.
* There is no separate volume: `/workspace` is a directory on the container
  disk (`--disk-gb`), which survives `stop`/start (billed for storage while
  stopped) and is deleted by `terminate`.
* `status` shows the host CPU, `$`/h, an uptime-based cost estimate and your
  account credit. Bandwidth is billed separately per TB (`vast-offers` shows
  the host's `$`/TB down/up; a first boot downloads the venv, SAM3 and the vLLM reader, tens of GB).
* A stopped instance can only restart on the same host; if its GPU was
  rented meanwhile, start waits. `pull`, `terminate`, rent another offer.
* The vLLM reader (tune arm Q) needs a CUDA 13 driver: `vast-offers --min-cuda 13`.
* **Marketplace hosts are third-party machines.** The host operator can read
  the container, including `/workspace/frc/.env` (Jev, HF, TBA keys) and the
  DB. Use keys you can rotate, prefer verified hosts, and `terminate` when done.

## Swarm: batch-process a past event

One pod per job, many cheap pods. On Nov 13 an RTX PRO 6000 pod ran a full
qm8 job in about 6-7.4 min with the GPU only ~27% busy and ~3.5 CPU cores in
use. Two jobs in one server process gave only 1.2x. The pipeline is bound by
serial Python, so a bigger GPU barely helps. To go faster, run more servers
with one job each. `swarm` (`deploy/runpod/swarm.py`) drives N ordinary pods
(same bundle, secrets and bootstrap as `up`) from one state file:
`deploy/runpod/.swarm/<name>.json` (gitignored), with member logs in
`deploy/runpod/.swarm/<name>/<podId>.log`. It never reads or writes the
single-pod `.state.json`, so a weekend pod and a swarm can coexist.

**GPU choice.** Cheap 48 GB cards are enough: L40S, RTX A6000, A40 and RTX 6000
Ada, in that default priority (`--gpu` to override, repeatable). Each pod runs
one job at a time, and the t48 tier has room for it. Ask for vCPUs
(`--min-vcpu`, default 8), because the job's CPU work is the bottleneck.
Members skip the ~9 GB vLLM reader venv (`bootstrap.sh --no-vllm`), since
the production reader is Jev. Pass `--with-vllm` if you need it. The default
volume is 80 GB: about 15 GB of venv and caches plus about 116 MB of video
per match.

**Cost math** (secure prices from the table above): 8 x A40 at $0.59/h =
$4.72/h. An 80-match event at about 7 min per match is about 560 pod-minutes,
so 8 pods take about 70 min plus about 15 min of bootstrap, about 1.4 h x
$4.72 ≈ **$7**. One RTX PRO 6000 would take 9+ h ≈ $25. Bootstrap time
is billed on every pod. `--terminate-when-done` stops each pod's billing as
soon as it has nothing left to do.

```bash
rp swarm plan --name cur --event 2026cur --download            # FIRST, before renting: schedule order; videos fetched HERE (home IP)
rp swarm up --name cur --count 8 --dry-run --secrets-from-env  # bodies, bundle manifest, $/h; creates NOTHING
rp swarm up --name cur --count 8 --secrets-from-env            # ONE "rent" prompt for all 8; parallel setup
rp swarm run --name cur --terminate-when-done                  # dispatch; Ctrl-C and re-run re-attaches
rp swarm status --name cur                                     # members, RunPod state, $ so far, progress
rp swarm pull --name cur                                       # (only if not terminated by run)
rp swarm import --name cur                                     # DRY RUN: what would be added locally
rp swarm import --name cur --apply                             # backup DB, then merge (--render: HUD videos)
rp swarm terminate --name cur                                  # whatever is left (refuses un-pulled pods)
```

* **up**: creates the pods and keeps the ones it got if some fail (no stock).
  It builds the bundle once (code + weights + a snapshot of the local DB +
  event state, the same as `up`) and gathers the secrets once. Then every
  member, in parallel: wait for ssh, ship, install secrets, bootstrap. Re-running
  resumes members that are not `ready`. `--count K` tops the swarm up to K
  live pods and `--add K` grows it. The bundle is uploaded from this machine
  once per pod (about 450 MB each).
* **plan**: matches come from the local DB `matches` table in schedule order
  (quals, then playoffs by set). `--matches` takes `all` (default), `qm`,
  `playoffs`, `sf`, a key `qm8` / `sf1m1` / `f1m2`, a range `qm1-qm40`, or
  a comma list of these. The video for a match is `<match_key>.mp4` from
  `--videos DIR`, then `data/videos`, then `data/test_footage`. With
  `--download`, missing videos are fetched via TBA video key + yt-dlp into
  `data/videos/` (`--cache-dir`) **from this machine**: YouTube blocks
  datacenter IPs as bots. Plan (and download) **before** `swarm up`: `plan` creates the
  swarm state without any pods, so nothing bills while videos download.
  Matches that already have a complete replay (local DB, a previous
  `data/runpod_pull/*` pull, or this swarm's run) are skipped unless you pass
  `--redo`.
* **run**: each pod gets a **contiguous block** of the schedule. The app's
  camera-move auto-recalibration and per-match corner calibration carry
  state from one match to the next on a server, so round-robin would hurt.
  An idle pod **steals** the tail half of the largest remaining block, or
  all of a dead pod's block. A failed match is retried once, on another pod
  if there is one. Per pod, the run uploads the video to
  `/workspace/batch/videos/` (skipped if the same size is already there),
  launches `pod_job.py` detached exactly like `tune` (out
  `/workspace/batch/jobs/<match_key>.a<attempt>.json`, one file per attempt,
  no profiler), and uploads the pod's next video while the job runs.
  Only one `swarm run` can drive a swarm at a time (`.swarm/<name>.run.lock`).
  The launch is guarded on the pod (`mkdir <out>.launch`), so retrying it
  after a dropped ssh session never starts a second driver. Before
  launching, the run asks the server for its active job, and an ssh failure
  counts as "unknown", never as "idle". Failures are handled by kind:
  * ssh failures (exit 255 or timeout) retry without limit. After 10
    minutes of them the RunPod API is asked about the pod. Only an API
    answer of "not RUNNING" or "gone" marks the member `lost`; if the API
    can't be reached, the run keeps retrying.
  * A remote command that really fails (disk full, missing venv) fails the
    match after 3 tries, so the pod doesn't retry it forever while billing.
  * A job counts as "wrote nothing" only when it has never written a state
    file for 5 minutes of reachable polling and its driver isn't running.

  `--config-json` sets knobs for every
  job. `--pod-config POD_ID=JSON` overrides knobs on one pod, for example
  for a quality sweep. With `--terminate-when-done`, a pod with nothing left
  to do and nothing to steal is pulled and then terminated (the flag is the
  confirmation). The assignment is a policy object (`--policy blocks`;
  `initial` / `place` / `pick` in `swarm.py`). A run can also grow while it
  is going (`swarm.add_items`, `run["open"]`), so a live staggered policy
  with a TBA arrival poller can plug in without changing the run loop.
* **pull**: `pull` for every member, in parallel, into the usual
  `data/runpod_pull/<podId>/`. It records which run jobs each pull covered.
  This includes `lost` members; an ssh failure there is reported per member.
  The same rules now apply to the single-pod `pull`:
  * A pull fails, and marks nothing, when the tar stream is truncated or
    ssh/tar exits non-zero.
  * A replay dir counts as pulled only once it has a `.pull_complete` marker.
    The marker is written after a clean extract, only for replays whose DB
    row was in the pulled snapshot. With `--with-video`, a replay whose HUD
    render is still queued or rendering gets no marker.
  * Replays without the marker are pulled again next time. That includes
    every replay pulled before this change: it is re-pulled once while its
    pod exists.
  * `import` and `terminate` only trust marked dirs.
* **import**: merges the pulled replays into the local app. The default is a
  dry run. `--apply` refuses while local :8004 reports an active job. It
  backs up `frc_scorer.db` to `frc_scorer.db.bak-swarm-<name>-<time>` first,
  then copies each replay dir to `web/replays/<id>/` and inserts its rows in
  one transaction. Rows come from `replays` plus every per-job table a job
  writes: `match_robot_identity`, `match_robot_crop`,
  `match_robot_identity_segment`, `match_robot_binding`,
  `appearance_gallery` and `replay_slot_corrections`. Only `complete`
  replays from marked pulls are imported. Paths are rewritten to
  the local layout, and the source video points at the local copy. An id
  that already exists is never overwritten (skipped and reported).
  `event_anchors` (event_lineup knob, off by default; tiles not pulled) is
  only reported. HUD mp4s are not pulled by default. An imported replay
  without one is set to `render_status=queued`, so the local server renders
  it from the local video (restart :8004, or pass `--render`). Calibration
  and event-state files from the pods are **not** merged: import lists the
  ones that differ from local.
* **stop / terminate**: one typed confirmation for every member (`stop swarm
  <name>`, `terminate swarm <name>`). `terminate` refuses a member whose last
  finished job is not covered by a pull, or which still has a job running,
  unless you pass `--force`. `stop` and the cost figures include `lost`
  members, which may still be billing. A re-run of `swarm run` puts a lost
  member back to work when the API reports it RUNNING again. For a lost pod
  that is stopped, start it in the RunPod console and `swarm pull` before
  terminating; `--force` deletes its unpulled results.

**Google Drive upload for remote runs.** Pods have no rclone and no Drive
credentials, and `GDRIVE_*` vars are never shipped to them (the
`gdrive_upload` knob is a no-op on a pod). The knobs still travel with the
results: each replay's `meta.json` records its job's knobs, so `import` uploads
from this machine instead:

```bash
rp swarm run --name cur --config-json '{"gdrive_upload": true, "gdrive_upload_video": true}'
rp swarm pull --name cur
rp swarm import --name cur                    # dry run: also lists the Drive uploads
rp swarm import --name cur --apply --render   # merge, then upload; :8004 uploads videos after rendering
```

After `--apply`, every imported `complete` match whose job had `gdrive_upload`
goes up in one `scripts/upload_gdrive.py <event> --match ...` call per event
(match files, then the event export once). Videos: a HUD mp4 that was pulled
(`pull --with-video`) goes up in the same call. A video whose render is queued
goes up from :8004 once it renders (its render hook re-reads the knobs), so
pass `--render` or restart :8004. `--gdrive` uploads every imported match
whatever its knobs; `--gdrive-video` also uploads videos. A failed upload makes
`import` exit 1 but never undoes the import; re-send with
`scripts/upload_gdrive.py`. `run` prints a reminder when its config has
`gdrive_upload`.

Caveats:

* In batch and sweep runs each pod's calibration state is independent (live mode syncs it). Each pod starts from the
  local event state, and calibrations it re-solves (camera moved) stay on
  that pod.
* The upload bandwidth from this machine is shared by all pods: the bundle
  once per pod, then about 116 MB per match.
* **Untested on a real swarm yet.** The dispatcher, import and plan are
  unit-tested offline with fakes. The first real run should be small
  (`--count 2 --matches qm1-qm6`).

**Item keys and the local backend (shared by batch, sweep and live).** A batch
run's items are keyed by match key, as before. A sweep item is keyed
`<config_id>/<match_key>` (`Q3/2026cur_qm8`), and a live re-posted video is
`<match_key>@r2`. Each item stores its real `match_key` (plus `config_id`,
`knobs`, `event` when it has them). Items in state files written before this
have none of these fields and fall back to the key, so old `.swarm/*.json`
files still load and resume. Per-job knobs are the run config, then the
item's knobs, then `--pod-config`. On a pod, job files use the key with `/`
replaced by `__` (`/workspace/batch/jobs/Q3__2026cur_qm8.a1.json`). Video
uploads are per match, so every config of one match shares one upload.

* `--restart-each` (any dispatching command) restarts the member's app
  server before every job (`serve.sh restart`, which waits for :8004). The
  app leaks about 0.7 GB of CUDA tensors per job, and after a few jobs
  tracking changes, so without a restart job order confounds a config
  comparison. It is on by default for sweeps on pods and off otherwise.
* `--local` (on `run`, `sweep run` and `live`) dispatches to the local app
  at `http://127.0.0.1:8004` instead of pods. `pod_job.py` runs here as a
  subprocess (`--no-profile`, `--app-dir` = the main checkout, whose DB and
  `web/replays` the main-server worktree symlinks), with the same busy
  guard and launch-once guard. Results stay in the local DB, so nothing is
  pulled or imported. A local `--restart-each` kills the pid in
  `/tmp/frc-vision-dev.pid` and runs `setsid -f bash
  .worktrees/main-server/web/start_all.sh`. It is off by default and the
  run prints a warning that job order may confound configs. Use `--local`
  to exercise any of this without renting.

## Quality sweep: pick the max-quality config

Plan: `docs/max-quality-swarm-plan-2026-11.md`. Each config gets one pod (the
`pinned` policy: config i runs on pod i). Pods differ in speed and host, so
this keeps those differences out of every within-config comparison. With
fewer pods than configs, a pod that finishes takes the next whole config.
A failed item is retried on its own config's pod. If a pod is lost, the
next pod that runs dry takes over its whole remaining config.

```bash
rp swarm sweep evalset --out data/eval/sweep_evalset.json --auto            # every hand-GT match (+ --add qm keys)
#   edit the file if needed: it is plain JSON (video path, truths per match)
rp swarm sweep plan --name qs --evalset data/eval/sweep_evalset.json \
                    --configs deploy/runpod/sweep_configs.example.json       # validates every knob; NO renting
rp swarm up --name qs --count 9 --dry-run --secrets-from-env                 # $/h for one pod per config
rp swarm up --name qs --count 9 --secrets-from-env [--with-vllm]             # RENTS (needs Nick's go-ahead); --with-vllm for Q4
rp swarm sweep run --name qs --terminate-when-done                           # restart before each job by default
rp swarm pull --name qs                                                      # (if not terminated by run)
rp swarm sweep score --name qs                                               # local scoring, cached per item
rp swarm sweep score --name qs --audit                                       # + Jev audit for no-GT matches (~$0.25 each; asks)
rp swarm sweep report --name qs                                              # ranking + noise band -> data/runpod_pull/sweeps/qs/

# the same machinery with nothing rented (local :8004, one job at a time):
rp swarm sweep plan --name qsl --evalset ... --configs ...
rp swarm sweep run --name qsl --local [--restart-each]
rp swarm sweep score --name qsl && rp swarm sweep report --name qsl
```

* **evalset**: `--auto` takes every match with hand GT in
  `data/eval/ground_truth/<match_key>/`, so new GT matches from other events
  and camera setups join on the next refresh. `--add MATCH_KEY` adds a match
  that has no GT. The video is found by the GT docs' video fingerprint
  (`web.field.ground_truth.video_fingerprint`; only same-size files in
  `web/uploads`, `data/videos` and `data/test_footage` are hashed), then by
  the GT doc's file name, then by `<match_key>.mp4`. Matches without a video
  are listed under `missing`. Per match the file records its truths:
  `hand_gt`, `manta` (if `data/manta_cache/<event>_qual_<n>.json` exists), or
  `none`. Re-running keeps entries you edited by hand.
* **configs**: `{"base": {...}, "baselines": ["Q0a","Q0b"], "configs": [{"id",
  "knobs"}]}`. See `sweep_configs.example.json`: Q0a/Q0b/Q1-Q6, with base =
  R1 (5 fps, corner stride 2, full match). Names and values are checked
  against `web/job_config.py`'s Knob registry (typos get a did-you-mean).
  Run the CLI with the main venv for value checks; other interpreters check
  names only. A config with `team_reader: vllm` gets its vLLM reader
  pre-started on its pod before each job (`vllm_ctl.py start`), so the cold
  start stays out of the measured wall time. That needs `swarm up
  --with-vllm`.
* **score**: each finished item is scored with `score_pulled.py` under the main
  venv: hand-GT coverage, error, labels right/wrong/unlabelled and misID s
  (period `auto` for `auto_period_only` configs), plus the Manta auto meanNN
  where cached. Pod items need a marked pull (`swarm pull`); local items are
  read from the local DB. Scores are cached in
  `data/runpod_pull/sweeps/<name>/scores/` and re-scored only when the job id
  changes (or with `--rescore`). `--audit` also runs the Jev label audit
  (`scripts/audit_replay.py` sample + jev + score) for matches **without**
  hand GT. It prints the estimate (~$0.25 per replay) and asks first.
* **report**: identity comes first. primary = right − 5 × wrong, in points of
  hand-GT robot-time, pooled over matches and weighted by each match's
  processed GT samples. Ties go to misID s per match, then coverage,
  position error median and Manta auto meanNN. The noise band for each
  metric is the spread between the baseline repeats (`--baselines`, default
  from the configs file). Verdicts:
  * **WIN** only if primary beats the *better* baseline by more than that
    spread.
  * **LOSS** if primary is below the *worse* baseline by more than the
    spread.
  * **noise** otherwise.
  * **VETOED** if it would win but coverage, position error or Manta
    regressed beyond max(spread, 3 pts / 5 cm / 10 cm) against the worse
    baseline.
  * **PARTIAL** if its GT matches differ from the baseline's (a failed
    item).

  With fewer than two baseline results the band is unknown: a config that beats
  the baseline shows as `better?`, never WIN (the report says so). The report also has a per-match table and mean wall time and peak
  VRAM per config. Outputs: `report.txt`, `report.json` and `report.md`.
* **Cost** (plan): about 9 pods × about 2.5 h × $2.69/h ≈ $60 on 96 GB cards. A
  restart adds about 1-1.5 min per job, and a Jev audit about $0.25 per
  no-GT replay. Nothing is rented by `evalset` / `plan` / `score` / `report`.
* `event_lineup` (Q5) learns from earlier matches on the same server. A
  sweep pod sees only the eval matches of its own config, in eval-set
  order.

## Live mode: staggered dispatch at an event

```bash
rp swarm up --name ev --count 5 --secrets-from-env                     # RENTS: ceil(T_match / cadence) + 1 pods
rp swarm live --name ev --event 2026xyz --auto-import                  # TBA poll every 60 s; Ctrl-C and re-run re-attaches
rp swarm live --name ev --event 2026xyz --hours 9                      # stop polling after 9 h, then drain
rp swarm live --name ev --event 2026xyz --hours 9 --terminate-when-done # ... then pull + terminate each pod
rp swarm live --name ev --event 2026xyz --reprocess-replayed           # a re-posted (replayed) match runs again as <mk>@r2

# rehearsal of a PAST event at its real cadence (phase 3); --local = no renting, tests the machinery
rp swarm live --name reh --event 2026cur --rehearse --speed 1 --matches qm1-qm30
rp swarm live --name rehl --event 2026cur --rehearse --speed 10 --matches qm1-qm6 --local
```

* **Arrivals**: TBA `/event/{key}/matches` (stdlib urllib, `TBA_API_KEY` from
  the main checkout `.env`), polled every `--poll-s` (default 60). A match
  with a YouTube video that has not been seen yet is downloaded **here**
  (`data/videos/`, home IP) and added to the open run. A failed download is
  tried again on the next poll. When a known match gets a new video id
  (a replay), it is logged and ignored unless you pass
  `--reprocess-replayed`.
* **Next day**: a run closed by `--hours` is finished, so the next `swarm
  live` for the same event starts a new run. That run takes over the
  earlier live runs' seen matches, and it never queues a match that already
  has a result: a done item of any earlier run of this swarm, or a complete
  replay in the local DB or a previous pull. Only a new video id of such a
  match counts, and only as a replay (`--reprocess-replayed`).
* **Billing at exit**: without `--terminate-when-done` the pods keep
  running when live exits. The last line says how many pods are still
  billing and at what $/h, with the `swarm pull` / `swarm terminate`
  commands. With the flag, each pod is pulled and then terminated once
  arrivals have stopped (`--hours` expired, or the rehearsal ended) and its
  queue is empty. The safety rules are the same as `swarm run`: a pod is
  never terminated while its last job is unpulled.
* **Dispatch** (`live` policy): the new match goes to the pod that has been
  idle longest. It never queues two matches on one pod while another pod is
  idle. When every pod is busy, nothing is pre-assigned: the pod that frees
  first takes the match. The pod predicted to free first (elapsed time vs
  median job time) gets the waiting video prefetched. Retries go before new
  arrivals. `--restart-each` works here too (off by default).
* **Calibration sync**: after each job, the dispatcher lists the pod's
  `calibration_*`, `composite_layout_*`, `zones_*` and `field_season_*`
  files (sha1). It fetches the ones that changed since the starting state
  (the local files the bundle shipped) and pushes them to every other ready
  pod that lacks that version. Each push untars into a temp dir in
  `web/uploads` and then `mv`s every file into place. A pod added mid-event
  gets the backlog; a failed push is retried after the next job. The last
  writer wins on a file two pods both changed. `clockscan_` caches are
  never synced. Every sync is logged in the run (`live.sync_log`) and
  printed.
* **Latency**: per match the run records video seen → downloaded →
  dispatched → launched → final replay. The live table shows p50 / p90 /
  max latency, the wait before dispatch, the pod predicted to free next and
  per-pod utilization. At the end it writes
  `data/runpod_pull/live/<name>/latency.json`.
* **`--auto-import`** pulls each finished match's pod and imports, with
  the `swarm import --apply` rules, only the replays of **this live run's**
  finished jobs. Replays of earlier batch, sweep or rehearsal runs on the
  same pods, and failed attempts, are never imported. The DB is backed up
  **once per live session**. HUD videos are not pulled, so after each
  import the dispatcher asks the local :8004 to render them (the same
  request as `swarm import --render`) and logs its answer. If :8004 is
  down, the replay stays `render_status=queued` until the app's startup
  sweep renders it. While local :8004 runs a job, the import waits until
  the next finished match. It is ignored with `--local`.
* **Rehearsal**: arrival time = now + (TBA `actual_time` − first match's) /
  `--speed`. Videos come from this swarm's earlier plan / runs, then
  `--videos DIR`, then `data/videos` / `data/test_footage` as
  `<match_key>.mp4`. Matches without a local video are skipped and listed.
  Everything else is identical to live. The run closes once the last match
  has arrived and finished.
* **Cost** (plan): 5 × L40S ≈ $5.5/h, about $150 for a 3-day event (about 9
  h/day); 5 × RTX PRO 6000 ≈ $13.5/h. Polling and rehearsing with `--local`
  rent nothing.

## 6. Shut down

```bash
rp stop          # type "stop": GPU billing stops; /workspace (venv, caches, DB, replays) is kept
rp up            # later: offers to start the same pod again; bootstrap re-runs (apt only)
rp pull && rp terminate   # type "terminate <podId>": pod + volume deleted
```

A stopped pod keeps its volume (billed for storage) but may not get its GPU
back on restart if the host is full. If that happens, `pull`, `terminate`,
and run `up` again.

## What lives where (pod)

| path | survives stop/start | content |
|---|---|---|
| /workspace/frc | yes | code, weights, `.venv`, `.env`, `profile.env`, `frc_scorer.db`, `web/uploads`, `web/replays` |
| /workspace/cache | yes | HF (SAM3, DINOv3), EasyOCR, uv, torch caches |
| /workspace/tools | yes | uv, deno, uv-managed Python |
| /workspace/logs | yes | `server.log`, `server.pid`, `bootstrap.log` |
| /workspace/tune | yes | test video, `host.json`, `jobs/R<n>.json` + logs + profiles (`.pyspy.txt`, `.gpu.csv`) |
| apt packages, /root | no | reinstalled by bootstrap |

## Troubleshooting

* **SSH never becomes ready**: `rp status` (does the pod have a public IP and
  a `22` port mapping?). Community cloud needs `supportPublicIp` (set). Check
  the pod's logs in the RunPod console. The image's sshd reads `PUBLIC_KEY`
  from `~/.ssh/id_ed25519.pub` (`RUNPOD_SSH_KEY=/path/key` to override).
* **SAM3 401/403 during prefetch**: the `HF_TOKEN` account has not been
  granted access to `facebook/sam3` (gated). Request access on
  huggingface.co, then `rp secrets --restart` and
  `rp ssh -- bash /workspace/frc/deploy/runpod/bootstrap.sh`. Emergency
  fallback: `TRACKING_BACKEND=sam2` in `profile.env` (ships the SAM2 small
  checkpoint; lower quality).
* **"no kernels for sm_XX" / CUDA not available**: the host driver is older
  than CUDA 12.8 (`allowedCudaVersions` should prevent this) or a non-cu128
  torch got installed. Check `rp ssh -- nvidia-smi` and the venv:
  `/workspace/frc/.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
* **Event load fails**: `TBA_API_KEY` missing or wrong (`rp secrets`).
* **Jobs run but no team labels**: `JEV_TYPESAFE_API_TOKEN` missing or wrong.
  The server log shows Jev errors.
* **`up` keeps failing to find a GPU**: widen `--gpu` (repeatable), try
  `--cloud COMMUNITY`, or lower `--min-vcpu` / `--min-ram`.

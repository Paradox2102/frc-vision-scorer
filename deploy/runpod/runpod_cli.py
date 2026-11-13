#!/usr/bin/env python3
"""RunPod deployment CLI for the FRC vision scorer (stdlib only).

Runs locally with the main checkout's .venv/bin/python or any system python3.
See deploy/runpod/README.md for the weekend runbook.

    runpod_cli.py gpus                         # prices / VRAM / stock (read-only)
    runpod_cli.py up --gpu "NVIDIA H100 80GB HBM3" [--dry-run] [--secrets-from-env]
    runpod_cli.py tunnel                       # http://localhost:8014 -> pod :8004
    runpod_cli.py status | logs | ssh | restart
    runpod_cli.py tune [--start R0] [--budget-min 8] [--apply R2]
    runpod_cli.py push [--ref main]            # re-ship code, restart server
    runpod_cli.py secrets                      # re-prompt + install pod .env
    runpod_cli.py pull                         # DB + replays -> data/runpod_pull/<podId>/
    runpod_cli.py stop | terminate             # typed confirmation

State (pod id, ssh endpoint) lives in deploy/runpod/.state.json (gitignored).
Secrets go to the pod ONLY over ssh into /workspace/frc/.env (chmod 600) —
never through RunPod's pod `env` field (stored in RunPod metadata) and never
RUNPOD_API / CLOUDFLARE_* / ROBOFLOW_API.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import io
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import profiles  # noqa: E402

STATE_PATH = HERE / ".state.json"
TUNE_DIR = HERE / "tune_results"
PULL_ROOT = REPO / "data" / "runpod_pull"

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
UA = "frc-runpod-cli/1.0"

POD_WS = "/workspace"
POD_APP = f"{POD_WS}/frc"
POD_TUNE = f"{POD_WS}/tune"
POD_LOG = f"{POD_WS}/logs/server.log"
POD_PORT = 8004                     # app port on the pod (serve.sh)

# runpod/pytorch: Ubuntu 24.04 (Python 3.12), CUDA 12.8.1, sshd + PUBLIC_KEY
# handling preconfigured. Its own torch is unused (the app venv pins torch
# 2.10.0+cu128); the tag is a stable (non-rc) release from Oct 30 2026.
DEFAULT_IMAGE = "runpod/pytorch:1.5.1-cu1281-torch291-ubuntu2404"
# Priority order for `up` when --gpu is not given (gpuTypePriority=custom).
DEFAULT_GPUS = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA H100 PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
]
ALLOWED_CUDA = ["12.8", "12.9", "13.0", "13.1", "13.2"]


def needs_cuda_filter(gpus) -> bool:
    """cu128 torch needs a CUDA>=12.8 host driver. Every Blackwell host already
    has one, and RunPod's allowedCudaVersions filter can hide real stock (Nov 12:
    "no instances" for RTX PRO 6000 with 10 DCs in stock), so only filter when a
    pre-Blackwell card is in the priority list."""
    return any("Blackwell" not in g and "B200" not in g for g in gpus)
DEFAULT_TUNNEL_PORT = 8014
QM8_VIDEO_NAME = "ff4be1d9-b8e5-49ff-ac78-4a6a0e77a697.mp4"   # cached 2026cur qm8 (web/uploads)
TUNE_MATCH_KEY = "2026cur_qm8"
TUNE_EVENT = "2026cur"
TUNE_MATCH_NUMBER = 8

# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SecretSpec:
    name: str
    required: bool
    why: str


# Everything the server can read from .env (grep os.environ / getenv over web/
# and src/, Nov 12 2026). Non-default reader paths are optional.
SECRET_SPECS = (
    SecretSpec("JEV_TYPESAFE_API_TOKEN", True,
               "team_reader=jev (default) + window-label fallback reads"),
    SecretSpec("HF_TOKEN", True,
               "facebook/sam3 checkpoint is gated on Hugging Face"),
    SecretSpec("TBA_API_KEY", True,
               "event load (rosters, match list, YouTube keys) from The Blue Alliance"),
    SecretSpec("ANTHROPIC_API_KEY", False,
               "only team_reader=claude (legacy, not default)"),
    SecretSpec("DEEPSEEK_API", False,
               "only team_reader=deepseek (not default)"),
)
# Never shipped to the pod, whatever the caller passes.
NEVER_SEND_EXACT = {"RUNPOD_API", "ROBOFLOW_API", "MANTA_API"}
NEVER_SEND_PREFIXES = ("CLOUDFLARE_", "RUNPOD_")


def never_send(name: str) -> bool:
    return name in NEVER_SEND_EXACT or name.startswith(NEVER_SEND_PREFIXES)


def mask(v: str | None) -> str:
    if not v:
        return "(unset)"
    return "****" + v[-4:] if len(v) > 8 else "****"


def parse_dotenv(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


def local_env() -> dict:
    """The main checkout's .env (worktrees have none)."""
    for p in (main_checkout() / ".env", REPO / ".env"):
        if p.exists():
            return parse_dotenv(p.read_text())
    return {}


def select_secrets(local: dict, *, interactive: bool, input_fn=input,
                   specs=SECRET_SPECS, out=print) -> dict:
    """Decide what goes into the pod .env.

    Non-interactive: every spec present in ``local``. Interactive: per var,
    Enter = keep the local value, ``-`` = skip, anything else = new value.
    Required vars missing at the end raise. Never returns a never_send name."""
    chosen: dict = {}
    for s in specs:
        if never_send(s.name):
            continue
        cur = local.get(s.name, "")
        if not interactive:
            if cur:
                chosen[s.name] = cur
            continue
        tag = "required" if s.required else "optional"
        prompt = (f"{s.name} ({tag}: {s.why}) [local .env: {mask(cur)}] "
                  f"Enter=keep / paste new / '-'=skip: ")
        ans = input_fn(prompt).strip()
        if ans == "-":
            continue
        if ans:
            chosen[s.name] = ans
        elif cur:
            chosen[s.name] = cur
    missing = [s.name for s in specs if s.required and s.name not in chosen]
    if missing:
        raise SystemExit(f"required secrets missing: {', '.join(missing)}")
    for k in list(chosen):
        if never_send(k):
            del chosen[k]
    return chosen


def render_dotenv(secrets: dict) -> str:
    lines = ["# Installed by deploy/runpod/runpod_cli.py — pod-only secrets (chmod 600)."]
    for k in sorted(secrets):
        if never_send(k):
            raise ValueError(f"refusing to write {k}")
        v = secrets[k]
        if "\n" in v:
            raise ValueError(f"{k}: newline in value")
        if all(c.isalnum() or c in "-_.:/+=@,~" for c in v):
            lines.append(f"{k}={v}")
        elif "'" not in v:
            lines.append(f"{k}='{v}'")
        else:
            raise ValueError(f"{k}: value contains a single quote; set it by hand")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Paths / git
# --------------------------------------------------------------------------
def main_checkout(repo: Path = REPO) -> Path:
    """The main checkout (weights, data, .env live there even in worktrees —
    same rule as web/students/paths.py)."""
    try:
        common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        p = Path(common)
        if not p.is_absolute():
            p = (repo / p).resolve()
        return p.parent
    except Exception:
        return repo


def git(*args, repo: Path = REPO, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode().strip()


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------
# Tracked paths shipped from `git archive <ref>`.
CODE_PATHS = ("web", "src", "data/field", "pyproject.toml", "deploy/runpod")
# ...minus these (prefix match on the archive path).
CODE_EXCLUDE_PREFIXES = (
    "web/uploads/", "web/replays/", "web/logs/",
    "data/field/reference_frames/",         # docs screenshots, never read
    "deploy/runpod/tune_results/",
)
CODE_EXCLUDE_SUFFIXES = (".step",)          # 35 MB CAD source of the field model


@dataclass(frozen=True)
class WeightSpec:
    rel: str            # path relative to the main checkout == path on the pod
    reader: str         # who loads it (for the manifest)
    required: bool = True


# Every weight file the DEFAULT config loads (Nov 12 2026), resolved the same
# way the code does: PROJECT_ROOT-relative in app.py, main-checkout-relative in
# web/students/paths.py (which falls back to the repo root when there is no
# git checkout, i.e. on the pod). Unused variants (s1_nosynth, s4_r*, base
# bumper/hub fallbacks, gapfill, sam2 large/base+) are deliberately absent.
WEIGHTS = (
    WeightSpec("runs/train/hub_zone_yolo26s_2026cur_v2/weights/best.pt",
               "app.py MODEL_PATH (main 3-class detector; preferred over hub_zone_yolo26s base)"),
    WeightSpec("runs/train/frc_bumpers_equipment_hardneg/weights/best.pt",
               "app.py BUMPER_MODEL_PATH (hard-negative bumper detector; preferred over frc_bumpers base)"),
    WeightSpec("runs/train/frc_bumpers_2026cur_v3/weights/best.pt",
               "web/students/paths.py BUMPER_WEIGHTS (window-label student detector)"),
    WeightSpec("yolo26n.pt", "app.py person detector (person veto / stream prefetch)"),
    WeightSpec("data/teacher/models/s1_digits.pt",
               "window_labels_reader=student_only (S1 digit reader)"),
    WeightSpec("data/teacher/models/s4_embedder.pt",
               "robot_embedder=s4 (S4 same-robot embedder)"),
    WeightSpec("checkpoints/sam2.1_hiera_small.pt",
               "tracking_backend=sam2 fallback only (if facebook/sam3 access fails)",
               required=False),
)
# Event state the server keeps as files in web/uploads (calibrations, camera
# layouts, zones, season overrides, clock-scan cache). Diagnostics, detection
# archives, frames and videos are NOT shipped.
EVENT_STATE_PREFIXES = ("calibration_", "composite_layout_", "zones_", "field_season_", "clockscan_")
EVENT_STATE_SUFFIXES = (".json", ".jpg", ".png")


def code_member_ok(name: str) -> bool:
    if name.endswith("/"):
        name = name[:-1]
    if any(name.startswith(p) or name + "/" == p for p in CODE_EXCLUDE_PREFIXES):
        return False
    if name.endswith(CODE_EXCLUDE_SUFFIXES):
        return False
    if "/__pycache__" in name or name.endswith(".pyc"):
        return False
    return True


def event_state_ok(name: str) -> bool:
    return name.startswith(EVENT_STATE_PREFIXES) and name.endswith(EVENT_STATE_SUFFIXES)


@dataclass
class BundlePlan:
    ref: str
    commit: str
    code_bytes: bytes                       # filtered tar (code only)
    code_files: list = field(default_factory=list)
    extra: list = field(default_factory=list)   # (src Path, arcname, kind)

    def total_size(self) -> int:
        return len(self.code_bytes) + sum(p.stat().st_size for p, _, _ in self.extra)

    def size_by_kind(self) -> dict:
        out = {"code": len(self.code_bytes)}
        for p, _, kind in self.extra:
            out[kind] = out.get(kind, 0) + p.stat().st_size
        return out


def filtered_code_tar(ref: str, repo: Path = REPO) -> tuple[bytes, list]:
    # paths missing from an older ref (e.g. deploy/runpod before it existed) are skipped
    present = [p for p in CODE_PATHS if git("ls-tree", "--name-only", ref, "--", p, repo=repo)]
    raw = git("archive", "--format=tar", ref, "--", *present, repo=repo, binary=True)
    src = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    buf = io.BytesIO()
    dst = tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT)
    names = []
    for m in src.getmembers():
        if m.name == "pax_global_header" or not code_member_ok(m.name):
            continue
        if m.isfile():
            dst.addfile(m, src.extractfile(m))
            names.append(m.name)
        elif m.isdir() or m.issym():
            dst.addfile(m)
    dst.close()
    return buf.getvalue(), names


def weights_manifest(main: Path, include_optional: bool = True) -> list:
    rows = []
    for w in WEIGHTS:
        if not w.required and not include_optional:
            continue
        p = main / w.rel
        if not p.exists():
            if w.required:
                raise SystemExit(f"required weight missing in main checkout: {p}")
            continue
        if p.stat().st_size < 4096:
            raise SystemExit(f"{p} looks like a git-LFS pointer ({p.stat().st_size} B)")
        rows.append((p, w.rel, "weights"))
    return rows


def event_state_files(main: Path) -> list:
    up = main / "web" / "uploads"
    if not up.is_dir():
        return []
    return [(p, f"web/uploads/{p.name}", "event_state")
            for p in sorted(up.iterdir()) if p.is_file() and event_state_ok(p.name)]


def snapshot_db(main: Path, dest: Path) -> Path:
    """Consistent copy of the live DB via the sqlite backup API (the local
    :8004 server may be writing). Never touches the source."""
    src = sqlite3.connect(f"file:{main / 'frc_scorer.db'}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def plan_bundle(ref: str, *, include_db: bool, include_event_state: bool,
                include_sam2: bool, scratch: Path, code_only: bool = False) -> BundlePlan:
    main = main_checkout()
    commit = git("rev-parse", "--short", ref)
    code, names = filtered_code_tar(ref)
    plan = BundlePlan(ref=ref, commit=commit, code_bytes=code, code_files=names)
    if code_only:
        return plan
    plan.extra += weights_manifest(main, include_optional=include_sam2)
    if include_event_state:
        plan.extra += event_state_files(main)
    if include_db:
        if (main / "frc_scorer.db").exists():
            plan.extra.append((snapshot_db(main, scratch / "frc_scorer.db"), "frc_scorer.db", "db"))
    return plan


def write_bundle(plan: BundlePlan, out: Path) -> int:
    with tarfile.open(out, "w:", format=tarfile.PAX_FORMAT) as tf:
        src = tarfile.open(fileobj=io.BytesIO(plan.code_bytes), mode="r:")
        for m in src.getmembers():
            tf.addfile(m, src.extractfile(m) if m.isfile() else None)
        for p, arc, _ in plan.extra:
            tf.add(str(p), arcname=arc, recursive=False)
        # provenance for `status`
        info = json.dumps({"ref": plan.ref, "commit": plan.commit,
                           "built_at": _dt.datetime.now().isoformat(timespec="seconds")}).encode()
        ti = tarfile.TarInfo("deploy/runpod/BUNDLE_INFO.json")
        ti.size = len(info)
        ti.mtime = int(time.time())
        tf.addfile(ti, io.BytesIO(info))
    return out.stat().st_size


def fmt_mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def print_plan(plan: BundlePlan) -> None:
    print(f"\nBundle: ref {plan.ref} @ {plan.commit}: {len(plan.code_files)} code files "
          f"({fmt_mb(len(plan.code_bytes))} tar)")
    print("Weights / state manifest (from the main checkout):")
    for p, arc, kind in plan.extra:
        print(f"  {fmt_mb(p.stat().st_size):>11}  {kind:<11} {arc}")
    by = plan.size_by_kind()
    print("Totals: " + ", ".join(f"{k} {fmt_mb(v)}" for k, v in by.items())
          + f" | TOTAL {fmt_mb(plan.total_size())}")


# --------------------------------------------------------------------------
# RunPod API
# --------------------------------------------------------------------------
def api_key() -> str:
    k = os.environ.get("RUNPOD_API") or local_env().get("RUNPOD_API")
    if not k:
        raise SystemExit("RUNPOD_API not found in the environment or the main checkout's .env")
    return k


def _request(method: str, url: str, body=None, timeout: float = 60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod {method} {url} -> HTTP {e.code}: {e.read()[:400].decode(errors='replace')}")


def graphql(query: str) -> dict:
    d = _request("POST", GRAPHQL, {"query": query})
    if d.get("errors"):
        raise SystemExit(f"GraphQL error: {d['errors'][:2]}")
    return d["data"]


def get_pod(pod_id: str) -> dict:
    return _request("GET", f"{REST}/pods/{pod_id}")


def gpu_types() -> list:
    q = ("query { gpuTypes { id displayName memoryInGb securePrice communityPrice secureCloud "
         "communityCloud lowestPrice(input:{gpuCount:1}) { stockStatus minVcpu minMemory } } }")
    return graphql(q)["gpuTypes"]


def build_create_body(*, name: str, gpus: list, cloud: str, image: str, volume_gb: int,
                      container_gb: int, pubkey: str, min_vcpu: int, min_ram: int,
                      data_centers: list | None = None, min_download_mbps: float | None = None,
                      network_volume_id: str | None = None) -> dict:
    """POST /v1/pods body. Only the PUBLIC ssh key goes in `env` — secrets are
    installed over ssh later. Only 22/tcp is exposed: the app has no auth, so
    it is reached through an ssh tunnel, never the public HTTP proxy."""
    if cloud not in ("SECURE", "COMMUNITY"):
        raise ValueError("cloud must be SECURE or COMMUNITY")
    body = {
        "name": name,
        "computeType": "GPU",
        "cloudType": cloud,
        "gpuTypeIds": list(gpus),
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "imageName": image,
        "containerDiskInGb": int(container_gb),
        "volumeInGb": int(volume_gb),
        "volumeMountPath": POD_WS,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pubkey.strip()},
        "supportPublicIp": True,
        "interruptible": False,
        "minVCPUPerGPU": int(min_vcpu),
        "minRAMPerGPU": int(min_ram),
    }
    if needs_cuda_filter(gpus):
        body["allowedCudaVersions"] = list(ALLOWED_CUDA)
    if data_centers:
        body["dataCenterIds"] = list(data_centers)
        body["dataCenterPriority"] = "custom"
    if min_download_mbps:
        body["minDownloadMbps"] = float(min_download_mbps)
    if network_volume_id:
        body["networkVolumeId"] = network_volume_id
        body.pop("volumeInGb")
    return body


def redact_body(body: dict) -> dict:
    b = json.loads(json.dumps(body))
    pk = b.get("env", {}).get("PUBLIC_KEY")
    if pk:
        b["env"]["PUBLIC_KEY"] = pk[:24] + "..." + pk[-12:]
    return b


# --------------------------------------------------------------------------
# State / ssh
# --------------------------------------------------------------------------
def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(st: dict) -> None:
    STATE_PATH.write_text(json.dumps(st, indent=2))


def require_pod(st: dict) -> str:
    if not st.get("pod_id"):
        raise SystemExit("no pod in deploy/runpod/.state.json — run `up` first")
    return st["pod_id"]


def ssh_key_path() -> Path:
    return Path(os.environ.get("RUNPOD_SSH_KEY", Path.home() / ".ssh" / "id_ed25519"))


def ssh_base(st: dict, *, tty: bool = False) -> list:
    if not st.get("ssh_host"):
        raise SystemExit("no ssh endpoint in state — `up` (or `status`) refreshes it once the pod has a public IP")
    kh = HERE / f".known_hosts_{st['pod_id']}"
    cmd = ["ssh", "-i", str(ssh_key_path()), "-p", str(st["ssh_port"]),
           "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={kh}",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
           "-o", "ConnectTimeout=15", "-o", "LogLevel=ERROR"]
    if tty:
        cmd.append("-t")
    return cmd + [f"root@{st['ssh_host']}"]


def ssh_run(st: dict, remote: str, *, check: bool = True, capture: bool = False,
            input_bytes: bytes | None = None, stdin_file=None, timeout: float | None = None):
    cmd = ssh_base(st) + [remote]
    r = subprocess.run(cmd, input=input_bytes, stdin=stdin_file,
                       capture_output=capture, timeout=timeout)
    if check and r.returncode != 0:
        err = (r.stderr or b"").decode(errors="replace")[-800:] if capture else ""
        raise SystemExit(f"remote command failed ({r.returncode}): {remote[:120]}\n{err}")
    return r


def ssh_out(st: dict, remote: str, timeout: float = 120, check: bool = True) -> str:
    r = ssh_run(st, remote, capture=True, timeout=timeout, check=check)
    return r.stdout.decode(errors="replace")


def refresh_endpoint(st: dict) -> dict:
    pod = get_pod(st["pod_id"])
    st["desired_status"] = pod.get("desiredStatus")
    st["cost_per_hr"] = pod.get("adjustedCostPerHr") or pod.get("costPerHr")
    st["gpu"] = (pod.get("gpu") or {}).get("displayName") or (pod.get("gpu") or {}).get("id") or st.get("gpu")
    ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
    if ip and port:
        st["ssh_host"], st["ssh_port"] = ip, int(port)
    save_state(st)
    return pod


def wait_for_ssh(st: dict, timeout_s: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pod = refresh_endpoint(st)
        if st.get("ssh_host"):
            r = subprocess.run(ssh_base(st) + ["echo ok"], capture_output=True, timeout=40)
            if r.returncode == 0 and b"ok" in r.stdout:
                print(f"ssh ready: root@{st['ssh_host']} -p {st['ssh_port']} "
                      f"({time.time() - t0:.0f}s)")
                return
        print(f"  waiting for pod/ssh... status={pod.get('desiredStatus')} "
              f"ip={pod.get('publicIp') or '-'} ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(10)
    raise SystemExit("ssh not ready in time — see README troubleshooting (`status`, RunPod console logs)")


def remote_job_active(st: dict) -> str | None:
    out = ssh_out(st, f"curl -s -m 10 http://127.0.0.1:{POD_PORT}/api/job/active || true", check=False).strip()
    try:
        return json.loads(out).get("job_id")
    except ValueError:
        return None


def parse_ts(s: str | None):
    """RunPod timestamps: ISO-8601 ('2026-11-12T10:00:00Z', with or without
    fractional seconds) or Go's '2026-11-12 10:00:00.123 +0000 UTC'."""
    if not s:
        return None
    s = s.strip().replace(" UTC", "")
    for cand in (s.replace("Z", "+00:00"), s):
        try:
            t = _dt.datetime.fromisoformat(cand)
            return t if t.tzinfo else t.replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return _dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def confirm(prompt: str, word: str) -> bool:
    ans = input(f"{prompt}\nType '{word}' to confirm: ").strip()
    return ans == word


# --------------------------------------------------------------------------
# Remote actions
# --------------------------------------------------------------------------
def push_bundle(st: dict, plan: BundlePlan, scratch: Path, *, replace_code: bool) -> None:
    tar_path = scratch / "bundle.tar"
    size = write_bundle(plan, tar_path)
    print(f"shipping {fmt_mb(size)} -> {POD_APP} ...", flush=True)
    clear = ""
    if replace_code:
        # remove tracked-code dirs first so files deleted upstream don't linger;
        # never touches .venv, .env, profile.env, frc_scorer.db, web/uploads,
        # web/replays, web/logs or weights.
        clear = (f"cd {POD_APP} && rm -rf src web/field web/scouting web/static web/students data/field && "
                 "{ [ ! -d web ] || find web -maxdepth 1 -type f "
                 "\\( -name '*.py' -o -name '*.sh' -o -name '*.md' \\) -delete; } && ")
    t0 = time.time()
    with open(tar_path, "rb") as f:
        ssh_run(st, f"mkdir -p {POD_APP} && {clear}tar -xf - -C {POD_APP} --no-same-owner", stdin_file=f)
    print(f"  shipped in {time.time() - t0:.0f}s")
    tar_path.unlink(missing_ok=True)


def run_bootstrap(st: dict, poll_s: float = 10) -> None:
    """Run bootstrap.sh DETACHED on the pod (a dropped ssh session must not
    kill a half-done venv install) and stream its log until it exits."""
    log = f"{POD_WS}/logs/bootstrap.log"
    ssh_run(st, f"mkdir -p {POD_WS}/logs && : > {log} && setsid -f nohup bash -c "
                f"'bash {POD_APP}/deploy/runpod/bootstrap.sh; echo BOOTSTRAP_EXIT=$?' "
                f">> {log} 2>&1 < /dev/null")
    seen = 0
    while True:
        time.sleep(poll_s)
        try:
            out = ssh_out(st, f"tail -n +{seen + 1} {log}", check=False, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        lines = out.splitlines()
        for ln in lines:
            print("  | " + ln, flush=True)
        seen += len(lines)
        done = [ln for ln in lines if ln.startswith("BOOTSTRAP_EXIT=")]
        if done:
            code = done[-1].split("=", 1)[1].strip()
            if code != "0":
                raise SystemExit(f"bootstrap failed (exit {code}); full log: {log} "
                                 f"(fix, then `push --bootstrap` or re-run `up`)")
            return


def install_secrets(st: dict, secrets: dict) -> None:
    text = render_dotenv(secrets).encode()
    ssh_run(st, f"umask 077 && mkdir -p {POD_APP} && cat > {POD_APP}/.env.tmp && "
                f"mv {POD_APP}/.env.tmp {POD_APP}/.env && chmod 600 {POD_APP}/.env",
            input_bytes=text)
    print("installed pod .env: " + ", ".join(f"{k}={mask(v)}" for k, v in sorted(secrets.items())))


def gather_secrets(args) -> dict:
    loc = local_env()
    # Keys always come from the operator: their own local .env (only with the
    # explicit --secrets-from-env) or typed at the prompt. Nothing in the repo
    # or the bundle carries a key, and --yes alone never reads .env.
    if getattr(args, "secrets_from_env", False):
        return select_secrets(loc, interactive=False)
    if getattr(args, "yes", False):
        raise SystemExit("--yes skips prompts, so the pod's API keys need --secrets-from-env "
                         "(reads YOUR local .env) or run without --yes to type them")
    print("\nSecrets for the pod (written over ssh to /workspace/frc/.env, chmod 600):")
    return select_secrets(loc, interactive=True, input_fn=lambda p: getpass.getpass(p))


def upload_cookies(st: dict, path: str) -> None:
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"cookies file not found: {p}")
    with open(p, "rb") as f:
        ssh_run(st, f"umask 077 && cat > {POD_WS}/youtube_cookies.txt", stdin_file=f)
    set_profile_vars(st, {"YTDLP_COOKIES_FILE": f"{POD_WS}/youtube_cookies.txt"})
    print("uploaded YouTube cookies -> /workspace/youtube_cookies.txt (YTDLP_COOKIES_FILE set)")


def read_remote_profile(st: dict) -> dict:
    return profiles.parse_env_file(ssh_out(st, f"cat {POD_APP}/profile.env 2>/dev/null || true"))


def set_profile_vars(st: dict, updates: dict, header: str = "") -> dict:
    env = read_remote_profile(st)
    env.update({k: str(v) for k, v in updates.items()})
    text = profiles.render_profile_env(env, header).encode()
    ssh_run(st, f"cat > {POD_APP}/profile.env.tmp && mv {POD_APP}/profile.env.tmp {POD_APP}/profile.env",
            input_bytes=text)
    return env


def restart_server(st: dict, force: bool = False) -> None:
    jid = remote_job_active(st)
    if jid and not force:
        raise SystemExit(f"job {jid} is running on the pod — not restarting (use --force)")
    ssh_run(st, f"bash {POD_APP}/deploy/runpod/serve.sh restart")


def tunnel_cmd(st: dict, port: int) -> list:
    kh = HERE / f".known_hosts_{st['pod_id']}"
    return ["ssh", "-N", "-i", str(ssh_key_path()), "-p", str(st["ssh_port"]),
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={kh}",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
            "-o", "ExitOnForwardFailure=yes",
            "-L", f"{port}:127.0.0.1:{POD_PORT}", f"root@{st['ssh_host']}"]


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_gpus(args) -> int:
    rows = [g for g in gpu_types()
            if g["id"].startswith("NVIDIA") and "MIG" not in g["id"]
            and args.min_gb <= (g.get("memoryInGb") or 0) <= args.max_gb]
    rows.sort(key=lambda g: (g["memoryInGb"], g.get("securePrice") or 99))
    print(f"{'gpuTypeId':<56} {'VRAM':>5} {'secure$/h':>9} {'comm$/h':>8} {'stock':>7} "
          f"{'vCPU':>5} {'RAM':>5} {'tier':>5} {'60h secure':>10}")
    for g in rows:
        lp = g.get("lowestPrice") or {}
        sp = g.get("securePrice") if g.get("secureCloud") else None
        cp = g.get("communityPrice") if g.get("communityCloud") else None
        tier = profiles.tier_for_vram((g["memoryInGb"] or 0) * 1024 * 0.97).name
        print(f"{g['id']:<56} {g['memoryInGb']:>4}G {sp if sp else '-':>9} {cp if cp else '-':>8} "
              f"{lp.get('stockStatus') or '-':>7} {lp.get('minVcpu') or '-':>5} {lp.get('minMemory') or '-':>5} "
              f"{tier:>5} {('$%.0f' % (60 * sp)) if sp else '-':>10}")
    print("\nPrices are per GPU-hour (pod volume storage is billed separately while stopped).")
    return 0


def _common_bundle_flags(ap):
    ap.add_argument("--ref", default="main", help="git ref to ship (default main)")
    ap.add_argument("--no-db", action="store_true",
                    help="start the pod with no frc_scorer.db and no event state (fresh)")
    ap.add_argument("--no-sam2-fallback", action="store_true",
                    help="skip checkpoints/sam2.1_hiera_small.pt (184 MB)")


def warn_dirty(ref: str) -> None:
    try:
        head_branch = git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except subprocess.CalledProcessError:
        return
    if dirty and ref in (head_branch, "HEAD"):
        print(f"WARNING: working tree has uncommitted changes; `{ref}` ships COMMITTED content only.")
    try:
        behind = git("rev-list", "--count", f"{ref}..origin/{ref}")
        if behind and behind != "0":
            print(f"WARNING: {ref} is {behind} commit(s) behind origin/{ref}.")
    except subprocess.CalledProcessError:
        pass


def cmd_up(args) -> int:
    st = load_state()
    gpus = args.gpu or DEFAULT_GPUS
    pubkey_path = Path(str(ssh_key_path()) + ".pub")
    pubkey = pubkey_path.read_text() if pubkey_path.exists() else "ssh-ed25519 <missing ~/.ssh/id_ed25519.pub>"
    name = args.name or f"frc-vision-{_dt.date.today().isoformat()}"
    body = build_create_body(name=name, gpus=gpus, cloud=args.cloud, image=args.image,
                             volume_gb=args.volume_gb, container_gb=args.container_gb, pubkey=pubkey,
                             min_vcpu=args.min_vcpu, min_ram=args.min_ram,
                             data_centers=args.data_center, min_download_mbps=args.min_download_mbps,
                             network_volume_id=args.network_volume_id)
    warn_dirty(args.ref)
    with tempfile.TemporaryDirectory(dir=args.scratch) as td:
        td = Path(td)
        plan = plan_bundle(args.ref, include_db=not args.no_db, include_event_state=not args.no_db,
                           include_sam2=not args.no_sam2_fallback, scratch=td)
        if args.dry_run:
            print("DRY RUN — nothing is created. POST https://rest.runpod.io/v1/pods with body:")
            print(json.dumps(redact_body(body), indent=2))
            print_plan(plan)
            if args.keep_bundle:
                out = Path(args.keep_bundle)
                size = write_bundle(plan, out)
                print(f"bundle tarball written to {out} ({fmt_mb(size)})")
            secrets = select_secrets(local_env(), interactive=False)
            print("\nsecrets that would be installed (from local .env): "
                  + ", ".join(f"{k}={mask(v)}" for k, v in sorted(secrets.items())))
            prof = profiles.tier_for_vram(80 * 1024 * 0.97)
            print("\nladder (shown for an 80 GB card; the pod picks its tier from nvidia-smi):")
            print(profiles.describe_ladder(prof))
            return 0

        secrets = gather_secrets(args)
        if st.get("pod_id"):
            pod = refresh_endpoint(st)
            print(f"reusing pod {st['pod_id']} ({pod.get('desiredStatus')}, {st.get('gpu')}, "
                  f"${st.get('cost_per_hr')}/h)")
            if pod.get("desiredStatus") == "EXITED":
                if not args.yes and not confirm("Pod is stopped; starting it resumes billing.", "start"):
                    return 1
                _request("POST", f"{REST}/pods/{st['pod_id']}/start")
        else:
            print(json.dumps(redact_body(body), indent=2))
            if not args.yes and not confirm(
                    f"This RENTS a GPU pod ({', '.join(gpus)}; {args.cloud}) and bills per hour "
                    f"until you stop/terminate it.", "rent"):
                return 1
            pod = _request("POST", f"{REST}/pods", body)
            st = {"pod_id": pod["id"], "created_at": time.time(), "gpu_priority": gpus,
                  "cost_per_hr": pod.get("costPerHr"), "image": args.image}
            save_state(st)
            print(f"created pod {pod['id']} at ${pod.get('costPerHr')}/h")
        wait_for_ssh(st)
        jid = remote_job_active(st)
        if jid:
            raise SystemExit(f"job {jid} is running on the pod; `up` would restart the server. "
                             f"Wait for it (or use `push --force`).")
        has_db =ssh_run(st, f"test -f {POD_APP}/frc_scorer.db", check=False).returncode == 0
        if has_db and not args.overwrite_db:
            # a re-run of `up` mid-event must never clobber the pod's DB or the
            # calibrations/layouts made on the pod
            plan.extra = [e for e in plan.extra if e[2] not in ("db", "event_state")]
            print("pod already has frc_scorer.db: keeping the pod's DB + event state "
                  "(--overwrite-db to replace them with the local copy)")
        push_bundle(st, plan, td, replace_code=True)
        install_secrets(st, secrets)
        if args.cookies:
            upload_cookies(st, args.cookies)
        print("bootstrapping (first boot: ~10-15 min for venv + ~3.5 GB SAM3 download)...", flush=True)
        run_bootstrap(st)
    st["pushed_ref"] = args.ref
    save_state(st)
    print(f"\nServer is up. Open the tunnel:\n  {Path(__file__).name} tunnel   "
          f"(then browse http://localhost:{DEFAULT_TUNNEL_PORT})")
    print("  raw: " + " ".join(shlex.quote(c) for c in tunnel_cmd(st, DEFAULT_TUNNEL_PORT)))
    return 0


def cmd_push(args) -> int:
    st = load_state()
    require_pod(st)
    refresh_endpoint(st)
    warn_dirty(args.ref)
    jid = remote_job_active(st)
    if jid and not args.force:
        raise SystemExit(f"job {jid} is running on the pod — push would restart the server (use --force)")
    with tempfile.TemporaryDirectory(dir=args.scratch) as td:
        plan = plan_bundle(args.ref, include_db=False, include_event_state=False,
                           include_sam2=False, scratch=Path(td), code_only=not args.with_weights)
        if args.with_weights:
            plan.extra = weights_manifest(main_checkout(), include_optional=True)
        print_plan(plan)
        push_bundle(st, plan, Path(td), replace_code=True)
    if args.bootstrap:
        run_bootstrap(st)
    else:
        restart_server(st, force=True)
    st["pushed_ref"] = args.ref
    save_state(st)
    return 0


def cmd_secrets(args) -> int:
    st = load_state()
    require_pod(st)
    refresh_endpoint(st)
    install_secrets(st, gather_secrets(args))
    if args.cookies:
        upload_cookies(st, args.cookies)
    if args.restart:
        restart_server(st, force=args.force)
    else:
        print("restart the server to pick them up: runpod_cli.py restart")
    return 0


def cmd_ssh(args) -> int:
    st = load_state()
    require_pod(st)
    refresh_endpoint(st)
    cmd = ssh_base(st, tty=not args.command) + (args.command or [])
    return subprocess.call(cmd)


def cmd_tunnel(args) -> int:
    st = load_state()
    require_pod(st)
    refresh_endpoint(st)
    cmd = tunnel_cmd(st, args.port)
    print(f"tunnel: http://localhost:{args.port} -> pod 127.0.0.1:8004 (Ctrl-C to close)")
    if args.print_only:
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    while True:
        rc = subprocess.call(cmd)
        if not args.reconnect:
            return rc
        print(f"tunnel exited ({rc}); reconnecting in 5 s (Ctrl-C to stop)")
        time.sleep(5)


def cmd_status(args) -> int:
    st = load_state()
    require_pod(st)
    pod = refresh_endpoint(st)
    cost = float(st.get("cost_per_hr") or 0)
    started = pod.get("lastStartedAt")
    t = parse_ts(started)
    up_h = (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds() / 3600 if t else None
    print(f"pod {st['pod_id']}  {pod.get('desiredStatus')}  {st.get('gpu')}  ${cost}/h  "
          f"image {pod.get('image')}")
    if up_h is not None:
        print(f"  up {up_h:.1f} h since {started} -> ~${up_h * cost:.2f} this session (GPU only; "
              f"see the RunPod billing page for storage)")
    if pod.get("desiredStatus") != "RUNNING" or not st.get("ssh_host"):
        return 0
    print(f"  ssh root@{st['ssh_host']} -p {st['ssh_port']}  (pushed ref: {st.get('pushed_ref')})")
    remote = (
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; "
        f"bash {POD_APP}/deploy/runpod/serve.sh status; "
        f"grep -E '^FRC_(GPU_TIER|PROFILE_RUNG)=|^SAM|^CORNER' {POD_APP}/profile.env 2>/dev/null | tr '\\n' ' '; echo; "
        f"cat {POD_APP}/deploy/runpod/BUNDLE_INFO.json 2>/dev/null; echo; df -h {POD_WS} | tail -1")
    print(ssh_out(st, remote, check=False))
    return 0


def cmd_logs(args) -> int:
    st = load_state()
    require_pod(st)
    refresh_endpoint(st)
    f = "-f " if args.follow else ""
    return subprocess.call(ssh_base(st) + [f"tail -n {args.lines} {f}{POD_LOG}"])


def cmd_restart(args) -> int:
    st = load_state()
    require_pod(st)
    refresh_endpoint(st)
    restart_server(st, force=args.force)
    return 0


def remote_tier(st: dict) -> profiles.Tier:
    out = ssh_out(st, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits").strip()
    name, mib = [s.strip() for s in out.splitlines()[0].rsplit(",", 1)]
    st["gpu_name"], st["vram_mib"] = name, float(mib)
    save_state(st)
    return profiles.tier_for_vram(float(mib))


def apply_rung(st: dict, tier: profiles.Tier, rung_name: str, force: bool) -> None:
    rung = profiles.rung_by_name(tier, rung_name)
    cur = read_remote_profile(st)
    keep = {k: v for k, v in cur.items()
            if k not in profiles.KNOB_ENV.values() and k not in profiles.BASE_ENV
            and k not in ("FRC_GPU_TIER", "FRC_PROFILE_RUNG")}
    env = profiles.profile_env(tier, rung, extra=keep)
    text = profiles.render_profile_env(env, f"applied {rung.name} on {st.get('gpu_name')} "
                                            f"{_dt.datetime.now().isoformat(timespec='seconds')}").encode()
    ssh_run(st, f"cat > {POD_APP}/profile.env.tmp && mv {POD_APP}/profile.env.tmp {POD_APP}/profile.env",
            input_bytes=text)
    print(f"profile.env -> {rung.name}: {rung.env()}")
    restart_server(st, force=force)


def cmd_profile(args) -> int:
    st = load_state()
    require_pod(st)
    refresh_endpoint(st)
    tier = remote_tier(st)
    if args.apply:
        apply_rung(st, tier, args.apply, args.force)
        return 0
    print(f"GPU {st.get('gpu_name')} ({st.get('vram_mib', 0) / 1024:.0f} GiB) -> {tier.name}")
    print(profiles.describe_ladder(tier))
    print("\npod profile.env:")
    print(ssh_out(st, f"cat {POD_APP}/profile.env", check=False))
    return 0


# ----------------------------------------------------------------- tune
def tune_results_path(pod_id: str) -> Path:
    return TUNE_DIR / f"{pod_id}.json"


def load_tune(pod_id: str) -> dict:
    p = tune_results_path(pod_id)
    return json.loads(p.read_text()) if p.exists() else {"pod_id": pod_id, "rungs": {}}


def save_tune(res: dict) -> None:
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    tune_results_path(res["pod_id"]).write_text(json.dumps(res, indent=2))


def over_budget(row: dict, budget_min: float, on: str = "final") -> bool:
    w = row.get("wall_final_s") if on == "final" else row.get("wall_job_s")
    if w is None:
        w = row.get("wall_job_s")
    return w is None or w / 60.0 > budget_min


def recommend(rows: dict, order: list, budget_min: float, on: str = "final") -> str | None:
    """Highest rung that finished within budget. Rungs are cumulative and
    ordered by expected quality, so the highest in-budget rung is the pick
    unless its scores say otherwise (the table shows them; noise is real)."""
    best = None
    for name in order:
        r = rows.get(name)
        if r and r.get("status") == "done" and not over_budget(r, budget_min, on):
            best = name
    return best


def ensure_test_video(st: dict, local_video: Path) -> str:
    remote = f"{POD_TUNE}/{TUNE_MATCH_KEY}.mp4"
    size = local_video.stat().st_size
    have = ssh_out(st, f"stat -c %s {remote} 2>/dev/null || echo 0", check=False).strip()
    if have != str(size):
        print(f"uploading test video {local_video.name} ({fmt_mb(size)}) -> {remote}", flush=True)
        with open(local_video, "rb") as f:
            ssh_run(st, f"mkdir -p {POD_TUNE} && cat > {remote}.part && mv {remote}.part {remote}",
                    stdin_file=f)
    return remote


def run_rung_remote(st: dict, rung: profiles.Rung, video: str, poll_s: float = 20) -> dict:
    """Launch (or re-attach to) the detached pod_job.py for this rung and
    poll its result file until it finishes."""
    out = f"{POD_TUNE}/jobs/{rung.name}.json"
    cur = ssh_out(st, f"cat {out} 2>/dev/null || true", check=False).strip()
    state = json.loads(cur) if cur else None
    alive = False
    if state and state.get("status") not in ("done", "error"):
        alive = ssh_run(st, f"kill -0 {int(state.get('pid', 0))} 2>/dev/null", check=False).returncode == 0
    if state and state.get("status") == "done":
        return state
    if not alive:
        jid = remote_job_active(st)
        if jid:
            raise SystemExit(f"pod is busy with job {jid}; tune waits for an idle server")
        cfg = json.dumps(rung.config())
        # setsid -f forks and returns at once; with a trailing `&` the list
        # ran in a background subshell that held ssh's stdout open, so ssh
        # never returned (Nov 12 R0 launch hit the 120 s timeout).
        launch = (f"mkdir -p {POD_TUNE}/jobs && cd {POD_APP} && setsid -f nohup .venv/bin/python "
                  f"deploy/runpod/pod_job.py --video {shlex.quote(video)} --match-key {TUNE_MATCH_KEY} "
                  f"--event {TUNE_EVENT} --label {rung.name} --config-json {shlex.quote(cfg)} "
                  f"--host 127.0.0.1:{POD_PORT} "
                  f"--out {out} > {POD_TUNE}/jobs/{rung.name}.log 2>&1 < /dev/null; echo started")
        ssh_out(st, f"rm -f {out}; " + launch)
        print(f"  [{rung.name}] launched {cfg}", flush=True)
    else:
        print(f"  [{rung.name}] re-attached to running job {state.get('job_id')}", flush=True)
    t0 = time.time()
    while True:
        time.sleep(poll_s)
        try:
            cur = ssh_out(st, f"cat {out} 2>/dev/null || true", check=False, timeout=60).strip()
        except subprocess.TimeoutExpired:
            continue
        if not cur:
            if time.time() - t0 > 300:
                log = ssh_out(st, f"tail -n 30 {POD_TUNE}/jobs/{rung.name}.log", check=False)
                raise SystemExit(f"pod_job.py wrote nothing after 5 min:\n{log}")
            continue
        try:
            state = json.loads(cur)
        except ValueError:
            continue
        el = time.time() - (state.get("t_submit") or t0)
        print(f"  [{rung.name}] {str(state.get('status')):<11} {el / 60:5.1f} min  "
              f"{state.get('progress') or ''}% {str(state.get('message') or '')[:70]}", flush=True)
        if state.get("status") in ("done", "error"):
            return state


def score_locally(st: dict, row: dict, local_video: Path, score_shift: float) -> dict:
    job_id = row["job_id"]
    dest = PULL_ROOT / st["pod_id"] / "tune" / job_id
    dest.mkdir(parents=True, exist_ok=True)
    meta = dest / "meta.json"
    if not meta.exists():
        with open(meta, "wb") as f:
            r = subprocess.run(ssh_base(st) + [f"cat {POD_APP}/web/replays/{job_id}/meta.json"], stdout=f)
        if r.returncode != 0:
            meta.unlink(missing_ok=True)
            return {"error": "could not pull meta.json"}
    rj = dest / "replay_row.json"
    rj.write_text(json.dumps(row.get("replay") or {"id": job_id, "match_key": TUNE_MATCH_KEY}))
    py = main_checkout() / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    r = subprocess.run([str(py), str(HERE / "score_pulled.py"), "--replay-json", str(rj),
                        "--meta", str(meta), "--video", str(local_video),
                        "--match", str(TUNE_MATCH_NUMBER), "--score-shift", str(score_shift)],
                       capture_output=True, text=True, cwd=str(REPO))
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-600:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": r.stdout[-400:]}


def _pct(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def print_tune_table(res: dict, order: list, budget_min: float, on: str) -> None:
    print(f"\n{'rung':<4} {'fps':>4} {'cs':>3} {'chunk':>5} {'job':>6} {'final':>6} {'peakVRAM':>9} "
          f"{'aggNN':>6} {'GTcov':>6} {'errMed':>6} {'lblOK':>6} {'wrong':>6} {'misID':>6}  note")
    for name in order:
        r = res["rungs"].get(name)
        if not r:
            continue
        k = r.get("knobs", {})
        sc = r.get("scores") or {}
        m = sc.get("manta") or {}
        g = sc.get("gt") or {}
        wj = f"{r['wall_job_s'] / 60:.1f}m" if r.get("wall_job_s") else "-"
        wf = f"{r['wall_final_s'] / 60:.1f}m" if r.get("wall_final_s") else "-"
        vr = f"{r['peak_vram_mib'] / 1024:.1f}G" if r.get("peak_vram_mib") else "-"
        note = "OVER BUDGET" if r.get("status") == "done" and over_budget(r, budget_min, on) else ""
        if r.get("status") != "done":
            note = f"{r.get('status')}: {str(r.get('error'))[:60]}"
        elif g.get("error") or m.get("error"):
            note += f" score err: {str(g.get('error') or m.get('error'))[:60]}"
        print(f"{name:<4} {k.get('sam3_track_fps', k.get('sam2_track_fps', '-')):>4} {k.get('corner_track_stride', '-'):>3} "
              f"{k.get('sam3_chunk_frames', '-'):>5} {wj:>6} {wf:>6} {vr:>9} "
              f"{m.get('agg_meanNN', '-'):>6} {_pct(g.get('coverage')):>6} "
              f"{g.get('err_median_m') if g.get('err_median_m') is not None else '-':>6} "
              f"{_pct(g.get('label_correct')):>6} {_pct(g.get('label_wrong')):>6} "
              f"{g.get('misid_s') if g.get('misid_s') is not None else '-':>6}  {note}")


def _table_order(res: dict, order: list) -> list:
    """Every measured rung (plain + tagged A/B arms), ladder order first."""
    return list(dict.fromkeys(list(order) + sorted(res.get("rungs", {}))))


def arm_base(rungs: list, rows: dict, rec: str | None) -> "profiles.Rung | None":
    """Rung the experimental arms sit on: the recommendation, else R1, else
    the highest rung that finished."""
    by = {r.name: r for r in rungs}
    if rec in by:
        return by[rec]
    done = [r for r in rungs if (rows.get(r.name) or {}).get("status") == "done"]
    if "R1" in by and any(r.name == "R1" for r in done):
        return by["R1"]
    return done[-1] if done else None


def run_arms(st: dict, res: dict, rungs: list, rec: str | None, remote_video: str,
             local_video: Path, args) -> None:
    """Measure profiles.ARMS on top of the base rung (never recommended or
    applied; a failed arm doesn't stop the others)."""
    base = arm_base(rungs, res["rungs"], rec)
    if base is None:
        print("no finished rung to run the experimental arms on; skipping arms")
        return
    arm_rungs = [profiles.arm_rung(base, a) for a in profiles.ARMS]
    print(f"\nexperimental arms on {base.name}: {', '.join(r.name for r in arm_rungs)}")
    for rung in arm_rungs:
        prev = res["rungs"].get(rung.name)
        if prev and prev.get("status") == "done" and not args.redo:
            print(f"  [{rung.name}] already measured (skip; --redo to re-run)")
            continue
        print(f"  [{rung.name}] {rung.rationale[:120]}")
        state = run_rung_remote(st, rung, remote_video)
        row = {"knobs": rung.knobs, "delta": rung.delta, "arm_of": base.name, **{k: state.get(k) for k in (
            "status", "error", "job_id", "wall_job_s", "wall_final_s", "peak_vram_mib",
            "timings", "replay", "window_labels_aborted")}}
        if row["status"] == "done":
            print(f"  [{rung.name}] scoring replay {row['job_id'][:8]} locally...", flush=True)
            row["scores"] = score_locally(st, row, local_video, args.score_shift)
        res["rungs"][rung.name] = row
        save_tune(res)
    print_tune_table(res, _table_order(res, [r.name for r in rungs]), args.budget_min, args.budget_on)


def cmd_tune(args) -> int:
    st = load_state()
    pod_id = require_pod(st)
    refresh_endpoint(st)
    tier = remote_tier(st)
    rungs = profiles.ladder(tier)
    extra = json.loads(args.with_json) if getattr(args, "with_json", None) else {}
    if extra:
        # A/B any knob against the ladder: R1 + {"sam3_parallel_views": true}
        # is measured as R1<tag>, next to the plain rungs in the same table.
        tag = args.tag or "X"
        rungs = [profiles.Rung(r.name + tag, {**r.knobs, **extra}, {**r.delta, **extra},
                               r.rationale + f" + {extra}") for r in rungs]
    order = [r.name for r in rungs]
    res = load_tune(pod_id)
    res.update({"gpu": st.get("gpu_name"), "vram_mib": st.get("vram_mib"), "tier": tier.name,
                "budget_min": args.budget_min, "budget_on": args.budget_on})
    if args.apply:
        apply_rung(st, tier, args.apply, args.force)
        res["applied"] = args.apply.upper()
        save_tune(res)
        return 0
    if args.show:
        print_tune_table(res, _table_order(res, order), args.budget_min, args.budget_on)
        return 0
    local_video = Path(args.video) if args.video else main_checkout() / "web" / "uploads" / QM8_VIDEO_NAME
    if not local_video.exists():
        raise SystemExit(f"test video not found: {local_video}")
    remote_video = ensure_test_video(st, local_video)
    sfx = (args.tag or "X") if extra else ""
    start = order.index(args.start.upper() + sfx)
    stop = order.index(args.stop.upper() + sfx) if args.stop else len(order) - 1
    climb = [] if getattr(args, "arms_only", False) else rungs[start:stop + 1]
    if climb:
        print(f"tuning on {st.get('gpu_name')} ({tier.name}); budget {args.budget_min} min "
              f"({args.budget_on}); rungs {order[start]}..{order[stop]}")
    for rung in climb:
        prev = res["rungs"].get(rung.name)
        if prev and prev.get("status") == "done" and not args.redo:
            print(f"  [{rung.name}] already measured (skip; --redo to re-run)")
        else:
            state = run_rung_remote(st, rung, remote_video)
            row = {"knobs": rung.knobs, "delta": rung.delta, **{k: state.get(k) for k in (
                "status", "error", "job_id", "wall_job_s", "wall_final_s", "peak_vram_mib",
                "timings", "replay", "window_labels_aborted")}}
            if row["status"] == "done":
                print(f"  [{rung.name}] scoring replay {row['job_id'][:8]} locally...", flush=True)
                row["scores"] = score_locally(st, row, local_video, args.score_shift)
            res["rungs"][rung.name] = row
            save_tune(res)
        r = res["rungs"][rung.name]
        print_tune_table(res, _table_order(res, order), args.budget_min, args.budget_on)
        if r.get("status") != "done":
            print(f"rung {rung.name} failed; stopping (fix, then re-run `tune --start {rung.name}`)")
            break
        if over_budget(r, args.budget_min, args.budget_on):
            print(f"rung {rung.name} exceeded the {args.budget_min} min budget; stopping the climb")
            break
    rec = recommend(res["rungs"], order, args.budget_min, args.budget_on)
    res["recommended"] = rec
    save_tune(res)
    if not extra and not getattr(args, "no_arms", False):
        run_arms(st, res, rungs, rec, remote_video, local_video, args)
    print(f"\nresults -> {tune_results_path(pod_id)}")
    if rec:
        print(f"recommended: {rec}  (apply: {Path(__file__).name} tune --apply {rec})")
        if args.auto_apply:
            apply_rung(st, tier, rec, args.force)
    else:
        print("no rung finished within budget")
    return 0


# ----------------------------------------------------------------- pull / stop
def cmd_pull(args) -> int:
    st = load_state()
    pod_id = require_pod(st)
    refresh_endpoint(st)
    dest = PULL_ROOT / pod_id
    dest.mkdir(parents=True, exist_ok=True)
    # 1. consistent DB snapshot on the pod, then copy
    ssh_run(st, f"cd {POD_APP} && .venv/bin/python -c \"import sqlite3; s=sqlite3.connect('file:frc_scorer.db?mode=ro', "
                f"uri=True); d=sqlite3.connect('/tmp/frc_pull.db'); s.backup(d); d.close()\"")
    db_out = dest / "frc_scorer.db"
    with open(db_out, "wb") as f:
        subprocess.run(ssh_base(st) + ["cat /tmp/frc_pull.db && rm -f /tmp/frc_pull.db"], stdout=f, check=True)
    print(f"DB -> {db_out} ({fmt_mb(db_out.stat().st_size)})")
    # 2. replays not pulled yet (+ event-state files: calibrations made at the event)
    rdir = dest / "web" / "replays"
    have = {p.name for p in rdir.iterdir() if (p / "meta.json").exists()} if rdir.exists() else set()
    listing = ssh_out(st, f"ls {POD_APP}/web/replays 2>/dev/null || true", check=False).split()
    new = [r for r in listing if r not in have]
    excl = "" if args.with_video else "--exclude='*.mp4' "
    lst = f"{POD_WS}/.pull_list"
    names = "".join(f"web/replays/{r}\n" for r in new).encode()
    pattern = "^(" + "|".join(EVENT_STATE_PREFIXES) + r").*\.(json|jpg|png)$"
    ssh_run(st, f"cd {POD_APP} && cat > {lst} && (ls web/uploads 2>/dev/null | grep -E '{pattern}' "
                f"| sed 's#^#web/uploads/#' >> {lst}; true)", input_bytes=names)
    p = subprocess.Popen(ssh_base(st) + [f"cd {POD_APP} && [ -s {lst} ] && tar -cf - {excl}-T {lst}"],
                         stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
            for m in tf:
                if m.name.startswith(("web/replays/", "web/uploads/")) and ".." not in m.name \
                        and (m.isfile() or m.isdir()):
                    tf.extract(m, dest, set_attrs=False,
                               **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    except tarfile.ReadError:
        pass  # nothing to pull
    p.wait()
    print(f"replays: {len(new)} new ({len(listing)} on pod) -> {dest / 'web' / 'replays'}"
          + ("" if args.with_video else " (HUD videos skipped; --with-video to include)"))
    print(f"event state -> {dest / 'web' / 'uploads'}")
    print("Local frc_scorer.db and web/replays were NOT touched.")
    return 0


def cmd_stop(args) -> int:
    st = load_state()
    pod_id = require_pod(st)
    pod = refresh_endpoint(st)
    jid = remote_job_active(st) if pod.get("desiredStatus") == "RUNNING" and st.get("ssh_host") else None
    msg = (f"STOP pod {pod_id} ({st.get('gpu')}). GPU billing stops; the /workspace volume is kept "
           f"(and billed for storage). Starting again may land on a machine without a free GPU.")
    if jid:
        msg += f"\nA job is RUNNING ({jid}) and will be killed."
    if not confirm(msg, "stop"):
        return 1
    _request("POST", f"{REST}/pods/{pod_id}/stop")
    print("stop requested")
    return 0


def cmd_terminate(args) -> int:
    st = load_state()
    pod_id = require_pod(st)
    refresh_endpoint(st)
    pulled = (PULL_ROOT / pod_id / "frc_scorer.db").exists()
    msg = (f"TERMINATE pod {pod_id}: the pod AND its /workspace volume (DB, replays, calibrations, "
           f"venv, caches) are DELETED permanently.\n"
           + ("Last `pull` exists locally." if pulled else "You have NOT run `pull` for this pod!"))
    if not confirm(msg, f"terminate {pod_id}"):
        return 1
    _request("DELETE", f"{REST}/pods/{pod_id}")
    st["terminated_at"] = time.time()
    STATE_PATH.rename(HERE / f".state.{pod_id}.terminated.json")
    print("terminated; state archived")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="runpod_cli.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gpus", help="GPU types: VRAM, prices, stock (read-only)")
    p.add_argument("--min-gb", type=int, default=48)
    p.add_argument("--max-gb", type=int, default=150)
    p.set_defaults(fn=cmd_gpus)

    p = sub.add_parser("up", help="create (or reuse) the pod, ship, install secrets, bootstrap, start")
    p.add_argument("--gpu", action="append", help="gpuTypeId; repeat for a priority list "
                                                  f"(default: {', '.join(DEFAULT_GPUS)})")
    p.add_argument("--cloud", default="SECURE", choices=("SECURE", "COMMUNITY"))
    p.add_argument("--volume-gb", type=int, default=150)
    p.add_argument("--container-gb", type=int, default=40)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--name")
    p.add_argument("--min-vcpu", type=int, default=8)
    p.add_argument("--min-ram", type=int, default=32, help="GB RAM per GPU")
    p.add_argument("--data-center", action="append", help="dataCenterId (repeatable)")
    p.add_argument("--min-download-mbps", type=float)
    p.add_argument("--network-volume-id")
    p.add_argument("--cookies", help="YouTube cookies.txt to upload (sets YTDLP_COOKIES_FILE)")
    p.add_argument("--yes", action="store_true", help="skip the rent/confirm prompts (keys still need --secrets-from-env)")
    p.add_argument("--secrets-from-env", action="store_true", help="take the pod API keys from YOUR local .env instead of prompting")
    p.add_argument("--dry-run", action="store_true", help="print the API body + bundle manifest; create nothing")
    p.add_argument("--keep-bundle", metavar="PATH", help="with --dry-run: also write the tarball here")
    p.add_argument("--overwrite-db", action="store_true",
                   help="reusing a pod that already has a DB: replace it (and event state) with the local copy")
    p.add_argument("--scratch", default=None, help="temp dir for the bundle (default system temp)")
    _common_bundle_flags(p)
    p.set_defaults(fn=cmd_up)

    p = sub.add_parser("push", help="re-ship code from a git ref and restart the server")
    p.add_argument("--ref", default="main")
    p.add_argument("--with-weights", action="store_true", help="re-ship the weights manifest too")
    p.add_argument("--bootstrap", action="store_true", help="re-run bootstrap.sh (deps changed)")
    p.add_argument("--force", action="store_true", help="even if a job is running")
    p.add_argument("--scratch", default=None)
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("secrets", help="re-prompt and install the pod .env")
    p.add_argument("--secrets-from-env", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--cookies", help="YouTube cookies.txt to upload")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_secrets)

    p = sub.add_parser("ssh", help="interactive shell (or run a command after --)")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_ssh)

    p = sub.add_parser("tunnel", help=f"forward localhost:{DEFAULT_TUNNEL_PORT} -> pod :8004")
    p.add_argument("--port", type=int, default=DEFAULT_TUNNEL_PORT)
    p.add_argument("--reconnect", action="store_true", help="re-open when it drops")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(fn=cmd_tunnel)

    p = sub.add_parser("status", help="pod state, cost so far, GPU, server, active job")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("logs", help="tail the server log")
    p.add_argument("-n", "--lines", type=int, default=100)
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("restart", help="restart the server (refuses while a job runs)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_restart)

    p = sub.add_parser("profile", help="show the tier/ladder/profile.env, or --apply R<n>")
    p.add_argument("--apply")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("tune", help="timed qm8 full-match ladder; pick the best rung within budget")
    p.add_argument("--start", default="R0")
    p.add_argument("--stop", default=None, help="last rung to try (default: top)")
    p.add_argument("--budget-min", type=float, default=8.0)
    p.add_argument("--budget-on", choices=("final", "job"), default="final",
                   help="final = submit -> labelled replay ready (default); job = submit -> ws complete")
    p.add_argument("--redo", action="store_true", help="re-run rungs already in the results file")
    p.add_argument("--apply", help="write rung R<n> into profile.env and restart")
    p.add_argument("--auto-apply", action="store_true", help="apply the recommended rung at the end")
    p.add_argument("--show", action="store_true", help="print the results table only")
    p.add_argument("--with", dest="with_json", metavar="JSON",
                   help='extra per-job knobs merged into every rung, e.g. \'{"sam3_parallel_views": true}\'')
    p.add_argument("--tag", help="suffix for --with rung names (default X): R1 -> R1P")
    p.add_argument("--no-arms", action="store_true",
                   help="skip the experimental arms (profiles.ARMS, e.g. R1M = sam3_multiplex) "
                        "measured after the ladder on the recommended rung")
    p.add_argument("--arms-only", action="store_true",
                   help="skip the ladder climb; only run the experimental arms on the "
                        "already-measured recommended rung")
    p.add_argument("--video", help="local copy of the qm8 video (default: main checkout web/uploads)")
    p.add_argument("--score-shift", type=float, default=0.475,
                   help="GT scoring half-robot shift (scripts/score_ground_truth.py --score-shift)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("pull", help="copy pod DB + replays + event state to data/runpod_pull/<podId>/")
    p.add_argument("--with-video", action="store_true", help="include replay HUD mp4s")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("stop", help="stop the pod (typed confirmation)")
    p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("terminate", help="terminate the pod + volume (typed confirmation)")
    p.set_defaults(fn=cmd_terminate)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

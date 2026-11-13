"""Per-job profiling for `runpod_cli.py tune` (stdlib only).

Two halves:

* Pure parsing / summarising (unit-tested in tests/test_runpod_profiling.py):
  py-spy ``--format raw`` collapsed stacks, nvidia-smi csv / dmon samples,
  /proc readers, and the text tables the CLI prints.
* ``JobProfiler`` (runs ON the pod inside pod_job.py): attaches
  ``py-spy record --gil`` to the app server and logs ``nvidia-smi`` GPU
  samples for exactly the lifetime of one tune job, then summarises both
  into the rung's ``profile`` dict. Never raises: every failure is recorded
  in the summary (``pyspy_error`` / ``gpu_error``) so a profiler problem can
  never fail a rung.

Collapsed-stack format (one line per distinct stack, root first)::

    <module> (app.py:16018);run (uvicorn/main.py:606);...;leaf (field/occlusion.py:149) 7

py-spy shortens filenames to the part after the sys.path entry, so the app's
own modules (``python web/app.py`` puts web/ on sys.path) appear as
``app.py``, ``field/occlusion.py``, ``students/s4_embedder.py``, and
site-packages as ``torch/...``, ``sam3/...``. "Project" frames are those
whose first path segment is a top-level name in web/ or src/, plus sam3/sam2.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parents[1]

THIRD_PARTY_PROJECT = ("sam3", "sam2")   # vendored-ish model code we tune
_WALK_SKIP = {"static", "uploads", "replays", "logs", "node_modules", "__pycache__"}
_FRAME_RE = re.compile(r"^(?P<func>.*?) \((?P<file>[^()]*?)(?::(?P<line>\d+))?\)$")


# ------------------------------------------------------------------ py-spy
def project_roots(app_dir: Path | str = APP_DIR) -> set:
    """First path segments that mark a frame as ours: every top-level module /
    package in web/ and src/ (as py-spy shortens them), plus sam3/sam2."""
    roots = set(THIRD_PARTY_PROJECT)
    for sub in ("web", "src"):
        d = Path(app_dir) / sub
        try:
            for p in d.iterdir():
                if p.name.startswith((".", "_")):
                    continue
                if p.is_dir() or p.suffix == ".py":
                    roots.add(p.name)
        except OSError:
            pass
        # py-spy keeps only the *package* part of a path (it walks up while
        # dirs have __init__.py): a module in a plain subdir shows as its
        # bare basename, so add those too.
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [n for n in dirnames if not n.startswith((".", "_"))
                           and n not in _WALK_SKIP]
            if dirpath != str(d) and "__init__.py" not in filenames:
                roots.update(f for f in filenames if f.endswith(".py"))
    if len(roots) == len(THIRD_PARTY_PROJECT):     # no checkout here: known layout
        roots |= {"app.py", "field", "students", "scouting", "analysis_prefetch.py",
                  "bumper_register.py", "clef_reader.py", "jev_reader.py", "embedder.py",
                  "hud_draw.py", "hud_render.py", "reid.py", "database.py"}
    return roots


def parse_frame(frame: str) -> tuple[str, str, int | None]:
    """'func (path/file.py:123)' -> ('func', 'path/file.py', 123). Frames
    without a parenthesised location come back as (frame, '', None)."""
    m = _FRAME_RE.match(frame.strip())
    if not m:
        return frame.strip(), "", None
    line = m.group("line")
    return m.group("func"), m.group("file"), int(line) if line else None


def frame_key(frame: str) -> str:
    """Function identity without the line number: 'func (path/file.py)'."""
    func, file, _ = parse_frame(frame)
    return f"{func} ({file})" if file else func


def parse_collapsed(text: str) -> list[tuple[list[str], int]]:
    """py-spy raw output -> [(frames root..leaf, count)]. Malformed lines are
    skipped; an empty stack (' 4') is kept with no frames (it still counts
    towards the sample total)."""
    out = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        stack, sep, cnt = line.rpartition(" ")
        if not sep:
            continue
        try:
            n = int(cnt)
        except ValueError:
            continue
        if n <= 0:
            continue
        frames = [f for f in stack.split(";") if f.strip()] if stack.strip() else []
        out.append((frames, n))
    return out


def is_project_file(file: str, roots: set) -> bool:
    if not file or file.startswith("<"):
        return False
    parts = [p for p in file.replace("\\", "/").split("/") if p]
    if not parts:
        return False
    if file.startswith("/"):      # --full-filenames style
        if "site-packages" in parts or "dist-packages" in parts:
            i = max(i for i, p in enumerate(parts) if p in ("site-packages", "dist-packages"))
            return i + 1 < len(parts) and parts[i + 1] in THIRD_PARTY_PROJECT
        return "web" in parts or "src" in parts
    return parts[0] in roots


def _top(counter: dict, total: int, n: int) -> list:
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [{"frame": k, "samples": v, "share": round(v / total, 4) if total else 0.0}
            for k, v in rows]


def summarise_pyspy(text: str, roots: set | None = None, top_inclusive: int = 15,
                    top_self: int = 10, passthrough_frac: float = 0.98) -> dict:
    """GIL-held sample summary of a ``py-spy record --gil --format raw`` dump.

    gil_top:   project/sam3 functions by INCLUSIVE share (a function counts at
               most once per stack, line numbers stripped): "of the time some
               thread held the GIL, how much was under this function". Pure
               pass-through frames are dropped (see below); each row also
               carries its self_share.
    self_top: leaf frames (any code) by SELF share: where the GIL time is
               actually spent.
    project_self_top: innermost project frame of each stack (who to blame).
    """
    roots = project_roots() if roots is None else roots
    stacks = parse_collapsed(text)
    total = sum(n for _, n in stacks)
    incl: dict = {}
    self_: dict = {}
    proj_self: dict = {}
    edge: dict = {}            # (F, next distinct function below F) -> samples
    proj_samples = 0
    for frames, n in stacks:
        if not frames:
            continue
        keys = [frame_key(f) for f in frames]
        isproj = [is_project_file(parse_frame(f)[1], roots) for f in frames]
        seen = set()
        inner_proj = None
        for i, k in enumerate(keys):
            if not isproj[i]:
                continue
            inner_proj = k
            if k in seen or i == 0:     # root frame (<module> (app.py)) = "main thread"
                continue
            seen.add(k)
            incl[k] = incl.get(k, 0) + n
            # next distinct PROJECT function below (through torch/stdlib frames)
            child = next((c for c, p in zip(keys[i + 1:], isproj[i + 1:]) if p and c != k), None)
            edge[(k, child)] = edge.get((k, child), 0) + n
        leaf = keys[-1]
        self_[leaf] = self_.get(leaf, 0) + n
        if inner_proj:
            proj_samples += n
            proj_self[inner_proj] = proj_self.get(inner_proj, 0) + n
    # Pure pass-throughs (>= passthrough_frac of a function's samples go on
    # to one and the same project callee: websocket_infer ->
    # _run_job_serialized -> process_video, sam3's handle_stream_request
    # chain) repeat the callee's number; keep only the deepest frame of such
    # a chain. The innermost project frame is never hidden.
    best_child: dict = {}
    for (k, child), v in edge.items():
        if child is not None and v > best_child.get(k, 0):
            best_child[k] = v
    hidden = {k for k, v in incl.items() if v and best_child.get(k, 0) >= passthrough_frac * v}
    shown = {k: v for k, v in incl.items() if k not in hidden}
    gil_top = _top(shown, total, top_inclusive)
    for r in gil_top:
        r["self_share"] = round(self_.get(r["frame"], 0) / total, 4) if total else 0.0
    return {
        "pyspy_samples": total,
        "pyspy_stacks": len(stacks),
        "project_share": round(proj_samples / total, 4) if total else 0.0,
        "gil_top": gil_top,
        "passthrough_hidden": len(hidden),
        "self_top": _top(self_, total, top_self),
        "project_self_top": _top(proj_self, total, top_self),
    }


# ------------------------------------------------------------------ GPU
def _num(s: str) -> float | None:
    s = s.strip().rstrip("%").replace("MiB", "").strip()
    if s in ("", "-", "N/A", "[N/A]", "[Not Supported]"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_gpu_samples(text: str) -> list[dict]:
    """GPU samples -> [{'gpu': idx, 'util': %, 'mem': MiB}] from either

    * ``nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used
      --format=csv[,nounits] -l 1`` (header line required; repeated headers
      and unit suffixes tolerated), or
    * ``nvidia-smi dmon -s um`` (``# gpu sm mem enc dec ... fb`` header).
    """
    out = []
    cols: list | None = None
    mode = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            names = line.lstrip("#").split()
            if "gpu" in names or "Idx" in names:
                if "sm" in names:
                    cols, mode = [n.lower() for n in names], "dmon"
            continue
        if "utilization.gpu" in line or "memory.used" in line:
            cols = [re.sub(r"\s*\[.*?\]", "", c).strip().lower() for c in line.split(",")]
            mode = "csv"
            continue
        if cols is None:
            continue
        if mode == "csv":
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(cols):
                continue
            row = dict(zip(cols, vals))
            util = _num(row.get("utilization.gpu", ""))
            mem = _num(row.get("memory.used", ""))
            idx = _num(row.get("index", "0"))
        else:
            vals = line.split()
            if len(vals) != len(cols):
                continue
            row = dict(zip(cols, vals))
            util = _num(row.get("sm", ""))
            mem = _num(row.get("fb", "")) if "fb" in row else None
            idx = _num(row.get("gpu", row.get("idx", "0")))
        if util is None and mem is None:
            continue
        out.append({"gpu": int(idx or 0), "util": util, "mem": mem})
    return out


def _median(xs: list) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def summarise_gpu(samples: list[dict], busy_pct: float = 50.0) -> dict:
    """Mean / median util, fraction of samples >= busy_pct, peak memory, for
    the lowest-index GPU (the pod has one)."""
    if not samples:
        return {"gpu_samples": 0}
    g0 = min(s["gpu"] for s in samples)
    rows = [s for s in samples if s["gpu"] == g0]
    utils = [s["util"] for s in rows if s["util"] is not None]
    mems = [s["mem"] for s in rows if s["mem"] is not None]
    out = {"gpu_samples": len(rows)}
    if utils:
        out.update(gpu_util_mean=round(sum(utils) / len(utils), 1),
                   gpu_util_p50=round(_median(utils), 1),
                   gpu_busy_frac=round(sum(u >= busy_pct for u in utils) / len(utils), 3))
    if mems:
        out["gpu_mem_peak_mib"] = int(max(mems))
    return out


# ------------------------------------------------------------------ /proc
def listen_inodes(proc_net_tcp: str, port: int) -> set:
    """Socket inodes LISTENing on ``port`` from /proc/net/tcp{,6} text."""
    inodes = set()
    for line in proc_net_tcp.splitlines()[1:]:
        f = line.split()
        if len(f) < 10:
            continue
        local, state, inode = f[1], f[3], f[9]
        try:
            if state == "0A" and int(local.rsplit(":", 1)[1], 16) == port and inode != "0":
                inodes.add(inode)
        except (IndexError, ValueError):
            continue
    return inodes


def proc_cpu_ticks(stat_text: str) -> int | None:
    """utime+stime+cutime+cstime (clock ticks) from /proc/<pid>/stat."""
    try:
        rest = stat_text[stat_text.rindex(")") + 2:].split()
        # rest[0] is field 3 (state); utime..cstime are fields 14..17
        return sum(int(x) for x in rest[11:15])
    except (ValueError, IndexError):
        return None


def _cmdline(proc: Path, pid: str) -> str:
    try:
        return (proc / pid / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def find_server_pid(port: int, proc_root: str | Path = "/proc",
                    pidfile: str | Path | None = None, cmd_hint: str = "app.py") -> tuple[int | None, str]:
    """PID of the app server: (1) owner of the LISTEN socket on ``port``
    (/proc/net/tcp + /proc/*/fd, needs no ss/lsof), (2) serve.sh's pid file,
    (3) a process whose cmdline mentions ``cmd_hint``. Returns (pid, how)."""
    proc = Path(proc_root)
    inodes = set()
    for name in ("tcp", "tcp6"):
        try:
            inodes |= listen_inodes((proc / "net" / name).read_text(), port)
        except OSError:
            pass
    pids = [p.name for p in proc.iterdir() if p.name.isdigit()] if proc.is_dir() else []
    if inodes:
        owners = []
        targets = {f"socket:[{i}]" for i in inodes}
        for pid in pids:
            try:
                for fd in (proc / pid / "fd").iterdir():
                    try:
                        if os.readlink(fd) in targets:
                            owners.append(int(pid))
                            break
                    except OSError:
                        continue
            except OSError:
                continue
        if owners:
            hinted = [p for p in owners if cmd_hint in _cmdline(proc, str(p))]
            return min(hinted or owners), "port"
    if pidfile:
        try:
            pid = int(Path(pidfile).read_text().strip())
            if (proc / str(pid)).exists():
                return pid, "pidfile"
        except (OSError, ValueError):
            pass
    cands = sorted(int(p) for p in pids if cmd_hint in _cmdline(proc, p)
                   and "python" in _cmdline(proc, p) and "pod_job" not in _cmdline(proc, p))
    if cands:
        return cands[0], "cmdline"
    return None, "not found"


def kill_stale(match_all: tuple, sig: int = signal.SIGINT, proc_root: str | Path = "/proc") -> list:
    """Signal leftover profilers from a killed pod_job (cmdline contains every
    string in ``match_all``). Returns the pids signalled."""
    proc = Path(proc_root)
    hit = []
    me = os.getpid()
    for p in proc.iterdir() if proc.is_dir() else []:
        if not p.name.isdigit() or int(p.name) == me:
            continue
        cmd = _cmdline(proc, p.name)
        if cmd and all(m in cmd for m in match_all):
            try:
                os.kill(int(p.name), sig)
                hit.append(int(p.name))
            except OSError:
                pass
    return hit


# ------------------------------------------------------------------ runner (pod side)
def pyspy_binary(app_dir: Path = APP_DIR) -> str | None:
    cand = Path(app_dir) / ".venv" / "bin" / "py-spy"
    if cand.exists():
        return str(cand)
    return shutil.which("py-spy")


class JobProfiler:
    """py-spy (GIL samples) + nvidia-smi (1 Hz util/memory) for one job.

    start() right before the job is submitted, stop() once its final result
    exists; summary() is the rung's ``profile`` dict. Raw files:
    ``<dir>/<label>.pyspy.txt`` (collapsed stacks), ``<label>.pyspy.log``,
    ``<label>.gpu.csv``. py-spy gets a hard ``--duration`` cap so an orphan
    (pod_job killed) can never sample forever; pod_job also SIGINTs stale
    profilers for the same files before starting new ones.
    """

    def __init__(self, out_dir: Path | str, label: str, port: int, *, rate: int = 20,
                 pidfile: str | None = None, pyspy: str | None = None,
                 nvidia_smi: str = "nvidia-smi", max_s: int = 7200, app_dir: Path = APP_DIR,
                 startup_check_s: float = 3.0):
        self.dir = Path(out_dir)
        self.label = label
        self.port = port
        self.rate = rate
        self.pidfile = pidfile
        self.pyspy = pyspy if pyspy is not None else pyspy_binary(app_dir)
        self.nvidia_smi = nvidia_smi
        self.max_s = max_s
        self.app_dir = Path(app_dir)
        self.startup_check_s = startup_check_s
        self.pyspy_txt = self.dir / f"{label}.pyspy.txt"
        self.pyspy_log = self.dir / f"{label}.pyspy.log"
        self.gpu_csv = self.dir / f"{label}.gpu.csv"
        self._py: subprocess.Popen | None = None
        self._gpu: subprocess.Popen | None = None
        self._pid: int | None = None
        self._t0 = self._t1 = None
        self._ticks0 = None
        self._ticks1 = None
        self._stopped = False
        self.info: dict = {"pyspy_rate": rate}

    # -- helpers
    def _ticks(self):
        if not self._pid:
            return None
        try:
            return proc_cpu_ticks(Path(f"/proc/{self._pid}/stat").read_text())
        except OSError:
            return None

    def _launch_pyspy(self, nonblocking: bool) -> subprocess.Popen:
        cmd = [self.pyspy, "record", "--gil", "--rate", str(self.rate), "--format", "raw",
               "--duration", str(self.max_s), "--output", str(self.pyspy_txt), "--pid", str(self._pid)]
        if nonblocking:
            cmd.append("--nonblocking")
        log = open(self.pyspy_log, "ab")
        try:
            return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        finally:
            log.close()

    def _log_tail(self, n: int = 400) -> str:
        try:
            return self.pyspy_log.read_text(errors="replace")[-n:].strip()
        except OSError:
            return ""

    # -- lifecycle
    def start(self) -> dict:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            for f in (self.pyspy_txt, self.pyspy_log, self.gpu_csv):
                f.unlink(missing_ok=True)
            stale = kill_stale(("py-spy", "record", str(self.pyspy_txt)))
            stale += kill_stale(("nvidia-smi", str(self.gpu_csv)), sig=signal.SIGTERM)
            if stale:
                self.info["stale_profilers_killed"] = stale
                time.sleep(1)
        except Exception as e:  # never fail the job
            self.info["profile_error"] = f"setup: {type(e).__name__}: {e}"[:300]
        self._t0 = time.time()
        self._start_pyspy()
        self._start_gpu()
        return {k: v for k, v in (("pyspy_pid", self._py.pid if self._py else None),
                                  ("gpu_log_pid", self._gpu.pid if self._gpu else None),
                                  ("server_pid", self._pid)) if v}

    def _start_pyspy(self) -> None:
        try:
            self._pid, how = find_server_pid(self.port, pidfile=self.pidfile)
            self.info["server_pid"], self.info["server_pid_from"] = self._pid, how
            if not self._pid:
                self.info["pyspy_error"] = f"no server process listening on :{self.port}"
                return
            self._ticks0 = self._ticks()
            if not self.pyspy:
                self.info["pyspy_error"] = "py-spy not installed (.venv/bin/py-spy or PATH)"
                return
            for nonblocking in (False, True):
                self._py = self._launch_pyspy(nonblocking)
                try:
                    rc = self._py.wait(timeout=self.startup_check_s)
                except subprocess.TimeoutExpired:
                    self.info["pyspy_mode"] = "nonblocking" if nonblocking else "blocking"
                    self.info.pop("pyspy_error", None)
                    return
                self.info["pyspy_error"] = (f"py-spy exited rc={rc} "
                                            f"({'nonblocking' if nonblocking else 'blocking'}): "
                                            f"{self._log_tail()}")[:500]
                self._py = None
        except Exception as e:
            self._py = None
            self.info["pyspy_error"] = f"{type(e).__name__}: {e}"[:500]

    def _start_gpu(self) -> None:
        try:
            if not shutil.which(self.nvidia_smi):
                self.info["gpu_error"] = f"{self.nvidia_smi} not found"
                return
            self._gpu = subprocess.Popen(
                [self.nvidia_smi, "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                 "--format=csv,nounits", "-l", "1", "-f", str(self.gpu_csv)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        except Exception as e:
            self._gpu = None
            self.info["gpu_error"] = f"{type(e).__name__}: {e}"[:300]

    def stop(self) -> None:
        """Idempotent. SIGINT py-spy (it flushes the raw file on Ctrl-C), stop
        the GPU logger."""
        if self._stopped:
            return
        self._stopped = True
        self._t1 = time.time()
        self._ticks1 = self._ticks()
        if self._py is not None:
            try:
                if self._py.poll() is None:
                    self._py.send_signal(signal.SIGINT)
                    try:
                        self._py.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        self._py.kill()
                        self._py.wait(timeout=10)
                        self.info["pyspy_error"] = "py-spy did not exit 60 s after SIGINT (killed)"
                elif self._py.returncode not in (0, None):
                    self.info["pyspy_error"] = f"py-spy exited early rc={self._py.returncode}: {self._log_tail()}"[:500]
            except Exception as e:
                self.info["pyspy_error"] = f"stop: {type(e).__name__}: {e}"[:300]
        if self._gpu is not None:
            try:
                self._gpu.terminate()
                self._gpu.wait(timeout=10)
            except Exception:
                try:
                    self._gpu.kill()
                except Exception:
                    pass

    def summary(self) -> dict:
        """Stop (if needed) and summarise; never raises."""
        try:
            self.stop()
        except Exception as e:
            self.info["profile_error"] = f"stop: {e}"[:300]
        out = dict(self.info)
        if self._t0 and self._t1:
            out["profiled_s"] = round(self._t1 - self._t0, 1)
            if self._ticks0 is not None and self._ticks1 is not None and self._t1 > self._t0:
                hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
                out["server_cpu_cores_mean"] = round((self._ticks1 - self._ticks0) / hz
                                                     / (self._t1 - self._t0), 2)
        try:
            if self.pyspy_txt.exists():
                s = summarise_pyspy(self.pyspy_txt.read_text(errors="replace"),
                                    project_roots(self.app_dir))
                out.update(s)
                if out.get("profiled_s") and self.rate:
                    # with --gil only GIL-holding samples are kept
                    out["gil_held_frac"] = round(s["pyspy_samples"] / (out["profiled_s"] * self.rate), 3)
            elif "pyspy_error" not in out and self._py is not None:
                out["pyspy_error"] = f"no output file: {self._log_tail()}"[:500]
        except Exception as e:
            out["pyspy_error"] = f"summarise: {type(e).__name__}: {e}"[:300]
        try:
            if self.gpu_csv.exists():
                out.update(summarise_gpu(parse_gpu_samples(self.gpu_csv.read_text(errors="replace"))))
        except Exception as e:
            out["gpu_error"] = f"summarise: {type(e).__name__}: {e}"[:300]
        out["files"] = [p.name for p in (self.pyspy_txt, self.gpu_csv) if p.exists()]
        return out


# ------------------------------------------------------------------ reports (local)
def format_profile_report(name: str, prof: dict) -> str:
    if not prof:
        return f"{name}: no profile recorded (run without --no-profile, or `pull` + re-summarise)"
    head = [f"{prof.get('pyspy_samples', 0)} GIL samples"
            + (f" (GIL held {100 * prof['gil_held_frac']:.0f}% of wall)" if prof.get("gil_held_frac") is not None else "")]
    if prof.get("profiled_s") is not None:
        head.insert(0, f"profiled {prof['profiled_s']} s")
    if prof.get("pyspy_mode"):
        head.append(f"py-spy {prof['pyspy_mode']} @ {prof.get('pyspy_rate', '?')} Hz")
    if prof.get("server_cpu_cores_mean") is not None:
        head.append(f"server CPU {prof['server_cpu_cores_mean']} cores avg")
    lines = [f"profile {name}: " + ", ".join(head)]
    if prof.get("gpu_samples"):
        lines.append(f"GPU: util mean {prof.get('gpu_util_mean', '-')}% p50 {prof.get('gpu_util_p50', '-')}% "
                     f"busy(>=50%) {_pct(prof.get('gpu_busy_frac'))} peak mem "
                     f"{prof.get('gpu_mem_peak_mib', '-')} MiB ({prof['gpu_samples']} samples)")
    for k in ("pyspy_error", "gpu_error", "profile_error"):
        if prof.get(k):
            lines.append(f"{k}: {prof[k]}")
    for title, key in (("project/sam3 functions by INCLUSIVE share of GIL samples", "gil_top"),
                       ("leaf frames by SELF share", "self_top"),
                       ("innermost project frame (self)", "project_self_top")):
        rows = prof.get(key) or []
        if not rows:
            continue
        lines.append(f"\n  {title}:")
        if key == "gil_top":
            lines.append(f"  {'incl':>6} {'self':>6} {'samples':>7}  function")
        for r in rows:
            selfcol = f"{100 * r['self_share']:5.1f}% " if "self_share" in r else ""
            lines.append(f"  {100 * r['share']:5.1f}% {selfcol}{r['samples']:>7}  {r['frame']}")
    return "\n".join(lines)


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def format_host_line(host: dict) -> str:
    """One-line host fingerprint for the start of `tune`."""
    if not host:
        return "host: (no fingerprint)"
    if host.get("error"):
        return f"host: fingerprint failed: {host['error']}"
    cpu = host.get("cpu") or {}
    b = host.get("bench") or {}
    gpu = host.get("gpu") or {}
    quota = cpu.get("cgroup_cpu_quota")
    parts = [
        f"{cpu.get('model', '?')}"
        + (f" (max {cpu['mhz_max']:.0f} MHz)" if cpu.get("mhz_max") else ""),
        f"{cpu.get('affinity', '?')}/{cpu.get('logical', '?')} cpus"
        + (f", quota {quota:g}" if quota else ", no quota"),
        f"{host.get('ram_gb', '?')} GB RAM"
        + (f" (cgroup {host['cgroup_mem_gb']} GB)" if host.get("cgroup_mem_gb") else ""),
    ]
    if gpu.get("name"):
        parts.append(f"{gpu['name']} drv {gpu.get('driver', '?')}")
    if b:
        parts.append(f"1T: py {b.get('py_mops', '?')} Mops/s, SAM3 frame load "
                     f"{b.get('load_ms_per_frame', '?')} ms (PIL {b.get('pil_ms_per_frame', '?')} ms); "
                     f"{b.get('load_threads', '?')}T load {b.get('thread_speedup', '?')}x")
    return "host: " + " | ".join(parts)

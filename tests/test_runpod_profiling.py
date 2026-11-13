"""deploy/runpod profiling: py-spy / nvidia-smi parsing + summaries, the
JobProfiler lifecycle (fake py-spy / nvidia-smi, no GPU, no pod), the host
fingerprint script, and the tune CLI glue (no network, no pods)."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNPOD = REPO / "deploy" / "runpod"
PY = sys.executable


def _load(name):
    sys.path.insert(0, str(RUNPOD))
    spec = importlib.util.spec_from_file_location(name, RUNPOD / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prof = _load("profiling")
hb = _load("host_bench")
cli = _load("runpod_cli")

ROOTS = {"app.py", "field", "students", "sam3", "sam2"}


# ----------------------------------------------------------------- collapsed stacks
def test_parse_frame_and_key():
    assert prof.parse_frame("occluded_by_hub (field/occlusion.py:149)") == \
        ("occluded_by_hub", "field/occlusion.py", 149)
    assert prof.parse_frame("_load_unlocked (<frozen importlib._bootstrap>:935)") == \
        ("_load_unlocked", "<frozen importlib._bootstrap>", 935)
    assert prof.parse_frame("f (x.py)") == ("f", "x.py", None)          # --nolineno
    assert prof.parse_frame("weird") == ("weird", "", None)
    # nested parens in the function name (<locals>, <genexpr>) survive
    k = prof.frame_key("InferenceJob.process_video.<locals>._hub (app.py:12)")
    assert k == "InferenceJob.process_video.<locals>._hub (app.py)"


def test_parse_collapsed_tolerates_junk():
    text = ("a (app.py:1);b (field/x.py:2) 5\n"
            " 4\n"                      # empty stack (real Nov 12 file has one)
            "garbage line without count\n"
            "\n"
            "a (app.py:1);c (torch/nn.py:3) 0\n"
            "a (app.py:1);name with spaces (field/y.py:9) 2\n")
    st = prof.parse_collapsed(text)
    assert st == [(["a (app.py:1)", "b (field/x.py:2)"], 5), ([], 4),
                  (["a (app.py:1)", "name with spaces (field/y.py:9)"], 2)]


def test_is_project_file():
    assert prof.is_project_file("field/occlusion.py", ROOTS)
    assert prof.is_project_file("app.py", ROOTS)
    assert prof.is_project_file("sam3/model/io_utils.py", ROOTS)
    assert not prof.is_project_file("torch/nn/modules/module.py", ROOTS)
    assert not prof.is_project_file("<frozen importlib._bootstrap>", ROOTS)
    assert not prof.is_project_file("threading.py", ROOTS)
    # --full-filenames style paths
    assert prof.is_project_file("/x/.venv/lib/python3.12/site-packages/sam3/a.py", ROOTS)
    assert not prof.is_project_file("/x/.venv/lib/python3.12/site-packages/torch/a.py", ROOTS)
    assert prof.is_project_file("/workspace/frc/web/field/occlusion.py", ROOTS)


def test_project_roots_from_checkout():
    roots = prof.project_roots(REPO)
    assert {"app.py", "field", "students", "sam3", "sam2"} <= roots
    assert "torch" not in roots
    # no checkout: falls back to the known layout
    assert {"app.py", "field"} <= prof.project_roots(Path("/nonexistent"))


SYNTH = textwrap.dedent("""\
    <module> (app.py:10);run (uvicorn/main.py:1);job (app.py:20);hot (field/occ.py:5);pip (field/occ.py:9) 60
    <module> (app.py:10);run (uvicorn/main.py:1);job (app.py:20);hot (field/occ.py:6) 10
    <module> (app.py:10);run (uvicorn/main.py:1);job (app.py:20);sam (sam3/x.py:1);fwd (torch/nn.py:3) 20
    Thread._bootstrap (threading.py:1);worker (field/occ.py:30);rec (field/r.py:1);rec (field/r.py:2) 10
""")


def test_summarise_pyspy_inclusive_self_and_passthrough():
    s = prof.summarise_pyspy(SYNTH, ROOTS)
    assert s["pyspy_samples"] == 100
    assert s["project_share"] == 1.0
    incl = {r["frame"]: r for r in s["gil_top"]}
    # root <module> never ranked; pip/hot inclusive; recursion counted once
    assert "<module> (app.py)" not in incl
    assert incl["hot (field/occ.py)"]["samples"] == 70
    assert incl["pip (field/occ.py)"]["samples"] == 60
    assert incl["sam (sam3/x.py)"]["samples"] == 20
    assert incl["rec (field/r.py)"]["samples"] == 10           # 2 frames, 1 count
    # job -> hot 70/90, -> sam 20/90: not a pass-through, kept
    assert incl["job (app.py)"]["samples"] == 90
    # worker -> rec 100%: pure pass-through, hidden behind its callee
    assert "worker (field/occ.py)" not in incl
    assert s["passthrough_hidden"] >= 1
    assert s["gil_top"][0]["frame"] == "job (app.py)"
    assert incl["hot (field/occ.py)"]["self_share"] == 0.1
    self_ = {r["frame"]: r["samples"] for r in s["self_top"]}
    assert self_ == {"pip (field/occ.py)": 60, "fwd (torch/nn.py)": 20,
                     "hot (field/occ.py)": 10, "rec (field/r.py)": 10}
    assert s["project_self_top"][0] == {"frame": "pip (field/occ.py)", "samples": 60, "share": 0.6}
    assert {"frame": "sam (sam3/x.py)", "samples": 20, "share": 0.2} in s["project_self_top"]


def test_summarise_pyspy_empty():
    s = prof.summarise_pyspy("", ROOTS)
    assert s["pyspy_samples"] == 0 and s["gil_top"] == [] and s["project_share"] == 0.0


NOV12 = cli.main_checkout(REPO) / "data" / "runpod_pull" / "1wjnhb11v7bgle" / "tune" / "pyspy_R1_gil.txt"


@pytest.mark.skipif(not NOV12.exists(), reason="Nov 12 pod pull not present")
def test_nov12_real_profile_finds_occlusion_hotspot():
    s = prof.summarise_pyspy(NOV12.read_text(errors="replace"), prof.project_roots(REPO))
    assert s["pyspy_samples"] == 2809
    top = [r["frame"] for r in s["gil_top"]]
    # below the job driver (process_video) and the tracker loop, the hub
    # occlusion test is the biggest project frame (~1/3 of GIL time) ...
    assert top[:3] == ["InferenceJob.process_video (app.py)",
                       "Sam2FieldTracker.track_selective (field/sam2_tracker.py)",
                       "occluded_by_hub (field/occlusion.py)"]
    occ = s["gil_top"][2]
    assert 0.30 < occ["share"] < 0.36
    # ... and its point-in-polygon loop is where the GIL time is spent
    assert s["self_top"][0]["frame"] == "_point_in_poly (field/occlusion.py)"
    assert s["project_self_top"][0]["frame"] == "_point_in_poly (field/occlusion.py)"
    assert "<module> (app.py)" not in top
    rep = prof.format_profile_report("R1", s)
    assert "occluded_by_hub (field/occlusion.py)" in rep and "INCLUSIVE" in rep


# ----------------------------------------------------------------- GPU samples
CSV_NOUNITS = """timestamp, index, utilization.gpu [%], memory.used [MiB]
2026/11/12 21:30:01.123, 0, 10, 9000
2026/11/12 21:30:02.123, 0, 60, 12000
2026/11/12 21:30:03.123, 0, 80, 15000
2026/11/12 21:30:04.123, 0, [N/A], 15100
"""

CSV_UNITS = """timestamp, index, utilization.gpu [%], memory.used [MiB]
2026/11/12 21:30:01.123, 0, 0 %, 100 MiB
2026/11/12 21:30:01.123, 1, 99 %, 50000 MiB
timestamp, index, utilization.gpu [%], memory.used [MiB]
2026/11/12 21:30:02.123, 0, 100 %, 300 MiB
"""

DMON = """# gpu     sm    mem    enc    dec    jpg    ofa     fb   bar1   ccpm
# Idx      %      %      %      %      %      %     MB     MB     MB
    0     77     30      0      0      0      0  14000      5      0
    0      -      -      -      -      -      -      -      -      -
    0     20      5      0      0      0      0  15000      5      0
"""


def test_parse_gpu_csv_nounits():
    s = prof.parse_gpu_samples(CSV_NOUNITS)
    assert [x["util"] for x in s] == [10, 60, 80, None]
    g = prof.summarise_gpu(s)
    assert g == {"gpu_samples": 4, "gpu_util_mean": 50.0, "gpu_util_p50": 60.0,
                 "gpu_busy_frac": 0.667, "gpu_mem_peak_mib": 15100}


def test_parse_gpu_csv_units_multi_gpu_uses_first():
    s = prof.parse_gpu_samples(CSV_UNITS)
    assert len(s) == 3
    g = prof.summarise_gpu(s)
    assert g["gpu_samples"] == 2 and g["gpu_util_mean"] == 50.0 and g["gpu_mem_peak_mib"] == 300


def test_parse_gpu_dmon():
    s = prof.parse_gpu_samples(DMON)
    assert [(x["util"], x["mem"]) for x in s] == [(77, 14000), (20, 15000)]
    g = prof.summarise_gpu(s)
    assert g["gpu_util_mean"] == 48.5 and g["gpu_busy_frac"] == 0.5 and g["gpu_mem_peak_mib"] == 15000


def test_summarise_gpu_empty():
    assert prof.summarise_gpu([]) == {"gpu_samples": 0}
    assert prof.parse_gpu_samples("10, 2000\nno header\n") == []


# ----------------------------------------------------------------- /proc helpers
def test_listen_inodes():
    text = textwrap.dedent("""\
          sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
           0: 00000000:1F44 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 4242 1 0
           1: 0100007F:1F44 0100007F:9C40 01 00000000:00000000 00:00000000 00000000     0        0 5555 1 0
           2: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 77 1 0
    """)
    assert prof.listen_inodes(text, 8004) == {"4242"}     # 0x1F44 = 8004, LISTEN only
    assert prof.listen_inodes(text, 22) == {"77"}
    assert prof.listen_inodes("header only\n", 8004) == set()


def test_proc_cpu_ticks():
    # comm with spaces and parens; utime..cstime = 100, 20, 3, 4
    stat = "1234 (python web (app).py) S 1 1234 1234 0 -1 4194560 100 0 0 0 100 20 3 4 20 0 30 0"
    assert prof.proc_cpu_ticks(stat) == 127
    assert prof.proc_cpu_ticks("garbage") is None


def test_find_server_pid_fake_proc(tmp_path):
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net" / "tcp").write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        "   0: 00000000:1F44 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 999 1 0\n")
    for pid, cmd, link in (("50", "bash\0serve.sh", "socket:[1]"),
                           ("300", "/w/.venv/bin/python\0web/app.py", "socket:[999]"),
                           ("400", "/w/.venv/bin/python\0deploy/runpod/pod_job.py", "pipe:[3]")):
        d = proc / pid
        (d / "fd").mkdir(parents=True)
        (d / "cmdline").write_text(cmd)
        os.symlink(link, d / "fd" / "3")
    assert prof.find_server_pid(8004, proc_root=proc) == (300, "port")
    # nobody on the port -> pid file -> cmdline
    (proc / "net" / "tcp").write_text("header\n")
    pidf = tmp_path / "server.pid"
    pidf.write_text("300\n")
    assert prof.find_server_pid(8004, proc_root=proc, pidfile=pidf) == (300, "pidfile")
    assert prof.find_server_pid(8004, proc_root=proc, pidfile=tmp_path / "nope") == (300, "cmdline")
    (proc / "300" / "cmdline").write_text("other")
    assert prof.find_server_pid(8004, proc_root=proc) == (None, "not found")


def test_find_server_pid_real_socket():
    """The real /proc path: a child process listening on a free port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    child = subprocess.Popen([PY, "-c", f"import socket,time; s=socket.socket(); "
                                        f"s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                                        f"s.bind(('127.0.0.1',{port})); s.listen(); print('up', flush=True); "
                                        f"time.sleep(30)"], stdout=subprocess.PIPE)
    try:
        assert child.stdout.readline().strip() == b"up"
        assert prof.find_server_pid(port, cmd_hint="socket") == (child.pid, "port")
    finally:
        child.kill()
        child.wait()


def test_kill_stale_signals_matching_processes():
    tag = f"frc-stale-test-{os.getpid()}"
    child = subprocess.Popen([PY, "-c", "import time; time.sleep(30)", tag])
    try:
        time.sleep(0.2)
        hit = prof.kill_stale((tag,), sig=signal.SIGTERM)
        assert hit == [child.pid]
        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:
            child.kill()


# ----------------------------------------------------------------- JobProfiler (fakes)
FAKE_PYSPY = textwrap.dedent("""\
    #!{py}
    import os, signal, sys, time
    a = sys.argv
    out = a[a.index("--output") + 1]
    mode = os.environ.get("FAKE_PYSPY", "ok")
    nb = "--nonblocking" in a
    if mode == "fail" or (mode == "fail_blocking" and not nb):
        print("Permission Denied: Try running again with elevated permissions")
        sys.exit(1)
    def flush(*_):
        with open(out, "w") as f:
            f.write("<module> (app.py:1);job (app.py:2);hot (field/occ.py:3) 30\\n")
            f.write("<module> (app.py:1);job (app.py:2);fwd (torch/nn.py:3) 10\\n")
        print("Wrote raw flamegraph data")
        sys.exit(0)
    signal.signal(signal.SIGINT, flush)
    while True:
        time.sleep(0.05)
""")

FAKE_SMI = textwrap.dedent("""\
    #!{py}
    import signal, sys, time
    a = sys.argv
    out = a[a.index("-f") + 1]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with open(out, "w") as f:
        f.write("timestamp, index, utilization.gpu [%], memory.used [MiB]\\n")
        i = 0
        while True:
            f.write(f"2026/11/13 10:00:0{{i % 10}}.000, 0, {{40 + 20 * (i % 2)}}, {{1000 + i}}\\n")
            f.flush()
            i += 1
            time.sleep(0.05)
""")


@pytest.fixture
def fakes(tmp_path):
    ps = tmp_path / "py-spy"
    ps.write_text(FAKE_PYSPY.format(py=PY))
    ps.chmod(0o755)
    smi = tmp_path / "nvidia-smi"
    smi.write_text(FAKE_SMI.format(py=PY))
    smi.chmod(0o755)
    # a "server" listening on a free port
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = subprocess.Popen([PY, "-c", f"import socket,time; s=socket.socket(); "
                                      f"s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                                      f"s.bind(('127.0.0.1',{port})); s.listen(); print('up', flush=True); "
                                      f"[sum(range(10000)) for _ in iter(int, 1)]"], stdout=subprocess.PIPE)
    assert srv.stdout.readline().strip() == b"up"
    yield {"pyspy": str(ps), "smi": str(smi), "port": port, "srv": srv, "dir": tmp_path / "jobs"}
    srv.kill()
    srv.wait()


def _run_profiler(f, monkeypatch, mode):
    monkeypatch.setenv("FAKE_PYSPY", mode)
    p = prof.JobProfiler(f["dir"], "R9", f["port"], pyspy=f["pyspy"], nvidia_smi=f["smi"],
                         startup_check_s=0.5, app_dir=REPO)
    started = p.start()
    time.sleep(0.6)
    s = p.summary()
    p.stop()                       # idempotent
    return started, s


def test_job_profiler_blocking_ok(fakes, monkeypatch):
    started, s = _run_profiler(fakes, monkeypatch, "ok")
    assert started["server_pid"] == fakes["srv"].pid and started["pyspy_pid"] and started["gpu_log_pid"]
    assert s["server_pid_from"] == "port" and s["pyspy_mode"] == "blocking"
    assert "pyspy_error" not in s and "gpu_error" not in s
    assert s["pyspy_samples"] == 40
    assert s["gil_top"][0]["frame"] == "job (app.py)" or s["gil_top"][0]["frame"] == "hot (field/occ.py)"
    assert {r["frame"] for r in s["gil_top"]} >= {"hot (field/occ.py)"}
    assert s["gpu_samples"] >= 2 and 40 <= s["gpu_util_mean"] <= 60
    assert s["server_cpu_cores_mean"] > 0.2         # the fake server spins
    assert s["profiled_s"] >= 0.5
    assert sorted(s["files"]) == ["R9.gpu.csv", "R9.pyspy.txt"]
    assert (fakes["dir"] / "R9.pyspy.txt").exists()


def test_job_profiler_falls_back_to_nonblocking(fakes, monkeypatch):
    _, s = _run_profiler(fakes, monkeypatch, "fail_blocking")
    assert s["pyspy_mode"] == "nonblocking" and "pyspy_error" not in s
    assert s["pyspy_samples"] == 40


def test_job_profiler_ptrace_denied_is_recorded_not_raised(fakes, monkeypatch):
    _, s = _run_profiler(fakes, monkeypatch, "fail")
    assert "Permission Denied" in s["pyspy_error"]
    assert "pyspy_samples" not in s
    assert s["gpu_samples"] >= 1          # GPU half still works


def test_job_profiler_without_tools(tmp_path):
    p = prof.JobProfiler(tmp_path, "R0", 1, pyspy="", nvidia_smi="/nonexistent/nvidia-smi")
    p.start()
    s = p.summary()
    assert s["pyspy_error"] and s["gpu_error"] and s["files"] == []


# ----------------------------------------------------------------- host fingerprint
def test_host_bench_parsers():
    assert hb.parse_cgroup_v2_cpu_max("max 100000\n") is None
    assert hb.parse_cgroup_v2_cpu_max("400000 100000\n") == 4.0
    assert hb.parse_cgroup_v2_cpu_max("150000 100000") == 1.5
    assert hb.parse_cgroup_v1_quota("-1\n", "100000\n") is None
    assert hb.parse_cgroup_v1_quota("800000", "100000") == 8.0
    assert hb.parse_meminfo_gb("MemTotal:       32000000 kB\nMemFree: 1 kB\n") == 30.5
    ci = hb.parse_cpuinfo("processor\t: 0\nmodel name\t: AMD EPYC 9354 32-Core Processor\ncpu MHz\t\t: 3000.0\n"
                          "processor\t: 1\nmodel name\t: AMD EPYC 9354 32-Core Processor\ncpu MHz\t\t: 1000.0\n")
    assert ci == {"model": "AMD EPYC 9354 32-Core Processor", "mhz_now_mean": 2000}
    ls = hb.parse_lscpu("Model name:            AMD EPYC 9354 32-Core Processor\nCPU max MHz:           3799.0703\n"
                        "Socket(s):             2\nCore(s) per socket:    32\nThread(s) per core:    2\n")
    assert ls == {"model": "AMD EPYC 9354 32-Core Processor", "mhz_max": 3799.0703, "sockets": 2,
                  "cores_per_socket": 32, "threads_per_core": 2}


def test_host_bench_runs_locally_json_shape():
    r = subprocess.run([PY, str(RUNPOD / "host_bench.py"), "--seconds", "0.6", "--no-gpu", "--threads", "2"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 1
    h = json.loads(lines[0])
    assert h["gpu"] == {"skipped": True}
    cpu = h["cpu"]
    assert cpu["model"] and cpu["logical"] >= 1 and 1 <= cpu["affinity"] <= cpu["logical"]
    assert "cgroup_cpu_quota" in cpu and cpu["cgroup"] in ("v1", "v2", "none")
    assert h["ram_gb"] > 0
    b = h["bench"]
    assert b["py_mops"] > 0 and b["jpeg_bytes"] > 10_000
    assert b["pil_ms_per_frame"] > 0 and b["load_ms_per_frame"] >= b["pil_ms_per_frame"] * 0.5
    assert b["load_op"] in ("torchvision", "numpy", "pil")
    assert b["load_threads"] == 2 and b["thread_speedup"] > 0
    line = prof.format_host_line(h)
    assert line.startswith("host: ") and cpu["model"] in line and "SAM3 frame load" in line


def test_host_bench_jpeg_is_deterministic():
    assert hb.make_jpeg() == hb.make_jpeg()


def test_format_host_line_error_and_quota():
    assert "failed" in prof.format_host_line({"error": "boom"})
    line = prof.format_host_line({"cpu": {"model": "EPYC", "affinity": 16, "logical": 128,
                                          "cgroup_cpu_quota": 16.0, "mhz_max": 3800.0},
                                  "ram_gb": 1007.6, "cgroup_mem_gb": 180.0,
                                  "gpu": {"name": "RTX PRO 6000", "driver": "570.1"},
                                  "bench": {"py_mops": 14.0, "load_ms_per_frame": 54.0,
                                            "pil_ms_per_frame": 20.0, "load_threads": 16,
                                            "thread_speedup": 9.5}})
    assert "16/128 cpus, quota 16" in line and "cgroup 180.0 GB" in line
    assert "RTX PRO 6000 drv 570.1" in line and "54.0 ms" in line and "16T load 9.5x" in line


# ----------------------------------------------------------------- CLI glue
def test_parse_host_bench_output():
    assert cli.parse_host_bench_output('warn\n{"a": 1}\n') == {"a": 1}
    assert "error" in cli.parse_host_bench_output("Traceback ...\n")
    assert "error" in cli.parse_host_bench_output("")


def test_tune_member_ok():
    assert cli.tune_member_ok("tune/jobs/R1.pyspy.txt")
    assert cli.tune_member_ok("tune/jobs/R1.gpu.csv")
    assert cli.tune_member_ok("tune/host.json")
    assert not cli.tune_member_ok("tune/2026cur_qm8.mp4")
    assert not cli.tune_member_ok("tune/jobs/R1.tmp")
    assert not cli.tune_member_ok("tune/../etc/passwd")
    assert not cli.tune_member_ok("web/replays/x/meta.json")


def test_tune_parser_profile_flags():
    a = cli.build_parser().parse_args(["tune"])
    assert a.no_profile is False and a.profile_report is None
    a = cli.build_parser().parse_args(["tune", "--no-profile", "--profile-report", "R1", "--pod-id", "abc"])
    assert a.no_profile and a.profile_report == "R1" and a.pod_id == "abc"


def test_print_tune_table_gpu_column(capsys):
    res = {"rungs": {
        "R0": {"status": "done", "knobs": {"sam3_track_fps": 5}, "wall_job_s": 300, "wall_final_s": 360,
               "profile": {"gpu_util_mean": 41.6}},
        "R1": {"status": "done", "knobs": {}, "wall_job_s": 300, "wall_final_s": 360},
    }}
    cli.print_tune_table(res, ["R0", "R1"], 8.0, "final")
    out = capsys.readouterr().out.splitlines()
    assert "gpu%" in out[1]
    r0 = next(line for line in out if line.startswith("R0"))
    r1 = next(line for line in out if line.startswith("R1"))
    assert " 42 " in r0 or r0.rstrip().endswith("42")
    assert "  -" in r1


def test_profile_report_from_results_and_pulled_raw(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "TUNE_DIR", tmp_path / "tune_results")
    monkeypatch.setattr(cli, "PULL_ROOT", tmp_path / "pull")
    cli.save_tune({"pod_id": "pod1", "rungs": {
        "R1": {"status": "done", "profile": prof.summarise_pyspy(SYNTH, ROOTS) | {"gpu_util_mean": 33.0,
                                                                                   "gpu_samples": 5}},
        "R2": {"status": "done", "profile": {"pyspy_error": "Permission Denied"}},
        "R3": {"status": "done"},
    }})
    name, p = cli.profile_report("r1", "pod1")
    assert name == "R1" and p["gil_top"][0]["frame"] == "job (app.py)"
    # rung row without gil_top: re-summarised from the pulled raw file
    raw = tmp_path / "pull" / "pod1" / "tune" / "jobs"
    raw.mkdir(parents=True)
    (raw / "R3.pyspy.txt").write_text(SYNTH)
    _, p3 = cli.profile_report("R3", "pod1")
    assert p3["pyspy_samples"] == 100
    _, p2 = cli.profile_report("R2", "pod1")
    assert "gil_top" not in p2 and "Permission" in prof.format_profile_report("R2", p2)
    with pytest.raises(SystemExit):
        cli.profile_report("R9", "pod1")
    # a local file path (+ its sibling .gpu.csv)
    f = tmp_path / "X.pyspy.txt"
    f.write_text(SYNTH)
    (tmp_path / "X.gpu.csv").write_text(CSV_NOUNITS)
    name, pf = cli.profile_report(str(f), None)
    assert name == "X.pyspy.txt" and pf["pyspy_samples"] == 100 and pf["gpu_util_mean"] == 50.0
    # through the CLI entry point: offline, no pod state needed
    monkeypatch.setattr(cli, "load_state", lambda: {})
    assert cli.main(["tune", "--profile-report", str(f)]) == 0
    assert "INCLUSIVE" in capsys.readouterr().out


def test_run_rung_remote_passes_no_profile(monkeypatch):
    sent = []
    done = json.dumps({"status": "done", "job_id": "j", "t_submit": time.time(), "profile": {"gpu_util_mean": 1}})
    calls = {"cat": 0}

    def fake_out(st, remote, timeout=120, check=True):
        sent.append(remote)
        if remote.startswith("cat "):
            calls["cat"] += 1
            return "" if calls["cat"] == 1 else done
        return "started"
    monkeypatch.setattr(cli, "ssh_out", fake_out)
    monkeypatch.setattr(cli, "remote_job_active", lambda st: None)
    rung = cli.profiles.Rung("R1", {"corner_track_stride": 2}, {}, "x")
    st = {"pod_id": "p"}
    state = cli.run_rung_remote(st, rung, "/workspace/tune/v.mp4", poll_s=0, profile=False)
    assert state["status"] == "done" and state["profile"] == {"gpu_util_mean": 1}
    launch = next(c for c in sent if "pod_job.py" in c)
    assert "--no-profile" in launch and "--label R1" in launch
    sent.clear()
    calls["cat"] = 0
    cli.run_rung_remote(st, rung, "/workspace/tune/v.mp4", poll_s=0)
    launch = next(c for c in sent if "pod_job.py" in c)
    assert "--no-profile" not in launch


def test_pod_job_has_profile_flags():
    r = subprocess.run([PY, str(RUNPOD / "pod_job.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--no-profile" in r.stdout and "--pyspy-rate" in r.stdout


def test_requirements_pin_pyspy():
    req = (RUNPOD / "requirements-runpod.txt").read_text()
    assert any(line.startswith("py-spy==") for line in req.splitlines())

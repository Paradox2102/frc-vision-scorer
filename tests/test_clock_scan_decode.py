"""Match-clock scan decode paths (no models, no real match video).

The scan used to seek (CAP_PROP_POS_FRAMES) to every sampled frame, which
re-decodes from the previous keyframe each time. The sequential path grabs
every frame once and only retrieves/OCRs the sampled ones, with OCR fanned
out over a thread pool. These tests pin the two paths to the same output on
a synthetic video whose frames encode their own index, so the fake reader
recovers the exact frame that was decoded.
"""
import hashlib
import json
import random
import shutil
import threading
import time

import cv2
import numpy as np
import pytest

import web.app as app_mod
from web.app import InferenceJob, scan_clock_samples

N_FRAMES = 150
WIDTH, HEIGHT = 128, 64
OVERHEAD_H = 40  # top strip handed to the reader
BITS = 8
BLOCK_W = WIDTH // BITS


def _encode(index: int) -> np.ndarray:
    """Frame whose overhead strip carries ``index`` as 8 black/white bars.
    The strip below the overhead is the bitwise inverse, so a reader that is
    handed the wrong slice decodes the wrong number."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for bit in range(BITS):
        on = (index >> bit) & 1
        x0 = bit * BLOCK_W
        frame[:OVERHEAD_H, x0:x0 + BLOCK_W] = 255 if on else 0
        frame[OVERHEAD_H:, x0:x0 + BLOCK_W] = 0 if on else 255
    return frame


def _decode(overhead: np.ndarray) -> int:
    assert overhead.shape[0] == OVERHEAD_H, overhead.shape
    value = 0
    for bit in range(BITS):
        x0 = bit * BLOCK_W
        patch = overhead[4:-4, x0 + 3:x0 + BLOCK_W - 3]
        if patch.mean() > 127:
            value |= 1 << bit
    return value


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("clockscan") / "synth.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (WIDTH, HEIGHT)
    )
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(_encode(i))
    writer.release()
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N_FRAMES
    cap.release()
    return path


def _expected(step, total=N_FRAMES):
    return [(i, i) for i in range(0, min(total, N_FRAMES), step)]


def _scan(path, step, *, sequential, workers=4, total=N_FRAMES, read_fn=_decode):
    return scan_clock_samples(
        path, total, step, OVERHEAD_H, read_fn,
        workers=workers, sequential=sequential,
    )


@pytest.mark.parametrize("step", [1, 7, 30, 149, 500])
def test_sequential_reads_exactly_the_sampled_frames(synth_video, step):
    assert _scan(synth_video, step, sequential=True) == _expected(step)


@pytest.mark.parametrize("step", [1, 7, 30])
def test_sequential_matches_seek_path(synth_video, step):
    seek = _scan(synth_video, step, sequential=False)
    seq = _scan(synth_video, step, sequential=True)
    assert seek == _expected(step)
    assert seq == seek


def test_worker_count_does_not_change_ordered_output(synth_video):
    # Jittered reader: completion order differs from submission order, the
    # output must still be in frame order.
    rng = random.Random(0)
    lock = threading.Lock()

    def jittery(overhead):
        with lock:
            delay = rng.random() * 0.004
        time.sleep(delay)
        return _decode(overhead)

    one = _scan(synth_video, 3, sequential=True, workers=1, read_fn=jittery)
    eight = _scan(synth_video, 3, sequential=True, workers=8, read_fn=jittery)
    assert one == eight == _expected(3)


@pytest.mark.parametrize("sequential", [True, False])
def test_total_frames_overshoot_stops_at_end_of_stream(synth_video, sequential):
    # CAP_PROP_FRAME_COUNT can overshoot the decodable frame count.
    got = _scan(synth_video, 10, sequential=sequential, total=N_FRAMES + 120)
    assert got == _expected(10)


@pytest.mark.parametrize("sequential", [True, False])
def test_total_frames_undershoot_stops_at_total(synth_video, sequential):
    got = _scan(synth_video, 10, sequential=sequential, total=95)
    assert got == _expected(10, total=95)


@pytest.mark.parametrize("sequential", [True, False])
def test_unopenable_video_returns_no_samples(tmp_path, sequential):
    missing = tmp_path / "missing.avi"
    assert _scan(missing, 10, sequential=sequential) == []


class _CountingCap:
    """Proxy over a real VideoCapture that counts seeks and retrieves."""

    instances = []

    def __init__(self, *args, **kwargs):
        self._cap = _REAL_CAP(*args, **kwargs)
        self.sets = 0
        self.retrieves = 0
        _CountingCap.instances.append(self)

    def set(self, *a):
        self.sets += 1
        return self._cap.set(*a)

    def retrieve(self, *a):
        self.retrieves += 1
        return self._cap.retrieve(*a)

    def __getattr__(self, name):
        return getattr(self._cap, name)


_REAL_CAP = cv2.VideoCapture


@pytest.fixture
def counting_cap(monkeypatch):
    _CountingCap.instances = []
    monkeypatch.setattr(app_mod.cv2, "VideoCapture", _CountingCap)
    return _CountingCap


def test_sequential_never_seeks(synth_video, counting_cap):
    assert _scan(synth_video, 7, sequential=True) == _expected(7)
    (cap,) = counting_cap.instances
    assert cap.sets == 0
    assert cap.retrieves == len(_expected(7))


def test_in_flight_ocr_is_bounded(synth_video, counting_cap):
    # A stalled reader must not let the decoder buffer the whole video.
    release = threading.Event()

    def blocked(overhead):
        release.wait(5)
        return _decode(overhead)

    out = {}
    t = threading.Thread(
        target=lambda: out.setdefault(
            "samples", _scan(synth_video, 1, sequential=True, workers=2, read_fn=blocked)
        )
    )
    t.start()
    time.sleep(0.5)
    (cap,) = counting_cap.instances
    buffered = cap.retrieves
    release.set()
    t.join(10)
    assert not t.is_alive()
    assert buffered <= 2 * 2 + 1, buffered
    assert out["samples"] == _expected(1)


def test_kill_switch_selects_seek_path(synth_video, counting_cap, monkeypatch):
    monkeypatch.setattr(app_mod, "CLOCK_SCAN_SEQUENTIAL", False)
    got = scan_clock_samples(synth_video, N_FRAMES, 10, OVERHEAD_H, _decode)
    assert got == _expected(10)
    (cap,) = counting_cap.instances
    assert cap.sets == len(_expected(10))  # one seek per sample = old path


def test_default_selects_sequential_path(synth_video, counting_cap, monkeypatch):
    monkeypatch.setattr(app_mod, "CLOCK_SCAN_SEQUENTIAL", True)
    got = scan_clock_samples(synth_video, N_FRAMES, 10, OVERHEAD_H, _decode)
    assert got == _expected(10)
    (cap,) = counting_cap.instances
    assert cap.sets == 0


def test_default_worker_count():
    assert app_mod.CLOCK_SCAN_OCR_WORKERS >= 1


# ---------------------------------------------------------------------------
# _scan_match_clock: cache key/format/behaviour unchanged
# ---------------------------------------------------------------------------

def _stub_job(path, fps=30.0):
    job = InferenceJob.__new__(InferenceJob)
    job.video_path = path
    job.video_fps = fps
    job.total_frames = N_FRAMES
    job.overhead_height = OVERHEAD_H
    return job


def _expected_cache_name(path, sample_step):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    h.update(f"{path.stat().st_size}:{sample_step}:{OVERHEAD_H}".encode())
    return f"clockscan_{h.hexdigest()}.json"


def test_scan_match_clock_writes_and_hits_same_cache(synth_video, tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", tmp_path)
    job = _stub_job(synth_video)
    calls = []

    def reader(overhead):
        calls.append(1)
        return _decode(overhead)

    job.read_match_clock = reader
    step = 15  # round(30 fps * 0.5 s)
    first = job._scan_match_clock()
    assert first == _expected(step)
    cache = tmp_path / _expected_cache_name(synth_video, step)
    assert cache.exists()
    assert json.loads(cache.read_text()) == [[i, v] for i, v in _expected(step)]

    n_calls = len(calls)
    assert job._scan_match_clock() == first
    assert len(calls) == n_calls  # served from cache


def test_scan_match_clock_honours_preexisting_cache(synth_video, tmp_path, monkeypatch):
    # A cache written by the old code (json.dumps of a list of tuples, None
    # for unread samples) must still be hit, not rescanned.
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", tmp_path)
    job = _stub_job(synth_video)
    old = [(0, None), (15, 20), (30, 19)]
    (tmp_path / _expected_cache_name(synth_video, 15)).write_text(json.dumps(old))

    def boom(_overhead):
        raise AssertionError("cache should have been hit")

    job.read_match_clock = boom
    assert job._scan_match_clock() == old


# ---------------------------------------------------------------------------
# read_match_clock is safe to call from worker threads
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_read_match_clock_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    job = InferenceJob.__new__(InferenceJob)
    job.ocr_regions = {"match_clock": (10, 10, 120, 40)}
    strips = []
    for text in ("1:27", "0:23", "2:15", "0:05", "1:59", "0:41"):
        img = np.zeros((60, 160, 3), np.uint8)
        cv2.putText(img, text, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        strips.append(img)
    strips = strips * 4
    serial = [job.read_match_clock(s) for s in strips]
    assert any(v is not None for v in serial)  # the reads are meaningful
    with ThreadPoolExecutor(8) as pool:
        parallel = list(pool.map(job.read_match_clock, strips))
    assert parallel == serial

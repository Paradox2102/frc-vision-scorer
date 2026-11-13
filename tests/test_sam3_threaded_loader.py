"""sam3_frame_load_workers: the threaded chunk frame loader must be
byte-identical to upstream's single-threaded PIL loader."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
io_utils = pytest.importorskip("sam3.model.io_utils")

from web.field import sam3_tracker  # noqa: E402
from web.field.sam3_tracker import (  # noqa: E402
    _uninstall_threaded_frame_loader,
    install_threaded_frame_loader,
)

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


@pytest.fixture(autouse=True)
def _restore_loader():
    _uninstall_threaded_frame_loader()
    upstream = io_utils.load_video_frames_from_image_folder
    yield upstream
    _uninstall_threaded_frame_loader()
    assert io_utils.load_video_frames_from_image_folder is upstream


def _write_frames(folder, n=12, w=64, h=48):
    from PIL import Image

    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"{i:08d}.jpg", quality=90)


def _load(fn, folder, *, async_loading=False):
    return fn(
        image_folder=str(folder),
        image_size=32,
        offload_video_to_cpu=True,
        img_mean=MEAN,
        img_std=STD,
        async_loading_frames=async_loading,
    )


def test_threaded_loader_byte_identical(tmp_path, _restore_loader):
    upstream = _restore_loader
    _write_frames(tmp_path)
    ref_imgs, ref_h, ref_w = _load(upstream, tmp_path)

    assert install_threaded_frame_loader(4) is True
    threaded = io_utils.load_video_frames_from_image_folder
    assert threaded is not upstream
    got_imgs, got_h, got_w = _load(threaded, tmp_path)

    assert got_imgs.dtype == ref_imgs.dtype == torch.float16
    assert got_imgs.shape == ref_imgs.shape == (12, 3, 32, 32)
    assert torch.equal(got_imgs, ref_imgs)
    assert (got_h, got_w) == (ref_h, ref_w) == (48, 64)


def test_patched_global_is_what_load_resource_calls(tmp_path, _restore_loader):
    # The real session path: load_resource_as_video_frames -> load_video_frames
    # -> load_video_frames_from_image_folder via module globals.
    _write_frames(tmp_path, n=5)
    ref = io_utils.load_resource_as_video_frames(
        str(tmp_path), 32, True, MEAN, STD)
    install_threaded_frame_loader(3)
    calls = []
    inner = io_utils.load_video_frames_from_image_folder

    def spy(**kw):
        calls.append(kw)
        return inner(**kw)

    io_utils.load_video_frames_from_image_folder = spy
    try:
        got = io_utils.load_resource_as_video_frames(
            str(tmp_path), 32, True, MEAN, STD)
    finally:
        io_utils.load_video_frames_from_image_folder = inner
    assert len(calls) == 1
    assert torch.equal(got[0], ref[0]) and got[1:] == ref[1:]


def test_threaded_loader_empty_dir_raises(tmp_path):
    install_threaded_frame_loader(4)
    (tmp_path / "notes.txt").write_text("not a frame")
    with pytest.raises(RuntimeError, match="no images found"):
        _load(io_utils.load_video_frames_from_image_folder, tmp_path)


def test_async_loading_delegates_to_upstream(tmp_path, _restore_loader):
    upstream = _restore_loader
    _write_frames(tmp_path, n=3)
    calls = []

    def fake_upstream(**kw):
        calls.append(kw)
        return "sentinel", 1, 2

    io_utils.load_video_frames_from_image_folder = fake_upstream
    try:
        assert install_threaded_frame_loader(4)
        out = _load(io_utils.load_video_frames_from_image_folder, tmp_path,
                    async_loading=True)
    finally:
        _uninstall_threaded_frame_loader()
        io_utils.load_video_frames_from_image_folder = upstream
    assert out == ("sentinel", 1, 2)
    assert len(calls) == 1 and calls[0]["async_loading_frames"] is True


def test_install_idempotent_and_off_for_one_worker(_restore_loader):
    upstream = _restore_loader
    assert install_threaded_frame_loader(1) is False
    assert install_threaded_frame_loader(0) is False
    assert io_utils.load_video_frames_from_image_folder is upstream

    assert install_threaded_frame_loader(4)
    first = io_utils.load_video_frames_from_image_folder
    assert install_threaded_frame_loader(4)
    assert install_threaded_frame_loader(2)  # smaller request: keep the 4-pool
    assert io_utils.load_video_frames_from_image_folder is first
    assert sam3_tracker._ORIG_LOAD_FROM_IMAGE_FOLDER is upstream

    assert install_threaded_frame_loader(8)  # larger: re-installed, same orig
    assert io_utils.load_video_frames_from_image_folder is not first
    assert sam3_tracker._ORIG_LOAD_FROM_IMAGE_FOLDER is upstream
    assert io_utils.load_video_frames_from_image_folder._frc_workers == 8


def test_pool_growth_never_shuts_down_a_pool_in_use():
    """sam3_parallel_views: several views open chunk sessions concurrently on
    the module-shared pool. Growing the pool must not shut down the one
    another thread already fetched (its ``pool.map`` would raise "cannot
    schedule new futures after shutdown")."""
    old = sam3_tracker._frame_load_pool(2)
    new = sam3_tracker._frame_load_pool(4)
    assert new is not old
    assert sam3_tracker._frame_load_pool(3) is new  # smaller: reuse
    assert list(old.map(lambda x: x * 2, range(5))) == [0, 2, 4, 6, 8]


def test_concurrent_loads_share_pool_byte_identical(tmp_path, _restore_loader):
    """Concurrent sessions (parallel views) mapping on the shared pool get
    the same frames as a lone upstream load."""
    import concurrent.futures as cf

    upstream = _restore_loader
    _write_frames(tmp_path)
    ref, _, _ = _load(upstream, tmp_path)
    assert install_threaded_frame_loader(4)
    patched = io_utils.load_video_frames_from_image_folder
    with cf.ThreadPoolExecutor(3) as ex:
        outs = list(ex.map(lambda _: _load(patched, tmp_path)[0], range(6)))
    for out in outs:
        assert torch.equal(out, ref)

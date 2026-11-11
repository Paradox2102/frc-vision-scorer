"""HUD renderer (web/hud_render.py; spec section 5) on a synthetic source video:
one frame per meta frame, final labels, the thumbnail decode check, missing
draw-list lines, the atomic replace and hud.live.mp4 kept once, and the CLI."""
import asyncio
import gzip
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import web.database as database
from web import hud_draw, hud_render

REPO = Path(__file__).resolve().parents[1]
W, H, OVERHEAD = 320, 240, 160
VIEWS = [{"name": "corner_left", "crop": [0, 150, 160, 90], "masks": []},
         {"name": "corner_right", "crop": [160, 150, 160, 90], "masks": [[0, 0, 20, 10]]}]


def write_source(path: Path, n: int = 40, fps: float = 30.0, near_static: bool = False) -> None:
    """A tiny mp4 whose every frame differs (a bar sweeping across), or, with
    ``near_static``, a static background with one small blob moving 1 px a
    frame (a fixed camera at 60 fps)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if near_static:
        rng = np.random.default_rng(0)
        bg = cv2.GaussianBlur(rng.integers(60, 120, (H, W, 3), dtype=np.uint8), (7, 7), 0)
        for i in range(n):
            f = bg.copy()
            cv2.circle(f, (40 + i, 80), 8, (230, 230, 230), -1)
            vw.write(f)
        vw.release()
        return
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.stack([(xx // 2) % 256, (yy // 2) % 256, ((xx + yy) // 4) % 256],
                    -1).astype(np.uint8)
    for i in range(n):
        f = base.copy()
        x = (i * 7) % (W - 30)
        cv2.rectangle(f, (x, 10), (x + 30, 140), (255, 255, 255), -1)
        cv2.rectangle(f, (W - x - 30, 160), (W - x, 230), (0, 0, 0), -1)
        vw.write(f)
    vw.release()


def decoded_thumbs(src: Path, first: int, frame_nums) -> dict:
    """main-view thumbnail per frame, decoded as the live loop does."""
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out, fn, want = {}, first, set(frame_nums)
    while fn <= max(want):
        ok, img = cap.read()
        assert ok
        if fn in want:
            out[fn] = hud_draw.main_thumb(img[:OVERHEAD, :])
        fn += 1
    cap.release()
    return out


def make_replay(tmp_path, rid="r1", n_meta=12, first=5, skip=2, draw=True, live_video=False,
                robots=None, n_src=40, near_static=False, thumb_offset=0):
    """A replay dir + DB row: meta.json (+ hud_static and draw.jsonl.gz when
    ``draw``), a synthetic source video. Returns (rid, dir, db_path)."""
    db = database.DB_PATH
    d = tmp_path / rid
    d.mkdir()
    src = tmp_path / f"{rid}_src.mp4"
    write_source(src, n=n_src, near_static=near_static)
    frame_nums = [first + skip * k for k in range(n_meta)]
    robots = robots or [
        {"slot": "B1", "alliance": "blue", "team_number": "111", "confirmed": True,
         "field_pos": [3.0, 2.0]},
        {"slot": "R1", "alliance": "red", "team_number": "AMBIGUOUS", "confirmed": False,
         "field_pos": [12.0, 5.0]}]
    frames = [{"frame_num": fn, "scores": {}, "robots": [dict(r) for r in robots],
               "i": i, "t": i / 8.0} for i, fn in enumerate(frame_nums)]
    summary = {"fps": 8.0, "frame_count": n_meta, "processing_ranges": [[first, frame_nums[-1]]]}
    if draw:
        summary["hud_static"] = {
            "flags": hud_draw.flags_to_dict(hud_draw.HudFlags()),
            "ocr_regions": {"blue_score": [10, 5, 40, 12]}, "field_wireframe_overlay": False,
            "overhead_height": OVERHEAD, "views": VIEWS, "panel_max_width": 720,
            "reference_paths": [], "event_key": None, "camera": "main",
            "calibration_version": None, "field_season": "2026-rebuilt-welded"}
        thumbs = decoded_thumbs(src, first + thumb_offset,
                                [fn + thumb_offset for fn in frame_nums])
        with gzip.open(d / "draw.jsonl.gz", "wt") as fh:
            for i, fn in enumerate(frame_nums):
                fh.write(json.dumps({
                    "i": i, "frame_num": fn, "main_thumb": thumbs[fn + thumb_offset],
                    "main": [{"slot": "B1", "slot_id": 0, "alliance": "blue",
                              "box": [40, 40, 90, 100], "ghost": False, "score": 2,
                              "mask_poly": [[45, 45], [85, 45], [85, 95], [45, 95]]},
                             {"slot": "R1", "slot_id": 3, "alliance": "red",
                              "box": [200, 60, 250, 120], "ghost": False, "score": 0,
                              "mask_poly": None},
                             {"slot": "R2", "slot_id": 4, "alliance": "red",
                              "box": [260, 20, 300, 60], "ghost": False, "score": 0,
                              "mask_poly": None}],
                    "corners": {"corner_left": [
                        {"slot": "B1", "alliance": "blue", "box": [10, 10, 50, 50],
                         "score": 2, "live_text": "222? (40%) S2"},
                        {"slot": None, "alliance": "red", "box": [80, 10, 120, 50],
                         "score": None, "live_text": "red bumper no #"}]},
                    "hubs": [{"alliance": "blue", "x": 150.0, "y": 80.0}],
                    "hub_boxes": [[140, 70, 160, 90, 0.9]],
                    "radar": [{"slot": "B1", "alliance": "blue", "pos": [3.0, 2.0],
                               "state": "visible", "corner_only": False,
                               "trail": [[2.5, 2.0], [3.0, 2.0]]}]}) + "\n")
    (d / "meta.json").write_text(json.dumps({"summary": summary, "frames": frames,
                                             "events": []}))
    if live_video:
        (d / "hud.mp4").write_bytes(b"LIVE-ENCODED")
    asyncio.run(database.insert_replay({
        "id": rid, "created_at": "2026-11-10T12:00:00", "label": rid, "event_key": None,
        "match_key": None, "video_path": str(d / "hud.mp4"), "meta_path": str(d / "meta.json"),
        "thumb_path": str(d / "thumb.jpg"), "source_video_path": str(src), "fps": 8.0,
        "frame_count": n_meta, "codec": None if draw else "h264", "status": "complete"}))
    return rid, d, db


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    asyncio.run(database.init_db())
    return tmp_path / "t.db"


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_overhead_ratio_matches_the_app():
    import web.app as app_mod
    assert hud_render.OVERHEAD_CROP_RATIO == app_mod.OVERHEAD_CROP_RATIO


def test_one_output_frame_per_meta_frame(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)
    stats = hud_render.render_replay(rid)
    assert stats["frames"] == 12 and _frame_count(d / "hud.mp4") == 12
    assert stats["decode_checked"] == 12 and stats["decode_mismatches"] == 0
    assert stats["source"] == "draw_list" and stats["codec"] in ("h264", "avc1", "mp4v")
    assert (stats["width"], stats["height"]) == (W, OVERHEAD + 300)
    assert (d / "thumb.jpg").exists() and json.loads((d / "render.json").read_text()) == stats
    assert not (d / "hud.render.mp4").exists()


def test_final_labels_are_drawn_not_the_live_ones(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path)
    texts = []
    real = hud_draw.draw_robot
    monkeypatch.setattr(hud_draw, "draw_robot",
                        lambda img, box, alliance, label, flags, **kw:
                        (texts.append((kw.get("slot_id"), label)),
                         real(img, box, alliance, label, flags, **kw))[1])
    stats = hud_render.render_replay(rid)
    per_frame = set(texts)
    # main: B1 final "111" confirmed, R1 AMBIGUOUS -> "no #"; corner B1 is the
    # same tag as main (not the recorded live "222? (40%) S2"); the slot-less
    # corner keeps its live string.
    assert per_frame == {(0, "111 S2"), (3, "no # S0"), (None, "red bumper no #")}
    assert stats["phantom_boxes"] == 12           # R2 has no meta row: grey slot-only box
    assert stats["corrections"] == {} and stats["corrections_note"] == "no recorded corrections"


def test_corrections_relabel_frames_and_are_counted(db, tmp_path, monkeypatch):
    rid, d, _ = make_replay(tmp_path)
    asyncio.run(database.insert_slot_correction(rid, 0, "111", "999"))
    asyncio.run(database.insert_slot_correction(rid, 3, None, "4256"))
    texts = []
    real = hud_draw.draw_robot
    monkeypatch.setattr(hud_draw, "draw_robot",
                        lambda img, box, alliance, label, flags, **kw:
                        (texts.append(label), real(img, box, alliance, label, flags, **kw))[1])
    stats = hud_render.render_replay(rid)
    assert {"999 S2", "4256 S0"} <= set(texts) and "111 S2" not in texts
    assert stats["corrections"] == {"B1": {"old": "111", "new": "999", "frames_relabelled": 12},
                                    "R1": {"old": None, "new": "4256", "frames_relabelled": 12}}
    assert stats["corrections_note"] is None


def _rewrite_thumbs(d: Path, bad_indices):
    import base64
    lines = [json.loads(l) for l in gzip.open(d / "draw.jsonl.gz", "rt")]
    for ln in lines:
        if ln["i"] in bad_indices:
            raw = bytes(255 - b for b in base64.b64decode(ln["main_thumb"]))
            ln["main_thumb"] = base64.b64encode(raw).decode()
    with gzip.open(d / "draw.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(ln) + "\n" for ln in lines)


def test_thumb_mismatches_beyond_two_percent_abort(db, tmp_path):
    rid, d, _ = make_replay(tmp_path, n_meta=60, first=3, skip=1, n_src=70)
    _rewrite_thumbs(d, {10})                     # 1 of 60 = 1.7%: tolerated
    assert hud_render.render_replay(rid)["decode_mismatches"] == 1
    _rewrite_thumbs(d, {20})                     # 2 of 60 = 3.3%: abort
    before = (d / "hud.mp4").read_bytes()
    with pytest.raises(hud_render.RenderError, match="decode misaligned"):
        hud_render.render_replay(rid)
    assert (d / "hud.mp4").read_bytes() == before and not (d / "hud.render.mp4").exists()


@pytest.mark.parametrize("offset", [1, 2])
def test_a_near_static_video_off_by_one_or_two_frames_aborts(db, tmp_path, offset):
    rid, d, _ = make_replay(tmp_path, n_meta=30, first=3, skip=1, n_src=40,
                            near_static=True, thumb_offset=offset)
    with pytest.raises(hud_render.RenderError, match="decode misaligned"):
        hud_render.render_replay(rid)
    assert not (d / "hud.mp4").exists() and not (d / "hud.render.mp4").exists()


def test_a_near_static_video_aligned_has_no_mismatches(db, tmp_path):
    rid, d, _ = make_replay(tmp_path, n_meta=30, first=3, skip=1, n_src=40, near_static=True)
    stats = hud_render.render_replay(rid)
    assert stats["decode_checked"] == 30 and stats["decode_mismatches"] == 0


def test_a_failing_encoder_never_replaces_hud_mp4(db, tmp_path, monkeypatch):
    import web.replay_recorder as rr
    rid, d, _ = make_replay(tmp_path, live_video=True)
    monkeypatch.setattr(rr.shutil, "which", lambda name: "/bin/sh")
    real_popen = subprocess.Popen

    def failing(cmd, **kw):                       # writes a partial file, exits 1
        return real_popen(["/bin/sh", "-c", f"head -c 100 > '{cmd[-1]}'; exit 1"], **kw)
    monkeypatch.setattr(rr.subprocess, "Popen", failing)
    with pytest.raises(hud_render.RenderError, match="encoder exited 1"):
        hud_render.render_replay(rid)
    assert (d / "hud.mp4").read_bytes() == b"LIVE-ENCODED"
    assert not (d / "hud.render.mp4").exists() and not (d / "render.json").exists()


def test_an_encoded_frame_count_short_of_meta_is_a_render_error(db, tmp_path, monkeypatch):
    from web.replay_recorder import ReplayRecorder
    rid, d, _ = make_replay(tmp_path, live_video=True)
    real_add = ReplayRecorder.add_frame
    calls = []

    def dropping(self, frame, record, draw=None):  # the encoder loses one frame
        calls.append(1)
        if len(calls) != 5:
            real_add(self, frame, record, draw)
    monkeypatch.setattr(ReplayRecorder, "add_frame", dropping)
    with pytest.raises(hud_render.RenderError, match="11 frames, expected 12"):
        hud_render.render_replay(rid)
    assert (d / "hud.mp4").read_bytes() == b"LIVE-ENCODED"
    assert not (d / "hud.render.mp4").exists()


def _rewrite_meta(d: Path, fn):
    meta = json.loads((d / "meta.json").read_text())
    fn(meta)
    (d / "meta.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("bad", [None, "behind"])
def test_a_frame_num_behind_the_decoder_aborts(db, tmp_path, bad):
    rid, d, _ = make_replay(tmp_path, live_video=True)

    def edit(meta):                          # frame 6 repeats frame 5's number (or none)
        meta["frames"][6]["frame_num"] = (meta["frames"][5]["frame_num"]
                                          if bad == "behind" else None)
    _rewrite_meta(d, edit)
    with pytest.raises(hud_render.RenderError,
                       match="decode misaligned: meta frame 6 has frame_num"):
        hud_render.render_replay(rid)
    assert (d / "hud.mp4").read_bytes() == b"LIVE-ENCODED"
    assert not (d / "hud.render.mp4").exists()


@pytest.mark.parametrize("skip", [1, 2])      # 1: read() runs out; 2: grab() does
def test_a_source_that_ends_early_aborts(db, tmp_path, skip):
    rid, d, _ = make_replay(tmp_path, n_meta=12, first=5, skip=skip, live_video=True)
    src = Path(database.get_replay_sync(rid)["source_video_path"])
    write_source(src, n=5 + skip * 6)        # only 6 of the 12 meta frames exist
    with pytest.raises(hud_render.RenderError, match="source video ended at frame"):
        hud_render.render_replay(rid)
    assert (d / "hud.mp4").read_bytes() == b"LIVE-ENCODED"
    assert not (d / "hud.render.mp4").exists()


def test_a_missing_draw_list_line_still_emits_a_frame(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)
    lines = [json.loads(l) for l in gzip.open(d / "draw.jsonl.gz", "rt")]
    with gzip.open(d / "draw.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(ln) + "\n" for ln in lines if ln["i"] != 4)
    stats = hud_render.render_replay(rid)
    assert stats["draw_lines_missing"] == 1 and stats["decode_checked"] == 11
    assert _frame_count(d / "hud.mp4") == 12


def test_a_truncated_draw_list_keeps_the_lines_before_the_damage(tmp_path):
    p = tmp_path / "draw.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for i in range(50):
            fh.write(json.dumps({"i": i, "frame_num": i, "pad": "x" * 200}) + "\n")
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])
    got = hud_render.load_draw_list(p)
    assert 0 < len(got) < 50 and sorted(got) == list(range(len(got)))


def test_replace_is_atomic_and_the_live_video_is_kept_once(db, tmp_path):
    rid, d, _ = make_replay(tmp_path, live_video=True)
    s1 = hud_render.render_replay(rid)
    assert s1["kept_live_video"] is True
    assert (d / "hud.live.mp4").read_bytes() == b"LIVE-ENCODED"
    assert _frame_count(d / "hud.mp4") == 12
    s2 = hud_render.render_replay(rid)          # re-render: the rendered video is not "live"
    assert s2["kept_live_video"] is False
    assert (d / "hud.live.mp4").read_bytes() == b"LIVE-ENCODED"


def test_no_live_copy_for_a_replay_that_never_had_one(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)
    assert hud_render.render_replay(rid)["kept_live_video"] is False
    assert not (d / "hud.live.mp4").exists()


def test_missing_source_video_is_a_render_error(db, tmp_path):
    rid, d, _ = make_replay(tmp_path)
    Path(database.get_replay_sync(rid)["source_video_path"]).unlink()
    with pytest.raises(hud_render.RenderError, match="source video missing"):
        hud_render.render_replay(rid)


def test_cli_runs_in_a_subprocess(db, tmp_path):
    rid, d, dbp = make_replay(tmp_path)
    p = subprocess.run([sys.executable, "-m", "web.hud_render", rid, "--db", str(dbp)],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["frames"] == 12 and (d / "hud.mp4").exists()
    bad = subprocess.run([sys.executable, "-m", "web.hud_render", "nope", "--db", str(dbp)],
                         cwd=REPO, capture_output=True, text=True, timeout=120)
    assert bad.returncode == 1 and json.loads(bad.stdout)["error"] == "replay not found"

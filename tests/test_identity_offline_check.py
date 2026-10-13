"""scripts/identity_offline_check.py: identity_log reconstruction from
match_robot_binding rows + the shared whole-match per-slot resolve."""
import importlib.util
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "identity_offline_check", ROOT / "scripts" / "identity_offline_check.py")
ioc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ioc)


def test_binding_takes_effect_after_its_start_frame():
    """start_frame is the loop's _current_frame at bind time — the PREVIOUS
    processed frame — so an identity holds its slot from the next processed
    frame through the next binding's start_frame (inclusive)."""
    frames = [0, 8, 16, 24, 32, 40]
    binds = [(1, 0, 0, None), (2, 0, 16, None)]     # (iid, slot, start, end)
    log = ioc.reconstruct_identity_log(binds, frames)
    held = {(r["frame_num"], r["identity_id"]) for r in log}
    assert held == {(8, 1), (16, 1), (24, 2), (32, 2), (40, 2)}
    assert all(r["slot_id"] == 0 and r["alliance"] == "blue" for r in log)


def test_end_frame_closes_the_hold():
    frames = [0, 8, 16, 24, 32]
    log = ioc.reconstruct_identity_log([(7, 4, 0, 16)], frames)
    assert [(r["frame_num"], r["identity_id"], r["alliance"]) for r in log] == [
        (8, 7, "red"), (16, 7, "red")]


def test_identity_on_two_slots_keeps_its_latest_binding():
    frames = [0, 8, 16, 24]
    binds = [(1, 0, 0, None), (1, 1, 8, None)]     # moved 0 -> 1, no new holder on 0
    log = ioc.reconstruct_identity_log(binds, frames)
    by_frame = {r["frame_num"]: r["slot_id"] for r in log if r["identity_id"] == 1}
    assert by_frame == {8: 0, 16: 1, 24: 1}


def test_run_check_scores_slots_against_truth():
    frames = list(range(0, 80, 8))
    binds = [(10 + s, s, 0, None) for s in range(6)]
    truth = {"B1": "4786", "B2": "3181", "B3": "2992",
             "R1": "6800", "R2": "1477", "R3": "4256"}
    reads = [{"type": "vlm_read", "frame": 16, "robot_id": 10 + s,
              "matched_team": t, "vote_weight": 1.0}
             for s, t in enumerate(truth.values()) if s != 2]     # B3 unread
    reads.append({"type": "vlm_read", "frame": 24, "robot_id": 13,
                  "matched_team": "4256", "vote_weight": 0.5})     # weaker misread
    res = ioc.run_check(binds, frames, reads, truth)
    assert res["correct"] == 5 and res["wrong"] == 0 and res["blank"] == 1
    assert res["slots"]["B3"]["status"] == "BLANK"
    assert res["slots"]["R1"]["resolved"] == "6800"
    text = ioc.format_report(res)
    assert "5/6 correct" in text and "B3" in text


def test_main_end_to_end_on_tmp_data(tmp_path, capsys):
    job = "abcd1234-0000"
    con = sqlite3.connect(tmp_path / "frc_scorer.db")
    con.execute("CREATE TABLE match_robot_binding (id INTEGER PRIMARY KEY, job_id TEXT,"
                " identity_id INTEGER, slot INTEGER, start_frame INTEGER, end_frame INTEGER)")
    con.execute("CREATE TABLE match_robot_identity (job_id TEXT, identity_id INTEGER,"
                " slot INTEGER, slot_label TEXT, alliance TEXT, team_number TEXT,"
                " confirmed INTEGER, source TEXT, confidence REAL)")
    for s in range(6):
        con.execute("INSERT INTO match_robot_binding (job_id, identity_id, slot, start_frame)"
                    " VALUES (?,?,?,?)", (job, s, s, 0))
        con.execute("INSERT INTO match_robot_identity VALUES (?,?,?,?,?,?,?,?,?)",
                    (job, s, s, "B1 B2 B3 R1 R2 R3".split()[s],
                     "blue" if s < 3 else "red", None, 0, "ocr", None))
    con.commit()
    con.close()
    (tmp_path / "web" / "uploads").mkdir(parents=True)
    (tmp_path / "web" / "replays" / job).mkdir(parents=True)
    (tmp_path / "web" / "replays" / job / "meta.json").write_text(json.dumps(
        {"frames": [{"frame_num": f, "robots": []} for f in (0, 8, 16)]}))
    (tmp_path / "web" / "uploads" / f"diagnostic_{job}.jsonl").write_text(
        json.dumps({"type": "vlm_read", "frame": 8, "robot_id": 4,
                    "matched_team": "1477", "vote_weight": 1.0}) + "\n")
    rc = ioc.main([job[:6], "--data-root", str(tmp_path),
                   "--truth", "B1=4786,B2=3181,B3=2992,R1=6800,R2=1477,R3=4256"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "R2" in out and "1/6 correct" in out and "today 0/6" in out


# --------------------------------------------------------------------------- #
# --swap: swap-aware time-varying resolve vs whole-match gir
# --------------------------------------------------------------------------- #
TRUTH = {"B1": "4786", "B2": "3181", "B3": "2992",
         "R1": "6800", "R2": "1477", "R3": "4256"}


def _swap_job():
    """R1/R3 exchange robots around f800 (positions touch there): R1 reads
    6800 early and 4256 late, R3 the opposite; others steady. One identity
    per slot; the replay positions feed the swap gate."""
    frames = list(range(0, 1600, 8))
    binds = [(10 + s, s, -8, None) for s in range(6)]

    def pos(s, f):
        y = 1.0 + 3.0 * s
        if s in (3, 5):
            t = min(abs(f - 800) / 360.0, 1.0)
            y = 13.0 + (y - 13.0) * t
        return [2.0 + 0.01 * f + (6.0 if s == 4 else 0.0), y]
    positions = {(f, s): pos(s, f) for f in frames for s in range(6)}
    reads = []
    for s, team in enumerate(TRUTH.values()):
        if s in (3, 5):
            continue
        reads += [{"type": "vlm_read", "frame": f, "robot_id": 10 + s,
                   "matched_team": team, "vote_weight": 1.0} for f in (160, 1200)]
    for i in range(5):
        for s, early, late in ((3, "6800", "4256"), (5, "4256", "6800")):
            reads.append({"type": "vlm_read", "frame": 160 + 40 * i, "robot_id": 10 + s,
                          "matched_team": early, "vote_weight": 1.0})
            reads.append({"type": "vlm_read", "frame": 1200 + 40 * i, "robot_id": 10 + s,
                          "matched_team": late, "vote_weight": 1.0})
    return binds, frames, reads, positions


def test_attach_positions_fills_pos():
    log = [{"frame_num": 8, "identity_id": 1, "slot_id": 0}]
    assert ioc.attach_positions(log, {(8, 0): [1, 2]})[0]["pos"] == [1, 2]
    assert ioc.attach_positions(log, None) is log


def test_run_swap_check_finds_the_swap_and_beats_gir():
    binds, frames, reads, positions = _swap_job()
    res = ioc.run_swap_check(binds, frames, reads, TRUTH, positions=positions,
                             auto_end=600, holdout_runs=10)
    assert [(s["alliance"], s["slots"]) for s in res["swaps"]] == [("red", ["R1", "R3"])]
    assert [seg[2] for seg in res["segments"]["R1"]] == ["6800", "4256"]
    assert [seg[2] for seg in res["segments"]["R3"]] == ["4256", "6800"]
    # gir ties R1/R3 (5 reads each way) -> both blank all match long.
    assert res["gir_labels"]["R1"] is None and res["gir_labels"]["R3"] is None
    assert res["auto_swap"]["slots_ok"] == 6 and res["auto_gir"]["slots_ok"] == 4
    assert res["auto_swap"]["frame_frac"] == 1.0
    ho = res["holdout"]
    assert ho["n"] > 0 and ho["swap"] > ho["gir"]
    text = ioc.format_swap_report(res)
    assert "R1<->R3" in text and "auto window: gir 4/6" in text and "held-out" in text


def test_run_swap_check_without_positions_has_open_gate():
    binds, frames, reads, _ = _swap_job()
    res = ioc.run_swap_check(binds, frames, reads, TRUTH, positions=None, holdout_runs=0)
    assert "holdout" not in res
    assert res["gate_open_frac"]["R1-R3"] == 1.0
    assert [(s["slots"]) for s in res["swaps"]] == [["R1", "R3"]]


def test_main_swap_flag_end_to_end(tmp_path, capsys):
    job = "swap1234-0000"
    binds, frames, reads, positions = _swap_job()
    con = sqlite3.connect(tmp_path / "frc_scorer.db")
    con.execute("CREATE TABLE match_robot_binding (id INTEGER PRIMARY KEY, job_id TEXT,"
                " identity_id INTEGER, slot INTEGER, start_frame INTEGER, end_frame INTEGER)")
    con.execute("CREATE TABLE match_robot_identity (job_id TEXT, identity_id INTEGER,"
                " slot INTEGER, slot_label TEXT, alliance TEXT, team_number TEXT,"
                " confirmed INTEGER, source TEXT, confidence REAL)")
    for iid, s, start, end in binds:
        con.execute("INSERT INTO match_robot_binding (job_id, identity_id, slot, start_frame)"
                    " VALUES (?,?,?,?)", (job, iid, s, start))
    con.commit()
    con.close()
    (tmp_path / "web" / "uploads").mkdir(parents=True)
    (tmp_path / "web" / "replays" / job).mkdir(parents=True)
    labels = ["B1", "B2", "B3", "R1", "R2", "R3"]
    (tmp_path / "web" / "replays" / job / "meta.json").write_text(json.dumps(
        {"frames": [{"frame_num": f, "robots": [
            {"slot": labels[s], "field_pos": positions[(f, s)]} for s in range(6)]}
            for f in frames]}))
    (tmp_path / "web" / "uploads" / f"diagnostic_{job}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in reads) + "\n")
    rc = ioc.main([job[:6], "--data-root", str(tmp_path), "--swap", "--fps", "60",
                   "--holdout-runs", "5", "--auto-end", "600",
                   "--truth", ",".join(f"{k}={v}" for k, v in TRUTH.items())])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[swap-resolve] lambda=3 prox=1.5m | blue: none | red: R1<->R3 @f" in out
    assert "auto window: gir 4/6 slots" in out and "swap 6/6 slots" in out


# --------------------------------------------------------------------------- #
# --segments: segment-level labels (identity_segment_resolve) + Clef floor
# --------------------------------------------------------------------------- #
def _segment_data(exact=False):
    """The swap job with the robot exchange logged as a discontinuity on R1
    and R3 at f800 (identity births on disrupted slots)."""
    binds, frames, reads, positions = _swap_job()
    rejects = [{"type": "identity_reject", "frame": 800, "slot_id": s,
                "slot_disrupted": True, "distance_m": 0.3} for s in (3, 5)]
    data = {"job": "seg", "bindings": binds, "frame_nums": frames, "reads": reads,
            "positions": positions, "votes_by_slot": {}, "reject_rows": rejects,
            "boundary_rows": []}
    if exact:
        data["boundary_rows"] = [{"type": "slot_boundary", "frame": 800, "slot_id": s,
                                  "kinds": ["birth"], "birth_disrupted": True}
                                 for s in (3, 5)]
    return data


def test_run_segment_check_labels_both_halves_and_beats_gir():
    res = ioc.run_segment_check(_segment_data(), TRUTH, auto_end=600, holdout_runs=10,
                                min_conf=0.65)
    assert res["boundary_source"] == "reconstructed" and res["boundary_rows"] == 2
    assert "R1=6800[0-792]>4256[800-1592]" in res["seg_summary"]
    assert "R3=4256[0-792]>6800[800-1592]" in res["seg_summary"]
    assert res["auto"]["seg"]["slots_ok"] == 6 and res["auto"]["gir"]["slots_ok"] == 4
    assert res["coverage"]["seg"] == 1.0
    assert res["exclusivity"]["seg"] == {"violations": 0, "duplicates": 0}
    ho = res["holdout"]
    assert ho["seg"]["agree"] > ho["gir"]["agree"]
    assert set(ho) >= {"gir", "swap", "seg", "gir+floor", "seg+floor"}
    assert res["floor"]["dropped"] == 0                  # no clef_probs rows
    text = ioc.format_segment_report(res)
    assert "boundaries: reconstructed" in text and "seg+floor" in text


def test_run_segment_check_prefers_exact_boundary_rows():
    res = ioc.run_segment_check(_segment_data(exact=True), TRUTH, holdout_runs=0)
    assert res["boundary_source"] == "exact" and "holdout" not in res
    assert "R1=6800[0-792]>4256[800-1592]" in res["seg_summary"]


def test_holdout_multi_matches_holdout_agreement_for_gir():
    binds, frames, reads, positions = _swap_job()
    ilog = ioc.attach_positions(ioc.reconstruct_identity_log(binds, frames), positions)
    blue, red = ["4786", "3181", "2992"], ["6800", "1477", "4256"]
    old = ioc.holdout_agreement(reads, ilog, {}, blue, red, ioc.SwapParams(), runs=7)

    def gir(train):
        gl = ioc._gir_labels(train, ilog, {}, blue, red, ioc._blank_meta())
        return lambda s, f: gl.get(s)
    new = ioc.holdout_multi(reads, ilog, blue, red, {"gir": gir}, runs=7)
    assert new["n"] == old["n"] and new["gir"]["agree"] == old["gir"]


def test_main_segments_flag_end_to_end(tmp_path, capsys):
    job = "segs1234-0000"
    binds, frames, reads, positions = _swap_job()
    con = sqlite3.connect(tmp_path / "frc_scorer.db")
    con.execute("CREATE TABLE match_robot_binding (id INTEGER PRIMARY KEY, job_id TEXT,"
                " identity_id INTEGER, slot INTEGER, start_frame INTEGER, end_frame INTEGER)")
    con.execute("CREATE TABLE match_robot_identity (job_id TEXT, identity_id INTEGER,"
                " slot INTEGER, slot_label TEXT, alliance TEXT, team_number TEXT,"
                " confirmed INTEGER, source TEXT, confidence REAL)")
    for iid, s, start, _end in binds:
        con.execute("INSERT INTO match_robot_binding (job_id, identity_id, slot, start_frame)"
                    " VALUES (?,?,?,?)", (job, iid, s, start))
    con.commit()
    con.close()
    (tmp_path / "web" / "uploads").mkdir(parents=True)
    (tmp_path / "web" / "replays" / job).mkdir(parents=True)
    labels = ["B1", "B2", "B3", "R1", "R2", "R3"]
    (tmp_path / "web" / "replays" / job / "meta.json").write_text(json.dumps(
        {"frames": [{"frame_num": f, "robots": [
            {"slot": labels[s], "field_pos": positions[(f, s)]} for s in range(6)]}
            for f in frames]}))
    rows = reads + [{"type": "identity_reject", "frame": 800, "slot_id": s,
                     "slot_disrupted": True, "distance_m": 0.3} for s in (3, 5)]
    (tmp_path / "web" / "uploads" / f"diagnostic_{job}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    rc = ioc.main([job[:6], "--data-root", str(tmp_path), "--segments", "--fps", "60",
                   "--holdout-runs", "5", "--auto-end", "600", "--min-conf", "0.65",
                   "--truth", ",".join(f"{k}={v}" for k, v in TRUTH.items())])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[segment-resolve] segments=8 read=8" in out
    assert "boundaries: reconstructed (2 rows)" in out
    assert "seg+floor" in out and "held-out: 5 x 30%" in out

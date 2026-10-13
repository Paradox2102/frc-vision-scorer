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

"""Score a replay pulled from the pod with the existing local scorers.

Runs LOCALLY under the main checkout's .venv (numpy etc.); called by
``runpod_cli.py tune`` after each rung. Reuses, unmodified:
  * scripts/ab_qm8.score()                   Manta auto-path mean-NN + coverage
  * scripts/score_ground_truth.score_replay() hand-GT coverage / error / labels / misID

Both read a replay row from ``frc_scorer.db`` and ``web/replays/<id>/meta.json``.
The pod's replay is NOT in the local DB (and must never be written there), so
this builds a throwaway sqlite DB in a temp dir holding just that replay row
(+ the match roster copied read-only from the main checkout's DB) and points the
scorer modules' DB_PATH / REPLAYS_DIR globals at it. Manta truth is read
cache-only from the main checkout's data/manta_cache (no API calls), and the GT
docs / calibration come from the main checkout too.

    .venv/bin/python deploy/runpod/score_pulled.py --replay-json row.json \
        --meta pulled/<id>/meta.json --video web/uploads/<qm8>.mp4 [--score-shift 0.475]
Prints one JSON object.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def main_checkout() -> Path:
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        p = Path(common)
        if not p.is_absolute():
            p = (REPO / p).resolve()
        return p.parent
    except Exception:
        return REPO


def build_db(db: Path, row: dict, meta: Path, video: Path, main_db: Path) -> None:
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE replays (id TEXT PRIMARY KEY, match_key TEXT, event_key TEXT, "
                "meta_path TEXT, created_at TEXT, blue_teams TEXT, red_teams TEXT, "
                "source_video_path TEXT, status TEXT)")
    con.execute("CREATE TABLE matches (id INTEGER PRIMARY KEY, tba_match_key TEXT)")
    con.execute("CREATE TABLE match_robots (match_id INTEGER, team_number TEXT, alliance TEXT)")
    con.execute("INSERT INTO replays VALUES (?,?,?,?,?,?,?,?,?)", (
        row["id"], row["match_key"], row.get("event_key") or row["match_key"].split("_")[0],
        str(meta), row.get("created_at") or "", row.get("blue_teams"), row.get("red_teams"),
        str(video), "complete"))
    if main_db.exists():
        src = sqlite3.connect(f"file:{main_db}?mode=ro", uri=True)
        try:
            for mid, key in src.execute("SELECT id, tba_match_key FROM matches WHERE tba_match_key = ?",
                                        (row["match_key"],)).fetchall():
                con.execute("INSERT INTO matches VALUES (?,?)", (mid, key))
                for t, al in src.execute("SELECT team_number, alliance FROM match_robots WHERE match_id = ?",
                                         (mid,)).fetchall():
                    con.execute("INSERT INTO match_robots VALUES (?,?,?)", (mid, t, al))
        finally:
            src.close()
    con.commit()
    con.close()


def run_audit(row: dict, audit_dir: Path, main: Path, *, n: int, ground_center_offset: float) -> dict:
    """Jev label audit of this replay (scripts/audit_replay.py sample -> jev ->
    score), inside the throwaway DB set up by main(). Never raises."""
    import argparse as _ap
    from scripts import audit_replay as ar
    ar.REPO = main            # .env (Jev token) + the job's detections file live in the main checkout
    audit_dir.mkdir(parents=True, exist_ok=True)
    answers = audit_dir / "answers_jev_multi.jsonl"
    score = audit_dir / "audit_score.json"
    try:
        if not (audit_dir / "items.jsonl").exists():
            ar.cmd_sample(_ap.Namespace(replay=row["id"], n=n, seed=0, unlabelled_frac=ar.UNLABELLED_FRAC,
                                        auto_min_share=ar.AUTO_MIN_SHARE,
                                        ground_center_offset=ground_center_offset,
                                        out=str(audit_dir), meta=None))
        if not answers.exists():
            ar.cmd_jev(_ap.Namespace(audit=str(audit_dir), mode="multi", out=str(answers), limit=0))
        ar.cmd_score(_ap.Namespace(audit=str(audit_dir), answers=str(answers), min_conf=ar.DEFAULT_MIN_CONF,
                                   no_gt=True, dump=None, json=str(score)))
        return json.loads(score.read_text())
    except (Exception, SystemExit) as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-json", required=True, help="pod replay row (id, match_key, ...) as JSON file")
    ap.add_argument("--meta", required=True, help="pulled meta.json")
    ap.add_argument("--video", required=True, help="local copy of the source video (GT fingerprint/fps)")
    ap.add_argument("--match", type=int, default=8, help="qual match number for the Manta score")
    ap.add_argument("--score-shift", type=float, default=0.475)
    ap.add_argument("--event", default=None, help="event for the Manta score (default: the match key's)")
    ap.add_argument("--no-manta", action="store_true", help="skip the Manta score (no cached truth)")
    ap.add_argument("--period", choices=("full", "auto"), default="full",
                    help="hand-GT scoring period (auto for auto_period_only jobs)")
    ap.add_argument("--audit-dir", default=None,
                    help="ALSO run the Jev label audit (scripts/audit_replay.py sample + jev + score; "
                         "CALLS THE JEV API, ~$0.25) into this dir")
    ap.add_argument("--audit-n", type=int, default=150)
    ap.add_argument("--ground-center-offset", type=float, default=0.0,
                    help="the job's ground_center_offset_m (audit crops)")
    a = ap.parse_args()

    row = json.loads(Path(a.replay_json).read_text())
    main = main_checkout()
    out: dict = {"replay_id": row["id"]}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        replays = td / "replays"
        (replays / row["id"]).mkdir(parents=True)
        shutil.copy(a.meta, replays / row["id"] / "meta.json")
        db = td / "frc_scorer.db"
        build_db(db, row, replays / row["id"] / "meta.json", Path(a.video), main / "frc_scorer.db")

        from scripts import compare_manta, score_ground_truth, manta_client
        from web.field import store as field_store
        compare_manta.DB_PATH = db
        compare_manta.REPLAYS_DIR = replays
        score_ground_truth.DB_PATH = db
        score_ground_truth.REPLAYS_DIR = replays
        # calibration for the scoring-side half-robot shift: the main checkout's
        # (identical to what was shipped to the pod)
        field_store.UPLOAD_DIR = main / "web" / "uploads"

        class CacheOnlyManta(manta_client.MantaClient):
            def _get(self, path, *, cache_key=None, refresh=False):
                cf = self.cache_dir / f"{cache_key}.json" if cache_key else None
                if cf is not None and cf.exists():
                    return json.loads(cf.read_text())
                raise LookupError(f"not in Manta cache: {cache_key}")

        cache_dir = main / "data" / "manta_cache"
        compare_manta.MantaClient = lambda: CacheOnlyManta(api_key="", cache_dir=cache_dir)

        from scripts import ab_qm8
        ab_qm8.MATCH = a.match
        ab_qm8.EVENT_KEY = a.event or row.get("event_key") or row["match_key"].split("_")[0]
        if a.no_manta:
            out["manta"] = {"skipped": "no cached Manta truth for this match"}
        else:
            try:
                out["manta"] = ab_qm8.score(expect_replay=row["id"])
            except Exception as e:
                out["manta"] = {"error": f"{type(e).__name__}: {e}"}

        from web.field.ground_truth import GT_DIR
        gt_dir = GT_DIR if GT_DIR.exists() else main / "data" / "eval" / "ground_truth"
        res = score_ground_truth.score_replay(row["id"], period=a.period, gt_dir=gt_dir,
                                              score_shift_m=a.score_shift)
        if "error" in res:
            out["gt"] = {"error": res["error"]}
        else:
            h, i = res["headline"], res["identity"]
            out["gt"] = {
                "n_robots": h["n_robots"], "n_processed": h.get("n_processed"),
                "coverage": h["coverage"],
                "err_median_m": h["err"]["median"], "err_p90_m": h["err"]["p90"],
                "label_correct": h["label_correct"], "label_wrong": h["label_wrong"],
                "unlabeled": h["unlabeled"], "slot_switches": h["slot_switches"],
                "misid_s": i["misid_s"], "lost_s": i["lost_s"],
                "warnings": res.get("warnings", []),
                "score_shift_m": a.score_shift, "period": a.period,
            }
        if a.audit_dir:
            out["audit"] = run_audit(row, Path(a.audit_dir), main, n=a.audit_n,
                                     ground_center_offset=a.ground_center_offset)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Persist camera calibrations as JSON under web/uploads/."""
from __future__ import annotations
import json
import time
import glob
from pathlib import Path
import numpy as np
from web.field.calibration import CameraCalibration

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"


def _path(event_key: str, camera: str) -> Path:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"calibration_{safe}.json"


def save(event_key: str, camera: str, cal: CameraCalibration) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _path(event_key, camera).write_text(json.dumps(cal.to_json(), indent=2))


def load(event_key: str, camera: str) -> CameraCalibration | None:
    """Most-recent calibration for back-compat callers that don't need epoch
    selection (spec: keep `load` returning the latest version). Considers both
    the versioned files and the legacy single file (via `load_versions`); returns
    None only if neither exists. Callers that need match-ordered epoch selection
    should use `load_versions` + `select_epoch` instead."""
    versions = load_versions(event_key, camera)
    if not versions:
        return None
    return max(versions, key=lambda v: v.get("created_at") or "")["calibration"]


def load_intrinsics(event_key: str, camera: str):
    cal = load(event_key, camera)
    return (None, None) if cal is None else (cal.K, np.asarray(cal.D).reshape(-1))


def _version_path(event_key: str, camera: str, version_id: str) -> Path:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"calibration_{safe}__{version_id}.json"


def _version_glob(event_key: str, camera: str) -> str:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return str(UPLOAD_DIR / f"calibration_{safe}__*.json")


def save_version(event_key, camera, cal, start_match=None, label=None,
                 source=None) -> str:
    """Write a NEW calibration version (never overwrites). Returns its version_id.
    ``source`` optionally records the solve inputs (correspondences + the frame's
    match/timestamp) so the calibrate UI can restore and re-preview them later."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    version_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time()*1e6) % 1_000_000:06d}"
    payload = {"calibration": cal.to_json(), "start_match": start_match,
               "label": label, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "version_id": version_id, "source": source}
    _version_path(event_key, camera, version_id).write_text(json.dumps(payload, indent=2))
    return version_id


def _read_version_file(p: Path) -> dict:
    d = json.loads(p.read_text())
    return {"calibration": CameraCalibration.from_json(d["calibration"]),
            "start_match": d.get("start_match"), "label": d.get("label"),
            "created_at": d.get("created_at"), "version_id": d["version_id"],
            "source": d.get("source")}


def load_versions(event_key, camera) -> list[dict]:
    """All calibration versions for (event,camera): versioned files + the legacy
    single file (as a start_match=None version). Sorted by created_at."""
    out = []
    for fp in glob.glob(_version_glob(event_key, camera)):
        try:
            out.append(_read_version_file(Path(fp)))
        except Exception:
            pass
    legacy = _path(event_key, camera)
    if legacy.exists():
        cal = CameraCalibration.from_json(json.loads(legacy.read_text()))
        out.append({"calibration": cal, "start_match": None, "label": "legacy",
                    "created_at": "", "version_id": "legacy", "source": None})
    out.sort(key=lambda v: v.get("created_at") or "")
    return out


def _drift_path(event_key: str) -> Path:
    safe = "".join(c for c in event_key if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"calibration_drift_{safe}.json"


def load_drift(event_key: str) -> dict:
    """Failed drift-check records: {camera: {match_key: {error_m, epoch, at}}}."""
    p = _drift_path(event_key)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def record_drift(event_key, camera, match_key, error_m, epoch) -> None:
    """Remember that calibration ``epoch`` failed the drift check on ``match_key``
    (camera likely moved). Cleared by clear_drift when a later check passes;
    consumers also treat a record as resolved once a different epoch covers the
    match (i.e. after recalibration)."""
    d = load_drift(event_key)
    d.setdefault(camera, {})[match_key] = {
        "error_m": round(float(error_m), 3), "epoch": epoch,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _drift_path(event_key).write_text(json.dumps(d, indent=2))


def clear_drift(event_key, camera, match_key) -> None:
    """Drop the failure record once a drift check passes for the match."""
    d = load_drift(event_key)
    if d.get(camera, {}).pop(match_key, None) is not None:
        _drift_path(event_key).write_text(json.dumps(d, indent=2))


def update_version(event_key, camera, version_id, cal) -> None:
    """Overwrite an existing version's calibration in place (property edits), keeping
    its start_match/label/created_at. 'legacy' updates the legacy single file."""
    if version_id == "legacy":
        save(event_key, camera, cal)
        return
    p = _version_path(event_key, camera, version_id)
    d = json.loads(p.read_text())
    d["calibration"] = cal.to_json()
    p.write_text(json.dumps(d, indent=2))

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
            "source": d.get("source"), "rejected": bool(d.get("rejected"))}


def load_versions(event_key, camera, include_rejected: bool = False) -> list[dict]:
    """All calibration versions for (event,camera): versioned files + the legacy
    single file (as a start_match=None version). Sorted by created_at.

    Versions an operator REJECTED (``set_version_rejected``) are left out unless
    ``include_rejected`` — so select_epoch, the job and every other consumer
    ignore them; listing UIs can opt back in."""
    out = []
    for fp in glob.glob(_version_glob(event_key, camera)):
        try:
            v = _read_version_file(Path(fp))
        except Exception:
            continue
        if v["rejected"] and not include_rejected:
            continue
        out.append(v)
    legacy = _path(event_key, camera)
    if legacy.exists():
        cal = CameraCalibration.from_json(json.loads(legacy.read_text()))
        out.append({"calibration": cal, "start_match": None, "label": "legacy",
                    "created_at": "", "version_id": "legacy", "source": None})
    out.sort(key=lambda v: v.get("created_at") or "")
    return out


def set_version_rejected(event_key, camera, version_id, rejected: bool = True) -> bool:
    """Flag (or un-flag) a version as rejected by an operator. The file is kept —
    rejection is reversible. Returns False if the version does not exist."""
    if version_id == "legacy":
        return False
    p = _version_path(event_key, camera, version_id)
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    d["rejected"] = bool(rejected)
    _atomic_write(p, json.dumps(d, indent=2).encode())
    return True


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


def _pending_autofix_path(event_key: str, camera: str) -> Path:
    safe = "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")
    return UPLOAD_DIR / f"calibration_autofix_{safe}.json"


def save_pending_autofix(event_key, camera, match_key, cal, summary, base_epoch=None) -> None:
    """Stage an auto-solved pose fix AWAITING HUMAN APPROVAL. Kept separate from the
    live versions so ``select_epoch`` never adopts it until approved — approval
    promotes it to a real version, rejection clears it (falling back to manual)."""
    p = _pending_autofix_path(event_key, camera)
    d = json.loads(p.read_text()) if p.exists() else {}
    d[match_key] = {"calibration": cal.to_json(), "summary": summary,
                    "base_epoch": base_epoch, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))


def load_pending_autofix(event_key, camera, match_key) -> dict | None:
    """Staged fix for a match, or None. Returns {calibration: CameraCalibration,
    summary, base_epoch, created_at}."""
    p = _pending_autofix_path(event_key, camera)
    if not p.exists():
        return None
    rec = json.loads(p.read_text()).get(match_key)
    if not rec:
        return None
    return {"calibration": CameraCalibration.from_json(rec["calibration"]),
            "summary": rec.get("summary"), "base_epoch": rec.get("base_epoch"),
            "created_at": rec.get("created_at")}


def clear_pending_autofix(event_key, camera, match_key) -> None:
    p = _pending_autofix_path(event_key, camera)
    if not p.exists():
        return
    d = json.loads(p.read_text())
    if d.pop(match_key, None) is not None:
        p.write_text(json.dumps(d, indent=2))


def _auto_safe(event_key: str, camera: str) -> str:
    return "".join(c for c in f"{event_key}_{camera}" if c.isalnum() or c in "_-")


def auto_record_path(event_key: str, camera: str) -> Path:
    return UPLOAD_DIR / f"calibration_auto_{_auto_safe(event_key, camera)}.json"


def auto_preview_name(event_key: str, camera: str) -> str:
    """File name (not path) of the auto-calibration overlay JPEG; it lives in
    UPLOAD_DIR next to the record."""
    return f"calibration_auto_{_auto_safe(event_key, camera)}.jpg"


def auto_preview_path(event_key: str, camera: str) -> Path:
    return UPLOAD_DIR / auto_preview_name(event_key, camera)


def auto_frame_name(event_key: str, camera: str) -> str:
    """File name of the raw temporal-median camera frame (no overlay)."""
    return f"calibration_auto_{_auto_safe(event_key, camera)}_frame.jpg"


def auto_frame_path(event_key: str, camera: str) -> Path:
    return UPLOAD_DIR / auto_frame_name(event_key, camera)


def _atomic_write(p: Path, data: bytes) -> None:
    # The calibrate page polls the record while the runner rewrites it — write to a
    # sibling temp file and rename so a reader never sees a half-written file.
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:      # unique temp name: concurrent writers
            f.write(data)                   # never share (or clobber) one temp file
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_auto_record(event_key, camera, record: dict) -> None:
    """Persist the automatic-calibration record for (event, camera). One record per
    pair, overwritten on each run. It is a REVIEW artifact: nothing here is ever
    read by ``load_versions``/``select_epoch`` — confirming promotes the candidate
    to a real version via ``save_version``."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(auto_record_path(event_key, camera),
                  json.dumps(record, indent=2).encode())


def load_auto_record(event_key, camera) -> dict | None:
    p = auto_record_path(event_key, camera)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def update_auto_record(event_key, camera, **fields) -> dict | None:
    """Merge ``fields`` into the existing record and persist it. Returns the updated
    record, or None if there is no record to update."""
    rec = load_auto_record(event_key, camera)
    if rec is None:
        return None
    rec.update(fields)
    save_auto_record(event_key, camera, rec)
    return rec


def save_auto_preview(event_key, camera, jpeg_bytes: bytes) -> str:
    """Write the overlay JPEG next to the record; returns its file name."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(auto_preview_path(event_key, camera), jpeg_bytes)
    return auto_preview_name(event_key, camera)


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


def save_auto_frame(event_key, camera, jpeg_bytes: bytes) -> str:
    """Write the raw median frame JPEG next to the record; returns its file name."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(auto_frame_path(event_key, camera), jpeg_bytes)
    return auto_frame_name(event_key, camera)


# ---- per-match corner calibration log ------------------------------------------

def permatch_log_path(event_key: str, camera: str) -> Path:
    return UPLOAD_DIR / f"calibration_permatch_{_auto_safe(event_key, camera)}.json"


def permatch_preview_name(event_key: str, camera: str, match_key: str) -> str:
    safe_m = "".join(c for c in str(match_key) if c.isalnum() or c in "_-")
    return f"calibration_permatch_{_auto_safe(event_key, camera)}__{safe_m}.jpg"


def permatch_preview_path(event_key: str, camera: str, match_key: str) -> Path:
    return UPLOAD_DIR / permatch_preview_name(event_key, camera, match_key)


def load_permatch_log(event_key, camera) -> dict:
    """{match_key: entry} for one camera; {} when none."""
    p = permatch_log_path(event_key, camera)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_permatch_entry(event_key, camera, match_key, entry: dict) -> dict:
    """Write (replace) one match's entry; returns it."""
    d = load_permatch_log(event_key, camera)
    d[str(match_key)] = entry
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(permatch_log_path(event_key, camera), json.dumps(d, indent=2).encode())
    return entry


def update_permatch_entry(event_key, camera, match_key, **fields) -> dict | None:
    d = load_permatch_log(event_key, camera)
    e = d.get(str(match_key))
    if e is None:
        return None
    e.update(fields)
    _atomic_write(permatch_log_path(event_key, camera), json.dumps(d, indent=2).encode())
    return e


def save_permatch_preview(event_key, camera, match_key, jpeg_bytes: bytes) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(permatch_preview_path(event_key, camera, match_key), jpeg_bytes)
    return permatch_preview_name(event_key, camera, match_key)

"""Diagnostic event/snapshot recorder for inference jobs.

Captures structured records of the identity pipeline's decisions and writes them
as JSONL (one record per line) for both live consumption via WebSocket and
post-match replay.

The recorder is the only place that knows the diagnostic schema — callers pass
typed arguments, never raw dicts. Failures are isolated: any I/O error degrades
the recorder to a no-op state so inference is never blocked.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

logger = logging.getLogger(__name__)

# Records are flushed to disk every N writes. Small enough to bound data loss
# on crash; large enough that we don't hammer fsync per event.
FLUSH_EVERY_N_WRITES = 32
DEFAULT_RING_SIZE = 2000


class NullRecorder:
    """Drop-in replacement that swallows all `record_*`, `flush`, `close` calls.

    Used when the recorder cannot open its output file. Inference keeps running;
    diagnostic data is silently dropped.
    """

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("record_") or name in ("flush", "close", "set_ws_send"):
            def noop(*_a: Any, **_kw: Any) -> None:
                return None
            return noop
        raise AttributeError(name)

    @property
    def is_null(self) -> bool:
        return True


class DiagnosticRecorder:
    """Writes diagnostic events/snapshots to JSONL and optionally pushes them
    via a WebSocket send callable.

    Args:
        out_path: JSONL file to append to (opened in 'a' mode).
        ws_send: Optional async function called with each record dict for live
                 streaming. Errors are caught and logged; never raised.
        ring_size: Number of most-recent records to keep in memory for late-
                   joining WS clients.
    """

    def __init__(
        self,
        out_path: Path,
        ws_send: Callable[[dict], Awaitable[None]] | None = None,
        ring_size: int = DEFAULT_RING_SIZE,
    ) -> None:
        self._out_path = Path(out_path)
        self._ws_send = ws_send
        self._ring_size = ring_size
        self._ring: list[dict] = []
        self._t0 = time.monotonic()
        self._writes_since_flush = 0
        try:
            self._fp = open(self._out_path, "a", buffering=1, encoding="utf-8")
        except OSError as e:
            logger.error("DiagnosticRecorder: failed to open %s: %s", self._out_path, e)
            self._fp = None

    @property
    def is_null(self) -> bool:
        return self._fp is None

    def _t_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    def _emit(self, record: dict) -> None:
        if self._fp is None:
            return
        try:
            line = json.dumps(record, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            logger.warning("DiagnosticRecorder: dropping malformed record (%s): %r", e, record)
            return
        try:
            self._fp.write(line + "\n")
            self._writes_since_flush += 1
            if self._writes_since_flush >= FLUSH_EVERY_N_WRITES:
                self._fp.flush()
                self._writes_since_flush = 0
        except OSError as e:
            logger.error("DiagnosticRecorder: write failed, degrading to null: %s", e)
            self._close_fp()
            return

        self._ring.append(record)
        if len(self._ring) > self._ring_size:
            del self._ring[: len(self._ring) - self._ring_size]

        if self._ws_send is not None:
            try:
                result = self._ws_send(record)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No running loop in this context — drop the coroutine cleanly
                        # so we don't get a "coroutine was never awaited" warning.
                        result.close()
                        logger.debug("DiagnosticRecorder: ws_send returned coroutine but no running loop; dropped")
                    else:
                        task = loop.create_task(result)
                        # Log exceptions instead of leaving them as "Task exception was never retrieved".
                        task.add_done_callback(
                            lambda t: logger.warning("ws_send task failed: %s", t.exception())
                            if t.exception() else None
                        )
            except Exception as e:
                logger.warning("DiagnosticRecorder: ws_send failed: %s", e)

    # ---- typed record_* methods ----

    def record_vlm_read(self, frame: int | None, robot_id: int | None,
                       crop_b64: str | None, legibility: float,
                       raw_read: str | None, matched_team: str | None,
                       alliance_read: str, confidence: float, vote_weight: float) -> None:
        if frame is None or robot_id is None or crop_b64 is None:
            logger.warning("record_vlm_read: missing required field; dropped")
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "vlm_read",
            "robot_id": int(robot_id), "crop_b64": crop_b64,
            "legibility": float(legibility), "raw_read": raw_read,
            "matched_team": matched_team, "alliance_read": alliance_read,
            "confidence": float(confidence), "vote_weight": float(vote_weight),
        })

    def record_vlm_filtered(self, frame: int | None, robot_id: int | None,
                           crop_b64: str | None, legibility: float, reason: str) -> None:
        if frame is None or robot_id is None:
            logger.warning("record_vlm_filtered: missing required field; dropped")
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "vlm_filtered",
            "robot_id": int(robot_id), "crop_b64": crop_b64,
            "legibility": float(legibility), "reason": reason,
        })

    def record_assignment(self, frame: int | None, robot_id: int | None, team: str | None,
                         alliance: str, confirmed: bool,
                         votes_for_assigned: float, votes_total: float,
                         top_team_share: float) -> None:
        if frame is None or robot_id is None or team is None:
            logger.warning("record_assignment: missing required field; dropped")
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "assignment",
            "robot_id": int(robot_id), "team": str(team), "alliance": alliance,
            "confirmed": bool(confirmed),
            "votes_for_assigned": float(votes_for_assigned),
            "votes_total": float(votes_total),
            "top_team_share": float(top_team_share),
        })

    def record_elimination(self, frame: int | None, robot_id: int | None,
                          team: str | None, alliance: str) -> None:
        if frame is None or robot_id is None or team is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "elimination",
            "robot_id": int(robot_id), "team": str(team), "alliance": alliance,
        })

    def record_overturn(self, frame: int | None, robot_id: int | None,
                       old_team: str | None, challenger_team: str | None,
                       current_votes: float, challenger_votes: float) -> None:
        if frame is None or robot_id is None or old_team is None or challenger_team is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "overturn",
            "robot_id": int(robot_id), "old_team": str(old_team),
            "challenger_team": str(challenger_team),
            "current_votes": float(current_votes),
            "challenger_votes": float(challenger_votes),
        })

    def record_reid_match(self, frame: int | None, track_id: int | None,
                         canonical_id: int | None, similarity: float,
                         second_best: float | None) -> None:
        if frame is None or track_id is None or canonical_id is None:
            return
        gap = (float(similarity) - float(second_best)) if second_best is not None else None
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "reid_match",
            "track_id": int(track_id), "canonical_id": int(canonical_id),
            "similarity": float(similarity),
            "second_best": float(second_best) if second_best is not None else None,
            "gap": gap,
        })

    def record_reid_drop(self, frame: int | None, track_id: int | None, reason: str,
                        would_have_matched: int | None, similarity: float | None) -> None:
        if frame is None or track_id is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "reid_drop",
            "track_id": int(track_id), "reason": reason,
            "would_have_matched": int(would_have_matched) if would_have_matched is not None else None,
            "similarity": float(similarity) if similarity is not None else None,
        })

    def record_snapshot(self, frame: int | None, robots_state: list[dict],
                       roster: dict, identified: dict) -> None:
        """`robots_state` is a list of dicts pre-shaped per the snapshot schema
        (including the `embedding` sub-dict). The caller in app.py is responsible
        for building it from self.robot_tracks + the embedder's last-match scores.
        The recorder validates field presence on emit but does not introspect
        RobotTrack or the embedder."""
        if frame is None:
            return
        self._emit({
            "frame": int(frame), "t_ms": self._t_ms(), "type": "snapshot",
            "robots": list(robots_state or []),
            "roster": roster or {},
            "identified": identified or {},
        })

    # ---- lifecycle ----

    def set_ws_send(self, ws_send: Callable[[dict], Awaitable[None]] | None) -> None:
        """Attach (or replace) the WebSocket push callable after construction.
        Called by InferenceJob once the websocket connects. Pass None to detach."""
        self._ws_send = ws_send

    def flush(self) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
            except OSError as e:
                logger.warning("DiagnosticRecorder.flush: %s", e)

    def close(self) -> None:
        self.flush()
        self._close_fp()

    def _close_fp(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None


def replay_iter(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file. Skip malformed lines and log the count
    of skipped lines on completion."""
    p = Path(path)
    if not p.exists():
        return
    skipped = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
    if skipped:
        logger.info("replay_iter(%s): skipped %d malformed lines", path, skipped)

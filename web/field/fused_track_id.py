"""Stable cross-camera fused track ids for the SAM2 tracklet path.

SAM2 emits per-camera object ids that collide across cameras and restart per
prompt. SP3 fusion clusters per-camera detections of one physical robot; this
registry gives each such cluster ONE integer id that is stable across frames,
so the Phase-1 FieldTracker sticky binding turns it into a fixed slot.

A cluster is identified by its set of member keys ``(camera, per_camera_id)``.
Because each camera's id is stable within that camera, two clusters that share
ANY member key are the same physical robot — even across a member joining (a
corner-only robot later seen by main) or dropping out (main occluded). That is
exactly union-find over member keys: each member key remembers its fused id, and
a cluster that touches several known ids collapses them to one.

Pure: no torch / SAM2 / cv2 imports."""
from __future__ import annotations


class FusedTrackRegistry:
    def __init__(self) -> None:
        # member key (camera, per_camera_id) -> fused id
        self._key_to_id: dict[tuple, int] = {}
        self._next_id = 0

    def assign(self, observations: list) -> None:
        """Stamp a stable ``track_id`` onto each FusedObservation (and its
        ``detection``), mutating in place. Call once per frame, in order."""
        for obs in observations:
            keys = [tuple(m) for m in obs.members]
            # Fused ids already known for any member key in this cluster.
            known = {self._key_to_id[k] for k in keys if k in self._key_to_id}
            if known:
                fused_id = min(known)
                # Collapse any bridged ids: every key currently pointing at one of
                # the `known` ids is repointed to the winner.
                if len(known) > 1:
                    losers = known - {fused_id}
                    for k, v in list(self._key_to_id.items()):
                        if v in losers:
                            self._key_to_id[k] = fused_id
            else:
                fused_id = self._next_id
                self._next_id += 1
            # Register every member key of this cluster to the chosen id.
            for k in keys:
                self._key_to_id[k] = fused_id
            obs.track_id = fused_id
            obs.detection.track_id = fused_id

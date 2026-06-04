"""Per-event appearance gallery for recovering obscured bumper numbers.

FRC-agnostic and storage-agnostic: holds DINOv2 embeddings keyed by
(team, alliance_color), supports diverse-set deposits and nearest-team queries.
The integration layer (web/app.py) owns persistence and vote casting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from web.embedder import cosine_similarity


class _Exemplar:
    __slots__ = ("embedding", "legibility", "verified")

    def __init__(self, embedding: np.ndarray, legibility: float, verified: bool = False):
        # Copy: track embeddings are EMA-updated in place elsewhere, so storing a
        # reference would let later updates silently mutate a stored exemplar.
        self.embedding = np.array(embedding, dtype=np.float32, copy=True)
        self.legibility = float(legibility)
        self.verified = bool(verified)


class AppearanceGallery:
    """Diverse, capped sets of appearance embeddings per (team, alliance_color)."""

    def __init__(self, dedup_sim: float = 0.92, max_exemplars: int = 12):
        self.dedup_sim = dedup_sim
        self.max_exemplars = max_exemplars
        # {(team, color): [_Exemplar, ...]}
        self._sets: dict[tuple[str, str], list[_Exemplar]] = {}

    def add(self, team: str, color: str, embedding, legibility: float,
            verified: bool = False) -> bool:
        """Deposit an exemplar. Returns True if stored, False if rejected as a
        near-duplicate. Keeps the set diverse (dedup) and capped (evict lowest
        legibility)."""
        if embedding is None or color not in ("red", "blue") or not team:
            return False
        emb = np.asarray(embedding, dtype=np.float32)
        bucket = self._sets.setdefault((team, color), [])
        for ex in bucket:
            if cosine_similarity(emb, ex.embedding) >= self.dedup_sim:
                return False  # too similar to an existing viewpoint
        bucket.append(_Exemplar(emb, legibility, verified))
        if len(bucket) > self.max_exemplars:
            # Evict the least valuable exemplar: prefer to drop UNVERIFIED ones, and
            # among equal verified-status the lowest legibility. This protects
            # human-verified exemplars (corrections) from being capped out by
            # higher-confidence auto-deposits.
            worst = min(range(len(bucket)),
                        key=lambda i: (bucket[i].verified, bucket[i].legibility))
            bucket.pop(worst)
        return True

    def query(self, embedding, color: Optional[str]) -> list[tuple[str, float, bool]]:
        """Best cosine similarity per team, filtered to `color` (both if None),
        sorted descending. On a similarity tie, a verified exemplar wins.

        Returns 3-tuples ``(team, sim, verified)`` where ``verified`` is the flag
        of the best-matching exemplar for that team."""
        if embedding is None:
            return []
        emb = np.asarray(embedding, dtype=np.float32)
        best: dict[str, tuple[float, bool]] = {}
        for (team, c), bucket in self._sets.items():
            if color in ("red", "blue") and c != color:
                continue
            for ex in bucket:
                sim = cosine_similarity(emb, ex.embedding)
                cur = best.get(team)
                # take higher sim; on equal sim prefer a verified exemplar
                if cur is None or sim > cur[0] or (sim == cur[0] and ex.verified and not cur[1]):
                    best[team] = (sim, ex.verified)
        ranked = [(team, s, v) for team, (s, v) in best.items()]
        ranked.sort(key=lambda t: (t[1], t[2]), reverse=True)  # sim desc, verified-first on tie
        return ranked

    # --- introspection (used by tests) ---
    def exemplar_count(self, team: str, color: str) -> int:
        return len(self._sets.get((team, color), []))

    def legibilities(self, team: str, color: str) -> list[float]:
        return [ex.legibility for ex in self._sets.get((team, color), [])]

    # --- persistence bridge ---
    def serialize(self) -> list[dict]:
        rows = []
        for (team, color), bucket in self._sets.items():
            for ex in bucket:
                rows.append({
                    "team": team,
                    "alliance_color": color,
                    "embedding": ex.embedding.astype(np.float32).tobytes(),
                    "legibility": ex.legibility,
                    "verified": ex.verified,
                })
        return rows

    @classmethod
    def from_rows(cls, rows, dedup_sim: float = 0.92, max_exemplars: int = 12):
        g = cls(dedup_sim=dedup_sim, max_exemplars=max_exemplars)
        for r in rows:
            emb = r["embedding"]
            if isinstance(emb, (bytes, bytearray)):
                emb = np.frombuffer(emb, dtype=np.float32)
            # load directly, skipping dedup (the DB is the source of truth for which
            # viewpoints are distinct), but still honor the cap below.
            g._sets.setdefault((r["team"], r["alliance_color"]), []).append(
                _Exemplar(emb, r.get("legibility") or 0.0, bool(r.get("verified"))))
        # Re-cap each bucket: the DB is append-only across matches, so a bucket can
        # hold more rows than max_exemplars. Keep the highest-legibility ones so
        # loaded sets stay bounded (memory + per-query cost).
        for key, bucket in g._sets.items():
            if len(bucket) > max_exemplars:
                # Keep verified exemplars first, then highest legibility, so human
                # corrections survive the cap when a bucket exceeds max_exemplars.
                bucket.sort(key=lambda ex: (ex.verified, ex.legibility), reverse=True)
                g._sets[key] = bucket[:max_exemplars]
        return g


def classify_match(ranked: list[tuple], *, confirm_sim: float,
                   confirm_gap: float, supplement_sim: float):
    """Decide what a gallery query result warrants.

    Returns (verdict, team, sim) where verdict is 'confirm' | 'supplement' | 'none'.
    'second-best team' is the next DISTINCT team in the ranked list.
    """
    if not ranked:
        return ("none", None, 0.0)
    team, sim = ranked[0][0], ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if sim >= confirm_sim and (sim - second) >= confirm_gap:
        return ("confirm", team, sim)
    if sim >= supplement_sim:
        return ("supplement", team, sim)
    return ("none", None, sim)

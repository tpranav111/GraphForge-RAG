"""Retrieval fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

from advanced_graphrag.domain import Evidence


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, ranked_lists: Sequence[tuple[float, Sequence[Evidence]]], *, limit: int) -> tuple[Evidence, ...]:
        scores: dict[str, float] = defaultdict(float)
        payloads: dict[str, Evidence] = {}
        for weight, evidences in ranked_lists:
            for rank, evidence in enumerate(evidences, start=1):
                scores[evidence.id] += weight / (self.k + rank)
                payloads.setdefault(evidence.id, evidence)
        fused = [replace(payloads[eid], score=payloads[eid].score + scores[eid]) for eid in scores]
        fused.sort(key=lambda evidence: evidence.score, reverse=True)
        return tuple(fused[:limit])


def dedupe_evidence(items: Iterable[Evidence]) -> tuple[Evidence, ...]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        out.append(item)
    return tuple(out)


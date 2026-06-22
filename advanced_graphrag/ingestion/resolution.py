"""Entity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from advanced_graphrag.domain import Entity, Relationship
from advanced_graphrag.utils import stable_id


@dataclass(frozen=True)
class ResolutionResult:
    entities: tuple[Entity, ...]
    relationships: tuple[Relationship, ...]
    remap: dict[str, str]


class ExactTypeNameResolver:
    """Conservative entity resolver.

    This follows the production bias in FalkorDB/Neo4j: prefer deterministic
    safe merges, then add LLM-assisted verification later for ambiguous cases.
    """

    def resolve(
        self,
        entities: Sequence[Entity],
        relationships: Sequence[Relationship],
    ) -> ResolutionResult:
        grouped: dict[tuple[str, str], list[Entity]] = {}
        for entity in entities:
            grouped.setdefault((entity.type.lower(), _norm(entity.name)), []).append(entity)

        remap: dict[str, str] = {}
        resolved: list[Entity] = []
        for (etype, name_key), group in grouped.items():
            canonical = group[0]
            canonical_id = stable_id("entity", etype, name_key)
            aliases = sorted({alias for item in group for alias in item.aliases} | {item.name for item in group})
            description = max((item.description for item in group), key=len, default="")
            merged = Entity(
                id=canonical_id,
                name=canonical.name,
                type=canonical.type,
                description=description,
                aliases=tuple(aliases),
                properties=dict(canonical.properties),
            )
            resolved.append(merged)
            for item in group:
                remap[item.id] = canonical_id

        rewritten: dict[str, Relationship] = {}
        for rel in relationships:
            src = remap.get(rel.source_entity_id, rel.source_entity_id)
            tgt = remap.get(rel.target_entity_id, rel.target_entity_id)
            if src == tgt:
                continue
            new_rel = Relationship(
                id=stable_id("rel", src, rel.type, tgt, ",".join(rel.source_chunk_ids)),
                source_entity_id=src,
                target_entity_id=tgt,
                type=rel.type,
                fact=rel.fact,
                confidence=rel.confidence,
                source_chunk_ids=rel.source_chunk_ids,
                properties=rel.properties,
            )
            rewritten[new_rel.id] = new_rel

        return ResolutionResult(tuple(resolved), tuple(rewritten.values()), remap)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


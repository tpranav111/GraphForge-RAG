"""In-memory stores.

These stores are intentionally simple but implement production boundaries:
graph, vector, and artifacts are separate concerns. They are useful for tests,
small local corpora, and as reference implementations for database adapters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from advanced_graphrag.domain import (
    Chunk,
    Community,
    CommunitySummary,
    Document,
    Entity,
    Relationship,
)
from advanced_graphrag.utils import cosine


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.documents_by_id: dict[str, Document] = {}
        self.chunks_by_id: dict[str, Chunk] = {}
        self.entities_by_id: dict[str, Entity] = {}
        self.relationships_by_id: dict[str, Relationship] = {}
        self.communities_by_id: dict[str, Community] = {}
        self.summaries_by_id: dict[str, CommunitySummary] = {}
        self._adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def upsert_document(self, document: Document) -> None:
        self.documents_by_id[document.id] = document

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self.chunks_by_id[chunk.id] = chunk

    def upsert_entities(self, entities: Sequence[Entity]) -> None:
        for entity in entities:
            self.entities_by_id[entity.id] = entity

    def upsert_relationships(self, relationships: Sequence[Relationship]) -> None:
        for rel in relationships:
            self.relationships_by_id[rel.id] = rel
        self._rebuild_adjacency()

    def upsert_communities(self, communities: Sequence[Community]) -> None:
        for community in communities:
            self.communities_by_id[community.id] = community

    def upsert_community_summaries(self, summaries: Sequence[CommunitySummary]) -> None:
        for summary in summaries:
            self.summaries_by_id[summary.id] = summary

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks_by_id.get(chunk_id)

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities_by_id.get(entity_id)

    def entities(self) -> list[Entity]:
        return list(self.entities_by_id.values())

    def relationships(self) -> list[Relationship]:
        return list(self.relationships_by_id.values())

    def chunks(self) -> list[Chunk]:
        return list(self.chunks_by_id.values())

    def community_summaries(self) -> list[CommunitySummary]:
        return list(self.summaries_by_id.values())

    def neighbors(self, entity_id: str) -> list[tuple[str, Relationship]]:
        result = []
        for neighbor_id, rel_id in self._adjacency.get(entity_id, []):
            rel = self.relationships_by_id.get(rel_id)
            if rel:
                result.append((neighbor_id, rel))
        return result

    def symbolic_read(self, query: str) -> list[dict[str, Any]]:
        """Safe symbolic query placeholder.

        Database adapters should validate and execute read-only Cypher. This
        fallback supports a tiny debug grammar: `entity:<name fragment>`.
        """

        query = query.strip()
        if not query.lower().startswith("entity:"):
            return []
        needle = query.split(":", 1)[1].strip().lower()
        rows = []
        for entity in self.entities():
            if needle in entity.name.lower():
                rows.append({"id": entity.id, "name": entity.name, "type": entity.type})
        return rows

    def _rebuild_adjacency(self) -> None:
        self._adjacency.clear()
        for rel in self.relationships_by_id.values():
            self._adjacency[rel.source_entity_id].append((rel.target_entity_id, rel.id))
            self._adjacency[rel.target_entity_id].append((rel.source_entity_id, rel.id))


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, tuple[str, list[float], dict[str, Any]]]] = defaultdict(dict)

    def upsert(self, namespace: str, items: Sequence[tuple[str, str, Sequence[float], dict[str, Any]]]) -> None:
        for item_id, text, vector, metadata in items:
            self._items[namespace][item_id] = (text, list(vector), dict(metadata))

    def search(
        self,
        namespace: str,
        query_vector: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        rows = []
        for item_id, (text, vector, metadata) in self._items.get(namespace, {}).items():
            if filters and any(metadata.get(k) != v for k, v in filters.items()):
                continue
            score = cosine(query_vector, vector)
            enriched = dict(metadata)
            enriched["text"] = text
            rows.append((item_id, score, enriched))
        rows.sort(key=lambda row: row[1], reverse=True)
        return rows[:top_k]


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def put_json(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = dict(value)

    def get_json(self, key: str) -> dict[str, Any] | None:
        item = self._data.get(key)
        return dict(item) if item is not None else None


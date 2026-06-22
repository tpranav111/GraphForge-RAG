"""Entity and relationship extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from advanced_graphrag.domain import Chunk, Entity, Relationship
from advanced_graphrag.models.base import ChatMessage, LLMClient
from advanced_graphrag.utils import keywords, stable_id


@dataclass(frozen=True)
class ExtractionResult:
    entities: tuple[Entity, ...]
    relationships: tuple[Relationship, ...]


class GraphExtractor:
    def extract(self, chunks: Sequence[Chunk]) -> ExtractionResult:
        raise NotImplementedError


class RuleBasedExtractor(GraphExtractor):
    """Deterministic extraction fallback.

    It provides usable local behavior without an LLM. Production ingestion
    should use `LLMJsonGraphExtractor` with a strong local extraction model.
    """

    _CAPITALIZED = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4})\b")

    def extract(self, chunks: Sequence[Chunk]) -> ExtractionResult:
        entities_by_name: dict[str, Entity] = {}
        relationships: dict[str, Relationship] = {}
        for chunk in chunks:
            names = self._candidate_entities(chunk.text)
            for name in names:
                entity = self._entity(name, chunk.text)
                entities_by_name.setdefault(entity.name.lower(), entity)
            linked = list(dict.fromkeys(names))[:8]
            for left, right in zip(linked, linked[1:]):
                src = self._entity(left, chunk.text)
                tgt = self._entity(right, chunk.text)
                rel = Relationship(
                    id=stable_id("rel", src.id, "RELATED_TO", tgt.id, chunk.id),
                    source_entity_id=src.id,
                    target_entity_id=tgt.id,
                    type="RELATED_TO",
                    fact=f"{left} is contextually related to {right}.",
                    confidence=0.45,
                    source_chunk_ids=(chunk.id,),
                )
                relationships.setdefault(rel.id, rel)
        return ExtractionResult(tuple(entities_by_name.values()), tuple(relationships.values()))

    def _candidate_entities(self, text: str) -> list[str]:
        names = [match.group(1).strip() for match in self._CAPITALIZED.finditer(text)]
        if names:
            return names
        return [word.title() for word in keywords(text, limit=5)]

    def _entity(self, name: str, context: str) -> Entity:
        normalized = " ".join(name.split())
        return Entity(
            id=stable_id("entity", normalized.lower()),
            name=normalized,
            type="Concept",
            description=f"Entity mentioned in source text: {normalized}.",
        )


class LLMJsonGraphExtractor(GraphExtractor):
    """LLM-backed graph extraction with strict JSON contracts.

    This mirrors Neo4j/FalkorDB two-step extraction and KAG's schema-constrained
    builder idea. The extractor is intentionally conservative: malformed chunks
    are skipped rather than silently poisoning the graph.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_entities_per_chunk: int = 24,
        fallback_extractor: GraphExtractor | None = None,
    ) -> None:
        self.llm = llm
        self.max_entities_per_chunk = max_entities_per_chunk
        self.fallback_extractor = fallback_extractor

    def extract(self, chunks: Sequence[Chunk]) -> ExtractionResult:
        entities: dict[str, Entity] = {}
        relationships: dict[str, Relationship] = {}
        for chunk in chunks:
            payload = self._extract_chunk(chunk)
            if self._should_fallback(payload) and self.fallback_extractor:
                fallback = self.fallback_extractor.extract([chunk])
                for entity in fallback.entities:
                    entities[entity.id] = entity
                for rel in fallback.relationships:
                    relationships[rel.id] = rel
                continue
            for item in payload.get("entities", [])[: self.max_entities_per_chunk]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                entity = Entity(
                    id=stable_id("entity", name.lower(), item.get("type", "Concept")),
                    name=name,
                    type=str(item.get("type", "Concept")),
                    description=str(item.get("description", "")),
                    aliases=tuple(_list_value(item.get("aliases"))),
                    properties=_dict_value(item.get("properties")),
                )
                entities[entity.id] = entity
            name_to_id = {e.name.lower(): e.id for e in entities.values()}
            for item in payload.get("relationships", []):
                if not isinstance(item, dict):
                    continue
                src_name = str(item.get("source", "")).strip().lower()
                tgt_name = str(item.get("target", "")).strip().lower()
                src_id = name_to_id.get(src_name)
                tgt_id = name_to_id.get(tgt_name)
                if not src_id or not tgt_id or src_id == tgt_id:
                    continue
                rel_type = str(item.get("type", "RELATED_TO")).upper()
                fact = str(item.get("fact") or item.get("description") or "")
                confidence = float(item.get("confidence", 0.75))
                rel = Relationship(
                    id=stable_id("rel", src_id, rel_type, tgt_id, chunk.id),
                    source_entity_id=src_id,
                    target_entity_id=tgt_id,
                    type=rel_type,
                    fact=fact,
                    confidence=confidence,
                    source_chunk_ids=(chunk.id,),
                    properties=_dict_value(item.get("properties")),
                )
                relationships[rel.id] = rel
        return ExtractionResult(tuple(entities.values()), tuple(relationships.values()))

    def _extract_chunk(self, chunk: Chunk) -> dict[str, object]:
        system = (
            "You extract concise knowledge graphs. Return valid JSON only. "
            "Do not include markdown fences, commentary, or explanations."
        )
        prompt = (
            "Extract a knowledge graph from the chunk. Return only JSON with "
            "keys `entities` and `relationships`.\n"
            "Entity schema: {\"name\": string, \"type\": string, \"description\": string, "
            "\"aliases\": string[], \"properties\": object}.\n"
            "Relationship schema: {\"source\": entity name, \"target\": entity name, "
            "\"type\": uppercase relation, \"fact\": grounded source fact, "
            "\"confidence\": number between 0 and 1, \"properties\": object}.\n"
            "Only extract facts explicitly supported by the text.\n\n"
            f"Chunk ID: {chunk.id}\nText:\n{chunk.text}"
        )
        response = self.llm.complete([ChatMessage("system", system), ChatMessage("user", prompt)], temperature=0.0)
        try:
            parsed = json.loads(_extract_json_object(response))
        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}
        if not isinstance(parsed, dict):
            return {"entities": [], "relationships": []}
        entities = parsed.get("entities", [])
        relationships = parsed.get("relationships", [])
        return {
            "entities": entities if isinstance(entities, list) else [],
            "relationships": relationships if isinstance(relationships, list) else [],
        }

    def _should_fallback(self, payload: dict[str, object]) -> bool:
        return not payload.get("entities") and not payload.get("relationships")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    text = _strip_fences(text)
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def _list_value(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def _dict_value(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}

"""Core domain objects.

The schema intentionally separates lexical provenance, extracted graph facts,
community summaries, and retrieval evidence. This is the common denominator of
Microsoft GraphRAG, LightRAG, HippoRAG, AWS/FalkorDB, Neo4j, and KAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    ENTITY = "Entity"
    RELATIONSHIP = "Relationship"
    COMMUNITY = "Community"
    COMMUNITY_SUMMARY = "CommunitySummary"


class QueryIntent(str, Enum):
    FACT = "fact"
    MULTI_HOP = "multi_hop"
    GLOBAL = "global"
    COMPARISON = "comparison"
    ENUMERATION = "enumeration"
    SYMBOLIC = "symbolic"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: str = "Concept"
    description: str = ""
    aliases: tuple[str, ...] = ()
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    id: str
    source_entity_id: str
    target_entity_id: str
    type: str
    fact: str
    confidence: float = 1.0
    source_chunk_ids: tuple[str, ...] = ()
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Community:
    id: str
    entity_ids: tuple[str, ...]
    level: int = 0
    parent_id: str | None = None


@dataclass(frozen=True)
class CommunitySummary:
    id: str
    community_id: str
    text: str
    entity_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryPlan:
    query: str
    intent: QueryIntent
    keywords: tuple[str, ...]
    entities: tuple[str, ...]
    subqueries: tuple[str, ...] = ()
    symbolic_query: str | None = None


@dataclass(frozen=True)
class Evidence:
    id: str
    kind: str
    text: str
    score: float
    source_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalBundle:
    query_plan: QueryPlan
    evidence: tuple[Evidence, ...]
    context: str


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[dict[str, Any], ...]
    graph_evidence: tuple[dict[str, Any], ...]
    confidence: float
    retrieval: RetrievalBundle


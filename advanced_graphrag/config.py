"""Configuration for the GraphRAG runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QueryMode(str, Enum):
    """Query modes inspired by LightRAG and Microsoft GraphRAG."""

    AUTO = "auto"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    SYMBOLIC = "symbolic"
    NAIVE = "naive"


@dataclass(frozen=True)
class ModelRoleConfig:
    """Model role separation.

    Production deployments should avoid using a single model for every role.
    This mirrors LightRAG's role-specific model configuration and KAG's planner,
    retriever, and generator split.
    """

    extraction_model: str = "local-extractor"
    planner_model: str = "local-planner"
    answer_model: str = "local-answer"
    judge_model: str = "local-judge"
    embedding_model: str = "local-embedding"
    reranker_model: str = "local-reranker"


@dataclass(frozen=True)
class RetrievalWeights:
    """Weights for reciprocal-rank and score fusion."""

    chunk_vector: float = 1.0
    chunk_keyword: float = 0.9
    entity_vector: float = 0.8
    relationship_vector: float = 1.1
    graph_walk: float = 1.1
    community: float = 0.8
    symbolic: float = 1.2


@dataclass(frozen=True)
class GraphRAGConfig:
    """Top-level runtime configuration."""

    chunk_size: int = 700
    chunk_overlap: int = 90
    max_context_tokens: int = 6000
    max_entities: int = 32
    max_relationships: int = 48
    max_chunks: int = 16
    max_communities: int = 6
    ppr_alpha: float = 0.65
    ppr_max_iter: int = 30
    ppr_tolerance: float = 1e-8
    min_relationship_confidence: float = 0.30
    enable_symbolic_queries: bool = True
    enable_community_summaries: bool = True
    model_roles: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    retrieval_weights: RetrievalWeights = field(default_factory=RetrievalWeights)


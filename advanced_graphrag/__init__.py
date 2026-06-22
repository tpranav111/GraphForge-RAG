"""Advanced GraphRAG package.

The package exposes a production-shaped GraphRAG stack with local defaults.
Backends are intentionally abstracted so the same orchestration can run with
in-memory stores in tests and graph/vector databases in production.
"""

from advanced_graphrag.config import GraphRAGConfig, ModelRoleConfig
from advanced_graphrag.engine import GraphRAGEngine
from advanced_graphrag.models.huggingface import (
    DEFAULT_HF_MODEL_PROFILE,
    HF_QWEN25_BGE_M3,
    HF_QWEN3_BGE_M3,
    HuggingFaceModelProfile,
    SentenceTransformerEmbeddingModel,
    TransformersCausalLLM,
    TransformersSequenceReranker,
)
from advanced_graphrag.models.local import HashingEmbeddingModel, HeuristicLLM
from advanced_graphrag.stores.filesystem import FileSystemArtifactStore
from advanced_graphrag.stores.memory import InMemoryArtifactStore, InMemoryGraphStore, InMemoryVectorStore

__all__ = [
    "DEFAULT_HF_MODEL_PROFILE",
    "FileSystemArtifactStore",
    "GraphRAGConfig",
    "GraphRAGEngine",
    "HF_QWEN25_BGE_M3",
    "HF_QWEN3_BGE_M3",
    "HashingEmbeddingModel",
    "HeuristicLLM",
    "HuggingFaceModelProfile",
    "InMemoryArtifactStore",
    "InMemoryGraphStore",
    "InMemoryVectorStore",
    "ModelRoleConfig",
    "SentenceTransformerEmbeddingModel",
    "TransformersCausalLLM",
    "TransformersSequenceReranker",
]

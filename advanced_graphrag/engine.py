"""GraphRAG facade."""

from __future__ import annotations

from advanced_graphrag.config import GraphRAGConfig
from advanced_graphrag.domain import Answer, RetrievalBundle
from advanced_graphrag.generation.generator import AnswerGenerator
from advanced_graphrag.ingestion.extraction import GraphExtractor
from advanced_graphrag.ingestion.pipeline import IngestionPipeline
from advanced_graphrag.models.base import EmbeddingModel, LLMClient
from advanced_graphrag.retrieval.retriever import MultiPathRetriever
from advanced_graphrag.stores.base import ArtifactStore, GraphStore, VectorStore


class GraphRAGEngine:
    """Single production facade for ingest, retrieve, and answer."""

    def __init__(
        self,
        *,
        config: GraphRAGConfig,
        graph_store: GraphStore,
        vector_store: VectorStore,
        artifact_store: ArtifactStore,
        embedder: EmbeddingModel,
        llm: LLMClient,
        extractor: GraphExtractor | None = None,
    ) -> None:
        self.config = config
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.artifact_store = artifact_store
        self.embedder = embedder
        self.llm = llm
        self.ingestion = IngestionPipeline(
            config=config,
            graph_store=graph_store,
            vector_store=vector_store,
            artifact_store=artifact_store,
            embedder=embedder,
            llm=llm,
            extractor=extractor,
        )
        self.retriever = MultiPathRetriever(
            config=config,
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
        )
        self.generator = AnswerGenerator(llm)

    def ingest_text(self, text: str, *, document_id: str | None = None, metadata: dict | None = None) -> str:
        return self.ingestion.ingest(text, document_id=document_id, metadata=metadata)

    def retrieve(self, query: str) -> RetrievalBundle:
        return self.retriever.retrieve(query)

    def answer(self, query: str) -> Answer:
        return self.generator.generate(self.retrieve(query))


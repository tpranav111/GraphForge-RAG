"""Production-shaped ingestion pipeline."""

from __future__ import annotations

from advanced_graphrag.config import GraphRAGConfig
from advanced_graphrag.domain import Document
from advanced_graphrag.ingestion.chunking import StructuralChunker
from advanced_graphrag.ingestion.communities import CommunitySummarizer, ConnectedComponentCommunities
from advanced_graphrag.ingestion.extraction import GraphExtractor, RuleBasedExtractor
from advanced_graphrag.ingestion.resolution import ExactTypeNameResolver
from advanced_graphrag.models.base import EmbeddingModel, LLMClient
from advanced_graphrag.stores.base import ArtifactStore, GraphStore, VectorStore
from advanced_graphrag.utils import stable_id


class IngestionPipeline:
    """Fixed, resumable ingestion flow.

    This follows the operational discipline of AWS/FalkorDB and the lexical
    graph idea in Neo4j/FalkorDB: provenance is mandatory, extraction is a
    replaceable strategy, and all derived artifacts are indexable.
    """

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
        self.chunker = StructuralChunker(config.chunk_size, config.chunk_overlap)
        self.extractor = extractor or RuleBasedExtractor()
        self.resolver = ExactTypeNameResolver()
        self.community_detector = ConnectedComponentCommunities()
        self.community_summarizer = CommunitySummarizer(llm)

    def ingest(self, text: str, *, document_id: str | None = None, metadata: dict | None = None) -> str:
        document_id = document_id or stable_id("doc", text)
        document = Document(id=document_id, text=text, metadata=dict(metadata or {}))
        self.graph_store.upsert_document(document)
        self.artifact_store.put_json(
            f"documents/{document_id}",
            {"document_id": document_id, "metadata": document.metadata},
        )

        chunks = self.chunker.chunk(document)
        self.graph_store.upsert_chunks(chunks)

        extraction = self.extractor.extract(chunks)
        extraction = self.resolver.resolve(extraction.entities, extraction.relationships)
        relationships = tuple(
            rel for rel in extraction.relationships if rel.confidence >= self.config.min_relationship_confidence
        )
        self.graph_store.upsert_entities(extraction.entities)
        self.graph_store.upsert_relationships(relationships)

        if self.config.enable_community_summaries:
            communities = self.community_detector.create(extraction.entities, relationships)
            self.graph_store.upsert_communities(communities)
            summaries = self.community_summarizer.summarize(
                communities,
                {entity.id: entity for entity in extraction.entities},
                relationships,
            )
            self.graph_store.upsert_community_summaries(summaries)

        self._index_chunks(chunks)
        self._index_entities(extraction.entities)
        self._index_relationships(relationships)
        self._index_community_summaries()
        return document_id

    def _index_chunks(self, chunks) -> None:
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.vector_store.upsert(
            "chunks",
            [
                (
                    chunk.id,
                    chunk.text,
                    vector,
                    {"document_id": chunk.document_id, "chunk_id": chunk.id, "ordinal": chunk.ordinal},
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )

    def _index_entities(self, entities) -> None:
        texts = [f"{entity.name}. {entity.type}. {entity.description}" for entity in entities]
        vectors = self.embedder.embed(texts)
        self.vector_store.upsert(
            "entities",
            [(entity.id, text, vector, {"entity_id": entity.id, "name": entity.name}) for entity, text, vector in zip(entities, texts, vectors)],
        )

    def _index_relationships(self, relationships) -> None:
        texts = [f"{rel.type}: {rel.fact}" for rel in relationships]
        vectors = self.embedder.embed(texts)
        self.vector_store.upsert(
            "relationships",
            [
                (
                    rel.id,
                    text,
                    vector,
                    {
                        "relationship_id": rel.id,
                        "source_entity_id": rel.source_entity_id,
                        "target_entity_id": rel.target_entity_id,
                    },
                )
                for rel, text, vector in zip(relationships, texts, vectors)
            ],
        )

    def _index_community_summaries(self) -> None:
        summaries = self.graph_store.community_summaries()
        vectors = self.embedder.embed([summary.text for summary in summaries])
        self.vector_store.upsert(
            "community_summaries",
            [
                (
                    summary.id,
                    summary.text,
                    vector,
                    {"community_id": summary.community_id, "entity_ids": list(summary.entity_ids)},
                )
                for summary, vector in zip(summaries, vectors)
            ],
        )


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from advanced_graphrag import (
    GraphRAGConfig,
    GraphRAGEngine,
    HashingEmbeddingModel,
    HeuristicLLM,
    InMemoryArtifactStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)


def main() -> None:
    engine = GraphRAGEngine(
        config=GraphRAGConfig(chunk_size=90, chunk_overlap=10),
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
        artifact_store=InMemoryArtifactStore(),
        embedder=HashingEmbeddingModel(dimension=128),
        llm=HeuristicLLM(),
    )

    engine.ingest_text(
        """
        Microsoft GraphRAG builds entity graphs and community summaries for global
        sensemaking. HippoRAG uses OpenIE, entity and fact embeddings, and
        Personalized PageRank over a graph memory. FalkorDB GraphRAG SDK uses
        multi-path retrieval over vector search, full-text search, graph traversal,
        and optional text-to-Cypher.
        """,
        document_id="graphrag_notes",
        metadata={"source": "local_quickstart"},
    )

    answer = engine.answer("How does HippoRAG retrieve evidence?")
    print(answer.answer)
    print("confidence:", answer.confidence)
    print("citations:", answer.citations)


if __name__ == "__main__":
    main()

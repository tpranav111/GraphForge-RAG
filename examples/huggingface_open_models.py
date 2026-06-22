import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from advanced_graphrag import (
    DEFAULT_HF_MODEL_PROFILE,
    FileSystemArtifactStore,
    GraphRAGConfig,
    GraphRAGEngine,
    InMemoryGraphStore,
    InMemoryVectorStore,
    SentenceTransformerEmbeddingModel,
    TransformersCausalLLM,
)


def build_engine() -> GraphRAGEngine:
    """Build GraphRAG with open HuggingFace Hub models loaded in-process."""

    profile = DEFAULT_HF_MODEL_PROFILE
    llm = TransformersCausalLLM(
        profile.generation_repo_id,
        max_new_tokens=1024,
        chat_template_kwargs={"enable_thinking": False},
    )
    embedder = SentenceTransformerEmbeddingModel(
        profile.embedding_repo_id,
        dimension=profile.embedding_dimension,
    )

    return GraphRAGEngine(
        config=GraphRAGConfig(max_context_tokens=8000),
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
        artifact_store=FileSystemArtifactStore(".graphrag_artifacts"),
        embedder=embedder,
        llm=llm,
    )


def main() -> None:
    engine = build_engine()
    engine.ingest_text(
        """
        Microsoft GraphRAG builds entity graphs and community summaries for
        global sensemaking. HippoRAG uses OpenIE, fact embeddings, and
        Personalized PageRank over graph memory. BGE-M3 is a multilingual
        embedding model commonly used for dense retrieval.
        """,
        document_id="hf_open_model_notes",
    )
    answer = engine.answer("How do graph walks and embeddings work together in GraphRAG?")
    print(answer.answer)
    print(answer.citations)


if __name__ == "__main__":
    main()

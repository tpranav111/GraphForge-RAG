import unittest
import tempfile

from advanced_graphrag import (
    FileSystemArtifactStore,
    GraphRAGConfig,
    GraphRAGEngine,
    HashingEmbeddingModel,
    HeuristicLLM,
    InMemoryArtifactStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)


class GraphRAGPipelineTest(unittest.TestCase):
    def make_engine(self) -> GraphRAGEngine:
        return GraphRAGEngine(
            config=GraphRAGConfig(chunk_size=80, chunk_overlap=8),
            graph_store=InMemoryGraphStore(),
            vector_store=InMemoryVectorStore(),
            artifact_store=InMemoryArtifactStore(),
            embedder=HashingEmbeddingModel(dimension=128),
            llm=HeuristicLLM(),
        )

    def test_ingest_retrieve_answer(self) -> None:
        engine = self.make_engine()
        engine.ingest_text(
            "Alice Johnson works at Acme Corp in London. Acme Corp builds graph retrieval systems.",
            document_id="doc1",
        )

        retrieval = engine.retrieve("Where does Alice Johnson work?")
        self.assertGreater(len(retrieval.evidence), 0)
        self.assertIn("Alice", retrieval.context)

        answer = engine.answer("Where does Alice Johnson work?")
        self.assertGreater(answer.confidence, 0.0)
        self.assertGreater(len(answer.citations), 0)

    def test_symbolic_entity_query(self) -> None:
        engine = self.make_engine()
        engine.ingest_text("Neo4j supports text to Cypher retrieval for graph question answering.", document_id="doc2")
        retrieval = engine.retrieve("cypher: entity:Neo4j")
        self.assertTrue(any(item.kind == "symbolic" for item in retrieval.evidence))

    def test_citations_are_deduped(self) -> None:
        engine = self.make_engine()
        engine.ingest_text("HippoRAG uses graph memory. HippoRAG uses graph memory.", document_id="doc3")
        answer = engine.answer("What uses graph memory?")
        pairs = {(item["chunk_id"], item["document_id"]) for item in answer.citations}
        self.assertEqual(len(pairs), len(answer.citations))

    def test_filesystem_artifact_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemArtifactStore(tmp)
            store.put_json("runs/test", {"ok": True, "count": 2})
            self.assertEqual(store.get_json("runs/test"), {"ok": True, "count": 2})


if __name__ == "__main__":
    unittest.main()

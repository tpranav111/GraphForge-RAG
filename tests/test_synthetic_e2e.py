import unittest

from advanced_graphrag import (
    GraphRAGConfig,
    GraphRAGEngine,
    HashingEmbeddingModel,
    HeuristicLLM,
    InMemoryArtifactStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from advanced_graphrag.domain import QueryIntent


class SyntheticGraphRAGE2ETest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph_store = InMemoryGraphStore()
        self.vector_store = InMemoryVectorStore()
        self.artifact_store = InMemoryArtifactStore()
        self.engine = GraphRAGEngine(
            config=GraphRAGConfig(
                chunk_size=900,
                chunk_overlap=80,
                max_chunks=8,
                max_entities=16,
                max_relationships=24,
                max_communities=8,
            ),
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            artifact_store=self.artifact_store,
            embedder=HashingEmbeddingModel(dimension=256),
            llm=HeuristicLLM(),
        )

    def test_full_pipeline_with_synthetic_robotics_corpus(self) -> None:
        corpus = {
            "atlas_ops": (
                "Atlas Robotics builds sensor fusion systems for Harbor Drones. "
                "Harbor Drones integrates Raven Battery packs for long bridge flights."
            ),
            "raven_supply": (
                "Raven Battery supplies modular cells to Atlas Robotics. "
                "Marina City deployed Harbor Drones for bridge inspection."
            ),
            "sentinel_monitoring": (
                "Sentinel Analytics monitors failures from Harbor Drones and reports "
                "risk alerts to Atlas Robotics."
            ),
        }

        for document_id, text in corpus.items():
            self.engine.ingest_text(text, document_id=document_id, metadata={"synthetic": True})

        self.assertEqual(set(self.graph_store.documents_by_id), set(corpus))
        self.assertGreaterEqual(len(self.graph_store.chunks()), 3)
        self.assertGreaterEqual(len(self.graph_store.entities()), 5)
        self.assertGreaterEqual(len(self.graph_store.relationships()), 4)
        self.assertGreaterEqual(len(self.graph_store.community_summaries()), 1)
        self.assertIsNotNone(self.artifact_store.get_json("documents/atlas_ops"))
        self.assertIn("chunks", self.vector_store._items)
        self.assertIn("entities", self.vector_store._items)
        self.assertIn("relationships", self.vector_store._items)
        self.assertIn("community_summaries", self.vector_store._items)

        entity_names = {entity.name for entity in self.graph_store.entities()}
        self.assertIn("Atlas Robotics", entity_names)
        self.assertIn("Harbor Drones", entity_names)
        self.assertIn("Raven Battery", entity_names)

        fact_retrieval = self.engine.retrieve("Who supplies modular cells to Atlas Robotics?")
        self.assertEqual(fact_retrieval.query_plan.intent, QueryIntent.FACT)
        self.assertTrue(fact_retrieval.evidence)
        self.assertIn("Raven Battery", fact_retrieval.context)
        self.assertTrue(any(item.kind == "chunk" for item in fact_retrieval.evidence))
        self.assertTrue(any(item.kind == "relationship" for item in fact_retrieval.evidence))
        self.assertTrue(any(item.kind == "graph_walk" for item in fact_retrieval.evidence))

        multi_hop = self.engine.retrieve("How is Atlas Robotics connected to Raven Battery?")
        self.assertEqual(multi_hop.query_plan.intent, QueryIntent.MULTI_HOP)
        self.assertTrue(any(item.kind == "graph_walk" for item in multi_hop.evidence))
        self.assertIn("Atlas Robotics", multi_hop.context)

        global_retrieval = self.engine.retrieve("Give an overall overview of the robotics deployment themes")
        self.assertEqual(global_retrieval.query_plan.intent, QueryIntent.GLOBAL)
        self.assertTrue(any(item.kind == "community" for item in global_retrieval.evidence))
        self.assertIn("[COMMUNITY]", global_retrieval.context)

        symbolic = self.engine.retrieve("cypher: entity:Harbor Drones")
        self.assertEqual(symbolic.query_plan.intent, QueryIntent.SYMBOLIC)
        self.assertTrue(any(item.kind == "symbolic" for item in symbolic.evidence))
        self.assertIn("Harbor Drones", symbolic.context)

        answer = self.engine.answer("Who supplies modular cells to Atlas Robotics?")
        self.assertGreater(answer.confidence, 0.0)
        self.assertTrue(answer.answer)
        self.assertTrue(answer.citations)
        self.assertTrue(answer.graph_evidence)


if __name__ == "__main__":
    unittest.main()

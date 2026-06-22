import unittest

from advanced_graphrag.domain import Chunk
from advanced_graphrag.ingestion.extraction import LLMJsonGraphExtractor, RuleBasedExtractor
from advanced_graphrag.models.base import ChatMessage


class _StaticLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages, *, temperature: float = 0.0) -> str:
        self.calls.append(list(messages))
        return self.response


class LLMJsonGraphExtractorTest(unittest.TestCase):
    def test_extracts_graph_from_wrapped_json(self) -> None:
        llm = _StaticLLM(
            """
            Here is the graph:
            ```json
            {
              "entities": [
                {"name": "Atlas Robotics", "type": "Organization", "description": "Drone integrator", "aliases": [], "properties": {}},
                {"name": "Raven Battery", "type": "Supplier", "description": "Battery supplier", "aliases": ["Raven"], "properties": {}}
              ],
              "relationships": [
                {"source": "Raven Battery", "target": "Atlas Robotics", "type": "SUPPLIES", "fact": "Raven Battery supplies Atlas Robotics.", "confidence": 0.91, "properties": {}}
              ]
            }
            ```
            """
        )
        chunk = Chunk(id="chunk_1", document_id="doc", text="Raven Battery supplies Atlas Robotics.", ordinal=0)

        result = LLMJsonGraphExtractor(llm).extract([chunk])

        self.assertEqual({entity.name for entity in result.entities}, {"Atlas Robotics", "Raven Battery"})
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(result.relationships[0].type, "SUPPLIES")
        self.assertEqual(result.relationships[0].source_chunk_ids, ("chunk_1",))
        self.assertEqual(llm.calls[0][0].role, "system")

    def test_falls_back_when_json_is_malformed(self) -> None:
        llm = _StaticLLM("not json")
        chunk = Chunk(id="chunk_2", document_id="doc", text="Atlas Robotics uses Harbor Drones.", ordinal=0)

        result = LLMJsonGraphExtractor(llm, fallback_extractor=RuleBasedExtractor()).extract([chunk])

        self.assertTrue(result.entities)
        self.assertTrue(result.relationships)


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import math
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from advanced_graphrag import (
    GraphRAGConfig,
    GraphRAGEngine,
    InMemoryArtifactStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from advanced_graphrag.models.openai_compatible import OpenAICompatibleEmbeddingModel, OpenAICompatibleLLM
from advanced_graphrag.utils import tokenize


class _SyntheticOpenAIHandler(BaseHTTPRequestHandler):
    embedding_dimension = 64

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/v1/chat/completions":
            self._send(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": self._completion(payload),
                            }
                        }
                    ]
                }
            )
            return
        if self.path == "/v1/embeddings":
            inputs = payload.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            self._send(
                {
                    "data": [
                        {"index": idx, "embedding": _hash_embedding(text, self.embedding_dimension)}
                        for idx, text in enumerate(inputs)
                    ]
                }
            )
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _completion(self, payload: dict[str, Any]) -> str:
        text = "\n".join(message.get("content", "") for message in payload.get("messages", []))
        if "community summary" in text.lower():
            return "Hosted HF community summary for synthetic operations evidence."
        if "Return a concise answer" in text:
            return "Hosted HF answer generated from retrieved synthetic GraphRAG context."
        return "Hosted HF extraction/planning response."

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _hash_embedding(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


class HostedModelE2ETest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SyntheticOpenAIHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_pipeline_uses_openai_compatible_hosted_model_boundary(self) -> None:
        engine = GraphRAGEngine(
            config=GraphRAGConfig(chunk_size=500, chunk_overlap=60),
            graph_store=InMemoryGraphStore(),
            vector_store=InMemoryVectorStore(),
            artifact_store=InMemoryArtifactStore(),
            embedder=OpenAICompatibleEmbeddingModel(
                base_url=self.base_url,
                model="synthetic-hf-embedding",
                dimension=_SyntheticOpenAIHandler.embedding_dimension,
            ),
            llm=OpenAICompatibleLLM(
                base_url=self.base_url,
                model="synthetic-hf-inference",
            ),
        )

        engine.ingest_text(
            "Northstar Labs coordinates Relay Grid. Relay Grid depends on Vector Cache. "
            "Vector Cache accelerates Northstar Labs retrieval workflows.",
            document_id="hosted_boundary_doc",
        )

        retrieval = engine.retrieve("How does Northstar Labs use Vector Cache?")
        self.assertTrue(retrieval.evidence)
        self.assertIn("Northstar Labs", retrieval.context)

        answer = engine.answer("How does Northstar Labs use Vector Cache?")
        self.assertIn("Hosted HF answer", answer.answer)
        self.assertTrue(answer.citations)
        self.assertGreater(answer.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

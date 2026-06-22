"""Answer generation."""

from __future__ import annotations

import re

from advanced_graphrag.domain import Answer, RetrievalBundle
from advanced_graphrag.models.base import ChatMessage, LLMClient


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, retrieval: RetrievalBundle) -> Answer:
        prompt = (
            "You are a GraphRAG answer generator.\n"
            "Use only the retrieved context. Cite source IDs when possible. "
            "If the context is insufficient, say so.\n\n"
            f"{retrieval.context}\n\n"
            "Return a concise answer."
        )
        text = self.llm.complete(
            [
                ChatMessage("system", "You answer from provided GraphRAG context only."),
                ChatMessage("user", prompt),
            ],
            temperature=0.0,
        )
        citations = self._citations(retrieval)
        graph_evidence = self._graph_evidence(retrieval)
        confidence = self._confidence(retrieval)
        return Answer(
            answer=text,
            citations=tuple(citations),
            graph_evidence=tuple(graph_evidence),
            confidence=confidence,
            retrieval=retrieval,
        )

    def _citations(self, retrieval: RetrievalBundle) -> list[dict[str, str]]:
        citations = []
        seen: set[tuple[str, str]] = set()
        for evidence in retrieval.evidence:
            if evidence.kind != "chunk":
                continue
            chunk_id = evidence.source_ids[0] if evidence.source_ids else evidence.id
            document_id = evidence.source_ids[1] if len(evidence.source_ids) > 1 else ""
            key = (chunk_id, document_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"chunk_id": chunk_id, "document_id": document_id, "reason": "retrieved source passage"})
        return citations[:8]

    def _graph_evidence(self, retrieval: RetrievalBundle) -> list[dict[str, str]]:
        rows = []
        pattern = re.compile(r"(.+?) -\[(.+?)\]-> (.+?): (.+)")
        for evidence in retrieval.evidence:
            if evidence.kind != "relationship":
                continue
            match = pattern.match(evidence.text)
            if match:
                rows.append(
                    {
                        "source": match.group(1),
                        "relation": match.group(2),
                        "target": match.group(3),
                        "fact": match.group(4),
                    }
                )
        return rows[:12]

    def _confidence(self, retrieval: RetrievalBundle) -> float:
        if not retrieval.evidence:
            return 0.0
        top = max(item.score for item in retrieval.evidence)
        support = min(len(retrieval.evidence) / 12.0, 1.0)
        return round(min(0.98, 0.35 + 0.4 * support + 0.25 * min(top, 1.0)), 3)

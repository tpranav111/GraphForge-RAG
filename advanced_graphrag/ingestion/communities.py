"""Community creation and summaries."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

from advanced_graphrag.domain import Community, CommunitySummary, Entity, Relationship
from advanced_graphrag.models.base import ChatMessage, LLMClient
from advanced_graphrag.utils import stable_id


class ConnectedComponentCommunities:
    """Dependency-light community detector.

    Production systems should swap this for Leiden/Louvain, as in Microsoft
    GraphRAG and NodeRAG. The interface is deliberately the same shape.
    """

    def create(
        self,
        entities: Sequence[Entity],
        relationships: Sequence[Relationship],
    ) -> list[Community]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        entity_ids = {entity.id for entity in entities}
        for rel in relationships:
            if rel.source_entity_id in entity_ids and rel.target_entity_id in entity_ids:
                adjacency[rel.source_entity_id].add(rel.target_entity_id)
                adjacency[rel.target_entity_id].add(rel.source_entity_id)
        seen: set[str] = set()
        communities: list[Community] = []
        for entity_id in sorted(entity_ids):
            if entity_id in seen:
                continue
            component = []
            queue = deque([entity_id])
            seen.add(entity_id)
            while queue:
                current = queue.popleft()
                component.append(current)
                for nxt in adjacency[current]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            communities.append(
                Community(
                    id=stable_id("community", ",".join(sorted(component))),
                    entity_ids=tuple(sorted(component)),
                    level=0,
                )
            )
        return communities


class CommunitySummarizer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def summarize(
        self,
        communities: Sequence[Community],
        entities_by_id: dict[str, Entity],
        relationships: Sequence[Relationship],
    ) -> list[CommunitySummary]:
        summaries: list[CommunitySummary] = []
        rels_by_entity: dict[str, list[Relationship]] = defaultdict(list)
        for rel in relationships:
            rels_by_entity[rel.source_entity_id].append(rel)
            rels_by_entity[rel.target_entity_id].append(rel)
        for community in communities:
            names = [entities_by_id[eid].name for eid in community.entity_ids if eid in entities_by_id]
            facts = []
            for eid in community.entity_ids:
                facts.extend(rel.fact for rel in rels_by_entity.get(eid, [])[:4])
            prompt = (
                "Write a compact community summary for GraphRAG retrieval.\n"
                f"Entities: {', '.join(names)}\n"
                f"Facts: {' '.join(facts[:16])}"
            )
            text = self.llm.complete([ChatMessage("user", prompt)], temperature=0.0)
            summaries.append(
                CommunitySummary(
                    id=stable_id("community_summary", community.id, text),
                    community_id=community.id,
                    text=text,
                    entity_ids=community.entity_ids,
                )
            )
        return summaries


"""Multi-path retrieval."""

from __future__ import annotations

from advanced_graphrag.config import GraphRAGConfig
from advanced_graphrag.domain import Evidence, QueryIntent, RetrievalBundle
from advanced_graphrag.models.base import EmbeddingModel
from advanced_graphrag.retrieval.context import ContextAssembler
from advanced_graphrag.retrieval.fusion import ReciprocalRankFusion
from advanced_graphrag.retrieval.graph_walk import PersonalizedPageRank
from advanced_graphrag.retrieval.query_planner import QueryPlanner
from advanced_graphrag.stores.base import GraphStore, VectorStore


class MultiPathRetriever:
    """Production-style hybrid retriever.

    It combines LightRAG local/global/hybrid behavior, FalkorDB multi-path
    retrieval, HippoRAG/FastGraphRAG PPR, Microsoft community summaries, and
    Neo4j/KAG symbolic hooks.
    """

    def __init__(
        self,
        *,
        config: GraphRAGConfig,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
    ) -> None:
        self.config = config
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.planner = QueryPlanner()
        self.fusion = ReciprocalRankFusion()
        self.context_assembler = ContextAssembler()
        self.ppr = PersonalizedPageRank(
            graph_store,
            alpha=config.ppr_alpha,
            max_iter=config.ppr_max_iter,
            tolerance=config.ppr_tolerance,
        )

    def retrieve(self, query: str) -> RetrievalBundle:
        plan = self.planner.plan(query)
        query_vector = self.embedder.embed([query])[0]
        weights = self.config.retrieval_weights

        ranked_lists: list[tuple[float, tuple[Evidence, ...]]] = []
        if plan.intent != QueryIntent.GLOBAL:
            ranked_lists.append((weights.chunk_vector, self._chunk_vector(query_vector)))
            ranked_lists.append((weights.chunk_keyword, self._chunk_keyword(plan.keywords)))
            ranked_lists.append((weights.entity_vector, self._entity_vector(query_vector)))
            ranked_lists.append((weights.relationship_vector, self._relationship_vector(query_vector)))
            ranked_lists.append((weights.graph_walk, self._graph_walk(query_vector)))

        if plan.intent in {QueryIntent.GLOBAL, QueryIntent.EXPLORATORY, QueryIntent.COMPARISON}:
            ranked_lists.append((weights.community, self._community_vector(query_vector)))

        if self.config.enable_symbolic_queries and plan.symbolic_query:
            ranked_lists.append((weights.symbolic, self._symbolic(plan.symbolic_query)))

        fused = self.fusion.fuse(ranked_lists, limit=self.config.max_chunks + self.config.max_relationships)
        context = self.context_assembler.assemble(plan, fused)
        return RetrievalBundle(query_plan=plan, evidence=fused, context=context)

    def _chunk_vector(self, query_vector: list[float]) -> tuple[Evidence, ...]:
        rows = self.vector_store.search("chunks", query_vector, top_k=self.config.max_chunks)
        return tuple(
            Evidence(
                id=f"chunk:{item_id}",
                kind="chunk",
                text=metadata["text"],
                score=score,
                source_ids=(metadata.get("chunk_id", item_id), metadata.get("document_id", "")),
                metadata=metadata,
            )
            for item_id, score, metadata in rows
        )

    def _chunk_keyword(self, kws: tuple[str, ...]) -> tuple[Evidence, ...]:
        if not kws:
            return ()
        results = []
        for chunk in self.graph_store.chunks():
            text_l = chunk.text.lower()
            hits = sum(1 for kw in kws if kw.lower() in text_l)
            if hits:
                results.append(
                    Evidence(
                        id=f"chunk_keyword:{chunk.id}",
                        kind="chunk",
                        text=chunk.text,
                        score=hits / len(kws),
                        source_ids=(chunk.id, chunk.document_id),
                    )
                )
        results.sort(key=lambda item: item.score, reverse=True)
        return tuple(results[: self.config.max_chunks])

    def _entity_vector(self, query_vector: list[float]) -> tuple[Evidence, ...]:
        rows = self.vector_store.search("entities", query_vector, top_k=self.config.max_entities)
        output = []
        for item_id, score, metadata in rows:
            entity = self.graph_store.get_entity(metadata.get("entity_id", item_id))
            if entity:
                output.append(
                    Evidence(
                        id=f"entity:{entity.id}",
                        kind="entity",
                        text=f"{entity.name} ({entity.type}): {entity.description}",
                        score=score,
                        source_ids=(entity.id,),
                        metadata={"entity_id": entity.id},
                    )
                )
        return tuple(output)

    def _relationship_vector(self, query_vector: list[float]) -> tuple[Evidence, ...]:
        rows = self.vector_store.search("relationships", query_vector, top_k=self.config.max_relationships)
        out = []
        for item_id, score, metadata in rows:
            rel = next((r for r in self.graph_store.relationships() if r.id == metadata.get("relationship_id", item_id)), None)
            if rel:
                src = self.graph_store.get_entity(rel.source_entity_id)
                tgt = self.graph_store.get_entity(rel.target_entity_id)
                out.append(
                    Evidence(
                        id=f"relationship:{rel.id}",
                        kind="relationship",
                        text=f"{src.name if src else rel.source_entity_id} -[{rel.type}]-> {tgt.name if tgt else rel.target_entity_id}: {rel.fact}",
                        score=score * rel.confidence,
                        source_ids=rel.source_chunk_ids,
                        metadata={"relationship_id": rel.id},
                    )
                )
        return tuple(out)

    def _graph_walk(self, query_vector: list[float]) -> tuple[Evidence, ...]:
        seed_rows = self.vector_store.search("entities", query_vector, top_k=8)
        seeds = {metadata.get("entity_id", item_id): max(score, 0.0) for item_id, score, metadata in seed_rows}
        ranked = self.ppr.rank(seeds)[: self.config.max_entities]
        out = []
        for entity_id, score in ranked:
            entity = self.graph_store.get_entity(entity_id)
            if not entity:
                continue
            neighbor_facts = [rel.fact for _, rel in self.graph_store.neighbors(entity_id)[:4]]
            out.append(
                Evidence(
                    id=f"graph_walk:{entity_id}",
                    kind="graph_walk",
                    text=f"{entity.name}: {entity.description} {' '.join(neighbor_facts)}",
                    score=score,
                    source_ids=(entity_id,),
                    metadata={"entity_id": entity_id},
                )
            )
        return tuple(out)

    def _community_vector(self, query_vector: list[float]) -> tuple[Evidence, ...]:
        rows = self.vector_store.search("community_summaries", query_vector, top_k=self.config.max_communities)
        return tuple(
            Evidence(
                id=f"community:{item_id}",
                kind="community",
                text=metadata["text"],
                score=score,
                source_ids=(metadata.get("community_id", item_id),),
                metadata=metadata,
            )
            for item_id, score, metadata in rows
        )

    def _symbolic(self, symbolic_query: str) -> tuple[Evidence, ...]:
        rows = self.graph_store.symbolic_read(symbolic_query)
        return tuple(
            Evidence(
                id=f"symbolic:{idx}",
                kind="symbolic",
                text=str(row),
                score=1.0,
                source_ids=tuple(str(v) for v in row.values() if isinstance(v, str)),
                metadata=row,
            )
            for idx, row in enumerate(rows)
        )


"""Context assembly."""

from __future__ import annotations

from advanced_graphrag.domain import Evidence, QueryPlan


class ContextAssembler:
    """Typed context builder.

    This follows FalkorDB/AWS style structured context assembly: direct graph
    answers, entities, facts, graph paths, summaries, and source passages are
    not flattened until the final prompt.
    """

    def assemble(self, plan: QueryPlan, evidence: tuple[Evidence, ...], *, max_chars: int = 24000) -> str:
        sections: dict[str, list[Evidence]] = {}
        for item in evidence:
            sections.setdefault(item.kind, []).append(item)

        parts = [
            f"Question: {plan.query}",
            f"Intent: {plan.intent.value}",
            f"Keywords: {', '.join(plan.keywords)}",
        ]
        for kind in ("symbolic", "entity", "relationship", "graph_walk", "community", "chunk"):
            values = sections.get(kind, [])
            if not values:
                continue
            parts.append(f"\n[{kind.upper()}]")
            for idx, item in enumerate(values, start=1):
                source = ", ".join(item.source_ids)
                parts.append(f"{idx}. score={item.score:.4f} sources={source}\n{item.text}")
        context = "\n".join(parts)
        return context[:max_chars]


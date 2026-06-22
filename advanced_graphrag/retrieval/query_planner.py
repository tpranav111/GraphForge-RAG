"""Query analysis and planning."""

from __future__ import annotations

import re

from advanced_graphrag.domain import QueryIntent, QueryPlan
from advanced_graphrag.utils import keywords


class QueryPlanner:
    """Rule-first planner with LLM extension point.

    KAG shows why planning matters. This default planner is deterministic and
    can be replaced by an LLM planner that emits the same `QueryPlan`.
    """

    _ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4}\b")

    def plan(self, query: str) -> QueryPlan:
        lowered = query.lower()
        if any(term in lowered for term in ("summarize", "overview", "themes", "overall")):
            intent = QueryIntent.GLOBAL
        elif any(term in lowered for term in ("compare", "difference", "versus", " vs ")):
            intent = QueryIntent.COMPARISON
        elif any(term in lowered for term in ("list", "all ", "which")):
            intent = QueryIntent.ENUMERATION
        elif any(term in lowered for term in ("path", "connect", "relationship", "between")):
            intent = QueryIntent.MULTI_HOP
        elif any(term in lowered for term in ("count", "how many", "cypher:", "query:")):
            intent = QueryIntent.SYMBOLIC
        else:
            intent = QueryIntent.FACT

        entities = tuple(dict.fromkeys(m.group(0) for m in self._ENTITY_RE.finditer(query)))
        kws = tuple(keywords(query, limit=12))
        subqueries = self._split_subqueries(query, intent)
        symbolic = None
        if lowered.startswith("cypher:") or lowered.startswith("query:"):
            symbolic = query.split(":", 1)[1].strip()
        elif intent == QueryIntent.SYMBOLIC and entities:
            symbolic = f"entity:{entities[0]}"
        return QueryPlan(query=query, intent=intent, keywords=kws, entities=entities, subqueries=subqueries, symbolic_query=symbolic)

    def _split_subqueries(self, query: str, intent: QueryIntent) -> tuple[str, ...]:
        if intent not in {QueryIntent.COMPARISON, QueryIntent.MULTI_HOP}:
            return ()
        parts = [part.strip(" ?") for part in re.split(r"\band\b|\bversus\b|\bvs\b", query, flags=re.I)]
        return tuple(part for part in parts if len(part.split()) >= 3)


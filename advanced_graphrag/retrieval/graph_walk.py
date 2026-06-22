"""Graph walk algorithms."""

from __future__ import annotations

from collections import defaultdict

from advanced_graphrag.stores.base import GraphStore


class PersonalizedPageRank:
    """Fixed-restart Personalized PageRank over the entity graph.

    This intentionally uses a stable restart vector, unlike some simplified
    graph-diffusion snippets. HippoRAG and FastGraphRAG both rely on this
    family of retrieval behavior.
    """

    def __init__(self, graph_store: GraphStore, *, alpha: float, max_iter: int, tolerance: float) -> None:
        self.graph_store = graph_store
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance

    def rank(self, seeds: dict[str, float]) -> list[tuple[str, float]]:
        if not seeds:
            return []
        total = sum(max(v, 0.0) for v in seeds.values()) or 1.0
        restart = {node: max(score, 0.0) / total for node, score in seeds.items()}
        scores = dict(restart)

        all_nodes = {entity.id for entity in self.graph_store.entities()}
        all_nodes.update(restart)
        for node in all_nodes:
            scores.setdefault(node, 0.0)

        for _ in range(self.max_iter):
            next_scores = defaultdict(float)
            for node in all_nodes:
                next_scores[node] += (1.0 - self.alpha) * restart.get(node, 0.0)
            for node, score in scores.items():
                neighbors = self.graph_store.neighbors(node)
                if not neighbors:
                    next_scores[node] += self.alpha * score
                    continue
                share = self.alpha * score / len(neighbors)
                for neighbor_id, _ in neighbors:
                    next_scores[neighbor_id] += share
            delta = sum(abs(next_scores.get(node, 0.0) - scores.get(node, 0.0)) for node in all_nodes)
            scores = dict(next_scores)
            if delta < self.tolerance:
                break
        return sorted(scores.items(), key=lambda row: row[1], reverse=True)


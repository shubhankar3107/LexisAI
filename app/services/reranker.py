from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.schemas.retrieval import RetrievalResult


class Reranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank retrieval candidates for a query."""
        raise NotImplementedError


class ScoreReranker(Reranker):
    """
    Deterministic baseline reranker.

    Uses the existing vector similarity score and returns
    candidates ordered from highest score to lowest score.
    """

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        if not results:
            return []

        ranked_results = sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )

        return ranked_results[:top_k]
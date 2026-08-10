from __future__ import annotations

from app.services.reranker import Reranker
from app.services.retriever import Retriever
from app.services.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


class RetrievalPipeline:
    """Orchestrates retrieval followed by optional reranking."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
    ):
        self._retriever = retriever
        self._reranker = reranker

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        if query.top_k <= 0:
            return []

        candidates = self._retriever.retrieve(
            query,
        )

        if not candidates:
            return []

        return self._reranker.rerank(
            query=query.query,
            results=candidates,
            top_k=query.top_k,
        )
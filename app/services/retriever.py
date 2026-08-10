from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.retrieval_service import RetrievalService
from app.services.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


class Retriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Retrieve relevant document chunks for a query."""
        raise NotImplementedError


class SemanticRetriever(Retriever):
    """
    Concrete semantic retriever backed by RetrievalService.

    Vector-index selection is delegated to RetrievalService,
    which resolves the ACTIVE index for the configured
    embedding profile and organization.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
    ):
        self._retrieval_service = retrieval_service

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        return self._retrieval_service.retrieve(
            query,
        )
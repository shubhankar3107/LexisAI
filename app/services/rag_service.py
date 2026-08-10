from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.schemas.rag import (
    RAGRequest,
    RAGResponse,
)


class RAGService(ABC):

    @abstractmethod
    def answer(
        self,
        request: RAGRequest,
    ) -> RAGResponse:
        """Generate an answer using retrieved document context."""
        raise NotImplementedError
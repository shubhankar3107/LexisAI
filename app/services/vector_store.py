from abc import ABC, abstractmethod

from app.services.schemas.retrieval import RetrievalResult
from app.services.schemas.vector import VectorRecord


class VectorStore(ABC):

    @abstractmethod
    def upsert(
        self,
        index_id: str,
        records: list[VectorRecord],
    ) -> None:
        """Insert or update vectors for a vector index."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        index_id: str,
        query_vector: list[float],
        organization_id: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Search a vector index for relevant records."""
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        index_id: str,
        chunk_ids: list[str],
    ) -> None:
        """Delete specific vectors from a vector index."""
        raise NotImplementedError

    @abstractmethod
    def delete_by_document_id(
        self,
        index_id: str,
        document_id: str,
    ) -> None:
        """Delete all vectors belonging to a document."""
        raise NotImplementedError
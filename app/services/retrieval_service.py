from __future__ import annotations

import uuid

from app.repositories.vector_index_repository import (
    VectorIndexRepository,
)
from app.services.embedding_service import EmbeddingService
from app.services.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)
from app.services.vector_store import VectorStore


class RetrievalService:
    """Application service for semantic document retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        vector_index_repository: VectorIndexRepository,
        profile_name: str = "legal-general",
    ):
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._vector_index_repository = vector_index_repository
        self._profile_name = profile_name

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        if query.top_k <= 0:
            return []

        vector_index = (
            self._vector_index_repository.get_active(
                organization_id=query.organization_id,
                profile_name=self._profile_name,
            )
        )

        if vector_index is None:
            return []

        query_vector = self._embedding_service.embed_query(
            query.query,
        )

        return self._vector_store.search(
            index_id=str(vector_index.id),
            query_vector=query_vector,
            organization_id=str(
                query.organization_id,
            ),
            top_k=query.top_k,
            document_id=(
                str(query.document_id)
                if query.document_id is not None
                else None
            ),
        )
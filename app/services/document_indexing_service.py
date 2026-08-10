from __future__ import annotations

import uuid

from app.enums.document_status import DocumentStatus
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingStateError,
)
from app.services.schemas.vector import VectorRecord
from app.services.vector_store import VectorStore


class DocumentIndexingService:
    """Application service for indexing processed document chunks."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_chunk_repository: DocumentChunkRepository,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self._document_repository = document_repository
        self._document_chunk_repository = document_chunk_repository
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def index_document(
        self,
        document_id: uuid.UUID,
        organization_id: uuid.UUID,
        index_id: uuid.UUID,
    ) -> None:
        document = self._document_repository.get_by_id(
            document_id,
            organization_id=organization_id,
        )

        if document is None:
            raise DocumentNotFoundError(
                f"Document {document_id} not found",
            )

        if document.status is not DocumentStatus.READY:
            raise DocumentProcessingStateError(
                f"Document {document_id} must be READY "
                f"before indexing, but is "
                f"{document.status.value}",
            )

        chunks = self._document_chunk_repository.list_by_document_id(
            document_id,
        )

        if not chunks:
            return

        texts = [
            chunk.content
            for chunk in chunks
        ]

        # Generate embeddings BEFORE deleting existing vectors.
        #
        # If embedding generation fails, the existing indexed
        # representation remains untouched.
        vectors = self._embedding_service.embed_documents(
            texts,
        )

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding provider returned an unexpected "
                "number of vectors: "
                f"expected {len(chunks)}, got {len(vectors)}",
            )

        records = [
            VectorRecord(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                organization_id=organization_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                vector=vector,
            )
            for chunk, vector in zip(
                chunks,
                vectors,
                strict=True,
            )
        ]

        # Remove the previous representation only after all
        # embeddings have been generated and validated.
        self._vector_store.delete_by_document_id(
            index_id=str(index_id),
            document_id=str(document_id),
        )

        # Persist the current representation.
        self._vector_store.upsert(
            index_id=str(index_id),
            records=records,
        )
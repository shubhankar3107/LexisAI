import uuid

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


class DocumentChunkRepository:

    def __init__(self, session: Session):
        self._session = session

    def add(
        self,
        chunk: DocumentChunk,
    ) -> DocumentChunk:
        self._session.add(chunk)
        return chunk

    def delete_by_document_id(
        self,
        document_id: uuid.UUID,
    ) -> None:
        """Delete all chunks belonging to a document."""
        statement = delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
        )

        self._session.execute(statement)

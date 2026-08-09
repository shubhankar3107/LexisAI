import uuid

from app.enums.document_status import DocumentStatus
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_content import DocumentContent
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.document_chunker import DocumentChunker
from app.services.document_text_extractor import DocumentTextExtractor
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingStateError,
)
from app.storage.file_storage import FileStorage


class DocumentProcessingService:

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_chunk_repository: DocumentChunkRepository,
        file_storage: FileStorage,
        text_extractor: DocumentTextExtractor,
        chunker: DocumentChunker,
        unit_of_work,
    ):
        self._document_repository = document_repository
        self._document_chunk_repository = document_chunk_repository
        self._file_storage = file_storage
        self._text_extractor = text_extractor
        self._chunker = chunker
        self._unit_of_work = unit_of_work

    def process_document(
        self,
        document_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> Document:
        document = self._document_repository.get_by_id(
            document_id,
            organization_id=organization_id,
        )

        if document is None:
            raise DocumentNotFoundError(
                f"Document {document_id} not found",
            )

        if document.status is DocumentStatus.PROCESSING:
            raise DocumentProcessingStateError(
                f"Document {document_id} is already processing",
            )

        if document.status is DocumentStatus.READY:
            raise DocumentProcessingStateError(
                f"Document {document_id} is already processed",
            )

        if document.status not in {
            DocumentStatus.UPLOADED,
            DocumentStatus.FAILED,
        }:
            raise DocumentProcessingStateError(
                f"Document {document_id} cannot be processed "
                f"from status {document.status}",
            )

        document.status = DocumentStatus.PROCESSING

        self._unit_of_work.commit()

        try:
            file = self._file_storage.read(
                document.asset.storage_path,
            )

            try:
                content, page_count = self._text_extractor.extract(
                    file,
                )
            finally:
                file.close()

            if document.content is not None:
                document.content.content = content
                document.content.page_count = page_count
            else:
                document.content = DocumentContent(
                    document_id=document.id,
                    content=content,
                    page_count=page_count,
                )

            self._document_chunk_repository.delete_by_document_id(
                document.id,
            )

            chunks = self._chunker.chunk(content)

            for chunk in chunks:
                self._document_chunk_repository.add(
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                    )
                )

            document.status = DocumentStatus.READY

            self._unit_of_work.commit()

        except Exception:
            self._unit_of_work.rollback()

            document.status = DocumentStatus.FAILED

            self._unit_of_work.commit()

            raise

        return document
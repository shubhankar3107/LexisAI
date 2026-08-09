import uuid
from typing import BinaryIO

from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.services.exceptions import DocumentNotFoundError, DocumentFileNotFoundError
from app.storage.file_storage import FileStorage


class DocumentService:

    def __init__(
        self,
        document_repository: DocumentRepository,
        unit_of_work,
        file_storage: FileStorage,
    ):
        self._document_repository = document_repository
        self._unit_of_work = unit_of_work
        self._file_storage = file_storage

    def get_document(
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

        return document

    def delete_document(
        self,
        document_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> None:
        document = self._document_repository.get_by_id(
            document_id,
            organization_id=organization_id,
        )

        if document is None:
            raise DocumentNotFoundError(
                f"Document {document_id} not found",
            )

        self._document_repository.delete(document)

        self._unit_of_work.commit()

    def list_documents(
        self,
        organization_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        documents = self._document_repository.list(
            organization_id=organization_id,
            limit=limit,
            offset=offset,
        )

        total = self._document_repository.count(
            organization_id=organization_id,
        )

        return documents, total

    def download_document(
        self,
        document_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> tuple[Document, BinaryIO]:
        document = self._document_repository.get_by_id(
            document_id,
            organization_id=organization_id,
        )

        if document is None:
            raise DocumentNotFoundError(
                f"Document {document_id} not found",
            )

        if document.asset is None:
            raise DocumentFileNotFoundError(
                f"Document {document_id} has no file asset",
            )

        try:
            file = self._file_storage.read(
                document.asset.storage_path,
            )
        except FileNotFoundError as exc:
            raise DocumentFileNotFoundError(
                f"File for document {document_id} not found",
            ) from exc

        return document, file
import uuid
from io import BytesIO

import pytest

from app.enums.document_status import DocumentStatus
from app.models.document import Document
from app.services.document_processing_service import DocumentProcessingService
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingStateError,
)
from app.services.document_chunker import DocumentChunker
from app.services.document_text_extractor import DocumentTextExtractor
from app.storage.file_storage import FileStorage
from app.models.document_asset import DocumentAsset


class FakeDocumentRepository:
    def __init__(self, document=None):
        self.document = document

    def get_by_id(
        self,
        document_id,
        organization_id=None,
    ):
        if self.document is None:
            return None

        if self.document.id != document_id:
            return None

        if (
            organization_id is not None
            and self.document.organization_id != organization_id
        ):
            return None

        return self.document


class FakeDocumentChunkRepository:
    def __init__(self):
        self.deleted_document_ids = []
        self.chunks = []

    def delete_by_document_id(self, document_id):
        self.deleted_document_ids.append(document_id)

    def add(self, chunk):
        self.chunks.append(chunk)
        return chunk


class FakeFileStorage(FileStorage):
    def __init__(self):
        self.files = {}

    def save(self, file, path):
        pass

    def delete(self, path):
        pass

    def read(self, path):
        return self.files[path]


class FakeTextExtractor(DocumentTextExtractor):
    def __init__(self, content="Extracted text", page_count=2):
        self.content = content
        self.page_count = page_count

    def extract(self, file):
        return self.content, self.page_count


class FakeChunker(DocumentChunker):
    def chunk(self, content):
        return [
            type(
                "FakeChunk",
                (),
                {
                    "chunk_index": 0,
                    "content": content,
                },
            )()
        ]


class FakeUnitOfWork:
    def __init__(self):
        self.commit_called = 0
        self.rollback_called = 0

    def commit(self):
        self.commit_called += 1

    def rollback(self):
        self.rollback_called += 1


def build_service(document=None):
    repository = FakeDocumentRepository(document)
    chunk_repository = FakeDocumentChunkRepository()
    file_storage = FakeFileStorage()
    text_extractor = FakeTextExtractor()
    chunker = FakeChunker()
    unit_of_work = FakeUnitOfWork()

    service = DocumentProcessingService(
        document_repository=repository,
        document_chunk_repository=chunk_repository,
        file_storage=file_storage,
        text_extractor=text_extractor,
        chunker=chunker,
        unit_of_work=unit_of_work,
    )

    return (
        service,
        repository,
        chunk_repository,
        file_storage,
        unit_of_work,
    )


def test_process_document_succeeds_for_correct_organization():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    service, repository, chunk_repository, file_storage, unit_of_work = (
        build_service(document)
    )

    document.asset = DocumentAsset(
        document_id=document.id,
        original_filename="Contract.pdf",
        stored_filename="contract.pdf",
        mime_type="application/pdf",
        file_size=11,
        checksum="test-checksum",
        storage_path="documents/contract.pdf",
    )

    file_storage.files["documents/contract.pdf"] = BytesIO(
        b"PDF content"
    )

    result = service.process_document(
        document_id,
        organization_id,
    )

    assert result is document
    assert result.status is DocumentStatus.READY

    assert chunk_repository.deleted_document_ids == [
        document_id,
    ]

    assert len(chunk_repository.chunks) == 1
    assert chunk_repository.chunks[0].document_id == document_id
    assert chunk_repository.chunks[0].content == "Extracted text"

    assert unit_of_work.commit_called == 2
    assert unit_of_work.rollback_called == 0


def test_process_document_rejects_wrong_organization():
    document_id = uuid.uuid4()
    document_organization_id = uuid.uuid4()
    requesting_organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=document_organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    service, _, chunk_repository, _, unit_of_work = build_service(
        document
    )

    with pytest.raises(DocumentNotFoundError):
        service.process_document(
            document_id,
            requesting_organization_id,
        )

    assert chunk_repository.deleted_document_ids == []
    assert chunk_repository.chunks == []
    assert unit_of_work.commit_called == 0
    assert unit_of_work.rollback_called == 0


def test_process_document_raises_when_not_found():
    service, _, chunk_repository, _, unit_of_work = build_service()

    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    with pytest.raises(DocumentNotFoundError):
        service.process_document(
            document_id,
            organization_id,
        )

    assert chunk_repository.deleted_document_ids == []
    assert chunk_repository.chunks == []
    assert unit_of_work.commit_called == 0
    assert unit_of_work.rollback_called == 0


def test_process_document_rejects_processing_document():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.PROCESSING,
    )

    service, _, _, _, unit_of_work = build_service(document)

    with pytest.raises(DocumentProcessingStateError):
        service.process_document(
            document_id,
            organization_id,
        )

    assert unit_of_work.commit_called == 0
    assert unit_of_work.rollback_called == 0


def test_process_document_rejects_ready_document():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.READY,
    )

    service, _, _, _, unit_of_work = build_service(document)

    with pytest.raises(DocumentProcessingStateError):
        service.process_document(
            document_id,
            organization_id,
        )

    assert unit_of_work.commit_called == 0
    assert unit_of_work.rollback_called == 0

def test_process_document_raises_when_organization_does_not_match():
    document_id = uuid.uuid4()
    document_organization_id = uuid.uuid4()
    wrong_organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=document_organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    service, repository, chunk_repository, file_storage, unit_of_work = (
        build_service(document)
    )

    with pytest.raises(DocumentNotFoundError):
        service.process_document(
            document_id,
            wrong_organization_id,
        )

    assert document.status is DocumentStatus.UPLOADED
    assert unit_of_work.commit_called == 0
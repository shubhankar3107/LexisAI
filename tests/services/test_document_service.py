import uuid
from datetime import datetime, timezone
from io import BytesIO

import pytest

from app.models.document import Document
from app.models.document_asset import DocumentAsset
from app.services.document_service import DocumentService
from app.services.exceptions import DocumentNotFoundError
from app.storage.file_storage import FileStorage


class FakeDocumentRepository:
    def __init__(self, document=None):
        self.document = document

    def get_by_id(self, document_id):
        if self.document is None:
            return None

        if self.document.id != document_id:
            return None

        if self.document.deleted_at is not None:
            return None

        return self.document

    def delete(self, document):
        document.deleted_at = datetime.now(
            timezone.utc,
        )


class FakeListRepository(FakeDocumentRepository):
    def __init__(self, documents):
        super().__init__()
        self.documents = documents

    def list(self, limit=50, offset=0):
        return self.documents[offset : offset + limit]

    def count(self):
        return len(self.documents)


class FakeFileStorage(FileStorage):
    def __init__(self):
        self.read_files = {}

    def save(self, file, path):
        pass

    def delete(self, path):
        pass

    def read(self, path):
        return self.read_files[path]


class FakeUnitOfWork:
    def __init__(self):
        self.commit_called = False
        self.rollback_called = False

    def commit(self):
        self.commit_called = True

    def rollback(self):
        self.rollback_called = True


def test_get_document_returns_document():
    document_id = uuid.uuid4()

    document = Document(
        id=document_id,
        title="Contract",
    )

    repository = FakeDocumentRepository(document)
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    result = service.get_document(document_id)

    assert result is document
    assert result.id == document_id
    assert result.title == "Contract"


def test_get_document_raises_when_not_found():
    repository = FakeDocumentRepository()
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    document_id = uuid.uuid4()

    with pytest.raises(DocumentNotFoundError):
        service.get_document(document_id)


def test_delete_document():
    document_id = uuid.uuid4()

    document = Document(
        id=document_id,
        title="Contract",
    )

    repository = FakeDocumentRepository(document)
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    service.delete_document(document_id)

    assert document.deleted_at is not None
    assert unit_of_work.commit_called is True
    assert unit_of_work.rollback_called is False


def test_delete_document_raises_when_not_found():
    repository = FakeDocumentRepository()
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    document_id = uuid.uuid4()

    with pytest.raises(DocumentNotFoundError):
        service.delete_document(document_id)

    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is False


def test_list_documents():
    document_one = Document(
        id=uuid.uuid4(),
        title="Contract",
    )

    document_two = Document(
        id=uuid.uuid4(),
        title="Roadmap",
    )

    repository = FakeListRepository(
        [
            document_one,
            document_two,
        ],
    )

    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    documents, total = service.list_documents()

    assert documents == [
        document_one,
        document_two,
    ]

    assert total == 2


def test_download_document():
    document_id = uuid.uuid4()

    document = Document(
        id=document_id,
        title="Contract",
    )

    asset = DocumentAsset(
        document_id=document_id,
        original_filename="Contract.pdf",
        stored_filename=f"{document_id}.pdf",
        mime_type="application/pdf",
        file_size=11,
        checksum="test-checksum",
        storage_path=f"documents/{document_id}/{document_id}.pdf",
    )

    document.asset = asset

    repository = FakeDocumentRepository(document)
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    file = BytesIO(b"PDF content")

    file_storage.read_files[asset.storage_path] = file

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    result_document, result_file = service.download_document(
        document_id,
    )

    assert result_document is document
    assert result_file is file
    assert result_file.read() == b"PDF content"


def test_download_document_raises_when_not_found():
    repository = FakeDocumentRepository()
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    document_id = uuid.uuid4()

    with pytest.raises(DocumentNotFoundError):
        service.download_document(document_id)


def test_get_document_raises_after_soft_delete():
    document_id = uuid.uuid4()

    document = Document(
        id=document_id,
        title="Contract",
    )

    repository = FakeDocumentRepository(document)
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    service.delete_document(document_id)

    with pytest.raises(DocumentNotFoundError):
        service.get_document(document_id)


def test_download_document_raises_after_soft_delete():
    document_id = uuid.uuid4()

    document = Document(
        id=document_id,
        title="Contract",
    )

    asset = DocumentAsset(
        document_id=document_id,
        original_filename="Contract.pdf",
        stored_filename=f"{document_id}.pdf",
        mime_type="application/pdf",
        file_size=11,
        checksum="test-checksum",
        storage_path=f"documents/{document_id}/{document_id}.pdf",
    )

    document.asset = asset

    repository = FakeDocumentRepository(document)
    unit_of_work = FakeUnitOfWork()
    file_storage = FakeFileStorage()

    service = DocumentService(
        repository,
        unit_of_work,
        file_storage,
    )

    service.delete_document(document_id)

    with pytest.raises(DocumentNotFoundError):
        service.download_document(document_id)

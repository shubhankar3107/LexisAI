import uuid
from io import BytesIO

from app.enums.document_status import DocumentStatus
from app.models.document_asset import DocumentAsset
from app.services.schemas.upload import UploadDocumentInput
from app.services.upload_service import UploadService
from app.storage.file_storage import FileStorage
from app.storage.storage_path import StoragePathBuilder


class FakeFileStorage(FileStorage):
    def __init__(self):
        self.saved_files = []

    def save(self, file, path):
        self.saved_files.append(
            (file, path),
        )

    def delete(self, path):
        pass

    def read(self, path):
        raise NotImplementedError


class FakeChecksumCalculator:
    def calculate(self, file):
        return "test-checksum"


class FakeDocumentRepository:
    def __init__(self):
        self.documents = []

    def add(self, document):
        self.documents.append(document)
        return document


class FakeUnitOfWork:
    def __init__(self):
        self.commit_called = False
        self.rollback_called = False

    def commit(self):
        self.commit_called = True

    def rollback(self):
        self.rollback_called = True


def test_upload_saves_file():
    organization_id = uuid.uuid4()
    file_storage = FakeFileStorage()
    checksum_calculator = FakeChecksumCalculator()
    document_repository = FakeDocumentRepository()
    unit_of_work = FakeUnitOfWork()

    service = UploadService(
        document_repository=document_repository,
        file_storage=file_storage,
        storage_path_builder=StoragePathBuilder(),
        checksum_calculator=checksum_calculator,
        unit_of_work=unit_of_work,
    )

    input_data = UploadDocumentInput(
        filename="Contract.PDF",
        content_type="application/pdf",
        file_size=1024,
    )

    file = BytesIO(b"PDF content")

    document_id, stored_filename, storage_path, checksum = service.upload(
        input_data,
        file,
        organization_id=organization_id,
    )

    assert isinstance(document_id, uuid.UUID)

    assert stored_filename == f"{document_id}.pdf"

    assert storage_path == (f"documents/{document_id}/{document_id}.pdf")

    assert checksum == "test-checksum"

    assert file.tell() == 0

    assert len(file_storage.saved_files) == 1

    saved_file, saved_path = file_storage.saved_files[0]

    assert saved_file is file
    assert saved_path == storage_path

    assert len(document_repository.documents) == 1

    document = document_repository.documents[0]

    assert document.id == document_id
    assert document.organization_id == organization_id
    assert document.title == "Contract"
    assert document.status is DocumentStatus.UPLOADED
    assert document.asset is not None
    assert isinstance(document.asset, DocumentAsset)

    asset = document.asset

    assert asset.document_id == document_id
    assert asset.original_filename == "Contract.PDF"
    assert asset.stored_filename == stored_filename
    assert asset.mime_type == "application/pdf"

    assert asset.file_size == 11

    assert asset.checksum == "test-checksum"
    assert asset.storage_path == storage_path

    assert unit_of_work.commit_called is True
    assert unit_of_work.rollback_called is False

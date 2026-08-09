from pathlib import Path

from app.api.dependencies import get_upload_service
from app.core.config import settings
from app.db.unit_of_work import UnitOfWork
from app.repositories.document_repository import DocumentRepository
from app.services.checksum import ChecksumCalculator
from app.services.upload_service import UploadService
from app.storage.local_file_storage import LocalFileStorage
from app.storage.storage_path import StoragePathBuilder


class FakeSession:
    pass


def test_get_upload_service():
    db = FakeSession()

    service = get_upload_service(db)

    assert isinstance(service, UploadService)

    assert isinstance(
        service._document_repository,
        DocumentRepository,
    )

    assert isinstance(
        service._unit_of_work,
        UnitOfWork,
    )

    assert isinstance(
        service._file_storage,
        LocalFileStorage,
    )

    assert isinstance(
        service._storage_path_builder,
        StoragePathBuilder,
    )

    assert isinstance(
        service._checksum_calculator,
        ChecksumCalculator,
    )

    assert service._file_storage._base_path == Path(
        settings.upload_directory,
    )

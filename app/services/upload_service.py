import uuid
from pathlib import Path
from typing import BinaryIO

from app.enums.document_status import DocumentStatus
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.services.checksum import ChecksumCalculator
from app.services.schemas.upload import UploadDocumentInput
from app.storage.file_storage import FileStorage
from app.storage.storage_path import StoragePathBuilder
from app.models.document_asset import DocumentAsset
from app.db.unit_of_work import UnitOfWork


class UploadService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        file_storage: FileStorage,
        storage_path_builder: StoragePathBuilder,
        checksum_calculator: ChecksumCalculator,
        unit_of_work: UnitOfWork,
    ):
        self._document_repository = document_repository
        self._file_storage = file_storage
        self._storage_path_builder = storage_path_builder
        self._checksum_calculator = checksum_calculator
        self._unit_of_work = unit_of_work

    def upload(
        self,
        input: UploadDocumentInput,
        file: BinaryIO,
        organization_id: uuid.UUID,
    ) -> tuple[uuid.UUID, str, str, str]:
        document_id = uuid.uuid4()

        stored_filename, storage_path = self._storage_path_builder.build(
            document_id,
            input.filename,
        )

        checksum = self._checksum_calculator.calculate(file)

        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)

        self._file_storage.save(
            file,
            storage_path,
        )

        title = Path(input.filename).stem

        document = Document(
            id=document_id,
            organization_id=organization_id,
            title=title,
            status=DocumentStatus.UPLOADED,
)

        asset = DocumentAsset(
            document_id=document_id,
            original_filename=input.filename,
            stored_filename=stored_filename,
            mime_type=input.content_type,
            file_size=file_size,
            checksum=checksum,
            storage_path=storage_path,
        )

        document.asset = asset

        try:
            self._document_repository.add(document)
            self._unit_of_work.commit()
        except Exception:
            self._unit_of_work.rollback()
            self._file_storage.delete(storage_path)
            raise

        return (
            document_id,
            stored_filename,
            storage_path,
            checksum,
        )

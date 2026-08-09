from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db.unit_of_work import UnitOfWork

from app.repositories.document_repository import DocumentRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository

from app.services.checksum import ChecksumCalculator
from app.services.upload_service import UploadService
from app.services.document_service import DocumentService
from app.services.document_processing_service import DocumentProcessingService
from app.services.pdf_text_extractor import PdfTextExtractor
from app.services.document_chunker import DocumentChunker

from app.storage.local_file_storage import LocalFileStorage
from app.storage.storage_path import StoragePathBuilder


def get_upload_service(
    db: Session = Depends(get_db),
) -> UploadService:
    document_repository = DocumentRepository(db)

    unit_of_work = UnitOfWork(db)

    file_storage = LocalFileStorage(
        settings.upload_directory,
    )

    storage_path_builder = StoragePathBuilder()

    checksum_calculator = ChecksumCalculator()

    return UploadService(
        document_repository=document_repository,
        file_storage=file_storage,
        storage_path_builder=storage_path_builder,
        checksum_calculator=checksum_calculator,
        unit_of_work=unit_of_work,
    )


def get_document_service(
    db: Session = Depends(get_db),
) -> DocumentService:
    document_repository = DocumentRepository(db)

    unit_of_work = UnitOfWork(db)

    file_storage = LocalFileStorage(
        settings.upload_directory,
    )

    return DocumentService(
        document_repository=document_repository,
        unit_of_work=unit_of_work,
        file_storage=file_storage,
    )


def get_document_processing_service(
    db: Session = Depends(get_db),
) -> DocumentProcessingService:
    document_repository = DocumentRepository(db)

    document_chunk_repository = DocumentChunkRepository(db)

    file_storage = LocalFileStorage(
        settings.upload_directory,
    )

    text_extractor = PdfTextExtractor()

    chunker = DocumentChunker()

    unit_of_work = UnitOfWork(db)

    return DocumentProcessingService(
        document_repository=document_repository,
        document_chunk_repository=document_chunk_repository,
        file_storage=file_storage,
        text_extractor=text_extractor,
        chunker=chunker,
        unit_of_work=unit_of_work,
    )

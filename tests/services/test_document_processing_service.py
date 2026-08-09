import uuid
from io import BytesIO

import pytest
from reportlab.pdfgen import canvas

from app.enums.document_status import DocumentStatus
from app.models.document import Document
from app.services.document_processing_service import DocumentProcessingService
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingStateError,
)
from app.services.document_chunker import DocumentChunker
from app.services.document_text_extractor import (
    DocumentTextExtractor,
    PageText,
)
from app.storage.file_storage import FileStorage
from app.models.document_asset import DocumentAsset
from app.services.pdf_text_extractor import PdfTextExtractor


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
    def __init__(self, content="Extracted text", page_count=1):
        self.content = content
        self.page_count = page_count

    def extract(self, file):
        return [
            PageText(
                page_number=page_number,
                content=self.content,
            )
            for page_number in range(1, self.page_count + 1)
        ]


class FakeChunker(DocumentChunker):
    def __init__(self, chunks_per_page=1):
        self.chunks_per_page = chunks_per_page

    def chunk(self, content):
        return [
            type(
                "FakeChunk",
                (),
                {
                    "chunk_index": index,
                    "content": content,
                },
            )()
            for index in range(self.chunks_per_page)
        ]


class FakeUnitOfWork:
    def __init__(self):
        self.commit_called = 0
        self.rollback_called = 0

    def commit(self):
        self.commit_called += 1

    def rollback(self):
        self.rollback_called += 1


def build_service(
    document=None,
    page_count=1,
    chunks_per_page=1,
    ):
    repository = FakeDocumentRepository(document)
    chunk_repository = FakeDocumentChunkRepository()
    file_storage = FakeFileStorage()
    text_extractor = FakeTextExtractor(page_count=page_count)
    chunker = FakeChunker(
    chunks_per_page=chunks_per_page,
    )
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


def test_process_document_preserves_page_number_on_multiple_chunks():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    service, repository, chunk_repository, file_storage, unit_of_work = (
        build_service(
            document,
            page_count=2,
            chunks_per_page=2,
        )
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

    service.process_document(
        document_id,
        organization_id,
    )

    assert len(chunk_repository.chunks) == 4

    assert chunk_repository.chunks[0].page_number == 1
    assert chunk_repository.chunks[1].page_number == 1
    assert chunk_repository.chunks[2].page_number == 2
    assert chunk_repository.chunks[3].page_number == 2

    assert chunk_repository.chunks[0].chunk_index == 0
    assert chunk_repository.chunks[1].chunk_index == 1
    assert chunk_repository.chunks[2].chunk_index == 2
    assert chunk_repository.chunks[3].chunk_index == 3


def test_process_document_marks_failed_when_extraction_fails():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    service, _, chunk_repository, file_storage, unit_of_work = (
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

    class FailingTextExtractor(DocumentTextExtractor):
        def extract(self, file):
            raise ValueError("Failed to extract PDF text")

    service._text_extractor = FailingTextExtractor()

    with pytest.raises(ValueError, match="Failed to extract PDF text"):
        service.process_document(
            document_id,
            organization_id,
        )

    assert document.status is DocumentStatus.FAILED

    assert chunk_repository.deleted_document_ids == []

    assert chunk_repository.chunks == []

    assert unit_of_work.rollback_called == 1

    assert unit_of_work.commit_called == 2


def test_process_document_marks_failed_when_chunking_fails():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    service, _, chunk_repository, file_storage, unit_of_work = (
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

    class FailingChunker(DocumentChunker):
        def chunk(self, content):
            raise ValueError("Failed to chunk document")

    service._chunker = FailingChunker()

    with pytest.raises(ValueError, match="Failed to chunk document"):
        service.process_document(
            document_id,
            organization_id,
        )

    assert document.status is DocumentStatus.FAILED
    assert chunk_repository.deleted_document_ids == [
        document_id,
    ]
    assert chunk_repository.chunks == []
    assert unit_of_work.rollback_called == 1
    assert unit_of_work.commit_called == 2


def test_process_document_updates_existing_content():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.FAILED,
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

    from app.models.document_content import DocumentContent

    existing_content = DocumentContent(
        document_id=document.id,
        content="Old extracted content",
        page_count=1,
    )

    document.content = existing_content

    service, _, chunk_repository, file_storage, unit_of_work = (
        build_service(document)
    )

    file_storage.files["documents/contract.pdf"] = BytesIO(
        b"PDF content"
    )

    result = service.process_document(
        document_id,
        organization_id,
    )

    assert result.status is DocumentStatus.READY

    assert result.content is existing_content
    assert result.content.content == "Extracted text"
    assert result.content.page_count == 1

    assert len(chunk_repository.chunks) == 1
    assert unit_of_work.commit_called == 2
    assert unit_of_work.rollback_called == 0


def test_process_document_can_retry_failed_document():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.FAILED,
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

    service, _, chunk_repository, file_storage, unit_of_work = (
        build_service(document)
    )

    file_storage.files["documents/contract.pdf"] = BytesIO(
        b"PDF content"
    )

    result = service.process_document(
        document_id,
        organization_id,
    )

    assert result is document
    assert document.status is DocumentStatus.READY

    assert chunk_repository.deleted_document_ids == [
        document_id,
    ]

    assert len(chunk_repository.chunks) == 1
    assert chunk_repository.chunks[0].content == "Extracted text"

    assert unit_of_work.commit_called == 2
    assert unit_of_work.rollback_called == 0


def test_process_document_assigns_page_numbers_to_multiple_chunks():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    service, _, chunk_repository, file_storage, unit_of_work = (
        build_service(
            document,
            page_count=2,
            chunks_per_page=2,
        )
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

    service.process_document(
        document_id,
        organization_id,
    )

    assert len(chunk_repository.chunks) == 4

    assert [
        chunk.page_number
        for chunk in chunk_repository.chunks
    ] == [1, 1, 2, 2]

    assert [
        chunk.chunk_index
        for chunk in chunk_repository.chunks
    ] == [0, 1, 2, 3]

    assert all(
        chunk.document_id == document_id
        for chunk in chunk_repository.chunks
    )

    assert document.status is DocumentStatus.READY
    assert unit_of_work.commit_called == 2
    assert unit_of_work.rollback_called == 0


def test_process_document_works_with_real_extractor_and_chunker():
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    document = Document(
        id=document_id,
        organization_id=organization_id,
        title="Contract",
        status=DocumentStatus.UPLOADED,
    )

    document.asset = DocumentAsset(
        document_id=document.id,
        original_filename="Contract.pdf",
        stored_filename="contract.pdf",
        mime_type="application/pdf",
        file_size=0,
        checksum="test-checksum",
        storage_path="documents/contract.pdf",
    )

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    pdf.drawString(100, 750, "Payment terms are described here.")
    pdf.showPage()

    pdf.drawString(100, 750, "Termination terms are described here.")
    pdf.showPage()

    pdf.save()

    buffer.seek(0)

    file_storage = FakeFileStorage()
    file_storage.files["documents/contract.pdf"] = buffer

    repository = FakeDocumentRepository(document)
    chunk_repository = FakeDocumentChunkRepository()
    unit_of_work = FakeUnitOfWork()

    service = DocumentProcessingService(
        document_repository=repository,
        document_chunk_repository=chunk_repository,
        file_storage=file_storage,
        text_extractor=PdfTextExtractor(),
        chunker=DocumentChunker(
            chunk_size=1000,
            chunk_overlap=100,
        ),
        unit_of_work=unit_of_work,
    )

    result = service.process_document(
        document_id,
        organization_id,
    )

    assert result is document
    assert result.status is DocumentStatus.READY

    assert len(chunk_repository.chunks) == 2

    assert [
        chunk.page_number
        for chunk in chunk_repository.chunks
    ] == [1, 2]

    assert [
        chunk.chunk_index
        for chunk in chunk_repository.chunks
    ] == [0, 1]

    assert "Payment terms" in chunk_repository.chunks[0].content
    assert "Termination terms" in chunk_repository.chunks[1].content

    assert unit_of_work.commit_called == 2
    assert unit_of_work.rollback_called == 0
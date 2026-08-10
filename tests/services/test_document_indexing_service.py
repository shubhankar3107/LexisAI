import uuid

import pytest

from app.enums.document_status import DocumentStatus
from app.services.document_indexing_service import (
    DocumentIndexingService,
)
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingStateError,
)


class FakeDocument:
    def __init__(
        self,
        document_id,
        organization_id,
        status=DocumentStatus.READY,
    ):
        self.id = document_id
        self.organization_id = organization_id
        self.status = status


class FakeDocumentChunk:
    def __init__(
        self,
        chunk_id,
        document_id,
        content,
        page_number=1,
        chunk_index=0,
    ):
        self.id = chunk_id
        self.document_id = document_id
        self.content = content
        self.page_number = page_number
        self.chunk_index = chunk_index


class FakeDocumentRepository:
    def __init__(self, document=None):
        self.document = document
        self.calls = []

    def get_by_id(
        self,
        document_id,
        organization_id,
    ):
        self.calls.append(
            (
                document_id,
                organization_id,
            )
        )

        if self.document is None:
            return None

        if self.document.id != document_id:
            return None

        if self.document.organization_id != organization_id:
            return None

        return self.document


class FakeDocumentChunkRepository:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls = []

    def list_by_document_id(
        self,
        document_id,
    ):
        self.calls.append(document_id)

        return [
            chunk
            for chunk in self.chunks
            if chunk.document_id == document_id
        ]


class FakeEmbeddingService:
    def __init__(
        self,
        vectors=None,
        error=None,
    ):
        self.vectors = vectors
        self.error = error
        self.calls = []

    def embed_documents(
        self,
        texts,
    ):
        self.calls.append(texts)

        if self.error is not None:
            raise self.error

        if self.vectors is not None:
            return self.vectors

        return [
            [float(index + 1)]
            for index, _ in enumerate(texts)
        ]


class FakeVectorStore:
    def __init__(self):
        self.upsert_calls = []
        self.delete_by_document_calls = []
        self.error = None

    def upsert(
        self,
        index_id,
        records,
    ):
        self.upsert_calls.append(
            (
                index_id,
                records,
            )
        )

        if self.error is not None:
            raise self.error

    def delete_by_document_id(
        self,
        index_id,
        document_id,
    ):
        self.delete_by_document_calls.append(
            (
                index_id,
                document_id,
            )
        )


@pytest.fixture
def organization_id():
    return uuid.uuid4()


@pytest.fixture
def document_id():
    return uuid.uuid4()


@pytest.fixture
def document(
    document_id,
    organization_id,
):
    return FakeDocument(
        document_id=document_id,
        organization_id=organization_id,
        status=DocumentStatus.READY,
    )


@pytest.fixture
def chunks(document_id):
    return [
        FakeDocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=document_id,
            content="First contract clause.",
            page_number=1,
            chunk_index=0,
        ),
        FakeDocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=document_id,
            content="Second contract clause.",
            page_number=1,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def document_repository(document):
    return FakeDocumentRepository(
        document=document,
    )


@pytest.fixture
def chunk_repository(chunks):
    return FakeDocumentChunkRepository(
        chunks=chunks,
    )


@pytest.fixture
def embedding_service():
    return FakeEmbeddingService(
        vectors=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )


@pytest.fixture
def vector_store():
    return FakeVectorStore()


@pytest.fixture
def service(
    document_repository,
    chunk_repository,
    embedding_service,
    vector_store,
):
    return DocumentIndexingService(
        document_repository=document_repository,
        document_chunk_repository=chunk_repository,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def test_index_document_embeds_chunks_and_upserts_vectors(
    service,
    document,
    chunks,
    embedding_service,
    vector_store,
    organization_id,
):
    index_id = uuid.uuid4()

    service.index_document(
        document_id=document.id,
        organization_id=organization_id,
        index_id=index_id,
    )

    assert embedding_service.calls == [
        [
            "First contract clause.",
            "Second contract clause.",
        ]
    ]

    assert len(vector_store.upsert_calls) == 1

    called_index_id, records = (
        vector_store.upsert_calls[0]
    )

    assert called_index_id == str(index_id)

    assert len(records) == 2

    assert records[0].chunk_id == chunks[0].id
    assert records[0].document_id == document.id
    assert records[0].organization_id == organization_id
    assert records[0].page_number == 1
    assert records[0].chunk_index == 0
    assert records[0].content == (
        "First contract clause."
    )
    assert records[0].vector == [
        0.1,
        0.2,
        0.3,
    ]

    assert records[1].chunk_id == chunks[1].id
    assert records[1].chunk_index == 1
    assert records[1].content == (
        "Second contract clause."
    )
    assert records[1].vector == [
        0.4,
        0.5,
        0.6,
    ]


def test_index_document_deletes_existing_document_vectors_before_upsert(
    service,
    document,
    vector_store,
    organization_id,
):
    index_id = uuid.uuid4()

    service.index_document(
        document_id=document.id,
        organization_id=organization_id,
        index_id=index_id,
    )

    assert vector_store.delete_by_document_calls == [
        (
            str(index_id),
            str(document.id),
        )
    ]

    assert len(vector_store.upsert_calls) == 1


def test_index_document_embeds_before_deleting_existing_vectors(
    service,
    document,
    embedding_service,
    vector_store,
    organization_id,
):
    index_id = uuid.uuid4()

    events = []

    original_embed = (
        embedding_service.embed_documents
    )

    original_delete = (
        vector_store.delete_by_document_id
    )

    def tracked_embed(texts):
        events.append("embed")
        return original_embed(texts)

    def tracked_delete(index_id, document_id):
        events.append("delete")
        return original_delete(
            index_id,
            document_id,
        )

    embedding_service.embed_documents = tracked_embed
    vector_store.delete_by_document_id = tracked_delete

    service.index_document(
        document_id=document.id,
        organization_id=organization_id,
        index_id=index_id,
    )

    assert events == [
        "embed",
        "delete",
    ]


def test_index_document_embedding_failure_preserves_existing_vectors(
    service,
    document,
    embedding_service,
    vector_store,
    organization_id,
):
    embedding_service.error = RuntimeError(
        "Embedding provider failed",
    )

    index_id = uuid.uuid4()

    with pytest.raises(
        RuntimeError,
        match="Embedding provider failed",
    ):
        service.index_document(
            document_id=document.id,
            organization_id=organization_id,
            index_id=index_id,
        )

    assert vector_store.delete_by_document_calls == []
    assert vector_store.upsert_calls == []


def test_index_document_vector_count_mismatch_preserves_existing_vectors(
    service,
    document,
    embedding_service,
    vector_store,
    organization_id,
):
    embedding_service.vectors = [
        [0.1, 0.2, 0.3],
    ]

    index_id = uuid.uuid4()

    with pytest.raises(
        ValueError,
        match="unexpected number of vectors",
    ):
        service.index_document(
            document_id=document.id,
            organization_id=organization_id,
            index_id=index_id,
        )

    assert vector_store.delete_by_document_calls == []
    assert vector_store.upsert_calls == []


def test_index_document_empty_document_does_not_delete_vectors(
    document_repository,
    embedding_service,
    organization_id,
):
    chunk_repository = FakeDocumentChunkRepository(
        chunks=[],
    )

    vector_store = FakeVectorStore()

    service = DocumentIndexingService(
        document_repository=document_repository,
        document_chunk_repository=chunk_repository,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    index_id = uuid.uuid4()

    service.index_document(
        document_id=document_repository.document.id,
        organization_id=organization_id,
        index_id=index_id,
    )

    assert embedding_service.calls == []

    assert vector_store.delete_by_document_calls == []
    assert vector_store.upsert_calls == []


def test_index_document_document_not_found(
    chunk_repository,
    embedding_service,
    vector_store,
    organization_id,
):
    missing_document_id = uuid.uuid4()

    document_repository = FakeDocumentRepository(
        document=None,
    )

    service = DocumentIndexingService(
        document_repository=document_repository,
        document_chunk_repository=chunk_repository,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=str(missing_document_id),
    ):
        service.index_document(
            document_id=missing_document_id,
            organization_id=organization_id,
            index_id=uuid.uuid4(),
        )

    assert embedding_service.calls == []
    assert vector_store.delete_by_document_calls == []
    assert vector_store.upsert_calls == []


def test_index_document_organization_isolation(
    document,
    chunk_repository,
    embedding_service,
    vector_store,
):
    wrong_organization_id = uuid.uuid4()

    document_repository = FakeDocumentRepository(
        document=document,
    )

    service = DocumentIndexingService(
        document_repository=document_repository,
        document_chunk_repository=chunk_repository,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    with pytest.raises(
        DocumentNotFoundError,
    ):
        service.index_document(
            document_id=document.id,
            organization_id=wrong_organization_id,
            index_id=uuid.uuid4(),
        )

    assert embedding_service.calls == []
    assert vector_store.delete_by_document_calls == []
    assert vector_store.upsert_calls == []


def test_index_document_upsert_failure_does_not_call_delete_again(
    service,
    document,
    vector_store,
    organization_id,
):
    vector_store.error = RuntimeError(
        "Vector store unavailable",
    )

    index_id = uuid.uuid4()

    with pytest.raises(
        RuntimeError,
        match="Vector store unavailable",
    ):
        service.index_document(
            document_id=document.id,
            organization_id=organization_id,
            index_id=index_id,
        )

    assert vector_store.delete_by_document_calls == [
        (
            str(index_id),
            str(document.id),
        )
    ]

    assert len(vector_store.upsert_calls) == 1
import uuid

import pytest

from app.services.schemas.retrieval import RetrievalResult
from app.services.schemas.vector import VectorRecord
from app.services.vector_store import VectorStore


class FakeVectorStore(VectorStore):

    def __init__(self):
        self.upsert_calls = []
        self.search_calls = []
        self.delete_calls = []
        self.delete_by_document_calls = []

    def upsert(
        self,
        index_id: str,
        records: list[VectorRecord],
    ) -> None:
        self.upsert_calls.append(
            (
                index_id,
                records,
            )
        )

    def search(
        self,
        index_id: str,
        query_vector: list[float],
        organization_id: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        self.search_calls.append(
            (
                index_id,
                query_vector,
                organization_id,
                top_k,
            )
        )

        return []

    def delete(
        self,
        index_id: str,
        chunk_ids: list[str],
    ) -> None:
        self.delete_calls.append(
            (
                index_id,
                chunk_ids,
            )
        )

    def delete_by_document_id(
        self,
        index_id: str,
        document_id: str,
    ) -> None:
        self.delete_by_document_calls.append(
            (
                index_id,
                document_id,
            )
        )


def test_vector_store_supports_upsert():
    store = FakeVectorStore()

    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    record = VectorRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        organization_id=organization_id,
        page_number=1,
        chunk_index=0,
        content="Contract text",
        vector=[0.1, 0.2, 0.3],
    )

    store.upsert(
        index_id="index-1",
        records=[record],
    )

    assert len(store.upsert_calls) == 1

    index_id, records = store.upsert_calls[0]

    assert index_id == "index-1"
    assert records == [record]


def test_vector_store_supports_search():
    store = FakeVectorStore()

    result = store.search(
        index_id="index-1",
        query_vector=[0.1, 0.2, 0.3],
        organization_id=str(uuid.uuid4()),
        top_k=5,
    )

    assert result == []
    assert len(store.search_calls) == 1

    (
        index_id,
        query_vector,
        organization_id,
        top_k,
    ) = store.search_calls[0]

    assert index_id == "index-1"
    assert query_vector == [0.1, 0.2, 0.3]
    assert top_k == 5


def test_vector_store_supports_delete():
    store = FakeVectorStore()

    chunk_ids = [
        str(uuid.uuid4()),
        str(uuid.uuid4()),
    ]

    store.delete(
        index_id="index-1",
        chunk_ids=chunk_ids,
    )

    assert store.delete_calls == [
        (
            "index-1",
            chunk_ids,
        )
    ]


def test_vector_store_supports_delete_by_document_id():
    store = FakeVectorStore()

    document_id = str(uuid.uuid4())

    store.delete_by_document_id(
        index_id="index-1",
        document_id=document_id,
    )

    assert store.delete_by_document_calls == [
        (
            "index-1",
            document_id,
        )
    ]


def test_vector_store_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()


def test_vector_store_requires_all_operations():
    assert hasattr(
        VectorStore,
        "upsert",
    )

    assert hasattr(
        VectorStore,
        "search",
    )

    assert hasattr(
        VectorStore,
        "delete",
    )

    assert hasattr(
        VectorStore,
        "delete_by_document_id",
    )
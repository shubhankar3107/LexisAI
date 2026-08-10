from __future__ import annotations

import uuid

from qdrant_client import QdrantClient

from app.services.qdrant_vector_store import QdrantVectorStore
from app.services.schemas.vector import VectorRecord
from qdrant_client.http import models


def make_record(
    organization_id: uuid.UUID,
    *,
    vector: list[float] | None = None,
    content: str = "Contract text",
) -> VectorRecord:
    return VectorRecord(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        organization_id=organization_id,
        page_number=1,
        chunk_index=0,
        content=content,
        vector=vector or [1.0, 0.0, 0.0],
    )


def make_store() -> tuple[
    QdrantClient,
    QdrantVectorStore,
]:
    client = QdrantClient(
        ":memory:",
    )

    store = QdrantVectorStore(
        client=client,
        collection_prefix="test-lexisai",
    )

    return client, store


def test_upsert_creates_collection_and_stores_vectors():
    client, store = make_store()

    organization_id = uuid.uuid4()

    record = make_record(
        organization_id,
    )

    store.upsert(
        index_id="index-1",
        records=[record],
    )

    collection_name = "test-lexisai_index1"

    assert client.collection_exists(
        collection_name,
    )

    response = client.query_points(
        collection_name=collection_name,
        query=[1.0, 0.0, 0.0],
        limit=1,
        with_payload=True,
    )

    assert len(response.points) == 1

    point = response.points[0]

    assert point.id == str(record.chunk_id)

    assert point.payload["chunk_id"] == str(
        record.chunk_id,
    )

    assert point.payload["document_id"] == str(
        record.document_id,
    )

    assert point.payload["organization_id"] == str(
        organization_id,
    )

    assert point.payload["page_number"] == 1
    assert point.payload["chunk_index"] == 0
    assert point.payload["content"] == "Contract text"


def test_search_returns_retrieval_results():
    _, store = make_store()

    organization_id = uuid.uuid4()

    record = make_record(
        organization_id,
        vector=[1.0, 0.0, 0.0],
    )

    store.upsert(
        index_id="index-1",
        records=[record],
    )

    results = store.search(
        index_id="index-1",
        query_vector=[1.0, 0.0, 0.0],
        organization_id=str(
            organization_id,
        ),
        top_k=10,
    )

    assert len(results) == 1

    result = results[0]

    assert result.document_id == record.document_id
    assert result.chunk_id == record.chunk_id
    assert result.page_number == record.page_number
    assert result.chunk_index == record.chunk_index
    assert result.content == record.content

    assert result.score > 0.99


def test_search_filters_by_organization():
    _, store = make_store()

    organization_a = uuid.uuid4()
    organization_b = uuid.uuid4()

    record_a = make_record(
        organization_a,
        vector=[1.0, 0.0, 0.0],
        content="Organization A",
    )

    record_b = make_record(
        organization_b,
        vector=[1.0, 0.0, 0.0],
        content="Organization B",
    )

    store.upsert(
        index_id="index-1",
        records=[
            record_a,
            record_b,
        ],
    )

    results = store.search(
        index_id="index-1",
        query_vector=[1.0, 0.0, 0.0],
        organization_id=str(
            organization_a,
        ),
        top_k=10,
    )

    assert len(results) == 1

    assert results[0].chunk_id == record_a.chunk_id
    assert results[0].content == "Organization A"


def test_search_respects_top_k():
    _, store = make_store()

    organization_id = uuid.uuid4()

    records = [
        make_record(
            organization_id,
            vector=[1.0, 0.0, 0.0],
            content="First",
        ),
        make_record(
            organization_id,
            vector=[0.9, 0.1, 0.0],
            content="Second",
        ),
        make_record(
            organization_id,
            vector=[0.8, 0.2, 0.0],
            content="Third",
        ),
    ]

    store.upsert(
        index_id="index-1",
        records=records,
    )

    results = store.search(
        index_id="index-1",
        query_vector=[1.0, 0.0, 0.0],
        organization_id=str(
            organization_id,
        ),
        top_k=2,
    )

    assert len(results) == 2


def test_search_missing_collection_returns_empty():
    _, store = make_store()

    results = store.search(
        index_id="missing-index",
        query_vector=[1.0, 0.0, 0.0],
        organization_id=str(
            uuid.uuid4(),
        ),
        top_k=10,
    )

    assert results == []


def test_search_with_non_positive_top_k_returns_empty():
    _, store = make_store()

    results = store.search(
        index_id="index-1",
        query_vector=[1.0, 0.0, 0.0],
        organization_id=str(
            uuid.uuid4(),
        ),
        top_k=0,
    )

    assert results == []


def test_delete_removes_vectors():
    _, store = make_store()

    organization_id = uuid.uuid4()

    record = make_record(
        organization_id,
    )

    store.upsert(
        index_id="index-1",
        records=[record],
    )

    store.delete(
        index_id="index-1",
        chunk_ids=[
            str(record.chunk_id),
        ],
    )

    results = store.search(
        index_id="index-1",
        query_vector=[1.0, 0.0, 0.0],
        organization_id=str(
            organization_id,
        ),
        top_k=10,
    )

    assert results == []


def test_delete_missing_collection_is_safe():
    _, store = make_store()

    store.delete(
        index_id="missing-index",
        chunk_ids=[
            str(uuid.uuid4()),
        ],
    )


def test_delete_empty_chunk_list_is_safe():
    _, store = make_store()

    store.delete(
        index_id="index-1",
        chunk_ids=[],
    )


def test_upsert_empty_records_is_safe():
    _, store = make_store()

    store.upsert(
        index_id="index-1",
        records=[],
    )


def test_delete_by_document_id_deletes_matching_vectors():
    from unittest.mock import Mock

    mock_client = Mock()

    mock_client.collection_exists.return_value = True

    store = QdrantVectorStore(
        client=mock_client,
        collection_prefix="lexisai",
    )

    document_id = str(uuid.uuid4())

    store.delete_by_document_id(
        index_id="index-1",
        document_id=document_id,
    )

    mock_client.collection_exists.assert_called_once_with(
        "lexisai_index1",
    )

    mock_client.delete.assert_called_once()

    call = mock_client.delete.call_args

    assert call.kwargs["collection_name"] == (
        "lexisai_index1"
    )

    selector = call.kwargs["points_selector"]

    assert isinstance(
        selector,
        models.FilterSelector,
    )

    filter_condition = selector.filter.must[0]

    assert isinstance(
        filter_condition,
        models.FieldCondition,
    )

    assert filter_condition.key == "document_id"

    assert isinstance(
        filter_condition.match,
        models.MatchValue,
    )

    assert filter_condition.match.value == document_id
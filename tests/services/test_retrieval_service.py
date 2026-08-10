from unittest.mock import Mock
import uuid

import pytest

from app.services.retrieval_service import RetrievalService
from app.services.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


def make_service(
    *,
    active_index=None,
):
    embedding_service = Mock()
    vector_store = Mock()
    vector_index_repository = Mock()

    vector_index_repository.get_active.return_value = (
        active_index
    )

    service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        vector_index_repository=vector_index_repository,
        profile_name="legal-general",
    )

    return (
        service,
        embedding_service,
        vector_store,
        vector_index_repository,
    )


def make_index():
    index = Mock()
    index.id = uuid.uuid4()
    return index


def make_result():
    return RetrievalResult(
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        content="Contract text",
        score=0.91,
    )


def test_retrieve_resolves_active_index():
    index = make_index()

    (
        service,
        embedding_service,
        vector_store,
        repository,
    ) = make_service(
        active_index=index,
    )

    query_vector = [0.1, 0.2, 0.3]

    embedding_service.embed_query.return_value = (
        query_vector
    )

    vector_store.search.return_value = []

    organization_id = uuid.uuid4()

    query = RetrievalQuery(
        query="termination notice",
        organization_id=organization_id,
        top_k=5,
    )

    service.retrieve(query)

    repository.get_active.assert_called_once_with(
        organization_id=organization_id,
        profile_name="legal-general",
    )

    embedding_service.embed_query.assert_called_once_with(
        "termination notice",
    )

    vector_store.search.assert_called_once_with(
        index_id=str(index.id),
        query_vector=query_vector,
        organization_id=str(organization_id),
        top_k=5,
        document_id=None,
    )


def test_retrieve_returns_vector_store_results():
    index = make_index()

    (
        service,
        embedding_service,
        vector_store,
        _,
    ) = make_service(
        active_index=index,
    )

    result = make_result()

    embedding_service.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    vector_store.search.return_value = [
        result,
    ]

    query = RetrievalQuery(
        query="termination notice",
        organization_id=uuid.uuid4(),
    )

    results = service.retrieve(query)

    assert results == [result]


def test_retrieve_passes_document_filter():
    index = make_index()

    (
        service,
        embedding_service,
        vector_store,
        _,
    ) = make_service(
        active_index=index,
    )

    embedding_service.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    vector_store.search.return_value = []

    organization_id = uuid.uuid4()
    document_id = uuid.uuid4()

    query = RetrievalQuery(
        query="termination notice",
        organization_id=organization_id,
        document_id=document_id,
        top_k=3,
    )

    service.retrieve(query)

    vector_store.search.assert_called_once_with(
        index_id=str(index.id),
        query_vector=[
            0.1,
            0.2,
            0.3,
        ],
        organization_id=str(organization_id),
        top_k=3,
        document_id=str(document_id),
    )


def test_retrieve_returns_empty_when_no_active_index():
    (
        service,
        embedding_service,
        vector_store,
        repository,
    ) = make_service(
        active_index=None,
    )

    query = RetrievalQuery(
        query="termination notice",
        organization_id=uuid.uuid4(),
    )

    results = service.retrieve(query)

    assert results == []

    embedding_service.embed_query.assert_not_called()
    vector_store.search.assert_not_called()
    repository.get_active.assert_called_once()


def test_retrieve_non_positive_top_k_returns_empty():
    (
        service,
        embedding_service,
        vector_store,
        repository,
    ) = make_service(
        active_index=make_index(),
    )

    query = RetrievalQuery(
        query="termination notice",
        organization_id=uuid.uuid4(),
        top_k=0,
    )

    results = service.retrieve(query)

    assert results == []

    repository.get_active.assert_not_called()
    embedding_service.embed_query.assert_not_called()
    vector_store.search.assert_not_called()


def test_retrieve_uses_configured_embedding_profile():
    index = make_index()

    (
        service,
        _,
        _,
        repository,
    ) = make_service(
        active_index=index,
    )

    query = RetrievalQuery(
        query="payment terms",
        organization_id=uuid.uuid4(),
    )

    service.retrieve(query)

    repository.get_active.assert_called_once_with(
        organization_id=query.organization_id,
        profile_name="legal-general",
    )


def test_retrieve_propagates_vector_store_failure():
    index = make_index()

    (
        service,
        embedding_service,
        vector_store,
        _,
    ) = make_service(
        active_index=index,
    )

    embedding_service.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    vector_store.search.side_effect = RuntimeError(
        "Qdrant unavailable",
    )

    query = RetrievalQuery(
        query="termination notice",
        organization_id=uuid.uuid4(),
        top_k=5,
    )

    with pytest.raises(
        RuntimeError,
        match="Qdrant unavailable",
    ):
        service.retrieve(query)

    embedding_service.embed_query.assert_called_once_with(
        "termination notice",
    )

    vector_store.search.assert_called_once()
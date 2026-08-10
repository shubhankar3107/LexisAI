import uuid

import pytest

from app.services.retriever import (
    Retriever,
    SemanticRetriever,
)
from app.services.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


def make_result():
    return RetrievalResult(
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        content="Contract text",
        score=0.92,
    )


class FakeRetrievalService:

    def __init__(
        self,
        results=None,
    ):
        self.results = results or []
        self.received_query = None

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        self.received_query = query

        return self.results


def test_retriever_returns_retrieval_results():
    result = make_result()

    retrieval_service = FakeRetrievalService(
        results=[result],
    )

    retriever = SemanticRetriever(
        retrieval_service=retrieval_service,
    )

    query = RetrievalQuery(
        query="What is the termination notice?",
        organization_id=uuid.uuid4(),
    )

    results = retriever.retrieve(query)

    assert len(results) == 1
    assert isinstance(
        results[0],
        RetrievalResult,
    )
    assert results[0] is result


def test_semantic_retriever_delegates_to_retrieval_service():
    retrieval_service = FakeRetrievalService(
        results=[],
    )

    retriever = SemanticRetriever(
        retrieval_service=retrieval_service,
    )

    query = RetrievalQuery(
        query="payment terms",
        organization_id=uuid.uuid4(),
        top_k=5,
    )

    results = retriever.retrieve(query)

    assert results == []
    assert retrieval_service.received_query is query


def test_semantic_retriever_preserves_retrieval_results():
    results = [
        make_result(),
        make_result(),
    ]

    retrieval_service = FakeRetrievalService(
        results=results,
    )

    retriever = SemanticRetriever(
        retrieval_service=retrieval_service,
    )

    query = RetrievalQuery(
        query="contract obligations",
        organization_id=uuid.uuid4(),
        top_k=10,
    )

    actual_results = retriever.retrieve(query)

    assert actual_results == results


def test_semantic_retriever_returns_empty_results():
    retrieval_service = FakeRetrievalService(
        results=[],
    )

    retriever = SemanticRetriever(
        retrieval_service=retrieval_service,
    )

    query = RetrievalQuery(
        query="termination clause",
        organization_id=uuid.uuid4(),
    )

    results = retriever.retrieve(query)

    assert results == []


def test_semantic_retriever_passes_query_unchanged():
    retrieval_service = FakeRetrievalService()

    retriever = SemanticRetriever(
        retrieval_service=retrieval_service,
    )

    query = RetrievalQuery(
        query="What are the payment terms?",
        organization_id=uuid.uuid4(),
        top_k=7,
        document_id=uuid.uuid4(),
        filters={
            "document_type": "contract",
        },
    )

    retriever.retrieve(query)

    assert retrieval_service.received_query is query
    assert retrieval_service.received_query.query == (
        "What are the payment terms?"
    )
    assert retrieval_service.received_query.organization_id == (
        query.organization_id
    )
    assert retrieval_service.received_query.top_k == 7
    assert retrieval_service.received_query.document_id == (
        query.document_id
    )
    assert retrieval_service.received_query.filters == {
        "document_type": "contract",
    }


def test_semantic_retriever_propagates_retrieval_failure():
    class FailingRetrievalService:

        def retrieve(
            self,
            query: RetrievalQuery,
        ) -> list[RetrievalResult]:
            raise RuntimeError(
                "Retrieval failed",
            )

    retriever = SemanticRetriever(
        retrieval_service=FailingRetrievalService(),
    )

    query = RetrievalQuery(
        query="termination clause",
        organization_id=uuid.uuid4(),
    )

    with pytest.raises(
        RuntimeError,
        match="Retrieval failed",
    ):
        retriever.retrieve(query)


def test_retriever_is_abstract():
    with pytest.raises(TypeError):
        Retriever()
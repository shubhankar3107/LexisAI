import uuid

import pytest

from app.services.reranker import Reranker
from app.services.retrieval_pipeline import (
    RetrievalPipeline,
)
from app.services.retriever import Retriever
from app.services.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


class FakeRetriever(Retriever):

    def __init__(
        self,
        results=None,
    ):
        self.results = results or []
        self.calls = []

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        self.calls.append(query)
        return self.results


class FakeReranker(Reranker):

    def __init__(
        self,
        results=None,
    ):
        self.results = results
        self.calls = []

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append(
            (
                query,
                results,
                top_k,
            )
        )

        if self.results is not None:
            return self.results

        return results[:top_k]


def make_result(
    score: float,
    content: str,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        content=content,
        score=score,
    )


@pytest.fixture
def organization_id():
    return uuid.uuid4()


@pytest.fixture
def query(
    organization_id,
):
    return RetrievalQuery(
        query="What is the termination notice?",
        organization_id=organization_id,
        top_k=2,
    )


def test_pipeline_retrieves_then_reranks(
    query,
):
    first = make_result(
        score=0.70,
        content="First",
    )

    second = make_result(
        score=0.90,
        content="Second",
    )

    retriever = FakeRetriever(
        results=[
            first,
            second,
        ],
    )

    reranker = FakeReranker(
        results=[
            second,
            first,
        ],
    )

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    results = pipeline.retrieve(query)

    assert results == [
        second,
        first,
    ]

    assert retriever.calls == [
        query,
    ]

    assert reranker.calls == [
        (
            query.query,
            [
                first,
                second,
            ],
            query.top_k,
        )
    ]


def test_pipeline_passes_query_text_to_reranker(
    query,
):
    retriever = FakeRetriever(
        results=[
            make_result(
                0.9,
                "Contract clause",
            ),
        ],
    )

    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    pipeline.retrieve(query)

    assert reranker.calls[0][0] == (
        "What is the termination notice?"
    )


def test_pipeline_passes_top_k_to_reranker(
    query,
):
    retriever = FakeRetriever(
        results=[
            make_result(
                0.9,
                "Contract clause",
            ),
        ],
    )

    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    pipeline.retrieve(query)

    assert reranker.calls[0][2] == 2


def test_pipeline_returns_empty_when_retriever_returns_empty(
    query,
):
    retriever = FakeRetriever(
        results=[],
    )

    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    results = pipeline.retrieve(query)

    assert results == []
    assert reranker.calls == []


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        -10,
    ],
)
def test_pipeline_returns_empty_for_non_positive_top_k(
    organization_id,
    top_k,
):
    query = RetrievalQuery(
        query="termination",
        organization_id=organization_id,
        top_k=top_k,
    )

    retriever = FakeRetriever(
        results=[
            make_result(
                0.9,
                "Contract",
            ),
        ],
    )

    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    results = pipeline.retrieve(query)

    assert results == []
    assert retriever.calls == []
    assert reranker.calls == []


def test_pipeline_propagates_retriever_failure(
    query,
):
    class FailingRetriever(Retriever):

        def retrieve(
            self,
            query,
        ):
            raise RuntimeError(
                "Retriever failed",
            )

    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        retriever=FailingRetriever(),
        reranker=reranker,
    )

    with pytest.raises(
        RuntimeError,
        match="Retriever failed",
    ):
        pipeline.retrieve(query)

    assert reranker.calls == []


def test_pipeline_propagates_reranker_failure(
    query,
):
    class FailingReranker(Reranker):

        def rerank(
            self,
            query,
            results,
            top_k,
        ):
            raise RuntimeError(
                "Reranker failed",
            )

    retriever = FakeRetriever(
        results=[
            make_result(
                0.9,
                "Contract",
            ),
        ],
    )

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=FailingReranker(),
    )

    with pytest.raises(
        RuntimeError,
        match="Reranker failed",
    ):
        pipeline.retrieve(query)


def test_pipeline_does_not_modify_retriever_results(
    query,
):
    first = make_result(
        0.40,
        "First",
    )

    second = make_result(
        0.90,
        "Second",
    )

    candidates = [
        first,
        second,
    ]

    retriever = FakeRetriever(
        results=candidates,
    )

    reranker = FakeReranker(
        results=[
            second,
            first,
        ],
    )

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    pipeline.retrieve(query)

    assert candidates == [
        first,
        second,
    ]
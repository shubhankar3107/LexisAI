import uuid

import pytest

from app.services.reranker import (
    Reranker,
    ScoreReranker,
)
from app.services.schemas.retrieval import RetrievalResult


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


def test_reranker_is_abstract():
    with pytest.raises(TypeError):
        Reranker()


def test_score_reranker_orders_by_score():
    reranker = ScoreReranker()

    low = make_result(
        score=0.40,
        content="Low score",
    )

    high = make_result(
        score=0.95,
        content="High score",
    )

    medium = make_result(
        score=0.70,
        content="Medium score",
    )

    results = reranker.rerank(
        query="contract termination",
        results=[
            low,
            high,
            medium,
        ],
        top_k=3,
    )

    assert results == [
        high,
        medium,
        low,
    ]


def test_score_reranker_respects_top_k():
    reranker = ScoreReranker()

    results = [
        make_result(0.95, "First"),
        make_result(0.90, "Second"),
        make_result(0.85, "Third"),
        make_result(0.80, "Fourth"),
    ]

    ranked = reranker.rerank(
        query="contract",
        results=results,
        top_k=2,
    )

    assert len(ranked) == 2

    assert ranked[0].content == "First"
    assert ranked[1].content == "Second"


def test_score_reranker_returns_empty_for_empty_results():
    reranker = ScoreReranker()

    results = reranker.rerank(
        query="contract",
        results=[],
        top_k=5,
    )

    assert results == []


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        -10,
    ],
)
def test_score_reranker_returns_empty_for_non_positive_top_k(
    top_k,
):
    reranker = ScoreReranker()

    results = [
        make_result(
            score=0.95,
            content="Contract",
        ),
    ]

    ranked = reranker.rerank(
        query="contract",
        results=results,
        top_k=top_k,
    )

    assert ranked == []


def test_score_reranker_preserves_result_objects():
    reranker = ScoreReranker()

    result = make_result(
        score=0.95,
        content="Contract clause",
    )

    ranked = reranker.rerank(
        query="contract",
        results=[result],
        top_k=1,
    )

    assert ranked[0] is result


def test_score_reranker_does_not_modify_input_list():
    reranker = ScoreReranker()

    first = make_result(
        score=0.40,
        content="First",
    )

    second = make_result(
        score=0.90,
        content="Second",
    )

    original = [
        first,
        second,
    ]

    reranker.rerank(
        query="contract",
        results=original,
        top_k=2,
    )

    assert original == [
        first,
        second,
    ]
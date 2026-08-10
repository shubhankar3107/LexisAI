import uuid

from app.services.schemas.rag import (
    RAGRequest,
    RAGResponse,
    RAGSource,
)


def test_rag_request_defaults():
    organization_id = uuid.uuid4()

    request = RAGRequest(
        query="What is the termination notice?",
        organization_id=organization_id,
    )

    assert request.query == (
        "What is the termination notice?"
    )
    assert request.organization_id == organization_id
    assert request.top_k == 10
    assert request.document_id is None
    assert request.filters == {}


def test_rag_request_preserves_optional_constraints():
    organization_id = uuid.uuid4()
    document_id = uuid.uuid4()

    request = RAGRequest(
        query="What is the notice period?",
        organization_id=organization_id,
        top_k=5,
        document_id=document_id,
        filters={
            "document_type": "contract",
        },
    )

    assert request.top_k == 5
    assert request.document_id == document_id
    assert request.filters == {
        "document_type": "contract",
    }


def test_rag_request_is_immutable():
    request = RAGRequest(
        query="payment terms",
        organization_id=uuid.uuid4(),
    )

    try:
        request.query = "changed"
    except Exception:
        pass
    else:
        raise AssertionError(
            "RAGRequest should be immutable",
        )


def test_rag_source_contains_provenance():
    document_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    source = RAGSource(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=4,
        chunk_index=7,
        score=0.9345,
    )

    assert source.document_id == document_id
    assert source.chunk_id == chunk_id
    assert source.page_number == 4
    assert source.chunk_index == 7
    assert source.score == 0.9345


def test_rag_response_defaults_to_empty_sources():
    response = RAGResponse(
        answer="The contract requires 30 days notice.",
    )

    assert response.answer == (
        "The contract requires 30 days notice."
    )
    assert response.sources == []


def test_rag_response_preserves_sources():
    source = RAGSource(
        document_id=str(uuid.uuid4()),
        chunk_id=str(uuid.uuid4()),
        page_number=2,
        chunk_index=3,
        score=0.91,
    )

    response = RAGResponse(
        answer="The contract requires 30 days notice.",
        sources=[source],
    )

    assert response.answer == (
        "The contract requires 30 days notice."
    )
    assert response.sources == [source]


def test_rag_response_is_immutable():
    response = RAGResponse(
        answer="Answer",
    )

    try:
        response.answer = "Changed"
    except Exception:
        pass
    else:
        raise AssertionError(
            "RAGResponse should be immutable",
        )
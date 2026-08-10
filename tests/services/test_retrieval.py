import uuid
import pytest

from app.services.schemas.retrieval import RetrievalResult


def test_retrieval_result_contains_retrieval_metadata():
    document_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    result = RetrievalResult(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=3,
        chunk_index=7,
        content="The agreement may be terminated with 30 days notice.",
        score=0.91,
    )

    assert result.document_id == document_id
    assert result.chunk_id == chunk_id
    assert result.page_number == 3
    assert result.chunk_index == 7
    assert result.content == (
        "The agreement may be terminated with 30 days notice."
    )
    assert result.score == 0.91


def test_retrieval_result_is_immutable():
    result = RetrievalResult(
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        content="Contract text",
        score=0.85,
    )

    with pytest.raises(AttributeError):
        result.score = 0.95
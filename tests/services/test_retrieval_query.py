import uuid
import pytest

from app.services.schemas.retrieval import RetrievalQuery


def test_retrieval_query_contains_required_information():
    organization_id = uuid.uuid4()

    query = RetrievalQuery(
        query="What is the termination notice period?",
        organization_id=organization_id,
    )

    assert query.query == "What is the termination notice period?"
    assert query.organization_id == organization_id
    assert query.top_k == 10
    assert query.document_id is None
    assert query.filters == {}


def test_retrieval_query_supports_document_scoping():
    organization_id = uuid.uuid4()
    document_id = uuid.uuid4()

    query = RetrievalQuery(
        query="What are the payment terms?",
        organization_id=organization_id,
        top_k=5,
        document_id=document_id,
    )

    assert query.organization_id == organization_id
    assert query.top_k == 5
    assert query.document_id == document_id


def test_retrieval_query_supports_metadata_filters():
    organization_id = uuid.uuid4()

    query = RetrievalQuery(
        query="termination",
        organization_id=organization_id,
        filters={
            "document_type": "contract",
            "language": "en",
        },
    )

    assert query.filters == {
        "document_type": "contract",
        "language": "en",
    }


def test_retrieval_query_is_immutable():
    query = RetrievalQuery(
        query="termination notice",
        organization_id=uuid.uuid4(),
    )

    with pytest.raises(AttributeError):
        query.top_k = 20
import uuid

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_current_organization_id,
    get_rag_orchestrator,
)
from app.main import app
from app.services.schemas.rag import (
    RAGRequest,
    RAGResponse,
    RAGSource,
)


class FakeRAGOrchestrator:

    def __init__(self):
        self.received_request = None

    def answer(
        self,
        request: RAGRequest,
    ) -> RAGResponse:
        self.received_request = request

        return RAGResponse(
            answer="The contract requires 30 days notice.",
            sources=[
                RAGSource(
                    document_id=str(uuid.uuid4()),
                    chunk_id=str(uuid.uuid4()),
                    page_number=4,
                    chunk_index=7,
                    score=0.91,
                ),
            ],
        )


def make_client(
    organization_id: uuid.UUID,
    orchestrator: FakeRAGOrchestrator,
):
    app.dependency_overrides[
        get_rag_orchestrator
    ] = lambda: orchestrator

    app.dependency_overrides[
        get_current_organization_id
    ] = lambda: organization_id

    return TestClient(app)


def cleanup_overrides():
    app.dependency_overrides.pop(
        get_rag_orchestrator,
        None,
    )

    app.dependency_overrides.pop(
        get_current_organization_id,
        None,
    )


def test_query_rag():
    organization_id = uuid.uuid4()

    orchestrator = FakeRAGOrchestrator()

    client = make_client(
        organization_id,
        orchestrator,
    )

    try:
        response = client.post(
            "/rag/query",
            headers={
                "X-Organization-ID": str(
                    organization_id,
                ),
            },
            json={
                "query": "What is the termination notice?",
                "organization_id": str(
                    organization_id,
                ),
                "top_k": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["answer"] == (
            "The contract requires 30 days notice."
        )

        assert len(data["sources"]) == 1

        assert data["sources"][0]["page_number"] == 4
        assert data["sources"][0]["chunk_index"] == 7
        assert data["sources"][0]["score"] == 0.91

        assert orchestrator.received_request is not None

        assert (
            orchestrator.received_request.query
            == "What is the termination notice?"
        )

        assert (
            orchestrator.received_request.organization_id
            == organization_id
        )

        assert orchestrator.received_request.top_k == 5

    finally:
        cleanup_overrides()


def test_query_rag_rejects_organization_mismatch():
    header_organization_id = uuid.uuid4()
    request_organization_id = uuid.uuid4()

    orchestrator = FakeRAGOrchestrator()

    client = make_client(
        header_organization_id,
        orchestrator,
    )

    try:
        response = client.post(
            "/rag/query",
            headers={
                "X-Organization-ID": str(
                    header_organization_id,
                ),
            },
            json={
                "query": "What is the termination notice?",
                "organization_id": str(
                    request_organization_id,
                ),
            },
        )

        assert response.status_code == 403

        data = response.json()

        assert data["detail"] == "Organization mismatch"

        assert orchestrator.received_request is None

    finally:
        cleanup_overrides()


def test_query_rag_requires_organization_id():
    orchestrator = FakeRAGOrchestrator()

    app.dependency_overrides[
        get_rag_orchestrator
    ] = lambda: orchestrator

    client = TestClient(app)

    try:
        response = client.post(
            "/rag/query",
            json={
                "query": "What is the termination notice?",
                "organization_id": str(
                    uuid.uuid4(),
                ),
            },
        )

        assert response.status_code == 422

        assert orchestrator.received_request is None

    finally:
        cleanup_overrides()

def test_query_rag_validates_request():
    organization_id = uuid.uuid4()

    orchestrator = FakeRAGOrchestrator()

    client = make_client(
        organization_id,
        orchestrator,
    )

    try:
        response = client.post(
            "/rag/query",
            headers={
                "X-Organization-ID": str(
                    organization_id,
                ),
            },
            json={
                "organization_id": str(
                    organization_id,
                ),
            },
        )

        assert response.status_code == 422

        assert orchestrator.received_request is None

    finally:
        cleanup_overrides()
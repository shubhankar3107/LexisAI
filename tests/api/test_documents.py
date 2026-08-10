import uuid
from datetime import datetime, timezone
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_indexing_service,
    get_document_processing_service,
    get_document_service,
    get_vector_index_service,
)
from app.enums.document_status import DocumentStatus
from app.enums.vector_index_status import VectorIndexStatus
from app.main import app
from app.services.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingStateError,
)
from app.services.vector_index_service import (
    IndexNotFoundError,
    InvalidIndexTransitionError,
)


class FakeDocumentAsset:
    def __init__(self):
        self.original_filename = "Contract.pdf"
        self.stored_filename = "contract.pdf"
        self.mime_type = "application/pdf"
        self.file_size = 3850
        self.checksum = "test-checksum"
        self.storage_path = "documents/test/contract.pdf"


class FakeDocument:
    def __init__(
        self,
        document_id,
        title,
        status,
        organization_id,
    ):
        self.id = document_id
        self.title = title
        self.status = status
        self.organization_id = organization_id
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.deleted_at = None
        self.asset = FakeDocumentAsset()


class FakeVectorIndex:
    def __init__(
        self,
        index_id,
        organization_id,
        status=VectorIndexStatus.BUILDING,
    ):
        self.id = index_id
        self.organization_id = organization_id
        self.status = status


@pytest.fixture
def document():
    return FakeDocument(
        document_id=uuid.uuid4(),
        title="Contract",
        status=DocumentStatus.UPLOADED,
        organization_id=uuid.uuid4(),
    )


@pytest.fixture
def vector_index(document):
    return FakeVectorIndex(
        index_id=uuid.uuid4(),
        organization_id=document.organization_id,
    )


@pytest.fixture
def client(document, vector_index):
    def override_get_document_service():
        class FakeDocumentService:

            def _verify_organization(
                self,
                organization_id,
            ):
                if organization_id != document.organization_id:
                    raise DocumentNotFoundError(
                        f"Document {document.id} not found",
                    )

            def get_document(
                self,
                document_id,
                organization_id,
            ):
                self._verify_organization(
                    organization_id,
                )

                if document_id != document.id:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                if document.deleted_at is not None:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                return document

            def delete_document(
                self,
                document_id,
                organization_id,
            ):
                self._verify_organization(
                    organization_id,
                )

                if document_id != document.id:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                if document.deleted_at is not None:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                document.deleted_at = datetime.now(
                    timezone.utc,
                )

            def list_documents(
                self,
                organization_id,
                limit=50,
                offset=0,
            ):
                if organization_id != document.organization_id:
                    return [], 0

                documents = [document]

                return (
                    documents[offset : offset + limit],
                    len(documents),
                )

            def download_document(
                self,
                document_id,
                organization_id,
            ):
                self._verify_organization(
                    organization_id,
                )

                if document_id != document.id:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                if document.deleted_at is not None:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                file = BytesIO(b"PDF content")

                return document, file

        return FakeDocumentService()

    app.dependency_overrides[
        get_document_service
    ] = override_get_document_service

    class FakeDocumentIndexingService:
        def __init__(self):
            self.calls = []
            self.error = None

        def index_document(
            self,
            document_id,
            organization_id,
            index_id,
        ):
            self.calls.append(
                (
                    document_id,
                    organization_id,
                    index_id,
                )
            )

            if self.error is not None:
                raise self.error

            if document_id != document.id:
                raise DocumentNotFoundError(
                    f"Document {document_id} not found",
                )

            if organization_id != document.organization_id:
                raise DocumentNotFoundError(
                    f"Document {document_id} not found",
                )

            return None

    indexing_service = FakeDocumentIndexingService()

    document.indexing_service = indexing_service

    class FakeVectorIndexService:
        def __init__(self):
            self.mark_ready_calls = []
            self.mark_failed_calls = []
            self.mark_ready_error = None
            self.mark_failed_error = None

        def require_building_index(
            self,
            index_id,
            organization_id,
        ):
            if (
                index_id != vector_index.id
                or organization_id != vector_index.organization_id
            ):
                raise IndexNotFoundError(
                    f"Vector index {index_id} not found",
                )

            if vector_index.status != VectorIndexStatus.BUILDING:
                raise InvalidIndexTransitionError(
                    "Vector index must be in BUILDING state "
                    "for indexing, but is "
                    f"{vector_index.status.value}",
                )

            return vector_index

        def mark_ready(
            self,
            index_id,
            organization_id,
        ):
            self.mark_ready_calls.append(
                (
                    index_id,
                    organization_id,
                )
            )

            if self.mark_ready_error is not None:
                raise self.mark_ready_error

            if (
                index_id != vector_index.id
                or organization_id != vector_index.organization_id
            ):
                raise IndexNotFoundError(
                    f"Vector index {index_id} not found",
                )

            if vector_index.status != VectorIndexStatus.BUILDING:
                raise InvalidIndexTransitionError(
                    "Invalid vector-index transition: "
                    f"{vector_index.status.value} -> ready",
                )

            vector_index.status = VectorIndexStatus.READY

            return vector_index

        def mark_failed(
            self,
            index_id,
            organization_id,
        ):
            self.mark_failed_calls.append(
                (
                    index_id,
                    organization_id,
                )
            )

            if self.mark_failed_error is not None:
                raise self.mark_failed_error

            if (
                index_id != vector_index.id
                or organization_id != vector_index.organization_id
            ):
                raise IndexNotFoundError(
                    f"Vector index {index_id} not found",
                )

            if vector_index.status != VectorIndexStatus.BUILDING:
                raise InvalidIndexTransitionError(
                    "Invalid vector-index transition: "
                    f"{vector_index.status.value} -> failed",
                )

            vector_index.status = VectorIndexStatus.FAILED

            return vector_index

    vector_index_service = FakeVectorIndexService()

    document.vector_index_service = vector_index_service

    app.dependency_overrides[
        get_document_indexing_service
    ] = lambda: indexing_service

    app.dependency_overrides[
        get_vector_index_service
    ] = lambda: vector_index_service

    yield TestClient(
        app,
        raise_server_exceptions=False,
    )

    app.dependency_overrides.clear()


def organization_headers(document):
    return {
        "X-Organization-ID": str(document.organization_id),
    }


def test_get_document(client, document):
    response = client.get(
        f"/documents/{document.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(document.id)
    assert data["title"] == "Contract"
    assert data["status"] == "UPLOADED"

    assert data["created_at"] is not None
    assert data["updated_at"] is not None

    assert data["asset"]["original_filename"] == "Contract.pdf"
    assert data["asset"]["stored_filename"] == "contract.pdf"
    assert data["asset"]["mime_type"] == "application/pdf"
    assert data["asset"]["file_size"] == 3850
    assert data["asset"]["checksum"] == "test-checksum"


def test_get_document_not_found(client, document):
    document_id = uuid.uuid4()

    response = client.get(
        f"/documents/{document_id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"


def test_get_document_wrong_organization(client, document):
    response = client.get(
        f"/documents/{document.id}",
        headers={
            "X-Organization-ID": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"


def test_delete_document(client, document):
    response = client.delete(
        f"/documents/{document.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 204


def test_delete_document_not_found(client, document):
    document_id = uuid.uuid4()

    response = client.delete(
        f"/documents/{document_id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"


def test_delete_document_wrong_organization(client, document):
    response = client.delete(
        f"/documents/{document.id}",
        headers={
            "X-Organization-ID": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 404

    assert document.deleted_at is None


def test_list_documents(client, document):
    response = client.get(
        "/documents/",
        headers=organization_headers(document),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1
    assert data["limit"] == 50
    assert data["offset"] == 0
    assert len(data["items"]) == 1

    assert data["items"][0]["id"] == str(document.id)
    assert data["items"][0]["title"] == "Contract"
    assert data["items"][0]["status"] == "UPLOADED"


def test_list_documents_custom_pagination(client, document):
    response = client.get(
        "/documents/?limit=10&offset=5",
        headers=organization_headers(document),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["limit"] == 10
    assert data["offset"] == 5
    assert data["total"] == 1
    assert data["items"] == []


def test_list_documents_wrong_organization(client, document):
    response = client.get(
        "/documents/",
        headers={
            "X-Organization-ID": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["items"] == []
    assert data["total"] == 0


def test_get_document_requires_organization_id(
    client,
    document,
):
    response = client.get(
        f"/documents/{document.id}",
    )

    assert response.status_code == 422


def test_delete_document_requires_organization_id(
    client,
    document,
):
    response = client.delete(
        f"/documents/{document.id}",
    )

    assert response.status_code == 422


def test_list_documents_requires_organization_id(client):
    response = client.get(
        "/documents/",
    )

    assert response.status_code == 422


def test_download_document(client, document):
    response = client.get(
        f"/documents/{document.id}/download",
        headers=organization_headers(document),
    )

    assert response.status_code == 200

    assert response.content == b"PDF content"

    assert response.headers["content-type"] == "application/pdf"

    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="Contract.pdf"'
    )


def test_download_document_not_found(client, document):
    document_id = uuid.uuid4()

    response = client.get(
        f"/documents/{document_id}/download",
        headers=organization_headers(document),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"


def test_download_document_requires_organization_id(
    client,
    document,
):
    response = client.get(
        f"/documents/{document.id}/download",
    )

    assert response.status_code == 422


def test_process_document(client, document):
    class FakeDocumentProcessingService:
        def process_document(
            self,
            document_id,
            organization_id,
        ):
            assert document_id == document.id
            assert organization_id == document.organization_id

            document.status = DocumentStatus.READY

            return document

    app.dependency_overrides[
        get_document_processing_service
    ] = lambda: FakeDocumentProcessingService()

    try:
        response = client.post(
            f"/documents/{document.id}/process",
            headers=organization_headers(document),
        )

        assert response.status_code == 200

        data = response.json()

        assert data["document_id"] == str(document.id)
        assert data["status"] == "READY"

    finally:
        app.dependency_overrides.pop(
            get_document_processing_service,
            None,
        )


def test_process_document_not_found(client, document):
    class FakeDocumentProcessingService:
        def process_document(
            self,
            document_id,
            organization_id,
        ):
            raise DocumentNotFoundError(
                f"Document {document_id} not found",
            )

    app.dependency_overrides[
        get_document_processing_service
    ] = lambda: FakeDocumentProcessingService()

    try:
        response = client.post(
            f"/documents/{uuid.uuid4()}/process",
            headers=organization_headers(document),
        )

        assert response.status_code == 404

        data = response.json()

        assert data["detail"] == "Document not found"

    finally:
        app.dependency_overrides.pop(
            get_document_processing_service,
            None,
        )


def test_process_document_conflict(client, document):
    class FakeDocumentProcessingService:
        def process_document(
            self,
            document_id,
            organization_id,
        ):
            raise DocumentProcessingStateError(
                f"Document {document_id} is already processing",
            )

    app.dependency_overrides[
        get_document_processing_service
    ] = lambda: FakeDocumentProcessingService()

    try:
        response = client.post(
            f"/documents/{document.id}/process",
            headers=organization_headers(document),
        )

        assert response.status_code == 409

        data = response.json()

        assert data["detail"] == (
            f"Document {document.id} is already processing"
        )

    finally:
        app.dependency_overrides.pop(
            get_document_processing_service,
            None,
        )


def test_process_document_wrong_organization(client, document):
    class FakeDocumentProcessingService:
        def process_document(
            self,
            document_id,
            organization_id,
        ):
            raise DocumentNotFoundError(
                f"Document {document_id} not found",
            )

    app.dependency_overrides[
        get_document_processing_service
    ] = lambda: FakeDocumentProcessingService()

    try:
        response = client.post(
            f"/documents/{document.id}/process",
            headers={
                "X-Organization-ID": str(uuid.uuid4()),
            },
        )

        assert response.status_code == 404

        data = response.json()

        assert data["detail"] == "Document not found"

    finally:
        app.dependency_overrides.pop(
            get_document_processing_service,
            None,
        )


def test_index_document(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == str(document.id)
    assert data["index_id"] == str(vector_index.id)
    assert data["status"] == "ready"

    assert vector_index.status == VectorIndexStatus.READY

    assert document.vector_index_service.mark_ready_calls == [
        (
            vector_index.id,
            vector_index.organization_id,
        )
    ]

    assert document.vector_index_service.mark_failed_calls == []


def test_index_document_index_not_found(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    missing_index_id = uuid.uuid4()

    response = client.post(
        f"/documents/{document.id}/index/{missing_index_id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Vector index not found"


def test_index_document_wrong_organization(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
        headers={
            "X-Organization-ID": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Vector index not found"

    assert vector_index.status == VectorIndexStatus.BUILDING


def test_index_document_requires_organization_id(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
    )

    assert response.status_code == 422

    assert vector_index.status == VectorIndexStatus.BUILDING


def test_index_document_rejects_ready_index(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    vector_index.status = VectorIndexStatus.READY

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 409

    data = response.json()

    assert data["detail"] == (
        "Vector index must be in BUILDING state "
        "for indexing, but is ready"
    )

    assert vector_index.status == VectorIndexStatus.READY


@pytest.mark.parametrize(
    "index_status",
    [
        VectorIndexStatus.ACTIVE,
        VectorIndexStatus.FAILED,
        VectorIndexStatus.DEPRECATED,
    ],
)
def test_index_document_rejects_non_building_index(
    client,
    document,
    vector_index,
    index_status,
):
    document.status = DocumentStatus.READY

    vector_index.status = index_status

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 409

    data = response.json()

    assert data["detail"] == (
        "Vector index must be in BUILDING state "
        f"for indexing, but is {index_status.value}"
    )

    assert vector_index.status == index_status


def test_index_document_failure_marks_index_failed(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    document.indexing_service.error = RuntimeError(
        "Embedding provider failed",
    )

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 500

    assert vector_index.status == VectorIndexStatus.FAILED

    assert document.vector_index_service.mark_ready_calls == []

    assert document.vector_index_service.mark_failed_calls == [
        (
            vector_index.id,
            vector_index.organization_id,
        )
    ]


def test_index_document_success_does_not_mark_failed(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 200

    assert vector_index.status == VectorIndexStatus.READY

    assert document.vector_index_service.mark_ready_calls == [
        (
            vector_index.id,
            vector_index.organization_id,
        )
    ]

    assert document.vector_index_service.mark_failed_calls == []


def test_index_document_does_not_mark_failed_when_mark_ready_fails(
    client,
    document,
    vector_index,
):
    document.status = DocumentStatus.READY

    document.vector_index_service.mark_ready_error = RuntimeError(
        "Database failure while marking index ready",
    )

    response = client.post(
        f"/documents/{document.id}/index/{vector_index.id}",
        headers=organization_headers(document),
    )

    assert response.status_code == 500

    assert vector_index.status == VectorIndexStatus.BUILDING

    assert document.vector_index_service.mark_ready_calls == [
        (
            vector_index.id,
            vector_index.organization_id,
        )
    ]

    assert document.vector_index_service.mark_failed_calls == []
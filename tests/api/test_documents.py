import uuid
from datetime import datetime, timezone
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_processing_service,
    get_document_service,
)
from app.enums.document_status import DocumentStatus
from app.main import app
from app.services.exceptions import DocumentNotFoundError, DocumentProcessingStateError


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


@pytest.fixture
def document():
    return FakeDocument(
        document_id=uuid.uuid4(),
        title="Contract",
        status=DocumentStatus.UPLOADED,
        organization_id=uuid.uuid4(),
    )


@pytest.fixture
def client(document):
    def override_get_document_service():
        class FakeDocumentService:

            def _verify_organization(self, organization_id):
                if organization_id != document.organization_id:
                    raise DocumentNotFoundError(
                        f"Document {document.id} not found",
                    )

            def get_document(
                self,
                document_id,
                organization_id,
            ):
                self._verify_organization(organization_id)

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
                self._verify_organization(organization_id)

                if document_id != document.id:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                if document.deleted_at is not None:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                document.deleted_at = datetime.now(timezone.utc)

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
                self._verify_organization(organization_id)

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

    yield TestClient(app)

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

    assert response.status_code == 404


def test_get_document_requires_organization_id(client, document):
    response = client.get(
        f"/documents/{document.id}",
    )

    assert response.status_code == 422


def test_delete_document_requires_organization_id(client, document):
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


def test_download_document_requires_organization_id(client, document):
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
import uuid
from datetime import datetime, timezone
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_document_service
from app.enums.document_status import DocumentStatus
from app.main import app
from app.services.exceptions import DocumentNotFoundError


class FakeDocumentAsset:
    def __init__(self):
        self.original_filename = "Contract.pdf"
        self.stored_filename = "contract.pdf"
        self.mime_type = "application/pdf"
        self.file_size = 3850
        self.checksum = "test-checksum"


class FakeDocument:
    def __init__(self, document_id, title, status):
        self.id = document_id
        self.title = title
        self.status = status
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
    )


@pytest.fixture
def client(document):
    def override_get_document_service():
        class FakeDocumentService:
            def get_document(self, document_id):
                if document_id != document.id:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                if document.deleted_at is not None:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                return document

            def delete_document(self, document_id):
                if document_id != document.id:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                if document.deleted_at is not None:
                    raise DocumentNotFoundError(
                        f"Document {document_id} not found",
                    )

                document.deleted_at = datetime.now(timezone.utc)

            def list_documents(self, limit=50, offset=0):
                documents = [document]

                return (
                    documents[offset : offset + limit],
                    len(documents),
                )

            def download_document(self, document_id):
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

    app.dependency_overrides[get_document_service] = override_get_document_service

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_get_document(client, document):
    response = client.get(
        f"/documents/{document.id}",
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


def test_get_document_not_found(client):
    document_id = uuid.uuid4()

    response = client.get(
        f"/documents/{document_id}",
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"


def test_get_document_invalid_uuid(client):
    response = client.get(
        "/documents/not-a-uuid",
    )

    assert response.status_code == 422


def test_delete_document(client, document):
    response = client.delete(
        f"/documents/{document.id}",
    )

    assert response.status_code == 204
    assert response.content == b""


def test_delete_document_not_found(client):
    document_id = uuid.uuid4()

    response = client.delete(
        f"/documents/{document_id}",
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"


def test_list_documents(client, document):
    response = client.get("/documents/")

    assert response.status_code == 200

    data = response.json()

    assert data["limit"] == 50
    assert data["offset"] == 0
    assert data["total"] == 1

    assert len(data["items"]) == 1

    item = data["items"][0]

    assert item["id"] == str(document.id)
    assert item["title"] == "Contract"
    assert item["status"] == "UPLOADED"
    assert item["created_at"] is not None
    assert item["updated_at"] is not None


def test_list_documents_custom_pagination(client):
    response = client.get(
        "/documents/?limit=10&offset=5",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["limit"] == 10
    assert data["offset"] == 5
    assert data["total"] == 1
    assert data["items"] == []


def test_list_documents_invalid_limit(client):
    response = client.get(
        "/documents/?limit=0",
    )

    assert response.status_code == 422


def test_list_documents_limit_too_large(client):
    response = client.get(
        "/documents/?limit=101",
    )

    assert response.status_code == 422


def test_list_documents_negative_offset(client):
    response = client.get(
        "/documents/?offset=-1",
    )

    assert response.status_code == 422


def test_download_document(client, document):
    response = client.get(
        f"/documents/{document.id}/download",
    )

    assert response.status_code == 200

    assert response.content == b"PDF content"

    assert response.headers["content-type"] == "application/pdf"

    assert (
        response.headers["content-disposition"] == 'attachment; filename="Contract.pdf"'
    )


def test_download_document_not_found(client):
    document_id = uuid.uuid4()

    response = client.get(
        f"/documents/{document_id}/download",
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"

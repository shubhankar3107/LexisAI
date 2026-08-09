import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_upload_service
from app.main import app


class FakeUploadService:
    def upload(self, input_data, file):
        assert input_data.filename == "Contract.pdf"
        assert input_data.content_type == "application/pdf"

        assert file.read() == b"PDF content"

        document_id = uuid.uuid4()

        return (
            document_id,
            f"{document_id}.pdf",
            f"documents/{document_id}/{document_id}.pdf",
            "test-checksum",
        )


def override_get_upload_service():
    return FakeUploadService()


@pytest.fixture
def client():
    app.dependency_overrides[get_upload_service] = override_get_upload_service

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_upload_document(client):
    response = client.post(
        "/documents/",
        files={
            "file": (
                "Contract.pdf",
                b"PDF content",
                "application/pdf",
            ),
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert "document_id" in data
    assert "stored_filename" in data
    assert "storage_path" in data
    assert "checksum" in data

    assert data["stored_filename"].endswith(".pdf")
    assert data["storage_path"].startswith("documents/")
    assert data["checksum"] == "test-checksum"

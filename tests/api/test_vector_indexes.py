from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_vector_index_service
from app.enums.vector_index_status import VectorIndexStatus
from app.main import app
from app.services.embedding_registry import (
    EmbeddingProfileNotFoundError,
)
from app.services.vector_index_service import (
    IndexNotFoundError,
    InvalidIndexTransitionError,
)


class FakeVectorIndex:
    def __init__(
        self,
        index_id,
        organization_id,
        profile_name="legal-general",
        status=VectorIndexStatus.BUILDING,
        name="legal-index",
    ):
        self.id = index_id
        self.name = name
        self.organization_id = organization_id
        self.profile_name = profile_name
        self.embedding_identity_id = uuid.uuid4()
        self.status = status
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at


@pytest.fixture
def vector_index():
    return FakeVectorIndex(
        index_id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
    )


@pytest.fixture
def client(vector_index):
    class FakeVectorIndexService:
        def __init__(self):
            self.create_index_called = False
            self.create_index_arguments = None

        def create_index(
            self,
            organization_id,
            name,
            profile_name,
        ):
            self.create_index_called = True

            self.create_index_arguments = {
                "organization_id": organization_id,
                "name": name,
                "profile_name": profile_name,
            }

            if profile_name != "legal-general":
                raise EmbeddingProfileNotFoundError(
                    f"Embedding profile '{profile_name}' not found",
                )

            created_index = FakeVectorIndex(
                index_id=uuid.uuid4(),
                organization_id=organization_id,
                profile_name=profile_name,
                status=VectorIndexStatus.BUILDING,
                name=name,
            )

            return created_index

        def get_index(
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

            return vector_index

        def get_active_index(
            self,
            organization_id,
            profile_name,
        ):
            if (
                organization_id != vector_index.organization_id
                or profile_name != vector_index.profile_name
                or vector_index.status != VectorIndexStatus.ACTIVE
            ):
                return None

            return vector_index

        def mark_ready(
            self,
            index_id,
            organization_id,
        ):
            self._verify_index(
                index_id,
                organization_id,
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
            self._verify_index(
                index_id,
                organization_id,
            )

            if vector_index.status != VectorIndexStatus.BUILDING:
                raise InvalidIndexTransitionError(
                    "Invalid vector-index transition: "
                    f"{vector_index.status.value} -> failed",
                )

            vector_index.status = VectorIndexStatus.FAILED
            return vector_index

        def activate_index(
            self,
            index_id,
            organization_id,
        ):
            self._verify_index(
                index_id,
                organization_id,
            )

            if vector_index.status != VectorIndexStatus.READY:
                raise InvalidIndexTransitionError(
                    "Invalid vector-index transition: "
                    f"{vector_index.status.value} -> active",
                )

            vector_index.status = VectorIndexStatus.ACTIVE
            return vector_index

        def deprecate_index(
            self,
            index_id,
            organization_id,
        ):
            self._verify_index(
                index_id,
                organization_id,
            )

            if vector_index.status != VectorIndexStatus.ACTIVE:
                raise InvalidIndexTransitionError(
                    "Invalid vector-index transition: "
                    f"{vector_index.status.value} -> deprecated",
                )

            vector_index.status = VectorIndexStatus.DEPRECATED
            return vector_index

        @staticmethod
        def _verify_index(
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

    service = FakeVectorIndexService()

    app.dependency_overrides[
        get_vector_index_service
    ] = lambda: service

    yield TestClient(app)

    app.dependency_overrides.clear()


def organization_headers(vector_index):
    return {
        "X-Organization-ID": str(
            vector_index.organization_id,
        ),
    }


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_vector_index(client, vector_index):
    response = client.post(
        "/vector-indexes/",
        headers=organization_headers(vector_index),
        json={
            "name": "legal-general-v1",
            "profile_name": "legal-general",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["id"] is not None
    assert data["name"] == "legal-general-v1"
    assert data["organization_id"] == str(
        vector_index.organization_id,
    )
    assert data["profile_name"] == "legal-general"
    assert data["embedding_identity_id"] is not None
    assert data["status"] == "building"

    assert data["created_at"] is not None
    assert data["updated_at"] is not None


def test_create_vector_index_uses_organization_from_header(
    client,
    vector_index,
):
    organization_id = vector_index.organization_id

    response = client.post(
        "/vector-indexes/",
        headers={
            "X-Organization-ID": str(organization_id),
        },
        json={
            "name": "organization-index",
            "profile_name": "legal-general",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["organization_id"] == str(
        organization_id,
    )


def test_create_vector_index_requires_organization_id(
    client,
):
    response = client.post(
        "/vector-indexes/",
        json={
            "name": "legal-index",
            "profile_name": "legal-general",
        },
    )

    assert response.status_code == 422


def test_create_vector_index_requires_name(
    client,
    vector_index,
):
    response = client.post(
        "/vector-indexes/",
        headers=organization_headers(vector_index),
        json={
            "profile_name": "legal-general",
        },
    )

    assert response.status_code == 422


def test_create_vector_index_requires_profile_name(
    client,
    vector_index,
):
    response = client.post(
        "/vector-indexes/",
        headers=organization_headers(vector_index),
        json={
            "name": "legal-index",
        },
    )

    assert response.status_code == 422


def test_create_vector_index_rejects_invalid_profile(
    client,
    vector_index,
):
    response = client.post(
        "/vector-indexes/",
        headers=organization_headers(vector_index),
        json={
            "name": "legal-index",
            "profile_name": "missing-profile",
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == (
        "Embedding profile 'missing-profile' not found"
    )


def test_create_vector_index_rejects_invalid_organization_id(
    client,
):
    response = client.post(
        "/vector-indexes/",
        headers={
            "X-Organization-ID": "not-a-uuid",
        },
        json={
            "name": "legal-index",
            "profile_name": "legal-general",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"] == "Invalid organization ID"


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


def test_get_vector_index(client, vector_index):
    response = client.get(
        f"/vector-indexes/{vector_index.id}",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(vector_index.id)
    assert data["name"] == "legal-index"
    assert data["organization_id"] == str(
        vector_index.organization_id,
    )
    assert data["profile_name"] == "legal-general"
    assert data["embedding_identity_id"] == str(
        vector_index.embedding_identity_id,
    )
    assert data["status"] == "building"

    assert data["created_at"] is not None
    assert data["updated_at"] is not None


def test_get_vector_index_not_found(client, vector_index):
    response = client.get(
        f"/vector-indexes/{uuid.uuid4()}",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Vector index not found"


def test_get_vector_index_wrong_organization(
    client,
    vector_index,
):
    response = client.get(
        f"/vector-indexes/{vector_index.id}",
        headers={
            "X-Organization-ID": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Vector index not found"


def test_get_vector_index_requires_organization_id(
    client,
    vector_index,
):
    response = client.get(
        f"/vector-indexes/{vector_index.id}",
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Active index
# ---------------------------------------------------------------------------


def test_get_active_vector_index(
    client,
    vector_index,
):
    vector_index.status = VectorIndexStatus.ACTIVE

    response = client.get(
        "/vector-indexes/active/legal-general",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(vector_index.id)
    assert data["profile_name"] == "legal-general"
    assert data["status"] == "active"


def test_get_active_vector_index_not_found(
    client,
    vector_index,
):
    response = client.get(
        "/vector-indexes/active/legal-general",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Active vector index not found"


def test_get_active_vector_index_requires_organization_id(
    client,
):
    response = client.get(
        "/vector-indexes/active/legal-general",
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_mark_vector_index_ready(
    client,
    vector_index,
):
    response = client.post(
        f"/vector-indexes/{vector_index.id}/ready",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(vector_index.id)
    assert data["status"] == "ready"
    assert vector_index.status == VectorIndexStatus.READY


def test_mark_vector_index_failed(
    client,
    vector_index,
):
    response = client.post(
        f"/vector-indexes/{vector_index.id}/fail",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(vector_index.id)
    assert data["status"] == "failed"
    assert vector_index.status == VectorIndexStatus.FAILED


def test_activate_vector_index(
    client,
    vector_index,
):
    vector_index.status = VectorIndexStatus.READY

    response = client.post(
        f"/vector-indexes/{vector_index.id}/activate",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(vector_index.id)
    assert data["status"] == "active"
    assert vector_index.status == VectorIndexStatus.ACTIVE


def test_deprecate_vector_index(
    client,
    vector_index,
):
    vector_index.status = VectorIndexStatus.ACTIVE

    response = client.post(
        f"/vector-indexes/{vector_index.id}/deprecate",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(vector_index.id)
    assert data["status"] == "deprecated"
    assert vector_index.status == VectorIndexStatus.DEPRECATED


@pytest.mark.parametrize(
    "endpoint",
    [
        "ready",
        "fail",
        "activate",
        "deprecate",
    ],
)
def test_lifecycle_endpoints_require_organization_id(
    client,
    vector_index,
    endpoint,
):
    response = client.post(
        f"/vector-indexes/{vector_index.id}/{endpoint}",
    )

    assert response.status_code == 422


def test_mark_ready_not_found(
    client,
    vector_index,
):
    response = client.post(
        f"/vector-indexes/{uuid.uuid4()}/ready",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Vector index not found"


def test_invalid_transition_returns_conflict(
    client,
    vector_index,
):
    vector_index.status = VectorIndexStatus.READY

    response = client.post(
        f"/vector-indexes/{vector_index.id}/ready",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 409

    data = response.json()

    assert data["detail"] == (
        "Invalid vector-index transition: ready -> ready"
    )


def test_activate_building_index_returns_conflict(
    client,
    vector_index,
):
    response = client.post(
        f"/vector-indexes/{vector_index.id}/activate",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 409

    data = response.json()

    assert data["detail"] == (
        "Invalid vector-index transition: building -> active"
    )


def test_deprecate_ready_index_returns_conflict(
    client,
    vector_index,
):
    vector_index.status = VectorIndexStatus.READY

    response = client.post(
        f"/vector-indexes/{vector_index.id}/deprecate",
        headers=organization_headers(vector_index),
    )

    assert response.status_code == 409

    data = response.json()

    assert data["detail"] == (
        "Invalid vector-index transition: ready -> deprecated"
    )
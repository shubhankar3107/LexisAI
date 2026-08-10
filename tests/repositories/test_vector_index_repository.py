from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.enums.vector_index_status import VectorIndexStatus
from app.models.embedding_identity import EmbeddingIdentity
from app.models.organization import Organization
from app.models.vector_index import VectorIndex
from app.repositories.vector_index_repository import VectorIndexRepository


def create_organization(
    session: Session,
) -> Organization:
    organization = Organization(
        name="Repository Test Organization",
        slug=f"repository-test-{uuid.uuid4()}",
    )

    session.add(organization)
    session.flush()

    return organization


def create_identity(
    session: Session,
    profile_name: str = "repository-test",
) -> EmbeddingIdentity:
    identity = EmbeddingIdentity(
        profile_name=profile_name,
        provider="test-provider",
        model="test-model",
        model_version="1",
        dimensions=1024,
        fingerprint=uuid.uuid4().hex,
    )

    session.add(identity)
    session.flush()

    return identity


def create_vector_index(
    session: Session,
    organization: Organization,
    identity: EmbeddingIdentity,
    status: VectorIndexStatus = VectorIndexStatus.BUILDING,
) -> VectorIndex:
    vector_index = VectorIndex(
        organization_id=organization.id,
        profile_name=identity.profile_name,
        embedding_identity_id=identity.id,
        name=f"repository-test-{uuid.uuid4()}",
        status=status,
    )

    session.add(vector_index)
    session.flush()

    return vector_index


def test_add_and_get_by_id(db_session: Session):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization,
        identity,
    )

    repository.add(vector_index)
    db_session.flush()

    result = repository.get_by_id(vector_index.id)

    assert result is vector_index
    assert result.id == vector_index.id


def test_get_by_id_enforces_organization_scope(
    db_session: Session,
):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    other_organization = create_organization(db_session)

    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization,
        identity,
    )

    result = repository.get_by_id(
        vector_index.id,
        organization_id=other_organization.id,
    )

    assert result is None


def test_get_active(db_session: Session):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization,
        identity,
        status=VectorIndexStatus.ACTIVE,
    )

    result = repository.get_active(
        organization.id,
        identity.profile_name,
    )

    assert result is vector_index


def test_list_by_organization(db_session: Session):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    identity = create_identity(db_session)

    first = create_vector_index(
        db_session,
        organization,
        identity,
        status=VectorIndexStatus.DEPRECATED,
    )

    second_identity = create_identity(db_session)

    second = create_vector_index(
        db_session,
        organization,
        second_identity,
        status=VectorIndexStatus.READY,
    )

    result = repository.list_by_organization(
        organization.id,
    )

    result_ids = {index.id for index in result}

    assert first.id in result_ids
    assert second.id in result_ids


def test_list_by_organization_filters_by_profile(
    db_session: Session,
):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)

    legal_identity = create_identity(
        db_session,
        profile_name="legal",
    )

    general_identity = create_identity(
        db_session,
        profile_name="general",
    )

    legal_index = create_vector_index(
        db_session,
        organization,
        legal_identity,
    )

    general_index = create_vector_index(
        db_session,
        organization,
        general_identity,
    )

    result = repository.list_by_organization(
        organization.id,
        profile_name="legal",
    )

    result_ids = {index.id for index in result}

    assert legal_index.id in result_ids
    assert general_index.id not in result_ids


def test_mark_ready(db_session: Session):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization,
        identity,
    )

    result = repository.mark_ready(vector_index)

    assert result is vector_index
    assert result.status == VectorIndexStatus.READY


def test_mark_deprecated(db_session: Session):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization,
        identity,
        status=VectorIndexStatus.READY,
    )

    result = repository.mark_deprecated(vector_index)

    assert result is vector_index
    assert result.status == VectorIndexStatus.DEPRECATED


def test_mark_failed(db_session: Session):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization,
        identity,
        status=VectorIndexStatus.BUILDING,
    )

    result = repository.mark_failed(vector_index)

    assert result is vector_index
    assert result.status == VectorIndexStatus.FAILED


def test_activate_deprecates_existing_active_index(
    db_session: Session,
):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)

    identity_one = create_identity(db_session)
    identity_two = create_identity(db_session)

    current = create_vector_index(
        db_session,
        organization,
        identity_one,
        status=VectorIndexStatus.ACTIVE,
    )

    replacement = create_vector_index(
        db_session,
        organization,
        identity_two,
        status=VectorIndexStatus.READY,
    )

    result = repository.activate(replacement)
    db_session.flush()

    assert result is replacement
    assert replacement.status == VectorIndexStatus.ACTIVE
    assert current.status == VectorIndexStatus.DEPRECATED


def test_activate_without_existing_active_index(
    db_session: Session,
):
    repository = VectorIndexRepository(db_session)

    organization = create_organization(db_session)
    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization,
        identity,
        status=VectorIndexStatus.READY,
    )

    result = repository.activate(vector_index)

    assert result is vector_index
    assert vector_index.status == VectorIndexStatus.ACTIVE
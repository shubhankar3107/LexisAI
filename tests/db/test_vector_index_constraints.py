from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.enums.vector_index_status import VectorIndexStatus
from app.models.embedding_identity import EmbeddingIdentity
from app.models.organization import Organization
from app.models.vector_index import VectorIndex


@pytest.fixture
def db_session() -> Session:
    engine = create_engine(
        settings.test_database_url,
        pool_pre_ping=True,
    )

    with engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

    if database_name != "lexisai_test":
        engine.dispose()
        pytest.fail(
            f"Refusing to run database tests against "
            f"unexpected database: {database_name!r}"
        )

    session = Session(
        bind=engine,
        expire_on_commit=False,
    )

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def create_organization(session: Session) -> Organization:
    organization = Organization(
        name="Test Organization",
        slug=f"test-{uuid.uuid4()}",
    )

    session.add(organization)
    session.flush()

    return organization


def create_identity(
    session: Session,
    *,
    profile_name: str = "legal-general",
    fingerprint: str | None = None,
) -> EmbeddingIdentity:
    identity = EmbeddingIdentity(
        profile_name=profile_name,
        provider="test-provider",
        model="test-model",
        model_version="1",
        dimensions=1024,
        fingerprint=fingerprint or uuid.uuid4().hex,
    )

    session.add(identity)
    session.flush()

    return identity


def create_vector_index(
    session: Session,
    *,
    organization: Organization,
    identity: EmbeddingIdentity,
    status: VectorIndexStatus = VectorIndexStatus.BUILDING,
    name: str | None = None,
) -> VectorIndex:
    vector_index = VectorIndex(
        organization_id=organization.id,
        profile_name=identity.profile_name,
        embedding_identity_id=identity.id,
        name=name or f"index-{uuid.uuid4()}",
        status=status,
    )

    session.add(vector_index)
    session.flush()

    return vector_index


def test_embedding_identity_fingerprint_must_be_unique(
    db_session: Session,
):
    fingerprint = uuid.uuid4().hex

    create_identity(
        db_session,
        fingerprint=fingerprint,
    )

    duplicate = EmbeddingIdentity(
        profile_name="legal-general",
        provider="test-provider",
        model="test-model",
        model_version="1",
        dimensions=1024,
        fingerprint=fingerprint,
    )

    db_session.add(duplicate)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_vector_index_requires_matching_identity_profile(
    db_session: Session,
):
    organization = create_organization(db_session)

    identity = create_identity(
        db_session,
        profile_name="legal-general",
    )

    vector_index = VectorIndex(
        organization_id=organization.id,
        profile_name="different-profile",
        embedding_identity_id=identity.id,
        name="invalid-profile-index",
        status=VectorIndexStatus.BUILDING,
    )

    db_session.add(vector_index)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_only_one_active_index_per_organization_and_profile(
    db_session: Session,
):
    organization = create_organization(db_session)

    identity_one = create_identity(
        db_session,
        profile_name="legal-general",
    )

    identity_two = create_identity(
        db_session,
        profile_name="legal-general",
    )

    create_vector_index(
        db_session,
        organization=organization,
        identity=identity_one,
        status=VectorIndexStatus.ACTIVE,
        name="active-index-one",
    )

    duplicate_active_index = VectorIndex(
        organization_id=organization.id,
        profile_name="legal-general",
        embedding_identity_id=identity_two.id,
        name="active-index-two",
        status=VectorIndexStatus.ACTIVE,
    )

    db_session.add(duplicate_active_index)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_multiple_non_active_indexes_are_allowed(
    db_session: Session,
):
    organization = create_organization(db_session)

    identity_one = create_identity(
        db_session,
        profile_name="legal-general",
    )

    identity_two = create_identity(
        db_session,
        profile_name="legal-general",
    )

    first = create_vector_index(
        db_session,
        organization=organization,
        identity=identity_one,
        status=VectorIndexStatus.DEPRECATED,
        name="deprecated-index-one",
    )

    second = create_vector_index(
        db_session,
        organization=organization,
        identity=identity_two,
        status=VectorIndexStatus.READY,
        name="ready-index-two",
    )

    assert first.id != second.id


def test_different_profiles_can_each_have_one_active_index(
    db_session: Session,
):
    organization = create_organization(db_session)

    legal_identity = create_identity(
        db_session,
        profile_name="legal-general",
    )

    multilingual_identity = create_identity(
        db_session,
        profile_name="multilingual",
    )

    legal_index = create_vector_index(
        db_session,
        organization=organization,
        identity=legal_identity,
        status=VectorIndexStatus.ACTIVE,
        name="legal-active",
    )

    multilingual_index = create_vector_index(
        db_session,
        organization=organization,
        identity=multilingual_identity,
        status=VectorIndexStatus.ACTIVE,
        name="multilingual-active",
    )

    assert legal_index.id != multilingual_index.id


def test_deleting_organization_cascades_to_vector_indexes(
    db_session: Session,
):
    organization = create_organization(db_session)

    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization=organization,
        identity=identity,
    )

    vector_index_id = vector_index.id
    organization_id = organization.id

    db_session.delete(organization)
    db_session.flush()

    remaining = db_session.get(
        VectorIndex,
        vector_index_id,
    )

    assert remaining is None

    assert db_session.get(
        Organization,
        organization_id,
    ) is None


def test_embedding_identity_cannot_be_deleted_while_referenced(
    db_session: Session,
):
    organization = create_organization(db_session)

    identity = create_identity(db_session)

    create_vector_index(
        db_session,
        organization=organization,
        identity=identity,
    )

    db_session.delete(identity)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_vector_index_status_uses_lowercase_database_values(
    db_session: Session,
):
    organization = create_organization(db_session)

    identity = create_identity(db_session)

    vector_index = create_vector_index(
        db_session,
        organization=organization,
        identity=identity,
        status=VectorIndexStatus.ACTIVE,
    )

    stored_status = db_session.execute(
        text(
            """
            SELECT status
            FROM vector_indexes
            WHERE id = :id
            """
        ),
        {"id": vector_index.id},
    ).scalar_one()

    assert stored_status == "active"
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.embedding_identity import EmbeddingIdentity
from app.repositories.embedding_identity_repository import (
    EmbeddingIdentityRepository,
)


def test_add_and_get_by_id(db_session: Session):
    repository = EmbeddingIdentityRepository(db_session)

    identity = EmbeddingIdentity(
        profile_name="repository-test",
        provider="test-provider",
        model="test-model",
        model_version="1",
        dimensions=1024,
        fingerprint=uuid.uuid4().hex,
    )

    repository.add(identity)
    db_session.flush()

    result = repository.get_by_id(identity.id)

    assert result is identity
    assert result.id == identity.id


def test_get_by_fingerprint(db_session: Session):
    repository = EmbeddingIdentityRepository(db_session)

    fingerprint = uuid.uuid4().hex

    identity = EmbeddingIdentity(
        profile_name="repository-test",
        provider="test-provider",
        model="test-model",
        model_version="1",
        dimensions=1024,
        fingerprint=fingerprint,
    )

    repository.add(identity)
    db_session.flush()

    result = repository.get_by_fingerprint(fingerprint)

    assert result is identity
    assert result.fingerprint == fingerprint
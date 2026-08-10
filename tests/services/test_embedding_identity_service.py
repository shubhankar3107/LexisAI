from __future__ import annotations

import uuid

from app.models.embedding_identity import EmbeddingIdentity
from app.services.embedding_identity_service import (
    EmbeddingIdentityService,
)
from app.services.schemas.embedding import EmbeddingIdentitySpec


class FakeEmbeddingIdentityRepository:
    def __init__(
        self,
        identity: EmbeddingIdentity | None = None,
    ):
        self.identity = identity
        self.add_called = False

    def get_by_fingerprint(
        self,
        fingerprint: str,
    ) -> EmbeddingIdentity | None:
        if (
            self.identity is not None
            and self.identity.fingerprint == fingerprint
        ):
            return self.identity

        return None

    def add(
        self,
        identity: EmbeddingIdentity,
    ) -> EmbeddingIdentity:
        self.add_called = True
        self.identity = identity
        return identity


def make_spec() -> EmbeddingIdentitySpec:
    return EmbeddingIdentitySpec(
        profile_name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v1",
        dimensions=4096,
    )


def make_identity(
    specification: EmbeddingIdentitySpec,
) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        id=uuid.uuid4(),
        profile_name=specification.profile_name,
        provider=specification.provider,
        model=specification.model,
        model_version=specification.model_version,
        dimensions=specification.dimensions,
        fingerprint=specification.fingerprint,
    )


def make_service(
    identity: EmbeddingIdentity | None = None,
):
    repository = FakeEmbeddingIdentityRepository(
        identity=identity,
    )

    service = EmbeddingIdentityService(
        embedding_identity_repository=repository,
    )

    return service, repository


def test_resolve_returns_existing_identity():
    specification = make_spec()
    existing_identity = make_identity(specification)

    service, repository = make_service(
        existing_identity,
    )

    result = service.resolve(specification)

    assert result is existing_identity
    assert repository.add_called is False


def test_resolve_creates_identity_when_not_found():
    specification = make_spec()

    service, repository = make_service()

    result = service.resolve(specification)

    assert isinstance(result, EmbeddingIdentity)

    assert result.profile_name == "legal-general"
    assert result.provider == "huggingface"
    assert result.model == "Qwen3-Embedding-8B"
    assert result.model_version == "v1"
    assert result.dimensions == 4096
    assert result.fingerprint == specification.fingerprint

    assert repository.add_called is True


def test_resolve_uses_specification_fingerprint():
    specification = make_spec()

    service, repository = make_service()

    result = service.resolve(specification)

    assert result.fingerprint == specification.fingerprint
    assert repository.identity is result


def test_resolve_reuses_same_identity_for_same_specification():
    specification = make_spec()

    existing_identity = make_identity(specification)

    service, repository = make_service(
        existing_identity,
    )

    first = service.resolve(specification)
    second = service.resolve(specification)

    assert first is existing_identity
    assert second is existing_identity

    assert repository.add_called is False


def test_resolve_creates_different_identity_for_different_specification():
    first_specification = make_spec()

    second_specification = EmbeddingIdentitySpec(
        profile_name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v2",
        dimensions=4096,
    )

    service, repository = make_service()

    first = service.resolve(first_specification)

    repository.identity = None
    repository.add_called = False

    second = service.resolve(second_specification)

    assert first.id != second.id
    assert first.fingerprint != second.fingerprint

    assert repository.add_called is True
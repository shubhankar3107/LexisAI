from __future__ import annotations

import uuid

from app.models.embedding_identity import EmbeddingIdentity
from app.repositories.embedding_identity_repository import (
    EmbeddingIdentityRepository,
)
from app.services.schemas.embedding import EmbeddingIdentitySpec


class EmbeddingIdentityService:
    """Resolve and persist immutable embedding identities."""

    def __init__(
        self,
        embedding_identity_repository: EmbeddingIdentityRepository,
    ):
        self._embedding_identity_repository = (
            embedding_identity_repository
        )

    def resolve(
        self,
        specification: EmbeddingIdentitySpec,
    ) -> EmbeddingIdentity:
        """Return the persistent identity for an embedding specification."""

        fingerprint = specification.fingerprint

        existing = (
            self._embedding_identity_repository.get_by_fingerprint(
                fingerprint,
            )
        )

        if existing is not None:
            return existing

        identity = EmbeddingIdentity(
            id=uuid.uuid4(),
            profile_name=specification.profile_name,
            provider=specification.provider,
            model=specification.model,
            model_version=specification.model_version,
            dimensions=specification.dimensions,
            fingerprint=fingerprint,
        )

        self._embedding_identity_repository.add(identity)

        return identity
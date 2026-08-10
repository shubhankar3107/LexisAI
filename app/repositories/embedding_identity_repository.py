from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.embedding_identity import EmbeddingIdentity


class EmbeddingIdentityRepository:
    """Persistence operations for immutable embedding identities."""

    def __init__(self, session: Session):
        self._session = session

    def add(
        self,
        identity: EmbeddingIdentity,
    ) -> EmbeddingIdentity:
        self._session.add(identity)
        return identity

    def get_by_id(
        self,
        identity_id: uuid.UUID,
    ) -> EmbeddingIdentity | None:
        statement = select(EmbeddingIdentity).where(
            EmbeddingIdentity.id == identity_id,
        )

        result = self._session.execute(statement)

        return result.scalar_one_or_none()

    def get_by_fingerprint(
        self,
        fingerprint: str,
    ) -> EmbeddingIdentity | None:
        statement = select(EmbeddingIdentity).where(
            EmbeddingIdentity.fingerprint == fingerprint,
        )

        result = self._session.execute(statement)

        return result.scalar_one_or_none()
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.enums.vector_index_status import VectorIndexStatus
from app.models.vector_index import VectorIndex


class VectorIndexRepository:
    """Persistence operations for vector-index lifecycle management."""

    def __init__(self, session: Session):
        self._session = session

    def add(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        self._session.add(vector_index)
        return vector_index

    def get_by_id(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID | None = None,
    ) -> VectorIndex | None:
        statement = select(VectorIndex).where(
            VectorIndex.id == index_id,
        )

        if organization_id is not None:
            statement = statement.where(
                VectorIndex.organization_id == organization_id,
            )

        result = self._session.execute(statement)

        return result.scalar_one_or_none()

    def get_active(
        self,
        organization_id: uuid.UUID,
        profile_name: str,
    ) -> VectorIndex | None:
        statement = select(VectorIndex).where(
            VectorIndex.organization_id == organization_id,
            VectorIndex.profile_name == profile_name,
            VectorIndex.status == VectorIndexStatus.ACTIVE,
        )

        result = self._session.execute(statement)

        return result.scalar_one_or_none()

    def list_by_organization(
        self,
        organization_id: uuid.UUID,
        profile_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[VectorIndex]:
        statement = select(VectorIndex).where(
            VectorIndex.organization_id == organization_id,
        )

        if profile_name is not None:
            statement = statement.where(
                VectorIndex.profile_name == profile_name,
            )

        statement = (
            statement
            .order_by(
                VectorIndex.created_at.desc(),
                VectorIndex.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = self._session.execute(statement)

        return result.scalars().all()

    def mark_ready(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        vector_index.status = VectorIndexStatus.READY

        return vector_index

    def activate(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        current_active_statement = (
            select(VectorIndex)
            .where(
                VectorIndex.organization_id
                == vector_index.organization_id,
                VectorIndex.profile_name
                == vector_index.profile_name,
                VectorIndex.status
                == VectorIndexStatus.ACTIVE,
            )
            .with_for_update()
        )

        current_active = self._session.execute(
            current_active_statement,
        ).scalar_one_or_none()

        if (
            current_active is not None
            and current_active.id != vector_index.id
        ):
            self._session.execute(
                VectorIndex.__table__.update()
                .where(VectorIndex.id == current_active.id)
                .values(status=VectorIndexStatus.DEPRECATED)
            )

            self._session.expire(
                current_active,
                ["status"],
            )

        self._session.execute(
            VectorIndex.__table__.update()
            .where(VectorIndex.id == vector_index.id)
            .values(status=VectorIndexStatus.ACTIVE)
        )

        vector_index.status = VectorIndexStatus.ACTIVE

        return vector_index

    def mark_deprecated(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        vector_index.status = VectorIndexStatus.DEPRECATED

        return vector_index

    def mark_failed(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        vector_index.status = VectorIndexStatus.FAILED

        return vector_index
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.enums.vector_index_status import VectorIndexStatus


class VectorIndex(Base):
    __tablename__ = "vector_indexes"

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "embedding_identity_id",
                "profile_name",
            ],
            [
                "embedding_identities.id",
                "embedding_identities.profile_name",
            ],
            name="fk_vector_index_embedding_identity_profile",
            ondelete="RESTRICT",
        ),
        Index(
            "uq_active_vector_index_per_org_profile",
            "organization_id",
            "profile_name",
            unique=True,
            postgresql_where=text(
                "status = 'active'",
            ),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "organizations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    profile_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    embedding_identity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    status: Mapped[VectorIndexStatus] = mapped_column(
        Enum(
            VectorIndexStatus,
            name="vector_index_status",
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
        default=VectorIndexStatus.BUILDING,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    organization: Mapped["Organization"] = relationship(
        back_populates="vector_indexes",
    )

    embedding_identity: Mapped["EmbeddingIdentity"] = relationship(
        back_populates="vector_indexes",
    )
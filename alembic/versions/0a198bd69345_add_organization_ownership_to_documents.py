"""Add organization ownership to documents

Revision ID: 0a198bd69345
Revises: 32124ae2caa4
Create Date: 2026-08-09 14:44:38.303031

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0a198bd69345"
down_revision: Union[str, Sequence[str], None] = "32124ae2caa4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


LEGACY_ORGANIZATION_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    """Upgrade schema."""

    organizations = sa.table(
        "organizations",
        sa.column("id", sa.UUID()),
        sa.column("name", sa.String()),
        sa.column("slug", sa.String()),
    )

    op.execute(
        sa.insert(organizations).values(
            id=LEGACY_ORGANIZATION_ID,
            name="Legacy Organization",
            slug="legacy-organization",
        )
    )

    op.add_column(
        "documents",
        sa.Column(
            "organization_id",
            sa.UUID(),
            nullable=True,
        ),
    )

    op.execute(
        sa.text(
            """
            UPDATE documents
            SET organization_id = CAST(:organization_id AS UUID)
            WHERE organization_id IS NULL
            """
        ).bindparams(
            organization_id=LEGACY_ORGANIZATION_ID,
        )
    )
    
    op.alter_column(
        "documents",
        "organization_id",
        existing_type=sa.UUID(),
        nullable=False,
    )

    op.create_index(
        op.f("ix_documents_organization_id"),
        "documents",
        ["organization_id"],
        unique=False,
    )

    op.create_foreign_key(
        "fk_documents_organization_id_organizations",
        "documents",
        "organizations",
        ["organization_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_constraint(
        "fk_documents_organization_id_organizations",
        "documents",
        type_="foreignkey",
    )

    op.drop_index(
        op.f("ix_documents_organization_id"),
        table_name="documents",
    )

    op.drop_column(
        "documents",
        "organization_id",
    )

    op.execute(
        sa.text(
            """
            DELETE FROM organizations
            WHERE id = :organization_id
            """
        ).bindparams(
            organization_id=LEGACY_ORGANIZATION_ID,
        )
    )
"""make document chunk page number nullable

Revision ID: e8ed66902805
Revises: 65073d68f1d6
Create Date: 2026-08-10 03:58:01.604645

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e8ed66902805"
down_revision: Union[str, Sequence[str], None] = "65073d68f1d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "document_chunks",
        "page_number",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "document_chunks",
        "page_number",
        existing_type=sa.Integer(),
        nullable=False,
    )

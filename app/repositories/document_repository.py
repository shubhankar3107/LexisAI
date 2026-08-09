import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document


class DocumentRepository:

    def __init__(self, session: Session):
        self._session = session

    def add(
        self,
        document: Document,
    ) -> Document:
        self._session.add(document)
        return document

    def get_by_id(
        self,
        document_id: uuid.UUID,
    ) -> Document | None:
        statement = select(Document).where(
            Document.id == document_id,
            Document.deleted_at.is_(None),
        )

        result = self._session.execute(statement)

        return result.scalar_one_or_none()

    def list_by_title(
        self,
        title: str,
    ) -> list[Document]:
        statement = select(Document).where(
            Document.title == title,
            Document.deleted_at.is_(None),
        )

        result = self._session.execute(statement)

        return result.scalars().all()

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        statement = (
            select(Document)
            .where(Document.deleted_at.is_(None))
            .limit(limit)
            .offset(offset)
        )

        result = self._session.execute(statement)

        return result.scalars().all()

    def count(self) -> int:
        statement = (
            select(func.count())
            .select_from(Document)
            .where(Document.deleted_at.is_(None))
        )

        result = self._session.execute(statement)

        return result.scalar_one()

    def delete(
        self,
        document: Document,
    ) -> None:
        document.deleted_at = datetime.now(
            timezone.utc,
        )

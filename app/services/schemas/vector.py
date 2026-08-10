from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """
    A vector and its metadata as persisted by a vector store.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    organization_id: uuid.UUID

    page_number: int = Field(
        ge=1,
    )

    chunk_index: int = Field(
        ge=0,
    )

    content: str

    vector: list[float]
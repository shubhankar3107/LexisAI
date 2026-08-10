from __future__ import annotations

from dataclasses import dataclass, field
import uuid


@dataclass(frozen=True)
class RAGRequest:
    """Application-level request for a retrieval-augmented answer."""

    query: str
    organization_id: uuid.UUID
    top_k: int = 10
    document_id: uuid.UUID | None = None
    filters: dict[str, object] = field(
        default_factory=dict,
    )


@dataclass(frozen=True)
class RAGSource:
    """Source metadata attached to a generated answer."""

    document_id: str
    chunk_id: str
    page_number: int
    chunk_index: int
    score: float


@dataclass(frozen=True)
class RAGResponse:
    """Application-level response containing an answer and its sources."""

    answer: str
    sources: list[RAGSource] = field(
        default_factory=list,
    )
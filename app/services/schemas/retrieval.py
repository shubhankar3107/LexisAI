from dataclasses import dataclass, field
import uuid


@dataclass(frozen=True)
class RetrievalResult:
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    page_number: int
    chunk_index: int
    content: str
    score: float


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    organization_id: uuid.UUID
    top_k: int = 10
    document_id: uuid.UUID | None = None
    filters: dict[str, object] = field(default_factory=dict)
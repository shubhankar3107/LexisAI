import uuid
from datetime import datetime

from pydantic import BaseModel

from app.enums.document_status import DocumentStatus


class DocumentAssetResponse(BaseModel):
    original_filename: str
    stored_filename: str
    mime_type: str
    file_size: int
    checksum: str


class DocumentResponse(BaseModel):
    id: uuid.UUID
    title: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    asset: DocumentAssetResponse


class DocumentListItemResponse(BaseModel):
    id: uuid.UUID
    title: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentListItemResponse]
    limit: int
    offset: int
    total: int

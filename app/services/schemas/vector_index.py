from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.enums.vector_index_status import VectorIndexStatus


class CreateVectorIndexRequest(BaseModel):
    name: str
    profile_name: str


class VectorIndexResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    organization_id: uuid.UUID
    profile_name: str
    embedding_identity_id: uuid.UUID
    status: VectorIndexStatus
    created_at: datetime
    updated_at: datetime
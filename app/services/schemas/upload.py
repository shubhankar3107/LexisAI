import uuid

from pydantic import BaseModel, Field


class UploadDocumentInput(BaseModel):
    filename: str
    content_type: str


class UploadDocumentResponse(BaseModel):
    document_id: uuid.UUID
    stored_filename: str
    storage_path: str
    checksum: str

from fastapi import APIRouter, Depends, File, UploadFile
import uuid
from app.api.dependencies import get_upload_service, get_current_organization_id
from app.services.schemas.upload import (
    UploadDocumentInput,
    UploadDocumentResponse,
)
from app.services.upload_service import UploadService


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.post(
    "/",
    response_model=UploadDocumentResponse,
    status_code=201,
)
def upload_document(
    file: UploadFile = File(...),
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: UploadService = Depends(get_upload_service),
):
    input_data = UploadDocumentInput(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
    )

    document_id, stored_filename, storage_path, checksum = service.upload(
        input_data,
        file.file,
        organization_id,
    )

    return UploadDocumentResponse(
        document_id=document_id,
        stored_filename=stored_filename,
        storage_path=storage_path,
        checksum=checksum,
    )

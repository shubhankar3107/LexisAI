import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    get_current_organization_id,
    get_document_processing_service,
    get_document_service,
)
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService
from app.services.exceptions import (
    DocumentFileNotFoundError,
    DocumentNotFoundError,
    DocumentProcessingStateError,
)
from app.services.schemas.document import (
    DocumentAssetResponse,
    DocumentListItemResponse,
    DocumentListResponse,
    DocumentResponse,
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    document_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: DocumentService = Depends(get_document_service),
):
    try:
        document = service.get_document(
            document_id,
            organization_id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from exc

    return DocumentResponse(
        id=document.id,
        title=document.title,
        status=document.status,
        created_at=document.created_at,
        updated_at=document.updated_at,
        asset=DocumentAssetResponse(
            original_filename=document.asset.original_filename,
            stored_filename=document.asset.stored_filename,
            mime_type=document.asset.mime_type,
            file_size=document.asset.file_size,
            checksum=document.asset.checksum,
        ),
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: DocumentService = Depends(get_document_service),
):
    try:
        service.delete_document(
            document_id,
            organization_id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from exc


@router.get(
    "/",
    response_model=DocumentListResponse,
)
def list_documents(
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: DocumentService = Depends(get_document_service),
):
    documents, total = service.list_documents(
        organization_id,
        limit=limit,
        offset=offset,
    )

    return DocumentListResponse(
        items=[
            DocumentListItemResponse(
                id=document.id,
                title=document.title,
                status=document.status,
                created_at=document.created_at,
                updated_at=document.updated_at,
            )
            for document in documents
        ],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.post(
    "/{document_id}/process",
)
def process_document(
    document_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: DocumentProcessingService = Depends(
        get_document_processing_service,
    ),
):
    try:
        document = service.process_document(
            document_id,
            organization_id,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from exc

    except DocumentProcessingStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return {
        "document_id": document.id,
        "status": document.status,
    }


@router.get(
    "/{document_id}/download",
)
def download_document(
    document_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: DocumentService = Depends(get_document_service),
):
    try:
        document, file = service.download_document(
            document_id,
            organization_id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from exc
    except DocumentFileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found",
        ) from exc

    return StreamingResponse(
        file,
        media_type=document.asset.mime_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{document.asset.original_filename}"'
            )
        },
    )
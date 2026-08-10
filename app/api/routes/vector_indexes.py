from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_current_organization_id,
    get_vector_index_service,
)
from app.services.embedding_registry import (
    EmbeddingProfileNotFoundError,
)
from app.services.schemas.vector_index import (
    CreateVectorIndexRequest,
    VectorIndexResponse,
)
from app.services.vector_index_service import (
    IndexNotFoundError,
    InvalidIndexTransitionError,
    VectorIndexService,
)


router = APIRouter(
    prefix="/vector-indexes",
    tags=["vector-indexes"],
)


def _to_response(
    vector_index,
) -> VectorIndexResponse:
    return VectorIndexResponse(
        id=vector_index.id,
        name=vector_index.name,
        organization_id=vector_index.organization_id,
        profile_name=vector_index.profile_name,
        embedding_identity_id=vector_index.embedding_identity_id,
        status=vector_index.status,
        created_at=vector_index.created_at,
        updated_at=vector_index.updated_at,
    )


@router.post(
    "/",
    response_model=VectorIndexResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_vector_index(
    payload: CreateVectorIndexRequest,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: VectorIndexService = Depends(
        get_vector_index_service,
    ),
):
    try:
        vector_index = service.create_index(
            organization_id=organization_id,
            name=payload.name,
            profile_name=payload.profile_name,
        )

    except EmbeddingProfileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return _to_response(vector_index)


@router.get(
    "/{index_id}",
    response_model=VectorIndexResponse,
)
def get_vector_index(
    index_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: VectorIndexService = Depends(
        get_vector_index_service,
    ),
):
    try:
        vector_index = service.get_index(
            index_id=index_id,
            organization_id=organization_id,
        )

    except IndexNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector index not found",
        ) from exc

    return _to_response(vector_index)


@router.get(
    "/active/{profile_name}",
    response_model=VectorIndexResponse,
)
def get_active_vector_index(
    profile_name: str,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: VectorIndexService = Depends(
        get_vector_index_service,
    ),
):
    vector_index = service.get_active_index(
        organization_id=organization_id,
        profile_name=profile_name,
    )

    if vector_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active vector index not found",
        )

    return _to_response(vector_index)


@router.post(
    "/{index_id}/ready",
    response_model=VectorIndexResponse,
)
def mark_vector_index_ready(
    index_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: VectorIndexService = Depends(
        get_vector_index_service,
    ),
):
    try:
        vector_index = service.mark_ready(
            index_id=index_id,
            organization_id=organization_id,
        )

    except IndexNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector index not found",
        ) from exc

    except InvalidIndexTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _to_response(vector_index)


@router.post(
    "/{index_id}/fail",
    response_model=VectorIndexResponse,
)
def mark_vector_index_failed(
    index_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: VectorIndexService = Depends(
        get_vector_index_service,
    ),
):
    try:
        vector_index = service.mark_failed(
            index_id=index_id,
            organization_id=organization_id,
        )

    except IndexNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector index not found",
        ) from exc

    except InvalidIndexTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _to_response(vector_index)


@router.post(
    "/{index_id}/activate",
    response_model=VectorIndexResponse,
)
def activate_vector_index(
    index_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: VectorIndexService = Depends(
        get_vector_index_service,
    ),
):
    try:
        vector_index = service.activate_index(
            index_id=index_id,
            organization_id=organization_id,
        )

    except IndexNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector index not found",
        ) from exc

    except InvalidIndexTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _to_response(vector_index)


@router.post(
    "/{index_id}/deprecate",
    response_model=VectorIndexResponse,
)
def deprecate_vector_index(
    index_id: uuid.UUID,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    service: VectorIndexService = Depends(
        get_vector_index_service,
    ),
):
    try:
        vector_index = service.deprecate_index(
            index_id=index_id,
            organization_id=organization_id,
        )

    except IndexNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vector index not found",
        ) from exc

    except InvalidIndexTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _to_response(vector_index)
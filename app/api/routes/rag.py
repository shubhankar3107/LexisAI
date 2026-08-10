from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_current_organization_id,
    get_rag_orchestrator,
)
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.schemas.rag import (
    RAGRequest,
    RAGResponse,
)


router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)


@router.post(
    "/query",
    response_model=RAGResponse,
)
def query_rag(
    request: RAGRequest,
    organization_id: uuid.UUID = Depends(
        get_current_organization_id,
    ),
    orchestrator: RAGOrchestrator = Depends(
        get_rag_orchestrator,
    ),
) -> RAGResponse:
    if request.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization mismatch",
        )

    return orchestrator.answer(
        request,
    )
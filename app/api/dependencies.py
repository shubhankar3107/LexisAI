from __future__ import annotations

import uuid

from fastapi import (
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.config.embeddings import EMBEDDING_PROFILES
from app.core.config import settings
from app.db.session import get_db
from app.db.unit_of_work import UnitOfWork

from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.repositories.embedding_identity_repository import (
    EmbeddingIdentityRepository,
)
from app.repositories.vector_index_repository import (
    VectorIndexRepository,
)

from app.services.checksum import ChecksumCalculator
from app.services.context_assembler import ContextAssembler
from app.services.document_chunker import DocumentChunker
from app.services.document_indexing_service import (
    DocumentIndexingService,
)
from app.services.document_processing_service import (
    DocumentProcessingService,
)
from app.services.document_service import DocumentService
from app.services.embedding_identity_service import (
    EmbeddingIdentityService,
)
from app.services.embedding_provider import EmbeddingProvider
from app.services.embedding_registry import EmbeddingRegistry
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.pdf_text_extractor import PdfTextExtractor
from app.services.prompt_builder import PromptBuilder
from app.services.qdrant_vector_store import QdrantVectorStore
from app.services.retrieval_service import RetrievalService
from app.services.retriever import Retriever, SemanticRetriever
from app.services.reranker import Reranker, ScoreReranker
from app.services.sentence_transformer_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)
from app.services.upload_service import UploadService
from app.services.vector_index_service import VectorIndexService
from app.services.vector_store import VectorStore

from app.storage.local_file_storage import LocalFileStorage
from app.storage.storage_path import StoragePathBuilder
from app.services.schemas.llm_config import (
    LLMProviderConfig,
)

from app.services.llm_provider import LLMProvider
from app.services.llm_provider_registry import (
    LLMProviderRegistry,
)

from app.services.rag_orchestrator import RAGOrchestrator

def get_current_organization_id(
    x_organization_id: str = Header(
        ...,
        alias="X-Organization-ID",
    ),
) -> uuid.UUID:
    try:
        return uuid.UUID(x_organization_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid organization ID",
        ) from exc


def get_upload_service(
    db: Session = Depends(get_db),
) -> UploadService:
    document_repository = DocumentRepository(db)

    unit_of_work = UnitOfWork(db)

    file_storage = LocalFileStorage(
        settings.upload_directory,
    )

    storage_path_builder = StoragePathBuilder()

    checksum_calculator = ChecksumCalculator()

    return UploadService(
        document_repository=document_repository,
        file_storage=file_storage,
        storage_path_builder=storage_path_builder,
        checksum_calculator=checksum_calculator,
        unit_of_work=unit_of_work,
    )


def get_document_service(
    db: Session = Depends(get_db),
) -> DocumentService:
    document_repository = DocumentRepository(db)

    unit_of_work = UnitOfWork(db)

    file_storage = LocalFileStorage(
        settings.upload_directory,
    )

    return DocumentService(
        document_repository=document_repository,
        unit_of_work=unit_of_work,
        file_storage=file_storage,
    )


def get_document_processing_service(
    db: Session = Depends(get_db),
) -> DocumentProcessingService:
    document_repository = DocumentRepository(db)

    document_chunk_repository = DocumentChunkRepository(db)

    file_storage = LocalFileStorage(
        settings.upload_directory,
    )

    text_extractor = PdfTextExtractor()

    chunker = DocumentChunker()

    unit_of_work = UnitOfWork(db)

    return DocumentProcessingService(
        document_repository=document_repository,
        document_chunk_repository=document_chunk_repository,
        file_storage=file_storage,
        text_extractor=text_extractor,
        chunker=chunker,
        unit_of_work=unit_of_work,
    )


def get_embedding_provider() -> EmbeddingProvider:
    profile = EMBEDDING_PROFILES["legal-general"]

    return SentenceTransformerEmbeddingProvider(
        model_name=profile.model,
        dimensions=profile.dimensions,
        device=settings.embedding_device,
    )


def get_embedding_service(
    embedding_provider: EmbeddingProvider = Depends(
        get_embedding_provider,
    ),
) -> EmbeddingService:
    return EmbeddingService(
        embedding_provider=embedding_provider,
    )


def get_vector_index_service(
    db: Session = Depends(get_db),
) -> VectorIndexService:
    vector_index_repository = VectorIndexRepository(db)

    embedding_registry = EmbeddingRegistry(
        EMBEDDING_PROFILES,
    )

    embedding_identity_repository = EmbeddingIdentityRepository(
        db,
    )

    embedding_identity_service = EmbeddingIdentityService(
        embedding_identity_repository=embedding_identity_repository,
    )

    unit_of_work = UnitOfWork(db)

    return VectorIndexService(
        vector_index_repository=vector_index_repository,
        embedding_registry=embedding_registry,
        embedding_identity_service=embedding_identity_service,
        unit_of_work=unit_of_work,
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )


def get_vector_store(
    client: QdrantClient = Depends(
        get_qdrant_client,
    ),
) -> VectorStore:
    return QdrantVectorStore(
        client=client,
        collection_prefix=settings.qdrant_collection_prefix,
    )


def get_document_indexing_service(
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(
        get_embedding_service,
    ),
    vector_store: VectorStore = Depends(
        get_vector_store,
    ),
) -> DocumentIndexingService:
    document_repository = DocumentRepository(db)

    document_chunk_repository = DocumentChunkRepository(db)

    return DocumentIndexingService(
        document_repository=document_repository,
        document_chunk_repository=document_chunk_repository,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def get_retrieval_service(
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(
        get_embedding_service,
    ),
    vector_store: VectorStore = Depends(
        get_vector_store,
    ),
) -> RetrievalService:
    vector_index_repository = VectorIndexRepository(db)

    return RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        vector_index_repository=vector_index_repository,
        profile_name="legal-general",
    )


def get_retriever(
    retrieval_service: RetrievalService = Depends(
        get_retrieval_service,
    ),
) -> Retriever:
    return SemanticRetriever(
        retrieval_service=retrieval_service,
    )


def get_reranker() -> Reranker:
    return ScoreReranker()


def get_context_assembler() -> ContextAssembler:
    return ContextAssembler()


def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


def get_llm_provider(
    request: Request,
) -> LLMProvider:
    registry = getattr(
        request.app.state,
        "llm_provider_registry",
        None,
    )

    if not isinstance(
        registry,
        LLMProviderRegistry,
    ):
        raise RuntimeError(
            "LLM provider registry is not initialized",
        )

    config = get_llm_provider_config()

    return registry.get(
        config.provider,
    )


def get_llm_service(
    provider: LLMProvider = Depends(
        get_llm_provider,
    ),
) -> LLMService:
    return LLMService(
        llm_provider=provider,
    )

def get_llm_provider_config() -> LLMProviderConfig:
    if settings.llm_provider is None:
        raise RuntimeError(
            "LLM provider is not configured",
        )

    return LLMProviderConfig(
        provider=settings.llm_provider,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )


def get_rag_orchestrator(
    retriever: Retriever = Depends(
        get_retriever,
    ),
    reranker: Reranker = Depends(
        get_reranker,
    ),
    context_assembler: ContextAssembler = Depends(
        get_context_assembler,
    ),
    prompt_builder: PromptBuilder = Depends(
        get_prompt_builder,
    ),
    llm_service: LLMService = Depends(
        get_llm_service,
    ),
) -> RAGOrchestrator:
    return RAGOrchestrator(
        retriever=retriever,
        reranker=reranker,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )
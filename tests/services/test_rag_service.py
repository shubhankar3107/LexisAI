from __future__ import annotations

import uuid

from qdrant_client import QdrantClient

from app.config.embeddings import EMBEDDING_PROFILES
from app.core.config import settings
from app.db.unit_of_work import UnitOfWork
from app.enums.document_status import DocumentStatus
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.organization import Organization
from app.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from app.repositories.document_repository import DocumentRepository
from app.repositories.embedding_identity_repository import (
    EmbeddingIdentityRepository,
)
from app.repositories.vector_index_repository import (
    VectorIndexRepository,
)
from app.services.context_assembler import ContextAssembler
from app.services.document_indexing_service import (
    DocumentIndexingService,
)
from app.services.embedding_identity_service import (
    EmbeddingIdentityService,
)
from app.services.embedding_registry import EmbeddingRegistry
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.ollama_llm_provider import OllamaLLMProvider
from app.services.prompt_builder import PromptBuilder
from app.services.qdrant_vector_store import QdrantVectorStore
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.retrieval_service import RetrievalService
from app.services.retriever import SemanticRetriever
from app.services.reranker import ScoreReranker
from app.services.schemas.llm_config import LLMProviderConfig
from app.services.schemas.rag import RAGRequest
from app.services.schemas.retrieval import RetrievalQuery
from app.services.vector_index_service import VectorIndexService


def test_real_rag_pipeline(db_session):
    organization = Organization(
        id=uuid.uuid4(),
        name="LexisAI Integration Test",
        slug=f"lexisai-{uuid.uuid4().hex}",
    )

    db_session.add(organization)
    db_session.flush()

    document = Document(
        id=uuid.uuid4(),
        organization_id=organization.id,
        title="Test Contract",
        status=DocumentStatus.READY,
    )

    db_session.add(document)
    db_session.flush()

    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=document.id,
            page_number=1,
            chunk_index=0,
            content=(
                "Either party may terminate this agreement "
                "by providing thirty days written notice."
            ),
        ),
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=document.id,
            page_number=2,
            chunk_index=1,
            content=(
                "All confidential information must be "
                "protected during and after the agreement."
            ),
        ),
    ]

    db_session.add_all(chunks)
    db_session.flush()

    document_repository = DocumentRepository(
        db_session,
    )

    chunk_repository = DocumentChunkRepository(
        db_session,
    )

    embedding_identity_repository = (
        EmbeddingIdentityRepository(
            db_session,
        )
    )

    vector_index_repository = VectorIndexRepository(
        db_session,
    )

    embedding_registry = EmbeddingRegistry(
        EMBEDDING_PROFILES,
    )

    embedding_identity_service = (
        EmbeddingIdentityService(
            embedding_identity_repository=(
                embedding_identity_repository
            ),
        )
    )

    unit_of_work = UnitOfWork(
        db_session,
    )

    vector_index_service = VectorIndexService(
        vector_index_repository=(
            vector_index_repository
        ),
        embedding_registry=embedding_registry,
        embedding_identity_service=(
            embedding_identity_service
        ),
        unit_of_work=unit_of_work,
    )

    vector_index = vector_index_service.create_index(
        organization_id=organization.id,
        name="Integration Test Index",
        profile_name="legal-general",
    )

    vector_index_service.mark_ready(
        index_id=vector_index.id,
        organization_id=organization.id,
    )

    vector_index_service.activate_index(
        index_id=vector_index.id,
        organization_id=organization.id,
    )

    from app.api.dependencies import get_embedding_provider

    embedding_service = EmbeddingService(
        embedding_provider=get_embedding_provider(),
    )

    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_prefix="lexisai-e2e",
    )

    collection_name = (
        "lexisai-e2e_"
        + str(vector_index.id).replace("-", "")
    )

    llm_provider = None

    try:
        # ---------------------------------------------------------
        # 1. INDEX DOCUMENT
        # ---------------------------------------------------------

        indexing_service = DocumentIndexingService(
            document_repository=document_repository,
            document_chunk_repository=chunk_repository,
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        indexing_service.index_document(
            document_id=document.id,
            organization_id=organization.id,
            index_id=vector_index.id,
        )

        assert qdrant_client.collection_exists(
            collection_name,
        )

        # ---------------------------------------------------------
        # 2. RETRIEVE
        # ---------------------------------------------------------

        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            vector_index_repository=(
                vector_index_repository
            ),
            profile_name="legal-general",
        )

        retriever = SemanticRetriever(
            retrieval_service=retrieval_service,
        )

        query = (
            "How many days notice is required "
            "to terminate the agreement?"
        )

        retrieval_results = retriever.retrieve(
            RetrievalQuery(
                query=query,
                organization_id=organization.id,
                top_k=3,
            ),
        )

        assert retrieval_results

        assert any(
            result.document_id == document.id
            for result in retrieval_results
        )

        assert any(
            "thirty days" in result.content.lower()
            for result in retrieval_results
        )

        # ---------------------------------------------------------
        # 3. BUILD RAG PIPELINE
        # ---------------------------------------------------------

        reranker = ScoreReranker()

        context_assembler = ContextAssembler()

        prompt_builder = PromptBuilder()

        llm_provider = OllamaLLMProvider(
            LLMProviderConfig(
                provider="ollama",
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout=settings.llm_timeout,
            ),
        )

        llm_service = LLMService(
            llm_provider=llm_provider,
        )

        orchestrator = RAGOrchestrator(
            retriever=retriever,
            reranker=reranker,
            context_assembler=context_assembler,
            prompt_builder=prompt_builder,
            llm_service=llm_service,
        )

        # ---------------------------------------------------------
        # 4. GENERATE ANSWER
        # ---------------------------------------------------------

        response = orchestrator.answer(
            RAGRequest(
                query=query,
                organization_id=organization.id,
                top_k=3,
            ),
        )

        # ---------------------------------------------------------
        # 5. VALIDATE RESPONSE
        # ---------------------------------------------------------

        assert response.answer.strip()

        assert response.sources

        assert any(
            source.document_id == str(document.id)
            for source in response.sources
        )

        answer = response.answer.lower()

        assert (
            "thirty days" in answer
            or "30 days" in answer
        )

    finally:
        if llm_provider is not None:
            llm_provider.close()

        if qdrant_client.collection_exists(
            collection_name,
        ):
            qdrant_client.delete_collection(
                collection_name,
            )

        qdrant_client.close()
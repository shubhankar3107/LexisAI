from __future__ import annotations

import os
import uuid

import pytest
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
from app.services.ollama_llm_provider import OllamaLLMProvider
from app.services.prompt_builder import PromptBuilder
from app.services.qdrant_vector_store import QdrantVectorStore
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.retrieval_service import RetrievalService
from app.services.retriever import SemanticRetriever
from app.services.reranker import ScoreReranker
from app.services.sentence_transformer_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)
from app.services.schemas.llm_config import LLMProviderConfig
from app.services.schemas.rag import RAGRequest
from app.services.vector_index_service import VectorIndexService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_INTEGRATION") != "1",
    reason=(
        "Real integration tests require "
        "RUN_REAL_INTEGRATION=1"
    ),
)


@pytest.fixture
def real_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )

    try:
        yield client
    finally:
        client.close()


def test_real_rag_pipeline(
    db_session,
    real_qdrant_client,
):
    organization_id = uuid.uuid4()

    vector_index_id = None

    try:
        # ---------------------------------------------------------
        # 1. Create organization
        # ---------------------------------------------------------

        organization = Organization(
            id=organization_id,
            name="LexisAI Real RAG Test",
            slug=f"lexisai-real-rag-{organization_id}",
        )

        db_session.add(organization)
        db_session.commit()

        # ---------------------------------------------------------
        # 2. Build vector-index lifecycle services
        # ---------------------------------------------------------

        vector_index_repository = VectorIndexRepository(
            db_session,
        )

        embedding_registry = EmbeddingRegistry(
            EMBEDDING_PROFILES,
        )

        embedding_identity_repository = (
            EmbeddingIdentityRepository(
                db_session,
            )
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

        # ---------------------------------------------------------
        # 3. Create BUILDING index
        # ---------------------------------------------------------

        vector_index = vector_index_service.create_index(
            organization_id=organization_id,
            name="Real RAG Integration Index",
            profile_name="legal-general",
        )

        vector_index_id = vector_index.id

        assert vector_index.status.value == "building"

        # ---------------------------------------------------------
        # 4. BUILDING -> READY -> ACTIVE
        # ---------------------------------------------------------

        vector_index_service.mark_ready(
            index_id=vector_index.id,
            organization_id=organization_id,
        )

        vector_index_service.activate_index(
            index_id=vector_index.id,
            organization_id=organization_id,
        )

        active_index = (
            vector_index_service.get_active_index(
                organization_id=organization_id,
                profile_name="legal-general",
            )
        )

        assert active_index is not None
        assert active_index.id == vector_index.id

        # ---------------------------------------------------------
        # 5. Create READY document
        # ---------------------------------------------------------

        document_id = uuid.uuid4()

        document = Document(
            id=document_id,
            organization_id=organization_id,
            title="Termination Notice Agreement",
            status=DocumentStatus.READY,
        )

        db_session.add(document)

        chunks = [
            DocumentChunk(
                id=uuid.uuid4(),
                document_id=document_id,
                page_number=1,
                chunk_index=0,
                content=(
                    "This agreement may be terminated by "
                    "either party by providing thirty days "
                    "written notice."
                ),
            ),
            DocumentChunk(
                id=uuid.uuid4(),
                document_id=document_id,
                page_number=2,
                chunk_index=1,
                content=(
                    "Any termination notice must be delivered "
                    "in writing to the address specified in "
                    "Section 12 of this agreement."
                ),
            ),
            DocumentChunk(
                id=uuid.uuid4(),
                document_id=document_id,
                page_number=3,
                chunk_index=2,
                content=(
                    "The agreement remains effective until "
                    "the applicable termination date."
                ),
            ),
        ]

        db_session.add_all(chunks)
        db_session.commit()

        # ---------------------------------------------------------
        # 6. Create REAL BGE-M3 embedding service
        # ---------------------------------------------------------

        profile = EMBEDDING_PROFILES["legal-general"]

        embedding_provider = (
            SentenceTransformerEmbeddingProvider(
                model_name=profile.model,
                dimensions=profile.dimensions,
                device=settings.embedding_device,
            )
        )

        embedding_service = EmbeddingService(
            embedding_provider=embedding_provider,
        )

        # ---------------------------------------------------------
        # 7. Create REAL Qdrant vector store
        # ---------------------------------------------------------

        vector_store = QdrantVectorStore(
            client=real_qdrant_client,
            collection_prefix=(
                settings.qdrant_collection_prefix
            ),
        )

        # ---------------------------------------------------------
        # 8. Index document through actual application service
        # ---------------------------------------------------------

        document_repository = DocumentRepository(
            db_session,
        )

        document_chunk_repository = (
            DocumentChunkRepository(
                db_session,
            )
        )

        indexing_service = DocumentIndexingService(
            document_repository=document_repository,
            document_chunk_repository=(
                document_chunk_repository
            ),
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        indexing_service.index_document(
            document_id=document_id,
            organization_id=organization_id,
            index_id=vector_index.id,
        )

        # ---------------------------------------------------------
        # 9. Create REAL retrieval service
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

        # ---------------------------------------------------------
        # 10. Create REAL Ollama/Qwen3 provider
        # ---------------------------------------------------------

        llm_config = LLMProviderConfig(
            provider="ollama",
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
        )

        llm_provider = OllamaLLMProvider(
            llm_config,
        )

        try:
            from app.services.llm_service import (
                LLMService,
            )

            llm_service = LLMService(
                llm_provider=llm_provider,
            )

            # -----------------------------------------------------
            # 11. Assemble REAL RAG pipeline
            # -----------------------------------------------------

            orchestrator = RAGOrchestrator(
                retriever=retriever,
                reranker=ScoreReranker(),
                context_assembler=ContextAssembler(),
                prompt_builder=PromptBuilder(),
                llm_service=llm_service,
            )

            # -----------------------------------------------------
            # 12. Execute REAL RAG query
            # -----------------------------------------------------

            response = orchestrator.answer(
                RAGRequest(
                    query=(
                        "How many days written notice "
                        "is required to terminate the agreement?"
                    ),
                    organization_id=organization_id,
                    top_k=3,
                    document_id=document_id,
                ),
            )

        finally:
            llm_provider.close()

        # ---------------------------------------------------------
        # 13. Validate generated answer
        # ---------------------------------------------------------

        assert response.answer
        assert len(response.answer.strip()) > 0

        # The answer should be grounded in the actual document.
        normalized_answer = response.answer.lower()

        assert (
            "30" in normalized_answer
            or "thirty" in normalized_answer
        )

        # ---------------------------------------------------------
        # 14. Validate source preservation
        # ---------------------------------------------------------

        assert response.sources

        assert all(
            source.document_id == str(document_id)
            for source in response.sources
        )

        assert any(
            source.page_number == 1
            for source in response.sources
        )

    finally:
        # ---------------------------------------------------------
        # 15. Remove Qdrant test collection
        # ---------------------------------------------------------

        if vector_index_id is not None:
            collection_name = (
                f"{settings.qdrant_collection_prefix}_"
                f"{str(vector_index_id).replace('-', '')}"
            )

            if real_qdrant_client.collection_exists(
                collection_name,
            ):
                real_qdrant_client.delete_collection(
                    collection_name,
                )

        # ---------------------------------------------------------
        # 16. Remove committed database test data
        # ---------------------------------------------------------

        db_session.delete(organization)
        db_session.commit()
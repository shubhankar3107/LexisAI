import pytest
from pathlib import Path

from qdrant_client import QdrantClient
from unittest.mock import patch, MagicMock

from app.api.dependencies import (
    get_embedding_provider,
    get_qdrant_client,
    get_upload_service,
    get_vector_index_service,
    get_vector_store,
    get_llm_provider,
    get_llm_service,
)
from app.core.config import settings
from app.db.unit_of_work import UnitOfWork
from app.repositories.document_repository import DocumentRepository
from app.repositories.embedding_identity_repository import (
    EmbeddingIdentityRepository,
)
from app.repositories.vector_index_repository import (
    VectorIndexRepository,
)
from app.services.checksum import ChecksumCalculator
from app.services.document_service import DocumentService
from app.services.embedding_identity_service import (
    EmbeddingIdentityService,
)
from app.services.embedding_registry import EmbeddingRegistry
from app.services.qdrant_vector_store import QdrantVectorStore
from app.services.upload_service import UploadService
from app.services.vector_index_service import VectorIndexService
from app.services.vector_store import VectorStore
from app.storage.local_file_storage import LocalFileStorage
from app.storage.storage_path import StoragePathBuilder
from app.services.embedding_provider import EmbeddingProvider
from app.services.sentence_transformer_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)

from app.services.document_indexing_service import (
    DocumentIndexingService,
)
from app.services.embedding_service import EmbeddingService
from app.api.dependencies import (
    get_embedding_provider,
    get_qdrant_client,
    get_retrieval_service,
    get_retriever,
    get_reranker,
    get_context_assembler,
    get_prompt_builder,
    get_upload_service,
    get_vector_index_service,
    get_vector_store,
)
from app.repositories.vector_index_repository import (
    VectorIndexRepository,
)
from app.services.context_assembler import ContextAssembler
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService
from app.services.retriever import Retriever, SemanticRetriever
from app.services.reranker import Reranker, ScoreReranker
from app.api.dependencies import (
    get_llm_provider_config,
)
from app.services.schemas.llm_config import (
    LLMProviderConfig,
)

from fastapi import FastAPI
from starlette.requests import Request

from app.services.llm_provider import LLMProvider
from app.services.llm_provider_registry import (
    LLMProviderRegistry,
)
from app.services.llm_service import LLMService

class FakeSession:
    pass


def test_get_upload_service():
    db = FakeSession()

    service = get_upload_service(db)

    assert isinstance(service, UploadService)

    assert isinstance(
        service._document_repository,
        DocumentRepository,
    )

    assert isinstance(
        service._unit_of_work,
        UnitOfWork,
    )

    assert isinstance(
        service._file_storage,
        LocalFileStorage,
    )

    assert isinstance(
        service._storage_path_builder,
        StoragePathBuilder,
    )

    assert isinstance(
        service._checksum_calculator,
        ChecksumCalculator,
    )

    assert service._file_storage._base_path == Path(
        settings.upload_directory,
    )


def test_get_vector_index_service():
    db = FakeSession()

    service = get_vector_index_service(db)

    assert isinstance(service, VectorIndexService)

    assert isinstance(
        service._vector_index_repository,
        VectorIndexRepository,
    )

    assert isinstance(
        service._embedding_registry,
        EmbeddingRegistry,
    )

    assert isinstance(
        service._embedding_identity_service,
        EmbeddingIdentityService,
    )

    assert isinstance(
        service._embedding_identity_service._embedding_identity_repository,
        EmbeddingIdentityRepository,
    )

    assert isinstance(
        service._unit_of_work,
        UnitOfWork,
    )


def test_get_qdrant_client():
    client = get_qdrant_client()

    assert isinstance(
        client,
        QdrantClient,
    )


def test_get_vector_store():
    client = QdrantClient(
        ":memory:",
    )

    store = get_vector_store(
        client=client,
    )

    assert isinstance(
        store,
        QdrantVectorStore,
    )

    assert isinstance(
        store,
        VectorStore,
    )

    assert store._client is client

    assert (
        store._collection_prefix
        == settings.qdrant_collection_prefix
    )


def test_get_embedding_provider():
    with patch(
        "app.api.dependencies."
        "SentenceTransformerEmbeddingProvider",
    ) as mock_provider:
        get_embedding_provider()

    mock_provider.assert_called_once_with(
        model_name="BAAI/bge-m3",
        dimensions=1024,
        device=settings.embedding_device,
    )


def test_get_embedding_service():
    from app.api.dependencies import get_embedding_service

    provider = MagicMock()

    service = get_embedding_service(
        embedding_provider=provider,
    )

    assert isinstance(
        service,
        EmbeddingService,
    )

    assert service._embedding_provider is provider


def test_get_document_indexing_service():
    from app.api.dependencies import (
        get_document_indexing_service,
    )

    db = FakeSession()

    embedding_service = MagicMock(
        spec=EmbeddingService,
    )

    vector_store = MagicMock()

    service = get_document_indexing_service(
        db=db,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    assert isinstance(
        service,
        DocumentIndexingService,
    )

    assert service._document_repository is not None
    assert service._document_chunk_repository is not None

    assert service._embedding_service is embedding_service
    assert service._vector_store is vector_store


def test_get_retrieval_service():
    db = FakeSession()

    embedding_service = MagicMock(
        spec=EmbeddingService,
    )

    vector_store = MagicMock(
        spec=VectorStore,
    )

    service = get_retrieval_service(
        db=db,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    assert isinstance(
        service,
        RetrievalService,
    )

    assert service._embedding_service is embedding_service
    assert service._vector_store is vector_store

    assert isinstance(
        service._vector_index_repository,
        VectorIndexRepository,
    )

    assert service._profile_name == "legal-general"


def test_get_retriever():
    retrieval_service = MagicMock(
        spec=RetrievalService,
    )

    retriever = get_retriever(
        retrieval_service=retrieval_service,
    )

    assert isinstance(
        retriever,
        SemanticRetriever,
    )

    assert isinstance(
        retriever,
        Retriever,
    )

    assert retriever._retrieval_service is (
        retrieval_service
    )


def test_get_reranker():
    reranker = get_reranker()

    assert isinstance(
        reranker,
        ScoreReranker,
    )

    assert isinstance(
        reranker,
        Reranker,
    )


def test_get_context_assembler():
    assembler = get_context_assembler()

    assert isinstance(
        assembler,
        ContextAssembler,
    )


def test_get_prompt_builder():
    builder = get_prompt_builder()

    assert isinstance(
        builder,
        PromptBuilder,
    )


def test_get_llm_provider_config():
    original_provider = settings.llm_provider
    original_base_url = settings.llm_base_url
    original_api_key = settings.llm_api_key
    original_model = settings.llm_model
    original_timeout = settings.llm_timeout

    try:
        settings.llm_provider = "test-provider"
        settings.llm_base_url = "http://localhost:8000/v1"
        settings.llm_api_key = "test-api-key"
        settings.llm_model = "test-model"
        settings.llm_timeout = 30.0

        config = get_llm_provider_config()

        assert isinstance(
            config,
            LLMProviderConfig,
        )

        assert config.provider == "test-provider"
        assert config.base_url == (
            "http://localhost:8000/v1"
        )
        assert config.api_key == "test-api-key"
        assert config.model == "test-model"
        assert config.timeout == 30.0

    finally:
        settings.llm_provider = original_provider
        settings.llm_base_url = original_base_url
        settings.llm_api_key = original_api_key
        settings.llm_model = original_model
        settings.llm_timeout = original_timeout


def test_get_llm_provider_config_requires_provider():
    original_provider = settings.llm_provider

    try:
        settings.llm_provider = None

        with pytest.raises(
            RuntimeError,
            match="LLM provider is not configured",
        ):
            get_llm_provider_config()

    finally:
        settings.llm_provider = original_provider


def test_get_llm_provider_resolves_registered_provider():
    provider = MagicMock(
        spec=LLMProvider,
    )

    app = FastAPI()

    app.state.llm_provider_registry = (
        LLMProviderRegistry(
            {
                "ollama": provider,
            },
        )
    )

    request = Request(
        scope={
            "type": "http",
            "app": app,
        },
    )

    with patch.object(
        settings,
        "llm_provider",
        "ollama",
    ):
        result = get_llm_provider(
            request,
        )

    assert result is provider



def test_get_llm_provider_rejects_uninitialized_registry():
    app = FastAPI()

    request = Request(
        scope={
            "type": "http",
            "app": app,
        },
    )

    with patch.object(
        settings,
        "llm_provider",
        "ollama",
    ):
        with pytest.raises(
            RuntimeError,
            match="LLM provider registry is not initialized",
        ):
            get_llm_provider(
                request,
            )


def test_get_llm_service_wraps_provider():
    provider = MagicMock(
        spec=LLMProvider,
    )

    service = get_llm_service(
        provider=provider,
    )

    assert isinstance(
        service,
        LLMService,
    )

    assert service._llm_provider is provider
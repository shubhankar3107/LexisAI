from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.api.routes.rag import router as rag_router
from app.api.routes.uploads import router as uploads_router
from app.api.routes.vector_indexes import (
    router as vector_indexes_router,
)
from app.api.dependencies import (
    get_llm_provider_config,
)
from app.services.llm_provider_factory import (
    LLMProviderFactory,
)
from app.services.llm_provider_registry import (
    LLMProviderRegistry,
)
from app.services.ollama_llm_provider import (
    OllamaLLMProvider,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    config = get_llm_provider_config()

    factory = LLMProviderFactory(
        {
            "ollama": OllamaLLMProvider,
        },
    )

    provider = factory.create(
        config,
    )

    registry = LLMProviderRegistry(
        {
            config.provider: provider,
        },
    )

    app.state.llm_provider_registry = registry

    try:
        yield
    finally:
        provider.close()


app = FastAPI(
    title="LexisAI",
    lifespan=lifespan,
)

app.include_router(uploads_router)
app.include_router(documents_router)
app.include_router(vector_indexes_router)
app.include_router(rag_router)
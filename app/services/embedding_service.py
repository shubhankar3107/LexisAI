from __future__ import annotations

from app.services.embedding_provider import EmbeddingProvider


class EmbeddingService:
    """Application service for generating text embeddings."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
    ):
        self._embedding_provider = embedding_provider

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._embedding_provider.embed_documents(
            texts,
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embedding_provider.embed_query(
            text,
        )
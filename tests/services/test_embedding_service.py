from __future__ import annotations

import pytest

from app.services.embedding_provider import EmbeddingProvider
from app.services.embedding_service import EmbeddingService


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.document_calls.append(texts)

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        self.query_calls.append(text)

        return [0.4, 0.5, 0.6]


def test_embedding_service_embeds_documents():
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(
        embedding_provider=provider,
    )

    result = service.embed_documents(
        [
            "first chunk",
            "second chunk",
        ],
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]

    assert provider.document_calls == [
        [
            "first chunk",
            "second chunk",
        ],
    ]


def test_embedding_service_embeds_query():
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(
        embedding_provider=provider,
    )

    result = service.embed_query(
        "What is the termination notice?",
    )

    assert result == [
        0.4,
        0.5,
        0.6,
    ]

    assert provider.query_calls == [
        "What is the termination notice?",
    ]


def test_embedding_service_preserves_document_order():
    class OrderedProvider(FakeEmbeddingProvider):

        def embed_documents(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [
                [float(len(text))]
                for text in texts
            ]

    provider = OrderedProvider()

    service = EmbeddingService(
        embedding_provider=provider,
    )

    result = service.embed_documents(
        [
            "short",
            "medium text",
            "longer text here",
        ],
    )

    assert result == [
        [5.0],
        [11.0],
        [16.0],
    ]


def test_embedding_service_passes_empty_document_batch():
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(
        embedding_provider=provider,
    )

    result = service.embed_documents([])

    assert result == []

    assert provider.document_calls == [[]]


def test_embedding_service_is_constructed_with_embedding_provider():
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(
        embedding_provider=provider,
    )

    assert service._embedding_provider is provider
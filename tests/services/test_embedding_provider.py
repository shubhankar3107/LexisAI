import pytest

from app.services.embedding_provider import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_embedding_provider_supports_document_embeddings():
    provider = FakeEmbeddingProvider()

    result = provider.embed_documents(
        [
            "first chunk",
            "second chunk",
        ],
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]


def test_embedding_provider_supports_query_embedding():
    provider = FakeEmbeddingProvider()

    result = provider.embed_query(
        "What is the termination notice?",
    )

    assert result == [0.1, 0.2, 0.3]


def test_embedding_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_embedding_provider_preserves_document_order():
    class OrderedEmbeddingProvider(EmbeddingProvider):

        def embed_documents(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [
                [float(len(text))]
                for text in texts
            ]

        def embed_query(
            self,
            text: str,
        ) -> list[float]:
            return [float(len(text))]

    provider = OrderedEmbeddingProvider()

    texts = [
        "short",
        "a much longer piece of text",
        "medium",
    ]

    result = provider.embed_documents(texts)

    assert result == [
        [5.0],
        [27.0],
        [6.0],
    ]


def test_embedding_provider_returns_numeric_query_vector():
    provider = FakeEmbeddingProvider()

    result = provider.embed_query(
        "contract text",
    )

    assert isinstance(result, list)
    assert all(
        isinstance(value, float)
        for value in result
    )
    assert len(result) > 0


def test_embedding_provider_returns_numeric_document_vectors():
    provider = FakeEmbeddingProvider()

    result = provider.embed_documents(
        ["contract text"],
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert all(
        isinstance(value, float)
        for value in result[0]
    )
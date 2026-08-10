from __future__ import annotations

from unittest.mock import patch, Mock

import numpy as np
import pytest

from app.services.sentence_transformer_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)


class FakeSentenceTransformer:
    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        self.model_name = model_name
        self.device = device

    def get_embedding_dimension(self):
        return 1024

    def encode(
        self,
        texts,
        normalize_embeddings,
    ):
        assert normalize_embeddings is True

        if isinstance(texts, str):
            texts = [texts]

        return np.array(
            [
                [
                    0.1,
                    0.2,
                    0.3,
                ]
                for _ in texts
            ],
            dtype=np.float32,
        )


def make_provider():
    with patch(
        "app.services.sentence_transformer_embedding_provider."
        "SentenceTransformer",
        FakeSentenceTransformer,
    ):
        provider = SentenceTransformerEmbeddingProvider(
            model_name="BAAI/bge-m3",
            dimensions=1024,
            device="mps",
        )

    return provider


def test_provider_loads_configured_model():
    model_constructor = Mock(
        return_value=FakeSentenceTransformer(
            "BAAI/bge-m3",
            "mps",
        ),
    )

    with patch(
        "app.services.sentence_transformer_embedding_provider."
        "SentenceTransformer",
        model_constructor,
    ):
        SentenceTransformerEmbeddingProvider(
            model_name="BAAI/bge-m3",
            dimensions=1024,
            device="mps",
        )

    model_constructor.assert_called_once_with(
        "BAAI/bge-m3",
        device="mps",
    )


def test_provider_returns_document_embeddings():
    provider = make_provider()

    result = provider.embed_documents(
        [
            "first document",
            "second document",
        ],
    )

    assert len(result) == 2

    assert result[0] == pytest.approx(
        [0.1, 0.2, 0.3],
    )

    assert result[1] == pytest.approx(
        [0.1, 0.2, 0.3],
    )


def test_provider_preserves_document_order():
    class OrderedFakeModel(FakeSentenceTransformer):

        def get_embedding_dimension(self):
            return 1

        def encode(
            self,
            texts,
            normalize_embeddings,
        ):
            return np.array(
                [
                    [float(len(text))]
                    for text in texts
                ],
                dtype=np.float32,
            )

    with patch(
        "app.services.sentence_transformer_embedding_provider."
        "SentenceTransformer",
        OrderedFakeModel,
    ):
        provider = SentenceTransformerEmbeddingProvider(
            model_name="BAAI/bge-m3",
            dimensions=1,
            device="mps",
        )

    result = provider.embed_documents(
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


def test_provider_returns_query_embedding():
    provider = make_provider()

    result = provider.embed_query(
        "What is the termination notice?",
    )

    assert result == pytest.approx([
        0.1,
        0.2,
        0.3,
    ])


def test_provider_returns_plain_python_floats():
    provider = make_provider()

    document_result = provider.embed_documents(
        ["contract text"],
    )

    query_result = provider.embed_query(
        "contract text",
    )

    assert isinstance(
        document_result[0],
        list,
    )

    assert all(
        isinstance(value, float)
        for value in document_result[0]
    )

    assert isinstance(
        query_result,
        list,
    )

    assert all(
        isinstance(value, float)
        for value in query_result
    )


def test_empty_document_batch_returns_empty_list():
    class RecordingFakeModel(FakeSentenceTransformer):

        def __init__(
            self,
            model_name: str,
            device: str,
        ):
            super().__init__(
                model_name,
                device,
            )
            self.encode_called = False

        def encode(
            self,
            texts,
            normalize_embeddings,
        ):
            self.encode_called = True

            return super().encode(
                texts,
                normalize_embeddings,
            )

    with patch(
        "app.services.sentence_transformer_embedding_provider."
        "SentenceTransformer",
        RecordingFakeModel,
    ):
        provider = SentenceTransformerEmbeddingProvider(
            model_name="BAAI/bge-m3",
            dimensions=1024,
            device="mps",
        )

        result = provider.embed_documents([])

        assert result == []
        assert provider._model.encode_called is False


def test_provider_rejects_dimension_mismatch():
    class WrongDimensionModel(FakeSentenceTransformer):

        def get_embedding_dimension(self):
            return 768

    with patch(
        "app.services.sentence_transformer_embedding_provider."
        "SentenceTransformer",
        WrongDimensionModel,
    ):
        with pytest.raises(
            ValueError,
            match=(
                "Embedding model dimension mismatch: "
                "expected 1024, got 768"
            ),
        ):
            SentenceTransformerEmbeddingProvider(
                model_name="BAAI/bge-m3",
                dimensions=1024,
                device="mps",
            )


def test_provider_uses_configured_device():
    model_constructor = Mock(
        return_value=FakeSentenceTransformer(
            "BAAI/bge-m3",
            "cpu",
        ),
    )

    with patch(
        "app.services.sentence_transformer_embedding_provider."
        "SentenceTransformer",
        model_constructor,
    ):
        SentenceTransformerEmbeddingProvider(
            model_name="BAAI/bge-m3",
            dimensions=1024,
            device="cpu",
        )

    model_constructor.assert_called_once_with(
        "BAAI/bge-m3",
        device="cpu",
    )
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.services.embedding_provider import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Sentence Transformers."""

    def __init__(
        self,
        model_name: str,
        dimensions: int,
        device: str = "mps",
    ):
        self._model_name = model_name
        self._dimensions = dimensions
        self._device = device

        self._model = SentenceTransformer(
            model_name,
            device=device,
        )

        actual_dimensions = (
            self._model.get_embedding_dimension()
        )

        if actual_dimensions != dimensions:
            raise ValueError(
                "Embedding model dimension mismatch: "
                f"expected {dimensions}, "
                f"got {actual_dimensions}",
            )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate normalized embeddings for document chunks."""

        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
        )

        return [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate a normalized embedding for a retrieval query."""

        embeddings = self._model.encode(
            [text],
            normalize_embeddings=True,
        )

        if len(embeddings) != 1:
            raise ValueError(
                "Embedding model returned an unexpected "
                f"number of query embeddings: {len(embeddings)}",
            )

        return [
            float(value)
            for value in embeddings[0]
        ]
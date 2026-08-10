from app.services.schemas.embedding import EmbeddingProfile


EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "legal-general": EmbeddingProfile(
        name="legal-general",
        provider="sentence-transformers",
        model="BAAI/bge-m3",
        model_version="v1",
        dimensions=1024,
    ),
}
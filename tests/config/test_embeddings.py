from app.config.embeddings import EMBEDDING_PROFILES
from app.services.schemas.embedding import EmbeddingProfile


def test_embedding_configuration_contains_legal_general_profile():
    profile = EMBEDDING_PROFILES["legal-general"]

    assert isinstance(profile, EmbeddingProfile)
    assert profile.provider == "sentence-transformers"
    assert profile.model == "BAAI/bge-m3"
    assert profile.model_version == "v1"
    assert profile.dimensions == 1024

def test_embedding_configuration_keys_match_profile_names():
    for name, profile in EMBEDDING_PROFILES.items():
        assert name == profile.name
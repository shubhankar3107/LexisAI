import pytest

from app.services.schemas.embedding import (
    EmbeddingIdentitySpec,
    EmbeddingProfile,
)


def test_embedding_profile_contains_model_configuration():
    profile = EmbeddingProfile(
        name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v1",
        dimensions=4096,
    )

    assert profile.name == "legal-general"
    assert profile.provider == "huggingface"
    assert profile.model == "Qwen3-Embedding-8B"
    assert profile.model_version == "v1"
    assert profile.dimensions == 4096


def test_embedding_identity_spec_contains_exact_embedding_information():
    identity = EmbeddingIdentitySpec(
        profile_name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v1",
        dimensions=4096,
    )

    assert identity.profile_name == "legal-general"
    assert identity.provider == "huggingface"
    assert identity.model == "Qwen3-Embedding-8B"
    assert identity.model_version == "v1"
    assert identity.dimensions == 4096


def test_embedding_profile_is_immutable():
    profile = EmbeddingProfile(
        name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v1",
        dimensions=4096,
    )

    with pytest.raises(AttributeError):
        profile.model = "another-model"


def test_embedding_identity_spec_is_immutable():
    identity = EmbeddingIdentitySpec(
        profile_name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v1",
        dimensions=4096,
    )

    with pytest.raises(AttributeError):
        identity.model_version = "v2"


def make_identity() -> EmbeddingIdentitySpec:
    return EmbeddingIdentitySpec(
        profile_name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v1",
        dimensions=4096,
    )


def test_embedding_identity_spec_has_deterministic_fingerprint():
    first = make_identity()
    second = make_identity()

    assert first.fingerprint == second.fingerprint


def test_different_embedding_identity_spec_has_different_fingerprint():
    first = make_identity()

    second = EmbeddingIdentitySpec(
        profile_name="legal-general",
        provider="huggingface",
        model="Qwen3-Embedding-8B",
        model_version="v2",
        dimensions=4096,
    )

    assert first.fingerprint != second.fingerprint


def test_fingerprint_is_sha256_hex():
    identity = make_identity()

    assert len(identity.fingerprint) == 64
    assert all(
        character in "0123456789abcdef"
        for character in identity.fingerprint
    )
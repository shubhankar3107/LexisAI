import pytest

from app.services.embedding_registry import (
    EmbeddingProfileNotFoundError,
    EmbeddingRegistry,
)
from app.services.schemas.embedding import EmbeddingProfile


def make_profile(
    name: str = "legal-general",
) -> EmbeddingProfile:
    return EmbeddingProfile(
    name="legal-general",
    provider="huggingface",
    model="Qwen3-Embedding-8B",
    model_version="v1",
    dimensions=4096,
)


def test_registry_returns_profile_by_name():
    profile = make_profile()

    registry = EmbeddingRegistry(
        profiles={
            profile.name: profile,
        },
    )

    result = registry.get("legal-general")

    assert result is profile


def test_registry_supports_multiple_profiles():
    general = make_profile("general")
    legal = make_profile("legal")

    registry = EmbeddingRegistry(
        profiles={
            "general": general,
            "legal": legal,
        },
    )

    assert registry.get("general") is general
    assert registry.get("legal") is legal


def test_registry_raises_for_unknown_profile():
    registry = EmbeddingRegistry(
        profiles={
            "legal-general": make_profile(),
        },
    )

    with pytest.raises(
        EmbeddingProfileNotFoundError,
        match="Embedding profile 'unknown' not found",
    ):
        registry.get("unknown")


def test_registry_does_not_share_profile_mapping():
    profile = make_profile()

    profiles = {
        "legal-general": profile,
    }

    registry = EmbeddingRegistry(
        profiles=profiles,
    )

    profiles.clear()

    assert registry.get("legal-general") is profile
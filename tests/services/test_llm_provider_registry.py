import pytest

from app.services.exceptions import (
    LLMProviderNotFoundError,
)
from app.services.llm_provider import LLMProvider
from app.services.llm_provider_registry import (
    LLMProviderRegistry,
)
from app.services.schemas.llm import (
    LLMRequest,
    LLMResponse,
)


class FakeLLMProvider(LLMProvider):

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        return LLMResponse(
            content="test response",
            model="test-model",
        )


def test_registry_returns_registered_provider():
    provider = FakeLLMProvider()

    registry = LLMProviderRegistry(
        {
            "test-provider": provider,
        },
    )

    result = registry.get(
        "test-provider",
    )

    assert result is provider


def test_registry_raises_for_unknown_provider():
    registry = LLMProviderRegistry({})

    with pytest.raises(
        LLMProviderNotFoundError,
        match="LLM provider 'missing' not found",
    ):
        registry.get("missing")


def test_registry_copies_provider_mapping():
    provider = FakeLLMProvider()

    providers = {
        "test-provider": provider,
    }

    registry = LLMProviderRegistry(
        providers,
    )

    providers.clear()

    assert registry.get(
        "test-provider",
    ) is provider